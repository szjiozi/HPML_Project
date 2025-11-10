#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
用 LongRefiner（三个 LoRA）对 retrieved.jsonl 打标签，产出三份训练数据：
- step1_qa_analysis.jsonl
- step2_doc_structuring.jsonl
- step3_global_selection.jsonl

输入 retrieved.jsonl 的每行需要包含：
{
  "id": "...",
  "question": "...",
  "docs": [
    {"doc_id": "...", "title": "...", "text": "..."}   # 或 {"contents": "Title\\nBody..."}
  ],
  "golden_answers": ["..."]   # 可选，用于记录
}
"""

import os, json, argparse, sys
from typing import List, Dict, Any
from tqdm import tqdm

# === 关键：从你给的代码中 import LongRefiner（保持文件/包路径能导入）===
# 如果你的 LongRefiner 就在同目录的 longrefiner_runner.py：
# from longrefiner_runner import LongRefiner
# 如果你就是把 LongRefiner 粘在本文件顶部，也可以直接用：
from longrefiner_runner import LongRefiner  # 按你的文件名改

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path: str, items: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def to_lr_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    LongRefiner.run_* 期望的每个文档格式是：{"contents": "{title}\\n{content}"}
    这里把 FlashRAG/LongRAG 常见的 {"title": "...", "text": "..."} 转成所需格式。
    """
    lr_docs = []
    for d in docs:
        if "contents" in d and isinstance(d["contents"], str):
            lr_docs.append({"contents": d["contents"]})
        else:
            title = d.get("title", "").strip()
            text = d.get("text", "").strip()
            lr_docs.append({"contents": f"{title}\n{text}".strip()})
    return lr_docs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieved_path", type=str, required=True, help="retrieved.jsonl（含 question + docs）")
    parser.add_argument("--out_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")

    # 三个 LoRA（作者发布的 Step1/2/3 LoRA 路径）
    parser.add_argument("--lora_step1", type=str, required=True, help="query_analysis LoRA 路径")
    parser.add_argument("--lora_step2", type=str, required=True, help="doc_structuring LoRA 路径")
    parser.add_argument("--lora_step3", type=str, required=True, help="global_selection LoRA 路径")

    # 打分模型：与 LongRefiner 代码保持一致，默认 bge-reranker-v2-m3
    parser.add_argument("--score_model_name", type=str, default="bge-reranker-v2-m3")
    parser.add_argument("--score_model_path", type=str, default="BAAI/bge-reranker-v2-m3")

    # 选择预算：和论文一致默认 2048，可按需改
    parser.add_argument("--budget", type=int, default=2048)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # === 初始化 LongRefiner（直接复用你给的实现）===
    lr = LongRefiner(
        base_model_path=args.base_model,
        query_analysis_module_lora_path=args.lora_step1,
        doc_structuring_module_lora_path=args.lora_step2,
        global_selection_module_lora_path=args.lora_step3,
        score_model_name=args.score_model_name,
        score_model_path=args.score_model_path,
        max_model_len=25000,
    )

    ids, questions, docs_for_lr, golds = [], [], [], []
    print("Loading retrieved data ...")
    for ex in tqdm(read_jsonl(args.retrieved_path)):
        ids.append(ex.get("id"))
        questions.append(ex["question"])
        docs_for_lr.append(to_lr_docs(ex["docs"]))
        golds.append(ex.get("golden_answers"))

    # ===== Step 1: Query Analysis =====
    print("Running Step1: Query Analysis ...")
    qa_results = lr.run_query_analysis(questions)
    # 产出：每条给出 Local / Global 概率（与原实现一致，是 softmax 后的近似概率）
    step1_items = []
    for _id, q, prob, ga in zip(ids, questions, qa_results, golds):
        step1_items.append({
            "id": _id,
            "question": q,
            "label": {"Local": prob.get("Local", 0.0), "Global": prob.get("Global", 0.0)},
            "golden_answers": ga
        })
    write_jsonl(os.path.join(args.out_dir, "step1_qa_analysis.jsonl"), step1_items)

    # ===== Step 2: Document Structuring =====
    print("Running Step2: Doc Structuring ...")
    struct_results = lr.run_doc_structuring(docs_for_lr)
    # 产出：每条是 list[structured_doc]，每个 structured_doc 包含 {title, abstract, sections{...}}
    step2_items = []
    for _id, q, sdoc_list, ga in zip(ids, questions, struct_results, golds):
        step2_items.append({
            "id": _id,
            "question": q,
            "structured_docs": sdoc_list,
            "golden_answers": ga
        })
    write_jsonl(os.path.join(args.out_dir, "step2_doc_structuring.jsonl"), step2_items)

    # ===== Step 3: Global Selection =====
    print("Running Step3: Global Selection ...")
    glob_sel = lr.run_global_selection(questions, struct_results)
    # 同时也可按论文的最终流程，合成“局部分数 + 全局择题 + 预算裁剪”，产出最终片段集合（可当强化/监督信号）
    print("Composing final nodes by budget (alpha from Step1, local scores from reranker) ...")
    final_contents = lr.run_all_search(
        question_list=questions,
        document_list=docs_for_lr,
        query_analysis_result=qa_results,
        doc_structuring_result=struct_results,
        budget=args.budget,
        ratio=None,
    )
    # 产出：selected_titles（每个 doc 选了哪些标题）+ final_contents（预算内最终上下文片段）
    step3_items = []
    for _id, q, sel_titles_per_doc, final_ctx, ga in zip(ids, questions, glob_sel, final_contents, golds):
        step3_items.append({
            "id": _id,
            "question": q,
            "selected_titles_per_doc": sel_titles_per_doc,   # 来自 run_global_selection
            "final_contents": final_ctx,                     # 来自 run_all_search（已结合 local+global+预算）
            "budget": args.budget,
            "golden_answers": ga
        })
    write_jsonl(os.path.join(args.out_dir, "step3_global_selection.jsonl"), step3_items)

    print("Done. Outputs are saved in:", args.out_dir)

if __name__ == "__main__":
    main()
