import json
from longrefiner import LongRefiner
from flashrag.config import Config
from flashrag.dataset.dataset import Dataset

# Initialize
query_analysis_module_lora_path = "data/jinjiajie/Query-Analysis-Qwen2.5-3B-Instruct"
doc_structuring_module_lora_path = "data/jinjiajie/Doc-Structuring-Qwen2.5-3B-Instruct"
selection_module_lora_path = "data/jinjiajie/Global-Selection-Qwen2.5-3B-Instruct"
data_path = "/scratch/jc13140/hpml_project/data/hotpotqa_train_10k.jsonl"
config = {
    "retrieval_method": "bm25",
    "bm25_backend": "pyserini",
    "retrieval_topk": 8,
    "index_path": "indexes/bm25",
    "corpus_path": "/scratch/jc13140/hpml_project/data/wiki18_100w.jsonl",
    "save_dir": "output/",
    "dataset_name": "hotpotqa",
    "use_multi_retriever": False,
    "save_retrieval_cache": False,
    "use_retrieval_cache": False,
    "retrieval_cache_path": None,
    "use_reranker": False,
}


refiner = LongRefiner(
    base_model_path="data/Qwen/Qwen2.5-3B-Instruct",
    query_analysis_module_lora_path=query_analysis_module_lora_path,
    doc_structuring_module_lora_path=doc_structuring_module_lora_path,
    global_selection_module_lora_path=selection_module_lora_path,
    score_model_name="bge-reranker-v2-m3",
    score_model_path="data/BAAI/bge-reranker-v2-m3",
    max_model_len=25000,
    save_training_data=True
)

with open("data/retrieval_result.json", "r") as f:
    retrieval_result = json.load(f)

data = Dataset(config, data_path)
questions = data.question
retrieval_docs = [retrieval_result.get(question, []) for question in questions]
# refined_result = refiner.batch_run(questions, retrieval_docs, budget=2048)

# ===== Step 1: Query Analysis =====
print("Running Step1: Query Analysis ...")
qa_results = refiner.run_query_analysis(questions)
# # 产出：每条给出 Local / Global 概率（与原实现一致，是 softmax 后的近似概率）
# step1_items = []
# for _id, q, prob, ga in zip(ids, questions, qa_results, golds):
#     step1_items.append({
#         "id": _id,
#         "question": q,
#         "label": {"Local": prob.get("Local", 0.0), "Global": prob.get("Global", 0.0)},
#         "golden_answers": ga
#     })
# write_jsonl(os.path.join(args.out_dir, "step1_qa_analysis.jsonl"), step1_items)

# ===== Step 2: Document Structuring =====
print("Running Step2: Doc Structuring ...")
struct_results = refiner.run_doc_structuring(retrieval_docs)
# # 产出：每条是 list[structured_doc]，每个 structured_doc 包含 {title, abstract, sections{...}}
# step2_items = []
# for _id, q, sdoc_list, ga in zip(ids, questions, struct_results, golds):
#     step2_items.append({
#         "id": _id,
#         "question": q,
#         "structured_docs": sdoc_list,
#         "golden_answers": ga
#     })
# write_jsonl(os.path.join(args.out_dir, "step2_doc_structuring.jsonl"), step2_items)

# ===== Step 3: Global Selection =====
print("Running Step3: Global Selection ...")
glob_sel = refiner.run_global_selection(questions, struct_results)
# 同时也可按论文的最终流程，合成“局部分数 + 全局择题 + 预算裁剪”，产出最终片段集合（可当强化/监督信号）
# print("Composing final nodes by budget (alpha from Step1, local scores from reranker) ...")
# final_contents = refiner.run_all_search(
#     question_list=questions,
#     document_list=retrieval_docs,
#     query_analysis_result=qa_results,
#     doc_structuring_result=struct_results,
#     budget=2048,
#     ratio=None,
# )
# # 产出：selected_titles（每个 doc 选了哪些标题）+ final_contents（预算内最终上下文片段）
# step3_items = []
# for _id, q, sel_titles_per_doc, final_ctx, ga in zip(ids, questions, glob_sel, final_contents, golds):
#     step3_items.append({
#         "id": _id,
#         "question": q,
#         "selected_titles_per_doc": sel_titles_per_doc,   # 来自 run_global_selection
#         "final_contents": final_ctx,                     # 来自 run_all_search（已结合 local+global+预算）
#         "budget": args.budget,
#         "golden_answers": ga
#     })
# write_jsonl(os.path.join(args.out_dir, "step3_global_selection.jsonl"), step3_items)

print("Done.")