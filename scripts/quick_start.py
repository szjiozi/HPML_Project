import os
import json
import time

# Set CUDA devices before importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# Import after setting environment variables
import torch  # noqa: E402
import wandb  # noqa: E402
from longrefiner import LongRefiner  # noqa: E402

# --- 1. 初始化 wandb (可選，如果環境變數未設置則使用 offline 模式) ---
wandb_mode = "online" if os.getenv("WANDB_API_KEY") else "disabled"
wandb.init(
    project="LongRefiner_QuickStart",
    job_type="inference",
    name="quick_start_demo",
    mode=wandb_mode,  # 如果沒有 API key 則禁用
)

# --- 2. 定義配置 ---
query_analysis_module_lora_path = (
    "jinjiajie/Query-Analysis-Qwen2.5-3B-Instruct"
)
doc_structuring_module_lora_path = (
    "jinjiajie/Doc-Structuring-Qwen2.5-3B-Instruct"
)
selection_module_lora_path = (
    "jinjiajie/Global-Selection-Qwen2.5-3B-Instruct"
)
base_model_path = "Qwen/Qwen2.5-3B-Instruct"
score_model_name = "bge-reranker-v2-m3"
score_model_path = "BAAI/bge-reranker-v2-m3"
max_model_len = 25000
budget = 2048

# 記錄配置到 wandb
config = {
    "base_model_path": base_model_path,
    "query_analysis_module_lora_path": query_analysis_module_lora_path,
    "doc_structuring_module_lora_path": doc_structuring_module_lora_path,
    "global_selection_module_lora_path": selection_module_lora_path,
    "score_model_name": score_model_name,
    "score_model_path": score_model_path,
    "max_model_len": max_model_len,
    "budget": budget,
    "cuda_visible_devices": os.environ.get(
        'CUDA_VISIBLE_DEVICES', ''
    ),
}
wandb.config.update(config)

# --- 3. 初始化模型 ---
print("Initializing LongRefiner...")
init_start_time = time.time()

refiner = LongRefiner(
    base_model_path=base_model_path,
    query_analysis_module_lora_path=query_analysis_module_lora_path,
    doc_structuring_module_lora_path=doc_structuring_module_lora_path,
    global_selection_module_lora_path=selection_module_lora_path,
    score_model_name=score_model_name,
    score_model_path=score_model_path,
    max_model_len=max_model_len,
)

init_time = time.time() - init_start_time
print(f"Model initialization completed in {init_time:.2f} seconds")

# 記錄初始化時間
wandb.log({"inference/model_init_time_sec": init_time})

# --- 4. 載入樣本資料 ---
with open("assets/sample_data.json", "r") as f:
    data = json.load(f)
question = list(data.keys())[0]
document_list = list(data.values())[0][:5]

# 記錄輸入統計
input_stats = {
    "input/num_documents": len(document_list),
    "input/question_length": len(question),
    "input/total_doc_length": sum(
        len(doc.get('content', '')) for doc in document_list
    ),
}
wandb.log(input_stats)

# --- 5. 處理文件並測量推理指標 ---
print("Processing documents...")

# 重置 VRAM 追蹤器
if torch.cuda.is_available():
    torch.cuda.reset_max_memory_allocated()

# 測量推理時間
inference_start_time = time.time()
refined_result = refiner.run(question, document_list, budget=budget)
inference_time = time.time() - inference_start_time

# 計算 VRAM 使用
if torch.cuda.is_available():
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
else:
    peak_vram_gb = 0

# --- 6. 記錄推理指標 ---
avg_time_per_doc = (
    inference_time / len(document_list) if document_list else 0
)
output_stats = {
    "inference/total_time_sec": inference_time,
    "inference/avg_time_per_doc_sec": avg_time_per_doc,
    "inference/peak_vram_gb": peak_vram_gb,
}

# 計算輸出統計
if isinstance(refined_result, list):
    total_output_length = sum(
        len(doc) for doc in refined_result if isinstance(doc, str)
    )
    avg_output_length = (
        total_output_length / len(refined_result)
        if refined_result else 0
    )
    output_stats.update({
        "output/num_refined_documents": len(refined_result),
        "output/total_output_length": total_output_length,
        "output/avg_output_length": avg_output_length,
    })

    # 計算壓縮比
    total_input_length = input_stats["input/total_doc_length"]
    if total_input_length > 0:
        compression_ratio = (
            total_input_length / total_output_length
            if total_output_length > 0 else 0
        )
        output_stats["output/compression_ratio"] = compression_ratio

wandb.log(output_stats)

# 記錄到 summary (最終指標)
wandb.summary.update({
    "inference/total_time_sec": inference_time,
    "inference/peak_vram_gb": peak_vram_gb,
    "input/num_documents": len(document_list),
})

if isinstance(refined_result, list) and refined_result:
    wandb.summary.update({
        "output/num_refined_documents": len(refined_result),
        "output/compression_ratio": output_stats.get(
            "output/compression_ratio", 0
        ),
    })

# --- 7. 輸出結果 ---
print("\n=== Inference Results ===")
print(f"Total time: {inference_time:.2f} seconds")
vram_msg = (
    f"Peak VRAM: {peak_vram_gb:.2f} GB"
    if peak_vram_gb > 0
    else "VRAM tracking not available"
)
print(vram_msg)
num_docs = (
    len(refined_result) if isinstance(refined_result, list) else 1
)
print(f"\nRefined result ({num_docs} documents):")
print(refined_result)

# 可選：將結果保存為 wandb artifact
if isinstance(refined_result, list) and refined_result:
    result_artifact = wandb.Artifact(
        name="quick_start_result",
        type="refined_output",
        metadata=config,
    )

    # 創建臨時文件保存結果
    result_file = "quick_start_result.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({
            "question": question,
            "input_documents": len(document_list),
            "refined_result": refined_result,
            "config": config,
        }, f, indent=2, ensure_ascii=False)

    result_artifact.add_file(result_file)
    wandb.log_artifact(result_artifact)

    # 清理臨時文件
    if os.path.exists(result_file):
        os.remove(result_file)

# 結束 wandb run
wandb.finish()
print("\n✅ Wandb logging completed!")
