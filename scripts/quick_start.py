import os
import json
import time

# Set CUDA devices before importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# Import after setting environment variables
import torch  # noqa: E402
import wandb  # noqa: E402
from longrefiner import LongRefiner  # noqa: E402

# --- 1. Initialize wandb (optional, use offline mode if env var not set) ---
wandb_mode = "online" if os.getenv("WANDB_API_KEY") else "disabled"
wandb.init(
    project="LongRefiner_QuickStart",
    job_type="inference",
    name="quick_start_demo",
    mode=wandb_mode,  # Disable if no API key
)

# --- 2. Define configuration ---
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

# Log configuration to wandb
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

# --- 3. Initialize model ---
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

# Log initialization time
wandb.log({"inference/model_init_time_sec": init_time})

# --- 4. Load sample data ---
with open("assets/sample_data.json", "r") as f:
    data = json.load(f)
question = list(data.keys())[0]
document_list = list(data.values())[0][:5]

# Log input statistics
input_stats = {
    "input/num_documents": len(document_list),
    "input/question_length": len(question),
    "input/total_doc_length": sum(
        len(doc.get('content', '')) for doc in document_list
    ),
}
wandb.log(input_stats)

# --- 5. Process documents and measure inference metrics ---
print("Processing documents...")

# Reset VRAM tracker
if torch.cuda.is_available():
    torch.cuda.reset_max_memory_allocated()

# Measure inference time
inference_start_time = time.time()
refined_result = refiner.run(question, document_list, budget=budget)
inference_time = time.time() - inference_start_time

# Calculate VRAM usage
if torch.cuda.is_available():
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
else:
    peak_vram_gb = 0

# --- 6. Log inference metrics ---
avg_time_per_doc = (
    inference_time / len(document_list) if document_list else 0
)
output_stats = {
    "inference/total_time_sec": inference_time,
    "inference/avg_time_per_doc_sec": avg_time_per_doc,
    "inference/peak_vram_gb": peak_vram_gb,
}

# Calculate output statistics
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

    # Calculate compression ratio
    total_input_length = input_stats["input/total_doc_length"]
    if total_input_length > 0:
        compression_ratio = (
            total_input_length / total_output_length
            if total_output_length > 0 else 0
        )
        output_stats["output/compression_ratio"] = compression_ratio

wandb.log(output_stats)

# Log to summary (final metrics)
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

# --- 7. Output results ---
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

# Optional: Save results as wandb artifact
if isinstance(refined_result, list) and refined_result:
    result_artifact = wandb.Artifact(
        name="quick_start_result",
        type="refined_output",
        metadata=config,
    )

    # Create temporary file to save results
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

    # Clean up temporary file
    if os.path.exists(result_file):
        os.remove(result_file)

# Finish wandb run
wandb.finish()
print("\n✅ Wandb logging completed!")

# --- 8. Clean up resources ---
print("Cleaning up resources...")
try:
    refiner.shutdown()
    print("✅ Resources cleaned up successfully.")
except Exception as e:
    print(f"Warning: Error during cleanup: {e}")
