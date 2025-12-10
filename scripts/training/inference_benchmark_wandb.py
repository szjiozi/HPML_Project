"""
範例腳本：使用 wandb 記錄推理指標

這個腳本展示了如何測量並記錄推理階段的指標，
包括延遲 (Latency)、VRAM 使用量等。

使用方式：
    # 設置環境變數
    export WANDB_API_KEY="your_api_key_here"
    export WANDB_PROJECT="LongRefiner_Training"
    
    # 運行腳本
    python inference_benchmark_wandb.py
"""

import wandb
import torch
import time
import os
from typing import Optional, List, Any


def benchmark_inference(
    model,
    query_dataset: List[Any],
    device: torch.device,
    config: dict,
    project_name: str = "LongRefiner_Training",
    warmup_iters: int = 5,
):
    """
    使用 wandb 記錄推理指標
    
    Args:
        model: PyTorch 模型
        query_dataset: 查詢資料集
        device: 計算設備
        config: 配置字典
        project_name: wandb 專案名稱
        warmup_iters: 預熱迭代次數
    """
    # --- 1. 初始化一個 "inference" Run ---
    run = wandb.init(
        project=project_name,
        job_type="inference",
        name=f"Inference_Benchmark_{config.get('model_variant', 'default')}_{config.get('quantization_strategy', 'none')}",
        config=config,
    )
    
    model.to(device)
    model.eval()
    
    # --- 2. 下載模型 (如果使用 Artifacts) ---
    # 範例：從 W&B 下載模型
    # artifact = run.use_artifact(f'{project_name}/model-{config["model_variant"]}:latest', type='model')
    # artifact_dir = artifact.download()
    # model_path = os.path.join(artifact_dir, f"model_{config['model_variant']}.pth")
    # model.load_state_dict(torch.load(model_path))
    
    # --- 3. 測量 Latency 和 VRAM ---
    latencies_ms = []
    
    if torch.cuda.is_available():
        torch.cuda.reset_max_memory_allocated()
    
    # 預熱階段
    print("Warming up...")
    with torch.no_grad():
        for i, query in enumerate(query_dataset[:warmup_iters]):
            if isinstance(query, dict):
                query = {k: v.to(device) if torch.is_tensor(v) else v for k, v in query.items()}
            else:
                query = query.to(device)
            
            if isinstance(query, dict):
                _ = model.generate(**query) if hasattr(model, "generate") else model(**query)
            else:
                _ = model.generate(query) if hasattr(model, "generate") else model(query)
    
    # 實際測量
    print("Measuring inference latency...")
    with torch.no_grad():
        for query in query_dataset:
            if isinstance(query, dict):
                query = {k: v.to(device) if torch.is_tensor(v) else v for k, v in query.items()}
            else:
                query = query.to(device)
            
            # 同步 CUDA 操作以獲得準確時間
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.time()
            
            if isinstance(query, dict):
                _ = model.generate(**query) if hasattr(model, "generate") else model(**query)
            else:
                _ = model.generate(query) if hasattr(model, "generate") else model(query)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            latencies_ms.append(latency_ms)
            
            # 記錄每個樣本的延遲
            wandb.log({
                "inference/latency_ms_per_sample": latency_ms,
            })
    
    # --- 4. 記錄到 wandb.summary ---
    if latencies_ms:
        avg_latency_ms = sum(latencies_ms) / len(latencies_ms)
        min_latency_ms = min(latencies_ms)
        max_latency_ms = max(latencies_ms)
        p50_latency_ms = sorted(latencies_ms)[len(latencies_ms) // 2]
        p95_latency_ms = sorted(latencies_ms)[int(len(latencies_ms) * 0.95)]
        p99_latency_ms = sorted(latencies_ms)[int(len(latencies_ms) * 0.99)]
        
        wandb.summary["inference/avg_latency_ms_per_sample"] = avg_latency_ms
        wandb.summary["inference/min_latency_ms_per_sample"] = min_latency_ms
        wandb.summary["inference/max_latency_ms_per_sample"] = max_latency_ms
        wandb.summary["inference/p50_latency_ms_per_sample"] = p50_latency_ms
        wandb.summary["inference/p95_latency_ms_per_sample"] = p95_latency_ms
        wandb.summary["inference/p99_latency_ms_per_sample"] = p99_latency_ms
        
        print(f"Average Latency: {avg_latency_ms:.2f} ms")
        print(f"Min Latency: {min_latency_ms:.2f} ms")
        print(f"Max Latency: {max_latency_ms:.2f} ms")
        print(f"P50 Latency: {p50_latency_ms:.2f} ms")
        print(f"P95 Latency: {p95_latency_ms:.2f} ms")
        print(f"P99 Latency: {p99_latency_ms:.2f} ms")
    
    if torch.cuda.is_available():
        peak_inference_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
        wandb.summary["inference/peak_vram_gb"] = peak_inference_vram_gb
        print(f"Peak Inference VRAM: {peak_inference_vram_gb:.2f} GB")
    
    # FLOPs per Query (通常是理論值，需要根據實際模型計算)
    flops_per_query_gflops = config.get("flops_per_query_gflops", 50.0)
    wandb.summary["inference/flops_per_query_gflops"] = flops_per_query_gflops
    
    # 結束 Run
    run.finish()
    
    return {
        "avg_latency_ms": avg_latency_ms if latencies_ms else 0,
        "peak_vram_gb": peak_inference_vram_gb if torch.cuda.is_available() else 0,
    }


if __name__ == "__main__":
    # 範例配置
    config = {
        "model_variant": "QLoRA",
        "quantization_strategy": "int4",
        "flops_per_query_gflops": 50.0,
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("這是一個範例腳本，請根據您的實際模型和資料集進行調整。")
    print("詳細說明請參考 docs/WANDB_INTEGRATION.md")

