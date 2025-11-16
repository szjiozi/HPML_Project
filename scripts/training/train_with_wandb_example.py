"""
範例腳本：展示如何在 LongRefiner 訓練流程中整合 wandb

這個腳本展示了如何將 wandb 整合到 PyTorch 訓練流程中，
記錄所有訓練和驗證指標。

使用方式：
    # 設置環境變數
    export WANDB_API_KEY="your_api_key_here"
    export WANDB_PROJECT="LongRefiner_Training"
    
    # 運行腳本
    python train_with_wandb_example.py
"""

import wandb
import torch
import time
import os
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

# 注意：這是一個範例腳本，您需要根據實際的模型和資料集進行調整


def calculate_accuracy(preds, labels):
    """計算準確率 (範例函式)"""
    # 替換成您實際的計算邏輯
    return 0.0


def calculate_f1(preds, labels):
    """計算 F1 分數 (範例函式)"""
    # 替換成您實際的計算邏輯
    return 0.0


def train_with_wandb(
    model,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer,
    device: torch.device,
    config: Dict[str, Any],
    project_name: str = "LongRefiner_Training",
):
    """
    使用 wandb 進行訓練的主函式
    
    Args:
        model: PyTorch 模型
        train_loader: 訓練資料載入器
        val_loader: 驗證資料載入器 (可選)
        optimizer: 優化器
        device: 計算設備
        config: 訓練配置字典
        project_name: wandb 專案名稱
    """
    # --- 1. 初始化 W&B Run ---
    run = wandb.init(
        project=project_name,
        config=config,
        name=f"{config.get('model_variant', 'default')}_{config.get('dataset', 'default')}_run_{int(time.time())}",
        job_type="training",
    )
    
    cfg = wandb.config
    
    # --- 2. 準備訓練 ---
    epochs = cfg.get("epochs", 5)
    log_steps = cfg.get("log_steps", 20)
    
    # 用於計算 FLOPs (需要根據實際模型計算)
    TFLOPs_per_step = cfg.get("tflops_per_step", 1.2)  # 假設值
    
    # 重置 VRAM 追蹤器
    if torch.cuda.is_available():
        torch.cuda.reset_max_memory_allocated()
    
    total_steps = 0
    global_step = 0
    
    print(f"--- Starting Training for {cfg.get('model_variant', 'default')} ---")
    
    # --- 3. 訓練迴圈 ---
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        
        total_train_samples = 0
        
        for step, batch in enumerate(train_loader):
            # 將 batch 移到設備上
            if isinstance(batch, dict):
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            else:
                batch = batch.to(device)
            
            # --- 前向傳播 ---
            if isinstance(batch, dict):
                outputs = model(**batch)
            else:
                outputs = model(batch)
            
            loss = outputs.loss if hasattr(outputs, "loss") else outputs
            
            # --- 反向傳播 ---
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # --- 記錄 Step-level 指標 ---
            if step % log_steps == 0:
                wandb.log({
                    "train/loss": loss.item() if isinstance(loss, torch.Tensor) else loss,
                    "epoch": epoch,
                    "step": global_step,
                })
            
            # 計算樣本數 (需要根據實際 batch 結構調整)
            if isinstance(batch, dict) and "input_ids" in batch:
                total_train_samples += len(batch["input_ids"])
            else:
                total_train_samples += batch.size(0) if torch.is_tensor(batch) else 1
            
            total_steps += 1
            global_step += 1
        
        # --- 記錄 Epoch-level 訓練指標 ---
        epoch_end_time = time.time()
        epoch_duration_sec = epoch_end_time - epoch_start_time
        epoch_duration_min = epoch_duration_sec / 60.0
        
        throughput = total_train_samples / epoch_duration_sec if epoch_duration_sec > 0 else 0
        
        wandb.log({
            "train/epoch_time_min": epoch_duration_min,
            "train/throughput_samples_per_sec": throughput,
            "epoch": epoch,
        })
        
        # --- 4. 驗證迴圈 (如果提供) ---
        if val_loader is not None:
            model.eval()
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, dict):
                        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    else:
                        batch = batch.to(device)
                    
                    # 生成預測 (需要根據實際模型調整)
                    if isinstance(batch, dict):
                        outputs = model.generate(**batch) if hasattr(model, "generate") else model(**batch)
                    else:
                        outputs = model.generate(batch) if hasattr(model, "generate") else model(batch)
                    
                    # 蒐集預測和標籤 (需要根據實際輸出格式調整)
                    # all_preds.extend(decode(outputs))
                    # all_labels.extend(decode(batch.get('labels', [])))
            
            # 計算驗證指標
            # val_accuracy = calculate_accuracy(all_preds, all_labels)
            # val_f1 = calculate_f1(all_preds, all_labels)
            
            # 範例假資料 (替換成實際計算)
            val_accuracy = 0.8 + (epoch / epochs) * 0.1
            val_f1 = 0.75 + (epoch / epochs) * 0.1
            
            print(f"Epoch {epoch}: Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
            
            wandb.log({
                "validation/accuracy": val_accuracy,
                "validation/f1_score": val_f1,
                "epoch": epoch,
            })
    
    # --- 5. 訓練結束 - 記錄最終指標 (Summary) ---
    print("--- Training Finished. Logging final summary metrics. ---")
    
    # System-Level: Training Metrics
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
        wandb.summary["train/peak_vram_gb"] = peak_vram_gb
        print(f"Peak Training VRAM: {peak_vram_gb:.2f} GB")
    
    # 計算總 FLOPs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    total_training_flops_tflops = TFLOPs_per_step * total_steps * num_gpus
    wandb.summary["train/total_training_flops_tflops"] = total_training_flops_tflops
    
    # Task-Level Metrics
    if val_loader is not None:
        wandb.summary["final/validation_accuracy"] = val_accuracy
        wandb.summary["final/validation_f1_score"] = val_f1
    
    # --- 6. 保存模型 (使用 W&B Artifacts) ---
    model_variant = cfg.get("model_variant", "default")
    model_path = f"model_{model_variant}.pth"
    
    # 保存模型
    if hasattr(model, "state_dict"):
        torch.save(model.state_dict(), model_path)
    else:
        torch.save(model, model_path)
    
    # 創建 Artifact
    artifact = wandb.Artifact(
        name=f"model-{model_variant}",
        type="model",
        metadata=dict(cfg),
    )
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    
    # 記錄模型大小
    model_size_gb = os.path.getsize(model_path) / (1024**3)
    wandb.summary["inference/model_size_gb"] = model_size_gb
    print(f"Model Size: {model_size_gb:.2f} GB")
    
    # 結束 Run
    run.finish()
    
    return model


if __name__ == "__main__":
    # 範例配置
    config = {
        "model_variant": "QLoRA",
        "quantization_strategy": "int4",
        "dataset": "LongRefiner_Combined",
        "epochs": 5,
        "learning_rate": 1e-4,
        "batch_size": 8,
        "log_steps": 20,
        "tflops_per_step": 1.2,
    }
    
    # 範例使用 (需要根據實際情況調整)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("這是一個範例腳本，請根據您的實際模型和資料集進行調整。")
    print("詳細說明請參考 docs/WANDB_INTEGRATION.md")

