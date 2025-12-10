# Weights & Biases (wandb) 整合指南

這是一個完整的教學，說明如何將 `wandb` 導入您的 PyTorch 專案，以捕捉所有訓練和推理指標。

-----

## 🚀 步驟 1: 安裝與登入 (HPC 環境)

### 1. 安裝函式庫

在您的 HPC 環境中 (可能是在您的 Slurm 腳本或互動式節點中)，確保 `wandb` 已安裝：

```bash
pip install wandb
```

或者使用 `uv` (推薦，與專案一致)：

```bash
uv add wandb
```

### 2. 登入 (HPC 最佳實踐)

由於 HPC 節點通常沒有瀏覽器，您需要使用 API Key 登入。

1. 到 [wandb.ai/authorize](https://wandb.ai/authorize) 獲取您的 API Key。

2. 在您的 HPC 登入節點上執行：

   ```bash
   wandb login
   ```

   當它提示時，貼上您的 API Key。

3. **(推薦)** 為了讓您的 Slurm 任務自動登入，最好的方法是將 API Key 設置為環境變數。在您的 `.bashrc` 或提交腳本中加入：

   ```bash
   export WANDB_API_KEY="YOUR_API_KEY_HERE"
   ```

-----

## 🛠️ 步驟 2: 整合到您的 PyTorch 腳本

`wandb` 的核心是 `wandb.init()` 和 `wandb.log()`。

- `wandb.init()`：在腳本開始時調用一次，用於初始化一個新的「Run」。
- `wandb.log()`：在訓練/驗證迴圈中調用，用於記錄指標。
- `wandb.summary`：用於儲存**最終**的單一值指標 (例如「Avg. F1」或「Peak VRAM」)。
- `wandb.Artifact`：用於儲存模型權重或資料集。

### 範例腳本結構 (Pseudo-code)

這是一個完整的範例，展示如何整合您所有的指標。

```python
import wandb
import torch
import time
import os
from torch.utils.data import DataLoader
from your_model_file import YourModel # 替換成您的模型
from your_dataset_file import YourDataset # 替換成您的資料集
from your_metrics_file import calculate_accuracy, calculate_f1 # 替換成您的計算函式

# --- 1. 定義您的實驗配置 ---
# 這些配置會被 wandb 記錄，並讓您能夠分組和過濾
config = {
    "model_variant": "QLoRA", # 'Full LoRA', 'LoRA+PTQ'
    "quantization_strategy": "int4", # 'None', 'int8'
    "dataset": "LongRefiner_Combined",
    "epochs": 5,
    "learning_rate": 1e-4,
    "batch_size": 8,
}

# --- 2. 初始化 W&B Run ---
# (確保 WANDB_API_KEY 已在環境中設置)
run = wandb.init(
    project="Your_Project_Name",  # 您的專案名稱
    config=config,               # 上方定義的配置
    name=f"{config['model_variant']}_{config['dataset']}_run_{int(time.time())}", # Run 的顯示名稱
    job_type="training"          # 將此 Run 標記為 "training"
)

# 使用 wandb.config 訪問配置 (這是一種最佳實踐)
cfg = wandb.config

# --- 3. 準備模型、資料和優化器 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YourModel(config=cfg).to(device)
train_dataset = YourDataset(split='train')
val_dataset = YourDataset(split='validation')
train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size)
val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

# (可選) 追蹤模型的梯度和拓撲
# wandb.watch(model, log='all', log_freq=100) # 每 100 步記錄一次

# --- 4. 訓練迴圈 ---
# 用於計算 FLOPs (這是一個簡化的例子，您需要一個更準確的估算器)
# 範例: 使用一個 library 或手動計算
# 假設: model.get_flops_per_step() 返回 TFLOPs
# TFLOPs_per_step = model.get_flops_per_step(cfg.batch_size) 

TFLOPs_per_step = 1.2 # 假設值 (TFLOPs)

# 重置 VRAM 追蹤器
if torch.cuda.is_available():
    torch.cuda.reset_max_memory_allocated()

total_steps = 0
global_step = 0
print(f"--- Starting Training for {cfg.model_variant} ---")

for epoch in range(cfg.epochs):
    epoch_start_time = time.time()
    model.train()
    
    total_train_samples = 0
    
    for step, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # --- 前向傳播 ---
        outputs = model(**batch)
        loss = outputs.loss
        
        # --- 反向傳播 ---
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # --- 記錄 Step-level 指標 ---
        if step % 20 == 0: # 每 20 步記錄一次
            wandb.log({
                "train/loss": loss.item(),
                "epoch": epoch,
                "step": global_step
            })
            
        total_train_samples += len(batch['input_ids'])
        total_steps += 1
        global_step += 1
    
    # --- 記錄 Epoch-level 訓練指標 ---
    epoch_end_time = time.time()
    epoch_duration_sec = epoch_end_time - epoch_start_time
    epoch_duration_min = epoch_duration_sec / 60.0
    
    # d. System-Level: Training Metrics
    throughput = total_train_samples / epoch_duration_sec
    
    wandb.log({
        "train/epoch_time_min": epoch_duration_min,
        "train/throughput_samples_per_sec": throughput,
        "epoch": epoch
    })
    
    # --- 5. 驗證迴圈 ---
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model.generate(**batch) # 假設
            
            # (蒐集您的預測和標籤)
            # all_preds.extend(decode(outputs))
            # all_labels.extend(decode(batch['labels']))
            pass 
    
    # c. Task-Level Metrics
    # (假設您有 all_preds 和 all_labels)
    # val_accuracy = calculate_accuracy(all_preds, all_labels)
    # val_f1 = calculate_f1(all_preds, all_labels)
    
    # 範例假資料
    val_accuracy = 0.8 + (epoch / cfg.epochs) * 0.1 # 假資料
    val_f1 = 0.75 + (epoch / cfg.epochs) * 0.1     # 假資料
    
    print(f"Epoch {epoch}: Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
    wandb.log({
        "validation/accuracy": val_accuracy,
        "validation/f1_score": val_f1,
        "epoch": epoch
    })

# --- 6. 訓練結束 - 記錄最終指標 (Summary) ---
print("--- Training Finished. Logging final summary metrics. ---")

# d. System-Level: Training Metrics
if torch.cuda.is_available():
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    wandb.summary["train/peak_vram_gb"] = peak_vram_gb
    print(f"Peak Training VRAM: {peak_vram_gb:.2f} GB")

# 假設 #GPUs = 1 (在 HPC 上您可能需要從 os.environ['SLURM_NTASKS'] 或 torch.cuda.device_count() 獲取)
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
total_training_flops_tflops = TFLOPs_per_step * total_steps * num_gpus
wandb.summary["train/total_training_flops_tflops"] = total_training_flops_tflops

# c. Task-Level Metrics (假設我們關心的是最後一個 epoch 的表現)
wandb.summary["final/validation_accuracy"] = val_accuracy
wandb.summary["final/validation_f1_score"] = val_f1

# (如果您計算了所有 dataset 的平均 F1，在這裡記錄)
# wandb.summary["final/avg_f1"] = avg_f1_all_datasets

# --- 7. 保存模型 (使用 W&B Artifacts) ---
# 這是追蹤「Model Size」的最佳方式
model_path = f"model_{cfg.model_variant}.pth"
torch.save(model.state_dict(), model_path)

# 創建一個 Artifact
artifact = wandb.Artifact(
    name=f"model-{cfg.model_variant}", # Artifact 的名稱
    type="model",                      # 類型
    metadata=cfg                       # 附加元數據
)
artifact.add_file(model_path) # 將模型文件加入
wandb.log_artifact(artifact) # 上傳 Artifact

# d. System-Level: Inference Metrics (Model Size)
model_size_gb = os.path.getsize(model_path) / (1024**3)
wandb.summary["inference/model_size_gb"] = model_size_gb
print(f"Model Size: {model_size_gb:.2f} GB")

# 結束 Run
run.finish()
```

-----

## 📊 步驟 3: 記錄 Inference 指標

您的 Inference 指標 (Latency, VRAM) 應該在一個**單獨的腳本**中測量，並記錄到一個**新的 `wandb` Run** 中。

```python
# inference_benchmark.py
import wandb
import torch
import time
from your_model_file import YourModel
from your_dataset_file import YourDataset

# --- 1. 初始化一個 "inference" Run ---
run = wandb.init(
    project="Your_Project_Name",
    job_type="inference", # 標記為 "inference"
    name="Inference_Benchmark_QLoRA_int4"
)

# --- 2. 下載模型 (如果使用 Artifacts) ---
# 這是從 W&B 下載模型的範例
# artifact = run.use_artifact('Your_Project_Name/model-QLoRA:latest', type='model')
# artifact_dir = artifact.download()
# model_path = os.path.join(artifact_dir, "model_QLoRA.pth")

# (載入模型...)
device = torch.device("cuda")
model = YourModel(...)
# model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

# (載入您的查詢資料)
# query_dataset = ...

# --- 3. 測量 Latency 和 VRAM ---
latencies_ms = []

if torch.cuda.is_available():
    torch.cuda.reset_max_memory_allocated()

with torch.no_grad():
    for query in query_dataset:
        query = query.to(device)
        
        # 預熱 (Warmup)
        # for _ in range(5):
        #     _ = model.generate(query)
            
        # 測量時間
        start_time = time.time()
        _ = model.generate(query)
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        latencies_ms.append(latency_ms)

# --- 4. 記錄到 wandb.summary ---
avg_latency_ms = sum(latencies_ms) / len(latencies_ms)
wandb.summary["inference/avg_latency_ms_per_sample"] = avg_latency_ms

if torch.cuda.is_available():
    peak_inference_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    wandb.summary["inference/peak_vram_gb"] = peak_inference_vram_gb

# (FLOPs per Query 通常是理論值，您可以直接記錄)
# GFLOPs_per_query = model.get_flops_per_query(query) / 1e9
wandb.summary["inference/flops_per_query_gflops"] = 50.0 # 假設值

print(f"Avg Latency: {avg_latency_ms:.2f} ms")
print(f"Peak Inference VRAM: {peak_inference_vram_gb:.2f} GB")

run.finish()
```

-----

## 📈 步驟 4: 在 `wandb` 儀表板上實現您的分析 (e, f)

這一步**不需要寫程式**，全都在 `wandb` 網頁介面上完成：

### 1. Convergence Curves (e.4)

- **方法:** 這會自動生成。`wandb` 會自動繪製您 `wandb.log()` 的所有指標 (如 `validation/f1_score`, `train/loss`) 對 `step` 或 `epoch` 的曲線。
- **比較:** 在您的專案頁面，`wandb` 會自動將所有 Run 的圖表疊加在一起，您可以輕鬆比較 `QLoRA` 和 `Full LoRA` 的 `validation/f1_score` 曲線。

### 2. Comparative Analysis (e.1, e.2)

- **方法:** 儀表板頂部有一個表格。點擊 "Columns" 按鈕，添加您記錄的 `summary` 指標，例如：
  - `final/validation_f1_score`
  - `train/peak_vram_gb`
  - `inference/avg_latency_ms_per_sample`
  - `inference/model_size_gb`
- **分組:** 點擊 "Group"，選擇 `config.model_variant`。`wandb` 現在會將 `QLoRA`, `Full LoRA` 等自動分組，並顯示每組的平均指標。這能讓您一目了然地看到「Performance Retention」和「Efficiency Gains」。

### 3. Pareto Frontier Visualization (e.3)

- **方法:** 在您的專案頁面，點擊 "Add panel" (或 "+" 圖示)，選擇 **"Scatter Plot"**。
- **設置:**
  - **X-axis:** 選擇 `inference/avg_latency_ms_per_sample` (或 `train/peak_vram_gb`)。
  - **Y-axis:** 選擇 `final/validation_f1_score`。
  - **Color (顏色):** 選擇 `config.model_variant`。
- **結果:** 您會得到一張圖，顯示所有實驗的「效率 vs 準確度」權衡，每個點的顏色代表它是哪個模型變體。這就是您的 Pareto Frontier。

透過這套流程，您在腳本中記錄的每項數據都會直接對應到您報告中需要的分析圖表。

-----

## 🔗 相關資源

- [wandb 官方文檔](https://docs.wandb.ai/)
- [wandb 授權頁面](https://wandb.ai/authorize)
- [wandb Python API 參考](https://docs.wandb.ai/ref/python)

