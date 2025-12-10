# Towards Efficient RAG: Quantization-Aware Adaptation of LongRefiner

**A NYU Course Project on Quantized LLM Efficiency**

This repository investigates the impact of quantization strategies (LoRA, QLoRA) on the efficiency and performance of LongRefiner, a hierarchical document refinement system for long-context Retrieval-Augmented Generation (RAG).

**Team Members:** Junzhi Chen, Chun-Ju Tao

**Base Project:** [LongRefiner](https://github.com/ignorejjj/LongRefiner) | [Paper (ACL 2025)](https://arxiv.org/pdf/2505.10413)

---

## Table of Contents

- [Project Description](#project-description)
  - [Overview](#overview)
  - [Goal / Objective](#goal--objective)
  - [Challenges](#challenges)
  - [Approach / Techniques](#approach--techniques)
  - [Implementation Details](#implementation-details)
  - [Key Components](#key-components)
- [Project Milestones](#project-milestones)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Conclusions](#conclusions)
- [References](#references)

---

## Project Description

### Overview

LongRefiner is an efficient plug-and-play refinement system for long-context RAG applications that achieves 10x compression while maintaining superior performance through hierarchical document refinement. This project extends LongRefiner by investigating **quantization-aware adaptation** to improve training efficiency and deployment cost.

### Goal / Objective

This project aims to investigate how different quantization strategies affect the efficiency and performance of large language model (LLM) fine-tuning in long-context Retrieval-augmented Generation tasks. We focus on two key modules from LongRefiner (Jin et al., 2025)—**Dual-Level Query Analysis (DQA)** and **Adaptive Document Refinement (ADR)**—which are critical for long-context RAG.

Specifically, we compare two training paradigms:

1. **Full LoRA (FP16)** — Standard fine-tuning with full-precision weights
2. **QLoRA** — Quantization-aware fine-tuning using 4-bit NF4 quantization during training

The goal is to determine whether quantization-aware fine-tuning can achieve better trade-offs between accuracy, training efficiency, and deployment cost.

### Challenges

1. **Balancing accuracy and efficiency**: Quantization reduces memory usage but introduces representational noise that may degrade DQA classification accuracy and ADR ranking stability.
2. **Ensuring fair comparison**: Each approach must share identical data, LoRA configurations, and optimization schedules to isolate the effect of quantization strategy.

### Approach / Techniques

We fix the Hierarchical Document Structuring (HDS) component using the original 3B LongRefiner-LoRA model (full precision), and train lighter student models for DQA and ADR. The study compares two versions of student fine-tuning pipelines under identical settings.

| Group | Training Mode | Quantization Stage | Backbone | Adapter Precision |
|-------|---------------|-------------------|----------|-------------------|
| **A. Full LoRA (Baseline)** | FP16 training | None | Qwen-0.5B | FP16 |
| **B. QLoRA** | 4-bit NF4 training (quantization-aware) | During training | Qwen-0.5B | FP16 |

### Implementation Details

**Hardware:** NYU HPC Greene Cluster
- GPU: NVIDIA A100 (40GB)
- Partition: `c12m85-a100-1`

**Frameworks:**
- Training: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for LoRA/QLoRA fine-tuning
- Inference: [vLLM](https://github.com/vllm-project/vllm) for efficient model serving
- Evaluation: Custom evaluation pipeline (FlashRAG-compatible)

**Dataset:** 
- **HotpotQA** (Multi-hop question answering)
  - Training: ~10,000 samples from FlashRAG's preprocessed HotpotQA dataset
  - Evaluation: 1,000 samples from the validation split
  - Retrieval: BM25 on Wikipedia corpus (wiki18_100w)

**Note:** Due to time and resource constraints, we focused on HotpotQA as the primary evaluation dataset. The original proposal included additional datasets (NQ, TriviaQA, PopQA, 2WikiMultiHopQA, ASQA, ELI5), which remain as future work.

### Key Components

LongRefiner consists of three modules that we fine-tune using different quantization strategies:

1. **Dual-Level Query Analysis (DQA)** - Classifies queries as local or global
2. **Hierarchical Document Structuring (HDS)** - Structures documents into XML format (fixed, using original 3B model)
3. **Adaptive Document Refinement (ADR)** - Selects relevant document sections

We train student models (Qwen-0.5B) using LoRA and QLoRA, comparing them against the teacher model (Qwen-3B).

---

## Project Milestones

### Stage 1 — Preparation
- [x] Literature review on LongRefiner, LoRA, QLoRA, quantization methods
- [x] Dataset selection and preprocessing (FlashRAG HotpotQA dataset)
- [x] Download and prepare evaluation data (1000 samples)
- [x] Build retrieval index using BM25
- [x] Prepare retrieval results for HotpotQA

### Stage 2 — Model Construction
- [x] Set up teacher inference pipeline using LongRefiner-3B LoRA
- [x] Build two student fine-tuning pipelines:
  - [x] Full LoRA (FP16)
  - [x] QLoRA (4-bit NF4)
- [ ] LoRA + PTQ (FP16 → INT8/INT4) — Not completed due to time constraints

### Stage 3 — Training & Distillation
- [x] Generate training data for 3 refinement steps using LongRefiner-3B LoRA
- [x] Train student models on Qwen-0.5B-Instruct
- [x] Collect training-time metrics (VRAM, throughput, training time, FLOPs)

### Stage 4 — Evaluation & Analysis
- [x] Reproduce QA results on HotpotQA (1000 samples)
- [x] Compare performance, efficiency, and convergence across all models
- [x] Collect system metrics (inference VRAM, latency, FLOPs, model size)
- [x] Collect task metrics (Exact Match, F1 score)

### Stage 5 — Demo Development
- [ ] Build interactive RAG demo comparing Base, LoRA, and QLoRA models
- [ ] Display DQA decisions, ADR refinement, latency, VRAM usage
- [ ] Prepare final report and presentation

---

## Repository Structure

```
LongRefiner/
├── assets/                          # Sample data and figures
│   ├── main_figure.jpg
│   └── sample_data.json
├── docs/                            # Documentation
│   └── WANDB_INTEGRATION.md        # Weights & Biases integration guide
├── eval_data/                       # Evaluation datasets
│   ├── hotpotqa_eval_1k.jsonl      # Ground truth (1000 samples)
│   └── hotpotqa_eval_1k_retrieval_result.json  # BM25 retrieval results
├── longrefiner/                     # Core LongRefiner package
│   ├── __init__.py
│   ├── refiner.py                  # Main refiner implementation
│   ├── prompt_template.py          # Official prompt templates
│   └── task_instruction.py         # Task-specific instructions
├── model/                           # Model checkpoints
│   ├── Qwen2.5-0.5B-Instruct/      # Base student model (0.5B)
│   ├── step1_model/                # LoRA: Query Analysis module
│   ├── step2_model/                # LoRA: Doc Structuring module
│   ├── step3_model/                # LoRA: Global Selection module
│   ├── step1_model_qlora/          # QLoRA: Query Analysis module
│   ├── step2_model_qlora/          # QLoRA: Doc Structuring module
│   └── step3_model_qlora/          # QLoRA: Global Selection module
├── results/                         # Evaluation results
│   ├── eval_result_base_hotpotqa.json    # Teacher model (3B) results
│   ├── eval_result_lora_hotpotqa.json    # LoRA student (0.5B) results
│   └── eval_result_qlora_hotpotqa.json   # QLoRA student (0.5B) results
├── scripts/
│   ├── evaluation/                  # Evaluation scripts
│   │   ├── run_eval.sh             # Main evaluation launcher
│   │   ├── run_eval.py             # Standalone vLLM evaluation (WORKING)
│   │   ├── run_eval_flashrag.py   # FlashRAG evaluation (broken, kept for reference)
│   │   ├── compare_all_results.py  # Compare all model results
│   │   ├── find_correct_incorrect.py  # Analyze correct/incorrect predictions
│   │   └── sample_docs.json        # Sample documents for testing
│   ├── training/                    # Training scripts (placeholder)
│   │   └── [Training scripts TBD]
│   └── quick_start.py              # Quick start example
├── run_hpc.sh                       # HPC cluster execution script
├── pyproject.toml                   # Project dependencies (uv)
├── uv.lock                          # Dependency lock file
└── README.md                        # This file
```

### Key Files

**Evaluation Pipeline:**
- `run_hpc.sh` - Automated HPC execution script
- `scripts/evaluation/run_eval.sh` - Configures experiment type and paths
- `scripts/evaluation/run_eval.py` - Main evaluation script using vLLM

**Analysis Tools:**
- `scripts/evaluation/compare_all_results.py` - Compare Base/LoRA/QLoRA results
- `scripts/evaluation/find_correct_incorrect.py` - Analyze prediction patterns

**Model Checkpoints:**
- `model/Qwen2.5-0.5B-Instruct/` - Base student model
- `model/step{1,2,3}_model/` - LoRA adapters (FP16)
- `model/step{1,2,3}_model_qlora/` - QLoRA adapters (4-bit NF4)

**Please note that the model checkpoints are not included in this repository due to size constraints. You can download them from the provided link.**

---

## 🛠️ Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, reliable package management.

### Step 1: Install `uv`

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 2: Sync Dependencies

```bash
# Clone the repository
git clone https://github.com/your-username/LongRefiner.git
cd LongRefiner

# Sync project dependencies (creates .venv automatically)
uv sync
```

This will:
- Create a virtual environment (`.venv`) if it doesn't exist
- Install all dependencies from `pyproject.toml` (vllm, transformers, torch, etc.)
- Install the project in editable mode
- Use PyTorch with CUDA 11.8 support for GPU acceleration

### Why `uv` and `pyproject.toml`?

Using `pyproject.toml` provides a standardized way to define project metadata and dependencies. `uv` is an extremely fast Python package installer (10-100x faster than pip) that makes environment setup significantly faster and more reliable than traditional methods.

---

## Quick Start

### Option 1: Using HPC Script (Recommended for Clusters)

For HPC environments, use the provided execution script which handles all setup automatically:

```bash
bash run_hpc.sh
```

**What `run_hpc.sh` does:**

This script automates the entire execution pipeline for HPC clusters (e.g., SLURM-based systems):

1. **Environment Validation**
   - Checks for CUDA availability and version
   - Verifies Python version (≥3.9 required)
   - Validates SLURM job configuration (GPU allocation, memory, etc.)

2. **Dependency Management**
   - Installs `uv` if not present
   - Syncs project dependencies using `uv sync`
   - Creates/updates virtual environment automatically

3. **Git Integration**
   - Pulls latest changes from repository (if `.env` configured)
   - Supports branch switching via `GH_BRANCH` environment variable

4. **Singularity Support** (Optional)
   - Detects if running inside Singularity container
   - Automatically executes script in Singularity environment if `SINGULARITY_BASH_PATH` is set

5. **Execution**
   - Runs `scripts/evaluation/run_eval.sh` with proper environment
   - Logs output to `slurm_logs/job_<id>.out` and `slurm_logs/job_<id>.err`

**Configuration:**

Create a `.env` file (copy from `.env.example`) to customize:
```bash
# Optional: Specify git branch
GH_BRANCH=main

# Optional: Use Singularity container
SINGULARITY_BASH_PATH=/path/to/singularity/bash.sh

# Optional: Wandb API key for experiment tracking
WANDB_API_KEY=your_wandb_key
```

### Option 2: Manual Execution

Run Python scripts using `uv run` (automatically uses the project's virtual environment):

```bash
# Run the quick start example
uv run python scripts/quick_start.py

# Or activate the virtual environment manually
source .venv/bin/activate
python scripts/quick_start.py
```

### Example Code

```python
import json
from longrefiner import LongRefiner

# Initialize LongRefiner with teacher model (3B)
refiner = LongRefiner(
    base_model_path="Qwen/Qwen2.5-3B-Instruct",
    query_analysis_module_lora_path="jinjiajie/Query-Analysis-Qwen2.5-3B-Instruct",
    doc_structuring_module_lora_path="jinjiajie/Doc-Structuring-Qwen2.5-3B-Instruct",
    global_selection_module_lora_path="jinjiajie/Global-Selection-Qwen2.5-3B-Instruct",
    score_model_name="bge-reranker-v2-m3",
    score_model_path="BAAI/bge-reranker-v2-m3",
    max_model_len=25000,
)

# Load sample data
with open("assets/sample_data.json", "r") as f:
    data = json.load(f)
question = list(data.keys())[0]
document_list = list(data.values())[0]

# Process documents
refined_result = refiner.run(question, document_list, budget=2048)
print(json.dumps(refined_result, indent=2, ensure_ascii=False))
```

---

## Training

**Note:** Training scripts are currently being organized and will be added to `scripts/training/` in a future update.

### Prerequisites

Install [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for training:

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

### Training Pipeline (Placeholder)

The training process involves three steps, each fine-tuning one module:

```bash
cd scripts/training

# Step 1: Train Query Analysis module (DQA)
llamafactory-cli train train_config_step1.yaml

# Step 2: Train Document Structuring module (HDS)
llamafactory-cli train train_config_step2.yaml

# Step 3: Train Global Selection module (ADR)
llamafactory-cli train train_config_step3.yaml
```

Each training configuration supports two variants:
- **Full LoRA**: Standard FP16 LoRA training
- **QLoRA**: 4-bit NF4 quantization-aware training

Training data is generated using the teacher model (LongRefiner-3B) on the HotpotQA dataset.

**Note:** LoRA + PTQ (post-training quantization) was part of the original proposal but was not completed due to time and resource constraints.

---

## Evaluation

### Evaluation Pipeline

The evaluation process consists of three stages:

```
run_hpc.sh → run_eval.sh → run_eval.py
```

### Step 1: Configure Experiment

Edit `scripts/evaluation/run_eval.sh` to select experiment type:

```bash
# Options: base (3B teacher), lora (0.5B LoRA), qlora (0.5B QLoRA)
export EXPERIMENT_TYPE="base"  # or "lora" or "qlora"

# Enable Weights & Biases logging (optional)
WANDB_ENABLED="--wandb_enabled"
```

### Step 2: Run Evaluation

```bash
cd scripts/evaluation
bash run_eval.sh
```

Or use the HPC script:

```bash
# From project root
bash run_hpc.sh
```

### Evaluation Scripts

#### `run_eval.py` (Primary, Working)

Our main evaluation script uses **vLLM** for efficient inference and implements custom metric calculation. We use this because:

- **FlashRAG Compatibility Issues**: The original project uses FlashRAG, but we encountered API compatibility issues with the latest version
- **Time Constraints**: Due to project deadlines, we implemented a standalone evaluation pipeline
- **Official Prompts**: We reference the original paper's prompt templates (`prompt_template.py`) to ensure fair comparison

**Key Features:**
- Standalone vLLM-based generation (no FlashRAG dependency)
- Implements official prompts from paper (Appendix D, Prompt C.1 & C.2)
- Robust answer extraction with cascading strategies
- Comprehensive metrics: EM, F1, VRAM, latency, FLOPs

**Usage:**
```bash
uv run python scripts/evaluation/run_eval.py \
    --dataset_name hotpotqa \
    --retrieval_result_path eval_data/hotpotqa_eval_1k_retrieval_result.json \
    --test_sample_num 1000 \
    --experiment_type base \
    --wandb_enabled
```

#### `run_eval_flashrag.py` (Reference, Broken)

This script attempts to use the FlashRAG framework but has compatibility issues:

- **Status**: Non-functional due to FlashRAG API changes
- **Purpose**: Kept for reference and potential future fixes
- **Issue**: `Config(config_dict=...)` API mismatch with FlashRAG 0.3.0

**Note:** While our base model results are lower than the original paper (22.4% vs ~30% EM), the **relative comparison** between Base, LoRA, and QLoRA remains valid and informative for our quantization study.

### Analysis Scripts

#### `compare_all_results.py`

Compares performance across Base (teacher), LoRA, and QLoRA models:

```bash
uv run python scripts/evaluation/compare_all_results.py
```

**Output:**
- Side-by-side EM and F1 comparison
- Answer length statistics
- Format compliance analysis
- Improvement breakdown

#### `find_correct_incorrect.py`

Analyzes prediction patterns by finding examples where models succeed or fail:

```bash
uv run python scripts/evaluation/find_correct_incorrect.py
```

**Output:**
- 10 correct predictions from base model (with LoRA/QLoRA comparison)
- 10 incorrect predictions from base model (with LoRA/QLoRA comparison)
- Identifies where quantized models improve or degrade

### Metrics Collected

**System Metrics:**
- **Inference VRAM (GB)**: Peak GPU memory during inference
- **Latency (ms/sample)**: End-to-end inference time per query
- **FLOPs per Query (GFLOPs)**: Estimated compute operations per inference
- **Model Size (GB)**: Disk size of model checkpoints

**Task Metrics:**
- **EM (Exact Match)**: Percentage of predictions that exactly match ground truth
- **F1 Score**: Token-level harmonic mean of precision and recall

---

## Results

### Performance Comparison (HotpotQA, 1000 samples)

| Model | Parameters | EM (%) | F1 (%) | Latency (ms) | Peak VRAM (GB) | FLOPs (GFLOPs) |
|-------|-----------|--------|--------|--------------|----------------|----------------|
| **Base (Teacher)** | 3B | **22.4** | **29.9** | 393.1 | 2.59 | 70,668 |
| **LoRA (Student)** | 0.5B | 20.9 | 28.6 | 636.2 | 2.59 | 40,418 |
| **QLoRA (Student)** | 0.5B | 20.5 | 27.4 | **238.0** | 2.59 | 40,418 |

### Key Observations

#### 1. Performance vs. Efficiency Trade-off

- **Base Model (3B)**: Highest accuracy (22.4% EM) but largest compute cost (70.7 TFLOPs)
- **LoRA (0.5B)**: Moderate accuracy (20.9% EM), 43% reduction in FLOPs, but **slower inference** (636ms)
- **QLoRA (0.5B)**: Competitive accuracy (20.5% EM), 43% reduction in FLOPs, **fastest inference** (238ms)

**Insight:** QLoRA achieves the best efficiency with only a 1.9% EM drop compared to the base model, while being **1.7x faster** than the base and **2.7x faster** than LoRA.

#### 2. Quantization Impact

The performance degradation from teacher (3B) to students (0.5B) is **minimal**:
- LoRA: -1.5% EM (-6.7% relative)
- QLoRA: -1.9% EM (-8.5% relative)

This suggests that **quantization-aware training (QLoRA) successfully preserves model quality** while enabling efficient deployment.

#### 3. Inference Speed Paradox

Surprisingly, LoRA (FP16) is **slower** than both Base and QLoRA:
- **Hypothesis**: Smaller models may have less optimized kernels in vLLM, or adapter loading overhead dominates
- **QLoRA advantage**: 4-bit quantization enables faster memory access and computation

#### 4. Comparison with Original Paper

Our base model EM (22.4%) is lower than the original paper (~30% on HotpotQA). Possible reasons:
- Different evaluation setup (vLLM vs. FlashRAG)
- Prompt formatting differences
- Answer extraction methodology

**However**, the **relative comparison** between Base, LoRA, and QLoRA remains valid and demonstrates the effectiveness of quantization strategies.

### Visualizations

#### Performance vs. Efficiency (Pareto Frontier)

```
EM (%)
  24 │
     │  ● Base (3B)
  22 │    ╲
     │     ╲  ● LoRA (0.5B)
  20 │      ╲   ● QLoRA (0.5B)
     │       ╲
  18 │        ╲
     └─────────────────────────
       0    200   400   600   800
              Latency (ms)
```

**QLoRA achieves the best efficiency-performance trade-off**, sitting on the Pareto frontier with minimal accuracy loss and maximum speed.

#### Model Size Comparison

```
Model Size (GB):
Base (3B):     **placeholder**
LoRA (0.5B):   **placeholder**
QLoRA (0.5B):  **placeholder**
```

**descriptioon placeholder**


## Conclusions

### Summary of Findings

1. **QLoRA is the most efficient approach** for deploying LongRefiner:
   - 43% FLOPs reduction
   - 1.7x faster inference than base model
   - Only 1.9% EM drop

2. **Quantization-aware training preserves quality**:
   - QLoRA (4-bit training) performs comparably to LoRA (FP16 training)
   - Minimal accuracy degradation despite aggressive quantization

3. **Student models (0.5B) are viable alternatives** to teacher models (3B):
   - Suitable for resource-constrained deployments
   - Maintain competitive performance on long-context RAG tasks

### Recommendations

**For Production Deployment:**
- Use **QLoRA (0.5B)** for edge devices and real-time applications
- Use **Base (3B)** only when maximum accuracy is critical

**For Future Research:**
- Investigate LoRA inference speed bottleneck
- Explore mixed-precision deployment (e.g., INT8 for some layers)
- Test on additional datasets (NQ, TriviaQA, ASQA, ELI5)

---

## References

1. **Jin, J., Li, X., Dong, G., Zhang, Y., Zhu, Y., Wu, Y., Li, Z., Ye, Q., & Dou, Z. (2025).** *Hierarchical Document Refinement for Long-context Retrieval-augmented Generation.* In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025), pages 3502–3520. [Paper](https://arxiv.org/pdf/2505.10413)

2. **Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023).** *QLoRA: Efficient Finetuning of Quantized LLMs.* Advances in Neural Information Processing Systems, 36, 10088-10115.

3. **Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2022).** *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR, 1(2), 3.

### Original Project

- **Repository**: [ignorejjj/LongRefiner](https://github.com/ignorejjj/LongRefiner)
- **HuggingFace Models**: [jinjiajie/longrefiner](https://huggingface.co/collections/jinjiajie/longrefiner-683ac32af1dc861d4c5d00e2)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Original LongRefiner authors for the base implementation
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for training infrastructure
- [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) for evaluation framework reference
- Course instructors for guidance and feedback
