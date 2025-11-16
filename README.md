# <div align="center">LongRefiner | Hierarchical Document Refinement for Long-context Retrieval-augmented Generation</div>
<div align="center">

**This is a fork of the original [LongRefiner](https://github.com/jinjiajie/LongRefiner) repository that uses [uv](https://github.com/astral-sh/uv) for package management and includes a complete HPC execution script for running the program on high-performance computing clusters.**

</div>

<div align="center">
<p>
<a href="#️-installation">Installation</a> |
<a href="#-quick-start">Quick-Start</a> |
<a href="#-training">Training</a> |
<a href="#-evaluation">Evaluation</a> |
<a href='https://huggingface.co/collections/jinjiajie/longrefiner-683ac32af1dc861d4c5d00e2'>Huggingface Models</a>
</p>
</div>

## 🔍 Overview

LongRefiner is an efficient plug-and-play refinement system for long-context RAG applications. It achieves 10x compression while maintaining superior performance through hierarchical document refinement.

<div align="center">
<img src="/assets/main_figure.jpg" width="800px">
</div>

## ✨ Key Features of this Fork

*   **Modern Dependency Management**: Utilizes `uv` and `pyproject.toml` for fast, reliable, and easy-to-manage dependency installation. This replaces `requirements.txt` for clearer separation of dependencies and better project structure.
*   **HPC Execution Script**: Includes a complete execution script (`run_hpc.sh`) that automates environment setup and program execution on HPC clusters, handling CUDA checks, Python version verification, dependency installation, and program execution.
*   **Original Features**: All original features of LongRefiner are preserved.

## 🛠️ Installation

This project uses `uv` for package management. If you don't have it, install it first:
```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then, sync the project environment and install all dependencies:
```bash
# Sync project dependencies (creates .venv automatically if needed)
uv sync
```

This will:
- Create a virtual environment (`.venv`) if it doesn't exist
- Install all dependencies defined in `pyproject.toml` (including `vllm`, `transformers`, `torch`, etc.)
- Install the project itself in editable mode
- Use the PyTorch CUDA 11.8 index for optimized GPU support

> **Why `uv` and `pyproject.toml`?**
> Using a `pyproject.toml` file provides a standardized way to define project metadata and dependencies. `uv` is an extremely fast Python package installer and resolver that reads this file, making the process of setting up development environments significantly faster and more reliable than traditional methods. The project is configured to use PyTorch with CUDA 11.8 support for optimal performance on NVIDIA GPUs.

## 🚀 Quick Start

### Running on HPC Clusters

For HPC environments, use the provided execution script which handles all setup automatically:

```bash
bash run_hpc.sh
```

This script will:
- Check for CUDA availability
- Verify Python version compatibility
- Install `uv` if not present
- Sync project dependencies using `uv sync`
- Execute the quick start example using `uv run`

### Manual Execution

You can run Python scripts using `uv run`, which automatically uses the project's virtual environment:

```bash
# Run a script with uv run (no need to activate venv)
uv run python your_script.py
```

Or use the provided quick start script:

```bash
uv run python scripts/quick_start.py
```

If you prefer the traditional approach, you can also activate the virtual environment manually:

```bash
# Activate the virtual environment
source .venv/bin/activate

# Then run your script
python your_script.py
```

Below is a code example you can use in your own scripts:

```python
import json
from longrefiner import LongRefiner

# Initialize
refiner = LongRefiner(
    base_model_path="Qwen/Qwen2.5-3B-Instruct",
    query_analysis_module_lora_path="jinjiajie/Query-Analysis-Qwen2.5-3B-Instruct",
    doc_structuring_module_lora_path="jinjiajie/Doc-Structuring-Qwen2.5-3B-Instruct",
    global_selection_module_lora_path="jinjiajie/Global-Selection-Qwen2.5-3B-Instruct",
    score_model_name="bge-reranker-v2-m3",
    score_model_path="BAAI/bge-reranker-v2-m3",
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

For advanced use cases where you need to explicitly import a specific backend, you can do so:
```python
# To explicitly use the vllm backend:
from longrefiner import LongRefinerVLLM

# To explicitly use the Hugging Face backend:
from longrefiner import LongRefinerHF
```

## 📚 Training

Training remains unchanged. For training purposes, please additionally install the `Llama-Factory` framework by following the instructions in the [official repository](https://github.com/hiyouga/LLaMA-Factory):

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
# If Llama-Factory uses uv, use uv sync; otherwise use their installation method
# Check their README for the recommended installation approach
```

Before training, prepare the datasets for three tasks in JSON format. Reference samples can be found in the training_data folder. We use the `Llama-Factory` framework for training. After setting up the training data, run:

```bash
cd scripts/training
# Train query analysis module
llamafactory-cli train train_config_step1.yaml  
# Train doc structuring module
llamafactory-cli train train_config_step2.yaml  
# Train global selection module
llamafactory-cli train train_config_step3.yaml  
```

## 📊 Evaluation

Evaluation remains unchanged. We use the [FlashRAG framework](https://github.com/RUC-NLPIR/FlashRAG) for RAG task evaluation. Required files:

- Evaluation dataset (recommended to obtain from FlashRAG's [official repository](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets))
- Retrieval results for each query in the dataset
- Model paths (same as above)

After preparation, configure the paths in `scripts/evaluation/run_eval.sh` and run:

```bash
cd scripts/evaluation
bash run_eval.sh
```


