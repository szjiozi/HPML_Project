# <div align="center">LongRefiner | Hierarchical Document Refinement for Long-context Retrieval-augmented Generation</div>
<div align="center">

**This is a fork of the original [LongRefiner](https://github.com/jinjiajie/LongRefiner) repository with added support for Apple Silicon (macOS) and modernized dependency management using [uv](https://github.com/astral-sh/uv).**

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

*   **Cross-Platform Execution**: Run inference on both NVIDIA GPUs (via `vllm`) for high performance and Apple Silicon Macs (via Hugging Face `transformers`) for local development and debugging.
*   **Modern Dependency Management**: Utilizes `uv` and `pyproject.toml` for fast, reliable, and easy-to-manage dependency installation. This replaces `requirements.txt` for clearer separation of dependencies and better project structure.
*   **Original Features**: All original features of LongRefiner are preserved.

## 🛠️ Installation

This project uses `uv` for package management. If you don't have it, install it first:
```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then, create a virtual environment and install the dependencies:
```bash
# Create venv
uv venv

# Activate venv
source .venv/bin/activate

# Install base dependencies (for Apple Silicon / CPU)
uv pip install -e .

# For NVIDIA GPU users, install with vllm extras for high performance
uv pip install -e ".[vllm]"

# To use memory-saving 4-bit quantization (recommended for local use):
uv pip install -e ".[quantization]"
```
> **Why `uv` and `pyproject.toml`?**
> Using a `pyproject.toml` file provides a standardized way to define project metadata and dependencies. `uv` is an extremely fast Python package installer and resolver that reads this file, making the process of setting up development environments significantly faster and more reliable than traditional methods.

### ⚙️ Configuration

You can control which backend the library uses via an environment variable. Create a `.env` file in the root of the project by copying the example:

```bash
cp .env.example .env
```

Then, edit the `.env` file to choose your desired backend (`hf` or `vllm`). The library will automatically load this setting when it starts.

## 🚀 Quick Start

The project will automatically detect the appropriate backend (`vllm` for HPC with NVIDIA GPUs, `hf` for Apple Silicon and others). You can override this behavior by setting the `LONGREFINER_BACKEND` environment variable to either `"vllm"` or `"hf"`.

For example, to force the Hugging Face backend on a machine where `vllm` is installed:
```bash
export LONGREFINER_BACKEND="hf"
python your_script.py
```

Below is a unified code example. It will work on any platform.

```python
import json
from longrefiner import LongRefiner # Dynamically selected based on your environment

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
uv pip install -e ".[torch,metrics]"
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


