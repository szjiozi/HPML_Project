#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# =============================================================================
# Experiment Configuration
# =============================================================================
# Options:
#   base     : original LongRefiner teacher (HF checkpoints, 3B)
#   lora     : student LoRA models (Qwen-0.5B)
#   qlora    : student QLoRA models (Qwen-0.5B)
EXPERIMENT_TYPE="${EXPERIMENT_TYPE:-base}"

# --- wandb settings ---
WANDB_PROJECT="LongRefiner_Evaluation"
WANDB_ENABLED="--wandb_enabled"  # Comment out to disable: WANDB_ENABLED=""

# --- Dataset settings ---
DATASET_NAME="hotpotqa"
SPLIT="validation"  # Use validation split (test split has no public answers)
TEST_SAMPLE_NUM=1000

# --- Retrieval results (must contain questions, docs, and answers) ---
RETRIEVAL_RESULT="eval_data/hotpotqa_eval_1k_retrieval_result.json"

# --- Generator model ---
GENERATOR_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
MAX_TOKENS=512
GPU_MEMORY_UTILIZATION=0.85
TENSOR_PARALLEL_SIZE=1  # Number of GPUs for generator

# =============================================================================
# Refiner Model Configuration (based on experiment type)
# =============================================================================
if [ "${EXPERIMENT_TYPE}" = "base" ]; then
    # Teacher: original LongRefiner 3B LoRA (HF checkpoints)
    BASE_REFINER_MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"
    QUERY_ANALYSIS_MODULE="jinjiajie/Query-Analysis-Qwen2.5-3B-Instruct"
    DOC_STRUCTURING_MODULE="jinjiajie/Doc-Structuring-Qwen2.5-3B-Instruct"
    GLOBAL_SELECTION_MODULE="jinjiajie/Global-Selection-Qwen2.5-3B-Instruct"
elif [ "${EXPERIMENT_TYPE}" = "lora" ]; then
    # Student: Qwen-0.5B + LoRA adapters
    BASE_REFINER_MODEL_PATH="model/Qwen2.5-0.5B-Instruct"
    QUERY_ANALYSIS_MODULE="model/step1_model"
    DOC_STRUCTURING_MODULE="model/step2_model"
    GLOBAL_SELECTION_MODULE="model/step3_model"
elif [ "${EXPERIMENT_TYPE}" = "qlora" ]; then
    # Student: Qwen-0.5B + QLoRA adapters
    BASE_REFINER_MODEL_PATH="model/Qwen2.5-0.5B-Instruct"
    QUERY_ANALYSIS_MODULE="model/step1_model_qlora"
    DOC_STRUCTURING_MODULE="model/step2_model_qlora"
    GLOBAL_SELECTION_MODULE="model/step3_model_qlora"
else
    echo "Unknown EXPERIMENT_TYPE: ${EXPERIMENT_TYPE}"
    echo "Please use: base, lora, qlora, or lora_ptq"
    exit 1
fi

SCORE_MODEL="bge-reranker-v2-m3"
SCORE_MODEL_PATH="BAAI/bge-reranker-v2-m3"

# --- Output directory ---
SAVE_DIR="results/"

# =============================================================================
# Run Evaluation
# =============================================================================
echo "=============================================="
echo "Running evaluation: ${EXPERIMENT_TYPE}"
echo "Dataset: ${DATASET_NAME}"
echo "=============================================="

# Choose one of the following commands to run the evaluation
# uv run python scripts/evaluation/run_eval_flashrag.py \
# uv run python scripts/evaluation/run_eval.py \

uv run python scripts/evaluation/run_eval.py \
    --dataset_name "${DATASET_NAME}" \
    --split "${SPLIT}" \
    --retrieval_result_path "${RETRIEVAL_RESULT}" \
    --test_sample_num ${TEST_SAMPLE_NUM} \
    --generator_model_path "${GENERATOR_MODEL_PATH}" \
    --framework "vllm" \
    --max_tokens ${MAX_TOKENS} \
    --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
    --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
    --base_refiner_model_path "${BASE_REFINER_MODEL_PATH}" \
    --query_analysis_module_lora_path "${QUERY_ANALYSIS_MODULE}" \
    --doc_structuring_module_lora_path "${DOC_STRUCTURING_MODULE}" \
    --global_selection_module_lora_path "${GLOBAL_SELECTION_MODULE}" \
    --score_model_name "${SCORE_MODEL}" \
    --score_model_path "${SCORE_MODEL_PATH}" \
    --save_dir "${SAVE_DIR}" \
    --experiment_type "${EXPERIMENT_TYPE}" \
    --save_note "${EXPERIMENT_TYPE}" \
    --wandb_project "${WANDB_PROJECT}" \
    ${WANDB_ENABLED}

echo "=============================================="
echo "Evaluation completed!"
echo "=============================================="
