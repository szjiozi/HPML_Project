#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# --- experiment variant ---
# Options:
#   base     : original LongRefiner teacher (HF checkpoints, 3B)
#   lora     : student LoRA models (step1/2/3_model on Qwen-0.5B)
#   qlora    : student QLoRA models (step1/2/3_model_qlora on Qwen-0.5B)
#   lora_ptq : student LoRA models with post-training quantization
EXPERIMENT_TYPE="base"

# --- wandb settings ---
WANDB_PROJECT="LongRefiner_Evaluation"
WANDB_ENABLED="--wandb_enabled"  # Comment out to disable wandb

# set datasets
DATASET_NAME="nq"
SPLIT="test"

# set generator model path
GENERATOR_MODEL="llama3.1-8B-instruct"
GENERATOR_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"

# set refiner model & module paths based on experiment type
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
elif [ "${EXPERIMENT_TYPE}" = "lora_ptq" ]; then
  # Student: Qwen-0.5B + LoRA adapters with PTQ (int8/int4)
  BASE_REFINER_MODEL_PATH="model/Qwen2.5-0.5B-Instruct"
  QUERY_ANALYSIS_MODULE="model/step1_model_ptq"
  DOC_STRUCTURING_MODULE="model/step2_model_ptq"
  GLOBAL_SELECTION_MODULE="model/step3_model_ptq"
else
  echo "Unknown EXPERIMENT_TYPE: ${EXPERIMENT_TYPE}. Please use 'base', 'lora', 'qlora', or 'lora_ptq'."
  exit 1
fi
SCORE_MODEL="bge-reranker-v2-m3"
SCORE_MODEL_PATH="BAAI/bge-reranker-v2-m3"

# set save directory
SAVE_DIR="results/"

# set retrieval result
RETRIEVAL_RESULT="eval_data/hotpotqa_eval_1k_retrieval_result.json"

# run script
uv runpython scripts/evaluation/run_eval.py \
    --dataset_name ${DATASET_NAME} \
    --split ${SPLIT} \
    --generator_model ${GENERATOR_MODEL} \
    --generator_model_path ${GENERATOR_MODEL_PATH} \
    --base_refiner_model_path ${BASE_REFINER_MODEL_PATH} \
    --query_analysis_module_lora_path ${QUERY_ANALYSIS_MODULE} \
    --doc_structuring_module_lora_path ${DOC_STRUCTURING_MODULE} \
    --global_selection_module_lora_path ${GLOBAL_SELECTION_MODULE} \
    --score_model_name ${SCORE_MODEL} \
    --score_model_path ${SCORE_MODEL_PATH} \
    --gpu_id "0,1,2,3,4,5,6,7" \
    --save_dir ${SAVE_DIR} \
    --retrieval_result_path ${RETRIEVAL_RESULT} \
    --framework "vllm" \
    --gpu_memory_utilization 0.85 \
    --generator_max_input_len 15000 \
    --max_tokens 512 \
    --test_sample_num 1000 \
    --save_note "${EXPERIMENT_TYPE}_experiment" \
    --experiment_type ${EXPERIMENT_TYPE} \
    --wandb_project ${WANDB_PROJECT} \
    ${WANDB_ENABLED}
