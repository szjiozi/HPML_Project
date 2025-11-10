python build_training_data_from_retrieved.py \
  --retrieved_path ./retrieved/hotpotqa_train_top8.jsonl \
  --out_dir ./training_data/ \
  --base_model Qwen/Qwen2.5-3B-Instruct \
  --lora_step1 /path/to/longrefiner_step1_qa_analysis_lora \
  --lora_step2 /path/to/longrefiner_step2_doc_struct_lora \
  --lora_step3 /path/to/longrefiner_step3_global_select_lora \
  --score_model_name bge-reranker-v2-m3 \
  --score_model_path BAAI/bge-reranker-v2-m3 \
  --budget 2048
