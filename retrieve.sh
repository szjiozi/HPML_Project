wget https://modelscope.cn/api/v1/datasets/FlashRAG_Dataset/retrieval_corpus/wiki18_100w_e5_index.zip -O wiki18_100w_e5_index.zip
unzip wiki18_100w_e5_index.zip -d ./indexes/wiki18_100w_e5_index
python -m flashrag.retriever.retriever \
  --query_path ./datasets/hotpotqa_train.json \
  --retrieval_method e5 \
  --model_path intfloat/e5-base-v2 \
  --index_dir ./indexes/wiki18_100w_e5_index \
  --save_path ./retrieved/hotpotqa_train_top8.jsonl \
  --top_k 8
