python -m flashrag.retriever.index_builder \
    --retrieval_method bm25 \
    --corpus_path /scratch/jc13140/hpml_project/data/wiki18_100w.jsonl \
    --bm25_backend pyserini \
    --save_dir indexes/ 
