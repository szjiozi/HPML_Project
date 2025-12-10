from flashrag.utils import get_retriever
from flashrag.config import Config
from flashrag.dataset.dataset import Dataset
import json


data_path = "/scratch/jc13140/hpml_project/data/hotpotqa_eval_1k.jsonl"
config = {
    "retrieval_method": "bm25",
    "bm25_backend": "pyserini",
    "retrieval_topk": 8,
    "index_path": "indexes/bm25",
    "corpus_path": "/scratch/jc13140/hpml_project/data/wiki18_100w.jsonl",
    "save_dir": "output/",
    "dataset_name": "hotpotqa",
    "use_multi_retriever": False,
    "save_retrieval_cache": False,
    "use_retrieval_cache": False,
    "retrieval_cache_path": None,
    "use_reranker": False,
}
retriever = get_retriever(config)
train_data = Dataset(config, data_path)
input_query = train_data.question
retrieval_results = retriever.batch_search(input_query)
train_data.update_output("retrieval_result", retrieval_results)
train_data.save("output/hotpotqa_eval_1k_retrieve.jsonl")
