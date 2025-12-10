import json

with open('/scratch/jc13140/hpml_project/FlashRAG/output/hotpotqa_eval_1k_retrieve.jsonl', 'r') as f:
    data = json.load(f)

transformed_data = {}
for d in data:
    transformed_data[d['question']] = d['output']['retrieval_result']

with open('/scratch/jc13140/hpml_project/data/hotpotqa_eval_1k_retrieval_result.json', 'w') as f:
    json.dump(transformed_data, f, indent=4)