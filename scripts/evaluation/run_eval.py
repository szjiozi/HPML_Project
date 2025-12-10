"""
Evaluation script for LongRefiner - No FlashRAG dependency
Uses vLLM for generation and simple metrics calculation
"""
import os
import re
import gc
import json
import string
import argparse
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any
from collections import Counter

import torch
import wandb
from vllm import LLM, SamplingParams
from datasets import load_dataset
from longrefiner import LongRefiner


# =============================================================================
# Evaluation Metrics (Simple implementation, no external dependency)
# =============================================================================

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Compute exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute F1 score between prediction and ground truth."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return float(prediction_tokens == ground_truth_tokens)
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_metrics(predictions: List[str], ground_truths: List[List[str]]) -> Dict[str, float]:
    """
    Compute EM and F1 metrics.
    
    Args:
        predictions: List of predicted answers
        ground_truths: List of lists of acceptable answers (multiple correct answers per question)
    
    Returns:
        Dictionary with 'em' and 'f1' scores
    """
    em_scores = []
    f1_scores = []
    
    for pred, gts in zip(predictions, ground_truths):
        # Handle single ground truth (string) or multiple (list)
        if isinstance(gts, str):
            gts = [gts]
        
        # Take max score across all acceptable answers
        em = max(exact_match_score(pred, gt) for gt in gts) if gts else 0.0
        f1 = max(f1_score(pred, gt) for gt in gts) if gts else 0.0
        
        em_scores.append(em)
        f1_scores.append(f1)
    
    return {
        "em": sum(em_scores) / len(em_scores) if em_scores else 0.0,
        "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
    }


# =============================================================================
# FLOPs Estimation
# =============================================================================

def estimate_model_params(model_path: str) -> float:
    """Estimate model parameters based on model path."""
    model_path_lower = model_path.lower()
    
    if "0.5b" in model_path_lower or "0_5b" in model_path_lower:
        return 0.5e9
    elif "3b" in model_path_lower:
        return 3.0e9
    elif "7b" in model_path_lower:
        return 7.0e9
    elif "8b" in model_path_lower:
        return 8.0e9
    elif "13b" in model_path_lower:
        return 13.0e9
    elif "70b" in model_path_lower:
        return 70.0e9
    else:
        return 3.0e9


def calculate_inference_flops(
    refiner_model_path: str,
    generator_model_path: str,
    avg_input_tokens: int = 2048,
    avg_output_tokens: int = 100,
    avg_retrieval_docs: int = 10,
) -> Tuple[float, float, float]:
    """Calculate estimated FLOPs for the inference pipeline."""
    refiner_params = estimate_model_params(refiner_model_path)
    generator_params = estimate_model_params(generator_model_path)
    
    # Refiner tokens estimation
    refiner_total_tokens = 50 + (500 * avg_retrieval_docs) + 1000
    
    # FLOPs = 2 * params * tokens (forward pass)
    refiner_flops = 2 * refiner_params * refiner_total_tokens
    generator_flops = 2 * generator_params * (avg_input_tokens + avg_output_tokens)
    
    # Convert to GFLOPs
    refiner_gflops = refiner_flops / 1e9
    generator_gflops = generator_flops / 1e9
    total_gflops = refiner_gflops + generator_gflops
    
    return refiner_gflops, generator_gflops, total_gflops


def get_model_size_gb(checkpoint_path: str) -> float:
    """Calculate the total size of model checkpoint files in GB."""
    if checkpoint_path is None:
        return 0.0
    
    path = Path(checkpoint_path)
    if not path.exists():
        return 0.0
    
    total_size = 0
    if path.is_file():
        total_size = path.stat().st_size
    else:
        for file in path.rglob("*"):
            if file.is_file():
                total_size += file.stat().st_size
    
    return total_size / (1024**3)


# =============================================================================
# Prompt Templates
# =============================================================================

def format_prompt(question: str, documents: List[str], dataset_name: str) -> str:
    """Format the prompt for the generator."""
    # Join documents
    if isinstance(documents, list):
        if documents and isinstance(documents[0], dict):
            doc_text = "\n\n".join([
                f"Document {i+1}: {doc.get('content', doc.get('text', str(doc)))}"
                for i, doc in enumerate(documents)
            ])
        else:
            doc_text = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(documents)])
    else:
        doc_text = str(documents)
    
    if dataset_name in ["eli5", "asqa"]:
        # Long-form QA
        prompt = f"""Find the useful content from the provided documents, then answer the question.
Answer the question directly. Your response should be very detailed.

The following are given documents.

{doc_text}

Question: {question}
Response: """
    else:
        # Short-form QA (NQ, HotpotQA, TriviaQA, etc.)
        prompt = f"""Find the useful content from the provided documents, then answer the question.
Answer the question directly. Your response should be very concise.
Please use 'So the final answer is:' as a prefix for the final answer.

Output format:
Question: What is the capital of France?
Response: The capital city of France is Paris. So the final answer is: Paris.

The following are given documents.

{doc_text}

Question: {question}
Response: """
    
    return prompt


def format_prompt_optimized(question: str, documents: List, tokenizer, dataset_name: str = "hotpotqa") -> str:
    """
    Use tokenizer's apply_chat_template for proper formatting.
    
    UPDATED: Now uses OFFICIAL prompts from LongRefiner paper (arXiv:2505.10413)
    Reference: Appendix D, Prompt C.1 and C.2
    
    Args:
        question: The question to answer
        documents: List of documents (str or dict)
        tokenizer: Tokenizer with apply_chat_template method
        dataset_name: Dataset name (for prompt customization)
        
    Returns:
        str: Formatted prompt with proper chat template
    """
    # Build document context
    if isinstance(documents, list):
        formatted_docs = []
        for i, doc in enumerate(documents):
            content = doc if isinstance(doc, str) else doc.get('content', doc.get('text', str(doc)))
            formatted_docs.append(f"Doc {i+1}: {content}")
        context_str = "\n".join(formatted_docs)
    else:
        context_str = str(documents)

    # Use OFFICIAL prompts from LongRefiner paper (Appendix D)
    if dataset_name not in ["eli5", "asqa"]:
        # Short-form QA (NQ, HotpotQA, TriviaQA, etc.) - Prompt C.1
        system_instruction = (
            "Find the useful content from the provided documents, then answer the question. "
            "Answer the question directly. Your response should be very concise. "
            "Please provide use 'So the final answer is:' as a prefix for the final answer."
            "\nOutput format:\n"
            "Question: What is the capital of France?\n"
            "Response:The capital city of France is Paris.So the final answer is: Paris."
            f"\n\nThe following are given documents.\n\n{context_str}"
        )
        user_input = (
            "Answer the question directly. Your response should be very concise. "
            "Please provide use 'So the final answer is:' as a prefix for the final answer.\n"
            f"Question: {question}\nResponse: "
        )
    else:
        # Long-form QA (ELI5, ASQA) - Prompt C.2
        system_instruction = (
            "Find the useful content from the provided documents, then answer the question. "
            "Answer the question directly. Your response should be very detailed."
            f"\n\nThe following are given documents.\n\n{context_str}"
        )
        user_input = (
            "Answer the question directly. Your response should be very detailed.\n"
            f"Question: {question}\nResponse: "
        )

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_input}
    ]
    
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_answer(response: str) -> str:
    """Extract the final answer from the response."""
    if "So the final answer is:" in response:
        return response.split("So the final answer is:")[-1].strip()
    return response.strip()


def extract_answer_robust(response: str) -> str:
    """
    Robust answer extraction with cascading strategies for Llama-3 verbose outputs.
    
    Args:
        response (str): Raw model output text
        
    Returns:
        str: Extracted concise answer
    """
    if not response:
        return ""
        
    # --- Stage 0: Preprocessing & Noise Removal ---
    # Remove (Note:...) or [Note:...] annotations common in Llama-3
    clean_text = re.sub(r'[\(\[]Note:.*?[\)\]]', '', response, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove "However, " transitional words at the start
    clean_text = re.sub(r'^However,?\s*', '', clean_text.strip(), flags=re.IGNORECASE)
    
    clean_text = clean_text.strip()
    
    # --- Stage 1: Explicit Format Markers (Golden Path) ---
    markers = [
        "So the final answer is:",
        "The final answer is:",
        "Final answer:",
        "Therefore, the answer is:",
        "The answer is:",
    ]
    
    for marker in markers:
        pattern = re.escape(marker)
        match = re.search(pattern, clean_text, re.IGNORECASE)
        if match:
            candidate = clean_text[match.end():].strip()
            
            # Remove trailing punctuation
            if candidate and candidate[-1] in string.punctuation:
                candidate = candidate.rstrip(string.punctuation)
            
            if candidate:
                return candidate

    # --- Stage 2: Refusal Detection ---
    refusal_patterns = [
        "not mentioned in the provided documents",
        "no information provided",
        "cannot answer",
        "context does not contain",
        "not found in the documents",
        "information is missing"
    ]
    for pattern in refusal_patterns:
        if pattern.lower() in clean_text.lower():
            if len(clean_text) < 150:
                return "no_answer"

    # --- Stage 3: Heuristic Fallback ---
    # Strategy A: Short text - likely the answer itself
    if len(clean_text) < 50:
        return clean_text.rstrip(string.punctuation)

    # Strategy B: Extract first sentence
    sentences = re.split(r'(?<=[.!?])\s+', clean_text)
    if sentences:
        first_sentence = sentences[0].strip()
        if len(first_sentence) < 100:
            # Remove common leading phrases
            first_sentence = re.sub(
                r'^(The answer is|The capital is|It is)\s+', 
                '', 
                first_sentence, 
                flags=re.IGNORECASE
            )
            return first_sentence.rstrip(string.punctuation)

    # Strategy C: Extract last sentence (for CoT reasoning)
    if len(sentences) > 1:
        last_sentence = sentences[-1].strip()
        if len(last_sentence) < 100:
            return last_sentence.rstrip(string.punctuation)

    # --- Stage 4: Forced Truncation (Fail-safe) ---
    return clean_text[:50].strip()


def load_eval_data(
    dataset_name: str,
    split: str,
    retrieval_result_path: str,
    test_sample_num: int,
) -> Tuple[List[str], List[List[str]], List[Any]]:
    """
    Load evaluation data.
    
    Returns:
        questions: List of questions
        ground_truths: List of ground truth answers
        retrieval_docs: List of retrieved documents for each question
    """
    # Try to load ground truth from corresponding .jsonl file first (most reliable)
    jsonl_path = retrieval_result_path.replace("_retrieval_result.json", ".jsonl")
    local_ground_truth = {}
    
    if os.path.exists(jsonl_path):
        print(f"Loading ground truth from local file: {jsonl_path}")
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        q = item.get("question", "")
                        ans = item.get("golden_answers", item.get("answer", ["unknown"]))
                        if isinstance(ans, str):
                            local_ground_truth[q] = [ans]
                        elif isinstance(ans, list):
                            local_ground_truth[q] = ans
                        else:
                            local_ground_truth[q] = [str(ans)]
            print(f"✓ Loaded {len(local_ground_truth)} ground truth answers from local file")
        except Exception as e:
            print(f"Warning: Failed to load local ground truth: {e}")
    else:
        print(f"Local ground truth file not found: {jsonl_path}")
    
    # Load retrieval results
    with open(retrieval_result_path, "r", encoding="utf-8") as f:
        retrieval_result = json.load(f)
    
    # Load ground truth from HuggingFace as fallback
    hf_ground_truth = {}
    if not local_ground_truth:
        print("ERROR: No local ground truth found")
    
    questions = []
    ground_truths = []
    retrieval_docs = []
    
    # Check if retrieval_result is already in the expected format
    if isinstance(retrieval_result, dict):
        # Format: {question: [docs]}
        for question, docs in list(retrieval_result.items())[:test_sample_num]:
            questions.append(question)
            retrieval_docs.append(docs)
            
            # Priority: local GT > HuggingFace GT > embedded in docs
            if question in local_ground_truth:
                ground_truths.append(local_ground_truth[question])
            elif question in hf_ground_truth:
                ground_truths.append(hf_ground_truth[question])
            elif isinstance(docs, dict):
                # Maybe docs contains answer
                answer = docs.get("answer", docs.get("golden_answers", ["unknown"]))
                ground_truths.append(answer if isinstance(answer, list) else [answer])
            else:
                ground_truths.append(["unknown"])
    
    elif isinstance(retrieval_result, list):
        # Format: [{"question": ..., "docs": ..., "answer": ...}, ...]
        for item in retrieval_result[:test_sample_num]:
            question = item.get("question", "")
            questions.append(question)
            retrieval_docs.append(item.get("docs", item.get("retrieval_result", [])))
            
            # Priority: local GT > HuggingFace GT > embedded in item
            if question in local_ground_truth:
                ground_truths.append(local_ground_truth[question])
            elif question in hf_ground_truth:
                ground_truths.append(hf_ground_truth[question])
            else:
                answer = item.get("answer", item.get("golden_answers", ["unknown"]))
                ground_truths.append(answer if isinstance(answer, list) else [answer])
    
    # Count how many have valid ground truth
    valid_gt_count = sum(1 for gt in ground_truths if gt != ["unknown"])
    print(f"\n{'='*60}")
    print("Data Loading Summary:")
    print(f"{'='*60}")
    print(f"Total samples loaded: {len(questions)}")
    print(f"✓ Valid ground truth: {valid_gt_count} ({valid_gt_count/len(questions)*100:.1f}%)")
    print(f"✗ Unknown ground truth: {len(questions) - valid_gt_count} ({(len(questions)-valid_gt_count)/len(questions)*100:.1f}%)")
    
    # Debug: Show first 3 examples
    if len(questions) > 0:
        print(f"\n{'='*60}")
        print("Sample Ground Truth (first 3 examples):")
        print(f"{'='*60}")
        for i in range(min(3, len(questions))):
            print(f"\n[{i+1}] Question: {questions[i][:80]}...")
            print(f"    Ground Truth: {ground_truths[i]}")
            print(f"    Source: {'Local' if questions[i] in local_ground_truth else 'HuggingFace' if questions[i] in hf_ground_truth else 'Unknown'}")
        print(f"{'='*60}\n")
    
    if valid_gt_count == 0:
        print("\n⚠️  WARNING: No valid ground truth found! Results will be meaningless.")
        print("   Please check that the ground truth data is correctly formatted.\n")
    
    return questions, ground_truths, retrieval_docs


# =============================================================================
# Main Functions
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run evaluation script for QA")
    
    # Dataset settings
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use")
    parser.add_argument("--retrieval_result_path", type=str, required=True, help="Path to retrieval results JSON")
    parser.add_argument("--test_sample_num", type=int, default=1000, help="Number of test samples")
    
    # Generator model settings
    parser.add_argument("--generator_model", type=str, default="llama3.1-8B-instruct", help="Name of generator model (FlashRAG)")
    parser.add_argument("--generator_model_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--framework", type=str, default="vllm", help="Framework to use (vllm, hf)")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5)
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--generator_max_input_len", type=int, default=15000, help="Maximum input length for generator")
    
    # Refiner model settings
    parser.add_argument("--base_refiner_model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--query_analysis_module_lora_path", type=str, default="jinjiajie/Query-Analysis-Qwen2.5-3B-Instruct")
    parser.add_argument("--doc_structuring_module_lora_path", type=str, default="jinjiajie/Doc-Structuring-Qwen2.5-3B-Instruct")
    parser.add_argument("--global_selection_module_lora_path", type=str, default="jinjiajie/Global-Selection-Qwen2.5-3B-Instruct")
    parser.add_argument("--score_model_name", type=str, default="bge-reranker-v2-m3")
    parser.add_argument("--score_model_path", type=str, default="BAAI/bge-reranker-v2-m3")
    
    # Output settings
    parser.add_argument("--save_dir", type=str, default="results/")
    parser.add_argument("--save_note", type=str, default="", help="Note to save with results")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU IDs to use")
    
    # Experiment tracking
    parser.add_argument("--experiment_type", type=str, default="base", choices=["base", "lora", "qlora", "lora_ptq"])
    parser.add_argument("--wandb_project", type=str, default="LongRefiner_Evaluation")
    parser.add_argument("--wandb_enabled", action="store_true", help="Enable wandb logging")
    parser.add_argument("--model_checkpoint_path", type=str, default=None)
    
    # FLOPs estimation
    parser.add_argument("--avg_input_tokens", type=int, default=2048)
    parser.add_argument("--avg_output_tokens", type=int, default=100)
    parser.add_argument("--avg_retrieval_docs", type=int, default=10)
    
    return parser.parse_args()


def run(args):
    """Run evaluation pipeline."""
    
    # Initialize wandb
    wandb_config = None
    if args.wandb_enabled:
        wandb_mode = "online" if os.getenv("WANDB_API_KEY") else "offline"
        wandb.init(
            project=args.wandb_project,
            job_type="evaluation",
            name=f"{args.experiment_type}_{args.dataset_name}",
            mode=wandb_mode,
        )
        wandb_config = vars(args)
        wandb.config.update(wandb_config)
    
    # Reset VRAM tracking
    if torch.cuda.is_available():
        torch.cuda.reset_max_memory_allocated()
    
    # --- Initialize components ---
    print("=" * 60)
    print("Initializing components...")
    init_start_time = time.time()
    
    # Initialize LongRefiner
    print("Loading LongRefiner...")
    refiner = LongRefiner(
        base_model_path=args.base_refiner_model_path,
        query_analysis_module_lora_path=args.query_analysis_module_lora_path,
        doc_structuring_module_lora_path=args.doc_structuring_module_lora_path,
        global_selection_module_lora_path=args.global_selection_module_lora_path,
        score_model_name=args.score_model_name,
        score_model_path=args.score_model_path,
        max_model_len=25000,
    )

    init_time = time.time() - init_start_time
    print(f"Component initialization completed in {init_time:.2f} seconds")
    
    # --- Load data ---
    print("Loading evaluation data...")
    questions, ground_truths, retrieval_docs = load_eval_data(
        dataset_name=args.dataset_name,
        split=args.split,
        retrieval_result_path=args.retrieval_result_path,
        test_sample_num=args.test_sample_num,
    )
    num_samples = len(questions)
    print(f"Processing {num_samples} samples...")
    
    # --- Run refinement ---
    print("Running document refinement...")
    refiner_start_time = time.time()
    refined_results = refiner.batch_run(questions, retrieval_docs, budget=2048)
    refiner_time = time.time() - refiner_start_time
    print(f"Refiner completed in {refiner_time:.2f} seconds")
    
    # Shutdown refiner to free GPU memory before loading generator
    print("Shutting down refiner to free GPU memory...")
    import gc
    refiner.shutdown()
    del refiner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Initialize vLLM generator (after refiner shutdown)
    print("Loading vLLM generator...")
    generator = LLM(
        model=args.generator_model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
    )
    
    # Get tokenizer for chat template formatting
    from transformers import AutoTokenizer
    generator_tokenizer = AutoTokenizer.from_pretrained(args.generator_model_path, trust_remote_code=True)
    print(f"✓ Loaded tokenizer")
    
    # Llama-3 special token IDs for proper stopping
    stop_token_ids = [128001, 128009]  # <|end_of_text|>, <|eot_id|>
    
    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=0.0,
        repetition_penalty=1.1,
        stop_token_ids=stop_token_ids,
        skip_special_tokens=True,
    )
    
    # Format prompts using tokenizer.apply_chat_template
    print("Formatting prompts with tokenizer.apply_chat_template...")
    prompts = [
        format_prompt_optimized(q, docs, generator_tokenizer, args.dataset_name)
        for q, docs in zip(questions, refined_results)
    ]
    
    # --- Generate answers ---
    print("Starting answer generation...")
    generator_start_time = time.time()
    outputs = generator.generate(prompts, sampling_params)
    generator_time = time.time() - generator_start_time
    print(f"Generator completed in {generator_time:.2f} seconds")
    
    # Extract predictions (using robust extraction)
    predictions = [extract_answer_robust(output.outputs[0].text) for output in outputs]
    
    # --- Compute metrics ---
    print("Computing metrics...")
    metrics = compute_metrics(predictions, ground_truths)
    print(f"Results: {metrics}")
    
    # --- Calculate system metrics ---
    total_inference_time = refiner_time + generator_time
    
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    else:
        peak_vram_gb = 0.0
    
    avg_latency_ms = (total_inference_time / num_samples) * 1000 if num_samples > 0 else 0
    refiner_latency_ms = (refiner_time / num_samples) * 1000 if num_samples > 0 else 0
    generator_latency_ms = (generator_time / num_samples) * 1000 if num_samples > 0 else 0
    
    model_size_gb = get_model_size_gb(args.model_checkpoint_path)
    
    refiner_gflops, generator_gflops, total_gflops = calculate_inference_flops(
        refiner_model_path=args.base_refiner_model_path,
        generator_model_path=args.generator_model_path,
        avg_input_tokens=args.avg_input_tokens,
        avg_output_tokens=args.avg_output_tokens,
        avg_retrieval_docs=args.avg_retrieval_docs,
    )
    
    # --- Log to wandb ---
    if args.wandb_enabled:
        wandb.log({
            # System metrics
            "system/peak_vram_gb": peak_vram_gb,
            "system/total_inference_time_sec": total_inference_time,
            "system/avg_latency_ms_per_sample": avg_latency_ms,
            "system/refiner_latency_ms_per_sample": refiner_latency_ms,
            "system/generator_latency_ms_per_sample": generator_latency_ms,
            "system/total_gflops_per_query": total_gflops,
            "system/num_samples": num_samples,
            # Task metrics
            "task/em": metrics["em"],
            "task/f1": metrics["f1"],
        })
        
        wandb.summary.update({
            "inference/peak_vram_gb": peak_vram_gb,
            "inference/avg_latency_ms_per_sample": avg_latency_ms,
            "inference/total_gflops_per_query": total_gflops,
            "final/em": metrics["em"],
            "final/f1": metrics["f1"],
            "experiment_type": args.experiment_type,
            "dataset_name": args.dataset_name,
        })
        
        if model_size_gb > 0:
            wandb.summary["inference/model_size_gb"] = model_size_gb
        
        wandb.finish()
        print("✅ Wandb logging completed!")
    
    # --- Save results ---
    os.makedirs(args.save_dir, exist_ok=True)
    result_file = os.path.join(
        args.save_dir, f"eval_result_{args.experiment_type}_{args.dataset_name}.json"
    )
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({
            "config": vars(args),
            "task_metrics": metrics,
            "system_metrics": {
                "peak_vram_gb": peak_vram_gb,
                "avg_latency_ms_per_sample": avg_latency_ms,
                "total_inference_time_sec": total_inference_time,
                "total_gflops_per_query": total_gflops,
                "model_size_gb": model_size_gb,
            },
            "predictions": predictions[:100],  # Save first 100 for inspection
        }, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {result_file}")
    
    # --- Cleanup generator to avoid Bus error ---
    print("Cleaning up resources...")
    try:
        del generator
        del outputs
        del prompts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("✓ Resources cleaned up successfully")
    except Exception as e:
        print(f"Warning: Cleanup error (non-critical): {e}")
    
    # --- Print summary ---
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Experiment Type: {args.experiment_type}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Samples: {num_samples}")
    print("-" * 60)
    print("Task Metrics:")
    print(f"  EM (Exact Match): {metrics['em']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print("-" * 60)
    print("System Metrics:")
    print(f"  Peak VRAM: {peak_vram_gb:.2f} GB")
    print(f"  Avg Latency: {avg_latency_ms:.2f} ms/sample")
    print(f"  FLOPs per Query: {total_gflops:.2f} GFLOPs")
    if model_size_gb > 0:
        print(f"  Model Size: {model_size_gb:.2f} GB")
    print("=" * 60)
    
    return metrics


if __name__ == "__main__":
    import sys
    args = parse_args()
    exit_code = 0
    try:
        run(args)
        print("\n✅ Evaluation completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        exit_code = 130
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        # Ensure proper cleanup
        print("Performing final cleanup...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Exit cleanly to avoid vLLM cleanup issues
        sys.exit(exit_code)