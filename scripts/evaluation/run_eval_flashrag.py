"""
Evaluation script for LongRefiner using FlashRAG
Uses the exact prompts from paper (arXiv:2505.10413, Appendix D)

This script uses FlashRAG's infrastructure (Config, PromptTemplate, Evaluator).
For standalone vLLM-only evaluation, use run_eval.py instead.

Key fix: Uses Config(config_dict=config_dict) instead of Config(config_dict)
to match FlashRAG 0.3.0 API.

Usage:
    python run_eval_flashrag.py \
        --dataset_name hotpotqa \
        --retrieval_result_path eval_data/hotpotqa_eval_1k_retrieval_result.json \
        --test_sample_num 1000 \
        --wandb_enabled
"""
import os
import gc
import json
import argparse
import time

import torch
import wandb
from longrefiner import LongRefiner
from flashrag.config import Config
from flashrag.evaluator import Evaluator
from flashrag.utils import get_generator, get_dataset
from flashrag.prompt import PromptTemplate


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run evaluation with FlashRAG infrastructure"
    )

    # Dataset settings
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Name of the dataset"
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Dataset split to use"
    )
    parser.add_argument(
        "--retrieval_result_path",
        type=str,
        required=True,
        help="Path to retrieval results JSON",
    )
    parser.add_argument(
        "--test_sample_num", type=int, default=1000, help="Number of test samples"
    )

    # Generator model settings
    parser.add_argument(
        "--generator_model",
        type=str,
        default="llama3.1-8B-instruct",
        help="Name of generator model",
    )
    parser.add_argument(
        "--generator_model_path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Path to generator model",
    )
    parser.add_argument(
        "--framework",
        type=str,
        default="vllm",
        help="Framework to use (vllm, hf, fschat)",
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.85, help="GPU memory ratio"
    )
    parser.add_argument(
        "--generator_max_input_len",
        type=int,
        default=15000,
        help="Maximum input length for generator",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=512, help="Max tokens to generate"
    )

    # Refiner model settings
    parser.add_argument(
        "--base_refiner_model_path",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Path to base refiner model",
    )
    parser.add_argument(
        "--query_analysis_module_lora_path",
        type=str,
        default="jinjiajie/Query-Analysis-Qwen2.5-3B-Instruct",
        help="Path to query analysis LoRA",
    )
    parser.add_argument(
        "--doc_structuring_module_lora_path",
        type=str,
        default="jinjiajie/Doc-Structuring-Qwen2.5-3B-Instruct",
        help="Path to doc structuring LoRA",
    )
    parser.add_argument(
        "--global_selection_module_lora_path",
        type=str,
        default="jinjiajie/Global-Selection-Qwen2.5-3B-Instruct",
        help="Path to global selection LoRA",
    )
    parser.add_argument(
        "--score_model_name",
        type=str,
        default="bge-reranker-v2-m3",
        help="Name of score model",
    )
    parser.add_argument(
        "--score_model_path",
        type=str,
        default="BAAI/bge-reranker-v2-m3",
        help="Path to score model",
    )

    # Output settings
    parser.add_argument(
        "--save_dir", type=str, default="results/", help="Directory to save results"
    )
    parser.add_argument(
        "--save_note", type=str, default="", help="Note to save with results"
    )
    parser.add_argument(
        "--gpu_id", type=str, default="0", help="GPU IDs to use"
    )

    # Experiment tracking
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="LongRefiner_Evaluation",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb_enabled", action="store_true", help="Enable wandb logging"
    )

    return parser.parse_args()


def get_official_prompts(dataset_name: str) -> tuple:
    """
    Get official prompts for the specified dataset.
    
    Reference: arXiv:2505.10413 (Appendix D, Prompt C.1 & C.2)
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        tuple: (system_prompt, user_prompt)
    """
    if dataset_name not in ["eli5", "asqa"]:
        # Short-form QA (NQ, HotpotQA, TriviaQA, etc.) - Prompt C.1
        system_prompt = (
            "Find the useful content from the provided documents, "
            "then answer the question. "
            "Answer the question directly. Your response should be very concise. "
            "Please provide use 'So the final answer is:' as a prefix "
            "for the final answer."
            "\nOutput format:\n"
            "Question: What is the capital of France?\n"
            "Response:The capital city of France is Paris."
            "So the final answer is: Paris."
            "\n\nThe following are given documents.\n\n{reference}"
        )
        user_prompt = (
            "Answer the question directly. Your response should be very concise. "
            "Please provide use 'So the final answer is:' as a prefix "
            "for the final answer.\n"
            "Question: {question}\nResponse: "
        )
    else:
        # Long-form QA (ELI5, ASQA) - Prompt C.2
        system_prompt = (
            "Find the useful content from the provided documents, "
            "then answer the question. "
            "Answer the question directly. Your response should be very detailed."
            "\n\nThe following are given documents.\n\n{reference}"
        )
        user_prompt = (
            "Answer the question directly. "
            "Your response should be very detailed.\n"
            "Question: {question}\nResponse: "
        )

    return system_prompt, user_prompt


def run(args):
    """
    Run evaluation using FlashRAG's Config API.
    
    Key fix: Uses Config(config_dict=config_dict) with keyword argument
    instead of Config(config_dict) to match FlashRAG 0.3.0 API.
    """
    print("=" * 60)
    print("LongRefiner Evaluation with FlashRAG")
    print("=" * 60)
    print("Using official prompts from paper (arXiv:2505.10413, Appendix D)")
    print("Answer extraction: split('So the final answer is:')[-1]")
    print("=" * 60)

    # Initialize wandb
    if args.wandb_enabled:
        wandb_mode = "online" if os.getenv("WANDB_API_KEY") else "offline"
        wandb.init(
            project=args.wandb_project,
            job_type="evaluation",
            name=f"flashrag_{args.dataset_name}",
            mode=wandb_mode,
        )
        wandb.config.update(vars(args))

    # Reset VRAM tracking
    if torch.cuda.is_available():
        torch.cuda.reset_max_memory_allocated()

    # Build FlashRAG config dictionary
    config_dict = {
        "generator_model": args.generator_model,
        "generator_model_path": args.generator_model_path,
        "gpu_id": args.gpu_id,
        "save_dir": args.save_dir,
        "framework": args.framework,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "generator_max_input_len": args.generator_max_input_len,
        "generation_params": {"max_tokens": args.max_tokens},
        "dataset_name": args.dataset_name,
        "test_sample_num": args.test_sample_num,
        "save_note": args.save_note,
    }

    # Initialize FlashRAG Config with correct API (keyword argument)
    print("Initializing FlashRAG Config...")
    config = Config(config_dict=config_dict)

    # Initialize FlashRAG generator
    print("Loading FlashRAG generator...")
    generator = get_generator(config)

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

    # Load dataset using FlashRAG
    print("Loading dataset...")
    all_split = get_dataset(config)
    data = all_split[args.split]

    # Load retrieval results
    with open(args.retrieval_result_path, "r") as f:
        retrieval_result = json.load(f)

    # Create PromptTemplate with official prompts
    system_prompt, user_prompt = get_official_prompts(args.dataset_name)
    template = PromptTemplate(config, system_prompt, user_prompt)

    # Process data
    questions = data.question
    retrieval_docs = [retrieval_result.get(question, []) for question in questions]

    # Run document refinement
    print("Running document refinement...")
    refiner_start = time.time()
    refined_results = refiner.batch_run(questions, retrieval_docs, budget=2048)
    refiner_time = time.time() - refiner_start
    print(f"Refiner completed in {refiner_time:.2f} seconds")

    # Format prompts using FlashRAG PromptTemplate
    input_prompts = [
        template.get_string(question, retrieval_result=docs)
        for question, docs in zip(questions, refined_results)
    ]

    # Generate answers
    print("Starting answer generation...")
    generator_start = time.time()
    output_list = generator.generate(input_prompts)
    generator_time = time.time() - generator_start
    print(f"Generator completed in {generator_time:.2f} seconds")

    # Update FlashRAG data object
    data.update_output("prompt", input_prompts)
    data.update_output("retrieval_results", retrieval_docs)
    data.update_output("refined_results", refined_results)
    data.update_output("raw_pred", output_list)

    # Extract final answers
    new_output_list = [
        i.split("So the final answer is:")[-1].strip() for i in output_list
    ]
    data.update_output("pred", new_output_list)

    # Evaluate using FlashRAG's Evaluator
    evaluator = Evaluator(config)
    result = evaluator.evaluate(data)
    print(f"\nFlashRAG Evaluation Result: {result}")

    # Calculate system metrics
    total_inference_time = refiner_time + generator_time

    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    else:
        peak_vram_gb = 0.0

    num_samples = len(questions)
    avg_latency_ms = (
        (total_inference_time / num_samples) * 1000 if num_samples > 0 else 0
    )

    # Log to wandb
    if args.wandb_enabled:
        wandb.log(
            {
                "system/peak_vram_gb": peak_vram_gb,
                "system/total_inference_time_sec": total_inference_time,
                "system/avg_latency_ms_per_sample": avg_latency_ms,
                "system/num_samples": num_samples,
                "task/result": result,
            }
        )
        wandb.finish()

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Result: {result}")
    print(f"Peak VRAM: {peak_vram_gb:.2f} GB")
    print(f"Avg Latency: {avg_latency_ms:.2f} ms/sample")
    print("=" * 60)

    # Cleanup
    refiner.shutdown()
    del refiner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    args = parse_args()

    try:
        result = run(args)
        print("\n✅ Evaluation completed!")
        return result
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    import sys
    result = main()
    sys.exit(0 if result is not None else 1)
