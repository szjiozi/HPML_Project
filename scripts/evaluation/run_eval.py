import os
import json
import argparse
import time
from pathlib import Path

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
        description="Run evaluation script for question answering"
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Name of the dataset"
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Dataset split to use"
    )
    parser.add_argument(
        "--generator_model",
        type=str,
        default="llama3.1-8B-instruct",
        help="Name of the generator model",
    )
    parser.add_argument(
        "--generator_model_path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Path to the base model",
    )
    parser.add_argument(
        "--base_refiner_model_path",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Path to the base model",
    )
    parser.add_argument(
        "--query_analysis_module_lora_path",
        type=str,
        default="model/Qwen2.5-3B-Instruct-query-analysis",
        help="Path to the query analysis module lora",
    )
    parser.add_argument(
        "--doc_structuring_module_lora_path",
        type=str,
        default="model/Qwen2.5-3B-Instruct-doc-structuring",
        help="Path to the doc structuring module lora",
    )
    parser.add_argument(
        "--global_selection_module_lora_path",
        type=str,
        default="model/Qwen2.5-3B-Instruct-global-selection",
        help="Path to the global selection module lora",
    )
    parser.add_argument(
        "--score_model_name",
        type=str,
        default="bge-reranker-v2-m3",
        help="Name of the score model",
    )
    parser.add_argument(
        "--score_model_path",
        type=str,
        default="BAAI/bge-reranker-v2-m3",
        help="Path to the score model",
    )
    parser.add_argument(
        "--gpu_id", type=str, default="0,1,2,3,4,5,6,7", help="GPU IDs to use"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/",
        help="Directory to save results",
    )
    parser.add_argument(
        "--retrieval_result_path",
        type=str,
        default="sample_docs.json",
        help="Path to the all docs file",
    )
    parser.add_argument(
        "--framework", type=str, default="vllm", help="Framework to use"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.85,
        help="GPU memory utilization ratio",
    )
    parser.add_argument(
        "--generator_max_input_len",
        type=int,
        default=15000,
        help="Maximum input length for generator",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--test_sample_num", type=int, default=1000, help="Number of test samples"
    )
    parser.add_argument(
        "--save_note", type=str, default="", help="Note to save with results"
    )

    # --- New arguments for wandb and experiment tracking ---
    parser.add_argument(
        "--model_variant",
        type=str,
        default="A",
        choices=["A", "B", "C"],
        help="Model variant: A (Full LoRA), B (LoRA+PTQ), C (QLoRA)",
    )
    parser.add_argument(
        "--quantization_strategy",
        type=str,
        default="none",
        choices=["none", "int8", "int4", "nf4"],
        help="Quantization strategy used",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="LongRefiner_Evaluation",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb_enabled", action="store_true", help="Enable wandb logging"
    )
    parser.add_argument(
        "--model_checkpoint_path",
        type=str,
        default=None,
        help="Path to the model checkpoint for measuring model size",
    )
    return parser.parse_args()


def get_prompt_template(config, dataset_name):
    """Get prompt template for the specified dataset"""
    if dataset_name not in ["eli5", "asqa"]:
        system_prompt = (
            "Find the useful content from the provided documents, then answer the question. "
            "Answer the question directly. Your response should be very concise. Please provide use 'So the final answer is:' as a prefix for the final answer."
            "\nOutput format:\nQuestion: What is the capital of France?\nResponse:The capital city of France is Paris.So the final answer is: Paris.\n\nThe following are given documents.\n\n{reference}"
        )
        user_prompt = "Answer the question directly. Your response should be very concise. Please provide use 'So the final answer is:' as a prefix for the final answer.\nQuestion: {question}\nResponse: "
    else:
        system_prompt = (
            "Find the useful content from the provided documents, then answer the question. "
            "Answer the question directly. Your response should be very detailed."
            "\n\nThe following are given documents.\n\n{reference}"
        )
        user_prompt = "Answer the question directly. Your response should be very detailed.\nQuestion: {question}\nResponse: "

    return PromptTemplate(config, system_prompt, user_prompt)


def get_model_size_gb(checkpoint_path: str) -> float:
    """Calculate the total size of model checkpoint files in GB"""
    if checkpoint_path is None:
        return 0.0

    path = Path(checkpoint_path)
    if not path.exists():
        return 0.0

    total_size = 0
    if path.is_file():
        total_size = path.stat().st_size
    else:
        # Sum all files in directory (for sharded checkpoints)
        for file in path.rglob("*"):
            if file.is_file():
                total_size += file.stat().st_size

    return total_size / (1024**3)  # Convert to GB


def get_variant_name(variant: str) -> str:
    """Get descriptive name for model variant"""
    variant_names = {"A": "Full_LoRA_Baseline", "B": "LoRA_PTQ", "C": "QLoRA"}
    return variant_names.get(variant, variant)


def run(args):
    """Run evaluation pipeline"""

    # --- Initialize wandb ---
    if args.wandb_enabled:
        wandb_mode = "online" if os.getenv("WANDB_API_KEY") else "offline"
        wandb.init(
            project=args.wandb_project,
            job_type="evaluation",
            name=f"{get_variant_name(args.model_variant)}_{args.dataset_name}_{args.quantization_strategy}",
            mode=wandb_mode,
        )

        # Log experiment configuration
        wandb_config = {
            # Experiment settings
            "model_variant": args.model_variant,
            "model_variant_name": get_variant_name(args.model_variant),
            "quantization_strategy": args.quantization_strategy,
            "dataset_name": args.dataset_name,
            "split": args.split,
            "test_sample_num": args.test_sample_num,
            # Model paths
            "base_refiner_model_path": args.base_refiner_model_path,
            "query_analysis_module_lora_path": args.query_analysis_module_lora_path,
            "doc_structuring_module_lora_path": args.doc_structuring_module_lora_path,
            "global_selection_module_lora_path": args.global_selection_module_lora_path,
            "generator_model": args.generator_model,
            "generator_model_path": args.generator_model_path,
            # Generation settings
            "max_tokens": args.max_tokens,
            "generator_max_input_len": args.generator_max_input_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "framework": args.framework,
        }
        wandb.config.update(wandb_config)

    # Initialize configuration
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

    # --- Reset VRAM tracking ---
    if torch.cuda.is_available():
        torch.cuda.reset_max_memory_allocated()

    # --- Initialize components with timing ---
    print("Initializing components...")
    init_start_time = time.time()

    config = Config(config_dict)
    generator = get_generator(config)
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

    # Prepare data
    all_split = get_dataset(config)
    data = all_split[args.split]
    with open(args.retrieval_result_path, "r") as f:
        retrieval_result = json.load(f)

    # Get prompt template
    template = get_prompt_template(config, args.dataset_name)

    # Process data and generate answers
    questions = data.question
    retrieval_docs = [retrieval_result.get(question, []) for question in questions]
    num_samples = len(questions)

    print(f"Processing {num_samples} samples...")

    # --- Measure refiner inference time ---
    print("Running document refinement...")
    refiner_start_time = time.time()
    refined_result = refiner.batch_run(questions, retrieval_docs, budget=2048)
    refiner_time = time.time() - refiner_start_time
    print(f"Refiner completed in {refiner_time:.2f} seconds")

    input_prompts = [
        template.get_string(question, retrieval_result=docs)
        for question, docs in zip(questions, refined_result)
    ]

    # --- Measure generator inference time ---
    print("Starting answer generation...")
    generator_start_time = time.time()
    output_list = generator.generate(input_prompts)
    generator_time = time.time() - generator_start_time
    print(f"Generator completed in {generator_time:.2f} seconds")

    # Total inference time
    total_inference_time = refiner_time + generator_time

    # Update data with outputs
    data.update_output("prompt", input_prompts)
    data.update_output("retrieval_results", retrieval_docs)
    data.update_output("refined_results", refined_result)
    data.update_output("raw_pred", output_list)
    new_output_list = [
        i.split("So the final answer is:")[-1].strip() for i in output_list
    ]
    data.update_output("pred", new_output_list)

    # --- Evaluate and get task metrics ---
    evaluator = Evaluator(config)
    result = evaluator.evaluate(data)
    print(result)
    print("------------------------\n")

    # --- Calculate system metrics ---
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    else:
        peak_vram_gb = 0.0

    avg_latency_ms = (
        (total_inference_time / num_samples) * 1000 if num_samples > 0 else 0
    )
    refiner_latency_ms = (refiner_time / num_samples) * 1000 if num_samples > 0 else 0
    generator_latency_ms = (
        (generator_time / num_samples) * 1000 if num_samples > 0 else 0
    )

    # Model size
    model_size_gb = get_model_size_gb(args.model_checkpoint_path)

    # --- Log to wandb ---
    if args.wandb_enabled:
        # System metrics
        system_metrics = {
            "system/peak_vram_gb": peak_vram_gb,
            "system/total_inference_time_sec": total_inference_time,
            "system/refiner_time_sec": refiner_time,
            "system/generator_time_sec": generator_time,
            "system/avg_latency_ms_per_sample": avg_latency_ms,
            "system/refiner_latency_ms_per_sample": refiner_latency_ms,
            "system/generator_latency_ms_per_sample": generator_latency_ms,
            "system/model_init_time_sec": init_time,
            "system/num_samples": num_samples,
        }

        if model_size_gb > 0:
            system_metrics["system/model_size_gb"] = model_size_gb

        wandb.log(system_metrics)

        # Task metrics (from FlashRAG evaluator result)
        # Result format is typically: {'em': 0.xxx, 'f1': 0.xxx, ...}
        task_metrics = {}
        if isinstance(result, dict):
            for metric_name, metric_value in result.items():
                # Normalize metric names
                if metric_name.lower() in ["em", "exact_match", "acc", "accuracy"]:
                    task_metrics["task/accuracy"] = metric_value
                elif metric_name.lower() == "f1":
                    task_metrics["task/f1"] = metric_value
                else:
                    task_metrics[f"task/{metric_name}"] = metric_value

        wandb.log(task_metrics)

        # Update summary with final metrics
        wandb.summary.update(
            {
                # System metrics summary
                "inference/peak_vram_gb": peak_vram_gb,
                "inference/avg_latency_ms_per_sample": avg_latency_ms,
                "inference/total_time_sec": total_inference_time,
                # Task metrics summary
                **{k.replace("task/", "final/"): v for k, v in task_metrics.items()},
                # Experiment info
                "model_variant": args.model_variant,
                "quantization_strategy": args.quantization_strategy,
                "dataset_name": args.dataset_name,
                "num_samples": num_samples,
            }
        )

        if model_size_gb > 0:
            wandb.summary["inference/model_size_gb"] = model_size_gb

        # Save results as artifact
        result_artifact = wandb.Artifact(
            name=f"eval_result_{args.model_variant}_{args.dataset_name}",
            type="evaluation_result",
            metadata={
                "model_variant": args.model_variant,
                "quantization_strategy": args.quantization_strategy,
                "dataset_name": args.dataset_name,
            },
        )

        # Save detailed results to file
        result_file = os.path.join(
            args.save_dir, f"eval_result_{args.model_variant}_{args.dataset_name}.json"
        )
        os.makedirs(args.save_dir, exist_ok=True)
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": wandb_config,
                    "task_metrics": (
                        result if isinstance(result, dict) else {"raw": str(result)}
                    ),
                    "system_metrics": {
                        "peak_vram_gb": peak_vram_gb,
                        "avg_latency_ms_per_sample": avg_latency_ms,
                        "total_inference_time_sec": total_inference_time,
                        "model_size_gb": model_size_gb,
                    },
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        result_artifact.add_file(result_file)
        wandb.log_artifact(result_artifact)

        # Finish wandb run
        wandb.finish()
        print("✅ Wandb logging completed!")

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Model Variant: {get_variant_name(args.model_variant)}")
    print(f"Quantization: {args.quantization_strategy}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Samples: {num_samples}")
    print("-" * 50)
    print("System Metrics:")
    print(f"  Peak VRAM: {peak_vram_gb:.2f} GB")
    print(f"  Avg Latency: {avg_latency_ms:.2f} ms/sample")
    if model_size_gb > 0:
        print(f"  Model Size: {model_size_gb:.2f} GB")
    print("-" * 50)
    print("Task Metrics:")
    print(f"  {result}")
    print("=" * 50)


if __name__ == "__main__":
    args = parse_args()
    run(args)
