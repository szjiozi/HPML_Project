"""
Find 10 correct and 10 incorrect answers from base model
Compare with LoRA and QLoRA predictions
Display in table format: #, Question, Ground Truth, Base, LoRA, QLoRA
"""
import json
import re
import string
from pathlib import Path
from typing import List, Dict, Tuple, Optional


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


def load_ground_truths(gt_file: str) -> List[List[str]]:
    """Load ground truth answers from JSONL file."""
    ground_truths = []
    with open(gt_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            gt = item.get('golden_answers', [])
            if isinstance(gt, str):
                gt = [gt]
            ground_truths.append(gt)
    return ground_truths


def load_questions(gt_file: str) -> List[str]:
    """Load questions from JSONL file."""
    questions = []
    with open(gt_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            questions.append(item.get('question', ''))
    return questions


def is_correct(prediction: str, ground_truths: List[str]) -> bool:
    """Check if prediction matches any ground truth."""
    if not ground_truths:
        return False
    return any(exact_match_score(prediction, gt) > 0.0 for gt in ground_truths)


def load_result_file(result_file: Path) -> Dict:
    """Load predictions from a result file."""
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    model_name = data['config'].get('save_note', result_file.stem)
    predictions = data['predictions']
    
    return {
        'model_name': model_name,
        'predictions': predictions,
        'em_score': data['task_metrics'].get('em', 0.0),
        'f1_score': data['task_metrics'].get('f1', 0.0)
    }


def truncate_text(text: str, max_len: int = 50) -> str:
    """Truncate text to max_len characters."""
    if len(text) <= max_len:
        return text
    return text[:max_len-3] + "..."


def format_table_row(items: List[str], widths: List[int]) -> str:
    """Format a table row with specified column widths."""
    formatted_items = []
    for item, width in zip(items, widths):
        truncated = truncate_text(str(item), width)
        formatted_items.append(f"{truncated:<{width}}")
    return " | ".join(formatted_items)


def print_separator(widths: List[int]) -> str:
    """Print a separator line."""
    total_width = sum(widths) + (len(widths) - 1) * 3
    return "-" * total_width


def print_comparison_table(examples: List[Dict], title: str):
    """Print a comparison table with Base, LoRA, QLoRA predictions."""
    if not examples:
        print(f"   No {title.lower()} examples found.")
        return
    
    # Column widths: #, Question, Ground Truth, Base, LoRA, QLoRA
    col_widths = [4, 45, 25, 30, 30, 30]
    
    print(f"\n{title} (showing {len(examples)} examples):")
    print(print_separator(col_widths))
    print(format_table_row(["#", "Question", "Ground Truth", "Base", "LoRA", "QLoRA"], col_widths))
    print(print_separator(col_widths))
    
    for ex in examples:
        # Mark correct predictions with ✓ and incorrect with ✗
        base_pred = ex['base_pred']
        lora_pred = ex['lora_pred']
        qlora_pred = ex['qlora_pred']
        gt = ex['ground_truth']
        
        # Check if each prediction is correct
        base_correct = is_correct(base_pred, [gt])
        lora_correct = is_correct(lora_pred, [gt])
        qlora_correct = is_correct(qlora_pred, [gt])
        
        # Add markers
        base_marker = "✓ " if base_correct else "✗ "
        lora_marker = "✓ " if lora_correct else "✗ "
        qlora_marker = "✓ " if qlora_correct else "✗ "
        
        print(format_table_row([
            str(ex['index']),
            ex['question'],
            gt,
            base_marker + base_pred,
            lora_marker + lora_pred,
            qlora_marker + qlora_pred
        ], col_widths))


def find_base_examples(base_predictions: List[str], ground_truths: List[List[str]], 
                       questions: List[str], num_samples: int = 10) -> Tuple[List[int], List[int]]:
    """Find indices of correct and incorrect examples from base model."""
    correct_indices = []
    incorrect_indices = []
    
    for idx, (pred, gts) in enumerate(zip(base_predictions, ground_truths)):
        if idx >= len(questions):
            break
        
        is_correct_answer = is_correct(pred, gts)
        
        if is_correct_answer and len(correct_indices) < num_samples:
            correct_indices.append(idx)
        elif not is_correct_answer and len(incorrect_indices) < num_samples:
            incorrect_indices.append(idx)
        
        if len(correct_indices) >= num_samples and len(incorrect_indices) >= num_samples:
            break
    
    return correct_indices, incorrect_indices


def main():
    """Main function to compare models based on base model results."""
    print("=" * 120)
    print("🔍 COMPARING MODELS BASED ON BASE MODEL RESULTS")
    print("=" * 120)
    
    # Load ground truth and questions
    gt_file = "eval_data/hotpotqa_eval_1k.jsonl"
    print(f"\n📂 Loading ground truth: {gt_file}")
    ground_truths = load_ground_truths(gt_file)
    questions = load_questions(gt_file)
    print(f"   Loaded {len(ground_truths)} ground truth entries")
    
    # Load result files
    results_dir = Path("results")
    base_file = results_dir / "eval_result_base_hotpotqa.json"
    lora_file = results_dir / "eval_result_lora_hotpotqa.json"
    qlora_file = results_dir / "eval_result_qlora_hotpotqa.json"
    
    # Check if files exist
    if not base_file.exists():
        print(f"\n❌ Base result file not found: {base_file}")
        return
    
    print(f"\n📊 Loading result files:")
    print(f"   - Base: {base_file.name}")
    base_data = load_result_file(base_file)
    
    lora_data = None
    if lora_file.exists():
        print(f"   - LoRA: {lora_file.name}")
        lora_data = load_result_file(lora_file)
    else:
        print(f"   ⚠️  LoRA file not found: {lora_file.name}")
    
    qlora_data = None
    if qlora_file.exists():
        print(f"   - QLoRA: {qlora_file.name}")
        qlora_data = load_result_file(qlora_file)
    else:
        print(f"   ⚠️  QLoRA file not found: {qlora_file.name}")
    
    # Find base model's correct and incorrect examples
    print(f"\n🔬 Analyzing base model results...")
    correct_indices, incorrect_indices = find_base_examples(
        base_data['predictions'], 
        ground_truths, 
        questions, 
        num_samples=10
    )
    print(f"   ✓ Found {len(correct_indices)} correct and {len(incorrect_indices)} incorrect examples")
    
    # Build comparison examples
    correct_examples = []
    for idx in correct_indices:
        example = {
            'index': idx,
            'question': questions[idx],
            'ground_truth': ground_truths[idx][0] if ground_truths[idx] else 'N/A',
            'base_pred': base_data['predictions'][idx] if idx < len(base_data['predictions']) else 'N/A',
            'lora_pred': lora_data['predictions'][idx] if lora_data and idx < len(lora_data['predictions']) else 'N/A',
            'qlora_pred': qlora_data['predictions'][idx] if qlora_data and idx < len(qlora_data['predictions']) else 'N/A'
        }
        correct_examples.append(example)
    
    incorrect_examples = []
    for idx in incorrect_indices:
        example = {
            'index': idx,
            'question': questions[idx],
            'ground_truth': ground_truths[idx][0] if ground_truths[idx] else 'N/A',
            'base_pred': base_data['predictions'][idx] if idx < len(base_data['predictions']) else 'N/A',
            'lora_pred': lora_data['predictions'][idx] if lora_data and idx < len(lora_data['predictions']) else 'N/A',
            'qlora_pred': qlora_data['predictions'][idx] if qlora_data and idx < len(qlora_data['predictions']) else 'N/A'
        }
        incorrect_examples.append(example)
    
    # Display results
    print("\n" + "=" * 120)
    print("📋 COMPARISON RESULTS (Based on Base Model)")
    print("=" * 120)
    
    print(f"\nModel Performance:")
    print(f"   Base:  EM={base_data['em_score']*100:.1f}%, F1={base_data['f1_score']*100:.1f}%")
    if lora_data:
        print(f"   LoRA:  EM={lora_data['em_score']*100:.1f}%, F1={lora_data['f1_score']*100:.1f}%")
    if qlora_data:
        print(f"   QLoRA: EM={qlora_data['em_score']*100:.1f}%, F1={qlora_data['f1_score']*100:.1f}%")
    
    # Display correct answers (where base is correct)
    print_comparison_table(correct_examples, "CORRECT ANSWERS (Base Model)")
    
    # Display incorrect answers (where base is incorrect)
    print_comparison_table(incorrect_examples, "INCORRECT ANSWERS (Base Model)")
    

    print("\nLegend: ✓ = Correct, ✗ = Incorrect")

    print("\n" + "=" * 120)
    print("Analysis complete!")
    print("=" * 120)
if __name__ == "__main__":
    main()
