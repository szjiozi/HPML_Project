"""
Simplified Smoke Test - No heavy dependencies
Tests answer extraction improvement on existing results
"""
import json
import re
import string
from collections import Counter

# =============================================================================
# Copy the extraction functions
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


def compute_metrics(predictions, ground_truths):
    """Compute EM and F1 metrics."""
    em_scores = []
    f1_scores = []
    
    for pred, gts in zip(predictions, ground_truths):
        if isinstance(gts, str):
            gts = [gts]
        
        em = max(exact_match_score(pred, gt) for gt in gts) if gts else 0.0
        f1 = max(f1_score(pred, gt) for gt in gts) if gts else 0.0
        
        em_scores.append(em)
        f1_scores.append(f1)
    
    return {
        "em": sum(em_scores) / len(em_scores) if em_scores else 0.0,
        "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
    }


def extract_answer(response: str) -> str:
    """OLD: Original extraction function."""
    if "So the final answer is:" in response:
        return response.split("So the final answer is:")[-1].strip()
    return response.strip()


def extract_answer_robust(response: str) -> str:
    """NEW: Robust answer extraction with cascading strategies."""
    if not response:
        return ""
        
    # --- Stage 0: Preprocessing & Noise Removal ---
    clean_text = re.sub(r'[\(\[]Note:.*?[\)\]]', '', response, flags=re.IGNORECASE | re.DOTALL)
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
    if len(clean_text) < 50:
        return clean_text.rstrip(string.punctuation)

    sentences = re.split(r'(?<=[.!?])\s+', clean_text)
    if sentences:
        first_sentence = sentences[0].strip()
        if len(first_sentence) < 100:
            first_sentence = re.sub(
                r'^(The answer is|The capital is|It is)\s+', 
                '', 
                first_sentence, 
                flags=re.IGNORECASE
            )
            return first_sentence.rstrip(string.punctuation)

    if len(sentences) > 1:
        last_sentence = sentences[-1].strip()
        if len(last_sentence) < 100:
            return last_sentence.rstrip(string.punctuation)

    return clean_text[:50].strip()


# =============================================================================
# Test
# =============================================================================

def main():
    print("="*70)
    print("🧪 SMOKE TEST: Answer Extraction Improvement")
    print("="*70)
    
    # Load results
    result_file = "results/eval_result_qlora_hotpotqa.json"
    print(f"\n📂 Loading: {result_file}")
    
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    predictions = data['predictions']
    
    # Load ground truth
    gt_file = "eval_data/hotpotqa_eval_1k.jsonl"
    print(f"📂 Loading: {gt_file}")
    
    ground_truths = []
    with open(gt_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= len(predictions):
                break
            item = json.loads(line)
            gt = item.get('golden_answers', [])
            ground_truths.append(gt if isinstance(gt, list) else [gt])
    
    # Test on first 50 samples
    test_size = min(50, len(predictions))
    test_preds = predictions[:test_size]
    test_gts = ground_truths[:test_size]
    
    print(f"\n📊 Testing on {test_size} samples...")
    
    # Extract answers
    old_extractions = [extract_answer(p) for p in test_preds]
    new_extractions = [extract_answer_robust(p) for p in test_preds]
    
    # Compute metrics
    old_metrics = compute_metrics(old_extractions, test_gts)
    new_metrics = compute_metrics(new_extractions, test_gts)
    
    # Statistics
    old_avg_len = sum(len(e) for e in old_extractions) / len(old_extractions)
    new_avg_len = sum(len(e) for e in new_extractions) / len(new_extractions)
    
    print(f"\n{'='*70}")
    print("📈 RESULTS COMPARISON")
    print(f"{'='*70}")
    
    print(f"\n📏 Average Answer Length:")
    print(f"   Old Method: {old_avg_len:.1f} chars")
    print(f"   New Method: {new_avg_len:.1f} chars")
    print(f"   Reduction: {(old_avg_len - new_avg_len)/old_avg_len*100:.1f}%")
    
    print(f"\n🎯 Metrics:")
    print(f"   EM (Old): {old_metrics['em']:.4f} ({old_metrics['em']*100:.1f}%)")
    print(f"   EM (New): {new_metrics['em']:.4f} ({new_metrics['em']*100:.1f}%)")
    print(f"   EM Gain: +{(new_metrics['em'] - old_metrics['em'])*100:.2f}%")
    
    print(f"\n   F1 (Old): {old_metrics['f1']:.4f} ({old_metrics['f1']*100:.1f}%)")
    print(f"   F1 (New): {new_metrics['f1']:.4f} ({new_metrics['f1']*100:.1f}%)")
    print(f"   F1 Gain: +{(new_metrics['f1'] - old_metrics['f1'])*100:.2f}%")
    
    # Show examples
    print(f"\n{'='*70}")
    print("📝 EXAMPLES (First 10)")
    print(f"{'='*70}")
    
    for i in range(min(10, test_size)):
        print(f"\n[{i+1}] Original length: {len(test_preds[i])} chars")
        print(f"    GT: {test_gts[i]}")
        print(f"    Old ({len(old_extractions[i])} chars): {old_extractions[i][:80]}...")
        print(f"    New ({len(new_extractions[i])} chars): {new_extractions[i]}")
        
        old_correct = any(
            normalize_answer(old_extractions[i]) == normalize_answer(gt) 
            for gt in test_gts[i]
        )
        new_correct = any(
            normalize_answer(new_extractions[i]) == normalize_answer(gt) 
            for gt in test_gts[i]
        )
        
        if not old_correct and new_correct:
            print(f"    ✅ FIXED!")
        elif old_correct and not new_correct:
            print(f"    ⚠️  REGRESSION!")
        elif new_correct:
            print(f"    ✓ Still correct")
        else:
            print(f"    ✗ Still wrong")
    
    print(f"\n{'='*70}")
    print("✅ Smoke Test Complete!")
    print(f"{'='*70}")
    
    # Summary
    improvement = (new_metrics['em'] - old_metrics['em']) * 100
    print(f"\n🎯 Summary:")
    if improvement > 5:
        print(f"   ✅ SIGNIFICANT IMPROVEMENT: +{improvement:.1f}% EM")
        print(f"   ✅ Recommended: Apply to run_eval.py and run full evaluation")
    elif improvement > 0:
        print(f"   ✓ Modest improvement: +{improvement:.1f}% EM")
    else:
        print(f"   ⚠️  No improvement or regression: {improvement:.1f}% EM")
    
    print(f"\n💡 Next Steps:")
    if improvement > 0:
        print(f"   1. Update run_eval.py to use extract_answer_robust()")
        print(f"   2. (Optional) Test format_prompt_optimized() with ChatML")
        print(f"   3. Run full evaluation on 1000 samples")
    else:
        print(f"   1. Review extraction logic")
        print(f"   2. Test on more samples")


if __name__ == "__main__":
    main()




