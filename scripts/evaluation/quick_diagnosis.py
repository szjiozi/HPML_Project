"""
Quick Diagnosis Script for LongRefiner Evaluation Issues
Run this to quickly identify potential problems
"""
import json
import re
from collections import Counter

def improved_extract_answer(response: str) -> str:
    """Improved answer extraction with multiple fallback strategies."""
    if not response or not response.strip():
        return ""
    
    # Strategy 1: Look for explicit answer markers
    patterns = [
        r"So the final answer is[:\s]+(.+?)(?:\.|$|\n)",
        r"[Ff]inal answer is[:\s]+(.+?)(?:\.|$|\n)",
        r"[Tt]herefore[,\s]+(?:the answer is[:\s]+)?(.+?)(?:\.|$|\n)",
        r"[Tt]he answer is[:\s]+(.+?)(?:\.|$|\n)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            # Remove trailing notes
            if "(Note:" in answer:
                answer = answer.split("(Note:")[0].strip()
            # Take first sentence if too long
            if len(answer) > 100:
                answer = answer.split('.')[0].split('\n')[0].strip()
            return answer
    
    # Strategy 2: First non-empty line under 100 chars
    lines = response.split('\n')
    for line in lines[:5]:
        line = line.strip()
        if 0 < len(line) < 100 and not line.startswith(("Document", "Question:", "Response:")):
            # Remove common prefixes
            for prefix in ["Answer:", "Response:", "So", "Therefore"]:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip().lstrip(':,').strip()
            return line.split('.')[0].strip()
    
    # Strategy 3: First sentence if reasonable length
    first_sentence = response.split('.')[0].strip()
    if 0 < len(first_sentence) < 100:
        return first_sentence
    
    # Last resort: first 50 characters
    return response[:50].strip()


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        import string
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


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


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Compute exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def analyze_predictions(result_file: str):
    """Analyze prediction patterns and issues."""
    print("="*70)
    print("🔍 QUICK DIAGNOSIS REPORT")
    print("="*70)
    
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    predictions = data['predictions']
    
    print(f"\n📊 Basic Statistics:")
    print(f"   Total predictions: {len(predictions)}")
    
    # Analyze prediction characteristics
    empty_count = sum(1 for p in predictions if not p or not p.strip())
    very_long_count = sum(1 for p in predictions if len(p) > 400)
    has_final_answer = sum(1 for p in predictions if "final answer" in p.lower())
    has_repetition = sum(1 for p in predictions if len(p) > 100 and p.count(p[:50]) > 2)
    starts_with_none = sum(1 for p in predictions if p.strip().lower().startswith('none'))
    
    print(f"   Empty predictions: {empty_count} ({empty_count/len(predictions)*100:.1f}%)")
    print(f"   Very long (>400 chars): {very_long_count} ({very_long_count/len(predictions)*100:.1f}%)")
    print(f"   Contains 'final answer': {has_final_answer} ({has_final_answer/len(predictions)*100:.1f}%)")
    print(f"   Has obvious repetition: {has_repetition} ({has_repetition/len(predictions)*100:.1f}%)")
    print(f"   Starts with 'None': {starts_with_none} ({starts_with_none/len(predictions)*100:.1f}%)")
    
    # Try improved extraction
    print(f"\n🔧 Testing Improved Answer Extraction:")
    
    # Load ground truth (assume same order as predictions)
    # For now, we'll just show extraction improvement
    improved_preds = [improved_extract_answer(p) for p in predictions]
    
    print(f"   Average original length: {sum(len(p) for p in predictions)/len(predictions):.1f} chars")
    print(f"   Average extracted length: {sum(len(p) for p in improved_preds)/len(improved_preds):.1f} chars")
    
    # Show examples
    print(f"\n📝 Sample Comparisons (first 10):")
    for i in range(min(10, len(predictions))):
        print(f"\n[{i+1}]")
        print(f"   Original ({len(predictions[i])} chars): {predictions[i][:100]}...")
        print(f"   Extracted: {improved_preds[i]}")
    
    # Pattern analysis
    print(f"\n🔍 Common Patterns in Predictions:")
    pattern_counts = {
        "Says 'not in documents'": sum(1 for p in predictions if 'not in' in p.lower() and 'document' in p.lower()),
        "Says 'not available'": sum(1 for p in predictions if 'not available' in p.lower()),
        "Says 'not found'": sum(1 for p in predictions if 'not found' in p.lower()),
        "Contains 'However'": sum(1 for p in predictions if 'however' in p.lower()),
        "Contains '(Note:'": sum(1 for p in predictions if '(note:' in p.lower()),
        "Ends abruptly": sum(1 for p in predictions if len(p) > 100 and not p.strip().endswith(('.', '?', '!'))),
    }
    
    for pattern, count in pattern_counts.items():
        print(f"   {pattern}: {count} ({count/len(predictions)*100:.1f}%)")
    
    print("\n" + "="*70)
    print("✅ Diagnosis Complete!")
    print("="*70)
    print("\nKey Findings:")
    if very_long_count > len(predictions) * 0.3:
        print("   ⚠️  High rate of very long predictions - likely hitting max_tokens limit")
    if has_repetition > len(predictions) * 0.2:
        print("   ⚠️  Significant repetition detected - model may be stuck in loops")
    if has_final_answer < len(predictions) * 0.3:
        print("   ⚠️  Most predictions don't use required format - prompt issue?")
    if pattern_counts["Says 'not in documents'"] > len(predictions) * 0.2:
        print("   ⚠️  Model frequently claims information not in documents - Refiner issue?")
    
    print("\nNext Steps:")
    print("   1. Check Refiner output quality (are answers in refined docs?)")
    print("   2. Run baseline without Refiner")
    print("   3. Test with Base model (3B) instead of QLoRA")
    print("   4. Improve answer extraction function")
    print("\nSee EVALUATION_ANALYSIS.md for detailed investigation plan.")


if __name__ == "__main__":
    import sys
    
    result_file = "eval_result_qlora_hotpotqa.json"
    if len(sys.argv) > 1:
        result_file = sys.argv[1]
    
    try:
        analyze_predictions(result_file)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {result_file}")
        print("Usage: python quick_diagnosis.py [result_file.json]")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


