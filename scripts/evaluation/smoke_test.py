"""
Smoke Test for Improved Answer Extraction and Prompt Optimization
Tests the improvements on a small sample before full evaluation
"""
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from scripts.evaluation.run_eval import (
    extract_answer,
    extract_answer_robust,
    compute_metrics,
    normalize_answer
)

def test_extraction_improvement():
    """Test answer extraction on actual predictions"""
    print("="*70)
    print("🧪 SMOKE TEST: Answer Extraction Comparison")
    print("="*70)
    
    # Load actual predictions from results
    result_files = [
        "results/eval_result_qlora_hotpotqa.json",
        "eval_result_qlora_hotpotqa.json",  # Fallback if in root
    ]
    
    result_file = None
    for f in result_files:
        if os.path.exists(f):
            result_file = f
            break
    
    if not result_file:
        print("❌ Error: Could not find result file")
        return
    
    print(f"\n📂 Loading: {result_file}")
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    predictions = data['predictions']
    
    # Load ground truth
    gt_files = [
        "eval_data/hotpotqa_eval_1k.jsonl",
    ]
    
    gt_file = None
    for f in gt_files:
        if os.path.exists(f):
            gt_file = f
            break
    
    if not gt_file:
        print("❌ Error: Could not find ground truth file")
        return
    
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
    test_predictions = predictions[:test_size]
    test_ground_truths = ground_truths[:test_size]
    
    print(f"\n📊 Testing on {test_size} samples...")
    
    # Extract with old method
    old_extractions = [extract_answer(p) for p in test_predictions]
    
    # Extract with new method
    new_extractions = [extract_answer_robust(p) for p in test_predictions]
    
    # Compute metrics
    old_metrics = compute_metrics(old_extractions, test_ground_truths)
    new_metrics = compute_metrics(new_extractions, test_ground_truths)
    
    # Statistics
    old_avg_len = sum(len(e) for e in old_extractions) / len(old_extractions)
    new_avg_len = sum(len(e) for e in new_extractions) / len(new_extractions)
    
    print(f"\n{'='*70}")
    print("📈 RESULTS COMPARISON")
    print(f"{'='*70}")
    
    print(f"\n📏 Average Answer Length:")
    print(f"   Old Method: {old_avg_len:.1f} chars")
    print(f"   New Method: {new_avg_len:.1f} chars")
    print(f"   Improvement: {(old_avg_len - new_avg_len)/old_avg_len*100:.1f}% reduction")
    
    print(f"\n🎯 Metrics:")
    print(f"   EM (Old): {old_metrics['em']:.4f} ({old_metrics['em']*100:.1f}%)")
    print(f"   EM (New): {new_metrics['em']:.4f} ({new_metrics['em']*100:.1f}%)")
    print(f"   EM Gain: +{(new_metrics['em'] - old_metrics['em'])*100:.1f}%")
    
    print(f"\n   F1 (Old): {old_metrics['f1']:.4f} ({old_metrics['f1']*100:.1f}%)")
    print(f"   F1 (New): {new_metrics['f1']:.4f} ({new_metrics['f1']*100:.1f}%)")
    print(f"   F1 Gain: +{(new_metrics['f1'] - old_metrics['f1'])*100:.1f}%")
    
    # Show detailed examples
    print(f"\n{'='*70}")
    print("📝 DETAILED EXAMPLES (First 10)")
    print(f"{'='*70}")
    
    for i in range(min(10, test_size)):
        print(f"\n[{i+1}] Question length: {len(test_predictions[i])} chars")
        print(f"    Ground Truth: {test_ground_truths[i]}")
        print(f"    Old Extraction ({len(old_extractions[i])} chars): {old_extractions[i][:100]}...")
        print(f"    New Extraction ({len(new_extractions[i])} chars): {new_extractions[i]}")
        
        # Check if correct
        old_correct = any(
            normalize_answer(old_extractions[i]) == normalize_answer(gt) 
            for gt in test_ground_truths[i]
        )
        new_correct = any(
            normalize_answer(new_extractions[i]) == normalize_answer(gt) 
            for gt in test_ground_truths[i]
        )
        
        if not old_correct and new_correct:
            print(f"    ✅ FIXED! (was wrong, now correct)")
        elif old_correct and not new_correct:
            print(f"    ⚠️  REGRESSION! (was correct, now wrong)")
        elif old_correct and new_correct:
            print(f"    ✓ Still correct")
        else:
            print(f"    ✗ Still wrong")
    
    print(f"\n{'='*70}")
    print("✅ Smoke Test Complete!")
    print(f"{'='*70}")
    
    # Summary
    print(f"\n🎯 Summary:")
    if new_metrics['em'] > old_metrics['em'] * 1.5:
        print(f"   ✅ SIGNIFICANT IMPROVEMENT detected!")
        print(f"   ✅ EM improved by {(new_metrics['em'] - old_metrics['em'])*100:.1f}%")
        print(f"   ✅ Recommended: Proceed with full evaluation")
    elif new_metrics['em'] > old_metrics['em']:
        print(f"   ✓ Modest improvement detected")
        print(f"   ✓ EM improved by {(new_metrics['em'] - old_metrics['em'])*100:.1f}%")
        print(f"   → Consider testing with optimized prompt as well")
    else:
        print(f"   ⚠️  No improvement or regression")
        print(f"   → Review extraction logic or prompt design")
    
    return new_metrics['em'], new_metrics['f1']


if __name__ == "__main__":
    try:
        em, f1 = test_extraction_improvement()
        print(f"\n✅ Test completed: EM={em:.4f}, F1={f1:.4f}")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


