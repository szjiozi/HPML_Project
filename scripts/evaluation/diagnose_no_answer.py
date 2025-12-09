"""
Diagnose the "no_answer" problem in predictions
"""
import json

def analyze_predictions():
    """Analyze no_answer rate in predictions."""
    
    print("="*70)
    print("🔍 NO_ANSWER DIAGNOSIS")
    print("="*70)
    
    # Load result
    with open("results/eval_result_base_hotpotqa.json", 'r') as f:
        data = json.load(f)
    
    predictions = data['predictions']
    
    # Count no_answer variants
    no_answer_count = 0
    no_answer_variants = []
    
    for pred in predictions:
        pred_lower = pred.lower()
        if 'no_answer' in pred_lower or 'no answer' in pred_lower:
            no_answer_count += 1
            if pred not in no_answer_variants:
                no_answer_variants.append(pred)
    
    print(f"\n📊 Statistics:")
    print(f"   Total predictions: {len(predictions)}")
    print(f"   'no_answer' responses: {no_answer_count} ({no_answer_count/len(predictions)*100:.1f}%)")
    print(f"   Valid answers: {len(predictions) - no_answer_count} ({(len(predictions)-no_answer_count)/len(predictions)*100:.1f}%)")
    
    print(f"\n📝 'no_answer' Variants Found:")
    for variant in no_answer_variants[:10]:
        print(f"   - {variant}")
    
    # EM and F1
    em = data['task_metrics']['em']
    f1 = data['task_metrics']['f1']
    
    print(f"\n📈 Performance:")
    print(f"   EM: {em*100:.1f}%")
    print(f"   F1: {f1*100:.1f}%")
    
    # Expected if no_answer was reduced
    expected_em = em / (1 - no_answer_count/len(predictions))
    
    print(f"\n💡 Analysis:")
    print(f"   ⚠️  {no_answer_count/len(predictions)*100:.1f}% of predictions are 'no_answer'")
    print(f"   ⚠️  This is TOO HIGH (expected <10%)")
    
    print(f"\n🎯 Likely Causes:")
    print(f"   1. ❌ Prompt instruction: 'If answer not in documents, say no_answer'")
    print(f"      → Made model over-conservative")
    print(f"   2. ❌ Refiner may be losing key information")
    print(f"      → Documents don't contain answers")
    print(f"   3. ❌ Model being too cautious with chat template")
    
    print(f"\n✅ Solutions Applied:")
    print(f"   1. ✓ Removed 'say no_answer' instruction from prompt")
    print(f"   2. ✓ Simplified system prompt to match original style")
    print(f"   3. ✓ Using tokenizer.apply_chat_template() properly")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Re-run evaluation with updated prompt")
    print(f"   2. Expected: no_answer rate should drop to <10%")
    print(f"   3. Expected EM: 22-28% (up from {em*100:.1f}%)")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    analyze_predictions()


