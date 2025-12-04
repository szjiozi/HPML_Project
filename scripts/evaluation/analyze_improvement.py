"""
Analyze the improvement from robust extraction
Compare original results with new results
"""
import json

def compare_results():
    """Compare old and new base results."""
    
    # Original results (from results folder, copied before changes)
    # You need to have the original file saved
    
    print("="*70)
    print("📊 EVALUATION IMPROVEMENT ANALYSIS")
    print("="*70)
    
    # Load new result
    with open("results/eval_result_base_hotpotqa.json", 'r') as f:
        new_data = json.load(f)
    
    new_em = new_data['task_metrics']['em']
    new_f1 = new_data['task_metrics']['f1']
    
    # Original metrics (from your first run)
    old_em = 0.034  # 3.4%
    old_f1 = 0.0932  # 9.32%
    
    print(f"\n📈 Base Model (3B) Performance:")
    print(f"{'='*70}")
    print(f"{'Metric':<15} {'Original':<15} {'New':<15} {'Change':<15}")
    print(f"{'-'*70}")
    print(f"{'EM':<15} {old_em*100:>6.2f}%        {new_em*100:>6.2f}%        {(new_em-old_em)*100:>+6.2f}%")
    print(f"{'F1':<15} {old_f1*100:>6.2f}%        {new_f1*100:>6.2f}%        {(new_f1-old_f1)*100:>+6.2f}%")
    
    print(f"\n💡 Analysis:")
    
    if new_em > old_em * 2:
        print(f"   ✅ Significant improvement! EM increased {new_em/old_em:.1f}x")
    elif new_em > old_em * 1.5:
        print(f"   ✓ Moderate improvement. EM increased {new_em/old_em:.1f}x")
    else:
        print(f"   ⚠️  Limited improvement. EM increased only {new_em/old_em:.1f}x")
    
    # Compare with smoke test expectations
    smoke_test_em = 0.26  # 26% from 50 samples
    
    print(f"\n🎯 Comparison with Smoke Test:")
    print(f"   Smoke Test (50 samples): {smoke_test_em*100:.1f}% EM")
    print(f"   Full Run (1000 samples): {new_em*100:.1f}% EM")
    print(f"   Difference: {(new_em - smoke_test_em)*100:.1f}%")
    
    if new_em < smoke_test_em * 0.7:
        print(f"\n⚠️  WARNING: Full results significantly lower than smoke test!")
        print(f"   Possible reasons:")
        print(f"   1. ChatML prompt caused issues (reverted now)")
        print(f"   2. First 50 samples were easier than average")
        print(f"   3. Some questions are particularly difficult")
        print(f"\n   📋 Next steps:")
        print(f"   1. Re-run with original prompt (already fixed)")
        print(f"   2. Analyze which questions are failing")
        print(f"   3. Check if answers are in refined documents")
    
    print(f"\n{'='*70}")
    print(f"Expected range after all fixes: 20-30% EM")
    print(f"Current: {new_em*100:.1f}% EM")
    
    if new_em < 0.20:
        print(f"Status: 🔴 Below expected range - needs investigation")
    elif new_em < 0.30:
        print(f"Status: 🟡 In expected range - acceptable")
    else:
        print(f"Status: 🟢 Above expected range - excellent!")
    print(f"{'='*70}")


if __name__ == "__main__":
    compare_results()

