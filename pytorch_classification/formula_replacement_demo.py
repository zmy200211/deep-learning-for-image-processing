"""
Direct demonstration of the formula replacement.

This script shows the exact replacement of the old uncertainty formula
with the new Jensen-Shannon Divergence implementation.
"""

import torch
from uncertainty_utils import jensen_shannon_divergence_uncertainty


def demonstrate_formula_replacement():
    """
    Show the exact formula replacement with side-by-side comparison.
    """
    print("FORMULA REPLACEMENT DEMONSTRATION")
    print("=" * 50)
    
    # Sample predictions (replace with your actual pred_sup_l and pred_sup_r)
    torch.manual_seed(42)
    pred_sup_l = torch.randn(3, 4)  # 3 samples, 4 classes
    pred_sup_r = torch.randn(3, 4)
    
    print("Input predictions:")
    print(f"pred_sup_l: {pred_sup_l}")
    print(f"pred_sup_r: {pred_sup_r}")
    print()
    
    # OLD FORMULA (to be replaced)
    print("OLD FORMULA:")
    print("uncertainty_l = abs(torch.max(torch.softmax(pred_sup_l, dim=1), dim=1)[0] - \\")
    print("                   torch.max(torch.softmax(pred_sup_r, dim=1), dim=1)[0])")
    print()
    
    # Execute old formula
    old_uncertainty_l = abs(torch.max(torch.softmax(pred_sup_l, dim=1), dim=1)[0] - 
                           torch.max(torch.softmax(pred_sup_r, dim=1), dim=1)[0])
    
    print(f"Old result: {old_uncertainty_l}")
    print()
    
    # NEW FORMULA (Jensen-Shannon Divergence)
    print("NEW FORMULA:")
    print("from uncertainty_utils import jensen_shannon_divergence_uncertainty")
    print("uncertainty_l = jensen_shannon_divergence_uncertainty(pred_sup_l, pred_sup_r)")
    print()
    
    # Execute new formula
    new_uncertainty_l = jensen_shannon_divergence_uncertainty(pred_sup_l, pred_sup_r)
    
    print(f"New result: {new_uncertainty_l}")
    print()
    
    # Show the mathematical steps of the new formula
    print("MATHEMATICAL STEPS OF NEW FORMULA:")
    print("1. Convert logits to probabilities:")
    ps = torch.softmax(pred_sup_l, dim=1)
    pt = torch.softmax(pred_sup_r, dim=1)
    print(f"   ps = softmax(pred_sup_l) = {ps}")
    print(f"   pt = softmax(pred_sup_r) = {pt}")
    print()
    
    print("2. Calculate mixture distribution:")
    M = 0.5 * (ps + pt)
    print(f"   M = 0.5 * (ps + pt) = {M}")
    print()
    
    print("3. Calculate KL divergences:")
    eps = 1e-10
    ps_safe = ps + eps
    pt_safe = pt + eps
    M_safe = M + eps
    
    kl_ps_M = torch.sum(ps * torch.log(ps_safe / M_safe), dim=1)
    kl_pt_M = torch.sum(pt * torch.log(pt_safe / M_safe), dim=1)
    print(f"   KL(ps || M) = {kl_ps_M}")
    print(f"   KL(pt || M) = {kl_pt_M}")
    print()
    
    print("4. Final Jensen-Shannon Divergence:")
    jsd = 0.5 * kl_ps_M + 0.5 * kl_pt_M
    print(f"   JSD = 0.5 * KL(ps || M) + 0.5 * KL(pt || M) = {jsd}")
    print()
    
    # Verify consistency
    print("VERIFICATION:")
    print(f"Manual calculation matches function: {torch.allclose(jsd, new_uncertainty_l)}")
    print()
    
    # Show the difference in values
    print("COMPARISON:")
    print(f"Old method values: {old_uncertainty_l.tolist()}")
    print(f"New method values: {new_uncertainty_l.tolist()}")
    print(f"Average old: {old_uncertainty_l.mean():.6f}")
    print(f"Average new: {new_uncertainty_l.mean():.6f}")


def show_simple_replacement():
    """
    Show the simplest way to replace the old code.
    """
    print("\n" + "=" * 50)
    print("SIMPLE CODE REPLACEMENT")
    print("=" * 50)
    
    print("BEFORE (OLD CODE):")
    print("---")
    print("uncertainty_l = abs(torch.max(torch.softmax(pred_sup_l, dim=1), dim=1)[0] - ")
    print("                   torch.max(torch.softmax(pred_sup_r, dim=1), dim=1)[0])")
    print()
    
    print("AFTER (NEW CODE):")
    print("---")
    print("from uncertainty_utils import uncertainty_l")
    print("uncertainty_l_result = uncertainty_l(pred_sup_l, pred_sup_r)")
    print()
    
    print("That's it! The function name remains the same for easy migration.")


if __name__ == "__main__":
    try:
        demonstrate_formula_replacement()
        show_simple_replacement()
        print("\nFormula replacement demonstration completed!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()