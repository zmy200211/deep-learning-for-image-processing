"""
Test script for uncertainty calculation utilities.

This script demonstrates the difference between the old absolute difference method
and the new Jensen-Shannon Divergence method for inter-model uncertainty calculation.
"""

import torch
import numpy as np
from uncertainty_utils import (
    old_uncertainty_calculation,
    jensen_shannon_divergence_uncertainty,
    calculate_uncertainty,
    uncertainty_l
)


def test_uncertainty_methods():
    """Test and compare different uncertainty calculation methods."""
    print("Testing Uncertainty Calculation Methods")
    print("=" * 50)
    
    # Set seed for reproducible results
    torch.manual_seed(42)
    
    # Create sample predictions (batch_size=4, num_classes=5)
    batch_size, num_classes = 4, 5
    pred_sup_l = torch.randn(batch_size, num_classes)
    pred_sup_r = torch.randn(batch_size, num_classes)
    
    print(f"Input shapes: pred_sup_l={pred_sup_l.shape}, pred_sup_r={pred_sup_r.shape}")
    print(f"Sample predictions:")
    print(f"pred_sup_l:\n{pred_sup_l}")
    print(f"pred_sup_r:\n{pred_sup_r}")
    print()
    
    # Calculate uncertainties using both methods
    old_uncertainty = old_uncertainty_calculation(pred_sup_l, pred_sup_r)
    jsd_uncertainty = jensen_shannon_divergence_uncertainty(pred_sup_l, pred_sup_r)
    
    print("Uncertainty Results:")
    print(f"Old method (abs max diff): {old_uncertainty}")
    print(f"New method (Jensen-Shannon): {jsd_uncertainty}")
    print()
    
    # Test the unified interface
    jsd_unified = calculate_uncertainty(pred_sup_l, pred_sup_r, method='jsd')
    abs_unified = calculate_uncertainty(pred_sup_l, pred_sup_r, method='abs_diff')
    
    print("Unified Interface Results:")
    print(f"JSD via unified interface: {jsd_unified}")
    print(f"Abs diff via unified interface: {abs_unified}")
    print()
    
    # Test backward compatibility function
    uncertainty_new = uncertainty_l(pred_sup_l, pred_sup_r)
    print(f"Backward compatibility function: {uncertainty_new}")
    print()
    
    # Verify numerical consistency
    print("Verification:")
    print(f"JSD methods match: {torch.allclose(jsd_uncertainty, jsd_unified)}")
    print(f"Abs diff methods match: {torch.allclose(old_uncertainty, abs_unified)}")
    print(f"Backward compatibility matches JSD: {torch.allclose(jsd_uncertainty, uncertainty_new)}")
    print()


def test_edge_cases():
    """Test edge cases for the uncertainty calculations."""
    print("Testing Edge Cases")
    print("=" * 30)
    
    # Test with identical predictions (should give zero uncertainty)
    pred_identical = torch.randn(2, 3)
    old_unc_identical = old_uncertainty_calculation(pred_identical, pred_identical)
    jsd_unc_identical = jensen_shannon_divergence_uncertainty(pred_identical, pred_identical)
    
    print("Identical predictions:")
    print(f"Old method uncertainty: {old_unc_identical}")
    print(f"JSD uncertainty: {jsd_unc_identical}")
    print(f"JSD close to zero: {torch.allclose(jsd_unc_identical, torch.zeros_like(jsd_unc_identical), atol=1e-6)}")
    print()
    
    # Test with very confident predictions
    pred_confident_l = torch.tensor([[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0]])
    pred_confident_r = torch.tensor([[-10.0, 10.0, -10.0], [10.0, -10.0, -10.0]])
    
    old_unc_confident = old_uncertainty_calculation(pred_confident_l, pred_confident_r)
    jsd_unc_confident = jensen_shannon_divergence_uncertainty(pred_confident_l, pred_confident_r)
    
    print("Confident but different predictions:")
    print(f"pred_confident_l:\n{pred_confident_l}")
    print(f"pred_confident_r:\n{pred_confident_r}")
    print(f"Old method uncertainty: {old_unc_confident}")
    print(f"JSD uncertainty: {jsd_unc_confident}")
    print()
    
    # Test with uniform distributions
    pred_uniform = torch.zeros(2, 4)  # All logits are 0, uniform after softmax
    pred_peaked = torch.tensor([[5.0, 0.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0]])
    
    old_unc_uniform = old_uncertainty_calculation(pred_uniform, pred_peaked)
    jsd_unc_uniform = jensen_shannon_divergence_uncertainty(pred_uniform, pred_peaked)
    
    print("Uniform vs peaked predictions:")
    print(f"Uniform softmax: {torch.softmax(pred_uniform, dim=1)}")
    print(f"Peaked softmax: {torch.softmax(pred_peaked, dim=1)}")
    print(f"Old method uncertainty: {old_unc_uniform}")
    print(f"JSD uncertainty: {jsd_unc_uniform}")
    print()


def demonstrate_mathematical_properties():
    """Demonstrate mathematical properties of Jensen-Shannon Divergence."""
    print("Mathematical Properties of Jensen-Shannon Divergence")
    print("=" * 55)
    
    torch.manual_seed(123)
    pred_a = torch.randn(3, 4)
    pred_b = torch.randn(3, 4)
    
    # Symmetry: JSD(P, Q) = JSD(Q, P)
    jsd_ab = jensen_shannon_divergence_uncertainty(pred_a, pred_b)
    jsd_ba = jensen_shannon_divergence_uncertainty(pred_b, pred_a)
    
    print("Symmetry Property:")
    print(f"JSD(A, B): {jsd_ab}")
    print(f"JSD(B, A): {jsd_ba}")
    print(f"Symmetric: {torch.allclose(jsd_ab, jsd_ba)}")
    print()
    
    # Non-negativity: JSD(P, Q) >= 0
    print("Non-negativity Property:")
    print(f"All JSD values >= 0: {torch.all(jsd_ab >= 0)}")
    print(f"Min JSD value: {torch.min(jsd_ab).item():.6f}")
    print()
    
    # Bounded: 0 <= JSD(P, Q) <= log(2)
    print("Boundedness Property:")
    log_2 = torch.log(torch.tensor(2.0))
    print(f"All JSD values <= log(2): {torch.all(jsd_ab <= log_2)}")
    print(f"Max JSD value: {torch.max(jsd_ab).item():.6f}")
    print(f"log(2): {log_2.item():.6f}")
    print()


if __name__ == "__main__":
    try:
        test_uncertainty_methods()
        test_edge_cases()
        demonstrate_mathematical_properties()
        print("All tests completed successfully!")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("This script requires PyTorch. Please install it with: pip install torch")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()