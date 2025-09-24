"""
Example usage of the Jensen-Shannon Divergence uncertainty calculation.

This example demonstrates how to replace the old uncertainty calculation 
with the new Jensen-Shannon Divergence method in practice.
"""

import torch
import torch.nn as nn
from uncertainty_utils import uncertainty_l, calculate_uncertainty


class DualModelUncertaintyEstimator:
    """
    Example class showing how to use the uncertainty calculation in practice.
    
    This could be used in scenarios like:
    - Semi-supervised learning with consistency regularization
    - Domain adaptation with source and target predictions
    - Ensemble methods with uncertainty quantification
    - Active learning for sample selection
    """
    
    def __init__(self, model_left, model_right):
        """
        Initialize with two models for uncertainty estimation.
        
        Args:
            model_left: First model (e.g., source domain model)
            model_right: Second model (e.g., target domain model)
        """
        self.model_left = model_left
        self.model_right = model_right
    
    def predict_with_uncertainty(self, x, method='jsd'):
        """
        Generate predictions with uncertainty estimates.
        
        Args:
            x: Input tensor
            method: Uncertainty calculation method ('jsd' or 'abs_diff')
            
        Returns:
            tuple: (predictions_left, predictions_right, uncertainty)
        """
        # Get predictions from both models
        with torch.no_grad():
            pred_sup_l = self.model_left(x)
            pred_sup_r = self.model_right(x)
        
        # OLD WAY (replaced):
        # uncertainty_l = abs(torch.max(torch.softmax(pred_sup_l, dim=1), dim=1)[0] - 
        #                    torch.max(torch.softmax(pred_sup_r, dim=1), dim=1)[0])
        
        # NEW WAY using Jensen-Shannon Divergence:
        uncertainty = calculate_uncertainty(pred_sup_l, pred_sup_r, method=method)
        
        return pred_sup_l, pred_sup_r, uncertainty
    
    def select_uncertain_samples(self, x, threshold=0.1, method='jsd'):
        """
        Select samples with uncertainty above threshold for active learning.
        
        Args:
            x: Input tensor
            threshold: Uncertainty threshold
            method: Uncertainty calculation method
            
        Returns:
            tuple: (uncertain_indices, uncertainties)
        """
        _, _, uncertainties = self.predict_with_uncertainty(x, method=method)
        uncertain_mask = uncertainties > threshold
        uncertain_indices = torch.where(uncertain_mask)[0]
        
        return uncertain_indices, uncertainties[uncertain_mask]


def example_migration():
    """
    Example showing how to migrate from old to new uncertainty calculation.
    """
    print("Migration Example: Old vs New Uncertainty Calculation")
    print("=" * 60)
    
    # Simulate some predictions from two models
    torch.manual_seed(42)
    batch_size, num_classes = 8, 10
    pred_sup_l = torch.randn(batch_size, num_classes)
    pred_sup_r = torch.randn(batch_size, num_classes)
    
    print(f"Processing {batch_size} samples with {num_classes} classes")
    print()
    
    # OLD CODE (what needs to be replaced):
    print("OLD CODE:")
    print("uncertainty_l = abs(torch.max(torch.softmax(pred_sup_l, dim=1), dim=1)[0] - ")
    print("                   torch.max(torch.softmax(pred_sup_r, dim=1), dim=1)[0])")
    
    old_uncertainty = abs(torch.max(torch.softmax(pred_sup_l, dim=1), dim=1)[0] - 
                         torch.max(torch.softmax(pred_sup_r, dim=1), dim=1)[0])
    print(f"Old result: {old_uncertainty}")
    print()
    
    # NEW CODE (Jensen-Shannon Divergence):
    print("NEW CODE:")
    print("from uncertainty_utils import uncertainty_l")
    print("uncertainty_l_new = uncertainty_l(pred_sup_l, pred_sup_r)")
    
    new_uncertainty = uncertainty_l(pred_sup_l, pred_sup_r)
    print(f"New result: {new_uncertainty}")
    print()
    
    # Comparison
    print("COMPARISON:")
    print(f"Old method mean: {old_uncertainty.mean():.4f}")
    print(f"New method mean: {new_uncertainty.mean():.4f}")
    print(f"Old method std:  {old_uncertainty.std():.4f}")
    print(f"New method std:  {new_uncertainty.std():.4f}")
    print()
    
    # Show benefits of new method
    print("BENEFITS OF JENSEN-SHANNON DIVERGENCE:")
    print("- Symmetric: JSD(P,Q) = JSD(Q,P)")
    print("- Bounded: 0 ≤ JSD ≤ log(2)")
    print("- Information-theoretic foundation")
    print("- Better captures distributional differences")
    print("- More robust to confident predictions")
    

def example_practical_usage():
    """
    Example of practical usage in a training loop context.
    """
    print("\nPractical Usage Example")
    print("=" * 30)
    
    # Create simple models for demonstration
    model_left = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 3))
    model_right = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 3))
    
    # Initialize uncertainty estimator
    uncertainty_estimator = DualModelUncertaintyEstimator(model_left, model_right)
    
    # Generate sample data
    torch.manual_seed(123)
    x = torch.randn(16, 10)
    
    # Get predictions with uncertainty
    pred_l, pred_r, uncertainty = uncertainty_estimator.predict_with_uncertainty(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Predictions left shape: {pred_l.shape}")
    print(f"Predictions right shape: {pred_r.shape}")
    print(f"Uncertainty shape: {uncertainty.shape}")
    print(f"Average uncertainty: {uncertainty.mean():.4f}")
    print()
    
    # Select uncertain samples (e.g., for active learning)
    uncertain_indices, uncertain_values = uncertainty_estimator.select_uncertain_samples(
        x, threshold=0.05
    )
    
    print(f"Samples with uncertainty > 0.05: {len(uncertain_indices)}")
    if len(uncertain_indices) > 0:
        print(f"Most uncertain sample index: {uncertain_indices[torch.argmax(uncertain_values)]}")
        print(f"Highest uncertainty value: {torch.max(uncertain_values):.4f}")


if __name__ == "__main__":
    try:
        example_migration()
        example_practical_usage()
        print("\nExample completed successfully!")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("This script requires PyTorch. Please install it with: pip install torch")
    except Exception as e:
        print(f"Error during example: {e}")
        import traceback
        traceback.print_exc()