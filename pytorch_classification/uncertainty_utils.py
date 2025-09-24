"""
Uncertainty calculation utilities for deep learning models.

This module provides functions for calculating inter-model uncertainty using
Jensen-Shannon Divergence, replacing the traditional absolute difference method.
"""

import torch
import torch.nn.functional as F


def old_uncertainty_calculation(pred_sup_l, pred_sup_r):
    """
    Original uncertainty calculation using absolute difference of maximum probabilities.
    
    Args:
        pred_sup_l (torch.Tensor): Logits from left/first model, shape (N, C)
        pred_sup_r (torch.Tensor): Logits from right/second model, shape (N, C)
    
    Returns:
        torch.Tensor: Uncertainty values, shape (N,)
    
    Formula:
        uncertainty_l = abs(torch.max(torch.softmax(pred_sup_l, dim=1), dim=1)[0] - 
                           torch.max(torch.softmax(pred_sup_r, dim=1), dim=1)[0])
    """
    softmax_l = torch.softmax(pred_sup_l, dim=1)
    softmax_r = torch.softmax(pred_sup_r, dim=1)
    
    max_prob_l = torch.max(softmax_l, dim=1)[0]
    max_prob_r = torch.max(softmax_r, dim=1)[0]
    
    uncertainty_l = torch.abs(max_prob_l - max_prob_r)
    return uncertainty_l


def jensen_shannon_divergence_uncertainty(pred_sup_l, pred_sup_r):
    """
    Calculate inter-model uncertainty using Jensen-Shannon Divergence.
    
    Jensen-Shannon Divergence is a method for measuring the similarity between 
    two probability distributions. It's symmetric and always finite, making it 
    ideal for uncertainty quantification between model predictions.
    
    Args:
        pred_sup_l (torch.Tensor): Logits from left/first model, shape (N, C)
        pred_sup_r (torch.Tensor): Logits from right/second model, shape (N, C)
    
    Returns:
        torch.Tensor: Jensen-Shannon divergence uncertainty values, shape (N,)
    
    Mathematical formulation:
        M = 1/2 * (ps + pt)
        JSD(ps || pt) = 1/2 * KL(ps || M) + 1/2 * KL(pt || M)
        
    where:
        - ps = softmax(pred_sup_l)
        - pt = softmax(pred_sup_r) 
        - M is the mixture distribution
        - KL is the Kullback-Leibler divergence
    """
    # Convert logits to probability distributions
    ps = torch.softmax(pred_sup_l, dim=1)
    pt = torch.softmax(pred_sup_r, dim=1)
    
    # Calculate the mixture distribution M = 1/2 * (ps + pt)
    M = 0.5 * (ps + pt)
    
    # Add small epsilon to prevent log(0) issues
    eps = 1e-10
    ps_safe = ps + eps
    pt_safe = pt + eps
    M_safe = M + eps
    
    # Calculate KL divergences: KL(P || Q) = sum(P * log(P / Q))
    kl_ps_M = torch.sum(ps * torch.log(ps_safe / M_safe), dim=1)
    kl_pt_M = torch.sum(pt * torch.log(pt_safe / M_safe), dim=1)
    
    # Jensen-Shannon Divergence: JSD = 1/2 * KL(ps || M) + 1/2 * KL(pt || M)
    jsd = 0.5 * kl_ps_M + 0.5 * kl_pt_M
    
    return jsd


def calculate_uncertainty(pred_sup_l, pred_sup_r, method='jsd'):
    """
    Calculate inter-model uncertainty using the specified method.
    
    Args:
        pred_sup_l (torch.Tensor): Logits from left/first model, shape (N, C)
        pred_sup_r (torch.Tensor): Logits from right/second model, shape (N, C)
        method (str): Uncertainty calculation method. Options:
                     - 'jsd': Jensen-Shannon Divergence (recommended)
                     - 'abs_diff': Absolute difference of max probabilities (legacy)
    
    Returns:
        torch.Tensor: Uncertainty values, shape (N,)
    
    Example:
        >>> pred_l = torch.randn(4, 10)  # 4 samples, 10 classes
        >>> pred_r = torch.randn(4, 10)
        >>> uncertainty = calculate_uncertainty(pred_l, pred_r, method='jsd')
        >>> print(uncertainty.shape)  # torch.Size([4])
    """
    if method == 'jsd':
        return jensen_shannon_divergence_uncertainty(pred_sup_l, pred_sup_r)
    elif method == 'abs_diff':
        return old_uncertainty_calculation(pred_sup_l, pred_sup_r)
    else:
        raise ValueError(f"Unknown uncertainty method: {method}. "
                        f"Available methods: 'jsd', 'abs_diff'")


# For backward compatibility, create an alias that replaces the old calculation
def uncertainty_l(pred_sup_l, pred_sup_r):
    """
    Updated uncertainty calculation using Jensen-Shannon Divergence.
    
    This function replaces the old uncertainty calculation:
    OLD: abs(torch.max(torch.softmax(pred_sup_l, dim=1), dim=1)[0] - 
             torch.max(torch.softmax(pred_sup_r, dim=1), dim=1)[0])
    NEW: Jensen-Shannon Divergence between softmax distributions
    
    Args:
        pred_sup_l (torch.Tensor): Logits from left/first model
        pred_sup_r (torch.Tensor): Logits from right/second model
    
    Returns:
        torch.Tensor: Inter-model uncertainty values
    """
    return jensen_shannon_divergence_uncertainty(pred_sup_l, pred_sup_r)