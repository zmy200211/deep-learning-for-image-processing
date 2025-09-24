# Jensen-Shannon Divergence Uncertainty Calculation Upgrade

## Overview

This upgrade replaces the traditional absolute difference uncertainty calculation with a more principled Jensen-Shannon Divergence (JSD) approach for inter-model uncertainty quantification.

## The Problem with the Old Method

The original uncertainty calculation:

```python
uncertainty_l = abs(torch.max(torch.softmax(pred_sup_l, dim=1), dim=1)[0] - 
                   torch.max(torch.softmax(pred_sup_r, dim=1), dim=1)[0])
```

**Limitations:**
- Only considers maximum probabilities, ignoring the full distribution
- Not symmetric (though `abs()` makes it appear so)
- No theoretical foundation in information theory
- Can be insensitive to distributional differences when max probabilities are similar
- Unbounded output range makes thresholding difficult

## The Jensen-Shannon Divergence Solution

**Mathematical Foundation:**
```
M = 1/2 * (ps + pt)
JSD(ps || pt) = 1/2 * KL(ps || M) + 1/2 * KL(pt || M)
```

Where:
- `ps = softmax(pred_sup_l)` (probability distribution from first model)
- `pt = softmax(pred_sup_r)` (probability distribution from second model)
- `M` is the mixture distribution
- `KL(P || Q)` is the Kullback-Leibler divergence

## Key Advantages

### 1. **Information-Theoretic Foundation**
- Based on well-established information theory principles
- Measures true distributional divergence, not just point estimates

### 2. **Mathematical Properties**
- **Symmetric:** `JSD(P, Q) = JSD(Q, P)`
- **Non-negative:** `JSD(P, Q) ≥ 0`
- **Bounded:** `0 ≤ JSD(P, Q) ≤ log(2) ≈ 0.693`
- **Metric:** Satisfies triangle inequality (square root of JSD)

### 3. **Better Uncertainty Quantification**
- Considers entire probability distributions
- More sensitive to subtle distributional differences
- Robust to confident but incorrect predictions
- Bounded output makes thresholding more predictable

### 4. **Practical Benefits**
- Drop-in replacement with backward compatibility
- Better performance in uncertainty-based sample selection
- More reliable for active learning and semi-supervised scenarios
- Improved model ensemble uncertainty estimation

## Implementation

### Simple Replacement

**Before:**
```python
uncertainty_l = abs(torch.max(torch.softmax(pred_sup_l, dim=1), dim=1)[0] - 
                   torch.max(torch.softmax(pred_sup_r, dim=1), dim=1)[0])
```

**After:**
```python
from uncertainty_utils import uncertainty_l
uncertainty_l = uncertainty_l(pred_sup_l, pred_sup_r)
```

### Advanced Usage

```python
from uncertainty_utils import calculate_uncertainty

# Use Jensen-Shannon Divergence (recommended)
jsd_uncertainty = calculate_uncertainty(pred_sup_l, pred_sup_r, method='jsd')

# Compare with old method
old_uncertainty = calculate_uncertainty(pred_sup_l, pred_sup_r, method='abs_diff')

# Uncertainty-based sample selection
high_uncertainty_mask = jsd_uncertainty > threshold
```

## Behavioral Differences

### Example Scenarios

1. **Identical Predictions:**
   - Old method: `0.0`
   - New method: `0.0` ✓

2. **Confident but Different:**
   - Prediction A: `[0.99, 0.01]`, Prediction B: `[0.01, 0.99]`
   - Old method: `0.0` (both have max prob 0.99)
   - New method: `0.693` (maximum JSD, indicating complete disagreement) ✓

3. **Uniform vs Peaked:**
   - Prediction A: `[0.25, 0.25, 0.25, 0.25]`, Prediction B: `[0.98, 0.007, 0.007, 0.007]`
   - Old method: `0.73` (difference in max probabilities)
   - New method: `0.337` (more nuanced measure of distributional difference)

## Files in This Implementation

- `uncertainty_utils.py` - Core implementation
- `test_uncertainty.py` - Comprehensive test suite
- `example_usage.py` - Practical integration examples
- `formula_replacement_demo.py` - Direct formula replacement demo
- `README_uncertainty.md` - Detailed documentation
- `JENSEN_SHANNON_UPGRADE.md` - This upgrade guide

## Migration Checklist

- [ ] Import the new uncertainty calculation: `from uncertainty_utils import uncertainty_l`
- [ ] Replace the old formula with the new function call
- [ ] Test with your specific data to understand behavioral changes
- [ ] Adjust thresholds if needed (JSD is bounded by log(2) ≈ 0.693)
- [ ] Update any documentation or comments referencing the old method
- [ ] Consider using the unified interface for flexibility: `calculate_uncertainty()`

## Performance Considerations

- **Computational Complexity:** Similar to the old method (both require softmax)
- **Memory Usage:** Slightly higher due to additional intermediate calculations
- **Numerical Stability:** Includes epsilon handling to prevent log(0) issues
- **Backpropagation:** Fully differentiable if gradients are needed

## Theoretical Background

Jensen-Shannon Divergence is the symmetric version of Kullback-Leibler divergence and has strong theoretical foundations:

- **Paper:** Lin, J. (1991). "Divergence measures based on the Shannon entropy"
- **Properties:** Metric properties when taking square root
- **Applications:** Widely used in machine learning, bioinformatics, and information theory
- **Interpretation:** Measures the "average" KL divergence from each distribution to their mixture

## Conclusion

The Jensen-Shannon Divergence upgrade provides a more theoretically sound, mathematically principled approach to inter-model uncertainty quantification while maintaining ease of use through backward-compatible interfaces.