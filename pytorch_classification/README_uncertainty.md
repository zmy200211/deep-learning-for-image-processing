# Inter-Model Uncertainty Calculation with Jensen-Shannon Divergence

This module provides an updated implementation for calculating inter-model uncertainty using Jensen-Shannon Divergence (JSD), replacing the traditional absolute difference method.

## Overview

The original uncertainty calculation used the absolute difference between maximum probabilities:

```python
# OLD METHOD
uncertainty_l = abs(torch.max(torch.softmax(pred_sup_l, dim=1), dim=1)[0] - 
                   torch.max(torch.softmax(pred_sup_r, dim=1), dim=1)[0])
```

The new method uses Jensen-Shannon Divergence, which provides a more principled approach:

```python
# NEW METHOD  
from uncertainty_utils import uncertainty_l
uncertainty_l_new = uncertainty_l(pred_sup_l, pred_sup_r)
```

## Mathematical Foundation

### Jensen-Shannon Divergence

The Jensen-Shannon Divergence is calculated as:

```
M = 1/2 * (ps + pt)
JSD(ps || pt) = 1/2 * KL(ps || M) + 1/2 * KL(pt || M)
```

Where:
- `ps = softmax(pred_sup_l)`: Probability distribution from first model
- `pt = softmax(pred_sup_r)`: Probability distribution from second model  
- `M`: Mixture distribution (average of ps and pt)
- `KL(P || Q)`: Kullback-Leibler divergence from P to Q

### Key Properties

1. **Symmetric**: `JSD(P, Q) = JSD(Q, P)`
2. **Non-negative**: `JSD(P, Q) ≥ 0`
3. **Bounded**: `0 ≤ JSD(P, Q) ≤ log(2)`
4. **Information-theoretic**: Based on information theory principles

## Usage

### Basic Usage

```python
import torch
from uncertainty_utils import calculate_uncertainty

# Your model predictions (logits)
pred_sup_l = torch.randn(batch_size, num_classes)  
pred_sup_r = torch.randn(batch_size, num_classes)

# Calculate uncertainty using Jensen-Shannon Divergence
uncertainty = calculate_uncertainty(pred_sup_l, pred_sup_r, method='jsd')

# For backward compatibility (replaces old uncertainty_l calculation)
from uncertainty_utils import uncertainty_l
uncertainty = uncertainty_l(pred_sup_l, pred_sup_r)
```

### Migration Guide

**Old Code:**
```python
uncertainty_l = abs(torch.max(torch.softmax(pred_sup_l, dim=1), dim=1)[0] - 
                   torch.max(torch.softmax(pred_sup_r, dim=1), dim=1)[0])
```

**New Code:**
```python
from uncertainty_utils import uncertainty_l
uncertainty_l = uncertainty_l(pred_sup_l, pred_sup_r)
```

### Advanced Usage

```python
from uncertainty_utils import calculate_uncertainty

# Compare different methods
jsd_uncertainty = calculate_uncertainty(pred_sup_l, pred_sup_r, method='jsd')
old_uncertainty = calculate_uncertainty(pred_sup_l, pred_sup_r, method='abs_diff')

# Use in uncertainty-based sample selection
high_uncertainty_mask = jsd_uncertainty > threshold
uncertain_samples = inputs[high_uncertainty_mask]
```

## Files

- `uncertainty_utils.py`: Main implementation with all uncertainty calculation functions
- `test_uncertainty.py`: Comprehensive test suite demonstrating differences between methods
- `example_usage.py`: Practical examples showing integration into existing code
- `README_uncertainty.md`: This documentation file

## Testing

Run the test suite to validate the implementation:

```bash
python test_uncertainty.py
```

Run the example to see practical usage:

```bash  
python example_usage.py
```

## Benefits of Jensen-Shannon Divergence

1. **Mathematical Rigor**: Based on information theory with well-understood properties
2. **Symmetry**: Order of model predictions doesn't matter
3. **Bounded Output**: Predictable range [0, log(2)] for better thresholding
4. **Distributional Awareness**: Considers entire probability distributions, not just maximum values
5. **Robustness**: Less sensitive to very confident but incorrect predictions

## Applications

This uncertainty calculation is particularly useful for:

- **Semi-supervised Learning**: Consistency regularization between models
- **Domain Adaptation**: Measuring prediction agreement across domains  
- **Active Learning**: Selecting informative samples for annotation
- **Ensemble Methods**: Quantifying prediction diversity
- **Model Calibration**: Understanding prediction reliability

## Requirements

- PyTorch
- Python 3.6+

## References

- Lin, J. (1991). Divergence measures based on the Shannon entropy. IEEE Transactions on Information theory, 37(1), 145-151.
- Endres, D. M., & Schindelin, J. E. (2003). A new metric for probability distributions. IEEE Transactions on Information theory, 49(7), 1858-1860.