# Semantic Segmentation Inference Tools - Usage Guide

This guide provides practical examples for using the refactored semantic segmentation inference tools.

## Quick Start

### 1. Setup
```bash
# Install dependencies
pip install torch torchvision numpy Pillow tqdm

# Navigate to FCN directory
cd pytorch_segmentation/fcn

# Check prerequisites
python example_usage.py --check
```

### 2. Generate Pseudo Labels
```bash
# Basic usage - generate pseudo labels for unlabeled images
python generate_pseudo_labels.py \
    --model-path ./save_weights/model_29.pth \
    --input-dir /path/to/unlabeled/images \
    --output-dir /path/to/pseudo/labels
```

### 3. Compute Test IoU
```bash
# Evaluate model on test dataset
python compute_test_iou.py \
    --model-path ./save_weights/model_29.pth \
    --image-dir /path/to/test/images \
    --mask-dir /path/to/test/masks \
    --verbose
```

## Advanced Examples

### High-Quality Pseudo Labels
For applications requiring high-quality pseudo labels, use a higher confidence threshold:

```bash
python generate_pseudo_labels.py \
    --model-path ./save_weights/best_model.pth \
    --input-dir ./unlabeled_data \
    --output-dir ./high_quality_pseudo_labels \
    --confidence-threshold 0.95 \
    --palette-path ./palette.json
```

### Batch Evaluation with Custom Settings
For large-scale evaluation with optimized settings:

```bash
python compute_test_iou.py \
    --model-path ./save_weights/best_model.pth \
    --image-dir ./large_test_set/images \
    --mask-dir ./large_test_set/masks \
    --batch-size 8 \
    --num-workers 16 \
    --output-file ./results/detailed_evaluation.json \
    --class-names ./pascal_voc_classes.json \
    --verbose
```

### PASCAL VOC Evaluation
Standard evaluation on PASCAL VOC validation set:

```bash
python compute_test_iou.py \
    --model-path ./save_weights/final_model.pth \
    --image-dir /data/VOCdevkit/VOC2012/JPEGImages \
    --mask-dir /data/VOCdevkit/VOC2012/SegmentationClass \
    --class-names ./pascal_voc_classes.json \
    --output-file ./voc_validation_results.json \
    --verbose
```

## Performance Tips

### For Pseudo Label Generation
- Use `--confidence-threshold 0.9` or higher for critical applications
- Add `--no-colored` to save storage space if visualizations aren't needed
- Use GPU (`--device cuda`) for faster processing

### For IoU Computation
- Increase `--batch-size` and `--num-workers` for faster evaluation
- Use `--verbose` only when needed to reduce output
- Save results to JSON for programmatic analysis

## Output Interpretation

### Pseudo Label Quality Metrics
The script reports high-confidence pseudo label ratios. Aim for:
- **>70%** for training data augmentation
- **>90%** for critical applications
- **>95%** for safety-critical domains

### IoU Results
Typical IoU ranges for PASCAL VOC:
- **Mean IoU >70%**: Excellent performance
- **Mean IoU 60-70%**: Good performance  
- **Mean IoU 50-60%**: Acceptable performance
- **Mean IoU <50%**: Needs improvement

## Troubleshooting

### Common Issues and Solutions

1. **Import Errors**
   ```bash
   # Ensure you're in the correct directory
   cd pytorch_segmentation/fcn
   # Check Python path
   python -c "from src import fcn_resnet50; print('OK')"
   ```

2. **Model Loading Issues**
   ```bash
   # Verify model file exists and format
   python -c "import torch; print(torch.load('model.pth').keys())"
   ```

3. **Dataset Format Issues**
   ```bash
   # Check image-mask correspondence
   python example_usage.py --setup  # Creates sample structure
   ```

4. **Memory Issues**
   ```bash
   # Reduce batch size
   --batch-size 1
   # Use CPU if GPU memory insufficient
   --device cpu
   ```

5. **Performance Issues**
   ```bash
   # Increase workers for faster data loading
   --num-workers 8
   # Use smaller image sizes if needed
   # (modify transforms in the script)
   ```

## Integration Examples

### Semi-Supervised Learning Pipeline
```bash
# Step 1: Generate pseudo labels
python generate_pseudo_labels.py \
    --model-path ./initial_model.pth \
    --input-dir ./unlabeled_pool \
    --output-dir ./pseudo_labels_round1 \
    --confidence-threshold 0.9

# Step 2: Evaluate current performance
python compute_test_iou.py \
    --model-path ./initial_model.pth \
    --image-dir ./validation_images \
    --mask-dir ./validation_masks \
    --output-file ./baseline_performance.json

# Step 3: Train with pseudo labels (using existing training script)
# Step 4: Repeat with improved model
```

### Model Comparison
```bash
# Compare multiple model checkpoints
for epoch in 20 25 29; do
    echo "Evaluating epoch $epoch..."
    python compute_test_iou.py \
        --model-path ./save_weights/model_${epoch}.pth \
        --image-dir ./test_images \
        --mask-dir ./test_masks \
        --output-file ./results/epoch_${epoch}_results.json
done
```

### Quality Assessment
```bash
# Generate pseudo labels with different thresholds
for threshold in 0.7 0.8 0.9 0.95; do
    python generate_pseudo_labels.py \
        --model-path ./model.pth \
        --input-dir ./unlabeled_sample \
        --output-dir ./pseudo_labels_${threshold} \
        --confidence-threshold ${threshold}
done
```

## API Usage (Programmatic)

For integration into larger systems, the tools can be used programmatically:

```python
from generate_pseudo_labels import PseudoLabelGenerator
from compute_test_iou import IoUEvaluator

# Initialize tools
generator = PseudoLabelGenerator(
    model_path="./model.pth",
    num_classes=20,
    device="cuda"
)

evaluator = IoUEvaluator(
    model_path="./model.pth", 
    num_classes=20,
    device="cuda"
)

# Process single image
pseudo_label, confidence, time = generator.generate_pseudo_label("image.jpg")

# Batch processing
generator.generate_batch_pseudo_labels("input_dir", "output_dir")

# Evaluation
results = evaluator.evaluate_dataset(test_loader)
print(f"Mean IoU: {results['mean_iou']:.2f}%")
```

## Best Practices

1. **Model Selection**: Use your best-performing checkpoint for inference
2. **Data Quality**: Ensure test images match training distribution
3. **Confidence Thresholding**: Start with 0.9 and adjust based on requirements
4. **Validation**: Always validate pseudo labels on a small subset first
5. **Storage**: Plan for output storage requirements (pseudo labels can be large)
6. **Monitoring**: Track inference times and confidence distributions
7. **Documentation**: Record hyperparameters and results for reproducibility

## Performance Benchmarks

Typical performance on modern hardware:

| Operation | GPU (RTX 3080) | CPU (16-core) | Memory |
|-----------|----------------|---------------|---------|
| Pseudo Label (512x512) | ~50ms | ~200ms | ~2GB |
| IoU Computation (batch=4) | ~100ms | ~400ms | ~4GB |
| Large dataset (1000 images) | ~2min | ~8min | ~8GB |

Results may vary based on model complexity and hardware configuration.