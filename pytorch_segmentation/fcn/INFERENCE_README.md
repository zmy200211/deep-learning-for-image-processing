# Semantic Segmentation Inference Tools

This directory contains refactored scripts for semantic segmentation inference tasks, focusing on two main functionalities:

1. **Pseudo Label Generation** - Generate pseudo labels for unlabeled data
2. **Mean IoU Computation** - Evaluate model performance on test datasets

## Features

- **Clean, modular design** - Easy to understand and extend
- **Batch processing** - Efficient handling of large datasets  
- **Confidence-based filtering** - Optional quality control for pseudo labels
- **Detailed reporting** - Comprehensive performance metrics
- **Flexible input formats** - Support for various image formats and directory structures

## Scripts Overview

### 1. Pseudo Label Generation (`generate_pseudo_labels.py`)

Generates pseudo labels for unlabeled data using a trained FCN model.

**Key Features:**
- Batch processing of image directories
- Configurable confidence thresholding
- Colored and grayscale output options
- Progress tracking and timing statistics

**Usage:**
```bash
python generate_pseudo_labels.py \
    --model-path ./save_weights/model_29.pth \
    --input-dir /path/to/unlabeled/images \
    --output-dir /path/to/pseudo/labels \
    --confidence-threshold 0.9 \
    --num-classes 20
```

**Arguments:**
- `--model-path`: Path to trained model weights (required)
- `--input-dir`: Directory containing unlabeled images (required)
- `--output-dir`: Directory to save pseudo labels (required)
- `--num-classes`: Number of segmentation classes excluding background (default: 20)
- `--confidence-threshold`: Minimum confidence for pseudo label generation (default: 0.9)
- `--no-confidence-filtering`: Disable confidence-based filtering
- `--no-colored`: Don't save colored pseudo label images
- `--palette-path`: Path to color palette JSON file (default: ./palette.json)
- `--device`: Device for inference - cuda/cpu (default: cuda)

### 2. Mean IoU Computation (`compute_test_iou.py`)

Computes mean IoU and detailed performance metrics for test datasets.

**Key Features:**
- Per-class and mean IoU computation
- Support for custom dataset structures
- Detailed performance reporting
- JSON output for further analysis

**Usage:**
```bash
python compute_test_iou.py \
    --model-path ./save_weights/model_29.pth \
    --image-dir /path/to/test/images \
    --mask-dir /path/to/test/masks \
    --output-file ./test_results.json \
    --verbose
```

**Arguments:**
- `--model-path`: Path to trained model weights (required)
- `--image-dir`: Directory containing test images (required)
- `--mask-dir`: Directory containing ground truth masks (required)
- `--num-classes`: Number of segmentation classes excluding background (default: 20)
- `--batch-size`: Batch size for evaluation (default: 1)
- `--num-workers`: Number of worker processes for data loading (default: 4)
- `--output-file`: Path to save evaluation results (default: ./test_iou_results.json)
- `--class-names`: Path to class names JSON file (default: ./pascal_voc_classes.json)
- `--verbose`: Print detailed per-class results
- `--device`: Device for inference - cuda/cpu (default: cuda)

## Output Formats

### Pseudo Label Generation Output

The script generates:
- **Grayscale labels**: PNG files with pixel values corresponding to class indices
- **Colored labels**: PNG files with color-coded segmentation masks (optional)
- **Console output**: Progress tracking, timing statistics, and quality metrics

Example output structure:
```
output_dir/
├── image1.png              # Grayscale pseudo label
├── image1_colored.png      # Colored pseudo label  
├── image2.png
├── image2_colored.png
└── ...
```

### IoU Computation Output

The script provides:
- **Console output**: Mean IoU, global accuracy, and per-class metrics (if --verbose)
- **JSON file**: Detailed results for further analysis

Example JSON output:
```json
{
  "mean_iou": 75.32,
  "global_accuracy": 89.45,
  "per_class_iou": {
    "background": 95.2,
    "aeroplane": 78.5,
    "bicycle": 68.3,
    ...
  },
  "per_class_accuracy": {
    "background": 97.8,
    "aeroplane": 85.2,
    "bicycle": 76.1,
    ...
  },
  "num_samples": 1449,
  "avg_inference_time": 0.045,
  "total_time": 65.2
}
```

## Dataset Requirements

### For Pseudo Label Generation
- Input directory with unlabeled images
- Supported formats: .jpg, .jpeg, .png, .bmp
- Images can be in subdirectories (recursive search)

### For IoU Computation
- Two directories: one for images, one for corresponding ground truth masks
- Image-mask pairs must have matching filenames (excluding extension)
- Example:
  ```
  images/
  ├── img001.jpg
  ├── img002.jpg
  └── ...
  
  masks/
  ├── img001.png
  ├── img002.png  
  └── ...
  ```

## Model Requirements

Both scripts work with FCN ResNet50 models trained using the existing training pipeline. The model should:
- Be saved with the complete state dict including 'model' key
- Have been trained with the same number of classes
- Use standard preprocessing (ImageNet normalization)

## Performance Considerations

### Pseudo Label Generation
- **Memory usage**: Depends on image size and batch processing
- **Speed**: ~20-50ms per image on GPU (depending on image size)
- **Quality control**: Use confidence thresholding to filter low-quality pseudo labels

### IoU Computation  
- **Accuracy**: Uses the same evaluation metrics as training
- **Speed**: Similar to pseudo label generation
- **Memory**: Minimal additional memory for confusion matrix

## Examples

### Example 1: Generate high-confidence pseudo labels
```bash
python generate_pseudo_labels.py \
    --model-path ./save_weights/best_model.pth \
    --input-dir ./unlabeled_data \
    --output-dir ./pseudo_labels \
    --confidence-threshold 0.95 \
    --num-classes 20
```

### Example 2: Evaluate on PASCAL VOC validation set
```bash
python compute_test_iou.py \
    --model-path ./save_weights/best_model.pth \
    --image-dir /data/VOCdevkit/VOC2012/JPEGImages \
    --mask-dir /data/VOCdevkit/VOC2012/SegmentationClass \
    --class-names ./pascal_voc_classes.json \
    --output-file ./voc_val_results.json \
    --verbose
```

### Example 3: Quick IoU check without detailed output
```bash
python compute_test_iou.py \
    --model-path ./save_weights/model_latest.pth \
    --image-dir ./test_images \
    --mask-dir ./test_masks \
    --batch-size 4
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're running from the FCN directory and have installed dependencies
2. **Model loading errors**: Check that model path exists and matches expected format
3. **Dataset errors**: Verify image-mask pairs exist and have matching filenames
4. **Memory errors**: Reduce batch size or image resolution
5. **CUDA errors**: Use `--device cpu` if GPU is unavailable

### Dependencies

Required packages:
- torch >= 1.13.1
- torchvision >= 0.11.1  
- numpy
- Pillow
- tqdm

Install with: `pip install torch torchvision numpy Pillow tqdm`

## Implementation Notes

### Removed Training Components
The refactored scripts have removed:
- Training loops and optimization logic
- Learning rate scheduling
- Data augmentation for training
- Model saving during training
- Distributed training utilities

### Retained Essential Components  
- Data preprocessing and transforms
- Model architecture and loading
- Evaluation metrics (IoU computation)
- Confusion matrix implementation
- Device handling and timing utilities

### Optimizations for Inference
- Disabled auxiliary classifier for faster inference
- Model set to eval mode with gradient computation disabled
- Efficient batch processing with progress tracking
- Memory-conscious data loading