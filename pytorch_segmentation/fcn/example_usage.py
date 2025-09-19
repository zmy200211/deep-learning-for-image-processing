#!/usr/bin/env python3
"""
Example usage script for the refactored semantic segmentation inference tools.

This script demonstrates how to use the pseudo label generation and IoU computation
tools with sample data and provides templates for common use cases.
"""

import os
import sys
import json
import argparse
from pathlib import Path


def create_sample_data_structure():
    """Create sample directory structure for testing."""
    base_dir = Path("./sample_data")
    
    # Create directories
    (base_dir / "unlabeled_images").mkdir(parents=True, exist_ok=True)
    (base_dir / "test_images").mkdir(parents=True, exist_ok=True)
    (base_dir / "test_masks").mkdir(parents=True, exist_ok=True)
    (base_dir / "pseudo_labels").mkdir(parents=True, exist_ok=True)
    (base_dir / "results").mkdir(parents=True, exist_ok=True)
    
    print(f"Created sample directory structure at: {base_dir.absolute()}")
    print("\nDirectory structure:")
    print("sample_data/")
    print("├── unlabeled_images/    # Put unlabeled images here")
    print("├── test_images/         # Put test images here")  
    print("├── test_masks/          # Put corresponding masks here")
    print("├── pseudo_labels/       # Generated pseudo labels will be saved here")
    print("└── results/            # Evaluation results will be saved here")
    
    return base_dir


def generate_sample_config():
    """Generate sample configuration files."""
    
    # Sample class names for PASCAL VOC
    voc_classes = {
        "0": "background",
        "1": "aeroplane", 
        "2": "bicycle",
        "3": "bird",
        "4": "boat",
        "5": "bottle",
        "6": "bus",
        "7": "car",
        "8": "cat",
        "9": "chair",
        "10": "cow",
        "11": "diningtable",
        "12": "dog",
        "13": "horse",
        "14": "motorbike",
        "15": "person",
        "16": "pottedplant",
        "17": "sheep",
        "18": "sofa",
        "19": "train",
        "20": "tvmonitor"
    }
    
    with open("sample_pascal_voc_classes.json", "w") as f:
        json.dump(voc_classes, f, indent=2)
    
    print("\nCreated sample_pascal_voc_classes.json")
    
    # Sample color palette (simplified)
    sample_palette = {
        "background": [0, 0, 0],
        "aeroplane": [128, 0, 0],
        "bicycle": [0, 128, 0],
        "bird": [128, 128, 0],
        "boat": [0, 0, 128],
        "bottle": [128, 0, 128],
        "bus": [0, 128, 128],
        "car": [128, 128, 128],
        "cat": [64, 0, 0],
        "chair": [192, 0, 0],
        "cow": [64, 128, 0],
        "diningtable": [192, 128, 0],
        "dog": [64, 0, 128],
        "horse": [192, 0, 128],
        "motorbike": [64, 128, 128],
        "person": [192, 128, 128],
        "pottedplant": [0, 64, 0],
        "sheep": [128, 64, 0],
        "sofa": [0, 192, 0],
        "train": [128, 192, 0],
        "tvmonitor": [0, 64, 128]
    }
    
    with open("sample_palette.json", "w") as f:
        json.dump(sample_palette, f, indent=2)
    
    print("Created sample_palette.json")


def print_usage_examples():
    """Print example command usage."""
    
    print("\n" + "="*80)
    print("USAGE EXAMPLES")
    print("="*80)
    
    print("\n1. PSEUDO LABEL GENERATION")
    print("-" * 30)
    print("# Basic usage:")
    print("python generate_pseudo_labels.py \\")
    print("    --model-path ./save_weights/model_29.pth \\")
    print("    --input-dir ./sample_data/unlabeled_images \\")
    print("    --output-dir ./sample_data/pseudo_labels")
    
    print("\n# With high confidence threshold:")
    print("python generate_pseudo_labels.py \\")
    print("    --model-path ./save_weights/model_29.pth \\")
    print("    --input-dir ./sample_data/unlabeled_images \\")
    print("    --output-dir ./sample_data/pseudo_labels \\")
    print("    --confidence-threshold 0.95 \\")
    print("    --palette-path ./sample_palette.json")
    
    print("\n# Without confidence filtering:")
    print("python generate_pseudo_labels.py \\")
    print("    --model-path ./save_weights/model_29.pth \\")
    print("    --input-dir ./sample_data/unlabeled_images \\")
    print("    --output-dir ./sample_data/pseudo_labels \\")
    print("    --no-confidence-filtering")
    
    print("\n2. IoU COMPUTATION")
    print("-" * 20)
    print("# Basic evaluation:")
    print("python compute_test_iou.py \\")
    print("    --model-path ./save_weights/model_29.pth \\")
    print("    --image-dir ./sample_data/test_images \\")
    print("    --mask-dir ./sample_data/test_masks")
    
    print("\n# Detailed evaluation with custom output:")
    print("python compute_test_iou.py \\")
    print("    --model-path ./save_weights/model_29.pth \\")
    print("    --image-dir ./sample_data/test_images \\")
    print("    --mask-dir ./sample_data/test_masks \\")
    print("    --class-names ./sample_pascal_voc_classes.json \\")
    print("    --output-file ./sample_data/results/test_results.json \\")
    print("    --verbose")
    
    print("\n# Batch evaluation:")
    print("python compute_test_iou.py \\")
    print("    --model-path ./save_weights/model_29.pth \\")
    print("    --image-dir ./sample_data/test_images \\")
    print("    --mask-dir ./sample_data/test_masks \\")
    print("    --batch-size 4 \\")
    print("    --num-workers 8")
    
    print("\n3. REAL DATASET EXAMPLES")
    print("-" * 28)
    print("# PASCAL VOC validation set:")
    print("python compute_test_iou.py \\")
    print("    --model-path ./save_weights/best_model.pth \\")
    print("    --image-dir /data/VOCdevkit/VOC2012/JPEGImages \\")
    print("    --mask-dir /data/VOCdevkit/VOC2012/SegmentationClass \\")
    print("    --class-names ./pascal_voc_classes.json \\")
    print("    --output-file ./voc_validation_results.json \\")
    print("    --verbose")
    
    print("\n# Generate pseudo labels for large dataset:")
    print("python generate_pseudo_labels.py \\")
    print("    --model-path ./save_weights/best_model.pth \\")
    print("    --input-dir /data/unlabeled_dataset \\")
    print("    --output-dir /data/pseudo_labels \\")
    print("    --confidence-threshold 0.9")


def check_prerequisites():
    """Check if required files and dependencies exist."""
    
    print("\n" + "="*80)
    print("PREREQUISITES CHECK")
    print("="*80)
    
    # Check if scripts exist
    scripts = ["generate_pseudo_labels.py", "compute_test_iou.py"]
    for script in scripts:
        if os.path.exists(script):
            print(f"✓ {script} found")
        else:
            print(f"✗ {script} NOT FOUND")
    
    # Check if core dependencies can be imported
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} available")
    except ImportError:
        print("✗ PyTorch not available")
    
    try:
        import torchvision
        print(f"✓ TorchVision {torchvision.__version__} available")
    except ImportError:
        print("✗ TorchVision not available")
    
    try:
        from src import fcn_resnet50
        print("✓ FCN model can be imported")
    except ImportError:
        print("✗ FCN model cannot be imported")
    
    # Check for model weights
    weight_dirs = ["./save_weights", "./weights"]
    found_weights = False
    for weight_dir in weight_dirs:
        if os.path.exists(weight_dir):
            weights = [f for f in os.listdir(weight_dir) if f.endswith('.pth')]
            if weights:
                print(f"✓ Model weights found in {weight_dir}: {weights[:3]}...")
                found_weights = True
                break
    
    if not found_weights:
        print("✗ No model weights found")
        print("  Please ensure you have trained model weights in ./save_weights/")
    
    # Check for config files
    config_files = ["palette.json", "pascal_voc_classes.json"]
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✓ {config_file} found")
        else:
            print(f"! {config_file} not found (will use defaults or sample)")


def main():
    parser = argparse.ArgumentParser(description="Example usage for segmentation inference tools")
    parser.add_argument("--setup", action="store_true", 
                       help="Create sample data structure and config files")
    parser.add_argument("--check", action="store_true",
                       help="Check prerequisites and dependencies")
    parser.add_argument("--examples", action="store_true",
                       help="Show usage examples")
    
    args = parser.parse_args()
    
    if args.setup:
        print("Setting up sample data structure and configuration files...")
        create_sample_data_structure()
        generate_sample_config()
        print("\nSetup complete! You can now:")
        print("1. Add your trained model to ./save_weights/")
        print("2. Add test images to ./sample_data/test_images/")
        print("3. Add corresponding masks to ./sample_data/test_masks/") 
        print("4. Add unlabeled images to ./sample_data/unlabeled_images/")
        print("5. Run the inference scripts!")
    
    if args.check:
        check_prerequisites()
    
    if args.examples:
        print_usage_examples()
    
    if not any([args.setup, args.check, args.examples]):
        print("Semantic Segmentation Inference Tools - Example Usage")
        print("\nAvailable options:")
        print("  --setup     Create sample data structure and config files")
        print("  --check     Check prerequisites and dependencies")
        print("  --examples  Show detailed usage examples")
        print("\nFor detailed documentation, see: INFERENCE_README.md")


if __name__ == "__main__":
    main()