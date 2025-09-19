"""
Mean IoU Computation Script for Test Dataset

This script computes the mean Intersection over Union (IoU) for a test dataset
using a trained segmentation model. It provides detailed per-class IoU metrics
and overall performance statistics.

Features:
- Flexible dataset loading from custom directory structures
- Per-class and mean IoU computation
- Detailed performance statistics and timing
- Support for various image formats
- Modular and extensible design
"""

import os
import json
import argparse
import time
from pathlib import Path

import torch
from torchvision import transforms
from torch.utils.data import default_collate
import numpy as np
from PIL import Image
from tqdm import tqdm

from src import fcn_resnet50
from train_utils.distributed_utils import ConfusionMatrix
import transforms as T


def time_synchronized():
    """Synchronized time measurement for CUDA operations."""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


class TestDataset(torch.utils.data.Dataset):
    """
    Custom dataset for test data with image-mask pairs.
    
    Args:
        image_dir (str): Directory containing test images
        mask_dir (str): Directory containing corresponding ground truth masks
        transforms: Data transformations to apply
        image_extensions (tuple): Supported image file extensions
    """
    
    def __init__(self, image_dir, mask_dir, transforms=None, 
                 image_extensions=('.jpg', '.jpeg', '.png', '.bmp')):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transforms = transforms
        
        # Find all image files
        self.image_files = []
        for ext in image_extensions:
            self.image_files.extend(self.image_dir.glob(f'*{ext}'))
            self.image_files.extend(self.image_dir.glob(f'*{ext.upper()}'))
        
        # Filter to only include images with corresponding masks
        valid_images = []
        for img_file in self.image_files:
            mask_file = self.mask_dir / f"{img_file.stem}.png"
            if mask_file.exists():
                valid_images.append(img_file)
            else:
                # Try alternative mask extensions
                for mask_ext in ['.png', '.jpg', '.jpeg']:
                    alt_mask_file = self.mask_dir / f"{img_file.stem}{mask_ext}"
                    if alt_mask_file.exists():
                        valid_images.append(img_file)
                        break
        
        self.image_files = sorted(valid_images)
        print(f"Found {len(self.image_files)} valid image-mask pairs")
        
        if len(self.image_files) == 0:
            raise ValueError(f"No valid image-mask pairs found in {image_dir} and {mask_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Load corresponding mask
        mask_path = self.mask_dir / f"{img_path.stem}.png"
        if not mask_path.exists():
            # Try alternative extensions
            for mask_ext in ['.jpg', '.jpeg']:
                alt_mask_path = self.mask_dir / f"{img_path.stem}{mask_ext}"
                if alt_mask_path.exists():
                    mask_path = alt_mask_path
                    break
        
        mask = Image.open(mask_path)
        
        # Apply transforms
        if self.transforms:
            image, mask = self.transforms(image, mask)
        
        return image, mask, str(img_path)


class SegmentationPresetEval:
    """Data preprocessing pipeline for evaluation."""
    
    def __init__(self, base_size=520, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.RandomResize(base_size, base_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    
    def __call__(self, img, target):
        """
        Apply transforms to image and target.
        
        Args:
            img: PIL Image
            target: PIL Image (ground truth mask)
            
        Returns:
            tuple: (transformed_image, transformed_target)
        """
        return self.transforms(img, target)


class IoUEvaluator:
    """
    Evaluates mean IoU for a segmentation model on test data.
    
    Args:
        model_path (str): Path to trained model weights
        num_classes (int): Number of segmentation classes (excluding background)
        device (str): Device for inference ('cuda' or 'cpu')
        class_names (list): Optional list of class names for detailed reporting
    """
    
    def __init__(self, model_path, num_classes=20, device="cuda", class_names=None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes + 1  # Include background
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.class_names = ["background"] + self.class_names
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Initialize confusion matrix
        self.confusion_matrix = ConfusionMatrix(self.num_classes)
        
        print(f"IoU Evaluator initialized on {self.device}")
        print(f"Model: FCN ResNet50, Classes: {num_classes}")
    
    def _load_model(self, model_path):
        """Load the trained segmentation model."""
        assert os.path.exists(model_path), f"Model weights {model_path} not found."
        
        # Create model (aux=False for inference)
        model = fcn_resnet50(aux=False, num_classes=self.num_classes)
        
        # Load weights
        weights_dict = torch.load(model_path, map_location='cpu')
        if 'model' in weights_dict:
            weights_dict = weights_dict['model']
        
        # Remove auxiliary classifier weights if present
        for k in list(weights_dict.keys()):
            if "aux" in k:
                del weights_dict[k]
        
        model.load_state_dict(weights_dict)
        model.to(self.device)
        model.eval()
        
        return model
    
    def evaluate_single(self, image, target):
        """
        Evaluate a single image-mask pair.
        
        Args:
            image: Input image tensor
            target: Ground truth mask tensor
            
        Returns:
            inference_time: Time taken for inference
        """
        with torch.no_grad():
            t_start = time_synchronized()
            output = self.model(image.to(self.device))
            t_end = time_synchronized()
            
            # Get predictions
            prediction = output['out'].argmax(1)
            
            # Update confusion matrix
            self.confusion_matrix.update(target.flatten(), prediction.flatten())
            
            return t_end - t_start
    
    def evaluate_dataset(self, test_loader):
        """
        Evaluate the entire test dataset.
        
        Args:
            test_loader: DataLoader for test dataset
            
        Returns:
            dict: Comprehensive evaluation results
        """
        print("Starting evaluation...")
        
        total_time = 0
        num_samples = 0
        
        # Reset confusion matrix
        self.confusion_matrix.reset()
        
        # Process each batch
        for image, target, img_paths in tqdm(test_loader, desc="Evaluating"):
            inference_time = self.evaluate_single(image, target)
            total_time += inference_time
            num_samples += image.size(0)
        
        # Compute metrics
        acc_global, acc_per_class, iou_per_class = self.confusion_matrix.compute()
        
        # Prepare results
        results = {
            'mean_iou': iou_per_class.mean().item() * 100,
            'global_accuracy': acc_global.item() * 100,
            'per_class_accuracy': acc_per_class.cpu().numpy() * 100,
            'per_class_iou': iou_per_class.cpu().numpy() * 100,
            'num_samples': num_samples,
            'avg_inference_time': total_time / num_samples if num_samples > 0 else 0,
            'total_time': total_time
        }
        
        return results
    
    def print_detailed_results(self, results):
        """Print detailed evaluation results."""
        print("\n" + "="*80)
        print("DETAILED EVALUATION RESULTS")
        print("="*80)
        
        print(f"Dataset size: {results['num_samples']} images")
        print(f"Total evaluation time: {results['total_time']:.2f}s")
        print(f"Average inference time: {results['avg_inference_time']*1000:.1f}ms per image")
        
        print(f"\nOverall Performance:")
        print(f"  Mean IoU: {results['mean_iou']:.2f}%")
        print(f"  Global Accuracy: {results['global_accuracy']:.2f}%")
        
        print(f"\nPer-Class Results:")
        print(f"{'Class':<15} {'IoU (%)':<10} {'Accuracy (%)':<15}")
        print("-" * 40)
        
        for i, class_name in enumerate(self.class_names):
            if i < len(results['per_class_iou']):
                iou = results['per_class_iou'][i]
                acc = results['per_class_accuracy'][i]
                print(f"{class_name:<15} {iou:<10.2f} {acc:<15.2f}")
        
        print("="*80)
    
    def save_results(self, results, output_path):
        """Save evaluation results to JSON file."""
        # Prepare serializable results
        serializable_results = {
            'mean_iou': float(results['mean_iou']),
            'global_accuracy': float(results['global_accuracy']),
            'per_class_iou': {name: float(iou) for name, iou in 
                             zip(self.class_names, results['per_class_iou'])},
            'per_class_accuracy': {name: float(acc) for name, acc in 
                                  zip(self.class_names, results['per_class_accuracy'])},
            'num_samples': int(results['num_samples']),
            'avg_inference_time': float(results['avg_inference_time']),
            'total_time': float(results['total_time'])
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to: {output_path}")


def load_class_names(class_names_path):
    """Load class names from JSON file."""
    if not os.path.exists(class_names_path):
        return None
    
    with open(class_names_path, 'r') as f:
        class_dict = json.load(f)
    
    # Handle different JSON formats
    if isinstance(class_dict, dict):
        if all(isinstance(v, list) for v in class_dict.values()):
            # Format: {"class_0": ["background"], "class_1": ["aeroplane"], ...}
            class_names = []
            for i in range(len(class_dict)):
                if str(i) in class_dict:
                    class_names.append(class_dict[str(i)][0])
                elif i in class_dict:
                    class_names.append(class_dict[i][0])
            return class_names
        else:
            # Format: {"0": "background", "1": "aeroplane", ...}
            return [class_dict[str(i)] for i in range(len(class_dict))]
    elif isinstance(class_dict, list):
        return class_dict
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Compute mean IoU for test dataset")
    
    # Model parameters
    parser.add_argument("--model-path", required=True,
                       help="Path to trained model weights")
    parser.add_argument("--num-classes", type=int, default=20,
                       help="Number of segmentation classes (excluding background)")
    parser.add_argument("--class-names", default="./pascal_voc_classes.json",
                       help="Path to class names JSON file")
    
    # Dataset parameters
    parser.add_argument("--image-dir", required=True,
                       help="Directory containing test images")
    parser.add_argument("--mask-dir", required=True,
                       help="Directory containing ground truth masks")
    
    # Processing options
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of worker processes for data loading")
    parser.add_argument("--device", default="cuda",
                       help="Device for inference (cuda/cpu)")
    
    # Output options
    parser.add_argument("--output-file", default="./test_iou_results.json",
                       help="Path to save evaluation results")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed per-class results")
    
    args = parser.parse_args()
    
    # Load class names
    class_names = load_class_names(args.class_names)
    if class_names is None:
        print(f"Warning: Could not load class names from {args.class_names}")
        class_names = [f"class_{i}" for i in range(args.num_classes)]
    
    # Create test dataset
    test_dataset = TestDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        transforms=SegmentationPresetEval(base_size=520)
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        collate_fn=lambda batch: default_collate(batch)  # Use default collate for simplicity
    )
    
    # Initialize evaluator
    evaluator = IoUEvaluator(
        model_path=args.model_path,
        num_classes=args.num_classes,
        device=args.device,
        class_names=class_names
    )
    
    # Run evaluation
    results = evaluator.evaluate_dataset(test_loader)
    
    # Print results
    if args.verbose:
        evaluator.print_detailed_results(results)
    else:
        print(f"Mean IoU: {results['mean_iou']:.2f}%")
        print(f"Global Accuracy: {results['global_accuracy']:.2f}%")
        print(f"Processed {results['num_samples']} images in {results['total_time']:.2f}s")
    
    # Save results
    evaluator.save_results(results, args.output_file)


if __name__ == '__main__':
    main()