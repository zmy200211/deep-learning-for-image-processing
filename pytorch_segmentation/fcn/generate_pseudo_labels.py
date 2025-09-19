"""
Pseudo Label Generation Script for Semantic Segmentation

This script generates pseudo labels for unlabeled data using a trained segmentation model.
The pseudo labels are saved as image files in a specified output directory.

Features:
- Batch processing of unlabeled images
- Configurable confidence thresholding
- Support for various image formats
- Modular and well-documented design
"""

import os
import json
import argparse
import time
from pathlib import Path

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm

from src import fcn_resnet50


def time_synchronized():
    """Synchronized time measurement for CUDA operations."""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


class PseudoLabelGenerator:
    """
    Generates pseudo labels for unlabeled data using a trained segmentation model.
    
    Args:
        model_path (str): Path to the trained model weights
        num_classes (int): Number of segmentation classes (excluding background)
        palette_path (str): Path to color palette JSON file
        device (str): Device for inference ('cuda' or 'cpu')
        confidence_threshold (float): Minimum confidence for pseudo label generation
    """
    
    def __init__(self, model_path, num_classes=20, palette_path="./palette.json", 
                 device="cuda", confidence_threshold=0.9):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        
        # Load color palette
        self.palette = self._load_palette(palette_path)
        
        # Initialize model
        self.model = self._load_model(model_path)
        
        # Data preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(520),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                               std=(0.229, 0.224, 0.225))
        ])
        
        print(f"Pseudo label generator initialized on {self.device}")
        print(f"Model: FCN ResNet50, Classes: {num_classes}, Confidence threshold: {confidence_threshold}")
    
    def _load_palette(self, palette_path):
        """Load color palette for visualization."""
        assert os.path.exists(palette_path), f"Palette file {palette_path} not found."
        
        with open(palette_path, "rb") as f:
            palette_dict = json.load(f)
            palette = []
            for v in palette_dict.values():
                palette += v
        return palette
    
    def _load_model(self, model_path):
        """Load the trained segmentation model."""
        assert os.path.exists(model_path), f"Model weights {model_path} not found."
        
        # Create model (aux=False for inference)
        model = fcn_resnet50(aux=False, num_classes=self.num_classes + 1)
        
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
    
    def _preprocess_image(self, image_path):
        """Preprocess a single image for inference."""
        original_img = Image.open(image_path).convert('RGB')
        original_size = original_img.size
        
        # Apply transformations
        img_tensor = self.transform(original_img)
        img_tensor = torch.unsqueeze(img_tensor, dim=0)
        
        return img_tensor, original_img, original_size
    
    def _postprocess_prediction(self, output, original_size, apply_confidence_threshold=True):
        """
        Post-process model output to generate pseudo labels.
        
        Args:
            output: Model output tensor
            original_size: Original image size (width, height)
            apply_confidence_threshold: Whether to apply confidence thresholding
            
        Returns:
            pseudo_label: Numpy array of predicted labels
            confidence_mask: Binary mask indicating high-confidence pixels
        """
        # Get class predictions and confidence scores
        probabilities = torch.softmax(output['out'], dim=1)
        max_probs, predictions = torch.max(probabilities, dim=1)
        
        # Squeeze batch dimension and move to CPU
        predictions = predictions.squeeze(0).cpu().numpy().astype(np.uint8)
        max_probs = max_probs.squeeze(0).cpu().numpy()
        
        # Apply confidence thresholding if requested
        confidence_mask = np.ones_like(predictions, dtype=bool)
        if apply_confidence_threshold:
            confidence_mask = max_probs >= self.confidence_threshold
            # Set low-confidence pixels to ignore index (255)
            predictions[~confidence_mask] = 255
        
        # Resize to original image size
        predictions_img = Image.fromarray(predictions)
        predictions_img = predictions_img.resize(original_size, Image.NEAREST)
        predictions = np.array(predictions_img)
        
        return predictions, confidence_mask
    
    def generate_pseudo_label(self, image_path, apply_confidence_threshold=True):
        """
        Generate pseudo label for a single image.
        
        Args:
            image_path (str): Path to input image
            apply_confidence_threshold (bool): Whether to apply confidence thresholding
            
        Returns:
            tuple: (pseudo_label_array, confidence_mask, inference_time)
        """
        # Preprocess image
        img_tensor, original_img, original_size = self._preprocess_image(image_path)
        
        # Inference
        with torch.no_grad():
            t_start = time_synchronized()
            output = self.model(img_tensor.to(self.device))
            t_end = time_synchronized()
            inference_time = t_end - t_start
        
        # Post-process
        pseudo_label, confidence_mask = self._postprocess_prediction(
            output, original_size, apply_confidence_threshold
        )
        
        return pseudo_label, confidence_mask, inference_time
    
    def save_pseudo_label(self, pseudo_label, output_path, save_colored=True):
        """
        Save pseudo label as image file.
        
        Args:
            pseudo_label: Numpy array of predicted labels
            output_path: Output file path
            save_colored: Whether to save colored version using palette
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save grayscale version
        label_img = Image.fromarray(pseudo_label)
        label_img.save(output_path)
        
        # Save colored version if requested
        if save_colored:
            colored_output_path = output_path.replace('.png', '_colored.png')
            colored_img = Image.fromarray(pseudo_label)
            colored_img.putpalette(self.palette)
            colored_img.save(colored_output_path)
    
    def generate_batch_pseudo_labels(self, input_dir, output_dir, 
                                   apply_confidence_threshold=True, 
                                   save_colored=True,
                                   image_extensions=('.jpg', '.jpeg', '.png', '.bmp')):
        """
        Generate pseudo labels for all images in a directory.
        
        Args:
            input_dir (str): Directory containing unlabeled images
            output_dir (str): Directory to save pseudo labels
            apply_confidence_threshold (bool): Whether to apply confidence thresholding
            save_colored (bool): Whether to save colored versions
            image_extensions (tuple): Supported image file extensions
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'**/*{ext}'))
            image_files.extend(input_path.glob(f'**/*{ext.upper()}'))
        
        if not image_files:
            print(f"No image files found in {input_dir}")
            return
        
        print(f"Found {len(image_files)} images to process")
        print(f"Output directory: {output_dir}")
        
        total_time = 0
        processed_count = 0
        high_confidence_count = 0
        
        # Process each image
        for img_file in tqdm(image_files, desc="Generating pseudo labels"):
            try:
                # Generate pseudo label
                pseudo_label, confidence_mask, inference_time = self.generate_pseudo_label(
                    str(img_file), apply_confidence_threshold
                )
                
                # Compute statistics
                total_time += inference_time
                processed_count += 1
                
                if apply_confidence_threshold:
                    high_conf_ratio = confidence_mask.mean()
                    if high_conf_ratio > 0.7:  # At least 70% high confidence
                        high_confidence_count += 1
                
                # Save pseudo label
                relative_path = img_file.relative_to(input_path)
                output_file = output_path / relative_path.with_suffix('.png')
                
                self.save_pseudo_label(pseudo_label, str(output_file), save_colored)
                
            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
                continue
        
        # Print statistics
        avg_time = total_time / processed_count if processed_count > 0 else 0
        print(f"\nProcessing completed!")
        print(f"Processed: {processed_count}/{len(image_files)} images")
        print(f"Average inference time: {avg_time:.3f}s per image")
        
        if apply_confidence_threshold:
            print(f"High-confidence pseudo labels: {high_confidence_count}/{processed_count}")
            print(f"Confidence threshold: {self.confidence_threshold}")


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo labels for semantic segmentation")
    
    # Model parameters
    parser.add_argument("--model-path", required=True, 
                       help="Path to trained model weights")
    parser.add_argument("--num-classes", type=int, default=20,
                       help="Number of segmentation classes (excluding background)")
    parser.add_argument("--palette-path", default="./palette.json",
                       help="Path to color palette JSON file")
    
    # Input/Output
    parser.add_argument("--input-dir", required=True,
                       help="Directory containing unlabeled images")
    parser.add_argument("--output-dir", required=True,
                       help="Directory to save pseudo labels")
    
    # Processing options
    parser.add_argument("--confidence-threshold", type=float, default=0.9,
                       help="Minimum confidence for pseudo label generation")
    parser.add_argument("--no-confidence-filtering", action="store_true",
                       help="Disable confidence-based filtering")
    parser.add_argument("--no-colored", action="store_true",
                       help="Don't save colored pseudo label images")
    parser.add_argument("--device", default="cuda",
                       help="Device for inference (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = PseudoLabelGenerator(
        model_path=args.model_path,
        num_classes=args.num_classes,
        palette_path=args.palette_path,
        device=args.device,
        confidence_threshold=args.confidence_threshold
    )
    
    # Generate pseudo labels
    generator.generate_batch_pseudo_labels(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        apply_confidence_threshold=not args.no_confidence_filtering,
        save_colored=not args.no_colored
    )


if __name__ == '__main__':
    main()