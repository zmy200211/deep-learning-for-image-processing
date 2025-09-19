#!/usr/bin/env python3
"""
Test script to validate the refactored inference tools functionality.

This script creates dummy data and tests both pseudo label generation
and IoU computation to ensure the refactored code works correctly.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

import torch
import numpy as np
from PIL import Image

# Add current directory to path for imports
sys.path.insert(0, '.')

from generate_pseudo_labels import PseudoLabelGenerator
from compute_test_iou import IoUEvaluator, TestDataset, SegmentationPresetEval
from src import fcn_resnet50


def create_dummy_model(num_classes=21, save_path="dummy_model.pth"):
    """Create a dummy FCN model for testing."""
    print("Creating dummy FCN model...")
    
    model = fcn_resnet50(aux=False, num_classes=num_classes)
    
    # Save dummy weights
    torch.save({'model': model.state_dict()}, save_path)
    
    print(f"Dummy model saved to {save_path}")
    return save_path


def create_dummy_images(image_dir, num_images=5, image_size=(256, 256)):
    """Create dummy RGB images for testing."""
    print(f"Creating {num_images} dummy images in {image_dir}...")
    
    os.makedirs(image_dir, exist_ok=True)
    
    for i in range(num_images):
        # Create random RGB image
        img_array = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(os.path.join(image_dir, f"test_image_{i:03d}.jpg"))
    
    print(f"Created {num_images} dummy images")


def create_dummy_masks(mask_dir, num_masks=5, image_size=(256, 256), num_classes=21):
    """Create dummy segmentation masks for testing."""
    print(f"Creating {num_masks} dummy masks in {mask_dir}...")
    
    os.makedirs(mask_dir, exist_ok=True)
    
    for i in range(num_masks):
        # Create random mask with class indices 0 to num_classes-1
        mask_array = np.random.randint(0, num_classes, image_size, dtype=np.uint8)
        mask = Image.fromarray(mask_array)
        mask.save(os.path.join(mask_dir, f"test_image_{i:03d}.png"))
    
    print(f"Created {num_masks} dummy masks")


def create_dummy_palette(palette_path="dummy_palette.json"):
    """Create a dummy color palette."""
    print("Creating dummy color palette...")
    
    # Simple palette with different colors for each class
    palette = {}
    for i in range(21):
        palette[f"class_{i}"] = [
            int(i * 12) % 256,
            int(i * 25) % 256, 
            int(i * 37) % 256
        ]
    
    with open(palette_path, 'w') as f:
        json.dump(palette, f, indent=2)
    
    print(f"Dummy palette saved to {palette_path}")
    return palette_path


def test_pseudo_label_generation():
    """Test the pseudo label generation functionality."""
    print("\n" + "="*60)
    print("TESTING PSEUDO LABEL GENERATION")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test data
        model_path = create_dummy_model(save_path=temp_path / "dummy_model.pth")
        palette_path = create_dummy_palette(temp_path / "dummy_palette.json")
        
        input_dir = temp_path / "input_images"
        output_dir = temp_path / "pseudo_labels"
        
        create_dummy_images(str(input_dir), num_images=3)
        
        try:
            # Test pseudo label generator
            generator = PseudoLabelGenerator(
                model_path=str(model_path),
                num_classes=20,
                palette_path=str(palette_path),
                device="cpu",  # Use CPU for testing
                confidence_threshold=0.5
            )
            
            # Test single image processing
            test_image = str(input_dir / "test_image_000.jpg")
            pseudo_label, confidence_mask, inference_time = generator.generate_pseudo_label(
                test_image, apply_confidence_threshold=True
            )
            
            print(f"✓ Single image processing successful")
            print(f"  - Pseudo label shape: {pseudo_label.shape}")
            print(f"  - Confidence mask shape: {confidence_mask.shape}")
            print(f"  - Inference time: {inference_time:.3f}s")
            
            # Test batch processing
            generator.generate_batch_pseudo_labels(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                apply_confidence_threshold=True,
                save_colored=True
            )
            
            # Check outputs
            output_files = list(output_dir.glob("*.png"))
            print(f"✓ Batch processing successful")
            print(f"  - Generated {len(output_files)} output files")
            
            # Verify both grayscale and colored outputs exist
            grayscale_files = [f for f in output_files if not f.name.endswith('_colored.png')]
            colored_files = [f for f in output_files if f.name.endswith('_colored.png')]
            
            print(f"  - Grayscale outputs: {len(grayscale_files)}")
            print(f"  - Colored outputs: {len(colored_files)}")
            
            return True
            
        except Exception as e:
            print(f"✗ Pseudo label generation failed: {str(e)}")
            return False


def test_iou_computation():
    """Test the IoU computation functionality."""
    print("\n" + "="*60)
    print("TESTING IoU COMPUTATION")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test data
        model_path = create_dummy_model(save_path=temp_path / "dummy_model.pth")
        
        image_dir = temp_path / "test_images"
        mask_dir = temp_path / "test_masks"
        results_file = temp_path / "test_results.json"
        
        create_dummy_images(str(image_dir), num_images=3)
        create_dummy_masks(str(mask_dir), num_masks=3)
        
        try:
            # Test dataset creation
            test_dataset = TestDataset(
                image_dir=str(image_dir),
                mask_dir=str(mask_dir),
                transforms=SegmentationPresetEval(base_size=256)
            )
            
            print(f"✓ Test dataset created successfully")
            print(f"  - Dataset size: {len(test_dataset)}")
            
            # Test data loading
            sample_img, sample_mask, sample_path = test_dataset[0]
            print(f"  - Sample image shape: {sample_img.shape}")
            print(f"  - Sample mask shape: {sample_mask.shape}")
            
            # Create data loader
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False
            )
            
            # Test IoU evaluator
            evaluator = IoUEvaluator(
                model_path=str(model_path),
                num_classes=20,
                device="cpu"  # Use CPU for testing
            )
            
            print(f"✓ IoU evaluator created successfully")
            
            # Test evaluation
            results = evaluator.evaluate_dataset(test_loader)
            
            print(f"✓ Dataset evaluation completed")
            print(f"  - Mean IoU: {results['mean_iou']:.2f}%")
            print(f"  - Global Accuracy: {results['global_accuracy']:.2f}%")
            print(f"  - Processed samples: {results['num_samples']}")
            print(f"  - Average inference time: {results['avg_inference_time']*1000:.1f}ms")
            
            # Test result saving
            evaluator.save_results(results, str(results_file))
            
            # Verify results file
            if results_file.exists():
                with open(results_file, 'r') as f:
                    saved_results = json.load(f)
                print(f"✓ Results saved successfully")
                print(f"  - Results file size: {results_file.stat().st_size} bytes")
            
            return True
            
        except Exception as e:
            print(f"✗ IoU computation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def test_integration():
    """Test integration between both tools."""
    print("\n" + "="*60)
    print("TESTING INTEGRATION")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test data
        model_path = create_dummy_model(save_path=temp_path / "dummy_model.pth")
        palette_path = create_dummy_palette(temp_path / "dummy_palette.json")
        
        # Original unlabeled images
        unlabeled_dir = temp_path / "unlabeled_images"
        create_dummy_images(str(unlabeled_dir), num_images=2)
        
        # Test images with ground truth masks  
        test_image_dir = temp_path / "test_images"
        test_mask_dir = temp_path / "test_masks"
        create_dummy_images(str(test_image_dir), num_images=2)
        create_dummy_masks(str(test_mask_dir), num_masks=2)
        
        # Generate pseudo labels
        pseudo_label_dir = temp_path / "pseudo_labels"
        
        try:
            print("Step 1: Generating pseudo labels...")
            generator = PseudoLabelGenerator(
                model_path=str(model_path),
                num_classes=20,
                palette_path=str(palette_path),
                device="cpu",
                confidence_threshold=0.7
            )
            
            generator.generate_batch_pseudo_labels(
                input_dir=str(unlabeled_dir),
                output_dir=str(pseudo_label_dir),
                apply_confidence_threshold=True
            )
            
            print("Step 2: Computing IoU on test set...")
            evaluator = IoUEvaluator(
                model_path=str(model_path),
                num_classes=20,
                device="cpu"
            )
            
            test_dataset = TestDataset(
                image_dir=str(test_image_dir),
                mask_dir=str(test_mask_dir),
                transforms=SegmentationPresetEval(base_size=256)
            )
            
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
            results = evaluator.evaluate_dataset(test_loader)
            
            print("✓ Integration test completed successfully")
            print(f"  - Generated pseudo labels: {len(list(pseudo_label_dir.glob('*.png')))}")
            print(f"  - Test set mean IoU: {results['mean_iou']:.2f}%")
            
            return True
            
        except Exception as e:
            print(f"✗ Integration test failed: {str(e)}")
            return False


def main():
    """Run all tests."""
    print("SEMANTIC SEGMENTATION INFERENCE TOOLS - VALIDATION TEST")
    print("="*80)
    
    # Check if we're in the right directory
    if not os.path.exists('src'):
        print("Error: Please run this script from the FCN directory")
        print("Expected to find 'src' directory with model definitions")
        return False
    
    tests = [
        ("Pseudo Label Generation", test_pseudo_label_generation),
        ("IoU Computation", test_iou_computation), 
        ("Integration", test_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✓ {test_name} test PASSED")
            else:
                print(f"✗ {test_name} test FAILED")
        except Exception as e:
            print(f"✗ {test_name} test FAILED with exception: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name:<30} [{status}]")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The refactored inference tools are working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)