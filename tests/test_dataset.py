"""Tests for dataset functionality."""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from steganalysis.data.dataset import (
    SteganalysisDataset, load_image_as_grayscale, apply_srm_residual
)


class TestDataset:
    """Test cases for dataset functionality."""
    
    def test_dataset_initialization(self, dummy_images):
        """Test dataset initialization."""
        data_root, cover_dir, stego_dir = dummy_images
        
        dataset = SteganalysisDataset(data_root, "cover", "stego")
        
        assert dataset.data_root == data_root
        assert dataset.cover_path == cover_dir
        assert dataset.stego_path == stego_dir
    
    def test_dataset_invalid_paths(self, temp_dir):
        """Test dataset with invalid paths."""
        with pytest.raises(ValueError, match="Cover directory does not exist"):
            SteganalysisDataset(temp_dir, "nonexistent_cover", "nonexistent_stego")
    
    def test_load_image_paths(self, dummy_images):
        """Test loading image paths and labels."""
        data_root, cover_dir, stego_dir = dummy_images
        dataset = SteganalysisDataset(data_root, "cover", "stego")
        
        image_paths, labels = dataset.load_image_paths()
        
        assert len(image_paths) == 10  # 5 cover + 5 stego
        assert len(labels) == 10
        
        # Check labels (first 5 should be 0 for cover, next 5 should be 1 for stego)
        assert np.all(labels[:5] == 0)
        assert np.all(labels[5:] == 1)
    
    def test_split_dataset(self, dummy_images):
        """Test dataset splitting."""
        data_root, cover_dir, stego_dir = dummy_images
        dataset = SteganalysisDataset(data_root, "cover", "stego")
        
        (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = \
            dataset.split_dataset(val_size=0.2, test_size=0.2, random_state=42)
        
        total_size = len(train_paths) + len(val_paths) + len(test_paths)
        assert total_size == 10
        
        # Check that all splits have both classes
        assert len(np.unique(train_labels)) <= 2
        assert len(np.unique(val_labels)) <= 2
        assert len(np.unique(test_labels)) <= 2
    
    def test_load_image_as_grayscale(self, dummy_images):
        """Test grayscale image loading."""
        data_root, cover_dir, stego_dir = dummy_images
        
        # Get first cover image
        cover_images = list(cover_dir.glob("*.png"))
        image_path = cover_images[0]
        
        image = load_image_as_grayscale(image_path)
        
        assert image is not None
        assert image.dtype == np.float32
        assert 0 <= image.min() <= image.max() <= 1
        assert image.ndim == 2  # Grayscale image
    
    def test_load_nonexistent_image(self):
        """Test loading nonexistent image."""
        fake_path = Path("nonexistent_image.png")
        image = load_image_as_grayscale(fake_path)
        
        assert image is None
    
    def test_apply_srm_residual(self, sample_image):
        """Test SRM residual filtering."""
        # Test first-order residual
        residual_first = apply_srm_residual(sample_image, "first_order")
        assert residual_first.shape == sample_image.shape
        assert residual_first.dtype == sample_image.dtype
        
        # Test second-order residual
        residual_second = apply_srm_residual(sample_image, "second_order")
        assert residual_second.shape == sample_image.shape
        assert residual_second.dtype == sample_image.dtype
    
    def test_apply_srm_residual_invalid_kernel(self, sample_image):
        """Test SRM residual with invalid kernel type."""
        with pytest.raises(ValueError, match="Unknown kernel type"):
            apply_srm_residual(sample_image, "invalid_kernel")