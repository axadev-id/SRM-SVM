"""Tests for patch extraction functionality."""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from steganalysis.features.patches import (
    extract_patches, patches_to_vectors, extract_patches_from_images
)


class TestPatchExtraction:
    """Test cases for patch extraction."""
    
    def test_extract_patches_basic(self, sample_image):
        """Test basic patch extraction."""
        patches = extract_patches(sample_image, patch_size=8, stride=4)
        
        assert patches.ndim == 3
        assert patches.shape[1] == 8
        assert patches.shape[2] == 8
        
        # Check number of patches
        expected_patches_x = (sample_image.shape[1] - 8) // 4 + 1
        expected_patches_y = (sample_image.shape[0] - 8) // 4 + 1
        expected_total = expected_patches_x * expected_patches_y
        
        assert patches.shape[0] == expected_total
    
    def test_extract_patches_small_image(self):
        """Test patch extraction with image smaller than patch size."""
        small_image = np.random.rand(4, 4).astype(np.float32)
        patches = extract_patches(small_image, patch_size=8, stride=4)
        
        assert patches.size == 0
    
    def test_patches_to_vectors(self, sample_image):
        """Test conversion of patches to vectors."""
        patches = extract_patches(sample_image, patch_size=8, stride=8)
        vectors = patches_to_vectors(patches)
        
        assert vectors.ndim == 2
        assert vectors.shape[0] == patches.shape[0]
        assert vectors.shape[1] == 8 * 8
    
    def test_patches_to_vectors_empty(self):
        """Test conversion of empty patches."""
        empty_patches = np.array([])
        vectors = patches_to_vectors(empty_patches)
        
        assert vectors.shape[0] == 0
    
    def test_extract_patches_from_images(self, sample_image):
        """Test patch extraction from multiple images."""
        images = [sample_image, sample_image * 0.5, sample_image * 2.0]
        
        all_patches = extract_patches_from_images(
            images, patch_size=8, stride=8, max_patches_per_image=10
        )
        
        assert all_patches.ndim == 2
        assert all_patches.shape[1] == 8 * 8
        assert all_patches.shape[0] <= 3 * 10  # At most 10 patches per image
    
    def test_extract_patches_different_stride(self, sample_image):
        """Test patch extraction with different stride values."""
        patches_stride_2 = extract_patches(sample_image, patch_size=8, stride=2)
        patches_stride_4 = extract_patches(sample_image, patch_size=8, stride=4)
        
        # Smaller stride should produce more patches
        assert patches_stride_2.shape[0] > patches_stride_4.shape[0]