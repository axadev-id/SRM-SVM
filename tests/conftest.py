"""Test configuration and fixtures."""

import pytest
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def dummy_images(temp_dir):
    """Create dummy images for testing."""
    cover_dir = temp_dir / "cover"
    stego_dir = temp_dir / "stego"
    
    cover_dir.mkdir()
    stego_dir.mkdir()
    
    # Create some dummy grayscale images
    for i in range(5):
        # Create random image
        img_array = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        
        # Save as cover image
        img.save(cover_dir / f"cover_{i}.png")
        
        # Create slightly different image for stego
        stego_array = img_array + np.random.randint(-5, 5, (64, 64), dtype=np.int16)
        stego_array = np.clip(stego_array, 0, 255).astype(np.uint8)
        stego_img = Image.fromarray(stego_array, mode='L')
        stego_img.save(stego_dir / f"stego_{i}.png")
    
    return temp_dir, cover_dir, stego_dir


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return np.random.rand(32, 32).astype(np.float32)


@pytest.fixture
def sample_patches():
    """Create sample patches for testing."""
    return np.random.rand(100, 64).astype(np.float32)  # 100 patches of 8x8=64 dimensions