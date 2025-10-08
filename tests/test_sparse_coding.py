"""Tests for sparse coding functionality."""

import sys
from pathlib import Path
import numpy as np
import pytest
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from steganalysis.features.sparse_coding import SparseDictionary, aggregate_sparse_codes


class TestSparseCoding:
    """Test cases for sparse coding."""
    
    def test_dictionary_initialization(self):
        """Test sparse dictionary initialization."""
        dictionary = SparseDictionary(n_components=64, random_state=42)
        
        assert dictionary.n_components == 64
        assert dictionary.random_state == 42
        assert not dictionary.is_fitted
    
    def test_dictionary_fit(self, sample_patches):
        """Test dictionary fitting."""
        dictionary = SparseDictionary(n_components=32, random_state=42)
        dictionary.fit(sample_patches)
        
        assert dictionary.is_fitted
        assert dictionary.dictionary_ is not None
        assert dictionary.dictionary_.shape == (32, sample_patches.shape[1])
    
    def test_dictionary_transform_omp(self, sample_patches):
        """Test sparse coding with OMP solver."""
        dictionary = SparseDictionary(n_components=32, random_state=42)
        dictionary.fit(sample_patches)
        
        # Transform a subset of patches
        test_patches = sample_patches[:10]
        codes = dictionary.transform(test_patches, solver='omp', n_nonzero_coefs=5)
        
        assert codes.shape == (10, 32)
        
        # Check sparsity (each code should have at most 5 non-zero coefficients)
        for i in range(codes.shape[0]):
            non_zero_count = np.count_nonzero(codes[i])
            assert non_zero_count <= 5
    
    def test_dictionary_transform_lasso(self, sample_patches):
        """Test sparse coding with Lasso solver."""
        dictionary = SparseDictionary(n_components=32, random_state=42)
        dictionary.fit(sample_patches)
        
        test_patches = sample_patches[:10]
        codes = dictionary.transform(test_patches, solver='lasso', alpha_coding=0.1)
        
        assert codes.shape == (10, 32)
    
    def test_dictionary_transform_empty(self, sample_patches):
        """Test transform with empty patches."""
        dictionary = SparseDictionary(n_components=32, random_state=42)
        dictionary.fit(sample_patches)
        
        empty_patches = np.array([]).reshape(0, -1)
        codes = dictionary.transform(empty_patches)
        
        assert codes.shape == (0, 32)
    
    def test_dictionary_save_load(self, sample_patches, temp_dir):
        """Test dictionary save and load."""
        dictionary = SparseDictionary(n_components=32, random_state=42)
        dictionary.fit(sample_patches)
        
        # Save dictionary
        save_path = temp_dir / "test_dict.npz"
        dictionary.save(save_path)
        
        # Load dictionary
        new_dictionary = SparseDictionary()
        new_dictionary.load(save_path)
        
        assert new_dictionary.is_fitted
        assert new_dictionary.n_components == 32
        assert new_dictionary.random_state == 42
        np.testing.assert_array_equal(dictionary.dictionary_, new_dictionary.dictionary_)
    
    def test_aggregate_sparse_codes(self):
        """Test sparse code aggregation."""
        # Create dummy codes
        codes = np.random.randn(50, 64)  # 50 patches, 64-dimensional codes
        
        features = aggregate_sparse_codes(codes)
        
        # Should have 3 * 64 features (avg pool + max pool + sign hist)
        assert features.shape == (3 * 64,)
        
        # Check that features are finite
        assert np.all(np.isfinite(features))
    
    def test_aggregate_sparse_codes_empty(self):
        """Test aggregation with empty codes."""
        empty_codes = np.array([]).reshape(0, -1)
        features = aggregate_sparse_codes(empty_codes)
        
        # Should return zeros with default size
        assert features.shape == (3 * 256,)
        assert np.all(features == 0)
    
    def test_dictionary_not_fitted_error(self, sample_patches):
        """Test error when using unfitted dictionary."""
        dictionary = SparseDictionary(n_components=32)
        
        with pytest.raises(ValueError, match="Dictionary must be fitted"):
            dictionary.transform(sample_patches)
        
        with pytest.raises(ValueError, match="Dictionary must be fitted"):
            dictionary.save(Path("dummy.npz"))
    
    def test_invalid_solver(self, sample_patches):
        """Test error with invalid solver."""
        dictionary = SparseDictionary(n_components=32, random_state=42)
        dictionary.fit(sample_patches)
        
        with pytest.raises(ValueError, match="Unknown solver"):
            dictionary.transform(sample_patches, solver='invalid')