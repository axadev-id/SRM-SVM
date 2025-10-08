import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from steganalysis.data.dataset import SteganalysisDataset
    from steganalysis.features.patches import extract_patches
    from steganalysis.features.sparse_coding import SparseDictionary
    from steganalysis.models.classifier import SteganalysisClassifier
    from steganalysis.utils.evaluation import setup_output_directory
    print('✓ All imports successful!')
    print('✓ Project structure is correct!')
except ImportError as e:
    print(f'✗ Import error: {e}')
except Exception as e:
    print(f'✗ Unexpected error: {e}')