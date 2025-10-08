#!/usr/bin/env python
"""Setup and validation script for SRM-SVM Steganalysis project."""

import os
import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10+ is required. Current version:", 
              f"{version.major}.{version.minor}.{version.micro}")
        return False
    else:
        print("✅ Python version:", f"{version.major}.{version.minor}.{version.micro}")
        return True


def check_dependencies():
    """Check if all required packages are installed."""
    required_packages = [
        'numpy', 'scipy', 'scikit-learn', 'matplotlib',
        'PIL', 'joblib', 'tqdm', 'click'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                __import__('PIL')
            elif package == 'scikit-learn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {missing_packages}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies installed!")
        return True


def check_project_structure():
    """Check if project structure is correct."""
    required_dirs = [
        'src/steganalysis',
        'src/steganalysis/data',
        'src/steganalysis/features', 
        'src/steganalysis/models',
        'src/steganalysis/utils',
        'scripts',
        'tests'
    ]
    
    required_files = [
        'src/steganalysis/__init__.py',
        'src/steganalysis/data/dataset.py',
        'src/steganalysis/features/patches.py',
        'src/steganalysis/features/sparse_coding.py',
        'src/steganalysis/models/classifier.py',
        'src/steganalysis/utils/evaluation.py',
        'scripts/train.py',
        'scripts/infer.py',
        'requirements.txt',
        'README.md'
    ]
    
    missing_items = []
    
    for item in required_dirs:
        if not Path(item).is_dir():
            print(f"❌ Directory missing: {item}")
            missing_items.append(item)
        else:
            print(f"✅ {item}/")
    
    for item in required_files:
        if not Path(item).is_file():
            print(f"❌ File missing: {item}")
            missing_items.append(item)
        else:
            print(f"✅ {item}")
    
    if missing_items:
        print(f"\n❌ Missing items: {len(missing_items)}")
        return False
    else:
        print("\n✅ Project structure is correct!")
        return True


def test_imports():
    """Test if all modules can be imported."""
    sys.path.insert(0, str(Path('src').absolute()))
    
    try:
        from steganalysis.data.dataset import SteganalysisDataset
        from steganalysis.features.patches import extract_patches
        from steganalysis.features.sparse_coding import SparseDictionary
        from steganalysis.models.classifier import SteganalysisClassifier
        from steganalysis.utils.evaluation import setup_output_directory
        print("✅ All modules can be imported successfully!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def check_dataset_structure(data_root="dataset"):
    """Check if dataset structure is correct."""
    if not Path(data_root).exists():
        print(f"⚠️  Dataset directory not found: {data_root}")
        print("   This is optional for setup validation.")
        return True
    
    cover_dir = Path(data_root) / "cover" 
    stego_dir = Path(data_root) / "stego"
    
    if not cover_dir.exists():
        print(f"⚠️  Cover directory not found: {cover_dir}")
        return False
    
    if not stego_dir.exists():
        print(f"⚠️  Stego directory not found: {stego_dir}")
        return False
    
    # Count images
    cover_images = list(cover_dir.glob("*.png")) + list(cover_dir.glob("*.jpg"))
    stego_images = list(stego_dir.glob("*.png")) + list(stego_dir.glob("*.jpg"))
    
    print(f"✅ Dataset found:")
    print(f"   Cover images: {len(cover_images)}")
    print(f"   Stego images: {len(stego_images)}")
    
    if len(cover_images) == 0 or len(stego_images) == 0:
        print("⚠️  No images found in one or both directories")
        return False
    
    return True


def create_sample_data():
    """Create sample dataset for testing."""
    from PIL import Image
    import numpy as np
    
    sample_dir = Path("sample_data")
    cover_dir = sample_dir / "cover"
    stego_dir = sample_dir / "stego"
    
    cover_dir.mkdir(parents=True, exist_ok=True)
    stego_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating sample dataset...")
    
    for i in range(10):
        # Create random grayscale image
        img_array = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        img.save(cover_dir / f"cover_{i:03d}.png")
        
        # Create slightly modified stego image
        stego_array = img_array + np.random.randint(-10, 10, (64, 64), dtype=np.int16)
        stego_array = np.clip(stego_array, 0, 255).astype(np.uint8)
        stego_img = Image.fromarray(stego_array, mode='L')
        stego_img.save(stego_dir / f"stego_{i:03d}.png")
    
    print(f"✅ Sample dataset created in: {sample_dir}")
    return sample_dir


def main():
    """Run all validation checks."""
    print("🔍 SRM-SVM Steganalysis Project Setup Validation")
    print("=" * 50)
    
    checks_passed = 0
    total_checks = 4
    
    # Check 1: Python version
    print("\n1. Checking Python version...")
    if check_python_version():
        checks_passed += 1
    
    # Check 2: Dependencies
    print("\n2. Checking dependencies...")
    if check_dependencies():
        checks_passed += 1
    
    # Check 3: Project structure
    print("\n3. Checking project structure...")
    if check_project_structure():
        checks_passed += 1
    
    # Check 4: Module imports
    print("\n4. Testing module imports...")
    if test_imports():
        checks_passed += 1
    
    # Optional: Check dataset
    print("\n5. Checking dataset (optional)...")
    dataset_path = "dataset/BOSSBase 1.01 + 0.4 WOW"
    if Path(dataset_path).exists():
        check_dataset_structure(dataset_path)
    else:
        print(f"⚠️  Dataset not found at: {dataset_path}")
        print("   You can use your own dataset or create sample data.")
        
        response = input("\nCreate sample dataset for testing? (y/n): ")
        if response.lower() == 'y':
            sample_path = create_sample_data()
            print(f"\n   You can now test with: --data-root {sample_path}")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"Setup validation: {checks_passed}/{total_checks} checks passed")
    
    if checks_passed == total_checks:
        print("🎉 Setup is complete! You can now run training.")
        print("\nExample commands:")
        print("python scripts/train.py --data-root sample_data --dict-size 64 --max-patches 10000")
        print("python example_run.py --mode train")
    else:
        print("❌ Setup incomplete. Please fix the issues above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)