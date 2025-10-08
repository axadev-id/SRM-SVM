#!/usr/bin/env python3
"""
TRAINING VALIDATION SCRIPT
Quick check to ensure everything is ready for optimal training.
"""

import os
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_environment():
    """Validate training environment."""
    logger.info("🔍 VALIDATING TRAINING ENVIRONMENT")
    logger.info("="*50)
    
    issues = []
    
    # Check Python path
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    
    # Check imports
    try:
        from steganalysis.data.dataset import SteganalysisDataset
        from steganalysis.features.patches import extract_patches_from_images
        from steganalysis.features.sparse_coding import SparseDictionary
        logger.info("✅ Custom modules import successfully")
    except ImportError as e:
        issues.append(f"❌ Import error: {e}")
    
    # Check sklearn
    try:
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import GridSearchCV
        logger.info("✅ Sklearn modules available")
    except ImportError as e:
        issues.append(f"❌ Sklearn error: {e}")
    
    # Check numpy/scipy
    try:
        import numpy as np
        import scipy
        import joblib
        logger.info("✅ Scientific computing libraries available")
    except ImportError as e:
        issues.append(f"❌ NumPy/SciPy error: {e}")
    
    # Check dataset
    dataset_path = Path('dataset/BOSSBase 1.01 + 0.4 WOW')
    if dataset_path.exists():
        cover_path = dataset_path / 'cover'
        stego_path = dataset_path / 'stego'
        
        if cover_path.exists() and stego_path.exists():
            cover_files = list(cover_path.glob('*.png'))
            stego_files = list(stego_path.glob('*.png'))
            
            logger.info(f"✅ Dataset found: {len(cover_files)} cover, {len(stego_files)} stego images")
            
            if len(cover_files) < 1000 or len(stego_files) < 1000:
                issues.append(f"⚠️  Small dataset: {len(cover_files)}+{len(stego_files)} images")
        else:
            issues.append("❌ Dataset directories not found")
    else:
        issues.append("❌ Dataset path not found")
    
    # Check existing dictionary
    dict_path = Path('outputs/20251008_102842/dictionary.npz')
    if dict_path.exists():
        try:
            import numpy as np
            dict_data = np.load(dict_path)
            dictionary = dict_data['dictionary']
            logger.info(f"✅ Existing dictionary found: {dictionary.shape}")
        except Exception as e:
            logger.info(f"⚠️  Dictionary file exists but can't load: {e}")
    else:
        logger.info("ℹ️  No existing dictionary - will train new one")
    
    # Check disk space
    import shutil
    total, used, free = shutil.disk_usage('.')
    free_gb = free // (1024**3)
    
    if free_gb < 5:
        issues.append(f"⚠️  Low disk space: {free_gb}GB free")
    else:
        logger.info(f"✅ Disk space: {free_gb}GB free")
    
    # Check memory (approximate)
    try:
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.available // (1024**3)
        
        if available_gb < 4:
            issues.append(f"⚠️  Low memory: {available_gb}GB available")
        else:
            logger.info(f"✅ Memory: {available_gb}GB available")
    except ImportError:
        logger.info("ℹ️  Can't check memory (psutil not available)")
    
    # Check output directory
    output_base = Path('outputs')
    if not output_base.exists():
        output_base.mkdir(exist_ok=True)
        logger.info("✅ Output directory created")
    else:
        logger.info("✅ Output directory exists")
    
    # Summary
    logger.info("="*50)
    if issues:
        logger.info("⚠️  VALIDATION ISSUES FOUND:")
        for issue in issues:
            logger.info(f"   {issue}")
        logger.info("")
        logger.info("🔧 Please resolve issues before training")
        return False
    else:
        logger.info("✅ ALL CHECKS PASSED!")
        logger.info("🚀 READY FOR OPTIMAL TRAINING!")
        logger.info("")
        logger.info("📋 RECOMMENDED COMMAND:")
        logger.info("   python scripts/train_single_ultimate.py")
        logger.info("")
        logger.info("⏰ Expected time: 6-8 hours")
        logger.info("🎯 Target accuracy: 80%+")
        return True

def estimate_training_time():
    """Estimate training time based on system."""
    logger.info("⏰ TRAINING TIME ESTIMATION")
    logger.info("="*50)
    
    # Try to get CPU info
    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=True)
        logger.info(f"💻 CPU cores: {cpu_count}")
        
        # Rough estimation based on cores
        if cpu_count >= 8:
            time_estimate = "4-6 hours (Fast system)"
        elif cpu_count >= 4:
            time_estimate = "6-8 hours (Good system)"
        else:
            time_estimate = "8-12 hours (Slower system)"
            
        logger.info(f"⏰ Estimated time: {time_estimate}")
        
    except ImportError:
        logger.info("⏰ Estimated time: 6-8 hours (typical)")
    
    logger.info("")
    logger.info("📊 Time breakdown:")
    logger.info("   • Data loading: 10-30 minutes")
    logger.info("   • Feature extraction: 30-60 minutes") 
    logger.info("   • Hyperparameter search: 4-6 hours ⏳")
    logger.info("   • Final evaluation: 10-30 minutes")
    logger.info("")
    logger.info("💡 Most time spent on grid search (worth it for 80%+ accuracy!)")

def main():
    logger.info("🔍 SRM-SVM OPTIMAL TRAINING VALIDATION")
    logger.info("="*60)
    
    ready = validate_environment()
    logger.info("")
    estimate_training_time()
    logger.info("")
    
    if ready:
        logger.info("🎉 SYSTEM READY FOR OPTIMAL TRAINING!")
        logger.info("="*60)
        logger.info("")
        logger.info("🚀 TO START TRAINING NOW:")
        logger.info("   python scripts/train_single_ultimate.py")
        logger.info("")
        logger.info("📚 FOR DETAILED GUIDE:")
        logger.info("   Read TRAINING_GUIDE.md")
        logger.info("")
        logger.info("Good luck achieving 80%+ accuracy! 🎯✨")
    else:
        logger.info("🔧 PLEASE FIX ISSUES BEFORE TRAINING")
        logger.info("="*60)

if __name__ == "__main__":
    main()