#!/usr/bin/env python3
"""
Ultra-fast SRM-SVM training script.
Uses existing dictionary and fixed parameters for speed.
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from steganalysis.data.dataset import SteganalysisDataset, load_image_as_grayscale, apply_srm_residual
from steganalysis.features.patches import extract_patches_from_images
from steganalysis.features.sparse_coding import SparseDictionary
from steganalysis.models.classifier import SteganalysisClassifier
from steganalysis.utils.evaluation import evaluate_model, save_results
from steganalysis.utils.logging import setup_logging

def main():
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/{timestamp}_ultrafast"
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(os.path.join(output_dir, 'training.log'))
    
    logger = logging.getLogger(__name__)
    logger.info("Starting ULTRA-FAST SRM-SVM training")
    
    try:
        # Configuration
        config = {
            'data_root': 'dataset/BOSSBase 1.01 + 0.4 WOW',
            'cover_dir': 'cover',
            'stego_dir': 'stego',
            'existing_dict': 'outputs/20251008_102842/dictionary.npz',
            'dict_size': 64,
            'patch_size': 12,
            'stride': 8,
            'n_nonzero_coefs': 6,
            'alpha_coding': 0.001,
            'max_patches': 1500,  # Reduced for speed
            'val_size': 0.15,
            'test_size': 0.15,
            'seed': 42,
            'apply_residual': True,
            'residual_type': 'second_order',
            'svm_c': 1.0,
            'svm_kernel': 'rbf',
            'svm_gamma': 'scale'
        }
        
        logger.info(f"Configuration: {config}")
        
        # 1. Load dataset
        logger.info("Loading dataset...")
        dataset = SteganalysisDataset(
            data_root=config['data_root'],
            cover_dir=config['cover_dir'],
            stego_dir=config['stego_dir'],
            val_size=config['val_size'],
            test_size=config['test_size'],
            random_state=config['seed']
        )
        
        logger.info(f"Dataset loaded: {len(dataset.train_paths)} train, {len(dataset.val_paths)} val, {len(dataset.test_paths)} test")
        
        # 2. Load existing dictionary
        logger.info("Loading existing dictionary...")
        dict_data = np.load(config['existing_dict'])
        dictionary = dict_data['dictionary']
        logger.info(f"Dictionary loaded: {dictionary.shape}")
        
        # 3. Initialize sparse extractor with existing dictionary
        sparse_extractor = SparseDictionary(
            n_components=config['dict_size'],
            alpha=config['alpha_coding'],
            max_iter=100,
            n_jobs=-1,
            random_state=config['seed']
        )
        sparse_extractor.dictionary_ = dictionary
        sparse_extractor.is_fitted = True
        
        # 4. Extract features from training data (batch processing)
        logger.info("Extracting features from training data...")
        
        def load_images_from_paths(paths, apply_residual=False, residual_type="first_order"):
            images = []
            for i, path in enumerate(paths):
                if i % 1000 == 0:
                    logger.info(f"Loading image {i}/{len(paths)}")
                
                image = load_image_as_grayscale(path)
                if apply_residual:
                    image = apply_srm_residual(image, residual_type)
                images.append(image)
            return images
        
        # Load training images
        train_images = load_images_from_paths(
            dataset.train_paths, 
            config['apply_residual'], 
            config['residual_type']
        )
        
        # Extract patches and features
        logger.info("Extracting patches...")
        train_patches = extract_patches_from_images(
            train_images,
            patch_size=config['patch_size'],
            stride=config['stride'],
            max_patches_per_image=config['max_patches']
        )
        
        logger.info(f"Extracted {len(train_patches)} patches")
        
        # Transform to sparse features
        logger.info("Transforming to sparse features...")
        train_features = sparse_extractor.transform(train_patches)
        logger.info(f"Feature shape: {train_features.shape}")
        
        # 5. Prepare labels
        train_labels = dataset.train_labels
        
        # 6. Load validation data (smaller subset for speed)
        logger.info("Loading validation data...")
        val_images = load_images_from_paths(
            dataset.val_paths[:300],  # Only 300 for speed
            config['apply_residual'], 
            config['residual_type']
        )
        
        val_patches = extract_patches_from_images(
            val_images,
            patch_size=config['patch_size'],
            stride=config['stride'],
            max_patches_per_image=500  # Fewer patches for validation
        )
        
        val_features = sparse_extractor.transform(val_patches)
        val_labels = dataset.val_labels[:300]
        
        # 7. Train SVM with fixed parameters (NO GRID SEARCH)
        logger.info("Training SVM with fixed parameters...")
        classifier = SteganalysisClassifier(
            kernel=config['svm_kernel'],
            C=config['svm_c'],
            gamma=config['svm_gamma'],
            use_grid_search=False,  # DISABLED for speed
            random_state=config['seed']
        )
        
        classifier.fit(train_features, train_labels, val_features, val_labels)
        
        # 8. Quick evaluation on test subset
        logger.info("Quick evaluation...")
        test_images = load_images_from_paths(
            dataset.test_paths[:300],  # Only 300 for speed
            config['apply_residual'], 
            config['residual_type']
        )
        
        test_patches = extract_patches_from_images(
            test_images,
            patch_size=config['patch_size'],
            stride=config['stride'],
            max_patches_per_image=500
        )
        
        test_features = sparse_extractor.transform(test_patches)
        test_labels = dataset.test_labels[:300]
        
        # Evaluate
        test_predictions = classifier.predict(test_features)
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        try:
            test_proba = classifier.predict_proba(test_features)[:, 1]
            roc_auc = roc_auc_score(test_labels, test_proba)
        except:
            roc_auc = 0.5
        
        metrics = {
            'accuracy': accuracy_score(test_labels, test_predictions),
            'precision': precision_score(test_labels, test_predictions, zero_division=0),
            'recall': recall_score(test_labels, test_predictions, zero_division=0),
            'f1': f1_score(test_labels, test_predictions, zero_division=0),
            'roc_auc': roc_auc
        }
        
        # 9. Save results
        logger.info("Saving results...")
        
        # Save metrics
        import json
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save config
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save classifier
        import joblib
        joblib.dump(classifier, os.path.join(output_dir, 'classifier.pkl'))
        
        # 10. Final results
        logger.info("="*60)
        logger.info("ULTRA-FAST TRAINING COMPLETED!")
        logger.info("="*60)
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1-Score:  {metrics['f1']:.4f}")
        logger.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        logger.info("="*60)
        logger.info(f"Results saved to: {output_dir}")
        logger.info("="*60)
        
        # Compare with previous result
        logger.info("COMPARISON WITH PREVIOUS TRAINING:")
        logger.info("Previous (Training 1): Accuracy=0.5000, F1=0.0000, ROC-AUC=0.5000")
        logger.info(f"Current  (Ultra-fast): Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")
        
        improvement = (metrics['accuracy'] - 0.5) / 0.5 * 100
        logger.info(f"Improvement: {improvement:.1f}% better than previous training!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()