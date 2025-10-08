#!/usr/bin/env python3
"""
Simple ultra-fast training using existing outputs.
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
from steganalysis.features.patches import extract_patches_from_images, patches_to_vectors
from steganalysis.features.sparse_coding import SparseDictionary
from steganalysis.utils.logging import setup_logging
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def main():
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/{timestamp}_simple"
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(os.path.join(output_dir, 'training.log'))
    
    logger = logging.getLogger(__name__)
    logger.info("Starting SIMPLE FAST SRM-SVM training")
    
    try:
        # Configuration
        data_root = 'dataset/BOSSBase 1.01 + 0.4 WOW'
        existing_dict = 'outputs/20251008_102842/dictionary.npz'
        
        logger.info("Loading existing dictionary...")
        dict_data = np.load(existing_dict)
        dictionary = dict_data['dictionary']
        logger.info(f"Dictionary loaded: {dictionary.shape}")
        
        # Load small subset of data for fast training
        logger.info("Loading dataset...")
        dataset = SteganalysisDataset(
            data_root=data_root,
            cover_dir='cover',
            stego_dir='stego',
            val_size=0.15,
            test_size=0.15,
            random_state=42
        )
        
        # Use only subset for ultra-fast training
        n_train = 1000  # Very small for speed
        n_test = 200
        
        logger.info(f"Using {n_train} training and {n_test} test samples")
        
        # Load training images
        train_paths = dataset.train_paths[:n_train]
        train_labels = dataset.train_labels[:n_train]
        
        train_images = []
        for i, path in enumerate(train_paths):
            if i % 100 == 0:
                logger.info(f"Loading training image {i}/{len(train_paths)}")
            
            image = load_image_as_grayscale(path)
            image = apply_srm_residual(image, 'second_order')  # Apply SRM
            train_images.append(image)
        
        # Extract patches
        logger.info("Extracting training patches...")
        train_patches = extract_patches_from_images(
            train_images,
            patch_size=12,
            stride=8,
            max_patches_per_image=800  # Reduced for speed
        )
        
        train_patch_vectors = patches_to_vectors(train_patches)
        logger.info(f"Training patches shape: {train_patch_vectors.shape}")
        
        # Initialize sparse coder
        sparse_coder = SparseDictionary(
            n_components=64,
            alpha=0.001,
            max_iter=50,
            random_state=42
        )
        
        # Set the existing dictionary
        sparse_coder.dictionary_ = dictionary
        sparse_coder.is_fitted = True
        
        # Transform to sparse features
        logger.info("Transforming to sparse features...")
        train_features = sparse_coder.transform(
            train_patch_vectors,
            solver='omp',
            n_nonzero_coefs=6,
            alpha_coding=0.001
        )
        
        logger.info(f"Training features shape: {train_features.shape}")
        
        # Load test data
        test_paths = dataset.test_paths[:n_test]
        test_labels = dataset.test_labels[:n_test]
        
        test_images = []
        for i, path in enumerate(test_paths):
            image = load_image_as_grayscale(path)
            image = apply_srm_residual(image, 'second_order')
            test_images.append(image)
        
        logger.info("Extracting test patches...")
        test_patches = extract_patches_from_images(
            test_images,
            patch_size=12,
            stride=8,
            max_patches_per_image=800
        )
        
        test_patch_vectors = patches_to_vectors(test_patches)
        test_features = sparse_coder.transform(
            test_patch_vectors,
            solver='omp',
            n_nonzero_coefs=6,
            alpha_coding=0.001
        )
        
        logger.info(f"Test features shape: {test_features.shape}")
        
        # Train SVM with fixed parameters
        logger.info("Training SVM...")
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        test_features_scaled = scaler.transform(test_features)
        
        svm = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        )
        
        svm.fit(train_features_scaled, train_labels)
        
        # Evaluate
        logger.info("Evaluating...")
        test_predictions = svm.predict(test_features_scaled)
        test_proba = svm.predict_proba(test_features_scaled)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(test_labels, test_predictions),
            'precision': precision_score(test_labels, test_predictions, zero_division=0),
            'recall': recall_score(test_labels, test_predictions, zero_division=0),
            'f1': f1_score(test_labels, test_predictions, zero_division=0),
            'roc_auc': roc_auc_score(test_labels, test_proba)
        }
        
        # Save results
        import json
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        import joblib
        joblib.dump({'svm': svm, 'scaler': scaler, 'sparse_coder': sparse_coder}, 
                   os.path.join(output_dir, 'model.pkl'))
        
        # Results
        logger.info("="*60)
        logger.info("SIMPLE FAST TRAINING COMPLETED!")
        logger.info("="*60)
        logger.info(f"Training samples: {n_train}")
        logger.info(f"Test samples: {n_test}")
        logger.info(f"Dictionary atoms: {dictionary.shape[0]}")
        logger.info(f"Feature dimension: {train_features.shape[1]}")
        logger.info("-"*60)
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1-Score:  {metrics['f1']:.4f}")
        logger.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        logger.info("="*60)
        
        # Compare with previous training
        logger.info("COMPARISON:")
        logger.info("Training 1 (Failed): Accuracy=0.5000, F1=0.0000, ROC-AUC=0.5000")
        logger.info(f"Current (Fast):       Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")
        
        if metrics['accuracy'] > 0.5:
            improvement = (metrics['accuracy'] - 0.5) / 0.5 * 100
            logger.info(f"🎉 IMPROVEMENT: {improvement:.1f}% better accuracy!")
        
        logger.info(f"Results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()