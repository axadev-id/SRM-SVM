#!/usr/bin/env python3
"""
Fast SRM-SVM steganalysis training using existing dictionary.
Optimized for memory efficiency and speed.
"""

import os
import sys
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from steganalysis.data.loader import load_boss_dataset
from steganalysis.features.patches import extract_patches_batch
from steganalysis.features.sparse_coding import SparseFeatureExtractor
from steganalysis.models.classifier import SteganalysisClassifier
from steganalysis.utils.evaluation import evaluate_model, save_results
from steganalysis.utils.logging import setup_logging

def load_existing_dictionary(dict_path):
    """Load pre-trained dictionary."""
    try:
        data = np.load(dict_path)
        dictionary = data['dictionary']
        logging.info(f"Loaded existing dictionary: {dictionary.shape}")
        return dictionary
    except Exception as e:
        logging.error(f"Failed to load dictionary from {dict_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Fast SRM-SVM Steganalysis Training')
    
    # Data parameters
    parser.add_argument('--data-root', required=True, help='Root directory of dataset')
    parser.add_argument('--cover-dir', default='cover', help='Cover images directory')
    parser.add_argument('--stego-dir', default='stego', help='Stego images directory')
    parser.add_argument('--existing-dict', help='Path to existing dictionary')
    
    # Memory-efficient parameters
    parser.add_argument('--dict-size', type=int, default=64, help='Dictionary size')
    parser.add_argument('--patch-size', type=int, default=12, help='Patch size')
    parser.add_argument('--stride', type=int, default=8, help='Stride for patch extraction')
    parser.add_argument('--sparse-solver', default='omp', choices=['omp', 'lars'], help='Sparse solver')
    parser.add_argument('--n-nonzero-coefs', type=int, default=6, help='Number of non-zero coefficients')
    parser.add_argument('--alpha-coding', type=float, default=0.001, help='Alpha for coding')
    parser.add_argument('--max-patches', type=int, default=2000, help='Max patches per image (reduced for speed)')
    
    # Efficient dataset split
    parser.add_argument('--val-size', type=float, default=0.15, help='Validation set size')
    parser.add_argument('--test-size', type=float, default=0.15, help='Test set size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # SRM parameters
    parser.add_argument('--apply-residual', action='store_true', default=True, help='Apply SRM residual')
    parser.add_argument('--residual-type', default='second_order', choices=['first_order', 'second_order'], help='SRM residual type')
    
    # Fixed SVM parameters (no grid search)
    parser.add_argument('--svm-c', type=float, default=1.0, help='SVM C parameter')
    parser.add_argument('--svm-kernel', default='rbf', choices=['linear', 'rbf'], help='SVM kernel')
    parser.add_argument('--svm-gamma', default='scale', help='SVM gamma parameter')
    
    # Output
    parser.add_argument('--output-dir', help='Output directory (auto-generated if not specified)')
    
    args = parser.parse_args()
    
    # Setup logging and output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"outputs/{timestamp}_fast"
    
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(os.path.join(args.output_dir, 'training.log'))
    
    logger = logging.getLogger(__name__)
    logger.info("Starting FAST SRM-SVM steganalysis training")
    logger.info(f"Configuration: {vars(args)}")
    
    try:
        # Load dataset with memory optimization
        logger.info("Loading dataset...")
        images_cover, images_stego, labels = load_boss_dataset(
            args.data_root, args.cover_dir, args.stego_dir
        )
        logger.info(f"Loaded {len(images_cover)} cover and {len(images_stego)} stego images")
        
        # Initialize sparse extractor
        sparse_extractor = SparseFeatureExtractor(
            dict_size=args.dict_size,
            alpha_dict=0.1,
            alpha_coding=args.alpha_coding,
            sparse_solver=args.sparse_solver,
            n_nonzero_coefs=args.n_nonzero_coefs,
            n_jobs=-1
        )
        
        # Try to load existing dictionary
        if args.existing_dict and os.path.exists(args.existing_dict):
            logger.info("Loading existing dictionary...")
            dictionary = load_existing_dictionary(args.existing_dict)
            if dictionary is not None:
                sparse_extractor.dictionary_ = dictionary
                sparse_extractor.is_fitted_ = True
                logger.info("Successfully loaded existing dictionary!")
            else:
                logger.warning("Failed to load existing dictionary, will create new one")
        
        # If no dictionary loaded, create minimal one
        if not hasattr(sparse_extractor, 'is_fitted_') or not sparse_extractor.is_fitted_:
            logger.info("Creating minimal dictionary for fast training...")
            # Extract minimal patches for dictionary
            sample_patches = []
            for i in range(min(50, len(images_cover))):  # Only 50 images for dict
                patches = extract_patches_batch(
                    [images_cover[i]], 
                    patch_size=args.patch_size,
                    stride=args.stride,
                    max_patches=100,  # Very few patches
                    apply_residual=args.apply_residual,
                    residual_type=args.residual_type
                )
                if len(patches) > 0:
                    sample_patches.extend(patches[:50])  # Max 50 patches per image
            
            if len(sample_patches) > 0:
                logger.info(f"Learning dictionary from {len(sample_patches)} patches...")
                sparse_extractor.fit(sample_patches)
                
                # Save dictionary
                dict_path = os.path.join(args.output_dir, 'dictionary_fast.npz')
                np.savez(dict_path, dictionary=sparse_extractor.dictionary_)
                logger.info(f"Dictionary saved to {dict_path}")
        
        # Extract features efficiently (batch processing)
        logger.info("Extracting features from all images...")
        all_images = images_cover + images_stego
        
        # Process in smaller batches to save memory
        batch_size = 500  # Process 500 images at a time
        all_features = []
        
        for i in range(0, len(all_images), batch_size):
            batch_images = all_images[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(all_images) + batch_size - 1)//batch_size}")
            
            # Extract patches
            batch_patches = extract_patches_batch(
                batch_images,
                patch_size=args.patch_size,
                stride=args.stride,
                max_patches=args.max_patches,
                apply_residual=args.apply_residual,
                residual_type=args.residual_type
            )
            
            if len(batch_patches) > 0:
                # Transform to sparse features
                batch_features = sparse_extractor.transform(batch_patches)
                all_features.append(batch_features)
                
                # Clear memory
                del batch_patches
        
        if len(all_features) > 0:
            features = np.vstack(all_features)
            logger.info(f"Generated features shape: {features.shape}")
        else:
            raise ValueError("No features extracted!")
        
        # Train classifier with fixed parameters (no grid search)
        logger.info("Training SVM classifier with fixed parameters...")
        classifier = SteganalysisClassifier(
            C=args.svm_c,
            kernel=args.svm_kernel, 
            gamma=args.svm_gamma,
            val_size=args.val_size,
            test_size=args.test_size,
            random_state=args.seed,
            n_jobs=-1,
            use_grid_search=False  # Disable grid search for speed
        )
        
        classifier.fit(features, labels)
        logger.info("Training completed!")
        
        # Evaluate
        logger.info("Evaluating model...")
        metrics = evaluate_model(classifier, features, labels)
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test F1-Score: {metrics['f1']:.4f}")
        logger.info(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
        
        # Save results
        logger.info("Saving results...")
        save_results(
            classifier, features, labels, metrics, 
            vars(args), args.output_dir
        )
        logger.info(f"Results saved to: {args.output_dir}")
        
        logger.info("Fast training completed successfully!")
        logger.info("="*50)
        logger.info("FINAL RESULTS:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-Score: {metrics['f1']:.4f}")
        logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()