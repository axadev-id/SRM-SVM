#!/usr/bin/env python3
"""
Improved training script with better parameters for SRM-SVM steganalysis.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from steganalysis.data.loader import load_boss_dataset
from steganalysis.features.patches import extract_patches_batch
from steganalysis.features.sparse_coding import SparseFeatureExtractor
from steganalysis.models.classifier import SteganalysisClassifier
from steganalysis.utils.evaluation import evaluate_model, save_results
from steganalysis.utils.logging import setup_logging

def main():
    parser = argparse.ArgumentParser(description='Improved SRM-SVM Steganalysis Training')
    
    # Data parameters
    parser.add_argument('--data-root', required=True, help='Root directory of dataset')
    parser.add_argument('--cover-dir', default='cover', help='Cover images directory')
    parser.add_argument('--stego-dir', default='stego', help='Stego images directory')
    
    # Improved feature extraction parameters
    parser.add_argument('--dict-size', type=int, default=128, help='Dictionary size (increased)')
    parser.add_argument('--patch-size', type=int, default=12, help='Patch size (increased)')
    parser.add_argument('--stride', type=int, default=8, help='Stride for patch extraction (reduced)')
    parser.add_argument('--sparse-solver', default='omp', choices=['omp', 'lars'], help='Sparse solver')
    parser.add_argument('--n-nonzero-coefs', type=int, default=8, help='Number of non-zero coefficients (increased)')
    parser.add_argument('--alpha-coding', type=float, default=0.001, help='Alpha for coding (reduced)')
    parser.add_argument('--max-patches', type=int, default=10000, help='Maximum patches per image (increased)')
    
    # Improved dataset split
    parser.add_argument('--val-size', type=float, default=0.15, help='Validation set size (increased)')
    parser.add_argument('--test-size', type=float, default=0.15, help='Test set size (increased)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # SRM parameters
    parser.add_argument('--apply-residual', action='store_true', default=True, help='Apply SRM residual (enabled)')
    parser.add_argument('--residual-type', default='spam686', choices=['first_order', 'spam686'], help='SRM residual type (improved)')
    
    # Output
    parser.add_argument('--output-dir', help='Output directory (auto-generated if not specified)')
    
    args = parser.parse_args()
    
    # Setup logging and output directory
    if args.output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"outputs/{timestamp}_improved"
    
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(os.path.join(args.output_dir, 'training.log'))
    
    logger = logging.getLogger(__name__)
    logger.info("Starting improved SRM-SVM steganalysis training")
    logger.info(f"Configuration: {vars(args)}")
    
    try:
        # Load dataset
        logger.info("Loading dataset...")
        images_cover, images_stego, labels = load_boss_dataset(
            args.data_root, args.cover_dir, args.stego_dir
        )
        logger.info(f"Loaded {len(images_cover)} cover and {len(images_stego)} stego images")
        
        # Extract patches with improved parameters
        logger.info("Extracting patches...")
        patches_cover = extract_patches_batch(
            images_cover, 
            patch_size=args.patch_size,
            stride=args.stride,
            max_patches=args.max_patches,
            apply_residual=args.apply_residual,
            residual_type=args.residual_type
        )
        
        patches_stego = extract_patches_batch(
            images_stego,
            patch_size=args.patch_size, 
            stride=args.stride,
            max_patches=args.max_patches,
            apply_residual=args.apply_residual,
            residual_type=args.residual_type
        )
        
        logger.info(f"Extracted {len(patches_cover)} cover and {len(patches_stego)} stego patches")
        
        # Sparse coding with improved parameters
        logger.info("Performing sparse coding...")
        sparse_extractor = SparseFeatureExtractor(
            dict_size=args.dict_size,
            alpha_dict=0.1,
            alpha_coding=args.alpha_coding,
            sparse_solver=args.sparse_solver,
            n_nonzero_coefs=args.n_nonzero_coefs,
            n_jobs=-1
        )
        
        # Fit dictionary and transform
        all_patches = patches_cover + patches_stego
        features = sparse_extractor.fit_transform(all_patches)
        logger.info(f"Generated features shape: {features.shape}")
        
        # Train classifier with improved parameters
        logger.info("Training classifier...")
        classifier = SteganalysisClassifier(
            C=10.0,  # Increased regularization
            kernel='rbf',  # Changed to RBF kernel
            gamma='scale',
            val_size=args.val_size,
            test_size=args.test_size,
            random_state=args.seed,
            n_jobs=-1
        )
        
        classifier.fit(features, labels)
        logger.info("Training completed")
        
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
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()