#!/usr/bin/env python3
"""
MINIMAL ultra-fast training with existing dictionary.
No complex dependencies - direct implementation.
"""

import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def main():
    logger.info("🚀 Starting MINIMAL ULTRA-FAST SRM-SVM training")
    
    try:
        # Check if existing dictionary exists
        dict_path = 'outputs/20251008_102842/dictionary.npz'
        if not os.path.exists(dict_path):
            logger.error(f"Dictionary not found at {dict_path}")
            return
        
        # Load dictionary
        logger.info("📚 Loading existing dictionary...")
        dict_data = np.load(dict_path)
        dictionary = dict_data['dictionary']
        logger.info(f"✅ Dictionary loaded: {dictionary.shape}")
        
        # Import after path setup
        from steganalysis.data.dataset import SteganalysisDataset, load_image_as_grayscale, apply_srm_residual
        from steganalysis.features.patches import extract_patches_from_images, patches_to_vectors
        from steganalysis.features.sparse_coding import SparseDictionary
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/{timestamp}_minimal"
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset
        logger.info("📂 Loading dataset...")
        dataset = SteganalysisDataset(
            data_root='dataset/BOSSBase 1.01 + 0.4 WOW',
            cover_dir='cover',
            stego_dir='stego'
        )
        
        # Split dataset
        logger.info("🔀 Splitting dataset...")
        (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = dataset.split_dataset(
            val_size=0.15,
            test_size=0.15,
            random_state=42
        )
        
        # Use small subset for ultra-fast training
        n_train = 500  # Super small for speed
        n_test = 100
        
        logger.info(f"🎯 Using {n_train} training, {n_test} test samples")
        
        # Load training data
        logger.info("🖼️ Loading training images...")
        train_images = []
        actual_train_labels = train_labels[:n_train]
        
        for i, path in enumerate(train_paths[:n_train]):
            if i % 50 == 0:
                logger.info(f"   Loading {i}/{n_train}...")
            
            image = load_image_as_grayscale(path)
            if image is not None:
                image = apply_srm_residual(image, 'second_order')  # Apply SRM residual
                train_images.append(image)
        
        # Extract patches (already returns vectors)
        logger.info("✂️ Extracting training patches...")
        train_vectors = extract_patches_from_images(
            train_images,
            patch_size=12,
            stride=8,  
            max_patches_per_image=500  # Small for speed  
        )
        
        logger.info(f"✅ Training patch vectors: {train_vectors.shape}")
        
        # Setup sparse coder with existing dictionary
        logger.info("🧮 Setting up sparse coder...")
        sparse_coder = SparseDictionary(
            n_components=64,
            alpha=0.001,
            max_iter=50,
            random_state=42
        )
        sparse_coder.dictionary_ = dictionary
        sparse_coder.is_fitted = True
        
        # Transform to sparse features
        logger.info("🔢 Transforming to sparse features...")
        train_sparse_codes = sparse_coder.transform(
            train_vectors,
            solver='omp',
            n_nonzero_coefs=6
        )
        logger.info(f"✅ Training sparse codes: {train_sparse_codes.shape}")
        
        # Aggregate patch features to image features (mean pooling)
        logger.info("📊 Aggregating patch features to image features...")
        patches_per_image = len(train_sparse_codes) // len(train_images)
        logger.info(f"   Patches per image: {patches_per_image}")
        
        train_features = []
        for i in range(len(train_images)):
            start_idx = i * patches_per_image
            end_idx = min((i + 1) * patches_per_image, len(train_sparse_codes))
            if start_idx < len(train_sparse_codes):
                # Mean pooling of patches for this image
                image_features = np.mean(train_sparse_codes[start_idx:end_idx], axis=0)
                train_features.append(image_features)
        
        train_features = np.array(train_features)
        logger.info(f"✅ Aggregated train features: {train_features.shape}")
        
        # Load test data  
        logger.info("🖼️ Loading test images...")
        test_images = []
        actual_test_labels = test_labels[:n_test]
        
        for i, path in enumerate(test_paths[:n_test]):
            image = load_image_as_grayscale(path)
            if image is not None:
                image = apply_srm_residual(image, 'second_order')
                test_images.append(image)
        
        # Extract test patches (already returns vectors)
        logger.info("✂️ Extracting test patches...")
        test_vectors = extract_patches_from_images(
            test_images,
            patch_size=12,
            stride=8,
            max_patches_per_image=500
        )
        
        test_sparse_codes = sparse_coder.transform(
            test_vectors,
            solver='omp',
            n_nonzero_coefs=6
        )
        logger.info(f"✅ Test sparse codes: {test_sparse_codes.shape}")
        
        # Aggregate patch features to image features (mean pooling)
        test_features = []
        for i in range(len(test_images)):
            start_idx = i * patches_per_image
            end_idx = min((i + 1) * patches_per_image, len(test_sparse_codes))
            if start_idx < len(test_sparse_codes):
                # Mean pooling of patches for this image
                image_features = np.mean(test_sparse_codes[start_idx:end_idx], axis=0)
                test_features.append(image_features)
        
        test_features = np.array(test_features)
        logger.info(f"✅ Aggregated test features: {test_features.shape}")
        
        # Train SVM (fixed parameters - no grid search)
        logger.info("🤖 Training SVM...")
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_features)
        test_scaled = scaler.transform(test_features)
        
        svm = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale', 
            probability=True,
            random_state=42
        )
        
        svm.fit(train_scaled, actual_train_labels)
        logger.info("✅ SVM training completed!")
        
        # Evaluate  
        logger.info("📊 Evaluating...")
        test_pred = svm.predict(test_scaled)
        test_proba = svm.predict_proba(test_scaled)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(actual_test_labels, test_pred),
            'precision': precision_score(actual_test_labels, test_pred, zero_division=0),
            'recall': recall_score(actual_test_labels, test_pred, zero_division=0),
            'f1': f1_score(actual_test_labels, test_pred, zero_division=0),
            'roc_auc': roc_auc_score(actual_test_labels, test_proba)
        }
        
        # Save results
        import json
        import joblib
        
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        joblib.dump({
            'svm': svm,
            'scaler': scaler,
            'sparse_coder': sparse_coder,
            'dictionary': dictionary
        }, os.path.join(output_dir, 'model.pkl'))
        
        # Display results
        logger.info("="*70)
        logger.info("🎉 MINIMAL ULTRA-FAST TRAINING COMPLETED!")
        logger.info("="*70)
        logger.info(f"📈 Training samples: {n_train}")
        logger.info(f"🧪 Test samples: {n_test}")
        logger.info(f"📚 Dictionary atoms: {dictionary.shape[0]}")
        logger.info(f"🔢 Feature dimension: {train_features.shape[1]}")
        logger.info(f"💾 Saved to: {output_dir}")
        logger.info("-"*70)
        logger.info(f"🎯 Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"🎯 Precision: {metrics['precision']:.4f}")
        logger.info(f"🎯 Recall:    {metrics['recall']:.4f}")
        logger.info(f"🎯 F1-Score:  {metrics['f1']:.4f}")
        logger.info(f"🎯 ROC-AUC:   {metrics['roc_auc']:.4f}")
        logger.info("="*70)
        
        # Comparison
        logger.info("📊 COMPARISON WITH PREVIOUS RESULTS:")
        logger.info("❌ Training 1 (Failed): Acc=0.5000, F1=0.0000, AUC=0.5000")
        logger.info(f"✅ Current (Minimal):   Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, AUC={metrics['roc_auc']:.4f}")
        
        if metrics['accuracy'] > 0.5:
            improvement = (metrics['accuracy'] - 0.5) / 0.5 * 100
            logger.info(f"🚀 IMPROVEMENT: {improvement:.1f}% better accuracy!")
        else:
            logger.info("⚠️  Need parameter tuning for better results")
        
        logger.info("✨ Training completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()