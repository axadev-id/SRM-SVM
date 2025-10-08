#!/usr/bin/env python3
"""
SINGLE ULTIMATE OPTIMAL TRAINING - TARGET: 80%+ ACCURACY
The most powerful single training script with all optimizations.
Perfect for running overnight for maximum accuracy.
"""

import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ultimate_training.log')
    ]
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def create_enhanced_features(sparse_features: np.ndarray, method: str = 'comprehensive') -> np.ndarray:
    """Create enhanced features for better classification."""
    logger.info(f"🧬 Creating enhanced features using {method} method...")
    
    if method == 'comprehensive':
        # Multiple feature transformations
        features = [sparse_features]
        
        # Statistical transformations
        features.append(np.power(sparse_features, 2))  # Squared
        features.append(np.sqrt(np.abs(sparse_features) + 1e-8))  # Square root
        features.append(np.tanh(sparse_features))  # Tanh transformation
        
        # Statistical moments per sample
        mean_feat = np.mean(sparse_features, axis=1, keepdims=True)
        std_feat = np.std(sparse_features, axis=1, keepdims=True) + 1e-8
        
        # Normalized features
        normalized = (sparse_features - mean_feat) / std_feat
        features.append(normalized)
        
        combined = np.hstack(features)
        
    elif method == 'statistical':
        # Focus on statistical properties
        features = [sparse_features]
        
        # Add statistical moments
        mean_feat = np.mean(sparse_features, axis=1, keepdims=True)
        std_feat = np.std(sparse_features, axis=1, keepdims=True)
        skew_feat = np.mean((sparse_features - mean_feat)**3, axis=1, keepdims=True) / (std_feat**3 + 1e-8)
        
        features.extend([mean_feat, std_feat, skew_feat])
        combined = np.hstack(features)
        
    else:  # 'basic'
        combined = sparse_features
    
    logger.info(f"✅ Enhanced features: {sparse_features.shape} -> {combined.shape}")
    return combined

def advanced_pooling(sparse_codes: np.ndarray, n_images: int, method: str = 'multi') -> np.ndarray:
    """Advanced pooling strategies for patch aggregation."""
    logger.info(f"📊 Advanced pooling using {method} method...")
    
    patches_per_image = len(sparse_codes) // n_images
    image_features = []
    
    for i in range(n_images):
        start_idx = i * patches_per_image
        end_idx = min((i + 1) * patches_per_image, len(sparse_codes))
        
        if start_idx < len(sparse_codes):
            patches = sparse_codes[start_idx:end_idx]
            
            if method == 'multi':
                # Multiple pooling strategies
                mean_pool = np.mean(patches, axis=0)
                max_pool = np.max(patches, axis=0)
                min_pool = np.min(patches, axis=0)
                std_pool = np.std(patches, axis=0)
                
                # Percentile pooling
                p25_pool = np.percentile(patches, 25, axis=0)
                p75_pool = np.percentile(patches, 75, axis=0)
                
                # Combine all
                combined = np.concatenate([
                    mean_pool, max_pool, min_pool, std_pool,
                    p25_pool, p75_pool
                ])
                
            elif method == 'weighted':
                # Weighted pooling based on activation strength
                weights = np.sum(np.abs(patches), axis=1)
                weights = weights / (np.sum(weights) + 1e-8)
                combined = np.average(patches, axis=0, weights=weights)
                
            else:  # 'mean'
                combined = np.mean(patches, axis=0)
            
            image_features.append(combined)
    
    result = np.array(image_features)
    logger.info(f"✅ Pooled features: {result.shape}")
    return result

def train_advanced_svm(X_train: np.ndarray, y_train: np.ndarray, 
                      X_val: np.ndarray, y_val: np.ndarray) -> tuple:
    """Train advanced SVM with comprehensive hyperparameter search."""
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.metrics import accuracy_score
    from sklearn.pipeline import Pipeline
    
    logger.info("🤖 Training advanced SVM with comprehensive search...")
    logger.info("⏰ This will take time but ensures optimal results!")
    
    # Try different scalers
    scalers = [
        ('standard', StandardScaler()),
        ('robust', RobustScaler())
    ]
    
    best_score = 0.0
    best_model = None
    best_scaler = None
    best_params = None
    
    for scaler_name, scaler in scalers:
        logger.info(f"   Testing with {scaler_name} scaler...")
        
        # Scale data
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Comprehensive parameter grid
        param_grid = [
            # RBF kernel with comprehensive C and gamma
            {
                'kernel': ['rbf'],
                'C': [1, 10, 50, 100, 500, 1000, 2000, 5000],
                'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1.0]
            },
            # Polynomial kernel
            {
                'kernel': ['poly'],
                'C': [1, 10, 100, 1000],
                'degree': [2, 3, 4],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            },
            # Linear kernel
            {
                'kernel': ['linear'],
                'C': [0.1, 1, 10, 100, 1000, 5000]
            }
        ]
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Grid search
        grid_search = GridSearchCV(
            SVC(probability=True, random_state=42),
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        logger.info(f"   Running grid search with {scaler_name} scaler...")
        grid_search.fit(X_train_scaled, y_train)
        
        # Validate on validation set
        val_pred = grid_search.best_estimator_.predict(X_val_scaled)
        val_score = accuracy_score(y_val, val_pred)
        
        logger.info(f"   {scaler_name} scaler results:")
        logger.info(f"     CV score: {grid_search.best_score_:.4f}")
        logger.info(f"     Val score: {val_score:.4f}")
        logger.info(f"     Best params: {grid_search.best_params_}")
        
        if val_score > best_score:
            best_score = val_score
            best_model = grid_search.best_estimator_
            best_scaler = scaler
            best_params = grid_search.best_params_
            best_cv_score = grid_search.best_score_
    
    logger.info(f"🏆 Best validation score: {best_score:.4f}")
    logger.info(f"🎯 Best parameters: {best_params}")
    
    return best_model, best_scaler, best_params, best_cv_score

def main():
    logger.info("="*80)
    logger.info("🚀 ULTIMATE OPTIMAL SRM-SVM TRAINING")
    logger.info("="*80)
    logger.info("🎯 TARGET: 80%+ ACCURACY")
    logger.info("💪 MAXIMUM OPTIMIZATION ENABLED")
    logger.info("⏰ EXPECTED TIME: 4-8 HOURS")
    logger.info("🔥 PERFECT FOR OVERNIGHT TRAINING")
    logger.info("="*80)
    
    try:
        # Import modules
        from steganalysis.data.dataset import SteganalysisDataset, load_image_as_grayscale, apply_srm_residual
        from steganalysis.features.patches import extract_patches_from_images
        from steganalysis.features.sparse_coding import SparseDictionary
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
        
        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/{timestamp}_ultimate"
        os.makedirs(output_dir, exist_ok=True)
        
        # Configuration for maximum performance
        config = {
            'n_train': 8000,      # Large training set
            'n_val': 1200,        # Good validation set
            'n_test': 1200,       # Comprehensive test set
            'dict_size': 128,     # Large dictionary
            'patch_size': 12,     # Optimal patch size
            'stride': 6,          # Dense sampling
            'max_patches': 1500,  # Many patches per image
            'alpha': 0.0005,      # Sparse coding parameter
            'n_nonzero_coefs': 10, # Rich representation
            'solver': 'omp',      # OMP solver
            'residual_type': 'second_order',
            'pooling_method': 'multi',
            'feature_method': 'comprehensive'
        }
        
        logger.info("📋 ULTIMATE CONFIGURATION:")
        for key, value in config.items():
            logger.info(f"   {key}: {value}")
        logger.info("")
        
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
        
        logger.info(f"🎯 Using {config['n_train']} train, {config['n_val']} val, {config['n_test']} test samples")
        
        # Dictionary handling
        existing_dict_path = 'outputs/20251008_102842/dictionary.npz'
        
        if os.path.exists(existing_dict_path):
            logger.info("📚 Loading existing high-quality dictionary...")
            dict_data = np.load(existing_dict_path)
            base_dictionary = dict_data['dictionary']
            
            # Expand dictionary if needed
            if base_dictionary.shape[0] < config['dict_size']:
                logger.info("🔧 Expanding dictionary for better representation...")
                # Use existing dictionary as starting point and train more atoms
                sparse_coder = SparseDictionary(
                    n_components=config['dict_size'],
                    alpha=config['alpha'],
                    max_iter=150,
                    random_state=42
                )
                
                # Load some images for dictionary expansion
                dict_images = []
                for i, path in enumerate(train_paths[:1500]):
                    if i % 300 == 0:
                        logger.info(f"   Loading dict images: {i}/1500")
                    image = load_image_as_grayscale(path)
                    if image is not None:
                        image = apply_srm_residual(image, config['residual_type'])
                        dict_images.append(image)
                
                dict_vectors = extract_patches_from_images(
                    dict_images,
                    patch_size=config['patch_size'],
                    stride=config['stride'],
                    max_patches_per_image=1000
                )
                
                sparse_coder.fit(dict_vectors)
                dictionary = sparse_coder.dictionary_
                logger.info(f"✅ Expanded dictionary: {dictionary.shape}")
            else:
                # Use subset of existing dictionary
                dictionary = base_dictionary[:config['dict_size']]
                sparse_coder = SparseDictionary(
                    n_components=config['dict_size'],
                    alpha=config['alpha'],
                    max_iter=100,
                    random_state=42
                )
                sparse_coder.dictionary_ = dictionary
                sparse_coder.is_fitted = True
                logger.info(f"✅ Using existing dictionary: {dictionary.shape}")
        else:
            logger.info("🧮 Training new high-quality dictionary...")
            # Train new dictionary with more data
            dict_images = []
            for i, path in enumerate(train_paths[:2000]):
                if i % 400 == 0:
                    logger.info(f"   Loading dict images: {i}/2000")
                image = load_image_as_grayscale(path)
                if image is not None:
                    image = apply_srm_residual(image, config['residual_type'])
                    dict_images.append(image)
            
            dict_vectors = extract_patches_from_images(
                dict_images,
                patch_size=config['patch_size'],
                stride=config['stride'],
                max_patches_per_image=1200
            )
            
            sparse_coder = SparseDictionary(
                n_components=config['dict_size'],
                alpha=config['alpha'],
                max_iter=200,
                random_state=42
            )
            
            sparse_coder.fit(dict_vectors)
            dictionary = sparse_coder.dictionary_
            logger.info(f"✅ New dictionary trained: {dictionary.shape}")
        
        # Save dictionary
        np.savez(os.path.join(output_dir, 'dictionary.npz'), dictionary=dictionary)
        
        # =================================================================
        # TRAINING DATA PROCESSING
        # =================================================================
        logger.info("🖼️ Processing training data...")
        train_images = []
        for i, path in enumerate(train_paths[:config['n_train']]):
            if i % 1000 == 0:
                logger.info(f"   Training images: {i}/{config['n_train']}")
            
            image = load_image_as_grayscale(path)
            if image is not None:
                image = apply_srm_residual(image, config['residual_type'])
                train_images.append(image)
        
        actual_train_labels = train_labels[:len(train_images)]
        logger.info(f"✅ Loaded {len(train_images)} training images")
        
        # Extract patches and features
        logger.info("✂️ Extracting training patches...")
        train_vectors = extract_patches_from_images(
            train_images,
            patch_size=config['patch_size'],
            stride=config['stride'],
            max_patches_per_image=config['max_patches']
        )
        
        logger.info("🔢 Computing sparse codes...")
        train_sparse_codes = sparse_coder.transform(
            train_vectors,
            solver=config['solver'],
            n_nonzero_coefs=config['n_nonzero_coefs']
        )
        
        # Advanced pooling
        train_features = advanced_pooling(
            train_sparse_codes, 
            len(train_images), 
            config['pooling_method']
        )
        
        # Enhanced features
        train_features = create_enhanced_features(
            train_features, 
            config['feature_method']
        )
        
        logger.info(f"✅ Final training features: {train_features.shape}")
        
        # =================================================================
        # VALIDATION DATA PROCESSING
        # =================================================================
        logger.info("🖼️ Processing validation data...")
        val_images = []
        for i, path in enumerate(val_paths[:config['n_val']]):
            if i % 300 == 0:
                logger.info(f"   Validation images: {i}/{config['n_val']}")
            
            image = load_image_as_grayscale(path)
            if image is not None:
                image = apply_srm_residual(image, config['residual_type'])
                val_images.append(image)
        
        actual_val_labels = val_labels[:len(val_images)]
        
        val_vectors = extract_patches_from_images(
            val_images,
            patch_size=config['patch_size'],
            stride=config['stride'],
            max_patches_per_image=config['max_patches']
        )
        
        val_sparse_codes = sparse_coder.transform(
            val_vectors,
            solver=config['solver'],
            n_nonzero_coefs=config['n_nonzero_coefs']
        )
        
        val_features = advanced_pooling(
            val_sparse_codes,
            len(val_images),
            config['pooling_method']
        )
        
        val_features = create_enhanced_features(
            val_features,
            config['feature_method']
        )
        
        logger.info(f"✅ Validation features: {val_features.shape}")
        
        # =================================================================
        # ADVANCED MODEL TRAINING
        # =================================================================
        logger.info("🤖 Starting advanced model training...")
        best_model, best_scaler, best_params, cv_score = train_advanced_svm(
            train_features, actual_train_labels,
            val_features, actual_val_labels
        )
        
        # =================================================================
        # TEST DATA PROCESSING AND EVALUATION
        # =================================================================
        logger.info("🧪 Processing test data...")
        test_images = []
        for i, path in enumerate(test_paths[:config['n_test']]):
            if i % 300 == 0:
                logger.info(f"   Test images: {i}/{config['n_test']}")
            
            image = load_image_as_grayscale(path)
            if image is not None:
                image = apply_srm_residual(image, config['residual_type'])
                test_images.append(image)
        
        actual_test_labels = test_labels[:len(test_images)]
        
        test_vectors = extract_patches_from_images(
            test_images,
            patch_size=config['patch_size'],
            stride=config['stride'],
            max_patches_per_image=config['max_patches']
        )
        
        test_sparse_codes = sparse_coder.transform(
            test_vectors,
            solver=config['solver'],
            n_nonzero_coefs=config['n_nonzero_coefs']
        )
        
        test_features = advanced_pooling(
            test_sparse_codes,
            len(test_images),
            config['pooling_method']
        )
        
        test_features = create_enhanced_features(
            test_features,
            config['feature_method']
        )
        
        logger.info(f"✅ Test features: {test_features.shape}")
        
        # Final evaluation
        logger.info("📊 Final evaluation...")
        test_scaled = best_scaler.transform(test_features)
        test_pred = best_model.predict(test_scaled)
        test_proba = best_model.predict_proba(test_scaled)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(actual_test_labels, test_pred),
            'precision': precision_score(actual_test_labels, test_pred, zero_division=0),
            'recall': recall_score(actual_test_labels, test_pred, zero_division=0),
            'f1': f1_score(actual_test_labels, test_pred, zero_division=0),
            'roc_auc': roc_auc_score(actual_test_labels, test_proba)
        }
        
        # =================================================================
        # SAVE RESULTS
        # =================================================================
        results = {
            'config': config,
            'metrics': metrics,
            'best_params': best_params,
            'cv_score': cv_score,
            'training_info': {
                'n_train_actual': len(train_images),
                'n_val_actual': len(val_images),
                'n_test_actual': len(test_images),
                'dict_size': dictionary.shape[0],
                'feature_dim': train_features.shape[1],
                'timestamp': timestamp
            }
        }
        
        with open(os.path.join(output_dir, 'ultimate_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        joblib.dump({
            'model': best_model,
            'scaler': best_scaler,
            'sparse_coder': sparse_coder,
            'dictionary': dictionary,
            'config': config
        }, os.path.join(output_dir, 'ultimate_model.pkl'))
        
        # =================================================================
        # DISPLAY RESULTS
        # =================================================================
        logger.info("="*80)
        logger.info("🎉 ULTIMATE OPTIMAL TRAINING COMPLETED!")
        logger.info("="*80)
        logger.info(f"📈 Training samples: {len(train_images)}")
        logger.info(f"🔍 Validation samples: {len(val_images)}")
        logger.info(f"🧪 Test samples: {len(test_images)}")
        logger.info(f"📚 Dictionary atoms: {dictionary.shape[0]}")
        logger.info(f"🔢 Feature dimension: {train_features.shape[1]}")
        logger.info(f"💾 Results saved to: {output_dir}")
        logger.info("="*80)
        
        logger.info("🏆 FINAL RESULTS:")
        logger.info(f"   Cross-Validation Score: {cv_score:.4f}")
        logger.info(f"   Best Parameters: {best_params}")
        logger.info(f"   🎯 Test Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        logger.info(f"   🎯 Test Precision: {metrics['precision']:.4f}")
        logger.info(f"   🎯 Test Recall:    {metrics['recall']:.4f}")
        logger.info(f"   🎯 Test F1-Score:  {metrics['f1']:.4f}")
        logger.info(f"   🎯 Test ROC-AUC:   {metrics['roc_auc']:.4f}")
        logger.info("="*80)
        
        # Achievement assessment
        if metrics['accuracy'] >= 0.80:
            logger.info("🏆 🎉 SUCCESS! TARGET ACHIEVED! 🎉 🏆")
            logger.info(f"🚀 ACCURACY: {metrics['accuracy']*100:.2f}% - EXCEEDS 80% TARGET!")
            logger.info("🌟 CONGRATULATIONS! OPTIMAL PERFORMANCE REACHED!")
        elif metrics['accuracy'] >= 0.75:
            logger.info("🎯 EXCELLENT! VERY CLOSE TO TARGET!")
            logger.info(f"📈 ACCURACY: {metrics['accuracy']*100:.2f}% - ALMOST THERE!")
            logger.info("💡 Minor tuning could push us over 80%!")
        elif metrics['accuracy'] >= 0.70:
            logger.info("⚡ GOOD RESULTS! SOLID PERFORMANCE!")
            logger.info(f"📊 ACCURACY: {metrics['accuracy']*100:.2f}% - STRONG FOUNDATION!")
            logger.info("🔧 Further optimization recommended for 80%+")
        else:
            logger.info("📈 BASELINE ESTABLISHED!")
            logger.info(f"📊 ACCURACY: {metrics['accuracy']*100:.2f}%")
            logger.info("🔬 Consider advanced techniques or more data")
        
        logger.info("="*80)
        logger.info("📋 DETAILED CLASSIFICATION REPORT:")
        logger.info(classification_report(actual_test_labels, test_pred,
                                        target_names=['Cover', 'Stego']))
        logger.info("="*80)
        logger.info("✨ ULTIMATE TRAINING COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()