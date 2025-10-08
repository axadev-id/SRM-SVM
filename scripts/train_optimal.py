#!/usr/bin/env python3
"""
ULTIMATE OPTIMAL SRM-SVM TRAINING - Target: 80%+ Accuracy
Full dataset, comprehensive hyperparameter tuning, advanced techniques.
Designed for maximum accuracy regardless of training time.
"""

import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import json
import joblib
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_optimal.log')
    ]
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def create_advanced_feature_combinations(sparse_features: np.ndarray) -> np.ndarray:
    """Create advanced feature combinations for better discrimination."""
    logger.info("🧬 Creating advanced feature combinations...")
    
    # Original features
    features = [sparse_features]
    
    # Statistical features
    features.append(np.power(sparse_features, 2))  # Squared features
    features.append(np.sqrt(np.abs(sparse_features) + 1e-8))  # Square root features
    
    # Interaction features (selected combinations to avoid explosion)
    n_features = sparse_features.shape[1]
    if n_features <= 64:  # Only for manageable sizes
        for i in range(0, min(n_features, 32), 4):
            for j in range(i+1, min(n_features, 32), 4):
                interaction = sparse_features[:, i:i+1] * sparse_features[:, j:j+1]
                features.append(interaction)
    
    combined = np.hstack(features)
    logger.info(f"✅ Enhanced features: {sparse_features.shape} -> {combined.shape}")
    return combined

def aggregate_patches_advanced(sparse_codes: np.ndarray, n_images: int, patches_per_image: int) -> np.ndarray:
    """Advanced patch aggregation with multiple pooling strategies."""
    logger.info("📊 Advanced patch aggregation...")
    
    image_features = []
    for i in range(n_images):
        start_idx = i * patches_per_image
        end_idx = min((i + 1) * patches_per_image, len(sparse_codes))
        
        if start_idx < len(sparse_codes):
            patches = sparse_codes[start_idx:end_idx]
            
            # Multiple pooling strategies
            mean_pool = np.mean(patches, axis=0)
            max_pool = np.max(patches, axis=0) 
            min_pool = np.min(patches, axis=0)
            std_pool = np.std(patches, axis=0)
            
            # L2 norm pooling
            l2_pool = np.sqrt(np.mean(patches**2, axis=0))
            
            # Combine all pooling strategies
            combined_features = np.concatenate([
                mean_pool, max_pool, min_pool, std_pool, l2_pool
            ])
            
            image_features.append(combined_features)
    
    result = np.array(image_features)
    logger.info(f"✅ Advanced aggregation: {result.shape}")
    return result

def train_ensemble_svm(X_train: np.ndarray, y_train: np.ndarray, 
                      X_val: np.ndarray, y_val: np.ndarray) -> List[Any]:
    """Train ensemble of SVMs with different configurations."""
    from sklearn.svm import SVC
    from sklearn.ensemble import VotingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    
    logger.info("🤖 Training ensemble of SVMs...")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Define different SVM configurations for ensemble
    svm_configs = [
        {'kernel': 'rbf', 'C': 10.0, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 100.0, 'gamma': 'auto'},
        {'kernel': 'rbf', 'C': 1000.0, 'gamma': 0.001},
        {'kernel': 'poly', 'C': 10.0, 'degree': 3},
        {'kernel': 'linear', 'C': 10.0},
    ]
    
    # Train individual SVMs
    svms = []
    for i, config in enumerate(svm_configs):
        logger.info(f"   Training SVM {i+1}/5: {config}")
        svm = SVC(probability=True, random_state=42, **config)
        svm.fit(X_train_scaled, y_train)
        
        # Validate
        val_pred = svm.predict(X_val_scaled)
        val_acc = accuracy_score(y_val, val_pred)
        logger.info(f"   SVM {i+1} validation accuracy: {val_acc:.4f}")
        
        svms.append((f'svm_{i+1}', svm))
    
    # Create ensemble
    ensemble = VotingClassifier(svms, voting='soft')
    ensemble.fit(X_train_scaled, y_train)
    
    # Validate ensemble
    ensemble_pred = ensemble.predict(X_val_scaled)
    ensemble_acc = accuracy_score(y_val, ensemble_pred)
    logger.info(f"✅ Ensemble validation accuracy: {ensemble_acc:.4f}")
    
    return ensemble, scaler

def comprehensive_grid_search(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
    """Comprehensive grid search for optimal hyperparameters."""
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    
    logger.info("🔍 Starting comprehensive grid search...")
    logger.info("⚠️  This will take several hours but will find optimal parameters!")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Comprehensive parameter grid
    param_grid = [
        {
            'kernel': ['rbf'],
            'C': [0.1, 1, 10, 100, 1000, 5000],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0]
        },
        {
            'kernel': ['poly'],
            'C': [0.1, 1, 10, 100, 1000],
            'degree': [2, 3, 4],
            'gamma': ['scale', 'auto', 0.001, 0.01]
        },
        {
            'kernel': ['linear'],
            'C': [0.1, 1, 10, 100, 1000, 5000]
        }
    ]
    
    # Use stratified k-fold for robust validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Grid search with comprehensive scoring
    grid_search = GridSearchCV(
        SVC(probability=True, random_state=42),
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,  # Use all CPU cores
        verbose=2,
        return_train_score=True
    )
    
    logger.info("🚀 Starting grid search (this will take time...)") 
    grid_search.fit(X_train_scaled, y_train)
    
    logger.info("✅ Grid search completed!")
    logger.info(f"🏆 Best score: {grid_search.best_score_:.4f}")
    logger.info(f"🎯 Best params: {grid_search.best_params_}")
    
    return {
        'best_estimator': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'scaler': scaler,
        'cv_results': grid_search.cv_results_
    }

def main():
    logger.info("="*80)
    logger.info("🚀 ULTIMATE OPTIMAL SRM-SVM TRAINING - TARGET: 80%+ ACCURACY")
    logger.info("="*80)
    logger.info("⏰ This training is designed for MAXIMUM accuracy")
    logger.info("💪 Using full dataset + comprehensive hyperparameter tuning")
    logger.info("🔥 Expected training time: 4-8 hours (be patient!)")
    logger.info("="*80)
    
    try:
        # Import after path setup
        from steganalysis.data.dataset import SteganalysisDataset, load_image_as_grayscale, apply_srm_residual
        from steganalysis.features.patches import extract_patches_from_images
        from steganalysis.features.sparse_coding import SparseDictionary
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
        
        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/{timestamp}_optimal"
        os.makedirs(output_dir, exist_ok=True)
        
        # Load full dataset
        logger.info("📂 Loading FULL dataset...")
        dataset = SteganalysisDataset(
            data_root='dataset/BOSSBase 1.01 + 0.4 WOW',
            cover_dir='cover',
            stego_dir='stego'
        )
        
        # Split dataset - use more data for training
        logger.info("🔀 Splitting dataset (80% train, 10% val, 10% test)...")
        (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = dataset.split_dataset(
            val_size=0.10,
            test_size=0.10,
            random_state=42
        )
        
        # Use substantial amount of data for optimal results
        n_train = min(8000, len(train_paths))  # Use up to 8000 training samples
        n_val = min(1000, len(val_paths))     # Use up to 1000 validation samples  
        n_test = min(1000, len(test_paths))   # Use up to 1000 test samples
        
        logger.info(f"🎯 Using {n_train} training, {n_val} validation, {n_test} test samples")
        logger.info("📊 This is a SUBSTANTIAL dataset for optimal results!")
        
        # Check for existing dictionary first
        existing_dict_path = 'outputs/20251008_102842/dictionary.npz'
        
        if os.path.exists(existing_dict_path):
            logger.info("📚 Loading existing high-quality dictionary...")
            dict_data = np.load(existing_dict_path)
            dictionary = dict_data['dictionary']
            logger.info(f"✅ Loaded dictionary: {dictionary.shape}")
            
            # Setup sparse coder with existing dictionary
            sparse_coder = SparseDictionary(
                n_components=64,
                alpha=0.001,
                max_iter=100,
                random_state=42
            )
            sparse_coder.dictionary_ = dictionary
            sparse_coder.is_fitted = True
            
        else:
            logger.info("🧮 Training NEW high-quality dictionary...")
            logger.info("⏰ This will take time but ensures optimal feature representation")
            
            # Load subset for dictionary training
            dict_images = []
            logger.info("🖼️ Loading images for dictionary training...")
            
            for i, path in enumerate(train_paths[:2000]):  # Use 2000 images for dictionary
                if i % 200 == 0:
                    logger.info(f"   Loading {i}/2000...")
                
                image = load_image_as_grayscale(path)
                if image is not None:
                    image = apply_srm_residual(image, 'second_order')
                    dict_images.append(image)
            
            # Extract patches for dictionary
            logger.info("✂️ Extracting patches for dictionary...")
            dict_vectors = extract_patches_from_images(
                dict_images,
                patch_size=12,
                stride=6,  # Dense sampling for better dictionary
                max_patches_per_image=2000  # More patches per image
            )
            
            logger.info(f"📊 Dictionary patches: {dict_vectors.shape}")
            
            # Train dictionary
            logger.info("🧮 Training high-quality dictionary...")
            sparse_coder = SparseDictionary(
                n_components=128,  # Larger dictionary for better representation
                alpha=0.001,
                max_iter=200,  # More iterations for convergence
                random_state=42
            )
            
            sparse_coder.fit(dict_vectors)
            dictionary = sparse_coder.dictionary_
            
            # Save dictionary
            np.savez(os.path.join(output_dir, 'dictionary.npz'), 
                    dictionary=dictionary)
            logger.info(f"✅ Dictionary trained and saved: {dictionary.shape}")
        
        # =================================================================
        # TRAINING DATA PROCESSING
        # =================================================================
        logger.info("🖼️ Loading training images...")
        train_images = []
        actual_train_labels = train_labels[:n_train]
        
        for i, path in enumerate(train_paths[:n_train]):
            if i % 500 == 0:
                logger.info(f"   Training images: {i}/{n_train}")
            
            image = load_image_as_grayscale(path)
            if image is not None:
                image = apply_srm_residual(image, 'second_order')
                train_images.append(image)
        
        logger.info(f"✅ Loaded {len(train_images)} training images")
        
        # Extract training patches
        logger.info("✂️ Extracting training patches...")
        train_vectors = extract_patches_from_images(
            train_images,
            patch_size=12,
            stride=8,
            max_patches_per_image=1000  # Good balance
        )
        
        logger.info(f"📊 Training patch vectors: {train_vectors.shape}")
        
        # Transform to sparse features  
        logger.info("🔢 Transforming to sparse features...")
        train_sparse_codes = sparse_coder.transform(
            train_vectors,
            solver='omp',
            n_nonzero_coefs=8  # More coefficients for richer representation
        )
        
        # Advanced patch aggregation
        patches_per_image = len(train_sparse_codes) // len(train_images)
        train_features = aggregate_patches_advanced(
            train_sparse_codes, len(train_images), patches_per_image
        )
        
        # Create advanced feature combinations
        train_features = create_advanced_feature_combinations(train_features)
        logger.info(f"✅ Final training features: {train_features.shape}")
        
        # =================================================================
        # VALIDATION DATA PROCESSING
        # =================================================================
        logger.info("🖼️ Loading validation images...")
        val_images = []
        actual_val_labels = val_labels[:n_val]
        
        for i, path in enumerate(val_paths[:n_val]):
            if i % 200 == 0:
                logger.info(f"   Validation images: {i}/{n_val}")
            
            image = load_image_as_grayscale(path)
            if image is not None:
                image = apply_srm_residual(image, 'second_order')
                val_images.append(image)
        
        # Process validation data (same pipeline)
        val_vectors = extract_patches_from_images(
            val_images,
            patch_size=12,
            stride=8,
            max_patches_per_image=1000
        )
        
        val_sparse_codes = sparse_coder.transform(
            val_vectors,
            solver='omp',
            n_nonzero_coefs=8
        )
        
        val_features = aggregate_patches_advanced(
            val_sparse_codes, len(val_images), patches_per_image
        )
        
        val_features = create_advanced_feature_combinations(val_features)
        logger.info(f"✅ Validation features: {val_features.shape}")
        
        # =================================================================
        # COMPREHENSIVE HYPERPARAMETER OPTIMIZATION
        # =================================================================
        logger.info("🔍 Starting comprehensive hyperparameter optimization...")
        logger.info("⏰ This phase will take 2-4 hours but is crucial for 80%+ accuracy")
        
        grid_results = comprehensive_grid_search(train_features, actual_train_labels)
        
        best_model = grid_results['best_estimator']
        best_scaler = grid_results['scaler']
        
        logger.info("🏆 Optimal hyperparameters found!")
        logger.info(f"   Best CV score: {grid_results['best_score']:.4f}")
        logger.info(f"   Best params: {grid_results['best_params']}")
        
        # =================================================================
        # ENSEMBLE TRAINING FOR MAXIMUM ROBUSTNESS
        # =================================================================
        logger.info("🤖 Training ensemble for maximum robustness...")
        ensemble_model, ensemble_scaler = train_ensemble_svm(
            train_features, actual_train_labels, 
            val_features, actual_val_labels
        )
        
        # =================================================================
        # TEST DATA PROCESSING AND FINAL EVALUATION
        # =================================================================
        logger.info("🧪 Processing test data...")
        test_images = []
        actual_test_labels = test_labels[:n_test]
        
        for i, path in enumerate(test_paths[:n_test]):
            if i % 200 == 0:
                logger.info(f"   Test images: {i}/{n_test}")
            
            image = load_image_as_grayscale(path)
            if image is not None:
                image = apply_srm_residual(image, 'second_order')
                test_images.append(image)
        
        # Process test data
        test_vectors = extract_patches_from_images(
            test_images,
            patch_size=12, 
            stride=8,
            max_patches_per_image=1000
        )
        
        test_sparse_codes = sparse_coder.transform(
            test_vectors,
            solver='omp',
            n_nonzero_coefs=8
        )
        
        test_features = aggregate_patches_advanced(
            test_sparse_codes, len(test_images), patches_per_image
        )
        
        test_features = create_advanced_feature_combinations(test_features)
        logger.info(f"✅ Test features: {test_features.shape}")
        
        # =================================================================
        # FINAL EVALUATION
        # =================================================================
        logger.info("📊 Final evaluation...")
        
        # Test best individual model
        test_scaled = best_scaler.transform(test_features)
        test_pred = best_model.predict(test_scaled)
        test_proba = best_model.predict_proba(test_scaled)[:, 1]
        
        # Test ensemble
        ensemble_test_scaled = ensemble_scaler.transform(test_features)
        ensemble_pred = ensemble_model.predict(ensemble_test_scaled)
        ensemble_proba = ensemble_model.predict_proba(ensemble_test_scaled)[:, 1]
        
        # Calculate metrics for both
        individual_metrics = {
            'accuracy': accuracy_score(actual_test_labels, test_pred),
            'precision': precision_score(actual_test_labels, test_pred, zero_division=0),
            'recall': recall_score(actual_test_labels, test_pred, zero_division=0),
            'f1': f1_score(actual_test_labels, test_pred, zero_division=0),
            'roc_auc': roc_auc_score(actual_test_labels, test_proba)
        }
        
        ensemble_metrics = {
            'accuracy': accuracy_score(actual_test_labels, ensemble_pred),
            'precision': precision_score(actual_test_labels, ensemble_pred, zero_division=0),
            'recall': recall_score(actual_test_labels, ensemble_pred, zero_division=0),
            'f1': f1_score(actual_test_labels, ensemble_pred, zero_division=0),
            'roc_auc': roc_auc_score(actual_test_labels, ensemble_proba)
        }
        
        # =================================================================
        # SAVE RESULTS
        # =================================================================
        
        # Save comprehensive results
        results = {
            'individual_model': {
                'metrics': individual_metrics,
                'best_params': grid_results['best_params'],
                'cv_score': grid_results['best_score']
            },
            'ensemble_model': {
                'metrics': ensemble_metrics
            },
            'training_info': {
                'n_train': n_train,
                'n_val': n_val,
                'n_test': n_test,
                'dict_size': dictionary.shape[0],
                'feature_dim': train_features.shape[1],
                'timestamp': timestamp
            }
        }
        
        with open(os.path.join(output_dir, 'comprehensive_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save models
        joblib.dump({
            'best_individual_model': best_model,
            'best_scaler': best_scaler,
            'ensemble_model': ensemble_model,
            'ensemble_scaler': ensemble_scaler,
            'sparse_coder': sparse_coder,
            'dictionary': dictionary,
            'grid_results': grid_results
        }, os.path.join(output_dir, 'optimal_models.pkl'))
        
        # =================================================================
        # DISPLAY RESULTS
        # =================================================================
        logger.info("="*80)
        logger.info("🎉 ULTIMATE OPTIMAL TRAINING COMPLETED!")
        logger.info("="*80)
        logger.info(f"📈 Training samples: {n_train}")
        logger.info(f"🔍 Validation samples: {n_val}")
        logger.info(f"🧪 Test samples: {n_test}")
        logger.info(f"📚 Dictionary atoms: {dictionary.shape[0]}")
        logger.info(f"🔢 Final feature dimension: {train_features.shape[1]}")
        logger.info(f"💾 Results saved to: {output_dir}")
        logger.info("="*80)
        
        logger.info("🏆 INDIVIDUAL BEST MODEL RESULTS:")
        logger.info(f"   Best CV Score: {grid_results['best_score']:.4f}")
        logger.info(f"   Best Params: {grid_results['best_params']}")
        logger.info(f"   🎯 Test Accuracy:  {individual_metrics['accuracy']:.4f}")
        logger.info(f"   🎯 Test Precision: {individual_metrics['precision']:.4f}")
        logger.info(f"   🎯 Test Recall:    {individual_metrics['recall']:.4f}")
        logger.info(f"   🎯 Test F1-Score:  {individual_metrics['f1']:.4f}")
        logger.info(f"   🎯 Test ROC-AUC:   {individual_metrics['roc_auc']:.4f}")
        
        logger.info("-"*80)
        
        logger.info("🤖 ENSEMBLE MODEL RESULTS:")
        logger.info(f"   🎯 Test Accuracy:  {ensemble_metrics['accuracy']:.4f}")
        logger.info(f"   🎯 Test Precision: {ensemble_metrics['precision']:.4f}")
        logger.info(f"   🎯 Test Recall:    {ensemble_metrics['recall']:.4f}")
        logger.info(f"   🎯 Test F1-Score:  {ensemble_metrics['f1']:.4f}")
        logger.info(f"   🎯 Test ROC-AUC:   {ensemble_metrics['roc_auc']:.4f}")
        logger.info("="*80)
        
        # Achievement assessment
        best_accuracy = max(individual_metrics['accuracy'], ensemble_metrics['accuracy'])
        
        if best_accuracy >= 0.80:
            logger.info("🏆 SUCCESS! TARGET ACHIEVED: 80%+ ACCURACY!")
            logger.info(f"🚀 Final accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        elif best_accuracy >= 0.75:
            logger.info("🎯 EXCELLENT! Very close to target!")
            logger.info(f"📈 Final accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        else:
            logger.info("⚡ GOOD PROGRESS! Further optimization may be needed.")
            logger.info(f"📊 Final accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        
        logger.info("="*80)
        logger.info("📋 DETAILED CLASSIFICATION REPORT:")
        logger.info("\nIndividual Model:")
        logger.info(classification_report(actual_test_labels, test_pred, 
                                        target_names=['Cover', 'Stego']))
        logger.info("\nEnsemble Model:")
        logger.info(classification_report(actual_test_labels, ensemble_pred,
                                        target_names=['Cover', 'Stego']))
        logger.info("="*80)
        
        logger.info("✨ OPTIMAL TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("💡 Models and results saved for deployment.")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()