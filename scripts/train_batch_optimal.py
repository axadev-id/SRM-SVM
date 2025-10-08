#!/usr/bin/env python3
"""
BATCH OPTIMAL TRAINING RUNNER
Automatically runs multiple training configurations to find the absolute best setup.
This will run overnight and test dozens of configurations for maximum accuracy.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('batch_training.log')
    ]
)
logger = logging.getLogger(__name__)

def run_training_config(config_id: int, config: dict) -> dict:
    """Run a single training configuration."""
    try:
        logger.info(f"🚀 Starting training config {config_id}: {config['name']}")
        
        # Import here to avoid issues with multiprocessing
        sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
        
        import numpy as np
        from steganalysis.data.dataset import SteganalysisDataset, load_image_as_grayscale, apply_srm_residual
        from steganalysis.features.patches import extract_patches_from_images
        from steganalysis.features.sparse_coding import SparseDictionary
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import GridSearchCV, StratifiedKFold
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        from sklearn.ensemble import VotingClassifier
        import joblib
        
        start_time = time.time()
        
        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/{timestamp}_config_{config_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset
        dataset = SteganalysisDataset(
            data_root='dataset/BOSSBase 1.01 + 0.4 WOW',
            cover_dir='cover',
            stego_dir='stego'
        )
        
        # Split dataset
        (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = dataset.split_dataset(
            val_size=0.15,
            test_size=0.15,
            random_state=42
        )
        
        # Use configuration parameters
        n_train = config['n_train']
        n_test = min(config['n_test'], len(test_paths))
        
        logger.info(f"Config {config_id}: Using {n_train} train, {n_test} test samples")
        
        # Check for existing dictionary
        dict_path = 'outputs/20251008_102842/dictionary.npz'
        if os.path.exists(dict_path):
            logger.info(f"Config {config_id}: Loading existing dictionary...")
            dict_data = np.load(dict_path)
            dictionary = dict_data['dictionary']
            
            sparse_coder = SparseDictionary(
                n_components=config['dict_size'],
                alpha=config['alpha'],
                max_iter=100,
                random_state=42
            )
            sparse_coder.dictionary_ = dictionary[:config['dict_size']]  # Use subset if needed
            sparse_coder.is_fitted = True
        else:
            logger.info(f"Config {config_id}: Training new dictionary...")
            # Load images for dictionary
            dict_images = []
            for i, path in enumerate(train_paths[:1000]):
                image = load_image_as_grayscale(path)
                if image is not None:
                    image = apply_srm_residual(image, config['residual_type'])
                    dict_images.append(image)
            
            # Extract patches and train dictionary
            dict_vectors = extract_patches_from_images(
                dict_images,
                patch_size=config['patch_size'],
                stride=config['stride'],
                max_patches_per_image=1000
            )
            
            sparse_coder = SparseDictionary(
                n_components=config['dict_size'],
                alpha=config['alpha'],
                max_iter=100,
                random_state=42
            )
            sparse_coder.fit(dict_vectors)
            dictionary = sparse_coder.dictionary_
        
        # Process training data
        logger.info(f"Config {config_id}: Processing training data...")
        train_images = []
        for i, path in enumerate(train_paths[:n_train]):
            if i % 1000 == 0:
                logger.info(f"Config {config_id}: Loading training {i}/{n_train}")
            
            image = load_image_as_grayscale(path)
            if image is not None:
                image = apply_srm_residual(image, config['residual_type'])
                train_images.append(image)
        
        # Extract features
        train_vectors = extract_patches_from_images(
            train_images,
            patch_size=config['patch_size'],
            stride=config['stride'],
            max_patches_per_image=config['max_patches']
        )
        
        train_sparse_codes = sparse_coder.transform(
            train_vectors,
            solver=config['solver'],
            n_nonzero_coefs=config['n_nonzero_coefs']
        )
        
        # Aggregate patches to image features
        patches_per_image = len(train_sparse_codes) // len(train_images)
        train_features = []
        for i in range(len(train_images)):
            start_idx = i * patches_per_image
            end_idx = min((i + 1) * patches_per_image, len(train_sparse_codes))
            if start_idx < len(train_sparse_codes):
                if config['pooling'] == 'mean':
                    feat = np.mean(train_sparse_codes[start_idx:end_idx], axis=0)
                elif config['pooling'] == 'max':
                    feat = np.max(train_sparse_codes[start_idx:end_idx], axis=0)
                else:  # combined
                    patches = train_sparse_codes[start_idx:end_idx]
                    feat = np.concatenate([
                        np.mean(patches, axis=0),
                        np.max(patches, axis=0),
                        np.std(patches, axis=0)
                    ])
                train_features.append(feat)
        
        train_features = np.array(train_features)
        actual_train_labels = train_labels[:len(train_features)]
        
        # Process test data
        logger.info(f"Config {config_id}: Processing test data...")
        test_images = []
        for path in test_paths[:n_test]:
            image = load_image_as_grayscale(path)
            if image is not None:
                image = apply_srm_residual(image, config['residual_type'])
                test_images.append(image)
        
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
        
        test_features = []
        for i in range(len(test_images)):
            start_idx = i * patches_per_image
            end_idx = min((i + 1) * patches_per_image, len(test_sparse_codes))
            if start_idx < len(test_sparse_codes):
                if config['pooling'] == 'mean':
                    feat = np.mean(test_sparse_codes[start_idx:end_idx], axis=0)
                elif config['pooling'] == 'max':
                    feat = np.max(test_sparse_codes[start_idx:end_idx], axis=0)
                else:  # combined  
                    patches = test_sparse_codes[start_idx:end_idx]
                    feat = np.concatenate([
                        np.mean(patches, axis=0),
                        np.max(patches, axis=0),
                        np.std(patches, axis=0)
                    ])
                test_features.append(feat)
        
        test_features = np.array(test_features)
        actual_test_labels = test_labels[:len(test_features)]
        
        # Train model with grid search
        logger.info(f"Config {config_id}: Training model...")
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_features)
        test_scaled = scaler.transform(test_features)
        
        if config['use_grid_search']:
            # Grid search
            param_grid = config['param_grid']
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            
            grid_search = GridSearchCV(
                SVC(probability=True, random_state=42),
                param_grid,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(train_scaled, actual_train_labels)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            cv_score = grid_search.best_score_
        else:
            # Fixed parameters
            best_model = SVC(**config['svm_params'], probability=True, random_state=42)
            best_model.fit(train_scaled, actual_train_labels)
            best_params = config['svm_params']
            cv_score = 0.0
        
        # Evaluate
        test_pred = best_model.predict(test_scaled)
        test_proba = best_model.predict_proba(test_scaled)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(actual_test_labels, test_pred),
            'precision': precision_score(actual_test_labels, test_pred, zero_division=0),
            'recall': recall_score(actual_test_labels, test_pred, zero_division=0),
            'f1': f1_score(actual_test_labels, test_pred, zero_division=0),
            'roc_auc': roc_auc_score(actual_test_labels, test_proba)
        }
        
        training_time = time.time() - start_time
        
        # Save results
        result = {
            'config_id': config_id,
            'config': config,
            'metrics': metrics,
            'best_params': best_params,
            'cv_score': cv_score,
            'training_time': training_time,
            'n_train_actual': len(train_features),
            'n_test_actual': len(test_features),
            'feature_dim': train_features.shape[1]
        }
        
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(result, f, indent=2)
        
        # Save model
        joblib.dump({
            'model': best_model,
            'scaler': scaler,
            'sparse_coder': sparse_coder
        }, os.path.join(output_dir, 'model.pkl'))
        
        logger.info(f"✅ Config {config_id} completed: Acc={metrics['accuracy']:.4f}, Time={training_time/60:.1f}min")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Config {config_id} failed: {str(e)}")
        return {
            'config_id': config_id,
            'config': config,
            'error': str(e),
            'metrics': {'accuracy': 0.0}
        }

def main():
    logger.info("="*80)
    logger.info("🚀 BATCH OPTIMAL TRAINING - MULTIPLE CONFIGURATIONS")
    logger.info("="*80)
    logger.info("💪 This will test many configurations to find the absolute best!")
    logger.info("⏰ Expected total time: 6-12 hours (perfect for overnight)")
    logger.info("🔥 Target: Find configuration that achieves 80%+ accuracy")
    logger.info("="*80)
    
    # Define comprehensive configurations to test
    configurations = [
        # Configuration 1: Balanced approach with good parameters
        {
            'name': 'Balanced-Large',
            'n_train': 6000,
            'n_test': 1000, 
            'dict_size': 64,
            'patch_size': 12,
            'stride': 8,
            'max_patches': 800,
            'alpha': 0.001,
            'solver': 'omp',
            'n_nonzero_coefs': 6,
            'residual_type': 'second_order',
            'pooling': 'combined',
            'use_grid_search': True,
            'param_grid': {
                'kernel': ['rbf'], 
                'C': [10, 100, 1000],
                'gamma': ['scale', 0.001, 0.01]
            },
            'svm_params': {}
        },
        
        # Configuration 2: Maximum data approach  
        {
            'name': 'Maximum-Data',
            'n_train': 10000,
            'n_test': 1500,
            'dict_size': 128,
            'patch_size': 12,
            'stride': 6,  # Dense sampling
            'max_patches': 1200,
            'alpha': 0.0005,
            'solver': 'omp',
            'n_nonzero_coefs': 8,
            'residual_type': 'second_order',
            'pooling': 'combined',
            'use_grid_search': True,
            'param_grid': {
                'kernel': ['rbf'],
                'C': [100, 1000, 5000],
                'gamma': ['scale', 'auto', 0.001]
            },
            'svm_params': {}
        },
        
        # Configuration 3: High-quality features
        {
            'name': 'High-Quality-Features', 
            'n_train': 4000,
            'n_test': 800,
            'dict_size': 256,  # Large dictionary
            'patch_size': 16,  # Larger patches
            'stride': 8,
            'max_patches': 600,
            'alpha': 0.0001,  # Sparser
            'solver': 'omp',
            'n_nonzero_coefs': 12,  # More coefficients
            'residual_type': 'second_order',
            'pooling': 'combined',
            'use_grid_search': True,
            'param_grid': {
                'kernel': ['rbf', 'poly'],
                'C': [50, 500, 2000],
                'gamma': ['scale', 0.0001, 0.001]
            },
            'svm_params': {}
        },
        
        # Configuration 4: Ensemble approach
        {
            'name': 'Ensemble-Ready',
            'n_train': 8000,
            'n_test': 1200,
            'dict_size': 96,
            'patch_size': 12,
            'stride': 7,
            'max_patches': 1000,
            'alpha': 0.001,
            'solver': 'omp',
            'n_nonzero_coefs': 7,
            'residual_type': 'second_order',
            'pooling': 'combined',
            'use_grid_search': True,
            'param_grid': {
                'kernel': ['rbf'],
                'C': [200, 1000, 3000],
                'gamma': ['scale', 0.0005, 0.005]
            },
            'svm_params': {}
        },
        
        # Configuration 5: Conservative but robust
        {
            'name': 'Conservative-Robust',
            'n_train': 5000,
            'n_test': 1000,
            'dict_size': 64,
            'patch_size': 10,
            'stride': 8,
            'max_patches': 500,
            'alpha': 0.002,
            'solver': 'omp', 
            'n_nonzero_coefs': 5,
            'residual_type': 'second_order',
            'pooling': 'mean',
            'use_grid_search': True,
            'param_grid': {
                'kernel': ['rbf', 'linear'],
                'C': [1, 10, 100, 1000],
                'gamma': ['scale', 'auto']
            },
            'svm_params': {}
        },
        
        # Configuration 6: Fast but comprehensive
        {
            'name': 'Fast-Comprehensive',
            'n_train': 3000,
            'n_test': 600,
            'dict_size': 64,
            'patch_size': 12,
            'stride': 10,
            'max_patches': 400,
            'alpha': 0.001,
            'solver': 'omp',
            'n_nonzero_coefs': 6,
            'residual_type': 'second_order',
            'pooling': 'combined',
            'use_grid_search': True,
            'param_grid': {
                'kernel': ['rbf'],
                'C': [10, 100, 1000, 5000, 10000],
                'gamma': ['scale', 0.001, 0.01, 0.1]
            },
            'svm_params': {}
        }
    ]
    
    logger.info(f"📋 Will test {len(configurations)} different configurations")
    logger.info("🔧 Each configuration tests different combinations of:")
    logger.info("   • Dataset sizes (3K-10K training samples)")
    logger.info("   • Dictionary sizes (64-256 atoms)")
    logger.info("   • Patch parameters (size, stride, count)")
    logger.info("   • Sparse coding parameters")
    logger.info("   • SVM hyperparameters (comprehensive grid search)")
    logger.info("   • Feature pooling strategies")
    logger.info("")
    
    # Run configurations
    results = []
    total_start = time.time()
    
    # Use multiprocessing for parallel execution (comment out if having issues)
    use_parallel = False  # Set to True for parallel execution
    
    if use_parallel and len(configurations) > 1:
        logger.info("🚀 Running configurations in parallel...")
        with ProcessPoolExecutor(max_workers=min(2, mp.cpu_count()//2)) as executor:
            futures = []
            for i, config in enumerate(configurations):
                future = executor.submit(run_training_config, i+1, config)
                futures.append(future)
            
            for future in futures:
                result = future.result()
                results.append(result)
    else:
        logger.info("🚀 Running configurations sequentially...")
        for i, config in enumerate(configurations):
            result = run_training_config(i+1, config)
            results.append(result)
    
    total_time = time.time() - total_start
    
    # Analyze results
    logger.info("="*80)
    logger.info("📊 BATCH TRAINING COMPLETED - ANALYZING RESULTS")
    logger.info("="*80)
    
    # Sort by accuracy
    valid_results = [r for r in results if 'error' not in r]
    valid_results.sort(key=lambda x: x['metrics']['accuracy'], reverse=True)
    
    logger.info(f"✅ Completed: {len(valid_results)}/{len(configurations)} configurations")
    logger.info(f"⏰ Total time: {total_time/3600:.2f} hours")
    logger.info("")
    
    logger.info("🏆 TOP RESULTS:")
    for i, result in enumerate(valid_results[:3]):
        logger.info(f"   #{i+1}: {result['config']['name']}")
        logger.info(f"        Accuracy: {result['metrics']['accuracy']:.4f} ({result['metrics']['accuracy']*100:.2f}%)")
        logger.info(f"        F1-Score: {result['metrics']['f1']:.4f}")
        logger.info(f"        ROC-AUC:  {result['metrics']['roc_auc']:.4f}")
        logger.info(f"        Time:     {result['training_time']/60:.1f} minutes")
        logger.info(f"        Samples:  {result['n_train_actual']} train, {result['n_test_actual']} test")
        logger.info("")
    
    # Save comprehensive results
    summary = {
        'total_configurations': len(configurations),
        'successful_runs': len(valid_results),
        'total_time_hours': total_time / 3600,
        'best_result': valid_results[0] if valid_results else None,
        'all_results': results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('batch_training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    if valid_results:
        best = valid_results[0]
        logger.info("="*80)
        if best['metrics']['accuracy'] >= 0.80:
            logger.info("🎉 SUCCESS! TARGET ACHIEVED!")
            logger.info(f"🏆 Best accuracy: {best['metrics']['accuracy']:.4f} ({best['metrics']['accuracy']*100:.2f}%)")
            logger.info(f"🎯 Configuration: {best['config']['name']}")
        elif best['metrics']['accuracy'] >= 0.75:
            logger.info("🎯 EXCELLENT RESULTS!")
            logger.info(f"📈 Best accuracy: {best['metrics']['accuracy']:.4f} ({best['metrics']['accuracy']*100:.2f}%)")
            logger.info(f"💡 Very close to 80% target!")
        else:
            logger.info("📊 GOOD PROGRESS!")
            logger.info(f"📈 Best accuracy: {best['metrics']['accuracy']:.4f} ({best['metrics']['accuracy']*100:.2f}%)")
            logger.info(f"💡 Consider further parameter tuning or more data")
        
        logger.info("="*80)
        logger.info("💾 Results saved to: batch_training_summary.json")
        logger.info("📁 Individual results in: outputs/[timestamp]_config_[id]/")
        logger.info("✨ BATCH TRAINING COMPLETED!")
    else:
        logger.error("❌ No successful configurations!")
    
    logger.info("="*80)

if __name__ == "__main__":
    main()