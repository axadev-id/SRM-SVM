"""Training script for SRM-SVM steganalysis."""

import logging
import sys
from pathlib import Path
from typing import List, Optional

import click
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from steganalysis.data.dataset import SteganalysisDataset, load_image_as_grayscale, apply_srm_residual
from steganalysis.features.patches import extract_patches_from_images
from steganalysis.features.sparse_coding import SparseDictionary, aggregate_sparse_codes
from steganalysis.models.classifier import SteganalysisClassifier
from steganalysis.utils.evaluation import (
    setup_output_directory, save_metrics, plot_roc_curve,
    plot_confusion_matrix, save_training_log, create_experiment_summary
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


def load_images_from_paths(
    image_paths: List[Path],
    apply_residual: bool = False,
    residual_type: str = "first_order",
) -> List[np.ndarray]:
    """Load images from paths with optional preprocessing.
    
    Args:
        image_paths: List of image file paths
        apply_residual: Whether to apply SRM residual filtering
        residual_type: Type of residual filtering
        
    Returns:
        List of loaded and preprocessed images
    """
    images = []
    failed_count = 0
    
    for path in tqdm(image_paths, desc="Loading images"):
        image = load_image_as_grayscale(path)
        
        if image is None:
            failed_count += 1
            continue
            
        if apply_residual:
            image = apply_srm_residual(image, residual_type)
        
        images.append(image)
    
    if failed_count > 0:
        logger.warning(f"Failed to load {failed_count} images")
    
    logger.info(f"Successfully loaded {len(images)} images")
    return images


def extract_features_from_images(
    images: List[np.ndarray],
    dictionary: SparseDictionary,
    patch_size: int = 8,
    stride: int = 4,
    sparse_solver: str = 'omp',
    n_nonzero_coefs: int = 5,
    alpha_coding: float = 0.01,
) -> np.ndarray:
    """Extract sparse coding features from images.
    
    Args:
        images: List of preprocessed images
        dictionary: Fitted sparse dictionary
        patch_size: Size of patches
        stride: Stride for patch extraction
        sparse_solver: Sparse coding solver
        n_nonzero_coefs: Max non-zero coefficients for OMP
        alpha_coding: Regularization for Lasso
        
    Returns:
        Feature matrix (n_images, n_features)
    """
    features = []
    
    for image in tqdm(images, desc="Extracting features"):
        # Extract patches
        patches = extract_patches_from_images(
            [image], patch_size, stride, max_patches_per_image=10000
        )
        
        if patches.size == 0:
            # Use zero features if no patches extracted
            zero_features = np.zeros(3 * dictionary.n_components)
            features.append(zero_features)
            continue
        
        # Sparse coding
        codes = dictionary.transform(
            patches, sparse_solver, n_nonzero_coefs, alpha_coding
        )
        
        # Aggregate codes to fixed-size feature vector
        feature_vector = aggregate_sparse_codes(codes)
        features.append(feature_vector)
    
    return np.array(features)


@click.command()
@click.option('--data-root', type=click.Path(exists=True), required=True,
              help='Root directory containing the dataset')
@click.option('--cover-dir', default='cover', help='Subdirectory containing cover images')
@click.option('--stego-dir', default='stego', help='Subdirectory containing stego images')
@click.option('--dict-size', default=256, help='Dictionary size (number of atoms)')
@click.option('--patch-size', default=8, help='Size of image patches')
@click.option('--stride', default=4, help='Stride for patch extraction')
@click.option('--sparse-solver', type=click.Choice(['omp', 'lasso']), default='omp',
              help='Sparse coding solver')
@click.option('--n-nonzero-coefs', default=5, help='Max non-zero coefficients for OMP')
@click.option('--alpha-coding', default=0.01, help='Regularization parameter for Lasso')
@click.option('--max-patches', default=200000, help='Maximum patches for dictionary learning')
@click.option('--val-size', default=0.2, help='Validation set size')
@click.option('--test-size', default=0.2, help='Test set size')
@click.option('--seed', default=42, help='Random seed')
@click.option('--apply-residual', is_flag=True, help='Apply SRM residual filtering')
@click.option('--residual-type', default='first_order', 
              type=click.Choice(['first_order', 'second_order']),
              help='Type of residual filtering')
@click.option('--output-dir', default='outputs', help='Base output directory')
def main(
    data_root: str,
    cover_dir: str,
    stego_dir: str,
    dict_size: int,
    patch_size: int,
    stride: int,
    sparse_solver: str,
    n_nonzero_coefs: int,
    alpha_coding: float,
    max_patches: int,
    val_size: float,
    test_size: float,
    seed: int,
    apply_residual: bool,
    residual_type: str,
    output_dir: str,
) -> None:
    """Train SRM-SVM steganalysis model."""
    
    # Set random seed
    np.random.seed(seed)
    
    # Setup output directory
    output_path = setup_output_directory(output_dir)
    
    # Setup file logging
    log_handler = logging.FileHandler(output_path / "training.log")
    log_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logging.getLogger().addHandler(log_handler)
    
    logger.info("Starting SRM-SVM steganalysis training")
    
    # Configuration for logging
    config = {
        'data_root': data_root,
        'cover_dir': cover_dir,
        'stego_dir': stego_dir,
        'dict_size': dict_size,
        'patch_size': patch_size,
        'stride': stride,
        'sparse_solver': sparse_solver,
        'n_nonzero_coefs': n_nonzero_coefs,
        'alpha_coding': alpha_coding,
        'max_patches': max_patches,
        'val_size': val_size,
        'test_size': test_size,
        'seed': seed,
        'apply_residual': apply_residual,
        'residual_type': residual_type,
    }
    
    try:
        # 1. Load dataset
        logger.info("Loading dataset...")
        dataset = SteganalysisDataset(Path(data_root), cover_dir, stego_dir)
        (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = \
            dataset.split_dataset(val_size, test_size, seed)
        
        # 2. Load training images for dictionary learning
        logger.info("Loading training images...")
        train_images = load_images_from_paths(train_paths, apply_residual, residual_type)
        
        # 3. Extract patches for dictionary learning
        logger.info("Extracting patches for dictionary learning...")
        training_patches = extract_patches_from_images(
            train_images, patch_size, stride, max_patches_per_image=1000
        )
        
        # Limit total patches if needed
        if len(training_patches) > max_patches:
            indices = np.random.choice(len(training_patches), max_patches, replace=False)
            training_patches = training_patches[indices]
        
        logger.info(f"Using {len(training_patches)} patches for dictionary learning")
        
        # 4. Learn sparse dictionary
        logger.info("Learning sparse dictionary...")
        dictionary = SparseDictionary(n_components=dict_size, random_state=seed)
        dictionary.fit(training_patches)
        
        # Save dictionary
        dict_path = output_path / "dictionary.npz"
        dictionary.save(dict_path)
        
        # 5. Extract features for all splits
        logger.info("Extracting features for training set...")
        X_train = extract_features_from_images(
            train_images, dictionary, patch_size, stride,
            sparse_solver, n_nonzero_coefs, alpha_coding
        )
        
        logger.info("Extracting features for validation set...")
        val_images = load_images_from_paths(val_paths, apply_residual, residual_type)
        X_val = extract_features_from_images(
            val_images, dictionary, patch_size, stride,
            sparse_solver, n_nonzero_coefs, alpha_coding
        )
        
        logger.info("Extracting features for test set...")
        test_images = load_images_from_paths(test_paths, apply_residual, residual_type)
        X_test = extract_features_from_images(
            test_images, dictionary, patch_size, stride,
            sparse_solver, n_nonzero_coefs, alpha_coding
        )
        
        # 6. Train classifier
        logger.info("Training SVM classifier...")
        classifier = SteganalysisClassifier(random_state=seed)
        classifier.fit(X_train, train_labels, X_val, val_labels)
        
        # Save classifier
        model_path = output_path / "classifier.joblib"
        classifier.save(model_path)
        
        # 7. Evaluate on test set
        logger.info("Evaluating on test set...")
        test_metrics = classifier.evaluate(X_test, test_labels)
        
        # Get predictions for detailed analysis
        test_predictions = classifier.predict(X_test)
        test_probabilities = classifier.predict_proba(X_test)[:, 1]
        confusion_mat = classifier.get_confusion_matrix(X_test, test_labels)
        classification_rep = classifier.get_classification_report(X_test, test_labels)
        
        # 8. Save results
        logger.info("Saving results...")
        save_metrics(test_metrics, output_path)
        
        # Save classification report
        with open(output_path / "classification_report.txt", 'w') as f:
            f.write(classification_rep)
        
        # Create visualizations
        plot_roc_curve(test_labels, test_probabilities, output_path)
        plot_confusion_matrix(confusion_mat, output_path)
        
        # Save training log
        training_log = {
            'config': config,
            'dataset_info': {
                'train_size': len(train_labels),
                'val_size': len(val_labels),
                'test_size': len(test_labels),
                'train_cover': int(np.sum(train_labels == 0)),
                'train_stego': int(np.sum(train_labels == 1)),
            },
            'dictionary_info': {
                'n_components': dict_size,
                'training_patches': len(training_patches),
                'patch_dimension': training_patches.shape[1] if len(training_patches) > 0 else 0,
            },
            'classifier_info': {
                'best_params': classifier.best_params_,
                'feature_dimension': X_train.shape[1],
            },
            'test_metrics': test_metrics,
        }
        save_training_log(training_log, output_path)
        
        # Create experiment summary
        create_experiment_summary(test_metrics, config, output_path)
        
        logger.info(f"Training completed successfully!")
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Test F1-Score: {test_metrics['f1']:.4f}")
        logger.info(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
        logger.info(f"Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()