"""Inference script for SRM-SVM steganalysis."""

import logging
import sys
from pathlib import Path
from typing import List

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from steganalysis.data.dataset import load_image_as_grayscale, apply_srm_residual
from steganalysis.features.patches import extract_patches_from_images
from steganalysis.features.sparse_coding import SparseDictionary, aggregate_sparse_codes
from steganalysis.models.classifier import SteganalysisClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_images_for_inference(
    image_dir: Path,
    apply_residual: bool = False,
    residual_type: str = "first_order",
) -> tuple[List[Path], List[np.ndarray]]:
    """Load images from directory for inference.
    
    Args:
        image_dir: Directory containing images
        apply_residual: Whether to apply SRM residual filtering
        residual_type: Type of residual filtering
        
    Returns:
        Tuple of (image_paths, loaded_images)
    """
    # Find all image files
    image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']:
        image_paths.extend(list(image_dir.glob(ext)))
        image_paths.extend(list(image_dir.glob(ext.upper())))
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in directory: {image_dir}")
    
    logger.info(f"Found {len(image_paths)} images in {image_dir}")
    
    # Load images
    images = []
    valid_paths = []
    
    for path in tqdm(image_paths, desc="Loading images"):
        image = load_image_as_grayscale(path)
        
        if image is None:
            logger.warning(f"Failed to load image: {path}")
            continue
            
        if apply_residual:
            image = apply_srm_residual(image, residual_type)
        
        images.append(image)
        valid_paths.append(path)
    
    logger.info(f"Successfully loaded {len(images)} images")
    return valid_paths, images


def extract_features_for_inference(
    images: List[np.ndarray],
    dictionary: SparseDictionary,
    patch_size: int = 8,
    stride: int = 4,
    sparse_solver: str = 'omp',
    n_nonzero_coefs: int = 5,
    alpha_coding: float = 0.01,
) -> np.ndarray:
    """Extract features from images for inference.
    
    Args:
        images: List of preprocessed images
        dictionary: Loaded sparse dictionary
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
@click.option('--image-dir', type=click.Path(exists=True), required=True,
              help='Directory containing images to analyze')
@click.option('--model-dir', type=click.Path(exists=True), required=True,
              help='Directory containing trained model files')
@click.option('--output-file', default='predictions.csv',
              help='Output CSV file for predictions')
@click.option('--patch-size', default=8, help='Size of image patches')
@click.option('--stride', default=4, help='Stride for patch extraction')
@click.option('--sparse-solver', type=click.Choice(['omp', 'lasso']), default='omp',
              help='Sparse coding solver')
@click.option('--n-nonzero-coefs', default=5, help='Max non-zero coefficients for OMP')
@click.option('--alpha-coding', default=0.01, help='Regularization parameter for Lasso')
@click.option('--apply-residual', is_flag=True, help='Apply SRM residual filtering')
@click.option('--residual-type', default='first_order',
              type=click.Choice(['first_order', 'second_order']),
              help='Type of residual filtering')
@click.option('--batch-size', default=100, help='Batch size for processing')
def main(
    image_dir: str,
    model_dir: str,
    output_file: str,
    patch_size: int,
    stride: int,
    sparse_solver: str,
    n_nonzero_coefs: int,
    alpha_coding: float,
    apply_residual: bool,
    residual_type: str,
    batch_size: int,
) -> None:
    """Run inference on images using trained SRM-SVM model."""
    
    logger.info("Starting SRM-SVM steganalysis inference")
    
    try:
        # Load trained models
        model_path = Path(model_dir)
        
        # Load dictionary
        dict_path = model_path / "dictionary.npz"
        if not dict_path.exists():
            raise FileNotFoundError(f"Dictionary file not found: {dict_path}")
        
        logger.info("Loading sparse dictionary...")
        dictionary = SparseDictionary()
        dictionary.load(dict_path)
        
        # Load classifier
        classifier_path = model_path / "classifier.joblib"
        if not classifier_path.exists():
            raise FileNotFoundError(f"Classifier file not found: {classifier_path}")
        
        logger.info("Loading classifier...")
        classifier = SteganalysisClassifier()
        classifier.load(classifier_path)
        
        # Load images
        logger.info("Loading images...")
        image_paths, images = load_images_for_inference(
            Path(image_dir), apply_residual, residual_type
        )
        
        # Process images in batches
        all_predictions = []
        all_probabilities = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_paths = image_paths[i:i + batch_size]
            
            logger.info(f"Processing batch {i // batch_size + 1}/{(len(images) + batch_size - 1) // batch_size}")
            
            # Extract features
            features = extract_features_for_inference(
                batch_images, dictionary, patch_size, stride,
                sparse_solver, n_nonzero_coefs, alpha_coding
            )
            
            # Make predictions
            predictions = classifier.predict(features)
            probabilities = classifier.predict_proba(features)
            
            all_predictions.extend(predictions)
            all_probabilities.extend(probabilities)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'image_path': [str(path) for path in image_paths],
            'image_name': [path.name for path in image_paths],
            'predicted_class': all_predictions,
            'predicted_label': ['stego' if pred == 1 else 'cover' for pred in all_predictions],
            'cover_probability': [prob[0] for prob in all_probabilities],
            'stego_probability': [prob[1] for prob in all_probabilities],
        })
        
        # Save results
        output_path = Path(output_file)
        results.to_csv(output_path, index=False)
        
        # Print summary
        n_cover = np.sum(results['predicted_class'] == 0)
        n_stego = np.sum(results['predicted_class'] == 1)
        
        logger.info(f"Inference completed!")
        logger.info(f"Total images processed: {len(results)}")
        logger.info(f"Predicted as cover: {n_cover} ({n_cover/len(results)*100:.1f}%)")
        logger.info(f"Predicted as stego: {n_stego} ({n_stego/len(results)*100:.1f}%)")
        logger.info(f"Results saved to: {output_path}")
        
        # Show some statistics
        mean_stego_prob = results['stego_probability'].mean()
        std_stego_prob = results['stego_probability'].std()
        
        logger.info(f"Average stego probability: {mean_stego_prob:.3f} ± {std_stego_prob:.3f}")
        
        # Show most confident predictions
        logger.info("\nMost confident stego predictions:")
        top_stego = results.nlargest(5, 'stego_probability')[['image_name', 'stego_probability']]
        for _, row in top_stego.iterrows():
            logger.info(f"  {row['image_name']}: {row['stego_probability']:.3f}")
        
        logger.info("\nMost confident cover predictions:")
        top_cover = results.nsmallest(5, 'stego_probability')[['image_name', 'stego_probability']]
        for _, row in top_cover.iterrows():
            logger.info(f"  {row['image_name']}: {row['stego_probability']:.3f}")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


if __name__ == "__main__":
    main()