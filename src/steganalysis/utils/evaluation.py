"""Evaluation and visualization utilities."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

logger = logging.getLogger(__name__)


def setup_output_directory(base_dir: str = "outputs") -> Path:
    """Create timestamped output directory.
    
    Args:
        base_dir: Base directory for outputs
        
    Returns:
        Path to the created output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory created: {output_dir}")
    return output_dir


def save_metrics(metrics: Dict[str, float], output_dir: Path) -> None:
    """Save evaluation metrics to JSON and CSV.
    
    Args:
        metrics: Dictionary of metric names and values
        output_dir: Directory to save the metrics
    """
    # Save as JSON
    json_path = output_dir / "metrics.json"
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save as CSV
    csv_path = output_dir / "metrics.csv"
    df = pd.DataFrame([metrics])
    df.to_csv(csv_path, index=False)
    
    logger.info(f"Metrics saved to {json_path} and {csv_path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_dir: Path,
    title: str = "ROC Curve",
) -> None:
    """Plot and save ROC curve.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        output_dir: Directory to save the plot
        title: Plot title
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = output_dir / "roc_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"ROC curve saved to {plot_path}")


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    output_dir: Path,
    class_names: Optional[list] = None,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save confusion matrix.
    
    Args:
        confusion_matrix: 2x2 confusion matrix
        output_dir: Directory to save the plot
        class_names: Names of classes
        title: Plot title
    """
    if class_names is None:
        class_names = ['Cover', 'Stego']
    
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = confusion_matrix.max() / 2.
    for i, j in np.ndindex(confusion_matrix.shape):
        plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                horizontalalignment="center",
                color="white" if confusion_matrix[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / "confusion_matrix.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved to {plot_path}")


def save_training_log(
    log_data: Dict[str, Any],
    output_dir: Path,
    filename: str = "training_log.json",
) -> None:
    """Save training configuration and results.
    
    Args:
        log_data: Dictionary containing training information
        output_dir: Directory to save the log
        filename: Name of the log file
    """
    log_path = output_dir / filename
    
    # Add timestamp
    log_data['timestamp'] = datetime.now().isoformat()
    
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2, default=str)
    
    logger.info(f"Training log saved to {log_path}")


def create_experiment_summary(
    metrics: Dict[str, float],
    config: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Create a comprehensive experiment summary.
    
    Args:
        metrics: Evaluation metrics
        config: Experiment configuration
        output_dir: Directory to save the summary
    """
    summary = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'output_directory': str(output_dir),
        },
        'configuration': config,
        'results': metrics,
        'key_findings': {
            'best_metric': max(metrics.items(), key=lambda x: x[1]),
            'accuracy_threshold': 'Good' if metrics.get('accuracy', 0) > 0.8 else 'Needs improvement',
        }
    }
    
    # Save detailed summary
    summary_path = output_dir / "experiment_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Create simple text summary
    text_summary_path = output_dir / "summary.txt"
    with open(text_summary_path, 'w') as f:
        f.write("STEGANALYSIS EXPERIMENT SUMMARY\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Timestamp: {summary['experiment_info']['timestamp']}\n")
        f.write(f"Output Directory: {summary['experiment_info']['output_directory']}\n\n")
        
        f.write("CONFIGURATION:\n")
        f.write("-" * 20 + "\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nRESULTS:\n")
        f.write("-" * 20 + "\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    logger.info(f"Experiment summary saved to {summary_path} and {text_summary_path}")