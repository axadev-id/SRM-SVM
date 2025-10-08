#!/usr/bin/env python
"""Example training script with sample configuration."""

import subprocess
import sys

def run_training():
    """Run training with example configuration."""
    
    # Example command for training
    cmd = [
        sys.executable, "scripts/train.py",
        "--data-root", "dataset/BOSSBase 1.01 + 0.4 WOW/",
        "--cover-dir", "cover",
        "--stego-dir", "stego", 
        "--dict-size", "256",
        "--patch-size", "8",
        "--stride", "4",
        "--sparse-solver", "omp",
        "--n-nonzero-coefs", "5",
        "--max-patches", "200000",
        "--val-size", "0.2",
        "--test-size", "0.2",
        "--seed", "42",
        "--output-dir", "outputs"
    ]
    
    print("Running training with configuration:")
    print(" ".join(cmd))
    print()
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: Training script not found. Make sure you're in the project root directory.")
        sys.exit(1)


def run_inference():
    """Run inference example."""
    
    # Example command for inference (you'll need to update the model-dir)
    cmd = [
        sys.executable, "scripts/infer.py",
        "--image-dir", "dataset/BOSSBase 1.01 + 0.4 WOW/cover",  # Example: test on cover images
        "--model-dir", "outputs/20241007_143022/",  # Update with actual output directory
        "--output-file", "inference_results.csv",
        "--patch-size", "8",
        "--stride", "4",
        "--sparse-solver", "omp",
        "--n-nonzero-coefs", "5"
    ]
    
    print("Running inference with configuration:")
    print(" ".join(cmd))
    print()
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Inference failed with error code {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: Inference script not found. Make sure you're in the project root directory.")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Example SRM-SVM steganalysis")
    parser.add_argument("--mode", choices=["train", "infer"], default="train",
                       help="Mode to run (train or infer)")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        run_training()
    elif args.mode == "infer":
        run_inference()
    else:
        print("Please specify --mode train or --mode infer")
        sys.exit(1)