#!/usr/bin/env python
"""Monitor training progress and check results."""

import os
import json
from pathlib import Path
from datetime import datetime


def find_latest_output():
    """Find the latest output directory."""
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        print("❌ No outputs directory found")
        return None
    
    # Find all output directories
    output_dirs = [d for d in outputs_dir.iterdir() if d.is_dir()]
    if not output_dirs:
        print("❌ No output directories found")
        return None
    
    # Sort by creation time (newest first)
    latest_dir = sorted(output_dirs, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    return latest_dir


def check_training_status():
    """Check current training status."""
    latest_dir = find_latest_output()
    if not latest_dir:
        return
    
    print(f"🔍 Checking latest training: {latest_dir.name}")
    print("=" * 50)
    
    # Check what files exist
    files_status = {
        "training.log": "📝 Training log",
        "dictionary.npz": "📚 Sparse dictionary", 
        "classifier.joblib": "🤖 Trained classifier",
        "metrics.json": "📊 Performance metrics",
        "roc_curve.png": "📈 ROC curve plot",
        "confusion_matrix.png": "🔲 Confusion matrix",
        "experiment_summary.json": "📋 Full summary"
    }
    
    for filename, description in files_status.items():
        filepath = latest_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"✅ {description}: {size:,} bytes")
        else:
            print(f"⏳ {description}: Not yet created")
    
    print()
    
    # Check training log for progress
    log_file = latest_dir / "training.log"
    if log_file.exists():
        print("📝 Latest log entries:")
        print("-" * 30)
        with open(log_file, 'r') as f:
            lines = f.readlines()
            # Show last 10 lines
            for line in lines[-10:]:
                print(f"   {line.strip()}")
    
    # Check metrics if available
    metrics_file = latest_dir / "metrics.json"
    if metrics_file.exists():
        print("\n📊 Training Results:")
        print("-" * 30)
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            for metric, value in metrics.items():
                print(f"   {metric}: {value:.4f}")


def check_system_resources():
    """Check current system resource usage."""
    import psutil
    
    print("\n💻 System Resources:")
    print("-" * 30)
    print(f"   CPU Usage: {psutil.cpu_percent(interval=1):.1f}%")
    print(f"   Memory Usage: {psutil.virtual_memory().percent:.1f}%")
    print(f"   Available Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    
    # Check for Python processes
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
        try:
            if 'python' in proc.info['name'].lower():
                python_processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    if python_processes:
        print(f"\n🐍 Python Processes:")
        for proc in python_processes:
            print(f"   PID {proc['pid']}: CPU {proc['cpu_percent']:.1f}%, "
                  f"Memory {proc['memory_percent']:.1f}%")


def main():
    """Main monitoring function."""
    print("🔍 SRM-SVM Training Monitor")
    print("=" * 50)
    
    try:
        check_training_status()
        check_system_resources()
        
        print("\n" + "=" * 50)
        print("✅ Monitoring complete!")
        
    except Exception as e:
        print(f"❌ Error during monitoring: {e}")


if __name__ == "__main__":
    main()