# Quick Start Guide

## 🚀 Rapid Setup (5 minutes)

### 1. Prerequisites
- Python 3.10+ 
- 2-8 GB RAM
- Basic command line knowledge

### 2. Installation
```bash
# Clone/download project to your folder
cd path/to/SRM-SVM-project

# Install dependencies
pip install -r requirements.txt

# Validate setup
python setup_validation.py
```

### 3. Prepare Dataset
Organize your images like this:
```
your_dataset/
├── cover/     # Clean images
│   ├── image1.png
│   └── image2.png
└── stego/     # Hidden message images  
    ├── image1.png
    └── image2.png
```

### 4. Train Model (Quick Test)
```bash
# Quick training (small model, fast)
python scripts/train.py \
  --data-root your_dataset \
  --dict-size 64 \
  --max-patches 10000 \
  --val-size 0.2 \
  --test-size 0.2

# Full training (better accuracy, slower)  
python scripts/train.py \
  --data-root your_dataset \
  --dict-size 256 \
  --max-patches 200000 \
  --apply-residual \
  --val-size 0.15 \
  --test-size 0.15
```

### 5. Test New Images
```bash
python scripts/infer.py \
  --image-dir path/to/new/images \
  --model-dir outputs/20241007_143022 \
  --output-file results.csv
```

### 6. View Results
- Check `outputs/TIMESTAMP/` for training results
- Open `results.csv` for predictions
- Training log: `outputs/TIMESTAMP/training.log`
- Accuracy plots: `outputs/TIMESTAMP/*.png`

## 🎯 Expected Results

**Good Performance Indicators:**
- Accuracy > 80%
- F1-Score > 0.75
- ROC-AUC > 0.85

**If Results Are Poor:**
- Increase `--dict-size` to 512 or 1024
- Add `--apply-residual` flag
- Reduce `--stride` to 2 or 3
- Increase `--max-patches`

## ⚡ Batch Helper (Windows)

Use the batch file for convenience:
```batch
# Setup everything
run.bat setup

# Create sample data and quick test
run.bat sample

# Train with your data
run.bat train "your_dataset_folder"

# Clean up files
run.bat clean
```

## 🔍 Troubleshooting

**Import Errors:**
```bash
python setup_validation.py
```

**Memory Issues:**
- Reduce `--max-patches` to 50000
- Use smaller `--dict-size` (128 or 64)

**Poor Accuracy:**
- Ensure balanced dataset (equal cover/stego images)
- Try different `--sparse-solver` (omp vs lasso)
- Enable `--apply-residual` for JPEG images

## 📊 Performance Guide

| Dataset Size | Dict Size | Max Patches | Training Time | Memory |
|--------------|-----------|-------------|---------------|--------|
| 1K images    | 64        | 10K         | 2-5 min       | 1 GB   |
| 10K images   | 256       | 100K        | 15-30 min     | 4 GB   |  
| 20K images   | 512       | 300K        | 60-120 min    | 8 GB   |

**Quick vs. Full Training:**
- **Quick**: `--dict-size 64 --max-patches 20000` (5-10 min, ~70-80% accuracy)
- **Full**: `--dict-size 512 --max-patches 300000` (1-2 hours, ~85-95% accuracy)