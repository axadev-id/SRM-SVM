@echo off
echo ==========================================
echo IMPROVED SRM-SVM STEGANALYSIS TRAINING
echo ==========================================
echo.

REM Change to project directory
cd /d "D:\kuliah\TA\SRM SVM"

echo Starting improved training with better parameters...
echo - Increased dictionary size: 128
echo - Larger patch size: 12x12
echo - More sparse coefficients: 8
echo - More training data: 10,000 patches per image
echo - Larger validation/test sets: 15% each
echo - RBF kernel SVM with higher regularization
echo - SPAM-686 SRM residual filters
echo.

python scripts/train.py ^
    --data-root "dataset/BOSSBase 1.01 + 0.4 WOW" ^
    --cover-dir cover ^
    --stego-dir stego ^
    --dict-size 128 ^
    --patch-size 12 ^
    --stride 8 ^
    --sparse-solver omp ^
    --n-nonzero-coefs 8 ^
    --alpha-coding 0.001 ^
    --max-patches 10000 ^
    --val-size 0.15 ^
    --test-size 0.15 ^
    --seed 42 ^
    --apply-residual ^
    --residual-type spam686

if %errorlevel% equ 0 (
    echo.
    echo ==========================================
    echo TRAINING COMPLETED SUCCESSFULLY!
    echo ==========================================
    echo Check the outputs folder for results.
) else (
    echo.
    echo ==========================================
    echo TRAINING FAILED!
    echo ==========================================
    echo Check the logs for error details.
)

pause