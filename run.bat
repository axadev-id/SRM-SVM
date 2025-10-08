@echo off
REM Batch script for SRM-SVM Steganalysis project

set PYTHON_CMD=python

if "%1"=="setup" (
    echo Setting up the project...
    %PYTHON_CMD% -m pip install -r requirements.txt
    %PYTHON_CMD% setup_validation.py
    goto end
)

if "%1"=="validate" (
    echo Validating setup...
    %PYTHON_CMD% setup_validation.py
    goto end
)

if "%1"=="test" (
    echo Running tests...
    %PYTHON_CMD% -m pytest tests/ -v
    goto end
)

if "%1"=="train" (
    if "%2"=="" (
        echo Usage: run.bat train [data_path]
        echo Example: run.bat train "sample_data"
        goto end
    )
    echo Training model with data from %2...
    %PYTHON_CMD% scripts/train.py --data-root "%2" --dict-size 128 --max-patches 50000 --val-size 0.2 --test-size 0.2
    goto end
)

if "%1"=="infer" (
    if "%3"=="" (
        echo Usage: run.bat infer [image_dir] [model_dir]
        echo Example: run.bat infer "sample_data/cover" "outputs/20241007_143022"
        goto end
    )
    echo Running inference...
    %PYTHON_CMD% scripts/infer.py --image-dir "%2" --model-dir "%3" --output-file predictions.csv
    goto end
)

if "%1"=="sample" (
    echo Creating sample dataset and training...
    %PYTHON_CMD% setup_validation.py
    %PYTHON_CMD% scripts/train.py --data-root "sample_data" --dict-size 64 --max-patches 10000 --val-size 0.3 --test-size 0.3
    goto end
)

if "%1"=="clean" (
    echo Cleaning up outputs...
    if exist "outputs" rmdir /s /q "outputs"
    if exist "sample_data" rmdir /s /q "sample_data"
    if exist "__pycache__" rmdir /s /q "__pycache__"
    if exist "src\steganalysis\__pycache__" rmdir /s /q "src\steganalysis\__pycache__"
    echo Cleanup complete.
    goto end
)

echo SRM-SVM Steganalysis Project Helper
echo Usage: run.bat [command]
echo.
echo Commands:
echo   setup     - Install dependencies and validate setup
echo   validate  - Validate project setup
echo   test      - Run unit tests
echo   train     - Train model (requires data path)
echo   infer     - Run inference (requires image dir and model dir)  
echo   sample    - Create sample data and train quick model
echo   clean     - Clean up generated files
echo.
echo Examples:
echo   run.bat setup
echo   run.bat train "dataset\BOSSBase 1.01 + 0.4 WOW"
echo   run.bat sample
echo   run.bat infer "new_images" "outputs\20241007_143022"

:end