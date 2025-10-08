# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2024-10-07

### Added
- Initial implementation of SRM-SVM steganalysis system
- Complete project structure with modular design
- Data loading and preprocessing with grayscale conversion
- Patch extraction with configurable size and stride
- Sparse dictionary learning using MiniBatchDictionaryLearning
- Sparse coding with OMP and Lasso solvers
- Feature aggregation with pooling and histogram statistics
- SVM classification with hyperparameter optimization
- Comprehensive evaluation with metrics and visualizations
- CLI interfaces for training and inference
- Unit tests for core functionality
- Setup validation and helper scripts
- Detailed documentation and examples

### Features
- **Data Handling**: 
  - Support for PNG, JPG, JPEG, BMP, TIFF formats
  - Automatic grayscale conversion
  - Optional SRM residual filtering
  - Stratified dataset splitting

- **Feature Extraction**:
  - 8×8 patch extraction with configurable stride
  - Dictionary learning with 256 atoms (configurable)
  - Sparse coding with OMP or Lasso solvers
  - Feature aggregation with multiple pooling strategies

- **Classification**:
  - SVM with RBF kernel
  - Automated hyperparameter search
  - Cross-validation with stratified splits
  - Class balancing options

- **Evaluation**:
  - Accuracy, Precision, Recall, F1-Score, ROC-AUC metrics
  - ROC curves and confusion matrix visualizations
  - Detailed classification reports
  - Comprehensive experiment logging

- **Usability**:
  - Command-line interfaces for training and inference
  - Batch processing support
  - Progress bars and detailed logging
  - Reproducible results with seed control
  - Setup validation scripts

### Documentation
- Complete README with usage examples
- Type hints throughout codebase
- Inline documentation for all modules
- Setup and validation guides
- Performance tuning recommendations

### Testing
- Unit tests for patch extraction
- Dictionary learning and sparse coding tests
- Dataset handling tests
- Import validation tests

### Configuration
- PEP8 compliant code formatting
- MyPy type checking support
- Flake8 linting configuration
- Project metadata in pyproject.toml