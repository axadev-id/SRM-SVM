# Contributing Guidelines

Thank you for your interest in contributing to the SRM-SVM Steganalysis project!

## 📋 Quick Contribution Checklist

- [ ] Code follows PEP 8 style guidelines
- [ ] All functions have type hints
- [ ] New features include unit tests
- [ ] Documentation is updated if needed
- [ ] Code passes existing tests

## 🛠 Development Setup

1. **Fork and Clone**
   ```bash
   git clone your-fork-url
   cd SRM-SVM-project
   ```

2. **Create Development Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or venv\Scripts\activate  # Windows
   
   pip install -r requirements.txt
   pip install pytest mypy flake8 black
   ```

3. **Validate Setup**
   ```bash
   python setup_validation.py
   pytest tests/
   ```

## 🧪 Testing Guidelines

### Running Tests
```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_patches.py

# With coverage
pytest tests/ --cov=src/steganalysis --cov-report=html
```

### Writing Tests
- Place tests in `tests/` directory
- Follow naming convention: `test_*.py`
- Use descriptive test names: `test_extract_patches_with_valid_input`
- Include both positive and negative test cases

Example test structure:
```python
class TestNewFeature:
    def test_normal_case(self):
        # Test expected behavior
        pass
    
    def test_edge_case(self):
        # Test boundary conditions
        pass
    
    def test_error_handling(self):
        # Test error conditions
        with pytest.raises(ValueError):
            # Test code that should raise ValueError
            pass
```

## 📝 Code Style

### Type Hints
All functions must have type hints:
```python
def process_image(image: np.ndarray, size: int = 8) -> Optional[np.ndarray]:
    """Process image with specified size."""
    pass
```

### Documentation
Use clear docstrings with Args and Returns:
```python
def extract_features(data: np.ndarray) -> np.ndarray:
    """Extract features from input data.
    
    Args:
        data: Input data array of shape (n_samples, n_features)
        
    Returns:
        Extracted features of shape (n_samples, feature_dim)
        
    Raises:
        ValueError: If data is empty or has wrong dimensions
    """
    pass
```

### Code Formatting
- Use Black for automatic formatting: `black src/ tests/`
- Line length: 88 characters
- Use meaningful variable names
- Add comments for complex logic

### Linting
```bash
# Check style
flake8 src/ tests/

# Type checking
mypy src/

# Format code
black src/ tests/
```

## 🏗 Project Structure

```
src/steganalysis/
├── data/           # Dataset handling
├── features/       # Feature extraction
├── models/         # ML models
└── utils/          # Utilities

scripts/            # CLI scripts
tests/              # Unit tests
docs/               # Documentation (if added)
```

## 📊 Performance Considerations

### Memory Usage
- Be mindful of large arrays and datasets
- Use generators for large data processing
- Include memory usage estimates in docstrings

### Computational Efficiency
- Profile code for bottlenecks
- Use vectorized operations when possible
- Consider parallel processing for independent operations

## 🐛 Bug Reports

Include in your bug report:
1. **Environment**: Python version, OS, package versions
2. **Reproduction Steps**: Minimal code to reproduce the issue
3. **Expected vs Actual Behavior**
4. **Error Messages**: Full traceback if applicable
5. **Data Information**: Dataset size, image formats, etc.

## ✨ Feature Requests

For new features, please include:
1. **Use Case**: Why is this feature needed?
2. **Proposed Implementation**: High-level approach
3. **Backward Compatibility**: Impact on existing code
4. **Testing Strategy**: How will it be tested?

## 🔄 Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write code following style guidelines
   - Add/update tests
   - Update documentation

3. **Test Changes**
   ```bash
   pytest tests/
   flake8 src/ tests/
   mypy src/
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

5. **Submit Pull Request**
   - Clear title and description
   - Reference related issues
   - Include testing information

## 📚 Areas for Contribution

### High Priority
- [ ] Additional sparse coding solvers
- [ ] More feature aggregation methods
- [ ] Enhanced preprocessing options
- [ ] Performance optimizations
- [ ] Better error handling

### Medium Priority
- [ ] Additional evaluation metrics
- [ ] Visualization improvements  
- [ ] Configuration file support
- [ ] Multi-class steganalysis
- [ ] Advanced dataset augmentation

### Documentation
- [ ] Tutorial notebooks
- [ ] API documentation
- [ ] Performance benchmarks
- [ ] Advanced usage examples

## 🚀 Getting Started

Good first contributions:
1. Fix typos in documentation
2. Add unit tests for existing functions
3. Improve error messages
4. Add input validation
5. Create usage examples

## 📞 Questions?

- Check existing issues and discussions
- Create a new issue with the "question" label
- Be specific about your use case or problem

## 🙏 Recognition

Contributors will be acknowledged in:
- CHANGELOG.md
- README.md contributors section
- Release notes

Thank you for helping improve SRM-SVM Steganalysis!