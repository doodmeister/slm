# SLM Development Guide

This guide covers the development aspects of the Simple Language Models (SLM) project.

## Development Setup

### Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd slm
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

### Development Dependencies

- **pytest**: Testing framework
- **pytest-cov**: Coverage reporting
- **ruff**: Modern linting and formatting (replaces black, isort, flake8)
- **mypy**: Type checking
- **pre-commit**: Git hooks

## Code Organization

```
slm/
├── __init__.py          # Package initialization
├── config.py            # Configuration management
├── exceptions.py        # Custom exceptions
├── utils.py             # Utility functions
├── logging_config.py    # Logging setup
├── cli.py              # Command-line interface
├── gui.py              # Graphical user interface
└── core/               # Core functionality
    ├── models.py       # Model definitions
    ├── data.py         # Data handling
    ├── trainer.py      # Training logic
    └── generator.py    # Text generation
```

## Coding Standards

### Style Guidelines

- Follow PEP 8 for Python code style
- Use type hints for all functions and methods
- Maximum line length: 88 characters (Black default)
- Use descriptive variable and function names
- Add docstrings to all public functions and classes

### Code Formatting and Linting

Use Ruff for comprehensive linting and formatting:
```bash
# Check and fix issues automatically
ruff check slm/ --fix

# Format code
ruff format slm/

# Run all checks
ruff check slm/ && ruff format --check slm/
```

### Type Checking

Run mypy for type checking:
```bash
mypy slm/
```

## Testing

### Running Tests

Run all tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=slm --cov-report=html
```

Run specific test files:
```bash
pytest tests/test_models.py
```

Run tests with specific markers:
```bash
pytest -m "not slow"  # Skip slow tests
pytest -m "integration"  # Run only integration tests
```

### Test Organization

- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete workflows

### Test Fixtures

Common fixtures are defined in `conftest.py`:
- `sample_text`: Standard text for testing
- `sample_vocab`: Standard vocabulary
- `temp_dir`: Temporary directory for file operations

### Mocking

Use `unittest.mock` for mocking external dependencies:
```python
from unittest.mock import Mock, patch

@patch('slm.utils.torch.cuda.is_available')
def test_device_selection(mock_cuda):
    mock_cuda.return_value = False
    # Test logic here
```

## Architecture Patterns

### SOLID Principles

The codebase follows SOLID principles:

1. **Single Responsibility**: Each class has one reason to change
2. **Open/Closed**: Open for extension, closed for modification
3. **Liskov Substitution**: Subtypes must be substitutable for base types
4. **Interface Segregation**: Clients shouldn't depend on unused interfaces
5. **Dependency Inversion**: Depend on abstractions, not concretions

### Design Patterns

- **Factory Pattern**: Used for model creation
- **Strategy Pattern**: Used for sampling methods
- **Observer Pattern**: Used for training callbacks
- **Template Method**: Used for base model structure

### Error Handling

- Use custom exceptions from `slm.exceptions`
- Provide detailed error messages with context
- Log errors at appropriate levels
- Handle errors gracefully in user interfaces

## Configuration Management

### Configuration Structure

Configuration is managed through Pydantic models:
- Type validation and conversion
- Documentation through field descriptions
- Serialization to/from YAML and JSON

### Adding New Configuration Options

1. Define the option in the appropriate config class
2. Add validation if necessary
3. Update default configuration file
4. Add tests for the new option

## Adding New Features

### Adding a New Model Architecture

1. Create the model class inheriting from `BaseModel`
2. Implement required methods (`forward`, `get_model_info`)
3. Add model type to `ModelType` enum
4. Update model factory function
5. Add comprehensive tests

### Adding New Sampling Methods

1. Create sampling strategy class inheriting from `SamplingStrategy`
2. Add method to `SamplingMethod` enum
3. Update generator factory method
4. Add tests and documentation

### Adding CLI Commands

1. Define command function with Click decorators
2. Add to CLI group in `cli.py`
3. Update help documentation
4. Add command tests

## Performance Considerations

### Memory Management

- Use context managers for resource cleanup
- Implement proper device management for GPU/CPU
- Monitor memory usage in training loops
- Use data loading optimizations

### Optimization Guidelines

- Profile code to identify bottlenecks
- Use appropriate batch sizes
- Implement gradient accumulation for large models
- Consider mixed precision training

## Documentation

### Docstring Format

Use Google-style docstrings:
```python
def train_model(data: str, epochs: int) -> Dict[str, Any]:
    """Train a language model.
    
    Args:
        data: Training data path.
        epochs: Number of training epochs.
        
    Returns:
        Dictionary containing training results.
        
    Raises:
        TrainingError: If training fails.
    """
```

### API Documentation

Generate API documentation using Sphinx:
```bash
sphinx-build -b html docs/ docs/_build/
```

## Git Workflow

### Branch Strategy

- `main`: Stable production code
- `develop`: Integration branch for features
- `feature/*`: Feature development branches
- `hotfix/*`: Critical fixes for production

### Commit Messages

Use conventional commits format:
```
type(scope): description

Body explaining the change (optional)

Fixes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

### Pre-commit Hooks

Install pre-commit hooks:
```bash
pre-commit install
```

Hooks run:
- Ruff linting and formatting
- mypy type checking
- Basic file checks

## Continuous Integration

### GitHub Actions

The project uses GitHub Actions for CI/CD:
- Run tests on multiple Python versions
- Check code formatting and linting
- Generate coverage reports
- Build and publish packages

### Quality Gates

- All tests must pass
- Coverage must be above 80%
- No linting errors
- Type checking must pass

## Debugging

### Common Issues

1. **CUDA out of memory**: Reduce batch size or model size
2. **Import errors**: Check virtual environment activation
3. **Configuration errors**: Validate YAML syntax
4. **Model loading failures**: Check checkpoint compatibility

### Debugging Tools

- Use `logger.debug()` for detailed logging
- Set `deterministic=True` for reproducible debugging
- Use smaller datasets for faster iteration
- Profile code with `cProfile` for performance issues

## Release Process

### Version Management

- Use semantic versioning (MAJOR.MINOR.PATCH)
- Update version in `pyproject.toml` and `__init__.py`
- Tag releases in Git

### Checklist

1. Update CHANGELOG.md
2. Run full test suite
3. Update documentation
4. Build and test package
5. Create GitHub release
6. Publish to PyPI (if applicable)

## Contributing

### Code Review Process

1. Create feature branch
2. Implement changes with tests
3. Update documentation
4. Submit pull request
5. Address review feedback
6. Merge after approval

### Guidelines for Contributors

- Follow existing code style
- Add tests for new functionality
- Update documentation
- Keep commits focused and atomic
- Write clear commit messages
