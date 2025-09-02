# 🧠 p_functions

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Comprehensive utilities for experimenting with probability-related functions and activation functions used in machine learning. This project provides implementations and benchmarking tools for various mathematical functions commonly used in neural networks and statistical computing.

## 📦 Features

### Probability Functions (`probability_functions.py`)
- **Logistic Sigmoid** - Binary classification activation with numerically stable implementation
- **Softmax** - Multi-class classification normalization with edge case handling
- **Gaussian PDF** - Normal distribution probability density with parameter validation

### Sigmoid Variants (`sigmoid.py`) - 10+ Implementations
- **Core**: Math, NumPy, SciPy sigmoid implementations with performance benchmarking
- **Mathematical Variants**: Hyperbolic tangent, arctangent, Gudermannian, error function
- **Advanced Functions**: Generalized logistic, smooth step, algebraic sigmoid
- **All functions** include proper type hints and comprehensive docstrings

### ReLU Family (`relu.py`) - 9+ Activation Functions
- **Standard**: ReLU, Leaky ReLU, Parametric ReLU with configurable parameters  
- **Modern Activations**: Swish, GELU, ELU, Softplus with sharpness control
- **Benchmarking**: Comprehensive timing comparisons for all implementations
- **Research Ready**: Implementations suitable for ML experimentation

## 🚀 Quick Start

### Interactive Interface
```bash
python interface.py
```

Or after install (editable or wheel):
```bash
p-functions
```

### Direct Usage
```python
# When installed as a package (src layout)
from p_functions import probability_functions as pf

# Sigmoid activation
result = pf.logistic_sigmoid(2.5)

# Softmax for multi-class
probs = pf.softmax([1.2, 0.8, 2.1])

# Gaussian PDF
density = pf.gaussian_pdf(1.5, mu=0, sigma=1)
```

### Benchmarking
```bash
# Compare sigmoid implementations
python sigmoid.py

# Compare ReLU variants  
python relu.py
```

## 🧪 Testing

Run the comprehensive test suite:
```bash
python tests.py
```

Or use pytest for unit tests:
```bash
pytest
```

## 📋 Requirements

- Python 3.11+
- NumPy 2.3.2+
- SciPy 1.16.0+

Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Function Reference

| Category | Functions | Features |
|----------|-----------|----------|
| **Core Probability** | sigmoid, softmax, gaussian_pdf | Input validation, edge case handling, numerical stability |
| **Sigmoid Variants** | 10+ implementations | Performance comparison, mathematical variants |
| **ReLU Family** | 9+ activation types | Modern ML activations, parameterized versions |

## 🆕 Recent Updates (v0.3.0)

- ✅ **Complete Implementation**: All 16 previously missing activation functions now implemented
- ✅ **Enhanced Interface**: Improved CLI with better error handling and user experience  
- ✅ **Dependencies Updated**: Latest stable NumPy (2.3.2) and SciPy (1.16.0)
- ✅ **Comprehensive Testing**: Full test coverage with mathematical validation
- ✅ **Documentation**: Professional README, CLAUDE.md for AI development
- ✅ **Project Infrastructure**: pyproject.toml, requirements.txt, multiple ignore files
- ✅ **Type Safety**: Complete type hints throughout codebase
- ✅ **Numerical Stability**: Improved implementations for edge cases

## 🛠️ Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
python tests.py

# Type checking
mypy . 

# Code formatting  
black .
isort .
```

## License

This project is licensed under the terms of the MIT License.
