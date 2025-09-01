# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **p_functions**, a machine learning utility library focusing on probability and activation functions. The project consists of three main functional modules plus an interactive interface:

- **`probability_functions.py`** - Core ML probability functions (sigmoid, softmax, gaussian PDF)
- **`sigmoid.py`** - Multiple sigmoid implementations with performance benchmarking 
- **`relu.py`** - Complete ReLU activation function family with timing comparisons
- **`interface.py`** - Text-based CLI for function evaluation

The architecture is designed for both direct programmatic use and performance comparison through standalone benchmarking scripts.

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install with dev dependencies  
pip install -e ".[dev]"
```

### Testing
```bash
# Run all tests
python tests.py

# Test specific function families
python sigmoid.py    # Benchmark sigmoid variants
python relu.py       # Benchmark ReLU variants

# Or with pytest
pytest
```

### Code Quality
```bash
# Type checking
mypy .

# Code formatting
black .
isort .
```

### Running the Interface
```bash
# Interactive CLI
python interface.py

# Or via installed script
p-functions
```

## Architecture Notes

### Function Organization
- **Core functions** in `probability_functions.py` have comprehensive error handling and input validation
- **Benchmark modules** (`sigmoid.py`, `relu.py`) serve dual purposes:
  - Direct import for function access  
  - Standalone execution for performance timing
- **Test coverage** is assertion-based with mathematical validation (not pytest framework)

### Import Patterns  
Preference is to import from the package when installed:
- `from p_functions import probability_functions as pf`

When running from the repo root, direct module imports also work:
- `import probability_functions as pf`

When adding new functions:
- Core probability functions go in `probability_functions.py`
- Sigmoid variants and comparisons go in `sigmoid.py`
- ReLU family functions go in `relu.py`
- Update `interface.py` if CLI access is needed

### Performance Considerations
Both `sigmoid.py` and `relu.py` include `timeit` benchmarking in their `__main__` sections. When modifying functions, maintain the benchmarking structure to preserve performance comparison capabilities.

### Dependencies
- **NumPy/SciPy**: Required for `sigmoid.py` and `relu.py` mathematical functions
- **Python 3.11+**: Type hints use modern syntax
- **No pytest**: Uses custom test runner in `tests.py`

## Code Style Configuration

- **Line length**: 88 characters (Black standard)
- **Type checking**: MyPy with `disallow_untyped_defs=true`
- **Import sorting**: isort with Black profile
- **Python versions**: 3.11, 3.12