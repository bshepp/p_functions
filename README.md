# p_functions

Utilities for experimenting with probability-related functions used in machine
learning. The repository now includes a simple text interface for evaluating
several common functions:

* Logistic sigmoid
* Softmax
* Gaussian probability density function (PDF)

## Usage

Run the interface with Python:

```bash
python interface.py
```

You will be prompted to choose a function and provide the required input values.
The program then prints the calculated probabilities.

## Running Tests

Unit tests for the probability functions are provided using `pytest`.
Run them from the repository root with:

```bash
pytest
```

## License

This project is licensed under the terms of the MIT License. See the
[LICENSE](LICENSE) file for details.
