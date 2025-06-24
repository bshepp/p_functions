"""Common probability-related functions for ML scenarios."""

import math
from typing import Iterable, List


def logistic_sigmoid(x: float) -> float:
    """Return the logistic sigmoid of ``x``.

    The implementation is numerically stable for large ``|x|`` values.
    """
    if x >= 0:
        # Standard form is stable for non-negative ``x``
        return 1.0 / (1.0 + math.exp(-x))
    # For large negative ``x`` avoid overflow in ``exp(-x)`` by
    # rearranging the expression.
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def softmax(values: Iterable[float]) -> List[float]:
    """Return the softmax of an iterable of numbers."""
    vals = list(values)
    if not vals:
        return []
    max_val = max(vals)
    exps = [math.exp(v - max_val) for v in vals]
    sum_exp = sum(exps)
    return [e / sum_exp for e in exps]


def gaussian_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Return the value of the Gaussian probability density function.

    Parameters
    ----------
    x : float
        Point at which the PDF is evaluated.
    mu : float, optional
        Mean of the distribution. Defaults to ``0.0``.
    sigma : float, optional
        Standard deviation of the distribution. Must be positive.

    Raises
    ------
    ValueError
        If ``sigma`` is not strictly positive.
    """

    if sigma <= 0:
        raise ValueError("sigma must be positive")

    coeff = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    return coeff * math.exp(-0.5 * ((x - mu) / sigma) ** 2)
