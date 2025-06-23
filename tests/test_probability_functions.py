import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import math
import pytest

import probability_functions as pf


def test_logistic_sigmoid_basic():
    assert pf.logistic_sigmoid(0) == 0.5
    assert pf.logistic_sigmoid(1) == pytest.approx(1 / (1 + math.exp(-1)))


def test_softmax_basic():
    result = pf.softmax([1, 2, 3])
    exp_values = [math.exp(x - 3) for x in [1, 2, 3]]
    expected = [ev / sum(exp_values) for ev in exp_values]
    assert all(pytest.approx(r) == e for r, e in zip(result, expected))


def test_gaussian_pdf_basic():
    result = pf.gaussian_pdf(0, 0, 1)
    expected = 1 / math.sqrt(2 * math.pi)
    assert result == pytest.approx(expected)
