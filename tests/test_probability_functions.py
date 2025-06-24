import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import math
import pytest
import probability_functions as pf


def test_logistic_sigmoid_zero():
    assert pf.logistic_sigmoid(0) == 0.5


def test_logistic_sigmoid_values():
    assert math.isclose(pf.logistic_sigmoid(1), 0.7310585786300049, rel_tol=1e-9)
    assert math.isclose(pf.logistic_sigmoid(-1), 0.2689414213699951, rel_tol=1e-9)


def test_logistic_sigmoid_large_negative():
    # Should not overflow and should be very close to 0
    result = pf.logistic_sigmoid(-1000)
    assert math.isclose(result, 0.0, abs_tol=1e-12)


def test_softmax_basic():
    data = [1.0, 2.0, 3.0]
    res = pf.softmax(data)
    exps = [math.exp(x - 3.0) for x in data]
    expected = [e / sum(exps) for e in exps]
    assert all(math.isclose(a, b, rel_tol=1e-9) for a, b in zip(res, expected))


def test_softmax_empty():
    assert pf.softmax([]) == []


def test_gaussian_pdf_standard_normal():
    result = pf.gaussian_pdf(0.0)
    expected = 1.0 / math.sqrt(2.0 * math.pi)
    assert math.isclose(result, expected, rel_tol=1e-9)


def test_gaussian_pdf_symmetry():
    assert math.isclose(
        pf.gaussian_pdf(0.5, mu=0.0, sigma=1.0),
        pf.gaussian_pdf(-0.5, mu=0.0, sigma=1.0),
        rel_tol=1e-9,
    )


def test_gaussian_pdf_invalid_sigma():
    with pytest.raises(ValueError):
        pf.gaussian_pdf(0.0, sigma=0)
