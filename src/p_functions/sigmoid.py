import math
import numpy
import scipy.stats
import scipy.special
import timeit

# Based on the equations from
# https://en.wikipedia.org/wiki/Sigmoid_function


def math_sig():
    x = 1
    math_sig_exp = 1 / (1 + math.exp(-x))
    return math_sig_exp


def numpy_sig():
    x = 1
    numpy_sig = 1 / (1 + numpy.exp(-x))
    return numpy_sig


def scipy_stats_logistics():
    x = 1
    scipy_stats_logistics_cdf = scipy.stats.logistic.cdf(x)
    return scipy_stats_logistics_cdf


def scipy_special():
    x = 1
    scipy_special_expit = scipy.special.expit(x)
    return scipy_special_expit


def hyperbolic_tangent_tanh():
    x = 1
    hyperbolic_tangent_th = math.tanh(x)
    return hyperbolic_tangent_th


def hyperbolic_tangent_exp():
    x = 1
    hyperbolic_tangent_e = (math.exp(x) - math.exp(-x)) / ((math.exp(x) + math.exp(-x)))
    return hyperbolic_tangent_e


def arc_tangent():
    x = 1
    at = math.atan(x)
    return at


def gudermannian(x: float = 1.0) -> float:
    """Gudermannian function: gd(x) = 2*arctan(tanh(x/2))"""
    return 2 * math.atan(math.tanh(x / 2))


def error_function(x: float = 1.0) -> float:
    """Error function approximation using tanh: erf(x) ≈ tanh(1.2*x)"""
    return math.tanh(1.2 * x)


def generalised_logistic_function(
    x: float = 1.0,
    A: float = 0.0,
    K: float = 1.0,
    B: float = 1.0,
    Q: float = 1.0,
    C: float = 1.0,
    M: float = 0.0,
) -> float:
    """Generalized logistic function: A + (K-A) / (C + Q*exp(-B*(x-M)))^(1/C)"""
    return A + (K - A) / ((C + Q * math.exp(-B * (x - M))) ** (1 / C))


def smooth_step(x: float = 1.0) -> float:
    """Smooth step function: 3x² - 2x³ for x ∈ [0,1], clamped elsewhere"""
    if x <= 0:
        return 0.0
    elif x >= 1:
        return 1.0
    return 3 * x * x - 2 * x * x * x


def algebraic_sigmoid(x: float = 1.0) -> float:
    """Algebraic sigmoid: x / sqrt(1 + x²)"""
    return x / math.sqrt(1 + x * x)


if __name__ == "__main__":
    print("Answer math.exp Module:", math_sig())
    print(timeit.timeit(math_sig, number=10000))
    print("Answer numpy_sig Module:", numpy_sig())
    print(timeit.timeit(numpy_sig, number=10000))
    print("Answer scipy.stats.logistics.cdf Module:", scipy_stats_logistics())
    print(timeit.timeit(scipy_stats_logistics, number=10000))
    print("Answer scipy.special.expit Module:", scipy_special())
    print(timeit.timeit(scipy_special, number=10000))
    print("Answer math.tanh Module:", hyperbolic_tangent_tanh())
    print(timeit.timeit(hyperbolic_tangent_tanh, number=10000))
    print("Answer math.tanh Module:", hyperbolic_tangent_exp())
    print(timeit.timeit(hyperbolic_tangent_exp, number=10000))
    print("Answer math.atan Module:", arc_tangent())
    print(timeit.timeit(arc_tangent, number=10000))
