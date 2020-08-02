import math
import numpy
import scipy.stats
import scipy.special
import timeit

#Based on the equations from
#https://en.wikipedia.org/wiki/Sigmoid_function

def math_sig():
    x=1
    math_sig_exp = 1/(1+math.exp(-x))
    return math_sig_exp

def numpy_sig():
    x=1
    numpy_sig = 1 / (1 + numpy.exp(-x))
    return numpy_sig

def scipy_stats_logistics():
    x=1
    scipy_stats_logistics_cdf = scipy.stats.logistic.cdf(x)
    return scipy_stats_logistics_cdf

def scipy_special():
    x=1
    scipy_special_expit = scipy.special.expit(x)
    return scipy_special_expit

def hyperbolic_tangent_tanh():
    x=1
    hyperbolic_tangent_th = math.tanh(x)
    return hyperbolic_tangent_th

def hyperbolic_tangent_exp():
    x=1
    hyperbolic_tangent_e = (math.exp(x) - math.exp(-x)) / ((math.exp(x) + math.exp(-x)))
    return hyperbolic_tangent_e

def arc_tangent():
    x=1
    at = math.atan(x)
    return at

def gudermannian():
    gm = 0
    return gm

def error_function():
    ef = 0
    return ef

def generalised_logistic_function():
    glf = 0
    return glf

def smooth_step():
    st = 0
    return st

def algerbraic_function_a():
    afa = 0
    return afa

if __name__ == "__main__":
    print("Answer math.exp Module:", math_sig())
    print(timeit.timeit(math_sig, number = 10000))
    print("Answer numpy_sig Module:", numpy_sig())
    print(timeit.timeit(numpy_sig, number = 10000))
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