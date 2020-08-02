import math
import timeit

#Based on the equation from
#https://en.wikipedia.org/wiki/Rectifier_(neural_networks)

def soft_plus():
    sp = 0
    return sp

def soft_plus_sharpness():
    sps = 0
    return sps

def swish():
    s = 0
    return s

def guassian_error_linear_unit():
    GELU = 0
    return GELU

def noisy_relu():
    nrelu = 0
    return nrelu

def leaky_relu():
    lr = 0
    return lr

def parametric_leaky_relu():
    plr = 0
    return plr

def exponential_linear_units():
    elu = 0
    return elu

def a():
    b = 0
    return b

if __name__ == "__main__":
    print("Answer for the softplu implementation:", soft_plus())
    print(timeit.timeit(soft_plus, number = 10000), "sec per loop")
    print("Answer for the soft_plus_sharpness implementation:", soft_plus_sharpness())
    print(timeit.timeit(soft_plus_sharpness, number=10000), "sec per loop")
    print("Answer for the swish implementation:", swish())
    print(timeit.timeit(swish, number=10000), "sec per loop")
    print("Answer for the guassian_error_linear_unit implementation:", guassian_error_linear_unit())
    print(timeit.timeit(guassian_error_linear_unit, number=10000), "sec per loop")
    print("Answer for the noisy_relu implementation:", noisy_relu())
    print(timeit.timeit(noisy_relu, number=10000), "sec per loop")
    print("Answer for the leaky_relu implementation:", leaky_relu())
    print(timeit.timeit(leaky_relu, number=10000), "sec per loop")
    print("Answer for the parametric_leaky_relu implementation:", parametric_leaky_relu())
    print(timeit.timeit(parametric_leaky_relu, number=10000), "sec per loop")
    print("Answer for the exponential_linear_units implementation:", exponential_linear_units())
    print(timeit.timeit(exponential_linear_units, number=10000), "sec per loop")
    print("Answer for the test:", a())
    print(timeit.timeit(a, number=10000), "sec per loop")
