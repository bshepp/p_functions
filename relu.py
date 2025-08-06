import math
import timeit

#Based on the equation from
#https://en.wikipedia.org/wiki/Rectifier_(neural_networks)

def soft_plus(x: float = 1.0) -> float:
    """Softplus function: ln(1 + exp(x))"""
    return math.log(1 + math.exp(x))

def soft_plus_sharpness(x: float = 1.0, beta: float = 1.0) -> float:
    """Softplus with sharpness parameter: (1/beta) * ln(1 + exp(beta*x))"""
    return (1 / beta) * math.log(1 + math.exp(beta * x))

def swish(x: float = 1.0) -> float:
    """Swish activation function: x * sigmoid(x) = x / (1 + exp(-x))"""
    return x / (1 + math.exp(-x))

def gaussian_error_linear_unit(x: float = 1.0) -> float:
    """GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))"""
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))

def noisy_relu(x: float = 1.0, noise: float = 0.0) -> float:
    """Noisy ReLU: max(0, x + N(0, σ)) - simplified without actual noise"""
    return max(0, x + noise)

def leaky_relu(x: float = 1.0, alpha: float = 0.01) -> float:
    """Leaky ReLU: max(αx, x) where α is small positive constant"""
    return max(alpha * x, x)

def parametric_relu(x: float = 1.0, alpha: float = 0.25) -> float:
    """Parametric ReLU: max(αx, x) where α is learnable parameter"""
    return max(alpha * x, x)

def exponential_linear_unit(x: float = 1.0, alpha: float = 1.0) -> float:
    """ELU: x if x > 0, else α(exp(x) - 1)"""
    return x if x > 0 else alpha * (math.exp(x) - 1)

def relu(x: float = 1.0) -> float:
    """Standard ReLU: max(0, x)"""
    return max(0, x)

if __name__ == "__main__":
    print("Answer for the softplus implementation:", soft_plus())
    print(timeit.timeit(soft_plus, number = 10000), "sec per loop")
    print("Answer for the soft_plus_sharpness implementation:", soft_plus_sharpness())
    print(timeit.timeit(soft_plus_sharpness, number=10000), "sec per loop")
    print("Answer for the swish implementation:", swish())
    print(timeit.timeit(swish, number=10000), "sec per loop")
    print("Answer for the gaussian_error_linear_unit implementation:", gaussian_error_linear_unit())
    print(timeit.timeit(gaussian_error_linear_unit, number=10000), "sec per loop")
    print("Answer for the noisy_relu implementation:", noisy_relu())
    print(timeit.timeit(noisy_relu, number=10000), "sec per loop")
    print("Answer for the leaky_relu implementation:", leaky_relu())
    print(timeit.timeit(leaky_relu, number=10000), "sec per loop")
    print("Answer for the parametric_relu implementation:", parametric_relu())
    print(timeit.timeit(parametric_relu, number=10000), "sec per loop")
    print("Answer for the exponential_linear_unit implementation:", exponential_linear_unit())
    print(timeit.timeit(exponential_linear_unit, number=10000), "sec per loop")
    print("Answer for the relu implementation:", relu())
    print(timeit.timeit(relu, number=10000), "sec per loop")
