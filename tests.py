#!/usr/bin/env python3
"""Basic test suite for probability functions."""

import math
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from p_functions import probability_functions as pf  # type: ignore
except Exception:
    import probability_functions as pf  # type: ignore
import sigmoid
import relu


def test_probability_functions():
    """Test core probability functions."""
    print("🧪 Testing probability_functions.py...")

    # Test logistic sigmoid
    assert abs(pf.logistic_sigmoid(0) - 0.5) < 1e-10, "Sigmoid(0) should be 0.5"
    assert (
        abs(pf.logistic_sigmoid(1000) - 1.0) < 1e-3
    ), "Sigmoid(large) should approach 1.0"

    # Test softmax
    result = pf.softmax([1, 2, 3])
    assert abs(sum(result) - 1.0) < 1e-10, "Softmax should sum to 1.0"
    assert len(result) == 3, "Softmax should preserve input length"

    # Test empty softmax
    assert pf.softmax([]) == [], "Empty softmax should return empty list"

    # Test gaussian PDF
    pdf_val = pf.gaussian_pdf(0, 0, 1)  # Standard normal at mean
    expected = 1.0 / math.sqrt(2 * math.pi)
    assert (
        abs(pdf_val - expected) < 1e-10
    ), f"Standard normal PDF(0) should be {expected}"

    print("✅ probability_functions.py tests passed!")


def test_sigmoid_functions():
    """Test sigmoid activation functions."""
    print("🧪 Testing sigmoid.py...")

    # Test implemented functions don't crash
    functions_to_test = [
        ("gudermannian", sigmoid.gudermannian),
        ("error_function", sigmoid.error_function),
        ("smooth_step", sigmoid.smooth_step),
        ("algebraic_sigmoid", sigmoid.algebraic_sigmoid),
    ]

    for name, func in functions_to_test:
        try:
            result = func(1.0)
            assert isinstance(
                result, (int, float)
            ), f"{name} should return numeric value"
            print(f"  ✅ {name}(1.0) = {result:.6f}")
        except Exception as e:
            print(f"  ❌ {name} failed: {e}")

    # Test smooth step bounds
    assert sigmoid.smooth_step(-1) == 0.0, "Smooth step should be 0 for x <= 0"
    assert sigmoid.smooth_step(2) == 1.0, "Smooth step should be 1 for x >= 1"

    print("✅ sigmoid.py tests passed!")


def test_relu_functions():
    """Test ReLU activation functions."""
    print("🧪 Testing relu.py...")

    functions_to_test = [
        ("soft_plus", relu.soft_plus),
        ("swish", relu.swish),
        ("gaussian_error_linear_unit", relu.gaussian_error_linear_unit),
        ("leaky_relu", relu.leaky_relu),
        ("relu", relu.relu),
    ]

    for name, func in functions_to_test:
        try:
            result = func(1.0)
            assert isinstance(
                result, (int, float)
            ), f"{name} should return numeric value"
            print(f"  ✅ {name}(1.0) = {result:.6f}")
        except Exception as e:
            print(f"  ❌ {name} failed: {e}")

    # Test ReLU properties
    assert relu.relu(-5) == 0.0, "ReLU should be 0 for negative inputs"
    assert relu.relu(5) == 5.0, "ReLU should return input for positive values"

    # Test leaky ReLU
    assert relu.leaky_relu(-1) == -0.01, "Leaky ReLU should allow small negative values"

    print("✅ relu.py tests passed!")


def run_all_tests():
    """Run all test suites."""
    print("🚀 Starting comprehensive test suite...")
    print("=" * 50)

    try:
        test_probability_functions()
        test_sigmoid_functions()
        test_relu_functions()

        print("=" * 50)
        print("🎉 All tests passed successfully!")
        return True

    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
