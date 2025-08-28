"""Simple text-based interface for evaluating probability functions."""

from typing import List, Optional
import sys

# Support both "installed package" (p_functions.probability_functions)
# and "repo root execution" (plain probability_functions on PYTHONPATH)
try:  # pragma: no cover - simple import fallback
    from p_functions import probability_functions as pf  # type: ignore
except Exception:  # pragma: no cover
    import probability_functions as pf  # type: ignore


def get_float(prompt: str, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
    """Get a float input with optional validation bounds."""
    while True:
        try:
            value = float(input(prompt))
            if min_val is not None and value < min_val:
                print(f"Value must be >= {min_val}. Please try again.")
                continue
            if max_val is not None and value > max_val:
                print(f"Value must be <= {max_val}. Please try again.")
                continue
            return value
        except ValueError:
            print("Please enter a valid number.")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            sys.exit(0)


def main() -> None:
    """Main interface loop for probability function evaluation."""
    print("🧮 Probability Function Evaluator")
    print("=" * 35)
    print("1. Logistic Sigmoid (binary classification)")
    print("2. Softmax (multi-class classification)")
    print("3. Gaussian PDF (normal distribution)")
    print("4. Exit")
    print("=" * 35)
    
    try:
        choice = input("Select an option (1-4): ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")
        sys.exit(0)

    if choice == "1":
        x = get_float("Enter a value: ")
        result = pf.logistic_sigmoid(x)
        print(f"Sigmoid({x}) = {result:.6f}")
        print(f"Probability: {result*100:.2f}%")
    elif choice == "2":
        values_str = input("Enter comma-separated values: ")
        try:
            values: List[float] = [float(v.strip()) for v in values_str.split(',') if v.strip()]
            if not values:
                print("❌ No valid numbers provided.")
                return
            if len(values) == 1:
                print("⚠️  Note: Softmax with single value always returns [1.0]")
        except ValueError:
            print("❌ Invalid input; please provide numbers separated by commas.")
            return
        
        result = pf.softmax(values)
        print(f"Softmax({values}) = {[round(r, 6) for r in result]}")
        print(f"Sum check: {sum(result):.6f} (should be 1.0)")
    elif choice == "3":
        x = get_float("Value x: ")
        mu = get_float("Mean (mu): ")
        sigma = get_float("Standard deviation (sigma): ", min_val=0.001)  # Prevent division by zero
        result = pf.gaussian_pdf(x, mu, sigma)
        print(f"Gaussian PDF({x}, μ={mu}, σ={sigma}) = {result:.6f}")
    elif choice == "4":
        print("Goodbye! 👋")
        sys.exit(0)
    else:
        print("❌ Invalid choice. Please select 1-4.")


if __name__ == "__main__":
    main()
