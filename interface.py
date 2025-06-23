"""Simple text-based interface for evaluating probability functions."""

from typing import List

import probability_functions as pf


def get_float(prompt: str) -> float:
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Please enter a valid number.")


def main() -> None:
    print("Probability function evaluator")
    print("1. Logistic Sigmoid (binary classification)")
    print("2. Softmax (multi-class classification)")
    print("3. Gaussian PDF")
    choice = input("Select an option: ")

    if choice == "1":
        x = get_float("Enter a value: ")
        print("Result:", pf.logistic_sigmoid(x))
    elif choice == "2":
        values_str = input("Enter comma-separated values: ")
        try:
            values: List[float] = [float(v) for v in values_str.split(',') if v]
        except ValueError:
            print("Invalid input; please provide numbers separated by commas.")
            return
        print("Result:", pf.softmax(values))
    elif choice == "3":
        x = get_float("Value x: ")
        mu = get_float("Mean mu: ")
        sigma = get_float("Std sigma: ")
        print("Result:", pf.gaussian_pdf(x, mu, sigma))
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
