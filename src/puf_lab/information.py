"""Gaussian-observation upper bound for correlated repeated PUF measurements.

This is not the Fisher information of the final binary/XOR response.
"""

import argparse
from pathlib import Path

import numpy as np

from .cli import write_csv


def gaussian_information(repeats, noise, correlation):
    if not isinstance(repeats, int) or repeats < 1 or noise <= 0 or not 0 <= correlation <= 1:
        raise ValueError("Need positive integer repeats, positive noise and correlation in [0, 1]")
    return repeats / (noise**2 * (1 + (repeats - 1) * correlation))


def verify():
    rows = []
    for repeats in (1, 3, 9):
        for rho in (0., .9, 1.):
            sigma = .25
            covariance = sigma**2 * ((1-rho)*np.eye(repeats)+rho*np.ones((repeats,repeats)))
            # Pseudoinverse also covers the perfectly correlated, rank-one limit.
            matrix_value = float(np.ones(repeats) @ np.linalg.pinv(covariance) @ np.ones(repeats))
            formula = gaussian_information(repeats, sigma, rho)
            np.testing.assert_allclose(matrix_value, formula, rtol=1e-12, atol=1e-12)
            rows.append({"reads": repeats, "correlation": rho, "noise": sigma,
                         "matrix_information": matrix_value, "formula_information": formula,
                         "fixed_budget_bound_ratio": 1/(1+(repeats-1)*rho)})
    return rows


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args=parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    rows=verify()
    write_csv(args.output / "information_bound.csv", rows)
    print(f"Verified {len(rows)} Gaussian information identities; binary accuracy ordering is not implied.")


if __name__ == "__main__":
    main()
