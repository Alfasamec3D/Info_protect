"""Demonstrate the limits of a centered population prior on unseen devices."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from .cli import write_csv
from .experiment import stream
from .model import random_features, sample_weights
from .summary import interval


def population_one_probability(chains, offset, noise):
    per_chain = ndtr(-offset / math.hypot(1, noise))
    return float((1 - (1 - 2 * per_chain)**chains) / 2)


def run(seed, devices, inputs):
    bits, chains = 64, 4
    noise = 0.1
    offsets = (0.0, 0.6)
    features = random_features(stream(seed, 30), inputs, bits)
    readings = np.empty((len(offsets), devices, chains))
    for i in range(devices):
        weights = sample_weights(stream(seed, 31, i), chains, bits)
        noisy_delays = features @ weights.T
        noisy_delays += stream(seed, 32, i).normal(0, noise, size=noisy_delays.shape)
        for j, offset in enumerate(offsets):
            bits_out = np.logical_xor.accumulate(noisy_delays + offset < 0, axis=-1)
            readings[j, i] = bits_out.mean(axis=0)
    rng = stream(seed, 33)
    draws = rng.integers(devices, size=(4000, devices))
    rows = []
    for j, offset in enumerate(offsets):
        for k in range(1, chains + 1):
            rates = readings[j, :, k - 1]
            proportion = float(rates.mean())
            low, high = interval(rates[draws].mean(axis=1))
            theory = population_one_probability(k, offset, noise)
            rows.append({
                "chains": k, "mean_delay_offset": offset, "noise": noise,
                "fraction_ones": proportion, "ci_low": low, "ci_high": high,
                "theory_fraction_ones": theory,
                "theory_majority_accuracy": max(theory, 1 - theory),
            })
    metadata = {
        "seed": seed, "devices": devices, "inputs": inputs, "challenge_bits": bits,
        "bootstrap_resamples": 4000, "shared_challenge_set": True,
    }
    return rows, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=128238)
    parser.add_argument("--devices", type=int, default=192)
    parser.add_argument("--inputs", type=int, default=8192)
    args = parser.parse_args(argv)
    if args.seed < 0 or args.devices < 2 or args.inputs < 2:
        parser.error("require nonnegative seed and at least two devices and inputs")
    rows, manifest = run(args.seed, args.devices, args.inputs)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "unseen_device.csv", rows)
    (args.output / "unseen_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
