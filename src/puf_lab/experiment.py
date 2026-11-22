"""Paired simulation of readout reliability and finite measurement budgets."""

import numpy as np

from .model import (
    disagreement_probability,
    random_features,
    read_error,
    sample_weights,
)


def stream(seed, *indices):
    """Separate deterministic streams by experiment, device and condition."""
    return np.random.default_rng(np.random.SeedSequence([seed, *indices]))


def trial_estimates(probabilities, enrollment_reads, budget, rng):
    """Compute one BER estimate per strategy and trial.

    Rows are independent trials. Strategies share the first available
    challenges, the verification read and prefixes of the enrollment reads.
    """
    verification = rng.random(probabilities.shape) < probabilities
    errors = np.zeros(probabilities.shape, dtype=np.int64)
    previous = 0
    for repeats in enrollment_reads:
        count = budget // (repeats + 1)
        errors[:, :count] += rng.binomial(repeats - previous, probabilities[:, :count])
        majority_wrong = errors[:, :count] > repeats // 2
        yield repeats, np.mean(majority_wrong != verification[:, :count], axis=1)
        previous = repeats


def device_rows(config, device, weights):
    bits = config.challenge_bits
    reference = random_features(
        stream(config.seed, 1, device), config.reference_challenges, bits
    ) @ weights.T
    columns = np.array(config.chains) - 1
    reference_signs = np.logical_xor.accumulate(reference < 0, axis=1)[:, columns]
    uniformity = reference_signs.mean(axis=0)
    reference_errors = {
        noise: read_error(reference, noise)[:, columns] for noise in config.noise_levels
    }
    characterization, measurements = [], []

    for noise, probability in reference_errors.items():
        noisy_ones = np.where(reference_signs, 1 - probability, probability).mean(axis=0)
        for i, k in enumerate(config.chains):
            p = probability[:, i]
            characterization.append({
                "device": device, "chains": k, "noise": noise,
                "uniformity": float(uniformity[i]),
                "majority_accuracy": float(max(noisy_ones[i], 1 - noisy_ones[i])),
                "ber": float(p.mean()),
                "reference_se": float(p.std(ddof=1) / np.sqrt(p.size)),
            })

    for budget in config.budgets:
        count = budget // (config.enrollment_reads[0] + 1)
        trial_delays = random_features(
            stream(config.seed, 2, device, budget), count * config.trials, bits
        ) @ weights.T
        for j, (noise, probability) in enumerate(reference_errors.items()):
            if noise == 0:
                continue
            trials = read_error(trial_delays, noise)[:, columns]
            for i, k in enumerate(config.chains):
                p = probability[:, i]
                target = p.mean()
                rng = stream(config.seed, 3, device, budget, k, j)
                estimates = trial_estimates(
                    trials[:, i].reshape(config.trials, count),
                    config.enrollment_reads, budget, rng,
                )
                for repeats, estimated in estimates:
                    challenges = budget // (repeats + 1)
                    expected = disagreement_probability(p, repeats).mean()
                    bias = expected - target
                    variance = expected * (1 - expected) / challenges
                    errors = estimated - target
                    measurements.append({
                        "device": device, "chains": k, "noise": noise,
                        "budget": budget, "enrollment_reads": repeats,
                        "challenges": challenges,
                        "target_ber": float(target),
                        "simulated_bias": float(errors.mean()),
                        "simulated_mse": float(np.mean(errors**2)),
                        "analytic_bias": float(bias),
                        "analytic_variance": float(variance),
                        "analytic_mse": float(bias**2 + variance),
                    })
    return characterization, measurements


def run(config, on_device=None):
    config.validate()
    weights = [
        sample_weights(stream(config.seed, 0, i), max(config.chains), config.challenge_bits)
        for i in range(config.devices)
    ]
    common = random_features(
        stream(config.seed, 4), config.uniqueness_challenges, config.challenge_bits
    )
    columns = np.array(config.chains) - 1
    signatures = np.stack([
        np.logical_xor.accumulate(common @ matrix.T < 0, axis=1)[:, columns]
        for matrix in weights
    ])
    pairs = [
        {
            "pair": i // 2, "chains": k,
            "hamming": float(np.mean(signatures[i, :, j] != signatures[i + 1, :, j])),
        }
        for i in range(0, config.devices, 2) for j, k in enumerate(config.chains)
    ]
    characterization, budget_trials = [], []
    for i, matrix in enumerate(weights):
        device, measurements = device_rows(config, i, matrix)
        characterization.extend(device)
        budget_trials.extend(measurements)
        if on_device is not None:
            on_device(i + 1, config.devices)
    return characterization, pairs, budget_trials
