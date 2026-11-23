"""Recompute attack metrics from stored individual predictions and audit inputs."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from puf_lab.attack import MODELS, Profile, sample_device
from puf_lab.experiment import stream
from puf_lab.model import read_error


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def check(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    import puf_lab.attack as implementation
    for name, digest in manifest["source_sha256"].items():
        assert hashlib.sha256(Path(implementation.__file__).with_name(name).read_bytes()).hexdigest() == digest
    profile = Profile(**{name: manifest[name] for name in Profile.__dataclass_fields__})
    rows = read_csv(directory / "per_device.csv")
    metrics = read_csv(directory / "metrics.csv")
    audits = read_csv(directory / "partition_audit.csv")
    validation = read_csv(directory / "validation.csv")
    conditions = len(profile.chains) * len(profile.noises) * len(profile.budgets)
    assert len(rows) == profile.devices * conditions * len(MODELS)
    assert len(metrics) == conditions * len(MODELS)
    assert len(audits) == profile.devices * 3
    assert len(validation) == profile.devices * conditions * 2 * profile.restarts
    index = {(r["device_id"], int(r["chains"]), float(r["noise"]), int(r["train_crps"]), r["model"]): r
             for r in rows}
    assert len(index) == len(rows)
    audit_index = {(r["device_id"], r["split"]): r for r in audits}
    estimates = {}

    def close(actual, expected):
        np.testing.assert_allclose(float(actual), expected, rtol=0, atol=1e-12)

    with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
        for device in range(profile.devices):
            raw, x, weights, slices = sample_device(profile, device)
            ident = f"{profile.seed}:{device}"
            np.testing.assert_array_equal(saved[f"d{device}_challenges"], raw)
            packed = np.packbits(raw, axis=1)
            assert len({r.tobytes() for r in packed}) == len(raw), "Duplicate or overlapping challenges"
            for name, part in zip(("train", "validation", "test"), slices):
                row = audit_index[ident, name]
                assert int(row["count"]) == len(raw[part]) and row["overlap"] == "0"
                assert row["challenge_sha256"] == hashlib.sha256(raw[part].tobytes()).hexdigest()
            test_slice = slices[-1]
            for chains in profile.chains:
                delays = x @ weights[:chains].T
                clean = np.logical_xor.reduce(delays < 0, axis=1)[test_slice]
                for noise_no, noise in enumerate(profile.noises):
                    prefix = f"d{device}_k{chains}_n{noise_no}"
                    noisy = delays + stream(profile.seed, 92, device, chains, noise_no).normal(0, noise, delays.shape)
                    labels = np.logical_xor.reduce(noisy < 0, axis=1)
                    np.testing.assert_array_equal(saved[f"{prefix}_labels"], labels)
                    np.testing.assert_array_equal(saved[f"{prefix}_clean"], clean)
                    truth = labels[test_slice]
                    oracle = (1 - read_error(delays[test_slice], noise)[:, -1]).mean()
                    for budget in profile.budgets:
                        for model in MODELS:
                            row = index[ident, chains, noise, budget, model]
                            pred = saved[f"{prefix}_b{budget}_{model}"]
                            assert pred.shape == truth.shape and pred.dtype == np.bool_
                            if model == "majority":
                                np.testing.assert_array_equal(pred, np.full(len(truth), labels[:budget].mean() >= .5))
                            values = {"accuracy": (pred == truth).mean(), "clean_accuracy": (pred == clean).mean(),
                                      "balanced_accuracy": np.mean([(pred[truth == b] == b).mean()
                                                                    for b in (False, True) if np.any(truth == b)]),
                                      "oracle_accuracy": oracle}
                            for key, value in values.items():
                                close(row[key], value)
                            estimates.setdefault((chains, noise, budget, model), []).append(values)
                            if model in ("product", "shuffled"):
                                trials = [v for v in validation if (v["device_id"], int(v["chains"]), float(v["noise"]),
                                          int(v["train_crps"]), v["model"]) == (ident, chains, noise, budget, model)]
                                assert len(trials) == profile.restarts
                                assert all(np.isfinite(float(t["loss"])) for t in trials)
                                chosen = [t for t in trials if t["selected"] == "True"]
                                assert len(chosen) == 1
                                close(chosen[0]["validation_accuracy"], max(float(t["validation_accuracy"]) for t in trials))
    for row in metrics:
        key = int(row["chains"]), float(row["noise"]), int(row["train_crps"]), row["model"]
        values = estimates[key]
        accuracy = np.array([v["accuracy"] for v in values])
        baseline = np.array([v["accuracy"] for v in estimates[(*key[:3], "majority")]])
        draws = stream(profile.seed, 95, key[0], int(key[1] * 1000), key[2]).integers(
            profile.devices, size=(profile.bootstrap, profile.devices))
        for field in ("accuracy", "clean_accuracy", "balanced_accuracy", "oracle_accuracy"):
            close(row[field], np.mean([v[field] for v in values]))
        for field, data, lo, hi in (("accuracy", accuracy, "ci_low", "ci_high"),
                                    ("gain", accuracy - baseline, "gain_ci_low", "gain_ci_high")):
            close(row[field], data.mean())
            bounds = np.quantile(data[draws].mean(axis=1), [.025, .975])
            close(row[lo], bounds[0])
            close(row[hi], bounds[1])
    return {r["device_id"] for r in rows}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/attack"))
    args = parser.parse_args()
    first, second = check(args.results), check(args.results / "repeat")
    assert first.isdisjoint(second)
    print(f"Verified attack predictions, challenge partitions and intervals for {len(first | second)} devices.")
