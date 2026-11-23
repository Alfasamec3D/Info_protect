"""Verify physical-query accounting, saved models, predictions and paired intervals."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from puf_lab.attack import Profile, predict_product, sample_device
from puf_lab.experiment import stream
from puf_lab.model import read_error
from puf_lab.query_study import GROUP, repeated_noise, stored_mlp_predict


def read(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def check(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    import puf_lab.query_study as implementation
    assert hashlib.sha256(Path(implementation.__file__).read_bytes()).hexdigest() == manifest["source_sha256"]
    profile = Profile(**{k: manifest[k] for k in Profile.__dataclass_fields__})
    rows = read(directory / "per_device.csv")
    metrics = read(directory / "metrics.csv")
    validation = read(directory / "validation.csv")
    audits = read(directory / "partition_audit.csv")
    ids = {r["device_id"] for r in rows}
    assert len(ids) == profile.devices
    assert len(rows) == profile.devices * 48
    assert len({r["record_id"] for r in rows}) == len(rows)
    assert len(validation) == len(rows) * profile.restarts
    assert len(audits) == profile.devices * 3

    def close(actual, expected):
        np.testing.assert_allclose(float(actual), expected, atol=1e-12, rtol=0)

    with np.load(directory / "models.npz", allow_pickle=False) as models, np.load(
            directory / "predictions.npz", allow_pickle=False) as arrays:
        for device in range(profile.devices):
            raw, x, weights, (tr, va, te) = sample_device(profile, device)
            ident = f"{profile.seed}:{device}"
            np.testing.assert_array_equal(raw, arrays[f"d{device}_challenges"])
            assert len({r.tobytes() for r in np.packbits(raw, axis=1)}) == len(raw)
            for name, part in (("train", tr), ("validation", va), ("test", te)):
                item = next(a for a in audits if a["device_id"] == ident and a["split"] == name)
                assert item["sha256"] == hashlib.sha256(raw[part].tobytes()).hexdigest()
                assert int(item["count"]) == len(raw[part])
            for row in (r for r in rows if r["device_id"] == ident):
                record = row["record_id"]
                k, budget, repeats, count = (int(row[f]) for f in ("chains", "budget", "reads", "unique_challenges"))
                noise, rho = float(row["noise"]), float(row["correlation"])
                assert count * repeats == budget
                assert int(row["total_queries"]) == budget + profile.validation
                assert int(row["validation_queries"]) == profile.validation
                assert np.isfinite(float(row["fit_seconds"])) and float(row["fit_seconds"]) > 0
                delays = x @ weights[:k].T
                clean = np.logical_xor.reduce(delays < 0, axis=1)
                noise_no = list(profile.noises).index(noise)
                observed = np.logical_xor.reduce(delays + stream(profile.seed, 100, device, k, noise_no).normal(
                    0, noise, delays.shape) < 0, axis=1)
                eps = repeated_noise(stream(profile.seed, 101, device, k, noise_no), max(profile.budgets), k, rho)
                negative_count = np.count_nonzero(delays[:count] + noise * eps[:repeats, :count] < 0, axis=-1)
                labels = np.count_nonzero(negative_count % 2, axis=0) > repeats//2
                np.testing.assert_array_equal(labels, arrays[f"{record}_train_labels"])
                np.testing.assert_array_equal(observed[te], arrays[f"{record}_test_labels"])
                np.testing.assert_array_equal(clean[te], arrays[f"{record}_clean"])
                if row["model"] == "product":
                    def predict(features):
                        return predict_product(features, models[f"{record}_weights"])
                else:
                    def predict(features):
                        return stored_mlp_predict(features, [models[f"{record}_coef{i}"] for i in range(3)],
                                                  [models[f"{record}_bias{i}"] for i in range(3)], models[f"{record}_classes"])
                pred = predict(x[te])
                np.testing.assert_array_equal(pred, arrays[f"{record}_prediction"])
                close(row["accuracy"], (pred == observed[te]).mean())
                close(row["clean_accuracy"], (pred == clean[te]).mean())
                close(row["train_label_error"], (labels != clean[:count]).mean())
                if noise == 0:
                    np.testing.assert_array_equal(labels, clean[:count])
                close(row["majority_accuracy"], ((labels.mean() >= .5) == observed[te]).mean())
                close(row["oracle_accuracy"], (1-read_error(delays[te], noise)[:, -1]).mean())
                attempts = [v for v in validation if v["record_id"] == record]
                selected = [v for v in attempts if v["selected"] == "True"]
                assert len(selected) == 1 and len(attempts) == profile.restarts
                close(selected[0]["validation_accuracy"], (predict(x[va]) == observed[va]).mean())
                close(selected[0]["validation_accuracy"], max(float(v["validation_accuracy"]) for v in attempts))
                baseline = next(r for r in rows if r["device_id"] == ident and r["kind"] == "benchmark"
                                and r["chains"] == row["chains"] and r["noise"] == row["noise"]
                                and r["budget"] == row["budget"] and r["model"] == "product")
                close(row["gain_over_single"], float(row["accuracy"])-float(baseline["accuracy"]))
    assert len(metrics) == len({tuple(r[k] for k in GROUP) for r in rows})
    assert len({tuple(r[k] for k in GROUP) for r in metrics}) == len(metrics)
    for row in metrics:
        subset = [r for r in rows if all(r[k] == row[k] for k in GROUP)]
        assert len(subset) == profile.devices
        draws = stream(profile.seed, 106, int(row["bootstrap_stream"])).integers(
            len(subset), size=(profile.bootstrap, len(subset)))
        for field in ("accuracy", "clean_accuracy", "gain_over_single", "train_label_error"):
            values = np.array([float(r[field]) for r in subset])
            close(row[field], values.mean())
            lo, hi = np.quantile(values[draws].mean(axis=1), [.025, .975])
            close(row[field + "_low"], lo)
            close(row[field + "_high"], hi)
    contrasts = read(directory / "interactions.csv")
    assert len(contrasts) == 4 * len(profile.budgets)
    assert len({(r['chains'], r['budget'], r['reads']) for r in contrasts}) == len(contrasts)
    for row in contrasts:
        groups = [[r for r in rows if r["kind"] == "budget" and r["correlation"] == str(rho)
                   and all(r[k] == row[k] for k in ("chains", "budget", "reads"))] for rho in (0., .9)]
        assert [r["device_id"] for r in groups[0]] == [r["device_id"] for r in groups[1]]
        delta = np.array([float(b["accuracy"])-float(a["accuracy"]) for a,b in zip(*groups)])
        draws = stream(profile.seed,107,int(row["chains"]),int(row["budget"]),int(row["reads"])).integers(
            profile.devices, size=(profile.bootstrap,profile.devices))
        close(row["correlated_minus_iid"],delta.mean())
        lo,hi=np.quantile(delta[draws].mean(axis=1),[.025,.975])
        close(row["ci_low"],lo); close(row["ci_high"],hi)
    thresholds = read(directory / "thresholds.csv")
    assert len(thresholds) == profile.devices * len(profile.chains) * len(profile.noises) * 2
    assert len({(r['device_id'], r['chains'], r['noise'], r['model']) for r in thresholds}) == len(thresholds)
    for row in thresholds:
        subset=[r for r in rows if r["kind"] == "benchmark" and all(r[k] == row[k] for k in
                ("device_id", "chains", "noise", "model"))]
        reached=[int(r["total_queries"]) for r in subset if float(r["accuracy"]) >= .95]
        assert row["first_tested_total_queries_at_95"] == (str(min(reached)) if reached else "not_reached")
    return ids


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/query_study"))
    args = parser.parse_args()
    a,b=check(args.results),check(args.results/"repeat")
    assert a.isdisjoint(b)
    print(f"Verified query budgets, exported models, predictions and intervals on {len(a|b)} devices.")
