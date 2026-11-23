"""Audit every candidate, selected prediction, query budget and paired interval."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from puf_lab.experiment import stream
from puf_lab.model import challenge_features
from puf_lab.observation_model import METHODS, objective
from puf_lab.observation_study import Design, cohort, design, verify_lock
import puf_lab.observation_study as implementation
from puf_lab.soft_labels import read_csv


def probabilities(x, weights):
    """Independent construction: sum the four joint Gaussian sign outcomes."""
    z = x @ weights.T
    log0, log1 = norm.logcdf(z), norm.logsf(z)
    one = logsumexp(np.array([log0[:, 0] + log1[:, 1], log1[:, 0] + log0[:, 1]]), axis=0)
    zero = logsumexp(np.array([log0[:, 0] + log0[:, 1], log1[:, 0] + log1[:, 1]]), axis=0)
    return one, zero


def close(a, b):
    np.testing.assert_allclose(float(a), float(b), atol=2e-11, rtol=2e-11)


def check(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    cfg = Design(**manifest['design'])
    for name, value in manifest['source_sha256'].items():
        assert hashlib.sha256(Path(implementation.__file__).with_name(name).read_bytes()).hexdigest() == value
    protocol_path = Path('configs/observation_study.json')
    assert hashlib.sha256(protocol_path.read_bytes()).hexdigest() == manifest['protocol_sha256']
    assert json.loads(json.dumps(asdict(design(manifest['protocol'], manifest['profile'])))) == manifest['design']
    if manifest['profile'] in ('main', 'repeat'):
        assert verify_lock(protocol_path) == manifest['protocol_lock_sha256']
    rows = read_csv(directory / 'per_device.csv'); trials = read_csv(directory / 'validation.csv')
    audit = read_csv(directory / 'partition_audit.csv')
    assert len(rows) == cfg.devices * len(METHODS)
    assert len(trials) == len(rows) * len(cfg.penalties) * cfg.restarts
    assert len(audit) == cfg.devices * 3
    assert len({r['record_id'] for r in rows}) == len(rows)
    rebuilt = []
    with np.load(directory / 'predictions.npz', allow_pickle=False) as saved, \
         np.load(directory / 'models.npz', allow_pickle=False) as models:
        assert len(models.files) == len(trials)
        for device in range(cfg.devices):
            raw, x, labels, bits, va, te, true_probability, clean = cohort(cfg, device)
            prefix = f'd{device}'
            for name, expected in (('challenges', raw), ('single_labels', labels), ('repeat_bits', bits),
                                   ('true_probability', true_probability), ('clean', clean)):
                np.testing.assert_array_equal(saved[prefix + '_' + name], expected)
            assert len(np.unique(raw, axis=0)) == len(raw)
            np.testing.assert_array_equal(bits[0], labels[:cfg.budget // cfg.reads])
            for name, part in (('train', slice(0, cfg.budget)), ('validation', va), ('test', te)):
                row = next(r for r in audit if r['device_id'] == f'{cfg.seed}:{device}' and r['split'] == name)
                assert int(row['count']) == len(raw[part])
                assert row['challenge_sha256'] == hashlib.sha256(raw[part].tobytes()).hexdigest()
            for method in METHODS:
                key = f'{prefix}_{method}'
                row = next(r for r in rows if r['record_id'] == key)
                n = cfg.budget if method == 'single' else cfg.budget // cfg.reads
                assert int(row['physical_training_queries']) == cfg.budget
                assert int(row['unique_challenges']) == n
                assert int(row['validation_queries']) == cfg.validation
                assert int(row['total_queries']) == cfg.budget + cfg.validation
                attempts = [r for r in trials if r['record_id'] == key]
                assert [int(r['candidate']) for r in attempts] == list(range(len(attempts)))
                assert [(float(r['penalty']), int(r['restart'])) for r in attempts] == [
                    (p, s) for p in cfg.penalties for s in range(cfg.restarts)]
                target = labels[:n] if method == 'single' else (bits.mean(axis=0) if method == 'fraction' else bits.sum(axis=0) > cfg.reads // 2)
                for attempt in attempts:
                    w = models[f"{key}_candidate{attempt['candidate']}"]
                    assert w.shape == (2, cfg.bits + 1) and np.isfinite(w).all()
                    a, b = probabilities(x[va], w)
                    close(attempt['validation_log_loss'], -(labels[va] * a + (1 - labels[va]) * b).mean())
                    close(attempt['validation_accuracy'], np.mean((a > b) == labels[va]))
                    loss, grad = objective(w.ravel(), x[:n], target, method, cfg.reads, float(attempt['penalty']))
                    close(attempt['loss'], loss); close(attempt['gradient_norm'], np.linalg.norm(grad))
                chosen = int(np.argmin([float(r['validation_log_loss']) for r in attempts]))
                assert [r['selected'] for r in attempts] == [str(i == chosen) for i in range(len(attempts))]
                assert int(row['candidate']) == chosen
                best = attempts[chosen]
                assert row['selected_converged'] == best['converged']
                assert int(row['selected_iterations']) == int(best['iterations'])
                close(row['selected_penalty'], best['penalty']); close(row['selected_restart'], best['restart'])
                close(row['validation_log_loss'], best['validation_log_loss'])
                a, b = probabilities(x[te], models[f'{key}_candidate{chosen}'])
                probability = np.exp(a); pred = a > b
                np.testing.assert_array_equal(pred, saved[key + '_prediction'])
                np.testing.assert_allclose(probability, saved[key + '_probability'], atol=1e-14, rtol=1e-13)
                values = {'accuracy': np.mean(pred == labels[te]), 'clean_accuracy': np.mean(pred == clean),
                          'expected_accuracy': np.mean(np.where(pred, true_probability, 1 - true_probability)),
                          'log_loss': -(labels[te] * a + (1 - labels[te]) * b).mean(),
                          'brier': np.mean((probability - labels[te])**2)}
                assert values['expected_accuracy'] <= np.maximum(true_probability, 1 - true_probability).mean() + 1e-13
                for name, value in values.items(): close(row[name], value)
                rebuilt.append({'method': method, 'device': device, **values})
    draws = stream(cfg.seed, 153).integers(cfg.devices, size=(cfg.bootstrap, cfg.devices))
    metrics = read_csv(directory / 'metrics.csv'); assert len(metrics) == len(METHODS)
    fields = ('accuracy', 'clean_accuracy', 'expected_accuracy', 'log_loss', 'brier')
    groups = {m: [r for r in rebuilt if r['method'] == m] for m in METHODS}
    for row in metrics:
        group = groups[row['method']]
        assert int(row['devices']) == cfg.devices
        assert int(row['selected_converged']) == sum(r['selected_converged'] == 'True' for r in rows if r['method'] == row['method'])
        for field in fields:
            values = np.array([r[field] for r in group]); lo, hi = np.quantile(values[draws].mean(axis=1), [.025, .975])
            close(row[field], values.mean()); close(row[field + '_low'], lo); close(row[field + '_high'], hi)
    contrasts = read_csv(directory / 'contrasts.csv'); assert len(contrasts) == 15
    assert sum(r['primary'] == 'True' for r in contrasts) == 1
    for row in contrasts:
        a, b = groups[row['first']], groups[row['second']]
        assert [r['device'] for r in a] == [r['device'] for r in b]
        values = np.array([x[row['endpoint']] - y[row['endpoint']] for x, y in zip(a, b)])
        lo, hi = np.quantile(values[draws].mean(axis=1), [.025, .975])
        close(row['difference'], values.mean()); close(row['ci_low'], lo); close(row['ci_high'], hi)
        assert (row['primary'] == 'True') == (row['first'] == 'fraction' and row['second'] == 'majority_aware' and row['endpoint'] == 'accuracy')
    return {r['device_id'] for r in rows}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=Path('results/observation_study'))
    p.add_argument('--single', action='store_true')
    args = p.parse_args()
    first = check(args.results)
    second = set() if args.single else check(args.results / 'repeat')
    assert first.isdisjoint(second)
    print(f'Verified observation likelihoods, all candidates, budgets and paired effects on {len(first | second)} devices.')
