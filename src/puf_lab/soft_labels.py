"""Compare empirical response fractions with majority labels on the same reads.

This extension consumes the completed query study; its physical query budget,
validation set, model architecture and initialization streams remain fixed.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
import csv
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits

from .attack import Profile, predict_product, sample_device
from .cli import write_csv
from .experiment import stream
from .model import read_error
from .query_study import repeated_noise

GROUP = ('chains', 'noise', 'budget', 'correlation', 'reads')
FIELDS = ('accuracy', 'clean_accuracy', 'expected_accuracy', 'gain_over_majority',
          'gain_over_single', 'expected_gain_over_majority', 'expected_gain_over_single')


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def soft_objective(flat, x, fractions, chains, penalty=1e-4):
    """BCE for P(response=1)=sigmoid(-product of estimated delays).

    Affine in fractions: exactly the average BCE of all repeated response bits.
    Putting fractional signs inside logaddexp would NOT have this property.
    """
    weights = flat.reshape(chains, x.shape[1])
    delays = x @ weights.T
    score = np.prod(delays, axis=1)
    loss = (np.logaddexp(0., score) - (1 - fractions) * score).mean()
    derivative = (expit(score) - (1 - fractions)) / len(x)
    gradient = np.empty_like(weights)
    for k in range(chains):
        others = np.prod(np.delete(delays, k, axis=1), axis=1)
        gradient[k] = x.T @ (derivative * others) + penalty * weights[k]
    return float(loss + penalty * np.sum(weights**2) / 2), gradient.ravel()


def fit_soft(x, fractions, valid_x, valid_y, chains, rng, restarts, maxiter):
    fractions = np.asarray(fractions, dtype=float)
    if fractions.shape != (len(x),) or not np.isfinite(fractions).all() or np.any((fractions < 0) | (fractions > 1)):
        raise ValueError('Response fractions must be a finite vector in [0, 1]')
    candidates, models = [], []
    for restart in range(restarts):
        initial = rng.normal(0, 1 / np.sqrt(x.shape[1]), (chains, x.shape[1]))
        fitted = minimize(soft_objective, initial.ravel(), args=(x, fractions, chains),
                          jac=True, method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-9})
        weights = fitted.x.reshape(chains, x.shape[1])
        models.append(weights)
        candidates.append({'restart': restart,
            'validation_accuracy': float(np.mean(predict_product(valid_x, weights) == valid_y)),
            'converged': bool(fitted.success), 'iterations': int(fitted.nit), 'loss': float(fitted.fun)})
    chosen = int(np.argmax([r['validation_accuracy'] for r in candidates]))
    for i, row in enumerate(candidates):
        row['selected'] = i == chosen
    return models[chosen], candidates


def summary(rows, profile):
    output = []
    for number, key in enumerate(sorted({tuple(r[k] for k in GROUP) for r in rows})):
        part = [r for r in rows if tuple(r[k] for k in GROUP) == key]
        draws = stream(profile.seed, 120, number).integers(len(part), size=(profile.bootstrap, len(part)))
        row = dict(zip(GROUP, key)) | {'devices': len(part), 'bootstrap_stream': number}
        for field in FIELDS:
            values = np.array([r[field] for r in part])
            lo, hi = np.quantile(values[draws].mean(axis=1), [.025, .975])
            row.update({field: float(values.mean()), field+'_low': float(lo), field+'_high': float(hi)})
        row.update({k: float(np.mean([r[k] for r in part])) for k in ('majority_accuracy', 'single_accuracy', 'fit_seconds')})
        row['devices_better_than_majority'] = sum(r['gain_over_majority'] > 0 for r in part)
        output.append(row)
    return output


def device_run(arguments):
    source, profile, device = arguments
    raw, x, weights, (_, va, te) = sample_device(profile, device)
    source_rows = read_csv(source / 'per_device.csv')
    part = [r for r in source_rows if int(r['device']) == device and r['kind'] == 'budget']
    rows, candidates, models, arrays = [], [], {}, {}
    arrays[f'd{device}_challenges'] = raw
    with threadpool_limits(limits=1), np.load(source / 'predictions.npz', allow_pickle=False) as baseline:
        np.testing.assert_array_equal(raw, baseline[f'd{device}_challenges'])
        for previous in part:
            k, r, budget = [int(previous[n]) for n in ('chains', 'reads', 'budget')]
            noise, rho = [float(previous[n]) for n in ('noise', 'correlation')]
            index = list(profile.noises).index(noise)
            count = budget // r
            delays = x @ weights[:k].T
            clean = np.logical_xor.reduce(delays < 0, axis=1)
            eps = repeated_noise(stream(profile.seed, 101, device, k, index), max(profile.budgets), k, rho)
            bits = np.logical_xor.reduce(delays[:count] + noise * eps[:r, :count] < 0, axis=-1)
            counts = bits.sum(axis=0).astype(np.uint8)
            fractions = counts / r
            key = previous['record_id']
            np.testing.assert_array_equal(fractions > .5, baseline[key+'_train_labels'])
            observed = np.logical_xor.reduce(delays + stream(profile.seed,100,device,k,index).normal(0,noise,delays.shape) < 0, axis=1)
            np.testing.assert_array_equal(observed[te], baseline[key+'_test_labels'])
            rng = stream(profile.seed, 102, device, k, index, budget, r)
            start = perf_counter()
            model, attempts = fit_soft(x[:count], fractions, x[va], observed[va], k, rng, profile.restarts, profile.maxiter)
            pred = predict_product(x[te], model)
            seconds = perf_counter() - start
            single = next(s for s in source_rows if s['device_id'] == previous['device_id']
                          and s['kind'] == 'benchmark' and s['model'] == 'product'
                          and int(s['chains']) == k and float(s['noise']) == noise and int(s['budget']) == budget)
            hard_pred = baseline[key+'_prediction']
            single_pred = baseline[single['record_id']+'_prediction']
            error = read_error(delays[te], noise)[:, -1]
            def expected(p):
                return float(np.where(p == clean[te], 1-error, error).mean())
            accuracy = float(np.mean(pred == observed[te]))
            row = {'record_id': key, 'device_id': previous['device_id'], 'device': device,
                   'chains': k, 'noise': noise, 'budget': budget, 'correlation': rho, 'reads': r,
                   'unique_challenges': count, 'validation_queries': profile.validation,
                   'total_queries': budget + profile.validation, 'accuracy': accuracy,
                   'clean_accuracy': float(np.mean(pred == clean[te])), 'expected_accuracy': expected(pred),
                   'majority_accuracy': float(previous['accuracy']), 'single_accuracy': float(single['accuracy']),
                   'gain_over_majority': accuracy - float(previous['accuracy']),
                   'gain_over_single': accuracy - float(single['accuracy']),
                   'expected_gain_over_majority': expected(pred)-expected(hard_pred),
                   'expected_gain_over_single': expected(pred)-expected(single_pred), 'fit_seconds': seconds}
            rows.append(row)
            candidates.extend({'record_id': key, **v} for v in attempts)
            models[key+'_weights'] = model
            arrays[key+'_prediction'] = pred
            arrays[key+'_response_counts'] = counts
    print(f'Soft labels seed {profile.seed}: device {device+1}/{profile.devices}', flush=True)
    return rows, candidates, models, arrays


def run(source, output, workers=1):
    manifest = json.loads((source / 'manifest.json').read_text())
    profile = Profile(**{f.name: manifest[f.name] for f in fields(Profile)})
    arguments = [(source, profile, i) for i in range(profile.devices)]
    rows, candidates, models, arrays = [], [], {}, {}
    if workers == 1:
        parts = map(device_run, arguments)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        parts = pool.map(device_run, arguments)
    try:
        for a, b, c, d in parts:
            rows.extend(a); candidates.extend(b); models.update(c); arrays.update(d)
    finally:
        if pool:
            pool.shutdown()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'per_device.csv', rows)
    write_csv(output / 'metrics.csv', summary(rows, profile))
    write_csv(output / 'validation.csv', candidates)
    np.savez_compressed(output / 'models.npz', **models)
    np.savez_compressed(output / 'predictions.npz', **arrays)
    fingerprint = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    details = {'profile': manifest, 'method': 'fractional BCE on the same final response bits',
               'selection': 'same starts and validation-accuracy selection as majority baseline',
               'source_sha256': {n: fingerprint(Path(__file__).with_name(n)) for n in
                   ('soft_labels.py','attack.py','query_study.py','model.py','experiment.py')},
               'reference_sha256': {n: fingerprint(source / n) for n in
                   ('manifest.json','per_device.csv','predictions.npz','models.npz')},
               'python': platform.python_version(), 'numpy': np.__version__, 'workers': workers,
               'expected_accuracy_scope': 'evaluator-only analytic expectation for one fresh noisy read at each fixed test challenge'}
    (output / 'manifest.json').write_text(json.dumps(details, indent=2)+'\n')
    plot(summary(rows, profile), output)


def plot(metrics, output):
    import matplotlib.pyplot as plt
    from .plots import save, PALETTE
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), layout='constrained')
    for ax, k in zip(axes, (1, 2)):
        selected = [r for r in metrics if r['chains'] == k]
        selected.sort(key=lambda r:(r['budget'],r['reads'],r['correlation']))
        for i, row in enumerate(selected):
            mean, lo, hi = [100*row['gain_over_majority'+s] for s in ('','_low','_high')]
            ax.errorbar(i, mean, yerr=[[max(0,mean-lo)],[max(0,hi-mean)]], fmt='o', capsize=4, color=PALETTE[0])
        ax.axhline(0, color='grey', linestyle='--', linewidth=1)
        ax.set(title=f'{k}-XOR, σ = 0,25', ylabel='Доля ответов минус большинство, п. п.',
               xticks=range(len(selected)), xticklabels=[f"{r['budget']}\nr={r['reads']}, ρ={r['correlation']}" for r in selected])
        ax.tick_params(axis='x', labelsize=7)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
    save(fig, output, 'soft_vs_majority')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('--workers must be positive')
    run(args.source, args.output, args.workers)


if __name__ == '__main__':
    main()
