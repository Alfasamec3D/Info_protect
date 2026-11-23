"""Independent-cohort study of information loss versus likelihood mismatch."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import scipy
from threadpoolctl import threadpool_limits

from .attack import Profile, sample_device
from .cli import write_csv
from .experiment import stream
from .observation_model import METHODS, fit, xor_logs

SOURCE_FILES = ('observation_model.py', 'observation_study.py', 'attack.py', 'model.py', 'experiment.py')


def verify_lock(protocol_path):
    lock_path = protocol_path.with_name('observation_protocol_lock.json')
    lock = json.loads(lock_path.read_text())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if digest(protocol_path) != lock['protocol_sha256']:
        raise ValueError('The frozen observation protocol has changed')
    if set(lock['scientific_source_sha256']) != set(SOURCE_FILES):
        raise ValueError('The protocol lock does not cover all scientific dependencies')
    for name, expected in lock['scientific_source_sha256'].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError(f'The frozen scientific implementation has changed: {name}')
    return digest(lock_path)


@dataclass(frozen=True)
class Design:
    seed: int
    devices: int = 16
    bits: int = 64
    budget: int = 4608
    reads: int = 9
    validation: int = 512
    test: int = 8192
    noise: float = .25
    restarts: int = 4
    penalties: tuple = (1e-5, 1e-4, 1e-3, 1e-2)
    maxiter: int = 1000
    bootstrap: int = 8000


def design(protocol, profile):
    cfg = Design(seed=protocol['seeds'].get(profile, 190001),
                 **{k: protocol[k] for k in Design.__dataclass_fields__ if k != 'seed'})
    if profile == 'smoke':
        return replace(cfg, seed=190000, devices=2, bits=16, budget=288,
                       validation=64, test=256, restarts=2, penalties=(1e-4, .01), maxiter=100, bootstrap=200)
    if profile == 'pilot':
        return replace(cfg, devices=2)
    return cfg


def cohort(cfg, device):
    profile = Profile(seed=cfg.seed, devices=cfg.devices, bits=cfg.bits, budgets=(cfg.budget,),
                      chains=(2,), validation=cfg.validation, test=cfg.test)
    raw, x, weights, (_, va, te) = sample_device(profile, device)
    delays = x @ weights.T
    # Separate split streams make training/selection independent of test-set size.
    labels = np.empty(len(x), dtype=bool)
    for split, part in enumerate((slice(0, cfg.budget), va, te)):
        noisy = delays[part] + stream(cfg.seed, 150, device, split).normal(0, cfg.noise, delays[part].shape)
        labels[part] = np.logical_xor.reduce(noisy < 0, axis=1)
    n = cfg.budget // cfg.reads
    bits = np.empty((cfg.reads, n), dtype=bool)
    bits[0] = labels[:n]
    noise = stream(cfg.seed, 151, device).normal(0, cfg.noise, (cfg.reads - 1, n, 2))
    bits[1:] = np.logical_xor.reduce(delays[None, :n] + noise < 0, axis=-1)
    a, b = xor_logs(x[te], weights / cfg.noise)
    return raw, x, labels, bits, va, te, np.exp(a), a > b


def evaluate(a, b, truth, true_probability, clean):
    probability = np.exp(a)
    pred = a > b
    return {'accuracy': float(np.mean(pred == truth)), 'clean_accuracy': float(np.mean(pred == clean)),
            'expected_accuracy': float(np.mean(np.where(pred, true_probability, 1 - true_probability))),
            'log_loss': float(-(truth * a + (1 - truth) * b).mean()),
            'brier': float(np.mean((probability - truth)**2))}


def device_run(args):
    cfg, device = args
    with threadpool_limits(limits=1):
        raw, x, labels, bits, va, te, true_probability, clean = cohort(cfg, device)
        initials = stream(cfg.seed, 152, device).normal(0, 1 / np.sqrt(x.shape[1]), (cfg.restarts, 2, x.shape[1]))
        prefix = f'd{device}'
        arrays = {prefix + '_challenges': raw, prefix + '_single_labels': labels,
                  prefix + '_repeat_bits': bits, prefix + '_true_probability': true_probability,
                  prefix + '_clean': clean}
        rows, trials, models = [], [], {}
        n = cfg.budget // cfg.reads
        fractions = bits.mean(axis=0)
        for method in METHODS:
            train_x = x[:cfg.budget] if method == 'single' else x[:n]
            target = labels[:cfg.budget] if method == 'single' else (fractions if method == 'fraction' else fractions > .5)
            start = perf_counter()
            chosen, weights, attempts = fit(train_x, target, x[va], labels[va], method, cfg.reads,
                                           initials, cfg.penalties, cfg.maxiter)
            a, b = xor_logs(x[te], weights[chosen])
            seconds = perf_counter() - start
            key = f'{prefix}_{method}'
            arrays[key + '_prediction'] = a > b
            arrays[key + '_probability'] = np.exp(a)
            for index, (w, trial) in enumerate(zip(weights, attempts)):
                models[f'{key}_candidate{index}'] = w
                trials.append({'record_id': key, 'candidate': index, **trial})
            best = attempts[chosen]
            rows.append({'record_id': key, 'device': device, 'device_id': f'{cfg.seed}:{device}',
                         'method': method, 'physical_training_queries': cfg.budget,
                         'unique_challenges': len(train_x), 'validation_queries': cfg.validation,
                         'total_queries': cfg.budget + cfg.validation, 'candidate': chosen,
                         'selected_penalty': best['penalty'], 'selected_restart': best['restart'],
                         'selected_converged': best['converged'], 'selected_iterations': best['iterations'],
                         'validation_log_loss': best['validation_log_loss'], 'fit_seconds': seconds,
                         **evaluate(a, b, labels[te], true_probability, clean)})
        baseline = np.full(len(clean), labels[:cfg.budget].mean() >= .5)
        arrays[prefix + '_constant'] = baseline
        audit = [{'device_id': f'{cfg.seed}:{device}', 'split': name, 'count': len(raw[part]),
                  'challenge_sha256': hashlib.sha256(raw[part].tobytes()).hexdigest()}
                 for name, part in (('train', slice(0, cfg.budget)), ('validation', va), ('test', te))]
    print(f'Observation study seed {cfg.seed}: device {device + 1}/{cfg.devices}', flush=True)
    return rows, trials, models, arrays, audit


def summarize(rows, cfg):
    metrics, contrasts = [], []
    draws = stream(cfg.seed, 153).integers(cfg.devices, size=(cfg.bootstrap, cfg.devices))
    fields = ('accuracy', 'clean_accuracy', 'expected_accuracy', 'log_loss', 'brier')
    groups = {method: [r for r in rows if r['method'] == method] for method in METHODS}
    for method, group in groups.items():
        row = {'method': method, 'devices': len(group), 'selected_converged': sum(r['selected_converged'] for r in group)}
        for field in fields:
            values = np.array([r[field] for r in group])
            lo, hi = np.quantile(values[draws].mean(axis=1), [.025, .975])
            row.update({field: float(values.mean()), field + '_low': float(lo), field + '_high': float(hi)})
        metrics.append(row)
    for first, second in (('fraction', 'majority_aware'), ('majority_aware', 'majority_naive'), ('fraction', 'single')):
        assert [r['device_id'] for r in groups[first]] == [r['device_id'] for r in groups[second]]
        for field in fields:
            values = np.array([a[field] - b[field] for a, b in zip(groups[first], groups[second])])
            lo, hi = np.quantile(values[draws].mean(axis=1), [.025, .975])
            contrasts.append({'first': first, 'second': second, 'endpoint': field,
                              'primary': first == 'fraction' and second == 'majority_aware' and field == 'accuracy',
                              'difference': float(values.mean()), 'ci_low': float(lo), 'ci_high': float(hi),
                              'devices': cfg.devices})
    return metrics, contrasts


def run(cfg, protocol_path, profile, output, workers):
    lock_hash = verify_lock(protocol_path) if profile in ('main', 'repeat') else None
    rows, trials, models, arrays, audit = [], [], {}, {}, []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for a, b, c, d, e in pool.map(device_run, [(cfg, i) for i in range(cfg.devices)]):
            rows.extend(a); trials.extend(b); models.update(c); arrays.update(d); audit.extend(e)
    metrics, contrasts = summarize(rows, cfg)
    output.mkdir(parents=True, exist_ok=True)
    for name, data in (('per_device', rows), ('validation', trials), ('metrics', metrics),
                       ('contrasts', contrasts), ('partition_audit', audit)):
        write_csv(output / (name + '.csv'), data)
    np.savez_compressed(output / 'models.npz', **models)
    np.savez_compressed(output / 'predictions.npz', **arrays)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {'profile': profile, 'design': asdict(cfg), 'protocol': json.loads(protocol_path.read_text()),
                'protocol_sha256': digest(protocol_path), 'workers': workers,
                'source_sha256': {n: digest(Path(__file__).with_name(n)) for n in SOURCE_FILES},
                'protocol_lock_sha256': lock_hash,
                'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                'evaluation_only': 'true_probability and clean use the simulator weights; never passed to fit'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    plot(contrasts, output)


def plot(contrasts, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [r for r in contrasts if r['endpoint'] == 'accuracy']
    center = np.array([r['difference'] for r in rows]) * 100
    low = np.array([r['ci_low'] for r in rows]) * 100
    high = np.array([r['ci_high'] for r in rows]) * 100
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.errorbar(center, np.arange(3), xerr=[center - low, high - center], fmt='o', color='#00756f', capsize=5)
    ax.set_yticks(np.arange(3), ['Доли − учёт большинства', 'Учёт большинства − наивная модель', 'Доли − разные запросы'])
    ax.axvline(0, color='#778899', linestyle='--'); ax.invert_yaxis()
    ax.set_xlabel('Разность точности, п. п.; парный 95% ДИ по устройствам')
    ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(output / 'observation_effects.png', dpi=180); plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=('main', 'repeat', 'pilot', 'smoke'), default='smoke')
    p.add_argument('--protocol', type=Path, default=Path('configs/observation_study.json'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=1)
    args = p.parse_args(argv)
    if args.workers < 1: p.error('--workers must be positive')
    protocol = json.loads(args.protocol.read_text())
    cfg = design(protocol, args.profile)
    if cfg.budget % cfg.reads: p.error('budget must be divisible by reads')
    run(cfg, args.protocol, args.profile, args.output, args.workers)


if __name__ == '__main__':
    main()
