"""Integrate and reproduce the two additional studies from the Ivan branch.

The original scientific files remain unchanged. This adapter supplies parallel
execution, common output locations, aggregate statistics and public provenance.
Run from the repository root, as with puf_lab.reproduce.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import sys
import tempfile

import numpy as np
import scipy
from scipy.stats import t

METHODS = ('unweighted', 'count_likelihood', 'fixed5', 'single')


def load_stopping():
    root = Path.cwd()
    folder = root / 'research/stopping'
    for name, file in (('method', 'method.py'), ('ivan_original_run', 'run.py')):
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, folder / file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        else:
            assert Path(sys.modules[name].__file__).resolve() == (folder / file).resolve()
    return sys.modules['method'], sys.modules['ivan_original_run']


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interval(values, family=1):
    values = np.asarray(values, float)
    n = len(values)
    half = float(t.ppf(1 - .05 / (2 * family), n - 1) * values.std(ddof=1) / np.sqrt(n))
    mean = float(values.mean())
    return dict(n=n, difference=mean, ci_low=mean-half, ci_high=mean+half)


def summarize_stopping(rows, config):
    summary, contrasts = [], []
    family = len(config['sigmas']) * len(config['budgets']) * len(config['caps'])
    metrics = ('probability_mse', 'expected_nll', 'accuracy', 'expected_accuracy', 'queries', 'unique')
    for sigma in config['sigmas']:
        for budget in config['budgets']:
            for cap in config['caps']:
                cell = dict(sigma=sigma, budget=budget, cap=cap)
                selected = [r for r in rows if all(r[k] == v for k, v in cell.items())]
                by = {method: {r['seed']: r for r in selected if r['method'] == method} for method in METHODS}
                assert all(set(items) == set(config['seeds']) for items in by.values())
                for method, items in by.items():
                    summary.append(dict(**cell, method=method, devices=len(items),
                                        **{m: float(np.mean([v[m] for v in items.values()])) for m in metrics}))
                for a, b in (('count_likelihood', 'unweighted'), ('count_likelihood', 'single'),
                             ('count_likelihood', 'fixed5')):
                    for metric in metrics[:4]:
                        primary = (a, b, metric) == ('count_likelihood', 'unweighted', 'probability_mse')
                        delta = [by[a][seed][metric] - by[b][seed][metric] for seed in config['seeds']]
                        contrasts.append(dict(**cell, a=a, b=b, metric=metric, primary=primary,
                                              family_size=family if primary else 1,
                                              **interval(delta, family if primary else 1)))
    return summary, contrasts


def summarize_confirmation(rows, config):
    summary, contrasts = [], []
    strategies = [v['name'] for v in config['strategies']]
    by = {s: {r['seed']: r for r in rows if r['strategy'] == s} for s in strategies}
    assert all(set(items) == set(config['seeds']) for items in by.values())
    for strategy, items in by.items():
        summary.append(dict(strategy=strategy, devices=len(items),
                            **{metric: float(np.mean([v[metric] for v in items.values()]))
                               for metric in ('accuracy', 'noisy_accuracy', 'queries_used', 'train_unique')}))
    for b in ('majority_5', 'single'):
        delta = [by['soft_5'][seed]['accuracy'] - by[b][seed]['accuracy'] for seed in config['seeds']]
        contrasts.append(dict(a='soft_5', b=b, metric='accuracy', primary=True, family_size=2,
                              **interval(delta, 2)))
    return summary, contrasts


def configuration(study, profile):
    filename = 'stopping_confirmatory.json' if study == 'stopping' else 'confirmation.json'
    config = json.loads((Path('configs/ivan') / filename).read_text())
    if profile == 'smoke':
        config.update(n=16, seeds=config['seeds'][:2], budgets=[512], test_size=512)
        if study == 'stopping':
            config['sigmas'] = [.75]
        else:
            config['validation_queries'] = 64
    return config


def provenance():
    origin = json.loads(Path('configs/ivan/origin.json').read_text())
    for name, expected in origin['verbatim_files_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    # Also check the installed/imported package against the source provenance.
    import puf_attacks
    package = Path(puf_attacks.__file__).parent
    for file in Path('src/puf_attacks').glob('*.py'):
        assert file.read_bytes() == (package / file.name).read_bytes(), file
    return origin['commit']


def stopping_worker(payload):
    config, seed, destination = payload
    _, original = load_stopping()
    config = dict(config, seeds=[seed])
    with tempfile.TemporaryDirectory(prefix='puf-ivan-') as temp:
        temp = Path(temp)
        cfg = temp / 'config.json'
        cfg.write_text(json.dumps(config))
        original.run(cfg, temp / 'results')
        for model in (temp / 'results/models').glob('*.npz'):
            shutil.copy2(model, Path(destination) / 'models' / model.name)
        return [json.loads(line) for line in (temp / 'results/runs.jsonl').read_text().splitlines()]


def confirmation_worker(payload):
    from puf_attacks.experiments import run_one
    config, job, destination = payload
    return run_one(config, job, Path(destination))


def plot(study, output, summary, contrasts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    primary = [r for r in contrasts if r['primary']]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    scale = 100 if study == 'confirmation' else 1
    y = np.arange(len(primary))
    center = np.array([r['difference'] for r in primary]) * scale
    low = np.array([r['ci_low'] for r in primary]) * scale
    high = np.array([r['ci_high'] for r in primary]) * scale
    labels = ([f"σ={r['sigma']}; B={r['budget']}; M={r['cap']}" for r in primary]
              if study == 'stopping' else ['Доли − большинство', 'Доли − разные запросы'])
    ax.errorbar(center, y, xerr=[center-low, high-center], fmt='o', color='#007f7b', capsize=4)
    ax.set_yticks(y, labels)
    ax.axvline(0, linestyle='--', color='gray')
    ax.invert_yaxis()
    ax.set_xlabel('Разность MSE вероятностей; интервалы Bonferroni, семейный уровень 95%'
                  if study == 'stopping' else 'Разность точности по чистому ответу, п. п.; Bonferroni 95%')
    fig.tight_layout()
    fig.savefig(output / 'primary_contrasts.png', dpi=180)
    plt.close(fig)
    if study == 'stopping':
        method, _ = load_stopping()
        p = np.linspace(0, 1, 501)
        fig, ax = plt.subplots(figsize=(8, 4))
        for cap in (5, 15):
            ax.plot(p, method.stopped_mean(p, cap)-p, label=f'M={cap}')
        ax.axhline(0, color='gray', linewidth=.8)
        ax.set(xlabel='Истинная вероятность ответа +1', ylabel='Смещение E[S/N] − p')
        ax.legend(); fig.tight_layout()
        fig.savefig(output / 'stopping_bias.png', dpi=180); plt.close(fig)


def run(study, profile, output, workers):
    origin_commit = provenance()
    output = Path(output).resolve()
    if (output / 'manifest.json').exists():
        raise ValueError('Use a fresh output directory to avoid mixing experiments.')
    for name in ('models', 'traces') if study == 'confirmation' else ('models',):
        (output / name).mkdir(parents=True, exist_ok=True)
    config = configuration(study, profile)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        if study == 'stopping':
            futures = [pool.submit(stopping_worker, (config, seed, str(output))) for seed in config['seeds']]
        else:
            from puf_attacks.experiments import jobs
            futures = [pool.submit(confirmation_worker, (config, job, str(output))) for job in jobs(config)]
        for future in as_completed(futures):
            value = future.result()
            rows.extend(value if isinstance(value, list) else [value])
    if study == 'stopping':
        rows.sort(key=lambda r: (r['seed'], r['sigma'], r['budget'], r['cap'], METHODS.index(r['method'])))
        summary, contrasts = summarize_stopping(rows, config)
        _, original = load_stopping()
        signature = original.signature(config)
    else:
        from puf_attacks.experiments import fingerprint
        order = [v['name'] for v in config['strategies']]
        rows.sort(key=lambda r: (r['seed'], order.index(r['strategy'])))
        summary, contrasts = summarize_confirmation(rows, config)
        signature = fingerprint(config)
    for name, values in (('runs', rows), ('summary', summary), ('contrasts', contrasts)):
        write_csv(output / (name + '.csv'), values)
    (output / 'runs.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    manifest = dict(study=study, profile=profile, config=config, scientific_signature=signature,
                    origin_commit=origin_commit, python=platform.python_version(), numpy=np.__version__,
                    scipy=scipy.__version__, workers=workers,
                    adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    plot(study, output, summary, contrasts)
    print(f'Ivan {study}: {len(rows)} fitted models, {len(config["seeds"])} independent devices.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=('all', 'confirmation', 'stopping'), default='all')
    parser.add_argument('--profile', choices=('full', 'smoke'), default='full')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    for study in ('confirmation', 'stopping') if args.study == 'all' else (args.study,):
        run(study, args.profile, args.output / study, args.workers)


if __name__ == '__main__':
    main()
