"""Reproducible, resumable experiments with a strict separation of oracle and learner."""
import csv
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import t

from .models import Attack
from .puf import ArbiterPUF, split_challenges


def fingerprint(config):
    h = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for p in sorted(Path(__file__).parent.glob('*.py')):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def evaluate(model, test, clean, p_positive):
    pred = model.predict(test)
    return {
        'accuracy': float(np.mean(pred == clean)),
        'noisy_accuracy': float(np.mean(np.where(pred == 1, p_positive, 1-p_positive))),
        'positive_prediction_rate': float(np.mean(pred == 1)),
    }


def jobs(config):
    for seed in config['seeds']:
        for k in config['chains']:
            for budget in config['budgets']:
                for noise in config['noise']:
                    for strategy in config['strategies']:
                        for model in config['models']:
                            yield dict(seed=seed, k=k, budget=budget, noise=noise,
                                       strategy=strategy['name'], repeats=strategy['repeats'],
                                       target=strategy['target'], model=model)


def run_one(config, job, out):
    start = time.perf_counter()
    seed, k, budget = job['seed'], job['k'], job['budget']
    repeats, noise = job['repeats'], job['noise']
    validation_queries = config['validation_queries']
    count = (budget-validation_queries)//repeats
    if count < 1 or repeats % 2 != 1:
        raise ValueError('positive training size and odd repetitions required')
    # Same instance and disjoint challenge stream across methods/strategies.
    puf = ArbiterPUF.random(config['n'], k, seed=seed)
    max_count = max(config['budgets'])-validation_queries
    pool, val, test = split_challenges(config['n'],
                                     [max_count, validation_queries, config['test_size']],
                                     seed+100_000)
    train = pool[:count]
    observed = puf.response(train, noise, repeats, np.random.default_rng(seed+200_000))
    y = np.mean((observed+1)/2, axis=1)
    if job['target'] == 'majority':
        y = (y > .5).astype(float)
    val_y = (puf.response(val, noise, 1, np.random.default_rng(seed+300_000))[:, 0]+1)/2
    clean = puf.response(test)[:, 0]
    p = puf.probability_positive(test, noise)
    model = Attack(job['model'], k, seed+400_000, epochs=config['epochs'],
                   restarts=config['restarts'], width=config['width'])
    fit_start = time.perf_counter()
    model.fit(train, y, val, val_y)
    fit_seconds = time.perf_counter()-fit_start
    key = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()[:20]
    model.save(out/'models'/f'{key}.npz')
    (out/'traces'/f'{key}.json').write_text(json.dumps(model.trace, indent=2))
    return {
        **job, 'id': key, 'n': config['n'], 'train_unique':count,
        'queries_used':count*repeats+validation_queries,
        'validation_queries':validation_queries, 'test_size':len(test),
        **evaluate(model, test, clean, p),
        'test_positive_rate':float(np.mean(clean == 1)),
        'validation_selected_epochs':len(model.trace),
        'optimizer_converged':model.converged,
        'fit_seconds':fit_seconds, 'total_seconds':time.perf_counter()-start,
    }


def run(config_path, output, limit=None):
    config_path, out = Path(config_path), Path(output)
    config = json.loads(config_path.read_text())
    required = {'seeds','chains','budgets','noise','strategies','models','n',
                'validation_queries','test_size','epochs','restarts','width'}
    if not required <= config.keys():
        raise ValueError(f'missing config fields: {required-config.keys()}')
    for name in ['models','traces','figures']:
        (out/name).mkdir(parents=True,exist_ok=True)
    signature = fingerprint(config)
    manifest_path = out/'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())['fingerprint'] != signature:
            raise ValueError('code/config changed; use a fresh output directory')
    else:
        try:
            git = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        except (OSError,subprocess.CalledProcessError):
            git = None
        manifest = dict(fingerprint=signature, config=config, python=platform.python_version(),
                        numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform(),
                        git_head=git,
                        note='fingerprint hashes actual source files and config, including uncommitted source')
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    rows_path = out/'runs.jsonl'
    rows = [json.loads(s) for s in rows_path.read_text().splitlines()] if rows_path.exists() else []
    completed = {r['id'] for r in rows}
    all_jobs = list(jobs(config))
    done = 0
    for i, job in enumerate(all_jobs):
        key = hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()[:20]
        if key in completed:
            continue
        result = run_one(config,job,out)
        with rows_path.open('a') as f:
            f.write(json.dumps(result,allow_nan=False)+'\n')
        rows.append(result)
        print(f"[{i+1}/{len(all_jobs)}] k={job['k']} B={job['budget']} noise={job['noise']} "
              f"{job['strategy']} {job['model']} seed={job['seed']} "
              f"acc={result['accuracy']:.4f} fit={result['fit_seconds']:.2f}s", flush=True)
        done += 1
        if limit is not None and done >= limit:
            break
    if rows:
        with (out/'runs.csv').open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def mean_ci(values):
    a = np.asarray(values,float)
    mean = float(a.mean())
    if len(a) < 2:
        return mean, None, None
    half = float(t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)))
    return mean, mean-half, mean+half
