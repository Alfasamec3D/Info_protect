"""Independently regenerate acquisition and audit all imported model predictions."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, log_ndtr
from scipy.stats import norm, t

from puf_attacks.puf import ArbiterPUF, split_challenges, features
from puf_attacks.experiments import fingerprint
from puf_lab.ivan_studies import load_stopping, provenance


def read(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def close(a, b):
    np.testing.assert_allclose(float(a), float(b), rtol=2e-11, atol=2e-11)


def stopped_counts(probabilities, seed, budget, cap):
    """Replay individual Bernoulli draws without using the acquisition code."""
    rng = np.random.default_rng(seed)
    successes, counts, used = [], [], 0
    for p in probabilities:
        if used + cap > budget:
            break
        first = rng.uniform(size=3) < p
        tail = rng.uniform(size=cap-3) < p if 0 < first.sum() < 3 else np.array([], bool)
        counts.append(3 + len(tail))
        successes.append(int(first.sum() + tail.sum()))
        used += counts[-1]
    assert 0 <= budget-used < cap
    return np.asarray(successes), np.asarray(counts)


def fixed_counts(probabilities, seed, budget, repeats):
    rng = np.random.default_rng(seed)
    return np.array([(rng.uniform(size=repeats) < p).sum() for p in probabilities[:budget//repeats]]), \
        np.full(budget//repeats, repeats)


def stopping_device(payload):
    root, config, seed, rows = payload
    root = Path(root)
    train, test = split_challenges(config['n'], [max(config['budgets']), config['test_size']], seed+1100000)
    assert len(np.unique(np.concatenate((train, test)), axis=0)) == len(train)+len(test)
    x, tx = features(train), features(test)
    weights = ArbiterPUF.random(config['n'], 1, seed).weights[0]
    for sigma in config['sigmas']:
        p, tp = norm.cdf(x @ weights / sigma), norm.cdf(tx @ weights / sigma)
        for budget in config['budgets']:
            for cap in config['caps']:
                s, n = stopped_counts(p, seed+2200000, budget, cap)
                assert np.all(np.isin(n, [3, cap]))
                fs, fn = fixed_counts(p, seed+3300000, budget, 5)
                ss, sn = fixed_counts(p, seed+4400000, budget, 1)
                for method, successes, counts in (('unweighted', s, n), ('count_likelihood', s, n),
                                                  ('fixed5', fs, fn), ('single', ss, sn)):
                    row = next(r for r in rows if (r['sigma'], r['budget'], r['cap'], r['method']) ==
                               (sigma, budget, cap, method))
                    assert row['queries'] == int(counts.sum()) and row['unique'] == len(counts)
                    assert row['queries'] <= budget
                    key = f'{seed}_{sigma}_{budget}_{cap}_{method}'
                    with np.load(root/'models'/(key+'.npz'), allow_pickle=False) as model:
                        w = model['weights']
                        assert model.files == ['weights'] and w.shape == (config['n']+1,) and np.isfinite(w).all()
                    z = tx @ w; q = norm.cdf(z)
                    expected = dict(probability_mse=np.mean((q-tp)**2),
                                    expected_nll=np.mean(-tp*norm.logcdf(z)-(1-tp)*norm.logsf(z)),
                                    accuracy=np.mean((q >= .5) == (tp >= .5)),
                                    expected_accuracy=np.mean(np.where(q >= .5, tp, 1-tp)),
                                    mean_confidence=np.mean(abs(q-.5)), truth_confidence=np.mean(abs(tp-.5)))
                    for metric, value in expected.items():
                        close(row[metric], value)
                    assert row['converged'] and 0 < row['iterations'] <= 300
                    # Check the gradient using a separately expressed probit score.
                    zt = x[:len(counts)] @ w
                    pos = successes/counts if method == 'unweighted' else successes
                    neg = 1-pos if method == 'unweighted' else counts-successes
                    dz = (-pos*np.exp(norm.logpdf(zt)-norm.logcdf(zt)) +
                          neg*np.exp(norm.logpdf(zt)-norm.logsf(zt))) / np.sum(pos+neg)
                    gradient = x[:len(counts)].T @ dz + 1e-6*w
                    assert np.max(abs(gradient)) < 2e-5, (key, np.max(abs(gradient)))
    return dict(seed=seed, train=len(train), test=len(test),
                challenge_sha256=hashlib.sha256(np.concatenate((train,test)).tobytes()).hexdigest())


def confirmation_device(payload):
    root, config, seed, rows = payload
    root = Path(root)
    pool, val, test = split_challenges(config['n'],
        [max(config['budgets'])-config['validation_queries'], config['validation_queries'], config['test_size']], seed+100000)
    joined = np.concatenate((pool, val, test))
    assert len(np.unique(joined, axis=0)) == len(joined)
    x, tx = features(pool), features(test)
    w = ArbiterPUF.random(config['n'], 1, seed).weights[0]
    p = norm.cdf(tx @ w/config['noise'][0]); clean = np.where(tx @ w >= 0, 1, -1)
    for row in rows:
        job = {k:row[k] for k in ('seed','k','budget','noise','strategy','repeats','target','model')}
        assert hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()[:20] == row['id']
        count = (row['budget']-row['validation_queries'])//row['repeats']
        assert row['train_unique'] == count
        assert row['queries_used'] == count*row['repeats']+row['validation_queries']
        assert 0 <= row['budget']-row['queries_used'] < row['repeats']
        assert row['model'] == 'lr' and row['k'] == 1 and row['optimizer_converged']
        with np.load(root/'models'/(row['id']+'.npz'), allow_pickle=False) as model:
            assert str(model['kind']) == 'lr' and int(model['k']) == 1
            params = model['params']
        pred = np.where(tx @ params >= 0, 1, -1)
        for metric, value in dict(accuracy=np.mean(pred == clean),
                noisy_accuracy=np.mean(np.where(pred == 1,p,1-p)),
                positive_prediction_rate=np.mean(pred == 1), test_positive_rate=np.mean(clean == 1)).items():
            close(row[metric], value)
        assert json.loads((root/'traces'/(row['id']+'.json')).read_text()) == []
        assert row['validation_selected_epochs'] == 0 and row['test_size'] == len(test)
        rng = np.random.default_rng(seed+200000)
        observed = (x[:count] @ w)[:,None] + rng.normal(scale=row['noise'], size=(count,row['repeats']))
        y = (observed >= 0).mean(axis=1)
        if row['target'] == 'majority': y = (y > .5).astype(float)
        penalty = params.copy(); penalty[-1] = 0
        gradient = x[:count].T @ (expit(x[:count] @ params)-y)/count + 1e-5*penalty
        assert np.max(abs(gradient)) < 1e-4, (row['id'], np.max(abs(gradient)))
    return dict(seed=seed, train=len(pool), validation=len(val), test=len(test),
                challenge_sha256=hashlib.sha256(joined.tobytes()).hexdigest())


def check(directory, study, workers=1, write_audit=False):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    config = manifest['config']
    if study == 'stopping':
        _, original = load_stopping()
        assert manifest.get('scientific_signature',manifest.get('sha256')) == original.signature(config)
        cells = len(config['sigmas'])*len(config['budgets'])*len(config['caps'])
        per_device = cells*4
    else:
        assert manifest.get('scientific_signature',manifest.get('fingerprint')) == fingerprint(config)
        per_device = 3
    rows = [json.loads(line) for line in (directory/'runs.jsonl').read_text().splitlines()]
    assert len(rows) == len(config['seeds'])*per_device
    csv_rows = read(directory/'runs.csv')
    assert len(csv_rows) == len(rows)
    for a,b in zip(rows,csv_rows):
        assert set(a) == set(b)
        for k,v in a.items():
            assert str(v) == b[k], (k,v,b[k])
    key_fields = ('seed','sigma','budget','cap','method') if study == 'stopping' else ('id',)
    assert len({tuple(r[k] for k in key_fields) for r in rows}) == len(rows)
    assert len(list((directory/'models').glob('*.npz'))) == len(rows)
    if study == 'confirmation': assert len(list((directory/'traces').glob('*.json'))) == len(rows)
    worker = stopping_device if study == 'stopping' else confirmation_device
    jobs = [(str(directory),config,seed,[r for r in rows if r['seed'] == seed]) for seed in config['seeds']]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        audit = list(pool.map(worker,jobs))
    if 'study' in manifest:
        # Recompute each interval directly, separately from the aggregation implementation.
        for r in read(directory/'contrasts.csv'):
            selected = rows
            if study == 'stopping':
                selected = [v for v in rows if all(v[k] == float(r[k]) for k in ('sigma','budget','cap'))]
            field = 'method' if study == 'stopping' else 'strategy'
            by = {m:{v['seed']:v for v in selected if v[field] == m} for m in (r['a'],r['b'])}
            delta = np.array([by[r['a']][seed][r['metric']]-by[r['b']][seed][r['metric']] for seed in config['seeds']])
            family = cells if study == 'stopping' and r['primary'] == 'True' else 2 if study == 'confirmation' else 1
            assert int(r['family_size']) == family and int(r['n']) == len(delta)
            half = t.isf(.05/(2*family),len(delta)-1)*np.std(delta,ddof=1)/np.sqrt(len(delta))
            for k,value in (('difference',delta.mean()),('ci_low',delta.mean()-half),('ci_high',delta.mean()+half)):
                close(r[k],value)
        for r in read(directory/'summary.csv'):
            field = 'method' if study == 'stopping' else 'strategy'
            selected = [v for v in rows if v[field] == r[field]]
            if study == 'stopping':
                selected = [v for v in selected if all(v[k] == float(r[k]) for k in ('sigma','budget','cap'))]
            assert len(selected) == int(r['devices'])
            for metric,value in r.items():
                if metric not in ('sigma','budget','cap','method','strategy','devices'):
                    close(value,np.mean([v[metric] for v in selected]))
    if write_audit:
        (directory/'audit.json').write_text(json.dumps(dict(status='passed',models=len(rows),
            devices=len(audit),scope='regenerated acquisition, saved weights, gradients, splits, metrics and familywise intervals',
            partitions=audit),indent=2)+'\n')
    print(f'Checked Ivan {study}: {len(rows)} models on {len(audit)} devices.',flush=True)
    return set(config['seeds'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path('results/ivan'))
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--original-branch',action='store_true')
    p.add_argument('--write-audit',action='store_true')
    a = p.parse_args()
    provenance()
    first = check(a.results/'confirmation','confirmation',a.workers,a.write_audit)
    second = check(a.results/('stopping_confirmatory' if a.original_branch else 'stopping'),
                   'stopping',a.workers,a.write_audit)
    assert first.isdisjoint(second)
