"""Run pre-specified stopped-sampling experiments; secrets stay in evaluator."""
import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import scipy
from scipy.special import log_ndtr
from puf_attacks.puf import ArbiterPUF, split_challenges, features
from method import Probit, BernoulliOracle, acquire


def signature(config):
    h=hashlib.sha256(json.dumps(config,sort_keys=True).encode())
    for p in [Path(__file__),Path(__file__).with_name('method.py'),ROOT/'src/puf_attacks/puf.py']:
        h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()


def run(config_path,out):
    config=json.loads(Path(config_path).read_text())
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest=dict(config=config,sha256=signature(config),numpy=np.__version__,scipy=scipy.__version__,
                  python=platform.python_version(),note='all queries charged; test probabilities evaluator-only')
    path=out/'manifest.json'
    if path.exists():
        assert json.loads(path.read_text())==manifest,'different source or environment: choose fresh output'
    else:path.write_text(json.dumps(manifest,indent=2)+'\n')
    path=out/'runs.jsonl'
    rows=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    completed={(r['seed'],r['sigma'],r['budget'],r['cap']) for r in rows if r['method']=='single'}
    for seed in config['seeds']:
        train,test=split_challenges(config['n'],[max(config['budgets']),config['test_size']],seed+1100000)
        x,tx=features(train),features(test)
        puf=ArbiterPUF.random(config['n'],1,seed)
        for sigma in config['sigmas']:
            p=puf.probability_positive(train,sigma)
            tp=puf.probability_positive(test,sigma)
            for budget in config['budgets']:
                for cap in config['caps']:
                    if (seed,sigma,budget,cap) in completed:continue
                    oracle=BernoulliOracle(p,seed+2200000)
                    s,n=acquire(oracle,budget,cap)
                    # Fixed-r control: both objectives must be identical (tested separately).
                    fo=BernoulliOracle(p,seed+3300000)
                    fs=np.array([fo.query(i,5) for i in range(budget//5)],float)
                    fn=np.full(len(fs),5.)
                    so=BernoulliOracle(p,seed+4400000)
                    ss=np.array([so.query(i,1) for i in range(budget)],float)
                    fits=[('unweighted',s,n,False),('count_likelihood',s,n,True),
                          ('fixed5',fs,fn,True),('single',ss,np.ones(budget),True)]
                    pending=[]
                    for method,successes,counts,weighted in fits:
                        start=time.perf_counter()
                        model=Probit.fit(x[:len(counts)],successes,counts,weighted)
                        q=model.predict(tx);z=tx@model.weights
                        nll=float(np.mean(-tp*log_ndtr(z)-(1-tp)*log_ndtr(-z)))
                        pending.append(dict(seed=seed,sigma=sigma,budget=budget,cap=cap,method=method,
                            queries=int(counts.sum()),unique=len(counts),converged=model.converged,
                            iterations=model.iterations,fit_seconds=time.perf_counter()-start,
                            probability_mse=float(np.mean((q-tp)**2)),expected_nll=nll,
                            accuracy=float(np.mean((q>=.5)==(tp>=.5))),
                            expected_accuracy=float(np.mean(np.where(q>=.5,tp,1-tp))),
                            mean_confidence=float(np.mean(np.abs(q-.5))),
                            truth_confidence=float(np.mean(np.abs(tp-.5)))))
                        key=f'{seed}_{sigma}_{budget}_{cap}_{method}'
                        (out/'models').mkdir(exist_ok=True)
                        np.savez_compressed(out/'models'/f'{key}.npz',weights=model.weights)
                    # Four models appended together: do not silently lose a partial group.
                    with path.open('a') as f:
                        f.write(''.join(json.dumps(r,allow_nan=False)+'\n' for r in pending))
                    rows.extend(pending)
                    print(seed,sigma,budget,cap,[(r['method'],round(r['probability_mse'],6)) for r in pending],flush=True)
    with (out/'runs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.config,args.output)
