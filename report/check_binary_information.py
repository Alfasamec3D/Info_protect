"""Regenerate binary information from Bernoulli probabilities and audit summaries."""
import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from scipy.stats import binom

from puf_lab.binary_information import Design
import puf_lab.binary_information as implementation
from puf_lab.experiment import stream
from puf_lab.model import challenge_features, sample_weights
from puf_lab.soft_labels import read_csv


def check(directory):
    manifest=json.loads((directory/'manifest.json').read_text())
    for name,digest in manifest['source_sha256'].items():
        assert hashlib.sha256(Path(implementation.__file__).with_name(name).read_bytes()).hexdigest()==digest
    cfg=Design(**{f.name:manifest[f.name] for f in fields(Design)})
    w=sample_weights(stream(cfg.seed,130),1,cfg.bits)[0]
    rng=stream(cfg.seed,131)
    rebuilt={r:[] for r in cfg.repeats}
    for start in range(0,cfg.trials,20):
        count=min(20,cfg.trials-start)
        raw=rng.integers(0,2,size=(count*cfg.budget,cfg.bits),dtype=np.uint8)
        t=(challenge_features(raw)@w)/cfg.sigma
        density_squared=np.exp(-t*t)/(2*np.pi)
        bernoulli_variance=ndtr(t)*ndtr(-t)
        info=(density_squared/(cfg.sigma**2*bernoulli_variance)).reshape(count,cfg.budget)
        for r in cfg.repeats:
            rebuilt[r].extend(r*info[:,:cfg.budget//r].sum(axis=1))
    saved=read_csv(directory/'trials.csv')
    assert len(saved)==cfg.trials*len(cfg.repeats)
    arrays={}
    for r in cfg.repeats:
        part=[row for row in saved if int(row['reads'])==r]
        assert [int(row['trial']) for row in part]==list(range(cfg.trials))
        arrays[r]=np.array([float(row['information_00']) for row in part])
        np.testing.assert_allclose(arrays[r],rebuilt[r],rtol=1e-13,atol=1e-12)
    summaries=read_csv(directory/'summary.csv')
    assert len(summaries)==len(cfg.repeats)
    estimates={r:{k:[] for k in ('mean_ratio','variance_ratio')} for r in cfg.repeats}
    rng=stream(cfg.seed,132)
    for start in range(0,cfg.bootstrap,100):
        draws=rng.integers(cfg.trials,size=(min(100,cfg.bootstrap-start),cfg.trials))
        for r in cfg.repeats:
            estimates[r]['mean_ratio'].extend(arrays[r][draws].mean(axis=1)/arrays[1][draws].mean(axis=1))
            estimates[r]['variance_ratio'].extend(arrays[r][draws].var(axis=1,ddof=1)/arrays[1][draws].var(axis=1,ddof=1))
    for row in summaries:
        r=int(row['reads']);a=arrays[r];base=arrays[1]
        expected={'designs':cfg.trials,'physical_queries':cfg.budget,'challenge_draws':cfg.budget//r,
                  'parameter_count':cfg.bits+1,'rank_upper_bound':min(cfg.bits+1,cfg.budget//r),
                  'mean_information':a.mean(),'variance_information':a.var(ddof=1),
                  'mean_ratio':a.mean()/base.mean(),'variance_ratio':a.var(ddof=1)/base.var(ddof=1),
                  'predicted_mean_ratio':1,'predicted_variance_ratio':r}
        for k in ('mean_ratio','variance_ratio'):
            lo,hi=np.quantile(estimates[r][k],[.025,.975])
            expected.update({k+'_low':lo,k+'_high':hi})
        for k,value in expected.items():
            np.testing.assert_allclose(float(row[k]),value,atol=1e-12,rtol=1e-12)
    retention=read_csv(directory/'majority_retention.csv')
    assert len(retention)==len(cfg.repeats)*5
    for row in retention:
        r=int(row['reads']);p=float(row['probability_one'])
        counts=np.arange(r+1)
        mass=binom.pmf(counts,r,p)
        scores=(counts-r*p)/(p*(1-p))
        # Conditional expected score for each majority outcome gives its information.
        q=mass[counts>r//2].sum()
        score_one=np.sum(mass[counts>r//2]*scores[counts>r//2])/q
        score_zero=np.sum(mass[counts<=r//2]*scores[counts<=r//2])/(1-q)
        ratio=(q*score_one**2+(1-q)*score_zero**2)/(r/(p*(1-p)))
        np.testing.assert_allclose(float(row['retained_information']),ratio,atol=1e-10,rtol=1e-10)
    return len(saved)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path('results/binary_information'))
    args=p.parse_args()
    count=check(args.results)+check(args.results/'repeat')
    print(f'Verified binary information on {count} random designs and exact majority-information checks.')
