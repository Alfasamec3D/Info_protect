"""Binary probit information and random-design variance at a fixed read budget.

Evaluator-only analysis of one Arbiter chain, known noise, independent reads.
Sampling with replacement is intentional: designs contain IID challenges.
"""
import argparse
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr
from scipy.stats import binom

from .cli import write_csv
from .experiment import stream
from .model import challenge_features, sample_weights


@dataclass(frozen=True)
class Design:
    seed: int = 930301
    bits: int = 64
    budget: int = 576
    trials: int = 4000
    sigma: float = .75
    repeats: tuple = (1,3,9)
    bootstrap: int = 2000


PROFILES={'main':Design(),'repeat':Design(seed=930302),
          'smoke':Design(bits=8,budget=72,trials=200,bootstrap=200)}


def coefficient(delays, sigma):
    """Scalar a(c) in I(c)=a(c) phi(c) phi(c)^T for a noisy sign bit."""
    if sigma<=0:
        raise ValueError('sigma must be positive')
    t=np.asarray(delays)/sigma
    return np.exp(-t*t-np.log(2*np.pi)-2*np.log(sigma)-log_ndtr(t)-log_ndtr(-t))


def majority_retention(probability, repeats):
    """Ratio of Fisher information about p after majority to all r IID bits."""
    p=np.asarray(probability,dtype=float)
    if repeats<1 or repeats%2!=1 or np.any((p<=0)|(p>=1)):
        raise ValueError('Need odd positive repeats and probabilities strictly between 0 and 1')
    threshold=repeats//2
    q=binom.sf(threshold,repeats,p)
    dq=repeats*binom.pmf(threshold,repeats-1,p)
    return dq*dq*p*(1-p)/(repeats*q*(1-q))


def generate(profile):
    if any(profile.budget%r for r in profile.repeats):
        raise ValueError('The budget must be divisible by every repeat count')
    weights=sample_weights(stream(profile.seed,130),1,profile.bits)[0]
    rng=stream(profile.seed,131)
    values={r:[] for r in profile.repeats}
    # Same IID challenge stream couples designs across r but not across trials.
    for start in range(0,profile.trials,20):
        size=min(20,profile.trials-start)
        raw=rng.integers(0,2,size=(size*profile.budget,profile.bits),dtype=np.uint8)
        information=coefficient(challenge_features(raw)@weights,profile.sigma).reshape(size,profile.budget)
        for r in profile.repeats:
            values[r].extend(r*information[:,:profile.budget//r].sum(axis=1))
    return {r:np.array(v) for r,v in values.items()}


def summarize(values, profile):
    base=values[1]
    rng=stream(profile.seed,132)
    # Paired resampling preserves common challenges across the repeat-count designs.
    mean_ratios={r:[] for r in profile.repeats}
    variance_ratios={r:[] for r in profile.repeats}
    for start in range(0,profile.bootstrap,100):
        draws=rng.integers(profile.trials,size=(min(100,profile.bootstrap-start),profile.trials))
        base_mean=base[draws].mean(axis=1)
        base_var=base[draws].var(axis=1,ddof=1)
        for r in profile.repeats:
            sample=values[r][draws]
            mean_ratios[r].extend(sample.mean(axis=1)/base_mean)
            variance_ratios[r].extend(sample.var(axis=1,ddof=1)/base_var)
    rows=[]
    for r in profile.repeats:
        row={'reads':r,'designs':profile.trials,'physical_queries':profile.budget,
             'challenge_draws':profile.budget//r,'parameter_count':profile.bits+1,
             'rank_upper_bound':min(profile.bits+1,profile.budget//r),
             'mean_information':float(values[r].mean()), 'variance_information':float(values[r].var(ddof=1)),
             'mean_ratio':float(values[r].mean()/base.mean()),
             'variance_ratio':float(values[r].var(ddof=1)/base.var(ddof=1)),
             'predicted_mean_ratio':1.,'predicted_variance_ratio':float(r)}
        for name,estimates in (('mean_ratio',mean_ratios[r]),('variance_ratio',variance_ratios[r])):
            low,high=np.quantile(estimates,[.025,.975])
            row.update({name+'_low':float(low),name+'_high':float(high)})
        rows.append(row)
    return rows


def run(profile, output):
    output.mkdir(parents=True,exist_ok=True)
    values=generate(profile)
    trials=[{'reads':r,'trial':i,'information_00':float(v)} for r in profile.repeats for i,v in enumerate(values[r])]
    rows=summarize(values,profile)
    write_csv(output/'trials.csv',trials)
    write_csv(output/'summary.csv',rows)
    retention=[{'reads':r,'probability_one':p,'retained_information':float(majority_retention(p,r))}
               for r in profile.repeats for p in (.1,.25,.5,.75,.9)]
    write_csv(output/'majority_retention.csv',retention)
    source=Path(__file__)
    manifest={**asdict(profile),'quantity':'J_00 of binary probit response, fixed device, random challenge design',
              'sampling':'IID uniform challenges with replacement; paired prefixes across r',
              'scope':'one chain; known sigma; independent read noise; evaluator-only',
              'source_sha256':{n:hashlib.sha256(source.with_name(n).read_bytes()).hexdigest()
                               for n in ('binary_information.py','model.py','experiment.py')},
              'numpy':np.__version__}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    plot(rows,output)
    print(f'Binary Fisher: {len(trials)} random designs; variance ratios '+', '.join(f"r={r['reads']}: {r['variance_ratio']:.3f}" for r in rows),flush=True)


def plot(rows,output):
    import matplotlib.pyplot as plt
    from .plots import save,PALETTE
    fig,axes=plt.subplots(1,2,figsize=(10.4,3.8),layout='constrained')
    x=[r['reads'] for r in rows]
    for ax,key,title in zip(axes,('mean_ratio','variance_ratio'),('Средняя условная информация','Разброс между дизайнами')):
        y=np.array([r[key] for r in rows]);lo=np.array([r[key+'_low'] for r in rows]);hi=np.array([r[key+'_high'] for r in rows])
        ax.errorbar(x,y,yerr=[np.maximum(0,y-lo),np.maximum(0,hi-y)],fmt='o',capsize=4,color=PALETTE[0],label='Monte Carlo, 95% ДИ')
        ax.plot(x,[r['predicted_'+key] for r in rows],'--',color=PALETTE[1],label='Тождество')
        ax.set(title=title,xlabel='Чтений одного запроса',ylabel='Отношение к одному чтению',xticks=x)
        ax.legend(frameon=False,fontsize=8)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.2)
    save(fig,output,'binary_information')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=PROFILES,default='main')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    run(PROFILES[args.profile],args.output)


if __name__=='__main__':
    main()
