"""Monte Carlo check of a derived fixed-query-budget Fisher-information identity.

The evaluator knows the simulator weights to calculate information. This is NOT
an attack, and no such information is supplied to any trained model.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from scipy.special import log_ndtr
from puf_attacks.puf import ArbiterPUF, features

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/fisher'
OUT.mkdir(exist_ok=True)
T,TRIALS,SIGMA=960,2000,.75
w=ArbiterPUF.random(64,1,2026).weights[0]
rows=[]
for r in [1,3,5]:
    rng=np.random.default_rng(8000+r)
    m=T//r
    for start in range(0,TRIALS,25):
        count=min(25,TRIALS-start)
        c=rng.choice(np.array([-1,1],dtype=np.int8),size=(count*m,64))
        z=(features(c)@w)/SIGMA
        # I_00(c) = gaussian_pdf(z)^2 / (sigma^2 Phi(z)Phi(-z)); phi_0^2=1.
        info=np.exp(-z*z-np.log(2*np.pi)-2*np.log(SIGMA)-log_ndtr(z)-log_ndtr(-z))
        total=r*info.reshape(count,m).sum(axis=1)
        rows.extend(dict(repeats=r,trial=start+i,queries=T,information=float(x))
                    for i,x in enumerate(total))
with (OUT/'trials.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows)
summary=[]
base=np.var([x['information'] for x in rows if x['repeats']==1],ddof=1)
for r in [1,3,5]:
    a=np.array([x['information'] for x in rows if x['repeats']==r])
    summary.append(dict(repeats=r,trials=len(a),mean=float(a.mean()),variance=float(a.var(ddof=1)),
                        variance_ratio=float(a.var(ddof=1)/base),predicted_ratio=r))
with (OUT/'summary.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(summary[0]))
    writer.writeheader();writer.writerows(summary)
(OUT/'manifest.json').write_text(json.dumps(dict(queries=T,trials=TRIALS,sigma=SIGMA,
    puf_source_sha256=hashlib.sha256((ROOT/'src/puf_attacks/puf.py').read_bytes()).hexdigest(),
    n=64,puf_seed=2026,sampling_seeds=[8001,8003,8005],numpy=np.__version__,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    note='evaluator-only information calculation; not an attack'),indent=2)+'\n')
print(json.dumps(summary,indent=2))
