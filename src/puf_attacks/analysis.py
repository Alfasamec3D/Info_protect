"""Aggregate by independent PUF instances; paired comparisons, not pooled CRP CIs."""
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from .experiments import mean_ci, jobs


def summarize(directory):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/puf-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(directory)
    rows = [json.loads(s) for s in (out/'runs.jsonl').read_text().splitlines()]
    config = json.loads((out/'manifest.json').read_text())['config']
    groups = defaultdict(list)
    for row in rows:
        groups[(row['k'],row['budget'],row['noise'],row['strategy'],row['model'])].append(row)
    summary = []
    for key, group in sorted(groups.items()):
        row = dict(zip(['k','budget','noise','strategy','model'],key))
        row['instances'] = len(group)
        for metric in ['accuracy','noisy_accuracy','fit_seconds']:
            m,lo,hi = mean_ci([r[metric] for r in group])
            row.update({metric:m,metric+'_ci_low':lo,metric+'_ci_high':hi})
        summary.append(row)
    with (out/'summary.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    comparisons = []
    # Paired differences against single-query collection, matched by instance.
    for row in summary:
        if row['strategy'] == 'single':
            continue
        current = groups[(row['k'],row['budget'],row['noise'],row['strategy'],row['model'])]
        base = groups.get((row['k'],row['budget'],row['noise'],'single',row['model']),[])
        by_seed = {r['seed']:r for r in base}
        differences = [r['accuracy']-by_seed[r['seed']]['accuracy']
                       for r in current if r['seed'] in by_seed]
        if differences:
            m,lo,hi = mean_ci(differences)
            comparisons.append({k:row[k] for k in ['k','budget','noise','strategy','model']} |
                               dict(pairs=len(differences),difference=m,ci_low=lo,ci_high=hi))
    if comparisons:
        with (out/'paired.csv').open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
    for k in sorted({r['k'] for r in rows}):
        for noise in sorted({r['noise'] for r in rows}):
            fig, ax = plt.subplots(figsize=(7,4.5),layout='constrained')
            for strategy,model in sorted({(r['strategy'],r['model']) for r in rows}):
                points = sorted([r for r in summary if r['k']==k and r['noise']==noise
                                 and r['strategy']==strategy and r['model']==model],key=lambda r:r['budget'])
                if not points:
                    continue
                x = [r['budget'] for r in points]
                y = np.array([r['accuracy'] for r in points])
                errors = [0 if r['accuracy_ci_low'] is None else r['accuracy']-r['accuracy_ci_low'] for r in points]
                ax.errorbar(x,y,yerr=errors,marker='o',capsize=3,label=f'{model}/{strategy}')
            ax.set(xscale='log',xlabel='Total oracle query budget (including validation)',
                   ylabel='Agreement with noiseless PUF',title=f'{k}-XOR Arbiter PUF, noise sigma={noise}')
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
            fig.savefig(out/'figures'/f'k{k}_noise{noise}.png',dpi=180)
            fig.savefig(out/'figures'/f'k{k}_noise{noise}.pdf')
            plt.close(fig)
    metadata = dict(completed=len(rows),expected=len(list(jobs(config))),
                    lbfgs_all_converged=all(r['optimizer_converged'] for r in rows if r['model'] in {'lr','raw_lr'}),
                    fit_seconds=sum(r['fit_seconds'] for r in rows),
                    ci='Student t, 95%, across independent PUF instances; exploratory, unadjusted',
                    test='noiseless targets; noisy_accuracy is exact expected single-read agreement')
    (out/'analysis.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))
    return summary
