"""Post-hoc optimization/regularization control for the large soft-label effect.

Fixed condition: 2-XOR, sigma=.25, B=4608, r=9, independent read noise.
All alternatives share measured bits, initial random streams and validation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from .attack import Profile,objective,predict_product,sample_device
from .cli import write_csv
from .experiment import stream
from .soft_labels import read_csv,soft_objective

SETTINGS=(('published',2,180,1e-4),('iterations',2,1000,1e-4),
          ('restarts',8,1000,1e-4),('regularization',8,1000,1e-2))


def settings(profile):
    """Keep the reference fit identical to the source, including smoke profiles."""
    return (('published',profile.restarts,profile.maxiter,1e-4),*SETTINGS[1:])


def fit(x,targets,vx,vy,kind,rng,restarts,maxiter,penalty):
    candidates,models=[],[]
    for restart in range(restarts):
        initial=rng.normal(0,1/np.sqrt(x.shape[1]),(2,x.shape[1]))
        fun=objective if kind=='majority' else soft_objective
        labels=1-2*targets.astype(float) if kind=='majority' else targets
        result=minimize(fun,initial.ravel(),args=(x,labels,2,penalty),jac=True,method='L-BFGS-B',
                        options={'maxiter':maxiter,'ftol':1e-9})
        weights=result.x.reshape(2,x.shape[1])
        models.append(weights)
        candidates.append({'restart':restart,'validation_accuracy':float(np.mean(predict_product(vx,weights)==vy)),
                           'iterations':int(result.nit),'converged':bool(result.success),
                           'loss':float(result.fun),'gradient_norm':float(np.linalg.norm(result.jac))})
    chosen=int(np.argmax([r['validation_accuracy'] for r in candidates]))
    for i,row in enumerate(candidates):row['selected']=i==chosen
    return models[chosen],candidates


def cohort(source,soft_source,device):
    manifest=json.loads((source/'manifest.json').read_text())
    profile=Profile(**{f.name:manifest[f.name] for f in fields(Profile)})
    rows=read_csv(source/'per_device.csv')
    # Smoke source uses its smaller first budget with the same strategy.
    budget=min(profile.budgets)
    row=next(r for r in rows if int(r['device'])==device and r['kind']=='budget' and r['chains']=='2'
             and float(r['noise'])==.25 and int(r['budget'])==budget and r['reads']=='9' and float(r['correlation'])==0.)
    key=row['record_id']
    raw,x,w,(_,va,te)=sample_device(profile,device)
    delays=x@w[:2].T
    noise_no=list(profile.noises).index(.25)
    noisy=delays+stream(profile.seed,100,device,2,noise_no).normal(0,.25,delays.shape)
    labels=(np.count_nonzero(noisy<0,axis=1)%2).astype(bool)
    with np.load(soft_source/'predictions.npz',allow_pickle=False) as saved:
        counts=saved[key+'_response_counts']
    with np.load(source/'predictions.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(labels[te],saved[key+'_test_labels'])
        np.testing.assert_array_equal(counts>4,saved[key+'_train_labels'])
    return profile,x[:budget//9],counts/9,x[va],labels[va],x[te],labels[te],key,budget


def device_run(arguments):
    source,soft_source,device=arguments
    profile,x,q,vx,vy,tx,ty,source_key,budget=cohort(source,soft_source,device)
    rows,trials,weights,predictions=[],[],{},{}
    with threadpool_limits(limits=1):
        for setting,restarts,maxiter,penalty in settings(profile):
            for kind in ('majority','fraction'):
                start=perf_counter()
                rng=stream(profile.seed,102,device,2,list(profile.noises).index(.25),budget,9)
                targets=q>.5 if kind=='majority' else q
                model,attempts=fit(x,targets,vx,vy,kind,rng,restarts,maxiter,penalty)
                pred=predict_product(tx,model)
                key=f'd{device}_{setting}_{kind}'
                chosen=next(t for t in attempts if t['selected'])
                rows.append({'record_id':key,'source_record_id':source_key,'device_id':f'{profile.seed}:{device}',
                             'device':device,'setting':setting,'kind':kind,'restarts':restarts,'maxiter':maxiter,
                             'penalty':penalty,'budget':budget,'reads':9,'validation_queries':profile.validation,
                             'total_queries':budget+profile.validation,'accuracy':float(np.mean(pred==ty)),
                             'validation_accuracy':chosen['validation_accuracy'],'selected_converged':chosen['converged'],
                             'selected_iterations':chosen['iterations'],'fit_seconds':perf_counter()-start})
                trials.extend({'record_id':key,**t} for t in attempts)
                weights[key]=model;predictions[key]=pred
    print(f'Optimizer control seed {profile.seed}: device {device+1}/{profile.devices}',flush=True)
    return rows,trials,weights,predictions


def summarize(rows,profile):
    summary=[]
    for number,(name,restarts,maxiter,penalty) in enumerate(settings(profile)):
        first=[r for r in rows if r['setting']==name and r['kind']=='majority']
        second=[r for r in rows if r['setting']==name and r['kind']=='fraction']
        assert [r['device_id'] for r in first]==[r['device_id'] for r in second]
        a=np.array([r['accuracy'] for r in first]);b=np.array([r['accuracy'] for r in second])
        draws=stream(profile.seed,140,number).integers(len(a),size=(profile.bootstrap,len(a)))
        lo,hi=np.quantile((b-a)[draws].mean(axis=1),[.025,.975])
        summary.append({'setting':name,'restarts':restarts,'maxiter':maxiter,'penalty':penalty,'devices':len(a),
                        'majority_accuracy':float(a.mean()),'fraction_accuracy':float(b.mean()),
                        'paired_gain':float((b-a).mean()),'ci_low':float(lo),'ci_high':float(hi),
                        'majority_converged':sum(r['selected_converged'] for r in first),
                        'fraction_converged':sum(r['selected_converged'] for r in second)})
    return summary


def run(source,soft_source,output,workers):
    cfg=json.loads((source/'manifest.json').read_text())
    profile=Profile(**{f.name:cfg[f.name] for f in fields(Profile)})
    arguments=[(source,soft_source,i) for i in range(profile.devices)]
    rows,trials,weights,predictions=[],[],{},{}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for a,b,c,d in pool.map(device_run,arguments):
            rows.extend(a);trials.extend(b);weights.update(c);predictions.update(d)
    output.mkdir(parents=True,exist_ok=True)
    write_csv(output/'per_device.csv',rows);write_csv(output/'validation.csv',trials)
    write_csv(output/'metrics.csv',summarize(rows,profile))
    np.savez_compressed(output/'models.npz',**weights)
    np.savez_compressed(output/'predictions.npz',**predictions)
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    info={'profile':cfg,'purpose':'post-hoc sensitivity analysis; does not replace original fixed-budget estimates',
          'condition':{'chains':2,'sigma':.25,'budget':min(profile.budgets),'reads':9,'correlation':0.},
          'settings':settings(profile),'source_sha256':{n:digest(Path(__file__).with_name(n)) for n in
            ('optimizer_control.py','attack.py','soft_labels.py','model.py','experiment.py')},
          'query_reference_sha256':{n:digest(source/n) for n in ('manifest.json','per_device.csv','predictions.npz')},
          'soft_reference_sha256':{n:digest(soft_source/n) for n in ('per_device.csv','predictions.npz')},'workers':workers}
    (output/'manifest.json').write_text(json.dumps(info,indent=2)+'\n')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--soft-source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=1)
    args=p.parse_args(argv)
    if args.workers<1:p.error('--workers must be positive')
    run(args.source,args.soft_source,args.output,args.workers)


if __name__=='__main__':main()
