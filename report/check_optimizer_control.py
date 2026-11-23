"""Audit the optimization control against the original cohorts and saved models."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from puf_lab.optimizer_control import cohort,settings
from puf_lab.attack import Profile
from dataclasses import fields
import puf_lab.optimizer_control as implementation
from puf_lab.soft_labels import read_csv,soft_objective
from puf_lab.attack import objective,predict_product
from puf_lab.experiment import stream


def check(directory,source,soft_source):
    manifest=json.loads((directory/'manifest.json').read_text())
    for name,value in manifest['source_sha256'].items():
        assert hashlib.sha256(Path(implementation.__file__).with_name(name).read_bytes()).hexdigest()==value
    for folder,field in ((source,'query_reference_sha256'),(soft_source,'soft_reference_sha256')):
        for name,value in manifest[field].items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==value
    rows=read_csv(directory/'per_device.csv');validation=read_csv(directory/'validation.csv')
    devices=manifest['profile']['devices']
    profile=Profile(**{f.name:manifest['profile'][f.name] for f in fields(Profile)})
    configurations=settings(profile)
    assert manifest['settings']==[list(s) for s in configurations]
    assert len(rows)==devices*len(configurations)*2
    assert len({r['record_id'] for r in rows})==len(rows)
    assert len(validation)==devices*sum(s[1] for s in configurations)*2
    def close(a,b):np.testing.assert_allclose(float(a),float(b),rtol=1e-11,atol=1e-12)
    with np.load(directory/'models.npz',allow_pickle=False) as models,np.load(directory/'predictions.npz',allow_pickle=False) as predictions, \
         np.load(source/'predictions.npz',allow_pickle=False) as original,np.load(soft_source/'predictions.npz',allow_pickle=False) as fractions:
        for device in range(devices):
            profile,x,q,vx,vy,tx,ty,source_key,budget=cohort(source,soft_source,device)
            for setting,restarts,maxiter,penalty in configurations:
                for kind in ('majority','fraction'):
                    key=f'd{device}_{setting}_{kind}'
                    row=next(r for r in rows if r['record_id']==key)
                    assert row['source_record_id']==source_key
                    assert int(row['restarts'])==restarts and int(row['maxiter'])==maxiter
                    close(row['penalty'],penalty)
                    assert int(row['budget'])==budget and int(row['reads'])==9
                    assert int(row['total_queries'])==budget+profile.validation
                    model=models[key];pred=predict_product(tx,model)
                    np.testing.assert_array_equal(pred,predictions[key])
                    close(row['accuracy'],np.mean(pred==ty))
                    close(row['validation_accuracy'],np.mean(predict_product(vx,model)==vy))
                    attempts=[r for r in validation if r['record_id']==key]
                    assert len(attempts)==restarts
                    selected=[r for r in attempts if r['selected']=='True'];assert len(selected)==1
                    chosen=selected[0]
                    close(chosen['validation_accuracy'],max(float(r['validation_accuracy']) for r in attempts))
                    close(chosen['validation_accuracy'],row['validation_accuracy'])
                    assert row['selected_converged']==chosen['converged']
                    assert row['selected_iterations']==chosen['iterations']
                    if kind=='majority':loss,grad=objective(model.ravel(),x,1-2*(q>.5).astype(float),2,penalty)
                    else:loss,grad=soft_objective(model.ravel(),x,q,2,penalty)
                    close(chosen['loss'],loss);close(chosen['gradient_norm'],np.linalg.norm(grad))
                    if setting=='published':
                        expected=(original if kind=='majority' else fractions)[source_key+'_prediction']
                        np.testing.assert_array_equal(pred,expected)
    metrics=read_csv(directory/'metrics.csv');assert len(metrics)==len(configurations)
    for number,(setting,restarts,maxiter,penalty) in enumerate(configurations):
        row=next(r for r in metrics if r['setting']==setting)
        part=[[r for r in rows if r['setting']==setting and r['kind']==kind] for kind in ('majority','fraction')]
        assert [r['device_id'] for r in part[0]]==[r['device_id'] for r in part[1]]
        a,b=[np.array([float(r['accuracy']) for r in group]) for group in part]
        draws=stream(profile.seed,140,number).integers(devices,size=(profile.bootstrap,devices))
        lo,hi=np.quantile((b-a)[draws].mean(axis=1),[.025,.975])
        for name,value in (('majority_accuracy',a.mean()),('fraction_accuracy',b.mean()),('paired_gain',(b-a).mean()),('ci_low',lo),('ci_high',hi)):
            close(row[name],value)
        for name,group in zip(('majority_converged','fraction_converged'),part):
            assert int(row[name])==sum(r['selected_converged']=='True' for r in group)
    return {r['device_id'] for r in rows}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path('results/optimizer_control'))
    p.add_argument('--source',type=Path,default=Path('results/query_study'))
    p.add_argument('--soft-source',type=Path,default=Path('results/soft_labels'))
    args=p.parse_args()
    first=check(args.results,args.source,args.soft_source)
    second=check(args.results/'repeat',args.source/'repeat',args.soft_source/'repeat')
    assert first.isdisjoint(second)
    print(f'Verified optimization controls, original baselines and paired intervals on {len(first|second)} devices.')
