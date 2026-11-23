"""Independently audit fractional-label training, exported predictions and effects."""
import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from puf_lab.attack import Profile, sample_device, predict_product
from puf_lab.experiment import stream
from puf_lab.soft_labels import read_csv, GROUP, FIELDS
import puf_lab.soft_labels as implementation


def check(directory, source):
    manifest=json.loads((directory/'manifest.json').read_text())
    for name,digest in manifest['source_sha256'].items():
        assert hashlib.sha256(Path(implementation.__file__).with_name(name).read_bytes()).hexdigest()==digest
    for name,digest in manifest['reference_sha256'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest
    cfg=manifest['profile']
    assert cfg==json.loads((source/'manifest.json').read_text())
    profile=Profile(**{f.name:cfg[f.name] for f in fields(Profile)})
    rows=read_csv(directory/'per_device.csv')
    validation=read_csv(directory/'validation.csv')
    original=read_csv(source/'per_device.csv')
    original_index={r['record_id']:r for r in original}
    expected={r['record_id'] for r in original if r['kind']=='budget'}
    assert {r['record_id'] for r in rows}==expected and len(rows)==len(expected)
    assert len(validation)==len(rows)*profile.restarts
    def close(a,b):
        np.testing.assert_allclose(float(a),b,rtol=0,atol=1e-12)
    measured=[]
    with np.load(directory/'predictions.npz',allow_pickle=False) as saved, \
         np.load(directory/'models.npz',allow_pickle=False) as models, \
         np.load(source/'predictions.npz',allow_pickle=False) as baseline:
        for device in range(profile.devices):
            raw,x,weights,(_,va,te)=sample_device(profile,device)
            np.testing.assert_array_equal(raw,saved[f'd{device}_challenges'])
            # Check all three partitions are unique and mutually disjoint.
            assert len({np.packbits(c).tobytes() for c in raw})==len(raw)
            for row in (r for r in rows if int(r['device'])==device):
                key=row['record_id']; previous=original_index[key]
                for field in (*GROUP,'device_id','unique_challenges','validation_queries','total_queries'):
                    assert row[field]==previous[field],field
                k,r,B=[int(row[n]) for n in ('chains','reads','budget')]
                sigma,rho=[float(row[n]) for n in ('noise','correlation')]
                noise_no=list(profile.noises).index(sigma)
                count=B//r
                assert count*r==B and int(row['total_queries'])==B+profile.validation
                delays=x@weights[:k].T
                rng=stream(profile.seed,101,device,k,noise_no)
                first=rng.normal(size=(max(profile.budgets),k))
                common=rng.normal(size=first.shape)
                other=rng.normal(size=(8,*first.shape))
                perturb=np.concatenate((first[None],rho*first+np.sqrt(rho*(1-rho))*common+np.sqrt(1-rho)*other))
                # Compute parity by counting negative delays, independently of generator XOR.
                bits=((delays[:count]+sigma*perturb[:r,:count]<0).sum(axis=-1)%2).astype(bool)
                counts=bits.sum(axis=0)
                np.testing.assert_array_equal(counts,saved[key+'_response_counts'])
                np.testing.assert_array_equal(counts>r//2,baseline[key+'_train_labels'])
                clean=((delays<0).sum(axis=-1)%2).astype(bool)
                noisy=delays+stream(profile.seed,100,device,k,noise_no).normal(0,sigma,delays.shape)
                labels=((noisy<0).sum(axis=-1)%2).astype(bool)
                np.testing.assert_array_equal(labels[te],baseline[key+'_test_labels'])
                w=models[key+'_weights']
                pred=(np.prod(x[te]@w.T,axis=1)<0)
                np.testing.assert_array_equal(pred,saved[key+'_prediction'])
                attempts=[r for r in validation if r['record_id']==key]
                assert len(attempts)==profile.restarts
                chosen=[r for r in attempts if r['selected']=='True']
                assert len(chosen)==1
                val_accuracy=np.mean(predict_product(x[va],w)==labels[va])
                close(chosen[0]['validation_accuracy'],val_accuracy)
                close(chosen[0]['validation_accuracy'],max(float(v['validation_accuracy']) for v in attempts))
                # Direct average Bernoulli BCE over all bits, rather than fractional-loss helper.
                scores=np.prod(x[:count]@w.T,axis=1)
                loss=np.logaddexp(0,-(1-2*bits.astype(float))*scores).mean()+1e-4*np.sum(w*w)/2
                close(chosen[0]['loss'],loss)
                single=next(v for v in original if v['device_id']==row['device_id'] and v['kind']=='benchmark'
                            and v['model']=='product' and int(v['chains'])==k and float(v['noise'])==sigma and int(v['budget'])==B)
                majority=baseline[key+'_prediction']; one=baseline[single['record_id']+'_prediction']
                # Exact XOR-one probability from chain probit probabilities, no read_error helper.
                p_one=(1-np.prod(1-2*ndtr(-delays[te]/sigma),axis=1))/2
                mean_expected=lambda p: float(np.where(p,p_one,1-p_one).mean())
                accuracy=float(np.mean(pred==labels[te]))
                values={'accuracy':accuracy,'clean_accuracy':float(np.mean(pred==clean[te])),
                        'expected_accuracy':mean_expected(pred),'gain_over_majority':accuracy-float(previous['accuracy']),
                        'gain_over_single':accuracy-float(single['accuracy']),
                        'expected_gain_over_majority':mean_expected(pred)-mean_expected(majority),
                        'expected_gain_over_single':mean_expected(pred)-mean_expected(one),
                        'majority_accuracy':float(np.mean(majority==labels[te])),
                        'single_accuracy':float(np.mean(one==labels[te]))}
                for field,value in values.items(): close(row[field],value)
                measured.append({**row,**values})
    summaries=read_csv(directory/'metrics.csv')
    keys={tuple(r[k] for k in GROUP) for r in measured}
    assert len(summaries)==len(keys) and {tuple(r[k] for k in GROUP) for r in summaries}==keys
    for row in summaries:
        part=[r for r in measured if all(r[k]==row[k] for k in GROUP)]
        assert len(part)==profile.devices==int(row['devices'])
        draws=stream(profile.seed,120,int(row['bootstrap_stream'])).integers(len(part),size=(profile.bootstrap,len(part)))
        for f in FIELDS:
            values=np.array([r[f] for r in part]);lo,hi=np.quantile(values[draws].mean(axis=1),[.025,.975])
            close(row[f],values.mean());close(row[f+'_low'],lo);close(row[f+'_high'],hi)
        for f in ('majority_accuracy','single_accuracy','fit_seconds'):
            close(row[f],np.mean([float(r[f]) for r in part]))
        assert int(row['devices_better_than_majority'])==sum(r['gain_over_majority']>0 for r in part)
    return {r['device_id'] for r in rows}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path('results/soft_labels'))
    p.add_argument('--source',type=Path,default=Path('results/query_study'))
    args=p.parse_args()
    first=check(args.results,args.source)
    second=check(args.results/'repeat',args.source/'repeat')
    assert first.isdisjoint(second)
    print(f'Verified fractional targets, saved models and paired effects on {len(first|second)} devices.')
