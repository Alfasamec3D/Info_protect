"""Compare a fresh full reproduction with all recorded numerical deliverables."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

IGNORED_CSV={'fit_seconds','total_seconds'}
IGNORED_JSON={'reference_sha256','query_reference_sha256','soft_reference_sha256','workers'}


def files(root,suffix):
    return {p.relative_to(root).as_posix():p for p in root.rglob('*'+suffix)
            if 'verification' not in p.relative_to(root).parts}


def csv_rows(path):
    with path.open(newline='',encoding='utf-8') as f:
        r=csv.DictReader(f)
        return r.fieldnames,list(r)


def normalized(value):
    if isinstance(value,dict):return {k:normalized(v) for k,v in value.items() if k not in IGNORED_JSON}
    if isinstance(value,list):return [normalized(v) for v in value]
    return value


def compare(reference,repeated):
    records=[]
    for suffix in ('.csv','.npz','.json'):
        first,second=files(reference,suffix),files(repeated,suffix)
        assert first.keys()==second.keys(),(suffix,'missing',sorted(first.keys()-second.keys()),'extra',sorted(second.keys()-first.keys()))
        for name,p in sorted(first.items()):
            q=second[name]
            record={'file':name,'kind':suffix[1:],'reference_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                    'repeated_sha256':hashlib.sha256(q.read_bytes()).hexdigest()}
            if suffix=='.csv':
                headers,a=csv_rows(p);other,b=csv_rows(q)
                assert headers==other and len(a)==len(b),name
                maximum=0.;unequal=0
                for line,(x,y) in enumerate(zip(a,b),2):
                    for key in headers:
                        if key in IGNORED_CSV:continue
                        if x[key]==y[key]:continue
                        try:left,right=float(x[key]),float(y[key])
                        except ValueError:raise AssertionError((name,line,key,x[key],y[key]))
                        assert math.isfinite(left) and math.isfinite(right),(name,line,key)
                        delta=abs(left-right);maximum=max(maximum,delta);unequal+=1
                        assert math.isclose(left,right,rel_tol=1e-12,abs_tol=1e-12),(name,line,key,left,right)
                record.update(rows=len(a),ignored_columns=sorted(set(headers)&IGNORED_CSV),
                              unequal_numeric_cells=unequal,maximum_absolute_difference=maximum)
            elif suffix=='.npz':
                maximum=0.;unequal=0
                with np.load(p,allow_pickle=False) as a,np.load(q,allow_pickle=False) as b:
                    assert set(a.files)==set(b.files),name
                    for key in a.files:
                        left,right=a[key],b[key]
                        assert left.shape==right.shape and left.dtype==right.dtype,(name,key)
                        if np.array_equal(left,right):continue
                        assert np.issubdtype(left.dtype,np.floating),(name,key)
                        # Imported models use NaN for an unused constant-predictor field.
                        # Equal missing-value masks are metadata, never unequal weights.
                        if np.isnan(left).any() or np.isnan(right).any():
                            assert np.array_equal(np.isnan(left),np.isnan(right)),(name,key,'missing values')
                            valid=~np.isnan(left)
                            left,right=left[valid],right[valid]
                            if np.array_equal(left,right):continue
                        assert np.isfinite(left).all() and np.isfinite(right).all(),(name,key)
                        maximum=max(maximum,float(np.max(np.abs(left-right))))
                        unequal+=int(np.sum(left!=right))
                        np.testing.assert_allclose(left,right,rtol=1e-12,atol=1e-12,err_msg=f'{name}:{key}')
                    record.update(arrays=len(a.files),unequal_numeric_elements=unequal,maximum_absolute_difference=maximum)
            else:
                a,b=json.loads(p.read_text()),json.loads(q.read_text())
                assert normalized(a)==normalized(b),name
                record['ignored_metadata_fields']=sorted(IGNORED_JSON.intersection(a))
            records.append(record)
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=Path('results'))
    parser.add_argument('--repeated',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    records=compare(args.reference,args.repeated)
    report={'checked_at':datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),
            'status':'passed','scope':'all CSV values, NPZ arrays and deterministic JSON metadata',
            'excluded':'wall-clock fit_seconds/total_seconds, execution worker counts and reference file hashes affected by those times; figures are checked in report review',
            'absolute_tolerance':1e-12,'relative_tolerance':1e-12,
            'files_compared':len(records),'files':records}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    changed=sum(r.get('unequal_numeric_cells',0)+r.get('unequal_numeric_elements',0) for r in records)
    print(f'Compared {len(records)} numerical artifacts; {changed} unequal numeric entries (within 1e-12 tolerance).')


if __name__=='__main__':main()
