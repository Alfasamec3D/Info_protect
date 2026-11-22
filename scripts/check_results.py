"""Audit included experiments independently of plotting/report generation."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for name in ['baseline','query_budget']:
    out = root/'results'/name
    manifest = json.loads((out/'manifest.json').read_text())
    config = manifest['config']
    rows = [json.loads(s) for s in (out/'runs.jsonl').read_text().splitlines()]
    expected = len(config['seeds'])*len(config['chains'])*len(config['budgets'])*len(config['noise'])*len(config['strategies'])*len(config['models'])
    assert len(rows) == expected, (name,len(rows),expected)
    assert len({r['id'] for r in rows}) == len(rows)
    for r in rows:
        assert r['queries_used'] == r['train_unique']*r['repeats']+r['validation_queries']
        assert 0 <= r['budget']-r['queries_used'] < r['repeats']
        assert 0 <= r['accuracy'] <= 1 and 0 <= r['noisy_accuracy'] <= 1
        assert (out/'models'/f"{r['id']}.npz").is_file()
        assert (out/'traces'/f"{r['id']}.json").is_file()
    h = hashlib.sha256(json.dumps(config,sort_keys=True).encode())
    for p in sorted((root/'src/puf_attacks').glob('*.py')):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    assert h.hexdigest() == manifest['fingerprint'], 'source differs from experiment provenance'
    print(f'{name}: {len(rows)}/{expected} runs, budgets, artifacts and source fingerprint OK')
out=root/'results/fisher'
manifest=json.loads((out/'manifest.json').read_text())
assert hashlib.sha256((root/'scripts/fisher_experiment.py').read_bytes()).hexdigest()==manifest['source_sha256']
assert hashlib.sha256((root/'src/puf_attacks/puf.py').read_bytes()).hexdigest()==manifest['puf_source_sha256']
import csv
with (out/'trials.csv').open() as f:
    trials=list(csv.DictReader(f))
assert len(trials)==3*manifest['trials']
assert all(int(r['queries'])==manifest['queries'] for r in trials)
assert all(float(r['information'])>=0 for r in trials)
print(f'fisher: {len(trials)} designs, query counts and source hashes OK')
