"""Small end-to-end demonstration without installation: python scripts/demo.py."""
import os
import sys
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from puf_attacks.experiments import run
from puf_attacks.analysis import summarize

if __name__ == '__main__':
    run(ROOT/'configs/smoke.json',ROOT/'results/smoke')
    summarize(ROOT/'results/smoke')
