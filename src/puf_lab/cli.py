"""Command line entry point for a reproducible experiment."""

import argparse
import csv
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np
import scipy

from . import __version__
from .config import Config
from .experiment import run
from .plots import plot
from .summary import summarize


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="JSON configuration")
    parser.add_argument("--output", type=Path, required=True, help="result directory")
    parser.add_argument("--quiet", action="store_true", help="hide device progress")
    args = parser.parse_args(argv)
    try:
        config = Config.load(args.config)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))

    def progress(done, total):
        print(f"Simulated devices: {done}/{total}", file=sys.stderr, flush=True)

    results = summarize(config, run(config, None if args.quiet else progress))
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in results.items():
        write_csv(args.output / f"{name}.csv", rows)
    plot(results, args.output)
    manifest = {
        "puf_lab": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libraries": {
            "numpy": np.__version__, "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "config": asdict(config),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
