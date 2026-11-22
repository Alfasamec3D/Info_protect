"""Regenerate all numerical deliverables of the course project."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys


def jobs(output):
    for profile, suffix in (("main", "forecast"), ("repeat", "forecast/repeat")):
        yield suffix, "forecast_cli", ("--profile", profile, "--quiet"), output / suffix
    yield "session study", "session_cli", (
        "--config", "configs/session_study.json", "--quiet",
    ), output
    for profile, suffix in (("main", "calibration"), ("repeat", "calibration/repeat")):
        yield suffix, "calibration_cli", ("--profile", profile, "--quiet"), output / suffix
    for profile, suffix in (("main", "equal-span"), ("repeat", "equal-span/repeat")):
        yield suffix, "calibration_cli", (
            "--family", "equal-span", "--profile", profile, "--quiet",
        ), output / "calibration" / suffix
    for profile, suffix in (("main", "sensitivity"), ("repeat", "sensitivity/repeat")):
        yield suffix, "sensitivity_cli", ("--profile", profile, "--quiet"), output / suffix
    yield "session replication", "session_cli", (
        "--config", "configs/session_replication.json", "--quiet",
    ), output / "replication"
    yield "reference study", "cli", ("--config", "configs/study.json", "--quiet"), output
    yield "device without prior observations", "control", (), output
    yield "one-time challenge verification", "defensive_auth", (), output / "defense"


def run_job(job):
    title, module, arguments, destination = job
    env = os.environ | {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1", "MPLBACKEND": "Agg"}
    print(f"Running {title}", flush=True)
    subprocess.run(
        (sys.executable, "-m", f"puf_lab.{module}", *arguments,
         "--output", str(destination)),
        check=True, env=env,
    )
    print(f"Finished {title}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1, help="parallel independent runs")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if not Path("configs/study.json").is_file():
        parser.error("run from the repository root")
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(run_job, jobs(args.output.resolve())))


if __name__ == "__main__":
    main()
