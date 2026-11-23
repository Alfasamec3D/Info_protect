"""Regenerate all numerical deliverables of the course project."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys


def jobs(output, query_workers=1):
    yield "Gaussian information bound", "information", (), output / "reference"
    for profile, suffix in (("main", "binary_information"), ("repeat", "binary_information/repeat")):
        yield suffix, "binary_information", ("--profile", profile), output / suffix
    for profile, suffix in (("main", "query_study"), ("repeat", "query_study/repeat")):
        yield suffix, "query_study", ("--profile", profile, "--workers", str(query_workers)), output / suffix
    for profile, suffix in (("main", "attack"), ("repeat", "attack/repeat")):
        yield suffix, "attack_cli", ("--profile", profile), output / suffix
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
    parser.add_argument("--pypuf-path", type=Path, help="optional separately installed pypuf 3.2.1 simulator")
    parser.add_argument("--query-workers", type=int, default=1, help="independent device workers per query study")
    args = parser.parse_args(argv)
    if args.jobs < 1 or args.query_workers < 1:
        parser.error("--jobs and --query-workers must be positive")
    if not Path("configs/study.json").is_file():
        parser.error("run from the repository root")
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(run_job, jobs(args.output.resolve(), args.query_workers)))
    # Fractional labels reuse the completed query cohorts and baseline predictions.
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(run_job, (
            ("soft labels " + suffix, "soft_labels",
             ("--source", str(args.output.resolve() / "query_study" / suffix),
              "--workers", str(args.query_workers)), args.output.resolve() / "soft_labels" / suffix)
            for suffix in ("", "repeat")
        )))
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(run_job, (
            ("optimizer control " + suffix, "optimizer_control",
             ("--source", str(args.output.resolve() / "query_study" / suffix),
              "--soft-source", str(args.output.resolve() / "soft_labels" / suffix),
              "--workers", str(args.query_workers)), args.output.resolve() / "optimizer_control" / suffix)
            for suffix in ("", "repeat")
        )))
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(run_job, (
            ('observation likelihood ' + profile, 'observation_study',
             ('--profile', profile, '--workers', str(args.query_workers)),
             args.output.resolve() / 'observation_study' / suffix)
            for profile, suffix in (('main', ''), ('repeat', 'repeat'))
        )))
    if args.pypuf_path:
        run_job(("pypuf reference check", "reference_check", ("--pypuf-path", str(args.pypuf_path.resolve())),
                 args.output.resolve() / "reference"))
    run_job(("Ivan branch studies", "ivan_studies", ("--workers", str(args.query_workers)),
             args.output.resolve() / "ivan"))


if __name__ == "__main__":
    main()
