
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR = REPO_ROOT / "runs_save"
RESULTS_ROOT = REPO_ROOT / "employment" / "results"
DONE_MARKER = "construction_impacts_annual.csv"


def iter_case_dirs(runs_dir: Path) -> list[Path]:
    containers = sorted(
        p
        for p in runs_dir.iterdir()
        if p.is_dir() and not p.name.startswith("_") and not p.name.startswith(".")
    )


    return [
        (
            p / p.name
            if not (p / "outputs").exists() and (p / p.name / "outputs").exists()
            else p
        )
        for p in containers
    ]


def is_done(case_dir: Path) -> bool:
    return (RESULTS_ROOT / case_dir.name / DONE_MARKER).exists()


def run_case(case_dir: Path, log_path: Path) -> tuple[bool, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "employment.run_employment_case",
        "--case-dir",
        str(case_dir),
    ]
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as fh:
        result = subprocess.run(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
        )
    elapsed = time.time() - t0
    return result.returncode == 0, elapsed


def fmt_duration(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument(
        "--force", action="store_true", help="Re-run even if results already exist."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without executing.",
    )
    args = parser.parse_args()

    cases = iter_case_dirs(args.runs_dir)
    pending = [c for c in cases if args.force or not is_done(c)]
    skipped = [c for c in cases if not args.force and is_done(c)]

    print(
        f"[{datetime.now():%H:%M:%S}] Found {len(cases)} case(s): "
        f"{len(pending)} to run, {len(skipped)} already done (skipping)."
    )
    for c in skipped:
        print(f"  SKIP  {c.name}")
    print()

    if args.dry_run:
        for c in pending:
            print(f"  WOULD RUN  {c.name}")
        return

    results: list[tuple[str, bool, float]] = []
    for i, case_dir in enumerate(pending, 1):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] ({i}/{len(pending)}) Starting: {case_dir.name} ...", flush=True)
        log_path = RESULTS_ROOT / case_dir.name / "employment_run.log"
        ok, elapsed = run_case(case_dir, log_path)
        status = "OK  " if ok else "FAIL"
        print(
            f"[{datetime.now():%H:%M:%S}] ({i}/{len(pending)}) {status} "
            f"{case_dir.name}  ({fmt_duration(elapsed)})  log->{log_path}",
            flush=True,
        )
        results.append((case_dir.name, ok, elapsed))

    print(f"\n{'='*60}")
    print(
        f"Summary: {len(results)} ran, "
        f"{sum(1 for _,ok,_ in results if ok)} succeeded, "
        f"{sum(1 for _,ok,_ in results if not ok)} failed"
    )
    total = sum(e for _, _, e in results)
    print(f"Total elapsed: {fmt_duration(total)}")
    failed = [(n, e) for n, ok, e in results if not ok]
    if failed:
        print("\nFailed cases:")
        for name, _ in failed:
            log = RESULTS_ROOT / name / "employment_run.log"
            print(f"  {name}  (log: {log})")
        sys.exit(1)


if __name__ == "__main__":
    main()
