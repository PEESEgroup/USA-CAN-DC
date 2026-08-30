from __future__ import annotations

import argparse
from pathlib import Path

from .aidc_runner import run_aidc_case
from .io import load_aidc_results, load_jedi_results
from .jedi_runner import JediRunArtifacts, run_jedi_case
from .monetary import load_monetary_basis
from .paths import (
    default_aidc_output_dir,
    default_case_root,
    default_jedi_output_dir,
)
from .reporting import write_data_dictionary, write_manifest, write_run_log
from .standardize import (
    merge_construction_impacts,
    merge_operating_impacts,
    write_outputs,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--aidc-output-dir", type=Path, default=None)
    parser.add_argument("--jedi-output-dir", type=Path, default=None)
    parser.add_argument("--run-aidc-only", action="store_true")
    parser.add_argument("--run-jedi-only", action="store_true")
    parser.add_argument("--skip-aidc-run", action="store_true")
    parser.add_argument("--skip-jedi-run", action="store_true")
    parser.add_argument("--reuse-existing-results", action="store_true")
    parser.add_argument("--aidc-assumptions-file", type=Path, default=None)
    parser.add_argument("--aidc-construction-method", default="lifecycle_intensity")
    parser.add_argument("--aidc-dollar-per-mw-construction", type=float, default=None)
    parser.add_argument("--jedi-money-year", type=int, default=2024)
    parser.add_argument(
        "--jedi-nameplate-scenario", choices=["low", "mid", "high"], default="mid"
    )
    parser.add_argument("--jedi-max-events", type=int, default=None)
    parser.add_argument("--jedi-coverage-only", action="store_true")
    parser.add_argument(
        "--jedi-include-transmission",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include interregional Transmission Line JEDI impacts (default: enabled).",
    )
    parser.add_argument(
        "--jedi-offshore-direct-runtime",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--jedi-no-popup-monitor", action="store_true")
    parser.add_argument(
        "--jedi-static-workers",
        type=int,
        default=1,
        help="Number of parallel JEDI worker processes (each with its own Excel "
        "session). Defaults to 1 (serial) -- running more than one Excel COM "
        "session concurrently causes intermittent 'RPC server unavailable' "
        "failures on some events; serial execution costs only ~10-15%% more "
        "wall time than 8-way parallel but produces zero dropped events.",
    )
    parser.add_argument(
        "--jedi-macro-workers",
        type=int,
        default=None,
        help="Number of parallel worker processes for macro-driven wind batches. "
        "Defaults to the local CPU count (auto-detected) when not specified -- "
        "unlike --jedi-static-workers this is left parallel by default since it "
        "wasn't exercised by the 1-worker/~20-minute serial verification run; "
        "pass 1 to force full-serial execution when a case's macro-driven wind "
        "events hit intermittent Excel COM failures under parallelism.",
    )
    return parser


def _resolve_dirs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    case_name = args.case_dir.name
    aidc_dir = args.aidc_output_dir or default_aidc_output_dir(case_name)
    jedi_dir = args.jedi_output_dir or default_jedi_output_dir(case_name)
    output_dir = args.output_dir or default_case_root(case_name)
    return aidc_dir, jedi_dir, output_dir


def _module_outputs_exist(output_dir: Path, filenames: tuple[str, ...]) -> bool:
    return all((output_dir / filename).exists() for filename in filenames)


def _detect_jedi_price_year(output_dir: Path, fallback: int) -> int:
    events_path = output_dir / "supported_build_events.csv"
    if events_path.exists():
        import pandas as pd

        values = pd.to_numeric(
            pd.read_csv(events_path, usecols=["money_year"])["money_year"],
            errors="coerce",
        )
        if values.isna().any():
            raise ValueError(
                f"JEDI events contain missing money_year values: {events_path}"
            )
        years = values.unique()
        if len(years) == 1:
            return int(years[0])
        raise ValueError(
            f"JEDI events contain mixed money_year values: {sorted(years)}"
        )
    return int(fallback)


def _should_run_module(
    skip_run: bool,
    reuse_existing_results: bool,
    output_dir: Path,
    expected_files: tuple[str, ...],
) -> bool:
    if skip_run:
        return False
    if reuse_existing_results and _module_outputs_exist(output_dir, expected_files):
        return False
    return True


def _write_partial_outputs(
    output_dir: Path,
    manifest: dict[str, object],
    logs: list[str],
) -> None:
    write_manifest(output_dir, manifest)
    write_data_dictionary(output_dir)
    write_run_log(output_dir, logs)


def _raise_jedi_failure(
    output_dir: Path,
    manifest: dict[str, object],
    logs: list[str],
    jedi_artifacts: JediRunArtifacts,
) -> None:
    logs.append("JEDI employment pipeline failed; combined outputs were not generated.")
    if jedi_artifacts.staged_output_dir:
        logs.append(
            f"Failed JEDI staging outputs were preserved at {jedi_artifacts.staged_output_dir}."
        )
    _write_partial_outputs(output_dir, manifest, logs)
    error_tail = (
        jedi_artifacts.stderr_tail
        or jedi_artifacts.stdout_tail
        or "No subprocess output captured."
    )
    raise RuntimeError(
        "JEDI employment pipeline failed. "
        f"Staged outputs: {jedi_artifacts.staged_output_dir or 'n/a'}. "
        f"Subprocess tail:\n{error_tail}"
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    aidc_dir, jedi_dir, output_dir = _resolve_dirs(args)
    logs: list[str] = []
    manifest: dict[str, object] = {
        "case_dir": str(args.case_dir),
        "aidc_output_dir": str(aidc_dir),
        "jedi_output_dir": str(jedi_dir),
        "output_dir": str(output_dir),
    }

    if args.run_aidc_only and args.run_jedi_only:
        raise ValueError("Cannot combine --run-aidc-only and --run-jedi-only.")

    run_aidc = not args.run_jedi_only and _should_run_module(
        skip_run=args.skip_aidc_run,
        reuse_existing_results=args.reuse_existing_results,
        output_dir=aidc_dir,
        expected_files=(
            "datacenter_construction_impacts_annual.csv",
            "datacenter_operating_impacts_annual.csv",
        ),
    )
    run_jedi = not args.run_aidc_only and _should_run_module(
        skip_run=args.skip_jedi_run,
        reuse_existing_results=args.reuse_existing_results,
        output_dir=jedi_dir,
        expected_files=(
            "construction_impacts_annual.csv",
            "operating_impacts_annual.csv",
        ),
    )

    if run_aidc:
        logs.append("Running AIDC employment pipeline.")
        aidc_proc = run_aidc_case(
            case_dir=args.case_dir,
            output_dir=aidc_dir,
            assumptions_file=args.aidc_assumptions_file,
            construction_method=args.aidc_construction_method,
            dollar_per_mw_construction=args.aidc_dollar_per_mw_construction,
        )
        manifest["aidc_stdout_tail"] = aidc_proc.stdout[-4000:]
        manifest["aidc_stderr_tail"] = aidc_proc.stderr[-4000:]
    else:
        if args.skip_aidc_run:
            logs.append("Skipping AIDC run and reusing existing outputs.")
        elif args.reuse_existing_results:
            logs.append("Reusing existing AIDC outputs without rerunning.")
        else:
            logs.append("AIDC run not requested.")

    jedi_artifacts: JediRunArtifacts | None = None
    if run_jedi:
        logs.append(
            "Running JEDI employment pipeline"
            + (
                " in coverage-only mode."
                if args.jedi_coverage_only
                else " with popup monitor workflow."
            )
        )
        jedi_artifacts = run_jedi_case(
            case_dir=args.case_dir,
            output_dir=jedi_dir,
            nameplate_scenario=args.jedi_nameplate_scenario,
            money_year=args.jedi_money_year,
            max_events=args.jedi_max_events,
            coverage_only=args.jedi_coverage_only,
            offshore_direct_runtime=args.jedi_offshore_direct_runtime,
            enable_popup_monitor=not args.jedi_no_popup_monitor,
            static_workers=args.jedi_static_workers,
            macro_workers=args.jedi_macro_workers,
            include_transmission=args.jedi_include_transmission,
        )
        manifest["jedi_run"] = {
            "command": jedi_artifacts.command,
            "monitor_command": jedi_artifacts.monitor_command,
            "monitor_pid": jedi_artifacts.monitor_pid,
            "monitor_log": jedi_artifacts.monitor_log,
            "output_dir": jedi_artifacts.output_dir,
            "staged_output_dir": jedi_artifacts.staged_output_dir,
            "status": jedi_artifacts.status,
            "stdout_tail": jedi_artifacts.stdout_tail,
            "stderr_tail": jedi_artifacts.stderr_tail,
        }
        if jedi_artifacts.status != "success":
            _raise_jedi_failure(output_dir, manifest, logs, jedi_artifacts)
    else:
        if args.skip_jedi_run:
            logs.append("Skipping JEDI run and reusing existing outputs.")
        elif args.reuse_existing_results:
            logs.append("Reusing existing JEDI outputs without rerunning.")
        else:
            logs.append("JEDI run not requested.")

    if args.jedi_coverage_only:
        logs.append(
            "JEDI coverage-only mode selected; combined outputs were not generated."
        )
        _write_partial_outputs(output_dir, manifest, logs)
        return

    if args.run_aidc_only:
        logs.append("AIDC-only mode selected; combined outputs were not generated.")
        _write_partial_outputs(output_dir, manifest, logs)
        return

    if args.run_jedi_only:
        logs.append("JEDI-only mode selected; combined outputs were not generated.")
        _write_partial_outputs(output_dir, manifest, logs)
        return

    aidc_results = load_aidc_results(aidc_dir)
    jedi_results = load_jedi_results(jedi_dir)
    jedi_price_year = _detect_jedi_price_year(jedi_dir, args.jedi_money_year)
    manifest["aidc_schema"] = aidc_results.schema
    manifest["jedi_schema"] = jedi_results.schema
    manifest["monetary_basis"] = {
        **load_monetary_basis(),
        "jedi_source_price_year": jedi_price_year,
    }
    manifest["complete_scenarios"] = {
        "low": "JEDI point estimate + AIDC low",
        "base": "JEDI point estimate + AIDC base",
        "high": "JEDI point estimate + AIDC high",
    }

    construction = merge_construction_impacts(
        args.case_dir.name, aidc_results, jedi_results, jedi_price_year
    )
    operating = merge_operating_impacts(
        args.case_dir.name, aidc_results, jedi_results, jedi_price_year
    )
    write_outputs(output_dir, construction, operating)
    write_manifest(output_dir, manifest)
    write_data_dictionary(output_dir)
    logs.append(f"Detected JEDI result schema: {jedi_results.schema}.")
    logs.append(
        "Wrote merged construction_impacts_annual.csv and operating_impacts_annual.csv."
    )
    logs.append(
        "Merged outputs use complete low/base/high scenarios and constant 2024 USD."
    )
    write_run_log(output_dir, logs)


if __name__ == "__main__":
    main()
