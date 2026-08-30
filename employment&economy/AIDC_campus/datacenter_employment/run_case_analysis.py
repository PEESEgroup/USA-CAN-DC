
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .assumptions import AssumptionRegister, DEFAULT_ASSUMPTIONS_PATH
from .public_bea_lq_io import DEFAULT_NORMALIZED_DIR
from .spend_io import DEFAULT_CONSTRUCTION_LOCALIZATION_PATH
from employment.monetary import load_monetary_basis


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[3]


def _construction_method_arg(value: str) -> str:
    if value == "workers_per_mw":
        raise argparse.ArgumentTypeError(
            "'workers_per_mw' was removed: use 'lifecycle_intensity'; its coefficient "
            "already includes peak/average workforce, duration, and PUE conversion"
        )
    return value


def _module_invocation(module: str, *args: str) -> list[str]:
    return [sys.executable, "-m", module, *args]


def _run_python_module(module: str, args: list[str]) -> None:
    subprocess.run(_module_invocation(module, *args), check=True)


def _write_manifest(output_dir: Path, payload: dict) -> None:
    with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=REPO_ROOT / "runs_save" / "v0626_G0-F0-S0",
    )
    parser.add_argument(
        "--assumptions-file", type=Path, default=DEFAULT_ASSUMPTIONS_PATH
    )
    parser.add_argument(
        "--construction-method",
        type=_construction_method_arg,
        choices=["lifecycle_intensity", "capex"],
        default="lifecycle_intensity",
    )
    parser.add_argument(
        "--indirect-method",
        choices=["public_bea_lq_io", "spend_io", "legacy_ratio"],
        default="public_bea_lq_io",
    )
    parser.add_argument(
        "--canada-spend-profile",
        choices=["statcan_regional", "us_proxy_sensitivity"],
        default="statcan_regional",
    )
    parser.add_argument("--dollar-per-mw-construction", type=float, default=None)
    parser.add_argument(
        "--construction-localization-csv",
        type=Path,
        default=DEFAULT_CONSTRUCTION_LOCALIZATION_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--public-bea-lq-dir", type=Path, default=DEFAULT_NORMALIZED_DIR
    )
    args = parser.parse_args()

    output_dir = args.output_dir or MODULE_DIR / "results" / args.case_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    campus_args = [
        "--case-dir",
        str(args.case_dir),
        "--output-dir",
        str(output_dir),
        "--assumptions-file",
        str(args.assumptions_file),
        "--construction-method",
        args.construction_method,
        "--indirect-method",
        args.indirect_method,
        "--canada-spend-profile",
        args.canada_spend_profile,
        "--construction-localization-csv",
        str(args.construction_localization_csv),
        "--public-bea-lq-dir",
        str(args.public_bea_lq_dir),
    ]
    if args.dollar_per_mw_construction is not None:
        campus_args.extend(
            ["--dollar-per-mw-construction", str(args.dollar_per_mw_construction)]
        )
    _run_python_module(
        "employment.AIDC_campus.datacenter_employment.campus_employment",
        campus_args,
    )

    assumption_register = AssumptionRegister.load(args.assumptions_file)
    _write_manifest(
        output_dir,
        {
            "assumptions_file": str(args.assumptions_file),
            "case_dir": str(args.case_dir),
            "construction_method": args.construction_method,
            "indirect_method": args.indirect_method,
            "canada_spend_profile": args.canada_spend_profile,
            "construction_localization_csv": str(args.construction_localization_csv),
            "public_bea_lq_dir": str(args.public_bea_lq_dir),
            "economic_method": (
                "public_bea_lq_io"
                if args.indirect_method == "public_bea_lq_io"
                else f"validation_{args.indirect_method}"
            ),
            "dollar_per_mw_construction": args.dollar_per_mw_construction,
            "direct_construction_job_years_per_mw_it": assumption_register.triplet(
                "direct_construction_job_years_per_mw_it"
            ).as_dict(),
            "direct_jobs_per_mw_operating": assumption_register.triplet(
                "direct_jobs_per_mw_operating"
            ).as_dict(),
            "capacity_basis": "IT",
            "pue_conversion": assumption_register.triplet(
                "employment_pue_conversion"
            ).base,
            "monetary_basis": load_monetary_basis(),
            "output_dir": str(output_dir),
        },
    )


if __name__ == "__main__":
    main()
