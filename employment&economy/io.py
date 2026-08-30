from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


CURRENT_AIDC_METHOD_VERSION = "dc_calibrated_v2"


@dataclass(frozen=True)
class ModuleResults:
    module: str
    construction: pd.DataFrame
    operations: pd.DataFrame
    schema: str
    output_dir: Path


def _require_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required output missing: {path}")
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:


        return pd.DataFrame()


def detect_jedi_schema(construction: pd.DataFrame, operations: pd.DataFrame) -> str:
    construction_full = {
        "construction_jobs_fte_direct",
        "construction_jobs_fte_indirect",
        "construction_jobs_fte_induced",
        "construction_jobs_fte_total",
    }.issubset(construction.columns)
    operating_full = {
        "operating_jobs_fte_direct",
        "operating_jobs_fte_indirect",
        "operating_jobs_fte_induced",
        "operating_jobs_fte_total",
    }.issubset(operations.columns)
    if construction_full and operating_full:
        return "full_breakdown"
    return "legacy_total_only"


def load_jedi_results(output_dir: Path) -> ModuleResults:
    construction = _require_csv(output_dir / "construction_impacts_annual.csv")
    operations = _require_csv(output_dir / "operating_impacts_annual.csv")
    return ModuleResults(
        module="jedi",
        construction=construction,
        operations=operations,
        schema=detect_jedi_schema(construction, operations),
        output_dir=output_dir,
    )


def load_aidc_results(output_dir: Path) -> ModuleResults:
    construction = _require_csv(
        output_dir / "datacenter_construction_impacts_annual.csv"
    )
    operations = _require_csv(output_dir / "datacenter_operating_impacts_annual.csv")
    for name, frame in (("construction", construction), ("operations", operations)):
        if frame.empty:
            continue
        if "economic_method_version" not in frame.columns:
            raise ValueError(
                f"AIDC {name} output is unversioned and cannot be combined with "
                f"{CURRENT_AIDC_METHOD_VERSION}; rerun that case"
            )
        versions = set(frame["economic_method_version"].dropna().astype(str))
        if (
            versions != {CURRENT_AIDC_METHOD_VERSION}
            or frame["economic_method_version"].isna().any()
        ):
            raise ValueError(
                f"AIDC {name} output has incompatible/mixed method versions: "
                f"{sorted(versions)}"
            )
    return ModuleResults(
        module="aidc",
        construction=construction,
        operations=operations,
        schema="aidc_direct_plus_indirect_induced",
        output_dir=output_dir,
    )
