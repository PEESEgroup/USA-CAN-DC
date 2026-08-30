
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_OUTPUTS = ("datacenter_it_inv.csv", "datacenter_it_cap.csv", "hierarchy.csv")
REQUIRED_INPUTS_CASE = ("modeledyears.csv",)
OPTIONAL_OUTPUTS = ("datacenter_load_st.csv", "datacenter_load_st_base.csv")


@dataclass(frozen=True)
class DatacenterCaseData:
    case_dir: Path
    case_name: str
    it_inv: pd.DataFrame
    it_cap: pd.DataFrame
    hierarchy: pd.DataFrame
    modeled_years: list[int]
    load_st: pd.DataFrame | None
    load_st_base: pd.DataFrame | None


def _require_file(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input missing: {path}\n"
            f"This file is produced by {produced_by}. The most likely cause is that this case "
            "was run with the data-center module disabled (e.g. Sw_DatacenterScenario unset) or "
            "the case has not finished a ReEDS solve yet."
        )
    return path


def _read_value_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "t" in df.columns:
        df["t"] = df["t"].astype(int)
    return df


def load_modeled_years(case_dir: Path) -> list[int]:
    path = _require_file(
        case_dir / "inputs_case" / "modeledyears.csv",
        "ReEDS's case-setup step (writes inputs_case/modeledyears.csv)",
    )
    with open(path, encoding="utf-8") as f:
        years = [int(tok) for tok in f.read().strip().split(",") if tok.strip()]
    return sorted(years)


def load_hierarchy(case_dir: Path) -> pd.DataFrame:
    path = _require_file(
        case_dir / "outputs" / "hierarchy.csv",
        "ReEDS's reporting step (writes outputs/hierarchy.csv)",
    )
    raw = pd.read_csv(path, usecols=["st", "country"])
    return raw.drop_duplicates(subset="st").reset_index(drop=True)


def load_datacenter_outputs(case_dir: Path) -> DatacenterCaseData:
    case_dir = Path(case_dir)
    outputs_dir = case_dir / "outputs"

    for fname in REQUIRED_OUTPUTS:
        if fname == "hierarchy.csv":
            continue
        _require_file(
            outputs_dir / fname,
            "input_processing/datacenter.py + ReEDS's data-center reporting step",
        )

    it_inv = _read_value_csv(outputs_dir / "datacenter_it_inv.csv")
    it_cap = _read_value_csv(outputs_dir / "datacenter_it_cap.csv")
    hierarchy = load_hierarchy(case_dir)
    modeled_years = load_modeled_years(case_dir)

    load_st = None
    load_st_base = None
    load_st_path = outputs_dir / "datacenter_load_st.csv"
    load_st_base_path = outputs_dir / "datacenter_load_st_base.csv"
    if load_st_path.exists():
        load_st = _read_value_csv(load_st_path)
    if load_st_base_path.exists():
        load_st_base = _read_value_csv(load_st_base_path)

    return DatacenterCaseData(
        case_dir=case_dir,
        case_name=case_dir.name,
        it_inv=it_inv,
        it_cap=it_cap,
        hierarchy=hierarchy,
        modeled_years=modeled_years,
        load_st=load_st,
        load_st_base=load_st_base,
    )
