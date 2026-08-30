from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import logging
import math
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import xlwings as xw


logger = logging.getLogger(__name__)




JEDI_WORKBOOK_MAX_YEAR = 2030
MIN_OUTPUT_IMPACT_YEAR: int | None = None





SINGLE_EVENT_RUNTIME_TIMEOUT_SECONDS = 1800
BATCH_RUNTIME_TIMEOUT_SECONDS = 7200

REPO_ROOT = Path(__file__).resolve().parents[2]
EMPLOYMENT_ROOT = Path(__file__).resolve().parents[1]
JEDI_DIR = EMPLOYMENT_ROOT / "energy_system" / "jedi_us"
JEDI_CANADA_DIR = EMPLOYMENT_ROOT / "energy_system" / "jedi_canada"
AUXILIARY_DATA_DIR = EMPLOYMENT_ROOT / "energy_system" / "auxiliary_data"
RUNTIME_INPUTS_DIR = AUXILIARY_DATA_DIR / "runtime_inputs"
STATE_ABBR_PATH = RUNTIME_INPUTS_DIR / "State_Abbr.xlsx"
CANADA_LOOKUP_PATH = RUNTIME_INPUTS_DIR / "canada_state_lookup.xlsx"
REEDS_CONSTRUCTION_RENAME_PATH = RUNTIME_INPUTS_DIR / "ReEDS_construction_rename.xlsx"
NAMEPLATE_TABLE_PATH = RUNTIME_INPUTS_DIR / "all_cap_fac_scenarios_with_canada.csv"
GAS_LOCAL_SHARE_PATH = RUNTIME_INPUTS_DIR / "localshare_ng_state_with_canada.xlsx"
COAL_LOCAL_SHARE_PATH = RUNTIME_INPUTS_DIR / "localshare_coal_state_with_canada.xlsx"


SUPPORTED_JEDI_TECHS = {
    "utility pv",
    "rooftop pv",
    "ng-cc",
    "ng-ct",
    "hydro",
    "pumped storage",
    "land-based wind",
    "offshore wind",
    "biopower",
    "coal",
    "csp",
    "geothermal",
}


MAPPED_BUT_UNSUPPORTED_REASONS = {
    "oil-gas-steam": "available petroleum workbook is not the user-modified OGS workbook referenced by LoCaTED",
    "battery_li": "battery user-modified JEDI workbooks are missing from energy_system/jedi_us",
    "h2-ct": "H2 user-modified JEDI workbook is missing from energy_system/jedi_us",
    "nuclear": "nuclear user-modified JEDI workbook is missing from energy_system/jedi_us",
}


OFFSHORE_DEFAULT_LOCAL_SHARE_ROWS = {
    19: 6,
    20: 7,
    22: 9,
    23: 10,
    25: 12,
    26: 13,
    27: 14,
    30: 17,
    33: 20,
    36: 23,
    39: 26,
    42: 29,
    46: 33,
    49: 36,
    52: 39,
    57: 44,
    58: 45,
    60: 47,
    61: 48,
    63: 50,
    64: 51,
    66: 53,
    67: 54,
    69: 56,
    70: 57,
    72: 59,
    73: 60,
    75: 62,
    76: 63,
    78: 65,
    79: 66,
    80: 67,
    81: 68,
    82: 69,
    83: 70,
    84: 71,
    86: 75,
    87: 76,
    88: 77,
    89: 78,
    90: 79,
    91: 80,
    94: 82,
    97: 85,
    98: 86,
    99: 87,
    100: 88,
    101: 89,
    102: 90,
    109: 99,
    110: 100,
    111: 101,
    112: 102,
    114: 104,
    115: 105,
    116: 107,
    117: 108,
    118: 109,
}


CANADA_OFFSHORE_ANALYSIS_AREA_PROXY = "North Atlantic [Region]"
CANADA_OFFSHORE_PROJECT_AREA_PROXY = "North Atlantic [Region]"
CANADA_OFFSHORE_REGION_PROXY = "North Atlantic"
CANADA_OFFSHORE_WEATHER_PROXY = "North Atlantic"
ONWIND_RUNTIME_BATCH_SIZE = 32


@dataclass(frozen=True)
class ImpactBreakdownSpec:
    direct_range: str | None
    indirect_range: str
    induced_range: str
    total_range: str


@dataclass(frozen=True)
class SupportedWorkbook:
    workbook_name: str
    workbook_key: str
    summary_sheet: str
    construction_breakdown: ImpactBreakdownSpec
    operating_breakdown: ImpactBreakdownSpec
    money_scale: float
    writer: Callable[[xw.main.Book, dict], None]
    runner: Callable[[xw.main.Book, xw.main.App, dict], None] | None = None


IMPACT_EFFECTS = ("direct", "indirect", "induced", "total")
IMPACT_METRICS = ("jobs_fte", "earnings_usd", "output_usd", "value_added_usd")


def _impact_column(prefix: str, metric: str, effect: str, annual: bool = False) -> str:
    suffix = "_annual" if annual else ""
    return f"{prefix}_{metric}_{effect}{suffix}"


def _legacy_total_column(prefix: str, metric: str, annual: bool = False) -> str:
    suffix = "_annual" if annual else ""
    return f"{prefix}_{metric}{suffix}"


def _scale_impact_values(
    values: list[float], money_scale: float, scale: float
) -> list[float]:
    return [
        values[0] * scale,
        values[1] * money_scale * scale,
        values[2] * money_scale * scale,
        values[3] * money_scale * scale,
    ]


def _read_breakdown_range(sheet: xw.main.Sheet, range_name: str) -> list[float]:
    return [coerce_scalar(v) for v in sheet.range(range_name).value]


def read_impact_breakdown(
    book: xw.main.Book, spec: SupportedWorkbook, breakdown: ImpactBreakdownSpec
) -> dict[str, list[float]]:
    sheet = book.sheets[spec.summary_sheet]
    indirect = _read_breakdown_range(sheet, breakdown.indirect_range)
    induced = _read_breakdown_range(sheet, breakdown.induced_range)
    total = _read_breakdown_range(sheet, breakdown.total_range)
    if breakdown.direct_range is None:
        direct = [
            total[idx] - indirect[idx] - induced[idx] for idx in range(len(total))
        ]
    else:
        direct = _read_breakdown_range(sheet, breakdown.direct_range)
    return {
        "direct": direct,
        "indirect": indirect,
        "induced": induced,
        "total": total,
    }


def add_scaled_impact_columns(
    row: dict,
    prefix: str,
    breakdown: dict[str, list[float]],
    money_scale: float,
    scale: float,
    annual: bool = False,
) -> None:
    for effect, values in breakdown.items():
        scaled = _scale_impact_values(values, money_scale, scale)
        for idx, metric in enumerate(IMPACT_METRICS):
            row[_impact_column(prefix, metric, effect, annual=annual)] = scaled[idx]

    if annual:
        for metric in IMPACT_METRICS:
            row[_legacy_total_column(prefix, metric, annual=True)] = row[
                _impact_column(prefix, metric, "total", annual=True)
            ]


def read_effect_value(
    row: dict, prefix: str, metric: str, effect: str, annual: bool = False
) -> float:
    primary = _impact_column(prefix, metric, effect, annual=annual)
    if primary in row and pd.notna(row[primary]):
        return float(row[primary])
    if effect == "total":
        legacy = _legacy_total_column(prefix, metric, annual=annual)
        if legacy in row and pd.notna(row[legacy]):
            return float(row[legacy])
    return 0.0


def ensure_excel_pythonpath() -> None:
    local_appdata = Path(os.environ.get("LOCALAPPDATA", ""))
    required_paths = [
        JEDI_DIR / "jedi-onwind-model",
        JEDI_DIR / "jedi-offwind-model",
        JEDI_CANADA_DIR / "jedi-onwind-model",
        JEDI_CANADA_DIR / "jedi-offwind-model",
        local_appdata / "landbosse-runtime" / "Lib" / "site-packages",
    ]
    existing = [
        segment
        for segment in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if segment
    ]
    merged: list[str] = []
    seen: set[str] = set()
    for path in [str(p) for p in required_paths if p.exists()] + existing:
        if path not in seen:
            seen.add(path)
            merged.append(path)
    os.environ["PYTHONPATH"] = os.pathsep.join(merged)


def require_runtime_input(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Required runtime input missing: {path} ({description}). "
            "Expected under energy_system/auxiliary_data/runtime_inputs/."
        )
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a construction-only JEDI batch for a ReEDS case. "
            "This prototype currently supports US-only utility PV, rooftop PV, NG-CC, NG-CT, hydro, "
            "pumped storage, land-based wind, offshore wind, biopower, coal, CSP, and geothermal."
        )
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=REPO_ROOT / "runs_save" / "v0622_G0-F0-S0",
        help="Path to a ReEDS case directory that contains an outputs folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EMPLOYMENT_ROOT / "energy_system" / "results" / "v0622_G0-F0-S0",
        help="Directory for generated CSV outputs.",
    )
    parser.add_argument(
        "--nameplate-scenario",
        choices=["low", "mid", "high"],
        default="mid",
        help="Scenario column to use from LoCaTED all_cap_fac_scenarios.csv.",
    )
    parser.add_argument(
        "--money-year",
        type=int,
        default=2024,
        help=f"Dollar year written into supported JEDI workbooks. Keep <= {JEDI_WORKBOOK_MAX_YEAR} for workbook compatibility.",
    )
    parser.add_argument(
        "--min-output-impact-year",
        type=int,
        default=MIN_OUTPUT_IMPACT_YEAR,
        help=(
            "Optional minimum impact year to keep in annual construction/operating outputs. "
            "Omit to keep the full modeled range."
        ),
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional cap on the number of supported build events to run through Excel for validation.",
    )
    parser.add_argument(
        "--coverage-only",
        action="store_true",
        help="Only generate the build coverage report, without opening Excel.",
    )
    parser.add_argument(
        "--jedi-dir",
        type=Path,
        default=None,
        help=(
            "Optional workbook root to use for both US and Canada events. "
            "If omitted, the script prefers energy_system/jedi_canada for Canada-supported "
            "events and falls back to energy_system/jedi_us."
        ),
    )
    parser.add_argument(
        "--canada-lookup-path",
        type=Path,
        default=CANADA_LOOKUP_PATH,
        help="Canada lookup workbook under energy_system/auxiliary_data/runtime_inputs/.",
    )
    parser.add_argument(
        "--macro-batch-size",
        type=int,
        default=8,
        help="Number of macro-driven wind events to process per Excel session before restarting it.",
    )
    parser.add_argument(
        "--macro-workers",
        type=int,
        default=None,
        help="Number of parallel worker processes for macro-driven wind batches. Defaults to "
        "the local CPU count (auto-detected via os.cpu_count()) when not specified -- unlike "
        "--static-workers, this was left at its original auto-parallel default because the "
        "1-worker/~20-minute serial verification run never exercised this path (its cases had "
        "no macro-driven wind events), so serializing it is untested.",
    )
    parser.add_argument(
        "--static-workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (each with its own Excel session) for "
        "static (non-macro) events such as PV, gas, hydro, biopower, coal, CSP, geothermal. "
        "Defaults to 1 (serial) -- running more than one Excel COM session concurrently "
        "causes intermittent 'RPC server unavailable' failures on some events; pass a "
        "higher value (e.g. os.cpu_count()) to opt back into parallel execution.",
    )
    parser.add_argument(
        "--offshore-direct-runtime",
        action="store_true",
        help=(
            "For offshore wind only, bypass the Excel VBA RunORBIT macro and instead call "
            "energy_system/offshore_orbit_direct.py via the installed orbit-runtime Python."
        ),
    )
    parser.add_argument(
        "--include-transmission",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also run the NREL Transmission Line JEDI workbook for positive ReEDS "
            "interregional AC/LCC/VSC capacity additions. Distribution-system assets "
            "remain outside scope; US/Canada and cross-border routes are allocated to their "
            "endpoint state/province JEDI workbooks."
        ),
    )
    return parser


def map_reeds_tech(raw_tech: str) -> str | None:
    if raw_tech == "distpv":
        return "rooftop pv"
    if raw_tech.startswith("upv_"):
        return "utility pv"
    if raw_tech.startswith("wind-ons_"):
        return "land-based wind"
    if raw_tech.startswith("wind-ofs_"):
        return "offshore wind"
    if raw_tech == "pumped-hydro":
        return "pumped storage"
    if raw_tech.startswith("hyd"):
        return "hydro"
    if raw_tech.startswith("egs_") or raw_tech.startswith("geohydro_"):
        return "geothermal"
    if raw_tech.startswith("csp"):
        return "csp"
    if raw_tech.startswith("biopower"):
        return "biopower"
    if raw_tech.startswith("coal") or raw_tech.startswith("Coal"):
        return "coal"
    if raw_tech.startswith("gas-CC") and "CCS" not in raw_tech and "H2" not in raw_tech:
        return "ng-cc"
    if raw_tech == "Gas-CT":
        return "ng-ct"
    if raw_tech in {"H2-CT", "Gas-CT_H2-CT"}:
        return "h2-ct"
    if raw_tech.startswith("o-g-s"):
        return "oil-gas-steam"
    if raw_tech.startswith("nuclear"):
        return "nuclear"
    if raw_tech == "battery_li":
        return "battery_li"
    return None


def load_canada_lookup(canada_lookup_path: Path) -> pd.DataFrame:
    if not canada_lookup_path.exists():
        return pd.DataFrame(
            columns=[
                "st",
                "state_name",
                "state_caps",
                "country",
                "is_canada_template",
                "onwind_region_proxy",
                "onwind_default_state_proxy",
                "offwind_analysis_area_proxy",
                "offwind_project_area_proxy",
                "offwind_orbit_region_proxy",
                "offwind_weather_proxy",
                "proxy_notes",
            ]
        )
    canada_lookup = pd.read_excel(canada_lookup_path)
    rename_map = {
        "Abbr": "st",
        "State": "state_name",
        "Caps": "state_caps",
        "Country": "country",
        "JEDIName": "jedi_name",
    }
    for old, new in rename_map.items():
        if old in canada_lookup.columns:
            canada_lookup = canada_lookup.rename(columns={old: new})
    if "country" not in canada_lookup.columns:
        canada_lookup["country"] = "Canada"
    canada_lookup["is_canada_template"] = True
    for column in [
        "onwind_region_proxy",
        "onwind_default_state_proxy",
        "offwind_analysis_area_proxy",
        "offwind_project_area_proxy",
        "offwind_orbit_region_proxy",
        "offwind_weather_proxy",
        "proxy_notes",
    ]:
        if column not in canada_lookup.columns:
            canada_lookup[column] = ""
    return canada_lookup[
        [
            "st",
            "state_name",
            "state_caps",
            "country",
            "is_canada_template",
            "onwind_region_proxy",
            "onwind_default_state_proxy",
            "offwind_analysis_area_proxy",
            "offwind_project_area_proxy",
            "offwind_orbit_region_proxy",
            "offwind_weather_proxy",
            "proxy_notes",
        ]
    ]


def load_state_lookup(canada_lookup_path: Path) -> pd.DataFrame:
    state_abbr = pd.read_excel(
        require_runtime_input(STATE_ABBR_PATH, "US state abbreviation lookup")
    ).rename(columns={"Abbr": "st", "State": "state_name", "Caps": "state_caps"})
    state_abbr["country"] = "USA"
    state_abbr["is_canada_template"] = False
    for column in [
        "onwind_region_proxy",
        "onwind_default_state_proxy",
        "offwind_analysis_area_proxy",
        "offwind_project_area_proxy",
        "offwind_orbit_region_proxy",
        "offwind_weather_proxy",
        "proxy_notes",
    ]:
        state_abbr[column] = ""
    canada_lookup = load_canada_lookup(canada_lookup_path)
    combined = pd.concat(
        [
            state_abbr[
                [
                    "st",
                    "state_name",
                    "state_caps",
                    "country",
                    "is_canada_template",
                    "onwind_region_proxy",
                    "onwind_default_state_proxy",
                    "offwind_analysis_area_proxy",
                    "offwind_project_area_proxy",
                    "offwind_orbit_region_proxy",
                    "offwind_weather_proxy",
                    "proxy_notes",
                ]
            ],
            canada_lookup,
        ],
        ignore_index=True,
    )
    combined = combined.drop_duplicates(subset=["st"], keep="last")
    return combined


def canada_supported_techs(jedi_canada_dir: Path) -> set[str]:
    if not jedi_canada_dir.exists():
        return set()
    try:
        is_base_us_dir = jedi_canada_dir.resolve() == JEDI_DIR.resolve()
    except OSError:
        is_base_us_dir = jedi_canada_dir == JEDI_DIR
    techs: set[str] = set()
    for tech, spec in supported_workbooks().items():
        if is_base_us_dir and tech != "offshore wind":
            continue
        if tech == "offshore wind":
            if (jedi_canada_dir / spec.workbook_name).exists() or (
                jedi_canada_dir == JEDI_CANADA_DIR
                and (JEDI_DIR / spec.workbook_name).exists()
            ):
                techs.add(tech)
            continue
        if spec.workbook_key == "onwind":


            name_path = Path(spec.workbook_name)
            pattern = f"{name_path.stem}__*{name_path.suffix}"
            if any((jedi_canada_dir / name_path.parent).glob(pattern)):
                techs.add(tech)
            continue
        if (jedi_canada_dir / spec.workbook_name).exists():
            techs.add(tech)
    return techs


def load_case_outputs(
    case_dir: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
]:
    outputs_dir = case_dir / "outputs"
    hierarchy = pd.read_csv(
        outputs_dir / "hierarchy.csv", usecols=["r", "st", "country"]
    )
    cap_new = pd.read_csv(outputs_dir / "cap_new_ivrt.csv")
    cap = pd.read_csv(outputs_dir / "cap.csv")
    gen = pd.read_csv(outputs_dir / "gen_ann.csv")
    avg_cf_path = outputs_dir / "avg_cf.csv"
    avg_cf = pd.read_csv(avg_cf_path) if avg_cf_path.exists() else None
    cap_ivrt_path = outputs_dir / "cap_ivrt.csv"
    cap_ivrt = pd.read_csv(cap_ivrt_path) if cap_ivrt_path.exists() else None
    gen_ivrt_path = outputs_dir / "gen_ivrt.csv"
    gen_ivrt = pd.read_csv(gen_ivrt_path) if gen_ivrt_path.exists() else None
    return (
        hierarchy,
        cap_new,
        cap.merge(hierarchy, on="r", how="left"),
        gen.merge(hierarchy, on="r", how="left"),
        avg_cf,
        cap_ivrt.merge(hierarchy, on="r", how="left") if cap_ivrt is not None else None,
        gen_ivrt.merge(hierarchy, on="r", how="left") if gen_ivrt is not None else None,
    )


def build_capacity_factor_table_from_avg_cf(
    hierarchy: pd.DataFrame,
    cap_new: pd.DataFrame,
    avg_cf: pd.DataFrame,
) -> pd.DataFrame:
    avg_cf = avg_cf.copy()
    avg_cf["mapped_tech"] = avg_cf["i"].map(map_reeds_tech)
    avg_cf = avg_cf[avg_cf["mapped_tech"].notna()]
    avg_cf = avg_cf.merge(hierarchy, on="r", how="left")

    supported_raw = cap_new.merge(hierarchy, on="r", how="left").copy()
    supported_raw["mapped_tech"] = supported_raw["i"].map(map_reeds_tech)
    supported_raw = supported_raw[
        supported_raw["mapped_tech"].notna() & (supported_raw["Value"] > 0)
    ]

    merged = supported_raw.merge(
        avg_cf[["i", "v", "r", "t", "Value"]],
        on=["i", "v", "r", "t"],
        how="left",
        suffixes=("_build", "_avg_cf"),
    )
    n_missing_raw = int(merged["Value_avg_cf"].isna().sum())
    if n_missing_raw:
        logger.warning(
            "build_capacity_factor_table_from_avg_cf: %d of %d supported raw build rows "
            "had no direct (i, v, r, t) match in avg_cf.csv; those groups will use fallback CF logic",
            n_missing_raw,
            len(merged),
        )

    matched = merged.dropna(subset=["Value_avg_cf"]).copy()
    if matched.empty:
        return pd.DataFrame(
            columns=["st", "country", "mapped_tech", "t", "capacity_factor"]
        )

    matched["weighted_cf"] = matched["Value_build"] * matched["Value_avg_cf"]
    aggregated = (
        matched.groupby(["st", "country", "mapped_tech", "t"], as_index=False)[
            ["weighted_cf", "Value_build"]
        ]
        .sum()
        .rename(columns={"Value_build": "build_mw"})
    )
    aggregated["capacity_factor"] = aggregated["weighted_cf"] / aggregated["build_mw"]
    return aggregated[["st", "country", "mapped_tech", "t", "capacity_factor"]]


def build_capacity_factor_table_from_gas_ct_region_year_ivrt_proxy(
    hierarchy: pd.DataFrame,
    cap_new: pd.DataFrame,
    avg_cf: pd.DataFrame | None,
    cap_ivrt: pd.DataFrame | None,
    gen_ivrt: pd.DataFrame | None,
) -> pd.DataFrame:
    if cap_ivrt is None or gen_ivrt is None:
        return pd.DataFrame(
            columns=["st", "country", "mapped_tech", "t", "capacity_factor"]
        )

    supported_raw = cap_new.merge(hierarchy, on="r", how="left").copy()
    supported_raw["mapped_tech"] = supported_raw["i"].map(map_reeds_tech)
    supported_raw = supported_raw[
        supported_raw["mapped_tech"].eq("ng-ct")
        & supported_raw["i"].eq("Gas-CT")
        & supported_raw["Value"].gt(0)
    ]
    if supported_raw.empty:
        return pd.DataFrame(
            columns=["st", "country", "mapped_tech", "t", "capacity_factor"]
        )

    if avg_cf is not None:
        supported_raw = supported_raw.merge(
            avg_cf[["i", "v", "r", "t", "Value"]],
            on=["i", "v", "r", "t"],
            how="left",
            suffixes=("_build", "_avg_cf"),
        )
        supported_raw = supported_raw[supported_raw["Value_avg_cf"].isna()].copy()
    else:
        supported_raw = supported_raw.rename(columns={"Value": "Value_build"})

    direct_gen_ivrt = gen_ivrt[gen_ivrt["i"].eq("Gas-CT")][
        ["i", "v", "r", "t", "Value"]
    ].rename(columns={"Value": "gen_ivrt_mwh"})
    supported_raw = supported_raw.merge(
        direct_gen_ivrt, on=["i", "v", "r", "t"], how="left"
    )
    supported_raw = supported_raw[supported_raw["gen_ivrt_mwh"].isna()].copy()
    if supported_raw.empty:
        return pd.DataFrame(
            columns=["st", "country", "mapped_tech", "t", "capacity_factor"]
        )

    region_year_cap = (
        cap_ivrt[cap_ivrt["i"].eq("Gas-CT")]
        .groupby(["r", "t"], as_index=False)["Value"]
        .sum()
        .rename(columns={"Value": "region_year_cap_mw"})
    )
    region_year_gen = (
        gen_ivrt[gen_ivrt["i"].eq("Gas-CT")]
        .groupby(["r", "t"], as_index=False)["Value"]
        .sum()
        .rename(columns={"Value": "region_year_gen_mwh"})
    )
    supported_raw = supported_raw.merge(region_year_cap, on=["r", "t"], how="left")
    supported_raw = supported_raw.merge(region_year_gen, on=["r", "t"], how="left")
    supported_raw["region_year_gen_mwh"] = supported_raw["region_year_gen_mwh"].fillna(
        0.0
    )
    supported_raw["region_year_cap_mw"] = supported_raw["region_year_cap_mw"].fillna(
        0.0
    )

    proxy_rows = supported_raw[
        supported_raw["region_year_gen_mwh"].gt(0.0)
        & supported_raw["region_year_cap_mw"].gt(0.0)
    ].copy()
    if proxy_rows.empty:
        return pd.DataFrame(
            columns=["st", "country", "mapped_tech", "t", "capacity_factor"]
        )

    proxy_rows["capacity_factor"] = proxy_rows["region_year_gen_mwh"] / (
        proxy_rows["region_year_cap_mw"] * 8760.0
    )
    proxy_rows["weighted_cf"] = (
        proxy_rows["Value_build"] * proxy_rows["capacity_factor"]
    )
    aggregated = (
        proxy_rows.groupby(["st", "country", "mapped_tech", "t"], as_index=False)[
            ["weighted_cf", "Value_build"]
        ]
        .sum()
        .rename(columns={"Value_build": "build_mw"})
    )
    aggregated["capacity_factor"] = aggregated["weighted_cf"] / aggregated["build_mw"]
    logger.warning(
        "build_capacity_factor_table_from_gas_ct_region_year_ivrt_proxy: applying Gas-CT region-year ivrt proxy "
        "to %d raw build rows across %d aggregated groups",
        len(proxy_rows),
        len(aggregated),
    )
    return aggregated[["st", "country", "mapped_tech", "t", "capacity_factor"]]


def build_capacity_factor_table(
    hierarchy: pd.DataFrame,
    cap_new: pd.DataFrame,
    cap: pd.DataFrame,
    gen: pd.DataFrame,
    avg_cf: pd.DataFrame | None = None,
    cap_ivrt: pd.DataFrame | None = None,
    gen_ivrt: pd.DataFrame | None = None,
) -> pd.DataFrame:
    avg_cf_capacity_factors = pd.DataFrame(
        columns=["st", "country", "mapped_tech", "t", "capacity_factor"]
    )
    if avg_cf is not None:
        avg_cf_capacity_factors = build_capacity_factor_table_from_avg_cf(
            hierarchy, cap_new, avg_cf
        )
    gas_ct_region_year_proxy = (
        build_capacity_factor_table_from_gas_ct_region_year_ivrt_proxy(
            hierarchy,
            cap_new,
            avg_cf,
            cap_ivrt,
            gen_ivrt,
        )
    )

    fallback_cap = cap_ivrt.copy() if cap_ivrt is not None else cap.copy()
    fallback_gen = gen_ivrt.copy() if gen_ivrt is not None else gen.copy()
    fallback_label = (
        "cap_ivrt/gen_ivrt"
        if cap_ivrt is not None and gen_ivrt is not None
        else "cap/gen"
    )

    if (cap_ivrt is None) ^ (gen_ivrt is None):
        logger.warning(
            "build_capacity_factor_table: only one of cap_ivrt/gen_ivrt is available; "
            "falling back to cap/gen annual outputs for consistency"
        )
        fallback_cap = cap.copy()
        fallback_gen = gen.copy()
        fallback_label = "cap/gen"

    fallback_cap["mapped_tech"] = fallback_cap["i"].map(map_reeds_tech)
    fallback_gen["mapped_tech"] = fallback_gen["i"].map(map_reeds_tech)

    fallback_cap = fallback_cap[fallback_cap["mapped_tech"].notna()]
    fallback_gen = fallback_gen[fallback_gen["mapped_tech"].notna()]

    cap_state = (
        fallback_cap.groupby(["st", "country", "mapped_tech", "t"], as_index=False)[
            "Value"
        ]
        .sum()
        .rename(columns={"Value": "cap_mw"})
    )
    gen_state = (
        fallback_gen.groupby(["st", "country", "mapped_tech", "t"], as_index=False)[
            "Value"
        ]
        .sum()
        .rename(columns={"Value": "gen_mwh"})
    )

    merged = cap_state.merge(
        gen_state,
        on=["st", "country", "mapped_tech", "t"],
        how="outer",
        indicator=True,
    )
    n_unmatched = (merged["_merge"] != "both").sum()
    if n_unmatched:
        logger.warning(
            "build_capacity_factor_table: %d of %d (st, country, mapped_tech, t) rows "
            "had no match in %s data; treating missing cap_mw/gen_mwh as 0.0",
            n_unmatched,
            len(merged),
            fallback_label,
        )
    merged = merged.drop(columns="_merge").fillna({"cap_mw": 0.0, "gen_mwh": 0.0})

    merged["capacity_factor"] = 0.0
    positive_cap = merged["cap_mw"] > 0
    merged.loc[positive_cap, "capacity_factor"] = merged.loc[
        positive_cap, "gen_mwh"
    ] / (merged.loc[positive_cap, "cap_mw"] * 8760.0)
    merged["capacity_factor"] = merged["capacity_factor"].clip(lower=0.0)
    if avg_cf_capacity_factors.empty and gas_ct_region_year_proxy.empty:
        return merged

    preferred = avg_cf_capacity_factors.copy()
    if not gas_ct_region_year_proxy.empty:
        preferred = pd.concat([preferred, gas_ct_region_year_proxy], ignore_index=True)
        preferred = preferred.drop_duplicates(
            subset=["st", "country", "mapped_tech", "t"], keep="first"
        )

    combined = merged.merge(
        preferred,
        on=["st", "country", "mapped_tech", "t"],
        how="outer",
        suffixes=("_fallback", ""),
    )
    preferred_mask = combined["capacity_factor"].notna()
    combined["capacity_factor"] = combined["capacity_factor"].fillna(
        combined["capacity_factor_fallback"]
    )
    n_preferred = int(preferred_mask.sum())
    logger.info(
        "build_capacity_factor_table: using avg_cf.csv for %d aggregated (st, country, mapped_tech, t) rows "
        "and %s fallback for the remainder",
        n_preferred,
        fallback_label,
    )
    return combined[["st", "country", "mapped_tech", "t", "capacity_factor"]]


def load_build_years() -> pd.DataFrame:
    rename = pd.read_excel(
        require_runtime_input(
            REEDS_CONSTRUCTION_RENAME_PATH, "ReEDS-to-JEDI construction rename table"
        ),
        sheet_name="main",
    )
    build_years = pd.read_csv(
        REPO_ROOT / "inputs" / "financials" / "construction_times_default.csv"
    )
    return rename.merge(build_years, left_on="ref", right_on="i", how="left")


def load_nameplate_table(nameplate_scenario: str) -> pd.DataFrame:
    source_path = require_runtime_input(
        NAMEPLATE_TABLE_PATH, "US+Canada nameplate capacity table"
    )
    logger.info("load_nameplate_table: using runtime input at %s", source_path)
    nameplate = pd.read_csv(source_path)
    return nameplate.query("scenario == @nameplate_scenario").rename(
        columns={"value": "nameplate_value"}
    )


def load_gas_local_share() -> pd.DataFrame:
    source_path = require_runtime_input(
        GAS_LOCAL_SHARE_PATH, "US+Canada local gas share table"
    )
    logger.info("load_gas_local_share: using runtime input at %s", source_path)
    ng_share = pd.read_excel(source_path)
    value_col = ng_share.columns[-1]
    return ng_share.rename(columns={"Abbr": "st", value_col: "local_gas_pct"})[
        ["st", "local_gas_pct"]
    ]


def load_coal_local_share() -> pd.DataFrame:
    source_path = require_runtime_input(
        COAL_LOCAL_SHARE_PATH, "US+Canada local coal share table"
    )
    logger.info("load_coal_local_share: using runtime input at %s", source_path)
    coal_share = pd.read_excel(source_path)
    value_col = coal_share.columns[-1]
    return coal_share.rename(columns={"Abbr": "st", value_col: "local_coal_pct"})[
        ["st", "local_coal_pct"]
    ]


def build_coverage_report(
    hierarchy: pd.DataFrame,
    cap_new: pd.DataFrame,
    state_lookup: pd.DataFrame,
    supported_canada_techs: set[str],
) -> pd.DataFrame:
    report = cap_new.merge(hierarchy, on="r", how="left").copy()
    report["mapped_tech"] = report["i"].map(map_reeds_tech)
    report["status"] = "unsupported_raw_tech"
    report["status_reason"] = (
        "no ReEDS->JEDI technology mapping defined in this prototype"
    )

    mapped = report["mapped_tech"].notna()
    report.loc[mapped, "status"] = "mapped_but_unsupported"
    report.loc[mapped, "status_reason"] = (
        report.loc[mapped, "mapped_tech"]
        .map(MAPPED_BUT_UNSUPPORTED_REASONS)
        .fillna("mapped tech is not yet supported in this prototype")
    )

    us_supported = (
        mapped
        & report["mapped_tech"].isin(SUPPORTED_JEDI_TECHS)
        & (report["country"] == "USA")
    )
    report.loc[us_supported, "status"] = "supported"
    report.loc[us_supported, "status_reason"] = "supported by the current Excel adapter"

    report = report.merge(
        state_lookup[
            [
                "st",
                "state_name",
                "state_caps",
                "is_canada_template",
                "onwind_region_proxy",
                "onwind_default_state_proxy",
                "offwind_analysis_area_proxy",
                "offwind_project_area_proxy",
                "offwind_orbit_region_proxy",
                "offwind_weather_proxy",
                "proxy_notes",
            ]
        ],
        on="st",
        how="left",
    )
    report["is_canada_template"] = (
        report["is_canada_template"].fillna(False).astype(bool)
    )
    canadian = (
        mapped
        & report["mapped_tech"].isin(SUPPORTED_JEDI_TECHS)
        & (report["country"] != "USA")
    )
    canadian_known = canadian & report["is_canada_template"]
    canadian_supported = canadian_known & report["mapped_tech"].isin(
        supported_canada_techs
    )
    report.loc[canadian_supported, "status"] = "supported"
    report.loc[canadian_supported, "status_reason"] = (
        "supported by the Canada-enhanced Excel adapter"
    )
    canada_onwind = canadian_supported & report["mapped_tech"].eq("land-based wind")
    report.loc[canada_onwind, "status_reason"] = (
        "supported by Canada-enhanced onshore wind adapter (US proxy state/region defaults)"
    )

    canadian_known_unsupported = canadian_known & ~report["mapped_tech"].isin(
        supported_canada_techs
    )
    report.loc[canadian_known_unsupported, "status"] = "mapped_but_unsupported"
    report.loc[canadian_known_unsupported, "status_reason"] = (
        "Canada lookup exists for this province, but no Canada-enhanced workbook exists for this tech"
    )
    canada_offwind = canadian_supported & report["mapped_tech"].eq("offshore wind")
    canada_offwind_missing_proxy = canada_offwind & (
        report["offwind_analysis_area_proxy"].fillna("").eq("")
        | report["offwind_project_area_proxy"].fillna("").eq("")
        | report["offwind_orbit_region_proxy"].fillna("").eq("")
        | report["offwind_weather_proxy"].fillna("").eq("")
    )
    report.loc[canada_offwind_missing_proxy, "status"] = "mapped_but_unsupported"
    report.loc[canada_offwind_missing_proxy, "status_reason"] = (
        "Canada offshore proxy is missing for this province; define province-specific offshore proxy inputs first"
    )
    canada_offwind = canada_offwind & ~canada_offwind_missing_proxy
    report.loc[canada_offwind, "status_reason"] = (
        "supported by province-specific offshore proxy using US offshore workbook inputs"
    )

    canadian_missing = canadian & ~report["is_canada_template"]
    report.loc[canadian_missing, "status"] = "unsupported_geography"
    report.loc[canadian_missing, "status_reason"] = (
        "Canada province lookup is missing; fill and enable the matching CA_*.xlsx template first"
    )

    return report


def prepare_supported_builds(
    coverage: pd.DataFrame,
    capacity_factors: pd.DataFrame,
    build_years: pd.DataFrame,
    nameplates: pd.DataFrame,
    gas_share: pd.DataFrame,
    coal_share: pd.DataFrame,
    state_lookup: pd.DataFrame,
) -> pd.DataFrame:
    supported = coverage.query("status == 'supported' and Value > 0").copy()
    supported = (
        supported.groupby(["st", "country", "mapped_tech", "t"], as_index=False)[
            "Value"
        ]
        .sum()
        .rename(
            columns={"Value": "build_mw", "mapped_tech": "tech", "t": "online_year"}
        )
    )
    supported = supported.merge(
        state_lookup[
            [
                "st",
                "state_caps",
                "state_name",
                "onwind_region_proxy",
                "onwind_default_state_proxy",
                "offwind_analysis_area_proxy",
                "offwind_project_area_proxy",
                "offwind_orbit_region_proxy",
                "offwind_weather_proxy",
                "proxy_notes",
            ]
        ],
        on="st",
        how="left",
    )
    supported = supported.merge(
        capacity_factors.rename(columns={"mapped_tech": "tech", "t": "online_year"})[
            ["st", "country", "tech", "online_year", "capacity_factor"]
        ],
        on=["st", "country", "tech", "online_year"],
        how="left",
    )

    tech_build = build_years.rename(columns={"construction_time": "build_years"})
    tech_build = tech_build[["i_x", "t_online", "build_years"]].rename(
        columns={"i_x": "tech", "t_online": "online_year"}
    )
    supported = supported.merge(tech_build, on=["tech", "online_year"], how="left")

    supported = supported.merge(
        nameplates.rename(columns={"state": "st", "tech": "tech"})[
            ["st", "tech", "nameplate_value"]
        ],
        on=["st", "tech"],
        how="left",
    )
    supported = supported.merge(gas_share, on="st", how="left")
    supported = supported.merge(coal_share, on="st", how="left")

    supported["capacity_factor"] = supported["capacity_factor"].fillna(0.0)
    supported["build_years"] = (
        supported["build_years"].fillna(1).astype(int).clip(lower=1)
    )
    n_nameplate_fallback = int(supported["nameplate_value"].isna().sum())
    if n_nameplate_fallback:
        logger.warning(
            "prepare_supported_builds: %d of %d events have no (st, tech) match in the nameplate "
            "capacity table and will use nameplate_value == build_mw (the entire aggregated build "
            "treated as one project) -- check energy_system/auxiliary_data/runtime_inputs if these are Canadian events",
            n_nameplate_fallback,
            len(supported),
        )
    supported["nameplate_value"] = supported["nameplate_value"].fillna(
        supported["build_mw"]
    )


    supported["construction_start_year"] = (
        supported["online_year"] - supported["build_years"]
    )
    supported["construction_start_year_workbook"] = supported[
        "construction_start_year"
    ].clip(upper=JEDI_WORKBOOK_MAX_YEAR)
    supported["workbook_year_is_proxy"] = (
        supported["construction_start_year_workbook"]
        != supported["construction_start_year"]
    )
    n_proxy_years = int(supported["workbook_year_is_proxy"].sum())
    if n_proxy_years:
        logger.warning(
            "prepare_supported_builds: %d of %d events have construction_start_year > %d "
            "and were clipped to fit JEDI's workbook lookup tables (see workbook_year_is_proxy column)",
            n_proxy_years,
            len(supported),
            JEDI_WORKBOOK_MAX_YEAR,
        )
    supported["const_period_months"] = supported["build_years"] * 12
    gas_events = supported["tech"].isin({"ng-cc", "ng-ct"})
    n_gas_fallback = int((gas_events & supported["local_gas_pct"].isna()).sum())
    if n_gas_fallback:
        logger.warning(
            "prepare_supported_builds: %d of %d gas events have no local_gas_pct match and will use "
            "local_gas_share == 0.0 (assumes 0%% in-state/in-province gas production) -- see "
            "energy_system/auxiliary_data/runtime_inputs if these are Canadian events",
            n_gas_fallback,
            int(gas_events.sum()),
        )
    coal_events = supported["tech"].eq("coal")
    n_coal_fallback = int((coal_events & supported["local_coal_pct"].isna()).sum())
    if n_coal_fallback:
        logger.warning(
            "prepare_supported_builds: %d of %d coal events have no local_coal_pct match and will use "
            "local_coal_share == 0.0 (assumes 0%% in-state/in-province coal production) -- see "
            "energy_system/auxiliary_data/runtime_inputs if these are Canadian events",
            n_coal_fallback,
            int(coal_events.sum()),
        )
    supported["local_gas_share"] = supported["local_gas_pct"].fillna(0.0) / 100.0
    supported["local_coal_share"] = supported["local_coal_pct"].fillna(0.0) / 100.0
    supported["project_scale_factor"] = 1.0
    supported["proxy_region_used"] = ""
    supported["weather_proxy_used"] = ""
    canada_onwind = (supported["country"] != "USA") & supported["tech"].eq(
        "land-based wind"
    )
    supported.loc[canada_onwind, "proxy_region_used"] = supported.loc[
        canada_onwind, "onwind_region_proxy"
    ]
    supported["onwind_province_state_caps"] = ""
    if canada_onwind.any():


















        us_caps_by_state_name = (
            state_lookup.loc[
                state_lookup["country"] == "USA", ["state_name", "state_caps"]
            ]
            .drop_duplicates(subset="state_name")
            .set_index("state_name")["state_caps"]
        )
        proxy_caps = supported.loc[canada_onwind, "onwind_default_state_proxy"].map(
            us_caps_by_state_name
        )
        n_missing_proxy = int(proxy_caps.isna().sum())
        if n_missing_proxy:
            logger.warning(
                "prepare_supported_builds: %d of %d Canada onshore wind events have no "
                "onwind_default_state_proxy resolvable against the US state lookup; falling back "
                "to the province's own state_caps, which the JEDI onshore wind workbook will "
                "silently approximate-match to an unrelated US state -- see "
                "energy_system/auxiliary_data/runtime_inputs/canada_state_lookup.xlsx",
                n_missing_proxy,
                int(canada_onwind.sum()),
            )
        supported.loc[canada_onwind, "onwind_province_state_caps"] = supported.loc[
            canada_onwind, "state_caps"
        ]
        supported.loc[canada_onwind, "state_caps"] = proxy_caps.fillna(
            supported.loc[canada_onwind, "state_caps"]
        )
    canada_offwind = (supported["country"] != "USA") & supported["tech"].eq(
        "offshore wind"
    )
    supported.loc[canada_offwind, "proxy_region_used"] = supported.loc[
        canada_offwind, "offwind_orbit_region_proxy"
    ]
    supported.loc[canada_offwind, "weather_proxy_used"] = supported.loc[
        canada_offwind, "offwind_weather_proxy"
    ]
    offshore_mask = supported["tech"].eq("offshore wind")
    supported.loc[offshore_mask, "project_scale_factor"] = (
        supported.loc[offshore_mask, "build_mw"] / 900.0
    )
    avg_facility_mask = supported["tech"].isin(
        {
            "ng-cc",
            "ng-ct",
            "hydro",
            "pumped storage",
            "land-based wind",
            "biopower",
            "coal",
            "csp",
            "geothermal",
        }
    )
    supported.loc[avg_facility_mask, "project_scale_factor"] = (
        supported.loc[avg_facility_mask, "build_mw"]
        / supported.loc[avg_facility_mask, "nameplate_value"]
    )
    return supported


def write_pv_inputs(book: xw.main.Book, event: dict) -> None:
    sheet = book.sheets["ProjectData"]
    average_system_kw = float(event["nameplate_value"])
    number_of_systems = (float(event["build_mw"]) * 1000.0) / average_system_kw
    sheet.range("B12").value = event["state_caps"]
    sheet.range("B14").value = int(event["construction_start_year_workbook"])
    sheet.range("B15").value = (
        "Residential" if event["tech"] == "rooftop pv" else "Utility"
    )
    sheet.range("B18").value = average_system_kw
    sheet.range("B19").value = number_of_systems
    sheet.range("B24").value = int(event["money_year"])


def write_ngas_inputs(book: xw.main.Book, event: dict) -> None:
    sheet = book.sheets["ProjectData"]
    sheet.range("B13").value = event["state_caps"]
    sheet.range("B14").value = int(event["construction_start_year_workbook"])
    sheet.range("B15").value = float(event["nameplate_value"])
    sheet.range("B16").value = float(event["capacity_factor"])
    sheet.range("B18").value = int(event["const_period_months"])
    sheet.range("B21").value = float(event["local_gas_share"])
    sheet.range("B24").value = int(event["money_year"])


def write_hydro_inputs(book: xw.main.Book, event: dict) -> None:
    sheet = book.sheets["ProjectData"]



    sheet.range("B22").value = "Simple"
    sheet.range("B15").value = event["state_name"]
    sheet.range("B17").value = int(event["construction_start_year_workbook"])
    sheet.range("B18").value = (
        "Pumped Storage" if event["tech"] == "pumped storage" else "New Site"
    )
    sheet.range("B19").value = float(event["nameplate_value"])
    sheet.range("B20").value = int(event["money_year"])
    sheet.range("B28").value = int(event["const_period_months"])
    sheet.range("B35").value = float(event["capacity_factor"])


def write_onshore_wind_inputs(book: xw.main.Book, event: dict) -> None:
    sheet = book.sheets["STEP 1 - Project Information"]
    sheet.range("C6").value = event["state_caps"]



    province_caps = str(event.get("onwind_province_state_caps") or "").strip()
    if province_caps and "Canada Wage Index" in [s.name for s in book.sheets]:
        book.sheets["Canada Wage Index"].range("B3").value = province_caps
    sheet.range("C8").value = float(event["nameplate_value"])
    sheet.range("C20").value = int(event["construction_start_year_workbook"])
    sheet.range("C21").value = int(event["money_year"])


def run_onshore_wind_model(book: xw.main.Book, app: xw.main.App, event: dict) -> None:
    _ = event
    app.api.Run(f"'{book.name}'!GoToStep2")


def apply_offshore_default_local_shares(book: xw.main.Book) -> None:
    default_sheet = book.sheets["Default Local"]
    local_sheet = book.sheets["Local Share - Step 3"]


    for target_row, source_row in OFFSHORE_DEFAULT_LOCAL_SHARE_ROWS.items():
        local_sheet.range(f"C{target_row}").value = default_sheet.range(
            f"D{source_row}"
        ).value


def map_offshore_analysis_area(state_name: str) -> str:
    if state_name == "New York":
        return "New York - Atlantic Coast"
    return state_name


def resolve_offshore_area_inputs(event: dict) -> tuple[str, str]:
    if str(event.get("country", "")).lower() != "usa":
        analysis_area = str(event.get("offwind_analysis_area_proxy") or "").strip()
        project_area = str(
            event.get("offwind_project_area_proxy") or analysis_area
        ).strip()
        if not analysis_area or not project_area:
            raise ValueError(
                f"Missing offshore proxy area for Canada event: province={event.get('st')} state_name={event.get('state_name')}"
            )
        return (analysis_area, project_area)
    analysis_area = str(
        event.get("offwind_analysis_area_proxy")
        or map_offshore_analysis_area(str(event["state_name"]))
    )
    project_area = str(event.get("offwind_project_area_proxy") or analysis_area)
    return (analysis_area, project_area)


def write_offshore_wind_inputs(book: xw.main.Book, event: dict) -> None:
    sheet = book.sheets["Project Data - Step 1"]
    analysis_area, project_area = resolve_offshore_area_inputs(event)
    sheet.range("D19").value = analysis_area
    sheet.range("D20").value = project_area
    sheet.range("D21").value = int(event["construction_start_year_workbook"])
    sheet.range("D22").value = int(event["money_year"])


def run_offshore_wind_model(book: xw.main.Book, app: xw.main.App, event: dict) -> None:
    _ = event
    app.api.Run(f"'{book.name}'!RunORBIT")
    apply_offshore_default_local_shares(book)


def resolve_orbit_runtime_python() -> Path | None:
    local = Path(os.environ.get("LOCALAPPDATA", "")) / "orbit-runtime" / "python.exe"
    return local if local.exists() else None


def resolve_landbosse_runtime_python() -> Path | None:
    local = (
        Path(os.environ.get("LOCALAPPDATA", "")) / "landbosse-runtime" / "python.exe"
    )
    return local if local.exists() else None


def validate_state_caps(value: object) -> str:
    state_caps = str(value).strip()
    if not state_caps or state_caps.lower() in ("nan", "none"):
        raise ValueError(f"Invalid state_caps value for JEDI workbook event: {value!r}")
    return state_caps


def run_onshore_runtime_subprocess(
    runtime_python: Path,
    workbook_path: Path,
    event: dict,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    helper_script = EMPLOYMENT_ROOT / "energy_system" / "onshore_landbosse_direct.py"
    temp_dir = Path(tempfile.mkdtemp(prefix="onshore_landbosse_direct_"))
    try:
        temp_workbook = temp_dir / workbook_path.name
        shutil.copy2(workbook_path, temp_workbook)

        cmd = [
            str(runtime_python),
            str(helper_script),
            "--workbook",
            str(temp_workbook),
            "--state-caps",
            validate_state_caps(event["state_caps"]),
            "--nameplate-value",
            str(float(event["nameplate_value"])),
            "--construction-year",
            str(int(event["construction_start_year_workbook"])),
            "--money-year",
            str(int(event["money_year"])),
        ]

        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SINGLE_EVENT_RUNTIME_TIMEOUT_SECONDS,
            check=False,
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "onshore_landbosse_direct failed: "
                f"returncode={proc.returncode} stdout={proc.stdout[-4000:]} stderr={proc.stderr[-4000:]}"
            )

        payload_line = None
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if line.startswith("{") and '"construction_breakdown"' in line:
                payload_line = line
                break
        if payload_line is None:
            raise RuntimeError(
                f"onshore_landbosse_direct returned no summary JSON. stdout={proc.stdout[-4000:]}"
            )

        payload = json.loads(payload_line)
        construction_breakdown = {
            effect: [
                coerce_scalar(v) for v in payload["construction_breakdown"][effect]
            ]
            for effect in IMPACT_EFFECTS
        }
        operating_breakdown = {
            effect: [coerce_scalar(v) for v in payload["operating_breakdown"][effect]]
            for effect in IMPACT_EFFECTS
        }
        return construction_breakdown, operating_breakdown
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_onshore_runtime_batch_subprocess(
    runtime_python: Path,
    workbook_path: Path,
    events: list[dict],
) -> tuple[list[dict], list[dict]]:
    helper_script = EMPLOYMENT_ROOT / "energy_system" / "onshore_landbosse_direct.py"
    temp_dir = Path(tempfile.mkdtemp(prefix="onshore_landbosse_batch_"))
    try:
        temp_workbook = temp_dir / workbook_path.name
        temp_input = temp_dir / "batch_input.json"
        shutil.copy2(workbook_path, temp_workbook)
        batch_payload = [
            {
                "event_id": idx,
                "state_caps": validate_state_caps(event["state_caps"]),
                "nameplate_value": float(event["nameplate_value"]),
                "construction_year": int(event["construction_start_year_workbook"]),
                "money_year": int(event["money_year"]),
                "project_scale_factor": float(event.get("project_scale_factor", 1.0)),
            }
            for idx, event in enumerate(events)
        ]
        temp_input.write_text(json.dumps(batch_payload), encoding="utf-8")

        cmd = [
            str(runtime_python),
            str(helper_script),
            "--workbook",
            str(temp_workbook),
            "--batch-input",
            str(temp_input),
        ]

        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=BATCH_RUNTIME_TIMEOUT_SECONDS,
            check=False,
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "onshore_landbosse_batch failed: "
                f"returncode={proc.returncode} stdout={proc.stdout[-4000:]} stderr={proc.stderr[-4000:]}"
            )

        payload_line = None
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if line.startswith("{") and '"results"' in line:
                payload_line = line
                break
        if payload_line is None:
            raise RuntimeError(
                f"onshore_landbosse_batch returned no result JSON. stdout={proc.stdout[-4000:]}"
            )

        payload = json.loads(payload_line)
        raw_rows: list[dict] = []
        error_rows: list[dict] = []
        for item in payload["results"]:
            event = events[int(item["event_id"])]
            if "error" in item:
                error_rows.append({**event, "error": item["error"]})
                continue
            construction_breakdown = {
                effect: [
                    coerce_scalar(v) for v in item["construction_breakdown"][effect]
                ]
                for effect in IMPACT_EFFECTS
            }
            operating_breakdown = {
                effect: [coerce_scalar(v) for v in item["operating_breakdown"][effect]]
                for effect in IMPACT_EFFECTS
            }
            row_event = dict(event)
            scale_override = item.get("project_scale_factor_override")
            if scale_override is not None:
                row_event["project_scale_factor"] = float(scale_override)
            raw_row = build_result_row(
                spec=supported_workbooks()["land-based wind"],
                event=row_event,
                construction_breakdown=construction_breakdown,
                operating_breakdown=operating_breakdown,
            )
            if item.get("fallback_proxy_cap_mw") is not None:
                raw_row["nameplate_value_fallback_proxy_mw"] = float(
                    item["fallback_proxy_cap_mw"]
                )
                raw_row["fallback_trigger_error"] = item.get(
                    "fallback_trigger_error", ""
                )
            raw_rows.append(raw_row)
        return raw_rows, error_rows
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_onshore_event_via_runtime(
    spec: SupportedWorkbook,
    event: dict,
    preferred_jedi_dir: Path | None = None,
) -> dict:
    runtime_python = resolve_landbosse_runtime_python()
    if runtime_python is None:
        raise FileNotFoundError(
            "landbosse-runtime/python.exe not found under LOCALAPPDATA"
        )

    source_workbook = resolve_workbook_path(spec, event, preferred_jedi_dir)
    original_nameplate = float(event["nameplate_value"])
    try:
        construction_breakdown, operating_breakdown = run_onshore_runtime_subprocess(
            runtime_python, source_workbook, event
        )
        return build_result_row(
            spec, event, construction_breakdown, operating_breakdown
        )
    except Exception:
        fallback_cap_mw = 3000.0
        if original_nameplate <= fallback_cap_mw:
            raise
        fallback_event = dict(event)
        fallback_event["nameplate_value"] = fallback_cap_mw
        fallback_event["project_scale_factor"] = float(
            event.get("project_scale_factor", 1.0)
        ) * (original_nameplate / fallback_cap_mw)
        construction_breakdown, operating_breakdown = run_onshore_runtime_subprocess(
            runtime_python, source_workbook, fallback_event
        )
        result = build_result_row(
            spec, event, construction_breakdown, operating_breakdown
        )
        result["project_scale_factor"] = fallback_event["project_scale_factor"]
        result["nameplate_value_fallback_proxy_mw"] = fallback_cap_mw
        return result


def run_offshore_event_via_runtime(
    spec: SupportedWorkbook,
    event: dict,
    preferred_jedi_dir: Path | None = None,
) -> dict:
    runtime_python = resolve_orbit_runtime_python()
    if runtime_python is None:
        raise FileNotFoundError("orbit-runtime/python.exe not found under LOCALAPPDATA")

    source_workbook = resolve_workbook_path(spec, event, preferred_jedi_dir)
    helper_script = EMPLOYMENT_ROOT / "energy_system" / "offshore_orbit_direct.py"
    data_library = runtime_python.parent / "data"
    temp_dir = Path(tempfile.mkdtemp(prefix="offshore_orbit_direct_"))
    try:
        temp_workbook = temp_dir / source_workbook.name
        shutil.copy2(source_workbook, temp_workbook)

        cmd = [
            str(runtime_python),
            str(helper_script),
            "--workbook",
            str(temp_workbook),
            "--analysis-area",
            resolve_offshore_area_inputs(event)[0],
            "--project-area",
            resolve_offshore_area_inputs(event)[1],
            "--construction-year",
            str(int(event["construction_start_year_workbook"])),
            "--money-year",
            str(int(event["money_year"])),
            "--data-library",
            str(data_library),
        ]

        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SINGLE_EVENT_RUNTIME_TIMEOUT_SECONDS,
            check=False,
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "offshore_orbit_direct failed: "
                f"returncode={proc.returncode} stdout={proc.stdout[-4000:]} stderr={proc.stderr[-4000:]}"
            )

        payload_line = None
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if line.startswith("{") and '"construction_breakdown"' in line:
                payload_line = line
                break
        if payload_line is None:
            raise RuntimeError(
                f"offshore_orbit_direct returned no summary JSON. stdout={proc.stdout[-4000:]}"
            )

        payload = json.loads(payload_line)
        construction_breakdown = {
            effect: [
                coerce_scalar(v) for v in payload["construction_breakdown"][effect]
            ]
            for effect in IMPACT_EFFECTS
        }
        operating_breakdown = {
            effect: [coerce_scalar(v) for v in payload["operating_breakdown"][effect]]
            for effect in IMPACT_EFFECTS
        }
        return build_result_row(
            spec, event, construction_breakdown, operating_breakdown
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def write_biopower_inputs(book: xw.main.Book, event: dict) -> None:
    sheet = book.sheets["ProjectData"]
    sheet.range("B14").value = event["state_caps"]
    sheet.range("B16").value = int(event["construction_start_year_workbook"])
    sheet.range("B17").value = int(event["const_period_months"])
    sheet.range("B18").value = float(event["nameplate_value"])
    sheet.range("B20").value = float(event["capacity_factor"])
    sheet.range("B33").value = int(event["money_year"])


def write_coal_inputs(book: xw.main.Book, event: dict) -> None:
    sheet = book.sheets["ProjectData"]
    sheet.range("B13").value = event["state_caps"]
    sheet.range("B15").value = int(event["construction_start_year_workbook"])
    sheet.range("B16").value = float(event["nameplate_value"])
    sheet.range("B17").value = float(event["capacity_factor"])
    sheet.range("B19").value = int(event["const_period_months"])
    sheet.range("B22").value = float(event["local_coal_share"])
    sheet.range("B25").value = int(event["money_year"])





    sheet.range("B27").value = "N"


def write_csp_inputs(book: xw.main.Book, event: dict) -> None:
    sheet = book.sheets["ProjectData"]
    sheet.range("B14").value = event["state_caps"]
    sheet.range("B17").value = int(event["construction_start_year_workbook"])
    sheet.range("B18").value = float(event["nameplate_value"])
    sheet.range("B20").value = float(event["capacity_factor"])
    sheet.range("B23").value = int(event["money_year"])


def write_geothermal_inputs(book: xw.main.Book, event: dict) -> None:
    sheet = book.sheets["ProjectData"]
    sheet.range("B13").value = event["state_caps"]
    sheet.range("B14").value = int(event["construction_start_year_workbook"])
    sheet.range("B15").value = int(event["const_period_months"])
    sheet.range("B16").value = float(event["nameplate_value"])
    sheet.range("B31").value = int(event["money_year"])


def supported_workbooks() -> dict[str, SupportedWorkbook]:
    return {
        "utility pv": SupportedWorkbook(
            workbook_name="jedi-pv-model.xlsm",
            workbook_key="pv",
            summary_sheet="SummaryResults",
            construction_breakdown=ImpactBreakdownSpec(
                "B30:E30", "B38:E38", "B39:E39", "B40:E40"
            ),
            operating_breakdown=ImpactBreakdownSpec(
                "B46:E46", "B47:E47", "B48:E48", "B49:E49"
            ),
            money_scale=1000.0,
            writer=write_pv_inputs,
        ),
        "rooftop pv": SupportedWorkbook(
            workbook_name="jedi-pv-model.xlsm",
            workbook_key="pv",
            summary_sheet="SummaryResults",
            construction_breakdown=ImpactBreakdownSpec(
                "B30:E30", "B38:E38", "B39:E39", "B40:E40"
            ),
            operating_breakdown=ImpactBreakdownSpec(
                "B46:E46", "B47:E47", "B48:E48", "B49:E49"
            ),
            money_scale=1000.0,
            writer=write_pv_inputs,
        ),
        "ng-cc": SupportedWorkbook(
            workbook_name="jedi-ngas-model.xlsm",
            workbook_key="ngas",
            summary_sheet="SummaryResults",
            construction_breakdown=ImpactBreakdownSpec(
                None, "B30:E30", "B31:E31", "B32:E32"
            ),
            operating_breakdown=ImpactBreakdownSpec(
                "B35:E35", "B36:E36", "B37:E37", "B38:E38"
            ),
            money_scale=1_000_000.0,
            writer=write_ngas_inputs,
        ),
        "ng-ct": SupportedWorkbook(
            workbook_name="jedi-ngas-model.xlsm",
            workbook_key="ngas",
            summary_sheet="SummaryResults",
            construction_breakdown=ImpactBreakdownSpec(
                None, "B30:E30", "B31:E31", "B32:E32"
            ),
            operating_breakdown=ImpactBreakdownSpec(
                "B35:E35", "B36:E36", "B37:E37", "B38:E38"
            ),
            money_scale=1_000_000.0,
            writer=write_ngas_inputs,
        ),
        "hydro": SupportedWorkbook(
            workbook_name="jedi-chydro-model.xlsm",
            workbook_key="hydro",
            summary_sheet="SummaryResults",
            construction_breakdown=ImpactBreakdownSpec(
                "B25:E25", "B26:E26", "B27:E27", "B28:E28"
            ),
            operating_breakdown=ImpactBreakdownSpec(
                "B34:E34", "B35:E35", "B36:E36", "B37:E37"
            ),
            money_scale=1_000_000.0,
            writer=write_hydro_inputs,
        ),
        "pumped storage": SupportedWorkbook(
            workbook_name="jedi-chydro-model.xlsm",
            workbook_key="hydro",
            summary_sheet="SummaryResults",
            construction_breakdown=ImpactBreakdownSpec(
                "B25:E25", "B26:E26", "B27:E27", "B28:E28"
            ),
            operating_breakdown=ImpactBreakdownSpec(
                "B34:E34", "B35:E35", "B36:E36", "B37:E37"
            ),
            money_scale=1_000_000.0,
            writer=write_hydro_inputs,
        ),
        "land-based wind": SupportedWorkbook(
            workbook_name="jedi-onwind-model/jedi-lbw-model.xlsm",
            workbook_key="onwind",
            summary_sheet="SummaryResults",
            construction_breakdown=ImpactBreakdownSpec(
                "B28:E28", "B29:E29", "B30:E30", "B31:E31"
            ),
            operating_breakdown=ImpactBreakdownSpec(
                "B34:E34", "B35:E35", "B36:E36", "B37:E37"
            ),
            money_scale=1_000_000.0,
            writer=write_onshore_wind_inputs,
            runner=run_onshore_wind_model,
        ),
        "offshore wind": SupportedWorkbook(
            workbook_name="jedi-offwind-model/jedi-osw-model.xlsm",
            workbook_key="offwind",
            summary_sheet="Economic Impact Results",
            construction_breakdown=ImpactBreakdownSpec(
                "D42:G42", "D53:G53", "D52:G52", "D54:G54"
            ),
            operating_breakdown=ImpactBreakdownSpec(
                "D59:G59", "D60:G60", "D61:G61", "D62:G62"
            ),
            money_scale=1_000_000.0,
            writer=write_offshore_wind_inputs,
            runner=run_offshore_wind_model,
        ),
        "biopower": SupportedWorkbook(
            workbook_name="jedi-biopower-model.xlsm",
            workbook_key="biopower",
            summary_sheet="SummaryResults",
            construction_breakdown=ImpactBreakdownSpec(
                "B43:E43", "B44:E44", "B45:E45", "B46:E46"
            ),
            operating_breakdown=ImpactBreakdownSpec(
                "B51:E51", "B55:E55", "B56:E56", "B57:E57"
            ),
            money_scale=1_000_000.0,
            writer=write_biopower_inputs,
        ),
        "coal": SupportedWorkbook(
            workbook_name="jedi-coal-model.xlsm",
            workbook_key="coal",
            summary_sheet="SummaryResults",
            construction_breakdown=ImpactBreakdownSpec(
                None, "B30:E30", "B31:E31", "B32:E32"
            ),
            operating_breakdown=ImpactBreakdownSpec(
                "B35:E35", "B36:E36", "B37:E37", "B38:E38"
            ),
            money_scale=1_000_000.0,
            writer=write_coal_inputs,
        ),
        "csp": SupportedWorkbook(
            workbook_name="jedi-csp-trough-model.xlsm",
            workbook_key="csp",
            summary_sheet="SummaryResults",
            construction_breakdown=ImpactBreakdownSpec(
                None, "B32:E32", "B33:E33", "B34:E34"
            ),
            operating_breakdown=ImpactBreakdownSpec(
                "B39:E39", "B40:E40", "B41:E41", "B42:E42"
            ),
            money_scale=1_000_000.0,
            writer=write_csp_inputs,
        ),
        "geothermal": SupportedWorkbook(
            workbook_name="jedi-geothermal-model.xlsm",
            workbook_key="geothermal",
            summary_sheet="Summary Results",
            construction_breakdown=ImpactBreakdownSpec(
                None, "B26:E26", "B27:E27", "B28:E28"
            ),
            operating_breakdown=ImpactBreakdownSpec(
                "B31:E31", "B32:E32", "B33:E33", "B34:E34"
            ),
            money_scale=1_000_000.0,
            writer=write_geothermal_inputs,
        ),
    }


def default_workbook_dir(preferred_jedi_dir: Path | None) -> Path:
    if preferred_jedi_dir is not None:
        return preferred_jedi_dir
    if JEDI_CANADA_DIR.exists():
        return JEDI_CANADA_DIR
    return JEDI_DIR


def resolve_workbook_path(
    spec: SupportedWorkbook,
    event: dict,
    preferred_jedi_dir: Path | None,
) -> Path:
    is_canada = str(event.get("country", "")).lower() != "usa"
    workbook_name = spec.workbook_name
    if spec.workbook_key == "onwind" and is_canada:






        province_abbr = str(event.get("st", "")).strip()
        name_path = Path(workbook_name)
        workbook_name = str(
            name_path.parent / f"{name_path.stem}__{province_abbr}{name_path.suffix}"
        )
    if preferred_jedi_dir is not None:
        return preferred_jedi_dir / workbook_name
    if is_canada:
        candidate = JEDI_CANADA_DIR / workbook_name
        if candidate.exists():
            return candidate
        if spec.workbook_key == "onwind":
            logger.warning(
                "resolve_workbook_path: no per-province onshore-wind workbook for "
                "st=%s at %s; falling back to the shared US workbook (proxy-state "
                "Location match only, no StatCan override for this province) -- "
                "run Canada_data/merge_canada_into_jedi.py to build one.",
                event.get("st"),
                candidate,
            )
    return JEDI_DIR / spec.workbook_name


def coerce_scalar(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, float) and math.isnan(value):
        return 0.0
    return float(value)


def create_excel_app() -> xw.main.App:
    import time

    ensure_excel_pythonpath()


    last_exc: Exception | None = None
    for attempt in range(6):
        try:
            app = xw.App(visible=False, add_book=False)
            break
        except Exception as exc:
            last_exc = exc
            wait = 2**attempt
            logger.warning(
                "create_excel_app: attempt %d failed (%s); retrying in %ds",
                attempt + 1,
                exc,
                wait,
            )
            time.sleep(wait)
    else:
        raise RuntimeError("create_excel_app: all retry attempts failed") from last_exc
    app.display_alerts = False
    app.screen_updating = False
    app.api.Visible = False
    app.api.DisplayAlerts = False
    app.api.DisplayStatusBar = False
    app.api.EnableEvents = False
    app.api.ScreenUpdating = False
    app.api.AskToUpdateLinks = False
    try:
        app.api.AutoRecover.Enabled = False
    except Exception:
        logger.debug("create_excel_app: failed to disable AutoRecover", exc_info=True)
    try:
        hwnd = int(app.api.Hwnd)
        ctypes.windll.user32.ShowWindow(hwnd, 0)
    except Exception:
        logger.debug(
            "create_excel_app: failed to hide Excel window via ShowWindow",
            exc_info=True,
        )
    return app


def build_result_row(
    spec: SupportedWorkbook,
    event: dict,
    construction_breakdown: dict[str, list[float]],
    operating_breakdown: dict[str, list[float]] | None = None,
) -> dict:
    scale = float(event.get("project_scale_factor", 1.0))
    row = {
        **event,
    }
    add_scaled_impact_columns(
        row, "construction", construction_breakdown, spec.money_scale, scale
    )
    if operating_breakdown is not None:
        if (
            event.get("tech") == "land-based wind"
            and float(event.get("nameplate_value", 0.0)) < 25.0
        ):





            operating_breakdown = {
                effect: [0.0, 0.0, 0.0, 0.0] for effect in IMPACT_EFFECTS
            }



        add_scaled_impact_columns(
            row, "operating", operating_breakdown, spec.money_scale, scale, annual=True
        )
    return row


def evaluate_event_in_book(
    book: xw.main.Book,
    app: xw.main.App,
    spec: SupportedWorkbook,
    event: dict,
) -> dict:
    spec.writer(book, event)
    if spec.runner is not None:
        spec.runner(book, app, event)
    app.api.CalculateFull()
    construction_breakdown = read_impact_breakdown(
        book, spec, spec.construction_breakdown
    )
    operating_breakdown = read_impact_breakdown(book, spec, spec.operating_breakdown)
    return build_result_row(spec, event, construction_breakdown, operating_breakdown)


def run_isolated_jedi_event(
    spec: SupportedWorkbook,
    event: dict,
    preferred_jedi_dir: Path | None = None,
    offshore_direct_runtime: bool = False,
) -> dict:
    if event["tech"] == "land-based wind":
        return run_onshore_event_via_runtime(spec, event, preferred_jedi_dir)
    if offshore_direct_runtime and event["tech"] == "offshore wind":
        return run_offshore_event_via_runtime(spec, event, preferred_jedi_dir)
    app = create_excel_app()
    book = None
    try:
        book = app.books.open(
            str(resolve_workbook_path(spec, event, preferred_jedi_dir))
        )
        return evaluate_event_in_book(book, app, spec, event)
    finally:
        if book is not None:
            try:
                book.close()
            except Exception:
                pass
        try:
            app.quit()
        except Exception:
            pass


def auto_worker_count() -> int:

    logical = os.cpu_count() or 4
    try:
        import subprocess as _sp

        result = _sp.run(
            ["wmic", "cpu", "get", "NumberOfCores", "/value"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        physical = sum(
            int(line.split("=")[1])
            for line in result.stdout.splitlines()
            if line.strip().startswith("NumberOfCores=")
        )
    except Exception:
        physical = max(1, logical // 2)
    cpu_limit = max(1, physical)


    try:
        import subprocess as _sp

        result = _sp.run(
            ["wmic", "OS", "get", "FreePhysicalMemory", "/value"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        free_kb = int(
            next(
                line.split("=")[1]
                for line in result.stdout.splitlines()
                if line.strip().startswith("FreePhysicalMemory=")
            )
        )
        free_gb = free_kb / (1024**2)
        ram_limit = max(1, int((free_gb - 4) / 0.5))
    except Exception:
        ram_limit = cpu_limit

    workers = min(cpu_limit, ram_limit, 16)
    logger.info(
        "auto_worker_count: physical_cores=%d, ram_limit=%d → %d workers",
        cpu_limit,
        ram_limit,
        workers,
    )
    return max(1, workers)


def chunk_records(records: list[dict], chunk_size: int) -> list[list[dict]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [records[i : i + chunk_size] for i in range(0, len(records), chunk_size)]


def run_macro_event_batch(
    tech: str,
    batch: list[dict],
    preferred_jedi_dir: Path | None = None,
    offshore_direct_runtime: bool = False,
) -> tuple[list[dict], list[dict]]:
    spec = supported_workbooks()[tech]
    if tech == "land-based wind":
        runtime_python = resolve_landbosse_runtime_python()
        if runtime_python is None:
            raise FileNotFoundError(
                "landbosse-runtime/python.exe not found under LOCALAPPDATA"
            )
        raw_rows: list[dict] = []
        error_rows: list[dict] = []
        workbook_groups: dict[str, list[dict]] = {}
        for event in batch:
            workbook_path = str(resolve_workbook_path(spec, event, preferred_jedi_dir))
            workbook_groups.setdefault(workbook_path, []).append(event)
        for workbook_path_str, group_events in workbook_groups.items():
            try:
                group_raw, group_errors = run_onshore_runtime_batch_subprocess(
                    runtime_python,
                    Path(workbook_path_str),
                    group_events,
                )
                raw_rows.extend(group_raw)
                error_rows.extend(group_errors)
            except Exception as exc:
                for event in group_events:
                    error_rows.append({**event, "error": repr(exc)})
        return raw_rows, error_rows
    if offshore_direct_runtime and tech == "offshore wind":
        raw_rows: list[dict] = []
        error_rows: list[dict] = []
        for event in batch:
            try:
                raw_rows.append(
                    run_offshore_event_via_runtime(spec, event, preferred_jedi_dir)
                )
            except Exception as exc:
                error_rows.append({**event, "error": repr(exc)})
        return raw_rows, error_rows
    app = create_excel_app()
    book = None
    raw_rows: list[dict] = []
    error_rows: list[dict] = []
    try:
        first_event = batch[0]
        book = app.books.open(
            str(resolve_workbook_path(spec, first_event, preferred_jedi_dir))
        )
        for event in batch:
            try:
                raw_rows.append(evaluate_event_in_book(book, app, spec, event))
            except Exception as exc:
                error_rows.append({**event, "error": repr(exc)})
    finally:
        if book is not None:
            try:
                book.close()
            except Exception:
                pass
        try:
            app.quit()
        except Exception:
            pass
    return raw_rows, error_rows


def sort_static_events(
    static_events: list[dict],
    workbook_specs: dict[str, "SupportedWorkbook"],
    preferred_jedi_dir: Path | None = None,
) -> list[dict]:
    return sorted(
        static_events,
        key=lambda event: (
            str(
                resolve_workbook_path(
                    workbook_specs[event["tech"]], event, preferred_jedi_dir
                )
            ),
            str(event["st"]),
            str(event["tech"]),
            int(event["online_year"]),
        ),
    )


def partition_into_n_chunks(records: list[dict], n: int) -> list[list[dict]]:
    if n <= 0:
        raise ValueError("n must be positive")
    n = min(n, len(records)) or 1
    chunk_size, remainder = divmod(len(records), n)
    chunks: list[list[dict]] = []
    start = 0
    for i in range(n):
        size = chunk_size + (1 if i < remainder else 0)
        if size == 0:
            continue
        chunks.append(records[start : start + size])
        start += size
    return chunks


def run_static_event_batch(
    batch: list[dict],
    preferred_jedi_dir: Path | None = None,
) -> tuple[list[dict], list[dict]]:
    import pythoncom
    import time


    pythoncom.CoInitialize()




    time.sleep((os.getpid() % 8) * 0.15)

    workbook_specs = supported_workbooks()


    needed: dict[str, Path] = {}
    for event in batch:
        orig = resolve_workbook_path(
            workbook_specs[event["tech"]], event, preferred_jedi_dir
        )
        needed[str(orig)] = orig

    temp_dir = Path(tempfile.mkdtemp(prefix="jedi_worker_"))

    workbook_map: dict[str, Path] = {}
    try:
        for orig_str, orig_path in needed.items():
            dest = temp_dir / orig_path.name
            shutil.copy2(str(orig_path), str(dest))


            dest.chmod(dest.stat().st_mode | 0o222)
            workbook_map[orig_str] = dest

        app = create_excel_app()
        raw_rows: list[dict] = []
        error_rows: list[dict] = []
        current_cache_key: str | None = None
        current_book = None
        try:
            for event in batch:
                tech = event["tech"]
                spec = workbook_specs[tech]
                try:
                    orig_path = resolve_workbook_path(spec, event, preferred_jedi_dir)
                    temp_path = workbook_map[str(orig_path)]
                    cache_key = str(temp_path)
                    if cache_key != current_cache_key:
                        if current_book is not None:
                            try:
                                current_book.close()
                            except Exception:
                                pass
                        current_book = app.books.open(
                            str(temp_path),
                            update_links=False,
                            notify=False,
                            ignore_read_only_recommended=True,
                        )
                        current_cache_key = cache_key
                    raw_rows.append(
                        evaluate_event_in_book(current_book, app, spec, event)
                    )
                except Exception as exc:
                    error_rows.append({**event, "error": repr(exc)})
            if current_book is not None:
                try:
                    current_book.close()
                except Exception:
                    pass
        finally:
            try:
                app.quit()
            except Exception:
                pass
    finally:
        shutil.rmtree(str(temp_dir), ignore_errors=True)

    return raw_rows, error_rows


def run_supported_jedi_events(
    events: pd.DataFrame,
    output_dir: Path,
    macro_batch_size: int = 8,
    macro_workers: int | None = None,
    static_workers: int | None = None,
    preferred_jedi_dir: Path | None = None,
    offshore_direct_runtime: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if macro_workers is None:
        macro_workers = auto_worker_count()
    if static_workers is None:
        static_workers = auto_worker_count()
    workbook_specs = supported_workbooks()
    raw_rows: list[dict] = []
    error_rows: list[dict] = []

    event_records = events.to_dict("records")
    macro_events_by_tech: dict[str, list[dict]] = {}
    static_events: list[dict] = []
    for event in event_records:
        tech = event["tech"]
        spec = workbook_specs[tech]
        if spec.runner is not None:
            macro_events_by_tech.setdefault(tech, []).append(event)
        else:
            static_events.append(event)

    static_events_sorted = sort_static_events(
        static_events, workbook_specs, preferred_jedi_dir
    )

    if static_events_sorted:
        chunks = partition_into_n_chunks(static_events_sorted, static_workers)
        logger.info(
            "run_supported_jedi_events: %d static events → %d chunks, "
            "each worker uses private temp copies of its .xlsm files (CPU count=%s)",
            len(static_events_sorted),
            len(chunks),
            os.cpu_count(),
        )
        if len(chunks) > 1:
            try:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=len(chunks)
                ) as executor:
                    futures = [
                        executor.submit(
                            run_static_event_batch, chunk, preferred_jedi_dir
                        )
                        for chunk in chunks
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        batch_raw, batch_errors = future.result()
                        raw_rows.extend(batch_raw)
                        error_rows.extend(batch_errors)
            except Exception:
                logger.exception(
                    "Parallel static-event dispatch failed; falling back to sequential."
                )
                raw_rows.clear()
                error_rows.clear()
                batch_raw, batch_errors = run_static_event_batch(
                    static_events_sorted, preferred_jedi_dir
                )
                raw_rows.extend(batch_raw)
                error_rows.extend(batch_errors)
        else:
            batch_raw, batch_errors = run_static_event_batch(
                static_events_sorted, preferred_jedi_dir
            )
            raw_rows.extend(batch_raw)
            error_rows.extend(batch_errors)

    macro_jobs: list[tuple[str, list[dict]]] = []
    for tech, records in macro_events_by_tech.items():
        tech_batch_size = (
            ONWIND_RUNTIME_BATCH_SIZE if tech == "land-based wind" else macro_batch_size
        )
        macro_jobs.extend(
            (tech, batch) for batch in chunk_records(records, tech_batch_size)
        )

    if macro_jobs:
        if macro_workers > 1 and len(macro_jobs) > 1:
            try:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=macro_workers
                ) as executor:
                    futures = [
                        executor.submit(
                            run_macro_event_batch,
                            tech,
                            batch,
                            preferred_jedi_dir,
                            offshore_direct_runtime,
                        )
                        for tech, batch in macro_jobs
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        batch_raw, batch_errors = future.result()
                        raw_rows.extend(batch_raw)
                        error_rows.extend(batch_errors)
            except Exception:
                for tech, batch in macro_jobs:
                    batch_raw, batch_errors = run_macro_event_batch(
                        tech, batch, preferred_jedi_dir, offshore_direct_runtime
                    )
                    raw_rows.extend(batch_raw)
                    error_rows.extend(batch_errors)
        else:
            for tech, batch in macro_jobs:
                batch_raw, batch_errors = run_macro_event_batch(
                    tech, batch, preferred_jedi_dir, offshore_direct_runtime
                )
                raw_rows.extend(batch_raw)
                error_rows.extend(batch_errors)

    raw = pd.DataFrame(raw_rows)
    errors = pd.DataFrame(error_rows)

    if not raw.empty:
        if "event_role" in raw.columns:
            raw.loc[raw["event_role"].eq("construction")].to_csv(
                output_dir / "construction_impacts_raw.csv", index=False
            )
            raw.loc[raw["event_role"].eq("operating_basis")].to_csv(
                output_dir / "operating_response_impacts_raw.csv", index=False
            )
        else:
            raw.to_csv(output_dir / "construction_impacts_raw.csv", index=False)
    if not errors.empty:
        errors.to_csv(output_dir / "construction_impacts_errors.csv", index=False)
    return raw, errors


def _compute_year_gaps(modeled_years: list[int]) -> dict[int, int]:
    years = sorted(set(modeled_years))
    return {
        year: (1 if i == 0 else year - years[i - 1]) for i, year in enumerate(years)
    }


def spread_construction_impacts(
    raw: pd.DataFrame,
    modeled_years: list[int] | None = None,
    min_output_impact_year: int | None = MIN_OUTPUT_IMPACT_YEAR,
) -> pd.DataFrame:
    annual_rows: list[dict] = []
    for row in raw.to_dict("records"):
        online_year = int(row["online_year"])
        build_years = max(1, int(row.get("build_years", 1)))
        constr_start = int(
            row.get("construction_start_year", online_year - build_years)
        )
        for impact_year in range(constr_start, online_year):
            if (
                min_output_impact_year is not None
                and impact_year < min_output_impact_year
            ):
                continue
            annual_rows.append(
                {
                    "impact_year": impact_year,
                    "online_year": online_year,
                    "construction_start_year": constr_start,
                    "workbook_year_is_proxy": bool(row["workbook_year_is_proxy"]),
                    "st": row["st"],
                    "country": row["country"],
                    "state_name": row["state_name"],
                    "tech": row["tech"],
                    "proxy_region_used": row.get("proxy_region_used", ""),
                    "weather_proxy_used": row.get("weather_proxy_used", ""),
                    "build_mw_online_year": row["build_mw"],
                    "build_years": build_years,
                    "money_year": row.get("money_year"),
                    "currency": "USD",
                    "price_year": row.get("money_year"),
                }
            )
            for metric in IMPACT_METRICS:
                annual_rows[-1][_legacy_total_column("construction", metric)] = (
                    read_effect_value(row, "construction", metric, "total")
                    / build_years
                )
                annual_rows[-1][_impact_column("construction", metric, "total")] = (
                    annual_rows[-1][_legacy_total_column("construction", metric)]
                )
                for effect in ("direct", "indirect", "induced"):
                    annual_rows[-1][_impact_column("construction", metric, effect)] = (
                        read_effect_value(row, "construction", metric, effect)
                        / build_years
                    )
    annual = pd.DataFrame(annual_rows)
    if annual.empty:
        return annual
    return annual.sort_values(["impact_year", "st", "tech", "online_year"]).reset_index(
        drop=True
    )


def save_summaries(annual: pd.DataFrame, output_dir: Path) -> None:
    if annual.empty:
        return
    annual.to_csv(output_dir / "construction_impacts_annual.csv", index=False)
    summary_columns = []
    for metric in IMPACT_METRICS:
        summary_columns.append(_legacy_total_column("construction", metric))
        summary_columns.extend(
            _impact_column("construction", metric, effect) for effect in IMPACT_EFFECTS
        )
    summary = (
        annual.groupby(["impact_year", "tech"], as_index=False)[summary_columns]
        .sum()
        .sort_values(["impact_year", "tech"])
    )
    summary.to_csv(output_dir / "construction_impacts_annual_summary.csv", index=False)


CF_RESPONSE_TECHS = {
    "ng-cc",
    "ng-ct",
    "hydro",
    "pumped storage",
    "biopower",
    "coal",
    "csp",
}


def prepare_operating_response_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    actual = events.reset_index(drop=True).copy()
    actual["operating_response_id"] = [f"response_{i}" for i in actual.index]
    actual["operating_response_anchor"] = "actual"
    responsive = actual[actual["tech"].isin(CF_RESPONSE_TECHS)]
    anchors = []
    for label, capacity_factor in (("cf0", 0.0), ("cf1", 1.0)):
        anchor = responsive.copy()
        anchor["capacity_factor"] = capacity_factor
        anchor["operating_response_anchor"] = label
        anchors.append(anchor)
    return pd.concat([actual, *anchors], ignore_index=True)


def build_operating_basis_events(
    cap: pd.DataFrame,
    hierarchy: pd.DataFrame,
    capacity_factors: pd.DataFrame,
    build_years: pd.DataFrame,
    nameplates: pd.DataFrame,
    gas_share: pd.DataFrame,
    coal_share: pd.DataFrame,
    state_lookup: pd.DataFrame,
    supported_canada_techs: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stock = cap.copy()
    stock["mapped_tech"] = stock["i"].map(map_reeds_tech)
    stock = stock[
        stock["mapped_tech"].isin(SUPPORTED_JEDI_TECHS) & stock["Value"].gt(0)
    ]
    if stock.empty:
        return pd.DataFrame(), pd.DataFrame()

    totals = stock.groupby(["st", "country", "mapped_tech", "t"], as_index=False)[
        "Value"
    ].sum()
    totals["reference_year_distance"] = (totals["t"] - 2022).abs()
    selected = (
        totals.sort_values(
            ["reference_year_distance", "Value"], ascending=[True, False]
        )
        .drop_duplicates(["st", "country", "mapped_tech"])
        .copy()
    )
    representative = stock.sort_values("Value", ascending=False).drop_duplicates(
        ["st", "country", "mapped_tech", "t"]
    )[["st", "country", "mapped_tech", "t", "i", "r"]]
    selected = selected.merge(
        representative,
        on=["st", "country", "mapped_tech", "t"],
        how="left",
    )
    synthetic_cap = selected[["i", "r", "t", "Value"]].copy()
    coverage = build_coverage_report(
        hierarchy, synthetic_cap, state_lookup, supported_canada_techs
    )
    basis = prepare_supported_builds(
        coverage=coverage,
        capacity_factors=capacity_factors,
        build_years=build_years,
        nameplates=nameplates,
        gas_share=gas_share,
        coal_share=coal_share,
        state_lookup=state_lookup,
    )
    if not basis.empty:
        basis["event_role"] = "operating_basis"
        basis["operating_basis_stock_mw"] = basis["build_mw"]
    return basis, coverage


def build_pumped_storage_capacity_factor_overrides(
    hierarchy: pd.DataFrame,
    avg_cf: pd.DataFrame | None,
    cap_ivrt: pd.DataFrame | None,
    stor_inout: pd.DataFrame | None = None,
    cap: pd.DataFrame | None = None,
) -> pd.DataFrame:
    columns = ["st", "country", "tech", "t", "capacity_factor"]
    if stor_inout is not None and not stor_inout.empty and cap is not None:
        direction_col = "*" if "*" in stor_inout.columns else "direction"
        discharge = stor_inout[
            stor_inout[direction_col].astype(str).str.lower().eq("out")
        ].copy()
        discharge["tech"] = discharge["i"].map(map_reeds_tech)
        discharge = discharge[discharge["tech"].eq("pumped storage")].merge(
            hierarchy, on="r", how="left"
        )
        generation = (
            discharge.groupby(["st", "country", "tech", "t"], as_index=False)["Value"]
            .sum()
            .rename(columns={"Value": "generation_mwh"})
        )
        capacity = cap.copy()
        capacity["tech"] = capacity["i"].map(map_reeds_tech)
        capacity = (
            capacity[capacity["tech"].eq("pumped storage")]
            .groupby(["st", "country", "tech", "t"], as_index=False)["Value"]
            .sum()
            .rename(columns={"Value": "capacity_mw"})
        )
        result = capacity.merge(
            generation, on=["st", "country", "tech", "t"], how="left"
        )
        result["generation_mwh"] = result["generation_mwh"].fillna(0.0)
        result["capacity_factor"] = 0.0
        positive = result["capacity_mw"].gt(0)
        result.loc[positive, "capacity_factor"] = result.loc[
            positive, "generation_mwh"
        ] / (result.loc[positive, "capacity_mw"] * 8760.0)
        return result[columns]
    if avg_cf is None or avg_cf.empty:
        return pd.DataFrame(columns=columns)
    avg = avg_cf.copy()
    avg["tech"] = avg["i"].map(map_reeds_tech)
    avg = avg[avg["tech"].eq("pumped storage")].merge(hierarchy, on="r", how="left")
    if avg.empty:
        return pd.DataFrame(columns=columns)
    if cap_ivrt is not None and not cap_ivrt.empty:
        weights = cap_ivrt[["i", "v", "r", "t", "Value"]].rename(
            columns={"Value": "capacity_mw"}
        )
        avg = avg.merge(weights, on=["i", "v", "r", "t"], how="left")
    else:
        avg["capacity_mw"] = 1.0
    avg["capacity_mw"] = avg["capacity_mw"].fillna(0.0).clip(lower=0.0)
    avg["weighted_cf"] = avg["Value"] * avg["capacity_mw"]
    grouped = avg.groupby(["st", "country", "tech", "t"], as_index=False)[
        ["weighted_cf", "capacity_mw"]
    ].sum()
    grouped = grouped[grouped["capacity_mw"].gt(0)].copy()
    grouped["capacity_factor"] = (grouped["weighted_cf"] / grouped["capacity_mw"]).clip(
        lower=0.0
    )
    return grouped[columns]


def spread_operating_impacts(
    raw: pd.DataFrame,
    cap: pd.DataFrame,
    modeled_years: list[int],
    gen: pd.DataFrame | None = None,
    capacity_factor_overrides: pd.DataFrame | None = None,
    min_output_impact_year: int | None = MIN_OUTPUT_IMPACT_YEAR,
) -> pd.DataFrame:
    if (
        "operating_jobs_fte_annual" not in raw.columns
        and _impact_column("operating", "jobs_fte", "total", annual=True)
        not in raw.columns
    ):
        return pd.DataFrame()

    solve_years = sorted(set(modeled_years))
    calendar_years = list(range(solve_years[0], solve_years[-1] + 1))
    id_cols = ["st", "country", "tech"]


    src_to_dst: dict[str, str] = {}
    for metric in IMPACT_METRICS:
        leg_src = _legacy_total_column("operating", metric, annual=True)
        if leg_src in raw.columns:
            src_to_dst[leg_src] = _legacy_total_column("operating", metric)
        for effect in ("direct", "indirect", "induced", "total"):
            src = _impact_column("operating", metric, effect, annual=True)
            if src in raw.columns:
                src_to_dst[src] = _impact_column("operating", metric, effect)
    src_cols = list(src_to_dst.keys())

    response_cols = ["operating_response_id", "operating_response_anchor"]
    has_response = all(column in raw.columns for column in response_cols)
    optional_cols = response_cols if has_response else []
    if "capacity_factor" in raw.columns:
        optional_cols = [*optional_cols, "capacity_factor"]
    if "event_role" in raw.columns:
        optional_cols = [*optional_cols, "event_role"]
    event_rates = raw.loc[
        raw["build_mw"].gt(0),
        id_cols + ["online_year", "build_mw"] + optional_cols + src_cols,
    ].copy()
    if "event_role" in event_rates.columns:
        event_rates = event_rates[event_rates["event_role"].eq("operating_basis")]
    for col in src_cols:
        event_rates[col] = pd.to_numeric(event_rates[col], errors="coerce").fillna(0.0)

    coefficient_cols: list[str] = []
    coefficient_rows: list[dict] = []
    if has_response:
        for response_id, group in event_rates.groupby("operating_response_id"):
            actual = group[group["operating_response_anchor"].eq("actual")]
            if actual.empty:
                continue
            base = actual.iloc[0]
            cf0 = group[group["operating_response_anchor"].eq("cf0")]
            cf1 = group[group["operating_response_anchor"].eq("cf1")]
            build_mw = float(base["build_mw"])
            row = {column: base[column] for column in id_cols + ["online_year"]}
            row["build_mw"] = build_mw
            row["operating_response_id"] = response_id
            row["operating_response_method"] = (
                "jedi_cf0_cf1" if not cf0.empty and not cf1.empty else "jedi_fixed_only"
            )
            row["response_breakpoint_cf"] = float(base.get("capacity_factor", 0.0))
            if "response_breakpoint_cf" not in coefficient_cols:
                coefficient_cols.append("response_breakpoint_cf")
            validation_errors: list[float] = []
            for src in src_cols:
                fixed_col = f"{src}__fixed_per_mw"
                variable_col = f"{src}__variable_per_mwh"
                actual_col = f"{src}__actual_per_mw"
                if fixed_col not in coefficient_cols:
                    coefficient_cols.extend([fixed_col, variable_col, actual_col])
                if not cf0.empty and not cf1.empty:
                    fixed_value = float(cf0.iloc[0][src]) / build_mw
                    variable_value = (
                        float(cf1.iloc[0][src]) - float(cf0.iloc[0][src])
                    ) / (build_mw * 8760.0)
                    actual_cf = float(base.get("capacity_factor", 0.0))
                    predicted = (
                        float(cf0.iloc[0][src])
                        + (float(cf1.iloc[0][src]) - float(cf0.iloc[0][src]))
                        * actual_cf
                    )
                    observed = float(base[src])
                    validation_errors.append(
                        abs(predicted - observed) / max(abs(observed), 1e-9)
                    )
                else:
                    fixed_value = float(base[src]) / build_mw
                    variable_value = 0.0
                row[fixed_col] = fixed_value
                row[variable_col] = variable_value
                row[actual_col] = float(base[src]) / build_mw
            row["operating_response_validation_error_max"] = (
                max(validation_errors) if validation_errors else 0.0
            )
            coefficient_rows.append(row)
    else:
        for _, base in event_rates.iterrows():
            build_mw = float(base["build_mw"])
            row = {column: base[column] for column in id_cols + ["online_year"]}
            row["build_mw"] = build_mw
            row["operating_response_method"] = "legacy_fixed_only"
            row["operating_response_validation_error_max"] = float("nan")
            row["response_breakpoint_cf"] = 0.0
            if "response_breakpoint_cf" not in coefficient_cols:
                coefficient_cols.append("response_breakpoint_cf")
            for src in src_cols:
                fixed_col = f"{src}__fixed_per_mw"
                variable_col = f"{src}__variable_per_mwh"
                actual_col = f"{src}__actual_per_mw"
                if fixed_col not in coefficient_cols:
                    coefficient_cols.extend([fixed_col, variable_col, actual_col])
                row[fixed_col] = float(base[src]) / build_mw
                row[variable_col] = 0.0
                row[actual_col] = float(base[src]) / build_mw
            coefficient_rows.append(row)

    coefficients = pd.DataFrame(coefficient_rows)
    if coefficients.empty:
        return pd.DataFrame()
    weighted_cols = []
    for column in coefficient_cols:
        weighted = f"{column}__weighted"
        coefficients[weighted] = coefficients[column] * coefficients["build_mw"]
        weighted_cols.append(weighted)

    def _aggregate_coefficients(group_cols: list[str]) -> pd.DataFrame:
        grouped = coefficients.groupby(group_cols, as_index=False)[
            weighted_cols + ["build_mw"]
        ].sum()
        for column, weighted in zip(coefficient_cols, weighted_cols):
            grouped[column] = grouped[weighted] / grouped["build_mw"]
        methods = coefficients.groupby(group_cols)["operating_response_method"].agg(
            lambda values: (
                "jedi_cf0_cf1"
                if values.eq("jedi_cf0_cf1").all()
                else (
                    "jedi_fixed_only"
                    if values.eq("jedi_fixed_only").all()
                    else (
                        "legacy_fixed_only"
                        if values.eq("legacy_fixed_only").all()
                        else "mixed"
                    )
                )
            )
        )
        validation = coefficients.groupby(group_cols)[
            "operating_response_validation_error_max"
        ].max()
        return (
            grouped[group_cols + coefficient_cols]
            .merge(
                methods.rename("operating_response_method").reset_index(), on=group_cols
            )
            .merge(
                validation.rename(
                    "operating_response_validation_error_max"
                ).reset_index(),
                on=group_cols,
            )
        )

    regional = _aggregate_coefficients(id_cols + ["online_year"])
    technology = _aggregate_coefficients(["tech", "online_year"])

    stock = cap.copy()
    stock["tech"] = stock["i"].map(map_reeds_tech)
    stock = stock[stock["tech"].isin(SUPPORTED_JEDI_TECHS)].copy()
    stock = stock.groupby(id_cols + ["t"], as_index=False)["Value"].sum()

    generation = pd.DataFrame(columns=id_cols + ["t", "Value"])
    if gen is not None and not gen.empty:
        generation = gen.copy()
        generation["tech"] = generation["i"].map(map_reeds_tech)
        generation = generation[generation["tech"].isin(SUPPORTED_JEDI_TECHS)]
        generation = generation.groupby(id_cols + ["t"], as_index=False)["Value"].sum()

    state_names = raw[["st", "country", "state_name"]].drop_duplicates(
        ["st", "country"]
    )
    money_years = (
        pd.to_numeric(raw["money_year"], errors="coerce").dropna().unique()
        if "money_year" in raw.columns
        else []
    )
    money_year = int(money_years[0]) if len(money_years) else None
    solve_set = set(solve_years)
    annual_frames: list[pd.DataFrame] = []

    def _calendar_coefficients(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(
                index=calendar_years, columns=coefficient_cols, dtype=float
            )
        anchors = (
            frame.groupby("online_year")[coefficient_cols]
            .mean()
            .reindex(calendar_years)
        )
        return anchors.interpolate(method="linear", limit_direction="both")

    tech_coefficients = {
        tech: _calendar_coefficients(group)
        for tech, group in technology.groupby("tech")
    }

    for key, stock_group in stock.groupby(id_cols):
        st, country, tech = key
        stock_series = (
            stock_group.set_index("t")["Value"]
            .reindex(solve_years, fill_value=0.0)
            .reindex(calendar_years)
            .interpolate(method="linear")
            .fillna(0.0)
            .clip(lower=0.0)
        )
        gen_group = generation[
            generation["st"].eq(st)
            & generation["country"].eq(country)
            & generation["tech"].eq(tech)
        ]
        if gen_group.empty:
            generation_series = pd.Series(0.0, index=calendar_years, dtype=float)
        else:
            generation_series = (
                pd.to_numeric(gen_group.set_index("t")["Value"], errors="coerce")
                .reindex(solve_years, fill_value=0.0)
                .reindex(calendar_years)
                .interpolate(method="linear")
                .fillna(0.0)
                .clip(lower=0.0)
            )
        if (
            capacity_factor_overrides is not None
            and not capacity_factor_overrides.empty
        ):
            override = capacity_factor_overrides[
                capacity_factor_overrides["st"].eq(st)
                & capacity_factor_overrides["country"].eq(country)
                & capacity_factor_overrides["tech"].eq(tech)
            ]
            if not override.empty:
                override_cf = (
                    pd.to_numeric(
                        override.set_index("t")["capacity_factor"], errors="coerce"
                    )
                    .reindex(solve_years)
                    .reindex(calendar_years)
                    .interpolate(method="linear", limit_direction="both")
                    .fillna(0.0)
                    .clip(lower=0.0)
                )
                generation_series = override_cf * stock_series * 8760.0
        fallback = tech_coefficients.get(tech)
        if fallback is None or fallback.dropna(how="all").empty:
            logger.warning(
                "spread_operating_impacts: no JEDI operating intensity for tech=%s; "
                "reporting %.3f MW of stock with NaN impacts",
                tech,
                stock_group["Value"].max(),
            )
            coefficient_frame = pd.DataFrame(
                float("nan"), index=calendar_years, columns=coefficient_cols
            )
            response_source = "missing"
            response_method = "missing"
            validation_error = float("nan")
        else:
            regional_group = regional[
                regional["st"].eq(st)
                & regional["country"].eq(country)
                & regional["tech"].eq(tech)
            ]
            local = _calendar_coefficients(regional_group)
            local_available = not local.dropna(how="all").empty
            coefficient_frame = (
                local.fillna(fallback) if local_available else fallback.copy()
            )
            response_source = "state_tech" if local_available else "technology_wide"
            method_values = (
                regional_group["operating_response_method"].dropna().unique()
            )
            response_method = (
                str(method_values[0])
                if local_available and len(method_values) == 1
                else str(
                    technology.loc[
                        technology["tech"].eq(tech), "operating_response_method"
                    ].iloc[0]
                )
            )
            validation_error = float(
                (
                    regional_group
                    if local_available
                    else technology[technology["tech"].eq(tech)]
                )["operating_response_validation_error_max"].max()
            )
        frame = pd.DataFrame({"impact_year": calendar_years})
        frame["st"] = st
        frame["country"] = country
        frame["tech"] = tech
        frame["operating_stock_mw"] = stock_series.to_numpy()
        frame["operating_generation_mwh"] = generation_series.to_numpy()
        denominator = frame["operating_stock_mw"] * 8760.0
        frame["actual_capacity_factor"] = 0.0
        positive = denominator.gt(0)
        frame.loc[positive, "actual_capacity_factor"] = (
            frame.loc[positive, "operating_generation_mwh"] / denominator[positive]
        )
        frame["operating_intensity_source"] = response_source
        frame["operating_response_method"] = response_method
        frame["operating_response_validation_error_max"] = validation_error
        frame["operating_response_nonlinear_flag"] = validation_error > 0.01
        if validation_error > 0.01 and response_method == "jedi_cf0_cf1":
            frame["operating_response_method"] = "jedi_piecewise_cf"
        frame["operating_response_extrapolation_flag"] = frame[
            "actual_capacity_factor"
        ].gt(1.0)
        frame["is_interpolated"] = ~frame["impact_year"].isin(solve_set)
        frame["money_year"] = money_year
        frame["currency"] = "USD"
        frame["price_year"] = money_year
        for src, dst in src_to_dst.items():
            fixed_col = f"{src}__fixed_per_mw"
            variable_col = f"{src}__variable_per_mwh"
            actual_col = f"{src}__actual_per_mw"
            fixed_values = coefficient_frame[fixed_col].to_numpy()
            variable_values = coefficient_frame[variable_col].to_numpy()
            actual_values = coefficient_frame[actual_col].to_numpy()
            frame[f"{dst}_fixed_per_mw"] = fixed_values
            frame[f"{dst}_variable_per_mwh"] = variable_values
            linear_impact = (
                frame["operating_stock_mw"] * fixed_values
                + frame["operating_generation_mwh"] * variable_values
            )
            frame[dst] = linear_impact
            if bool(frame["operating_response_nonlinear_flag"].any()):
                cf = frame["actual_capacity_factor"]
                breakpoint = pd.Series(
                    coefficient_frame["response_breakpoint_cf"].to_numpy(),
                    index=frame.index,
                )
                fixed = pd.Series(fixed_values, index=frame.index)
                actual_per_mw = pd.Series(actual_values, index=frame.index)
                cf1_per_mw = (
                    fixed + pd.Series(variable_values, index=frame.index) * 8760.0
                )
                effective = (
                    fixed + cf * pd.Series(variable_values, index=frame.index) * 8760.0
                )
                low = (
                    frame["operating_response_nonlinear_flag"]
                    & cf.le(breakpoint)
                    & breakpoint.gt(0)
                )
                high = (
                    frame["operating_response_nonlinear_flag"]
                    & cf.gt(breakpoint)
                    & breakpoint.lt(1)
                )
                effective.loc[low] = (
                    fixed.loc[low]
                    + (actual_per_mw.loc[low] - fixed.loc[low])
                    * cf.loc[low]
                    / breakpoint.loc[low]
                )
                effective.loc[high] = actual_per_mw.loc[high] + (
                    cf1_per_mw.loc[high] - actual_per_mw.loc[high]
                ) * (cf.loc[high] - breakpoint.loc[high]) / (1.0 - breakpoint.loc[high])
                frame.loc[frame["operating_response_nonlinear_flag"], dst] = (
                    frame.loc[
                        frame["operating_response_nonlinear_flag"], "operating_stock_mw"
                    ]
                    * effective.loc[frame["operating_response_nonlinear_flag"]]
                )
                frame[f"{dst}_effective_per_mw"] = effective
        annual_frames.append(frame)

    if not annual_frames:
        return pd.DataFrame()
    annual = pd.concat(annual_frames, ignore_index=True)
    annual = annual.merge(state_names, on=["st", "country"], how="left")
    annual["state_name"] = annual["state_name"].fillna(annual["st"])
    if min_output_impact_year is not None:
        annual = annual[annual["impact_year"] >= min_output_impact_year]
    return annual.sort_values(["impact_year", "st", "tech"]).reset_index(drop=True)


def save_operating_summaries(annual_operating: pd.DataFrame, output_dir: Path) -> None:
    if annual_operating.empty:
        return
    annual_operating.to_csv(output_dir / "operating_impacts_annual.csv", index=False)
    summary_columns = []
    for metric in IMPACT_METRICS:
        summary_columns.append(_legacy_total_column("operating", metric))
        summary_columns.extend(
            _impact_column("operating", metric, effect) for effect in IMPACT_EFFECTS
        )
    summary = (
        annual_operating.groupby(["impact_year", "tech"], as_index=False)[
            summary_columns
        ]
        .sum()
        .sort_values(["impact_year", "tech"])
    )
    summary.to_csv(output_dir / "operating_impacts_annual_summary.csv", index=False)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    parser = build_arg_parser()
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    state_lookup = load_state_lookup(args.canada_lookup_path)
    supported_canada = canada_supported_techs(
        JEDI_CANADA_DIR if args.jedi_dir is None else args.jedi_dir
    )
    hierarchy, cap_new, cap, gen, avg_cf, cap_ivrt, gen_ivrt = load_case_outputs(
        args.case_dir
    )
    coverage = build_coverage_report(hierarchy, cap_new, state_lookup, supported_canada)
    coverage.to_csv(output_dir / "build_coverage_report.csv", index=False)

    if args.coverage_only:
        return

    capacity_factors = build_capacity_factor_table(
        hierarchy,
        cap_new,
        cap,
        gen,
        avg_cf,
        cap_ivrt,
        gen_ivrt,
    )
    build_years = load_build_years()
    nameplates = load_nameplate_table(args.nameplate_scenario)
    gas_share = load_gas_local_share()
    coal_share = load_coal_local_share()

    events = prepare_supported_builds(
        coverage=coverage,
        capacity_factors=capacity_factors,
        build_years=build_years,
        nameplates=nameplates,
        gas_share=gas_share,
        coal_share=coal_share,
        state_lookup=state_lookup,
    )
    events["money_year"] = args.money_year
    events["event_role"] = "construction"
    events.to_csv(output_dir / "supported_build_events.csv", index=False)

    operating_basis, operating_basis_coverage = build_operating_basis_events(
        cap=cap,
        hierarchy=hierarchy,
        capacity_factors=capacity_factors,
        build_years=build_years,
        nameplates=nameplates,
        gas_share=gas_share,
        coal_share=coal_share,
        state_lookup=state_lookup,
        supported_canada_techs=supported_canada,
    )
    operating_basis["money_year"] = args.money_year
    operating_basis.to_csv(output_dir / "operating_basis_events.csv", index=False)
    operating_basis_coverage.to_csv(
        output_dir / "operating_basis_coverage_report.csv", index=False
    )

    if args.max_events is not None:
        events = events.head(args.max_events).copy()

    operating_response_events = prepare_operating_response_events(operating_basis)
    run_events = pd.concat([events, operating_response_events], ignore_index=True)
    raw, _errors = run_supported_jedi_events(
        run_events,
        output_dir,
        macro_batch_size=args.macro_batch_size,
        macro_workers=args.macro_workers,
        static_workers=args.static_workers,
        preferred_jedi_dir=args.jedi_dir,
        offshore_direct_runtime=args.offshore_direct_runtime,
    )
    logger.info(
        "run_supported_jedi_events: %d raw rows, %d error rows", len(raw), len(_errors)
    )
    if not _errors.empty:
        logger.warning(
            "JEDI errors saved to %s/construction_impacts_errors.csv", output_dir
        )
        for _, row in _errors.iterrows():
            logger.warning(
                "  error [%s %s %s]: %s",
                row.get("st"),
                row.get("tech"),
                row.get("online_year"),
                row.get("error"),
            )
    _my_path = args.case_dir / "inputs_case" / "modeledyears.csv"
    with open(_my_path, encoding="utf-8") as _f:
        modeled_years = sorted(
            int(t) for t in _f.read().strip().split(",") if t.strip()
        )

    construction_raw = (
        raw[raw["event_role"].eq("construction")].copy()
        if "event_role" in raw.columns
        else raw
    )
    annual = spread_construction_impacts(
        construction_raw,
        modeled_years=modeled_years,
        min_output_impact_year=args.min_output_impact_year,
    )
    save_summaries(annual, output_dir)

    stor_inout_path = args.case_dir / "outputs" / "stor_inout.csv"
    stor_inout = pd.read_csv(stor_inout_path) if stor_inout_path.exists() else None
    pumped_storage_cf = build_pumped_storage_capacity_factor_overrides(
        hierarchy, avg_cf, cap_ivrt, stor_inout=stor_inout, cap=cap
    )
    annual_operating = spread_operating_impacts(
        raw,
        cap=cap,
        modeled_years=modeled_years,
        gen=gen,
        capacity_factor_overrides=pumped_storage_cf,
        min_output_impact_year=args.min_output_impact_year,
    )
    save_operating_summaries(annual_operating, output_dir)

    if args.include_transmission:
        from employment.energy_system.transmission_jedi import run_transmission_jedi

        run_transmission_jedi(
            args.case_dir,
            output_dir,
            money_year=args.money_year,
            max_events=args.max_events,
        )
    from employment.energy_system.transmission_jedi import (
        write_electricity_system_coverage,
    )

    write_electricity_system_coverage(output_dir)


if __name__ == "__main__":
    main()
