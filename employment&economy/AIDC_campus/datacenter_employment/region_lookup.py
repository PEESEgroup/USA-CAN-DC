
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
EMPLOYMENT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_ABBR_PATH = (
    EMPLOYMENT_ROOT
    / "energy_system"
    / "auxiliary_data"
    / "runtime_inputs"
    / "State_Abbr.xlsx"
)
DEFAULT_CANADA_LOOKUP_PATH = (
    EMPLOYMENT_ROOT
    / "energy_system"
    / "auxiliary_data"
    / "runtime_inputs"
    / "canada_state_lookup.xlsx"
)

_COUNTRY_NORMALIZE = {"canada": "CAN", "can": "CAN", "usa": "USA", "us": "USA"}


def _normalize_country(value: str) -> str:
    return _COUNTRY_NORMALIZE.get(str(value).strip().lower(), str(value).strip().upper())


def load_state_country_lookup(
    state_abbr_path: Path = DEFAULT_STATE_ABBR_PATH,
    canada_lookup_path: Path = DEFAULT_CANADA_LOOKUP_PATH,
    fallback_hierarchy: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frames = []
    if Path(state_abbr_path).exists():
        us = pd.read_excel(state_abbr_path)[["State", "Abbr"]].rename(
            columns={"State": "state_name", "Abbr": "st"}
        )
        us["country"] = "USA"
        frames.append(us)
    else:
        logger.warning("State_Abbr.xlsx not found at %s; US state names will be unavailable.", state_abbr_path)

    if Path(canada_lookup_path).exists():
        can = pd.read_excel(canada_lookup_path)[["State", "Abbr", "Country"]].rename(
            columns={"State": "state_name", "Abbr": "st", "Country": "country"}
        )
        can["country"] = can["country"].map(_normalize_country)
        frames.append(can)
    else:
        logger.warning(
            "canada_state_lookup.xlsx not found at %s; Canadian province names will be unavailable.",
            canada_lookup_path,
        )

    if frames:
        return pd.concat(frames, ignore_index=True).drop_duplicates(subset="st").reset_index(drop=True)

    if fallback_hierarchy is not None:
        logger.warning(
            "Both JEDI state lookup files are missing; falling back to hierarchy.csv for "
            "st->country mapping only. state_name will equal st."
        )
        fallback = fallback_hierarchy[["st", "country"]].copy()
        fallback["state_name"] = fallback["st"]
        fallback["country"] = fallback["country"].map(_normalize_country)
        return fallback[["st", "state_name", "country"]]

    raise FileNotFoundError(
        "No state/country lookup available: both JEDI lookup files are missing and no fallback "
        "hierarchy frame was provided."
    )
