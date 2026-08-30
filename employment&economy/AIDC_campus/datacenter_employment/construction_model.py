
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_year_gaps(modeled_years: list[int]) -> dict[int, int]:
    years = sorted(set(modeled_years))
    gaps: dict[int, int] = {}
    for i, year in enumerate(years):
        gaps[year] = 1 if i == 0 else year - years[i - 1]
    return gaps


def _prepare_increment_frame(
    it_inv: pd.DataFrame, modeled_years: list[int]
) -> pd.DataFrame:
    df = it_inv.rename(columns={"Value": "mw_increment"}).copy()
    negative_mask = df["mw_increment"] < 0
    if negative_mask.any():
        logger.warning(
            "%d datacenter_it_inv.csv row(s) have negative MW increments (endogenous siting "
            "reallocated IT load away from a state); clipping to 0 for construction job-years. "
            "States/years affected: %s",
            int(negative_mask.sum()),
            sorted(
                df.loc[negative_mask, ["st", "t"]].itertuples(index=False, name=None)
            ),
        )
        df.loc[negative_mask, "mw_increment"] = 0.0
    gaps = compute_year_gaps(modeled_years)
    df["gap_years"] = df["t"].map(gaps)
    if df["gap_years"].isna().any():
        missing = sorted(df.loc[df["gap_years"].isna(), "t"].unique())
        raise ValueError(
            f"datacenter_it_inv.csv contains year(s) not present in modeledyears.csv: {missing}"
        )
    df["is_first_observed_increment"] = df["t"] == min(modeled_years)
    return df


def compute_construction_job_years_from_intensity(
    it_inv: pd.DataFrame,
    modeled_years: list[int],
    direct_construction_job_years_per_mw_it: float,
    construction_duration_years: float = 2.0,
) -> pd.DataFrame:
    df = _prepare_increment_frame(it_inv, modeled_years)
    df["direct_construction_job_years_per_mw_it"] = (
        direct_construction_job_years_per_mw_it
    )
    df["construction_duration_years"] = construction_duration_years
    df["direct_construction_job_years_total"] = (
        df["mw_increment"] * direct_construction_job_years_per_mw_it
    )
    df["dollar_per_mw_construction"] = float("nan")
    df["beta_local"] = float("nan")
    df["eta_constr"] = float("nan")
    df["capex_million"] = float("nan")
    df["data_quality_flag"] = False
    return df


def compute_construction_job_years(
    it_inv: pd.DataFrame,
    modeled_years: list[int],
    dollar_per_mw_construction: float | None,
    beta_local: float,
    eta_constr: float,
    construction_duration_years: float = 2.0,
) -> pd.DataFrame:
    df = _prepare_increment_frame(it_inv, modeled_years)
    df["dollar_per_mw_construction"] = dollar_per_mw_construction
    df["beta_local"] = beta_local
    df["eta_constr"] = eta_constr
    df["direct_construction_job_years_per_mw_it"] = float("nan")
    df["construction_duration_years"] = construction_duration_years

    if dollar_per_mw_construction is None:
        df["capex_million"] = float("nan")
        df["direct_construction_job_years_total"] = float("nan")
        df["data_quality_flag"] = True
    else:
        df["capex_million"] = df["mw_increment"] * dollar_per_mw_construction / 1e6
        df["direct_construction_job_years_total"] = (
            df["capex_million"] * beta_local * eta_constr
        )
        df["data_quality_flag"] = False
    return df


def spread_construction_job_years(
    per_increment: pd.DataFrame, min_output_impact_year: int | None = None
) -> pd.DataFrame:
    rows: list[dict] = []
    for rec in per_increment.to_dict("records"):
        total = rec["direct_construction_job_years_total"]
        duration = rec.get("construction_duration_years")

        duration = 2.0 if pd.isna(duration) else float(duration)
        if duration <= 0:
            raise ValueError("construction_duration_years must be positive")
        deployment_year = int(rec["t"])
        interval_start = deployment_year - duration
        first_year = int(interval_start // 1)
        last_year = deployment_year - 1
        for impact_year in range(first_year, last_year + 1):
            overlap = max(
                0.0,
                min(float(impact_year + 1), float(deployment_year))
                - max(float(impact_year), interval_start),
            )
            if overlap <= 0:
                continue
            if (
                min_output_impact_year is not None
                and impact_year < min_output_impact_year
            ):
                continue
            row = dict(rec)
            row["deployment_year"] = deployment_year
            row["deployment_mw"] = rec["mw_increment"]
            row["construction_allocation_fraction"] = overlap / duration
            row["impact_year"] = impact_year
            row["direct_construction_job_years"] = (
                total * overlap / duration if pd.notna(total) else float("nan")
            )
            rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["impact_year", "st"]).reset_index(drop=True)
