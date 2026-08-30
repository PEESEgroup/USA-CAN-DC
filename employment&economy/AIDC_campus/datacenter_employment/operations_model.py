
from __future__ import annotations

import pandas as pd


def interpolate_mw_stock(
    it_cap: pd.DataFrame, modeled_years: list[int]
) -> pd.DataFrame:
    years_sorted = sorted(set(modeled_years))
    full_calendar_years = list(range(years_sorted[0], years_sorted[-1] + 1))

    out_frames = []
    for st, group in it_cap.groupby("st"):
        modeled_series = (
            group.set_index("t")["Value"]
            .reindex(years_sorted, fill_value=0.0)
            .astype(float)
        )
        calendar_series = modeled_series.reindex(full_calendar_years).interpolate(
            method="linear"
        )
        frame = (
            calendar_series.rename("mw_stock")
            .reset_index()
            .rename(columns={"index": "t"})
        )
        frame["st"] = st
        frame["is_interpolated"] = ~frame["t"].isin(years_sorted)
        out_frames.append(frame)

    if not out_frames:
        return pd.DataFrame(columns=["st", "t", "mw_stock", "is_interpolated"])
    return pd.concat(out_frames, ignore_index=True)[
        ["st", "t", "mw_stock", "is_interpolated"]
    ]


def compute_operating_jobs(
    it_cap: pd.DataFrame,
    modeled_years: list[int],
    direct_jobs_per_mw_operating: float,
    min_output_impact_year: int | None = None,
) -> pd.DataFrame:
    interpolated = interpolate_mw_stock(it_cap, modeled_years)
    interpolated["direct_jobs_per_mw_operating"] = direct_jobs_per_mw_operating
    interpolated["direct_operating_jobs"] = (
        interpolated["mw_stock"] * direct_jobs_per_mw_operating
    )
    interpolated = interpolated.rename(columns={"t": "impact_year"})
    if min_output_impact_year is not None:
        interpolated = interpolated[
            interpolated["impact_year"] >= min_output_impact_year
        ]
    return interpolated.sort_values(["impact_year", "st"]).reset_index(drop=True)
