
from __future__ import annotations

import pandas as pd

from .assumptions import AssumptionRegister, GeoMultiplierTable


def apply_multipliers(
    direct: pd.Series,
    m_tot_dir: float | pd.Series,
) -> tuple[pd.Series, pd.Series]:
    total = direct * m_tot_dir
    indirect_plus_induced = total - direct
    return indirect_plus_induced, total


def build_geo_multiplier_series(
    region_col: pd.Series,
    geo_table: GeoMultiplierTable,
    phase: str,
    scenario: str,
) -> pd.Series:
    if phase == "construction":
        return region_col.map(lambda r: geo_table.constr_scalar(r, scenario))
    if phase == "operations":
        return region_col.map(lambda r: geo_table.ops_scalar(r, scenario))
    raise ValueError(
        f"Unknown phase: {phase!r}. Expected 'construction' or 'operations'."
    )







def build_geo_constr_indirect_series(
    region_col: pd.Series,
    geo_table: GeoMultiplierTable,
    scenario: str,
) -> pd.Series:
    return region_col.map(lambda r: geo_table.constr_indirect_scalar(r, scenario))


def build_geo_constr_induced_series(
    region_col: pd.Series,
    geo_table: GeoMultiplierTable,
    scenario: str,
) -> pd.Series:
    return region_col.map(lambda r: geo_table.constr_induced_scalar(r, scenario))


def build_geo_ops_indirect_series(
    region_col: pd.Series,
    geo_table: GeoMultiplierTable,
    scenario: str,
) -> pd.Series:
    return region_col.map(lambda r: geo_table.ops_indirect_scalar(r, scenario))


def build_geo_ops_induced_series(
    region_col: pd.Series,
    geo_table: GeoMultiplierTable,
    scenario: str,
) -> pd.Series:
    return region_col.map(lambda r: geo_table.ops_induced_scalar(r, scenario))







def build_geo_constr_earnings_per_job_series(
    region_col: pd.Series,
    geo_table: GeoMultiplierTable,
    effect: str,
    scenario: str,
) -> pd.Series:

    def _get(r: str) -> float:
        try:
            return geo_table.constr_earnings_per_job_scalar(r, effect, scenario)
        except KeyError:
            return float("nan")

    return region_col.map(_get)


def build_geo_ops_earnings_per_job_series(
    region_col: pd.Series,
    geo_table: GeoMultiplierTable,
    effect: str,
    scenario: str,
) -> pd.Series:

    def _get(r: str) -> float:
        try:
            return geo_table.ops_earnings_per_job_scalar(r, effect, scenario)
        except KeyError:
            return float("nan")

    return region_col.map(_get)


def build_geo_constr_output_per_job_series(
    region_col: pd.Series,
    geo_table: GeoMultiplierTable,
    effect: str,
    scenario: str,
) -> pd.Series:

    def _get(r: str) -> float:
        try:
            return geo_table.constr_output_per_job_scalar(r, effect, scenario)
        except KeyError:
            return float("nan")

    return region_col.map(_get)


def build_geo_ops_output_per_job_series(
    region_col: pd.Series,
    geo_table: GeoMultiplierTable,
    effect: str,
    scenario: str,
) -> pd.Series:

    def _get(r: str) -> float:
        try:
            return geo_table.ops_output_per_job_scalar(r, effect, scenario)
        except KeyError:
            return float("nan")

    return region_col.map(_get)


def build_geo_constr_metric_per_direct_job_series(
    region_col: pd.Series,
    geo_table: GeoMultiplierTable,
    metric: str,
    effect: str,
    scenario: str,
) -> pd.Series:

    def _get(region: str) -> float:
        try:
            return geo_table.constr_metric_per_direct_job_scalar(
                region, metric, effect, scenario
            )
        except KeyError:
            return float("nan")

    return region_col.map(_get)


def build_geo_ops_metric_per_direct_job_series(
    region_col: pd.Series,
    geo_table: GeoMultiplierTable,
    metric: str,
    effect: str,
    scenario: str,
) -> pd.Series:

    def _get(region: str) -> float:
        try:
            return geo_table.ops_metric_per_direct_job_scalar(
                region, metric, effect, scenario
            )
        except KeyError:
            return float("nan")

    return region_col.map(_get)
