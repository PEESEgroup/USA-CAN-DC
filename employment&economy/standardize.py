from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io import ModuleResults
from .monetary import load_monetary_basis, normalize_aidc_money, usd_price_factor



CONSTRUCTION_MERGED_COLUMNS = [
    "case_name",
    "impact_year",
    "tech",
    "st",
    "country",
    "scenario",
    "jobs_direct",
    "jobs_indirect",
    "jobs_induced",
    "jobs_indirect_induced",
    "jobs_total",
    "earnings_total_usd",
    "output_total_usd",
    "value_added_total_usd",
    "source_currency",
    "multiplier_source_currency",
    "source_price_year",
    "fx_source_currency_per_usd",
    "price_adjustment_factor",
    "multiplier_data_year",
    "multiplier_data_vintages",
    "deflator_source",
    "fx_application",
    "monetary_basis_verified",
    "currency",
    "price_year",
]

OPERATING_MERGED_COLUMNS = [
    "case_name",
    "impact_year",
    "tech",
    "st",
    "country",
    "scenario",
    "jobs_direct",
    "jobs_indirect",
    "jobs_induced",
    "jobs_indirect_induced",
    "jobs_total",
    "earnings_total_usd",
    "output_total_usd",
    "value_added_total_usd",
    "source_currency",
    "multiplier_source_currency",
    "source_price_year",
    "fx_source_currency_per_usd",
    "price_adjustment_factor",
    "multiplier_data_year",
    "multiplier_data_vintages",
    "deflator_source",
    "fx_application",
    "monetary_basis_verified",
    "currency",
    "price_year",
]


def _series_or_none(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series([None] * len(df), index=df.index, dtype="float64")


def _null_series(df: pd.DataFrame) -> pd.Series:
    return pd.Series([None] * len(df), index=df.index, dtype="float64")


def _ensure_aidc_money(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if "price_year" in df.columns and "currency" in df.columns:
        currencies = set(df["currency"].dropna().astype(str))
        years = set(
            pd.to_numeric(df["price_year"], errors="coerce").dropna().astype(int)
        )
        target_year = int(load_monetary_basis()["target_price_year"])
        if currencies != {"USD"} or years != {target_year}:
            raise ValueError(
                f"AIDC rows claim an incompatible monetary basis: currencies={currencies}, years={years}"
            )
        if df["price_year"].isna().any() or df["currency"].isna().any():
            raise ValueError(
                "AIDC rows contain missing currency or price_year metadata"
            )
        return df
    return normalize_aidc_money(df, columns)


def _empty_merged_rows(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _aidc_construction_rows(case_name: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_merged_rows(CONSTRUCTION_MERGED_COLUMNS)
    df = _ensure_aidc_money(
        df,
        [
            "total_construction_earnings_usd",
            "total_construction_output_usd",
            "total_construction_value_added_usd",
        ],
    )
    indirect = _series_or_none(df, "indirect_construction_job_years")
    induced = _series_or_none(df, "induced_construction_job_years")

    combined = _series_or_none(df, "indirect_induced_construction_job_years")
    has_split = "indirect_construction_job_years" in df.columns
    return pd.DataFrame(
        {
            "case_name": case_name,
            "impact_year": pd.to_numeric(df["impact_year"], errors="coerce"),
            "tech": "AIDC",
            "st": df["st"],
            "country": df["country"],
            "scenario": df.get(
                "scenario", pd.Series(["base"] * len(df), index=df.index)
            ),
            "jobs_direct": _series_or_none(df, "direct_construction_job_years"),
            "jobs_indirect": indirect if has_split else _null_series(df),
            "jobs_induced": induced if has_split else _null_series(df),
            "jobs_indirect_induced": (indirect + induced) if has_split else combined,
            "jobs_total": _series_or_none(df, "total_construction_job_years"),
            "earnings_total_usd": _series_or_none(
                df, "total_construction_earnings_usd"
            ),
            "output_total_usd": _series_or_none(df, "total_construction_output_usd"),
            "value_added_total_usd": _series_or_none(
                df, "total_construction_value_added_usd"
            ),
            "source_currency": df.get("source_currency", "USD"),
            "multiplier_source_currency": df.get("multiplier_source_currency", "USD"),
            "source_price_year": df.get("source_price_year", 2014),
            "fx_source_currency_per_usd": df.get("fx_source_currency_per_usd", 1.0),
            "price_adjustment_factor": df.get("price_adjustment_factor", 1.0),
            "multiplier_data_year": df.get("multiplier_data_year", 2014),
            "multiplier_data_vintages": df.get(
                "multiplier_data_vintages", df.get("multiplier_data_year", "2014")
            ),
            "deflator_source": df.get("deflator_source", "unknown"),
            "fx_application": df.get("fx_application", "unknown"),
            "monetary_basis_verified": df.get("monetary_basis_verified", False),
            "currency": df.get("currency", "USD"),
            "price_year": df.get("price_year", 2024),
        }
    )


def _aidc_operating_rows(case_name: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_merged_rows(OPERATING_MERGED_COLUMNS)
    df = _ensure_aidc_money(
        df,
        [
            "total_operating_earnings_usd",
            "total_operating_output_usd",
            "total_operating_value_added_usd",
        ],
    )
    indirect = _series_or_none(df, "indirect_operating_jobs")
    induced = _series_or_none(df, "induced_operating_jobs")
    combined = _series_or_none(df, "indirect_induced_operating_jobs")
    has_split = "indirect_operating_jobs" in df.columns
    row = pd.DataFrame(
        {
            "case_name": case_name,
            "impact_year": pd.to_numeric(df["impact_year"], errors="coerce"),
            "tech": "AIDC",
            "st": df["st"],
            "country": df["country"],
            "scenario": df.get(
                "scenario", pd.Series(["base"] * len(df), index=df.index)
            ),
            "jobs_direct": _series_or_none(df, "direct_operating_jobs"),
            "jobs_indirect": indirect if has_split else _null_series(df),
            "jobs_induced": induced if has_split else _null_series(df),
            "jobs_indirect_induced": (indirect + induced) if has_split else combined,
            "jobs_total": _series_or_none(df, "total_operating_jobs"),
            "earnings_total_usd": _series_or_none(df, "total_operating_earnings_usd"),
            "output_total_usd": _series_or_none(df, "total_operating_output_usd"),
            "value_added_total_usd": _series_or_none(
                df, "total_operating_value_added_usd"
            ),
            "source_currency": df.get("source_currency", "USD"),
            "multiplier_source_currency": df.get("multiplier_source_currency", "USD"),
            "source_price_year": df.get("source_price_year", 2014),
            "fx_source_currency_per_usd": df.get("fx_source_currency_per_usd", 1.0),
            "price_adjustment_factor": df.get("price_adjustment_factor", 1.0),
            "multiplier_data_year": df.get("multiplier_data_year", 2014),
            "multiplier_data_vintages": df.get(
                "multiplier_data_vintages", df.get("multiplier_data_year", "2014")
            ),
            "deflator_source": df.get("deflator_source", "unknown"),
            "fx_application": df.get("fx_application", "unknown"),
            "monetary_basis_verified": df.get("monetary_basis_verified", False),
            "currency": df.get("currency", "USD"),
            "price_year": df.get("price_year", 2024),
        }
    )
    return row


def _jedi_agg_rows(
    case_name: str,
    df: pd.DataFrame,
    prefix: str,
    fallback_price_year: int = 2024,
) -> pd.DataFrame:
    group_cols = ["impact_year", "tech", "st", "country"]
    col_map = {
        f"{prefix}_jobs_fte_direct": "jobs_direct",
        f"{prefix}_jobs_fte_indirect": "jobs_indirect",
        f"{prefix}_jobs_fte_induced": "jobs_induced",
        f"{prefix}_jobs_fte_total": "jobs_total",
        f"{prefix}_earnings_usd_total": "earnings_total_usd",
        f"{prefix}_output_usd_total": "output_total_usd",
        f"{prefix}_value_added_usd_total": "value_added_total_usd",
    }
    agg = {
        dst: pd.NamedAgg(column=src, aggfunc=lambda values: values.sum(min_count=1))
        for src, dst in col_map.items()
        if src in df.columns
    }
    grouped = df.groupby(group_cols, as_index=False).agg(**agg)
    if "jobs_indirect" in grouped.columns and "jobs_induced" in grouped.columns:
        grouped["jobs_indirect_induced"] = grouped["jobs_indirect"].fillna(0) + grouped[
            "jobs_induced"
        ].fillna(0)
    else:
        grouped["jobs_indirect_induced"] = pd.Series(
            [None] * len(grouped), index=grouped.index, dtype="float64"
        )
    price_values = df.get("price_year", fallback_price_year)
    if "currency" in df.columns:
        currencies = set(df["currency"].dropna().astype(str))
        if df["currency"].isna().any() or currencies != {"USD"}:
            raise ValueError(
                f"JEDI input contains mixed, missing, or non-USD currency metadata: {currencies}"
            )
    if isinstance(price_values, pd.Series):
        numeric_price_values = pd.to_numeric(price_values, errors="coerce")
        if numeric_price_values.isna().any():
            raise ValueError("JEDI input contains missing price years")
        source_years = numeric_price_values.unique()
        if len(source_years) != 1:
            raise ValueError(
                f"JEDI input contains mixed or missing price years: {source_years}"
            )
        source_year = int(source_years[0])
    else:
        source_year = int(fallback_price_year)
    factor, target_year = usd_price_factor(source_year)
    basis = load_monetary_basis()
    unknown_countries = sorted(set(grouped["country"].dropna()) - {"USA", "CAN"})
    if unknown_countries:
        raise ValueError(
            f"Unknown JEDI country codes for monetary metadata: {unknown_countries}"
        )
    canada = grouped["country"].eq("CAN")
    for column in ("earnings_total_usd", "output_total_usd", "value_added_total_usd"):
        if column in grouped.columns:
            grouped[column] = grouped[column] * factor
    grouped["case_name"] = case_name
    grouped["source_currency"] = "USD"
    grouped["multiplier_source_currency"] = "USD"
    grouped["source_price_year"] = source_year
    grouped["fx_source_currency_per_usd"] = 1.0
    grouped["price_adjustment_factor"] = factor
    grouped["multiplier_data_year"] = (
        grouped["tech"]
        .map({"utility pv": 2019, "rooftop pv": 2019, "land-based wind": 2016})
        .fillna(2014)
        .astype(int)
    )
    grouped["multiplier_data_vintages"] = grouped["multiplier_data_year"].astype(str)
    grouped["deflator_source"] = (
        f"ReEDS inputs/financials/deflator.csv; {source_year} USD to {target_year} USD"
    )
    grouped["fx_application"] = "none"
    can_basis = basis["CAN"]
    grouped.loc[canada, "source_currency"] = can_basis["source_currency"]
    grouped.loc[canada, "multiplier_source_currency"] = can_basis[
        "multiplier_source_currency"
    ]
    grouped.loc[canada, "fx_source_currency_per_usd"] = float(
        can_basis["fx_source_currency_per_usd"]
    )
    grouped.loc[canada, "multiplier_data_year"] = int(can_basis["multiplier_data_year"])
    grouped.loc[canada, "multiplier_data_vintages"] = str(
        can_basis["multiplier_data_year"]
    )
    grouped.loc[canada, "deflator_source"] = can_basis["deflator_source"]
    grouped.loc[canada, "fx_application"] = "embedded_in_jobs_multiplier"
    grouped["currency"] = "USD"
    grouped["price_year"] = target_year
    required_metadata = [
        "source_currency",
        "multiplier_source_currency",
        "source_price_year",
        "fx_source_currency_per_usd",
        "price_adjustment_factor",
        "multiplier_data_vintages",
        "deflator_source",
        "fx_application",
        "currency",
        "price_year",
    ]
    grouped["monetary_basis_verified"] = grouped[required_metadata].notna().all(axis=1)
    if not grouped["monetary_basis_verified"].all():
        raise ValueError("JEDI monetary metadata could not be verified for every row")
    grouped["impact_year"] = pd.to_numeric(grouped["impact_year"], errors="coerce")
    return grouped


def _validate_merged_money(merged: pd.DataFrame) -> None:
    if merged.empty:
        return
    currencies = set(merged["currency"].dropna().astype(str))
    years = set(pd.to_numeric(merged["price_year"], errors="raise").astype(int))
    if (
        currencies != {"USD"}
        or years != {2024}
        or merged["currency"].isna().any()
        or merged["price_year"].isna().any()
        or not merged["monetary_basis_verified"].fillna(False).all()
    ):
        raise ValueError(
            "Merged employment output requires verified currency=USD and price_year=2024; "
            f"found currencies={currencies}, years={years}"
        )


def merge_construction_impacts(
    case_name: str,
    aidc: ModuleResults,
    jedi: ModuleResults,
    jedi_price_year: int = 2024,
) -> pd.DataFrame:
    aidc_rows = _aidc_construction_rows(case_name, aidc.construction)
    jedi_rows = _jedi_agg_rows(
        case_name, jedi.construction, "construction", jedi_price_year
    )
    jedi_rows = pd.concat(
        [jedi_rows.assign(scenario=scenario) for scenario in ("low", "base", "high")],
        ignore_index=True,
    )
    for col in CONSTRUCTION_MERGED_COLUMNS:
        if col not in jedi_rows.columns:
            jedi_rows[col] = None
    merged = pd.concat(
        [
            aidc_rows[CONSTRUCTION_MERGED_COLUMNS],
            jedi_rows[CONSTRUCTION_MERGED_COLUMNS],
        ],
        ignore_index=True,
    )
    _validate_merged_money(merged)
    return merged.sort_values(["impact_year", "tech", "st", "scenario"]).reset_index(
        drop=True
    )


def merge_operating_impacts(
    case_name: str,
    aidc: ModuleResults,
    jedi: ModuleResults,
    jedi_price_year: int = 2024,
) -> pd.DataFrame:
    aidc_rows = _aidc_operating_rows(case_name, aidc.operations)
    jedi_rows = _jedi_agg_rows(case_name, jedi.operations, "operating", jedi_price_year)
    jedi_rows = pd.concat(
        [jedi_rows.assign(scenario=scenario) for scenario in ("low", "base", "high")],
        ignore_index=True,
    )
    for col in OPERATING_MERGED_COLUMNS:
        if col not in jedi_rows.columns:
            jedi_rows[col] = None
    merged = pd.concat(
        [aidc_rows[OPERATING_MERGED_COLUMNS], jedi_rows[OPERATING_MERGED_COLUMNS]],
        ignore_index=True,
    )
    _validate_merged_money(merged)
    return merged.sort_values(["impact_year", "tech", "st", "scenario"]).reset_index(
        drop=True
    )


def write_outputs(
    output_dir: Path, construction: pd.DataFrame, operating: pd.DataFrame
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    construction.to_csv(output_dir / "construction_impacts_annual.csv", index=False)
    operating.to_csv(output_dir / "operating_impacts_annual.csv", index=False)
    for scenario in ("low", "base", "high"):
        construction.loc[construction["scenario"].eq(scenario)].to_csv(
            output_dir / f"construction_impacts_annual_{scenario}.csv", index=False
        )
        operating.loc[operating["scenario"].eq(scenario)].to_csv(
            output_dir / f"operating_impacts_annual_{scenario}.csv", index=False
        )
