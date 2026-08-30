from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


DEFAULT_MONETARY_BASIS_PATH = Path(__file__).resolve().parent / "monetary_basis.yml"
DEFAULT_DEFLATOR_PATH = (
    Path(__file__).resolve().parents[1] / "inputs" / "financials" / "deflator.csv"
)


def load_monetary_basis(path: Path = DEFAULT_MONETARY_BASIS_PATH) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        basis = yaml.safe_load(handle)
    if basis.get("target_currency") != "USD":
        raise ValueError("Only a USD target currency is currently supported")
    return basis


def normalize_aidc_money(
    frame: pd.DataFrame,
    monetary_columns: list[str],
    basis_path: Path = DEFAULT_MONETARY_BASIS_PATH,
    multiplier_context: str | None = None,
    source_price_year: int = 2022,
) -> pd.DataFrame:
    result = frame.copy()
    basis = load_monetary_basis(basis_path)
    target_year = int(basis["target_price_year"])
    country_key = result["country"].replace({"Canada": "CAN"})
    unknown = sorted(set(country_key.dropna()) - {"USA", "CAN"})
    if unknown:
        raise ValueError(
            f"Unknown AIDC country codes for monetary normalization: {unknown}"
        )

    result["source_currency"] = country_key.map(lambda c: basis[c]["source_currency"])
    result["multiplier_source_currency"] = country_key.map(
        lambda c: basis[c].get(
            "multiplier_source_currency", basis[c]["source_currency"]
        )
    )
    source_price_year = int(source_price_year)
    price_factor, _ = usd_price_factor(source_price_year, basis_path=basis_path)
    result["source_price_year"] = source_price_year
    result["fx_source_currency_per_usd"] = country_key.map(
        lambda c: float(basis[c]["fx_source_currency_per_usd"])
    )
    result["price_adjustment_factor"] = price_factor
    result["multiplier_data_year"] = country_key.map(
        lambda c: int(basis[c]["multiplier_data_year"])
    )
    result["multiplier_data_vintages"] = (
        result["multiplier_data_year"].astype("Int64").astype(str)
    )
    result["deflator_source"] = (
        f"ReEDS inputs/financials/deflator.csv; {source_price_year} USD to {target_year} USD"
    )
    result["fx_application"] = country_key.map(
        lambda c: (
            "embedded_in_jobs_multiplier"
            if basis[c].get("fx_already_embedded")
            else "none"
        )
    )
    if multiplier_context not in {None, "construction", "operations"}:
        raise ValueError(
            "multiplier_context must be None, 'construction', or 'operations'"
        )
    if multiplier_context == "operations" and "st" in result.columns:




        mixed_nv = country_key.eq("USA") & result["st"].eq("NV")
        result.loc[mixed_nv, "multiplier_data_year"] = pd.NA
        result.loc[mixed_nv, "multiplier_data_vintages"] = "2014;2016"
        result.loc[mixed_nv, "deflator_source"] = (
            "Mixed JEDI Deflators sheets: standard-family 2014 and "
            "land-based wind 2016 NV/TCPU fallback"
        )
    for column in monetary_columns:
        if column in result.columns:
            result[column] = (
                pd.to_numeric(result[column], errors="coerce")
                * result["price_adjustment_factor"]
            )
    result["currency"] = basis["target_currency"]
    result["price_year"] = target_year
    required = [
        "source_currency",
        "source_price_year",
        "price_adjustment_factor",
        "multiplier_data_vintages",
        "deflator_source",
        "currency",
        "price_year",
    ]
    result["monetary_basis_verified"] = result[required].notna().all(axis=1)
    if not result["monetary_basis_verified"].all():
        raise ValueError("AIDC monetary basis could not be verified for every row")
    return result


def usd_price_factor(
    source_year: int,
    source: str = "standard_2014",
    basis_path: Path = DEFAULT_MONETARY_BASIS_PATH,
) -> tuple[float, int]:
    basis = load_monetary_basis(basis_path)
    target_year = int(basis["target_price_year"])
    deflator = pd.read_csv(DEFAULT_DEFLATOR_PATH)
    year_col = "*Dollar.Year" if "*Dollar.Year" in deflator.columns else "Dollar.Year"
    factors = deflator.set_index(pd.to_numeric(deflator[year_col], errors="raise"))[
        "Deflator"
    ]
    source_year = int(source_year)
    if source_year not in factors.index or target_year not in factors.index:
        raise ValueError(
            f"No ReEDS deflator configured for {source_year}->{target_year} in "
            f"{DEFAULT_DEFLATOR_PATH}"
        )
    return (
        float(factors.loc[source_year]) / float(factors.loc[target_year]),
        target_year,
    )
