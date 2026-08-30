
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SHARED_DATA = Path(__file__).resolve().parents[2] / "shared_data"
DEFAULT_CANADA_PROFILE_PATH = SHARED_DATA / "canada_regional_spend_profiles.csv"
DEFAULT_CANADA_LOCAL_SHARE_PATH = SHARED_DATA / "canada_product_local_shares.csv"
DEFAULT_CANADA_MARGIN_PATH = SHARED_DATA / "canada_product_margin_factors.csv"
DEFAULT_CANADA_ELECTRICITY_PRICE_PATH = SHARED_DATA / "canada_large_power_prices.csv"
DEFAULT_CANADA_PAYROLL_PATH = SHARED_DATA / "canada_payroll_parameters.csv"
CANADA_REGIONS = {"AB", "BC", "MB", "NB", "NL", "NS", "ON", "PE", "QC", "SK"}
QUALITY_ORDER = {
    "province_exact": 0,
    "province_modelled": 1,
    "canada_proxy": 2,
    "unresolved": 3,
}


def worst_quality(values) -> str:
    flags = [str(value) for value in values]
    return max(flags, key=lambda value: QUALITY_ORDER.get(value, 99))


@dataclass(frozen=True)
class CanadaRegionalData:
    profiles: pd.DataFrame
    local_shares: pd.DataFrame
    margins: pd.DataFrame
    electricity_prices: pd.DataFrame
    payroll: pd.DataFrame

    @classmethod
    def load(
        cls,
        profile_path: Path = DEFAULT_CANADA_PROFILE_PATH,
        local_share_path: Path = DEFAULT_CANADA_LOCAL_SHARE_PATH,
        margin_path: Path = DEFAULT_CANADA_MARGIN_PATH,
        electricity_price_path: Path = DEFAULT_CANADA_ELECTRICITY_PRICE_PATH,
        payroll_path: Path = DEFAULT_CANADA_PAYROLL_PATH,
    ) -> "CanadaRegionalData":
        data = cls(
            pd.read_csv(profile_path),
            pd.read_csv(local_share_path),
            pd.read_csv(margin_path),
            pd.read_csv(electricity_price_path),
            pd.read_csv(payroll_path),
        )
        data.validate()
        return data

    def validate(self) -> None:
        for name, frame in {
            "profiles": self.profiles,
            "local_shares": self.local_shares,
            "electricity_prices": self.electricity_prices,
            "payroll": self.payroll,
        }.items():
            missing = CANADA_REGIONS - set(frame["region"].astype(str))
            if missing:
                raise ValueError(f"Canada {name} missing provinces: {sorted(missing)}")
        flags = set(self.profiles["quality_flag"].astype(str))
        forbidden = {
            flag for flag in flags if "us_proxy" in flag or flag == "unresolved"
        }
        if forbidden:
            raise ValueError(
                f"Canada default profile contains forbidden quality flags: {forbidden}"
            )
        for column in ("local_share", "margin_share", "amount_per_anchor"):
            values = pd.to_numeric(self.profiles[column], errors="raise")
            if not np.isfinite(values).all():
                raise ValueError(f"Canada profile {column} must be finite")
            included = self.profiles["include_in_campus_total"].astype(bool)
            if column != "margin_share" and (values[included] < 0).any():
                raise ValueError(
                    f"Included Canada profile {column} must be non-negative"
                )
        if not self.profiles["local_share"].between(0, 1).all():
            raise ValueError("Canada profile local_share must be within [0, 1]")
        methods = set(self.profiles["scale_method"])
        expected = {
            "per_local_construction_dollar",
            "non_electric_composition_share",
        }
        if methods != expected:
            raise ValueError(f"Unexpected Canada scale methods: {sorted(methods)}")
        if not self.payroll["employee_wage_share"].between(0, 1).all():
            raise ValueError("Canada employee wage shares must be within [0, 1]")
        household_limits = {
            "wage_salary_share": 1.0,
            "dpi_to_personal_income": 1.0,
            "pce_to_dpi": 1.25,
            "employee_compensation_to_consumption": 1.0,
        }
        for column, upper in household_limits.items():
            values = pd.to_numeric(self.payroll[column], errors="raise")
            if not values.between(0, upper, inclusive="right").all():
                raise ValueError(f"Invalid Canada household parameter {column}")
        if (
            not self.payroll["employee_compensation_per_output"]
            .between(0, 1, inclusive="right")
            .all()
        ):
            raise ValueError("Canada employee-compensation/output shares are invalid")
        if not self.payroll["employee_compensation_per_job_2022_usd"].gt(0).all():
            raise ValueError("Canada employee compensation per job must be positive")
        if (
            not self.electricity_prices["electricity_price_2022_usd_per_mwh"]
            .gt(0)
            .all()
        ):
            raise ValueError("Canada large-power prices must be positive")
        margin_values = pd.to_numeric(self.margins["factor_share"], errors="raise")
        if not np.isfinite(margin_values).all():
            raise ValueError("Canada product margin factors must be finite")
        margin_sums = self.margins.groupby(["region", "statcan_product"])[
            "factor_share"
        ].sum()
        if not np.allclose(margin_sums, 1.0, rtol=0.0, atol=1e-8):
            raise ValueError(
                "Canada product margin factors must balance to purchaser price"
            )

    def rows(self, region: str, phase: str) -> pd.DataFrame:
        rows = self.profiles[
            self.profiles["region"].astype(str).eq(str(region))
            & self.profiles["phase"].eq(phase)
        ].copy()
        if rows.empty:
            raise KeyError(f"No Canada regional spend profile for {region}/{phase}")
        return rows.reset_index(drop=True)

    def employee_wage_share(self, region: str) -> float:
        row = self.payroll[self.payroll["region"].astype(str).eq(str(region))]
        if len(row) != 1:
            raise KeyError(f"No unique Canada payroll parameter for {region}")
        return float(row.iloc[0]["employee_wage_share"])

    def household_parameters(self, region: str) -> dict[str, object]:
        row = self.payroll[self.payroll["region"].astype(str).eq(str(region))]
        if len(row) != 1:
            raise KeyError(f"No unique Canada household parameter row for {region}")
        selected = row.iloc[0]
        return {
            "wage_salary_share": float(selected["wage_salary_share"]),
            "dpi_to_personal_income": float(selected["dpi_to_personal_income"]),
            "pce_to_dpi": float(selected["pce_to_dpi"]),
            "employee_compensation_to_consumption": float(
                selected["employee_compensation_to_consumption"]
            ),
            "negative_household_saving": str(selected["negative_household_saving"])
            .strip()
            .lower()
            in {"true", "1", "yes"},
            "household_factor_fallback": str(selected["household_factor_fallback"]),
        }

    def employee_compensation_per_job(self, region: str) -> float:
        row = self.payroll[self.payroll["region"].astype(str).eq(str(region))]
        if len(row) != 1:
            raise KeyError(f"No unique Canada compensation anchor for {region}")
        return float(row.iloc[0]["employee_compensation_per_job_2022_usd"])

    def electricity_price(self, region: str) -> float:
        row = self.electricity_prices[
            self.electricity_prices["region"].astype(str).eq(str(region))
        ]
        if len(row) != 1:
            raise KeyError(f"No unique Canada electricity price for {region}")
        return float(row.iloc[0]["electricity_price_2022_usd_per_mwh"])

    def local_share(
        self, region: str, industry: str, product: str | None = None
    ) -> float:
        rows = self.local_shares[
            self.local_shares["region"].astype(str).eq(str(region))
        ]
        if product is not None:
            exact = rows[rows["statcan_product"].astype(str).eq(str(product))]
            if not exact.empty and np.isfinite(exact.iloc[0]["local_share"]):
                return float(exact.iloc[0]["local_share"])
        aggregate = rows[
            rows["statcan_product"].eq("__JEDI_AGGREGATE__")
            & rows["jedi_industry"].eq(industry)
        ]
        if aggregate.empty:
            raise KeyError(f"No Canada local-share fallback for {region}/{industry}")
        return float(aggregate.iloc[0]["local_share"])

    def quality(self, region: str, phase: str) -> str:
        values = list(self.rows(region, phase)["quality_flag"])
        values.extend(
            self.payroll.loc[
                self.payroll["region"].astype(str).eq(str(region)), "quality_flag"
            ].tolist()
        )
        return worst_quality(values)
