
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .calibration import ECONOMIC_METHOD_VERSION, DataCenterCalibration


ECONOMIC_METHOD = "public_bea_lq_io"
MODEL_PRICE_YEAR = 2024
DEFAULT_NORMALIZED_DIR = (
    Path(__file__).resolve().parents[2] / "shared_data" / "public_bea_lq_io"
)


@dataclass(frozen=True)
class RegionalMatrices:

    type_i: np.ndarray
    type_ii: np.ndarray
    lps: np.ndarray
    jobs_per_output: np.ndarray
    earnings_per_output: np.ndarray
    value_added_per_output: np.ndarray
    wage_per_output: np.ndarray
    spectral_radius_type_i: float
    spectral_radius_type_ii: float


def _require_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def _finite_nonnegative(values: np.ndarray, name: str) -> None:
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{name} must be finite and non-negative")


def _metric_frame(
    industries: pd.DataFrame,
    output: np.ndarray,
    matrices: RegionalMatrices,
    effect: str,
    classification: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bea_industry": industries["bea_industry"].astype(str).to_numpy(),
            "industry_title": industries["industry_title"].astype(str).to_numpy(),
            "effect": effect,
            "classification": classification,
            "jobs": output * matrices.jobs_per_output,
            "earnings_usd": output * matrices.earnings_per_output,
            "output_usd": output,
            "value_added_usd": output * matrices.value_added_per_output,
        }
    )


class PublicBEALQIO:

    REQUIRED_FILES = {
        "industries": "bea_industries.csv",
        "matrix": "bea_domestic_requirements_2017.npz",
        "regional": "us_regional_coefficients_2024.csv",
        "pce": "us_state_pce_producer_shares_2024.csv",
        "assets": "us_aces_asset_profile.csv",
        "operations": "us_datacenter_operating_inputs.csv",
        "margins": "us_purchaser_margins_2017.csv",
    }

    def __init__(
        self,
        industries: pd.DataFrame,
        national_a: np.ndarray,
        regional: pd.DataFrame,
        pce: pd.DataFrame,
        assets: pd.DataFrame,
        operations: pd.DataFrame,
        margins: pd.DataFrame,
        calibration: DataCenterCalibration | None = None,
    ):
        self.industries = industries.reset_index(drop=True).copy()
        self.national_a = np.asarray(national_a, dtype=float)
        self.regional = regional.copy()
        self.pce = pce.copy()
        self.assets = assets.copy()
        self.operations = operations.copy()
        self.margins = margins.copy()
        self.calibration = calibration or DataCenterCalibration.from_assumptions()
        self._validate()
        self.codes = self.industries["bea_industry"].astype(str).tolist()
        self.index = {code: idx for idx, code in enumerate(self.codes)}
        self._matrix_cache: dict[str, RegionalMatrices] = {}

    @classmethod
    def load(
        cls,
        directory: Path,
        calibration: DataCenterCalibration | None = None,
    ) -> "PublicBEALQIO":
        directory = Path(directory)
        missing = [
            filename
            for filename in cls.REQUIRED_FILES.values()
            if not (directory / filename).exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Public BEA-LQ normalized inputs are incomplete. Run "
                "`python -m employment.data_pipeline download` followed by `build`. "
                f"Missing: {missing}"
            )
        matrix_file = np.load(directory / cls.REQUIRED_FILES["matrix"])
        if "a" not in matrix_file:
            raise ValueError("BEA requirements NPZ must contain array 'a'")
        return cls(
            pd.read_csv(directory / cls.REQUIRED_FILES["industries"]),
            matrix_file["a"],
            pd.read_csv(directory / cls.REQUIRED_FILES["regional"]),
            pd.read_csv(directory / cls.REQUIRED_FILES["pce"]),
            pd.read_csv(directory / cls.REQUIRED_FILES["assets"]),
            pd.read_csv(directory / cls.REQUIRED_FILES["operations"]),
            pd.read_csv(directory / cls.REQUIRED_FILES["margins"]),
            calibration,
        )

    def _validate(self) -> None:
        _require_columns(
            self.industries,
            {
                "bea_industry",
                "industry_title",
                "national_jobs_per_output_2024",
                "national_earnings_per_output_2024",
                "national_value_added_per_output_2024",
                "national_wage_per_output_2024",
                "source_data_year",
                "source_price_year",
                "currency",
                "price_year",
            },
            "bea_industries",
        )
        n = len(self.industries)
        if self.national_a.shape != (n, n):
            raise ValueError(
                f"Domestic requirements matrix shape {self.national_a.shape} != {(n, n)}"
            )
        _finite_nonnegative(self.national_a, "national domestic requirements")
        _require_columns(
            self.regional,
            {
                "region",
                "bea_industry",
                "lps",
                "lps_basis",
                "jobs_per_output_2024",
                "earnings_per_output_2024",
                "value_added_per_output_2024",
                "wage_per_output_2024",
                "qcew_fallback_level",
                "source_data_year",
                "source_price_year",
                "currency",
                "price_year",
                "download_url",
                "sha256",
                "fallback_status",
                "mapping_status",
            },
            "us_regional_coefficients_2024",
        )
        _require_columns(
            self.pce,
            {
                "region",
                "bea_industry",
                "producer_share",
                "wage_salary_share",
                "dpi_to_personal_income",
                "pce_to_dpi",
                "employee_compensation_to_consumption",
                "negative_household_saving",
                "household_factor_fallback",
            },
            "us_state_pce_producer_shares_2024",
        )
        _require_columns(
            self.assets,
            {"asset_component", "purchaser_industry", "asset_share", "excluded"},
            "us_aces_asset_profile",
        )
        _require_columns(
            self.operations,
            {"input_commodity", "non_electric_input_share", "electricity_flag"},
            "us_datacenter_operating_inputs",
        )
        _require_columns(
            self.margins,
            {"purchaser_industry", "producer_industry", "margin_share", "tax_flag"},
            "us_purchaser_margins_2017",
        )
        if self.industries["bea_industry"].astype(str).duplicated().any():
            raise ValueError("Duplicate BEA detailed industries")
        for frame_name, frame in (
            ("industries", self.industries),
            ("regional", self.regional),
            ("pce", self.pce),
            ("assets", self.assets),
            ("operations", self.operations),
            ("margins", self.margins),
        ):
            _require_columns(
                frame,
                {
                    "source_data_year",
                    "source_price_year",
                    "currency",
                    "price_year",
                    "download_url",
                    "sha256",
                    "fallback_status",
                    "mapping_status",
                },
                frame_name,
            )
            if set(frame["currency"].astype(str)) != {"USD"}:
                raise ValueError(f"{frame_name} must use USD")
            if set(pd.to_numeric(frame["price_year"])) != {MODEL_PRICE_YEAR}:
                raise ValueError(f"{frame_name} must use constant 2024 USD")
        if not self.regional["lps"].between(0.0, 1.0).all():
            raise ValueError("Regional LPS values must be within [0, 1]")
        if not set(self.regional["lps_basis"]).issubset({"faf", "wage_lq"}):
            raise ValueError("LPS basis must be FAF for goods or wage_lq otherwise")
        if not self.regional["mapping_status"].eq("resolved").all():
            bad = self.regional.loc[
                ~self.regional["mapping_status"].eq("resolved"),
                ["region", "bea_industry", "mapping_status"],
            ]
            raise ValueError(
                f"Unresolved production industry mappings:\n{bad.head(20)}"
            )
        expected = set(self.industries["bea_industry"].astype(str))
        for region, rows in self.regional.groupby("region"):
            actual = set(rows["bea_industry"].astype(str))
            if actual != expected:
                raise ValueError(
                    f"{region} regional coefficients do not cover all detailed industries"
                )
        pce_sums = self.pce.groupby("region")["producer_share"].sum()
        if (pce_sums <= 0).any() or (pce_sums > 1.0 + 1e-9).any():
            raise ValueError("State PCE producer shares must sum within (0, 1]")
        for column, upper in (
            ("wage_salary_share", 1.0),
            ("dpi_to_personal_income", 1.0),
            ("pce_to_dpi", 1.25),
            ("employee_compensation_to_consumption", 1.0),
        ):
            if not self.pce[column].between(0.0, upper, inclusive="right").all():
                raise ValueError(f"Invalid household parameter {column}")
        non_electric = self.operations.loc[
            ~self.operations["electricity_flag"].astype(bool),
            "non_electric_input_share",
        ]
        if not np.isclose(non_electric.sum(), 1.0, atol=1e-9):
            raise ValueError("Non-electric operating input shares must sum to one")
        included_assets = self.assets.loc[~self.assets["excluded"].astype(bool)]
        if not np.isclose(included_assets["asset_share"].sum(), 1.0, atol=1e-9):
            raise ValueError("Included ACES tangible asset shares must sum to one")
        margin_sums = self.margins.groupby("purchaser_industry")["margin_share"].sum()
        if not np.allclose(margin_sums, 1.0, atol=1e-8):
            raise ValueError(
                "Purchaser margins must sum to one, including excluded taxes"
            )

    def regional_matrices(self, region: str) -> RegionalMatrices:
        region = str(region)
        if region in self._matrix_cache:
            return self._matrix_cache[region]
        rows = (
            self.regional[self.regional["region"].astype(str).eq(region)]
            .assign(bea_industry=lambda x: x["bea_industry"].astype(str))
            .set_index("bea_industry")
            .reindex(self.industries["bea_industry"].astype(str))
        )
        required_numeric = [
            "lps",
            "jobs_per_output_2024",
            "earnings_per_output_2024",
            "value_added_per_output_2024",
            "wage_per_output_2024",
        ]
        if rows[required_numeric].isna().any().any():
            raise ValueError(f"Incomplete regional inputs for {region}")
        lps = rows["lps"].to_numpy(float)
        a_region = lps[:, None] * self.national_a
        radius_i = float(max(abs(np.linalg.eigvals(a_region))))
        if radius_i >= 1.0:
            raise ValueError(f"Type I spectral radius is {radius_i:.6f} for {region}")
        type_i = np.linalg.inv(np.eye(len(lps)) - a_region)

        regional_pce = self.pce[self.pce["region"].astype(str).eq(region)]
        eta_values = regional_pce["employee_compensation_to_consumption"].unique()
        if len(eta_values) != 1:
            raise ValueError(f"Expected one household consumption factor for {region}")
        eta = float(eta_values[0])
        pce_rows = (
            regional_pce.assign(bea_industry=lambda x: x["bea_industry"].astype(str))
            .groupby("bea_industry", as_index=True)["producer_share"]
            .sum()
            .reindex(self.industries["bea_industry"].astype(str), fill_value=0.0)
        )


        wage_row = rows["earnings_per_output_2024"].to_numpy(float)
        augmented = np.zeros((len(lps) + 1, len(lps) + 1), dtype=float)
        augmented[:-1, :-1] = a_region
        augmented[:-1, -1] = eta * pce_rows.to_numpy(float)
        augmented[-1, :-1] = wage_row
        radius_ii = float(max(abs(np.linalg.eigvals(augmented))))
        if radius_ii >= 1.0:
            raise ValueError(f"Type II spectral radius is {radius_ii:.6f} for {region}")
        inverse_augmented = np.linalg.inv(np.eye(len(lps) + 1) - augmented)
        type_ii = inverse_augmented[:-1, :-1]
        matrices = RegionalMatrices(
            type_i=type_i,
            type_ii=type_ii,
            lps=lps,
            jobs_per_output=rows["jobs_per_output_2024"].to_numpy(float),
            earnings_per_output=rows["earnings_per_output_2024"].to_numpy(float),
            value_added_per_output=rows["value_added_per_output_2024"].to_numpy(float),
            wage_per_output=wage_row,
            spectral_radius_type_i=radius_i,
            spectral_radius_type_ii=radius_ii,
        )
        self._matrix_cache[region] = matrices
        return matrices

    def _producer_vector(
        self, region: str, purchaser_spend: pd.DataFrame
    ) -> np.ndarray:
        _require_columns(
            purchaser_spend, {"purchaser_industry", "purchaser_spend_usd"}, "demand"
        )
        expanded = purchaser_spend.merge(
            self.margins,
            on="purchaser_industry",
            how="left",
            validate="many_to_many",
        )
        if expanded["margin_share"].isna().any():
            missing = expanded.loc[
                expanded["margin_share"].isna(), "purchaser_industry"
            ].unique()
            raise KeyError(f"Missing purchaser margins for {missing.tolist()}")
        expanded = expanded.loc[~expanded["tax_flag"].astype(bool)].copy()
        expanded["producer_spend_usd"] = (
            expanded["purchaser_spend_usd"] * expanded["margin_share"]
        )
        regional = self.regional[self.regional["region"].astype(str).eq(str(region))][
            ["bea_industry", "lps"]
        ].copy()
        regional["bea_industry"] = regional["bea_industry"].astype(str)
        expanded["producer_industry"] = expanded["producer_industry"].astype(str)
        expanded = expanded.merge(
            regional,
            left_on="producer_industry",
            right_on="bea_industry",
            how="left",
            validate="many_to_one",
        )
        if expanded["lps"].isna().any():
            missing = expanded.loc[expanded["lps"].isna(), "producer_industry"].unique()
            raise KeyError(f"Missing LPS for producer industries {missing.tolist()}")
        expanded["local_producer_spend_usd"] = (
            expanded["producer_spend_usd"] * expanded["lps"]
        )
        vector = np.zeros(len(self.industries), dtype=float)
        for row in expanded.itertuples(index=False):
            vector[self.index[str(row.producer_industry)]] += float(
                row.local_producer_spend_usd
            )
        return vector

    def _equipment_local_final_demand(self, tangible: pd.DataFrame) -> np.ndarray:
        _require_columns(
            tangible, {"purchaser_industry", "purchaser_spend_usd"}, "equipment demand"
        )
        expanded = tangible[["purchaser_industry", "purchaser_spend_usd"]].merge(
            self.margins,
            on="purchaser_industry",
            how="left",
            validate="many_to_many",
        )
        if expanded["margin_share"].isna().any():
            missing = expanded.loc[
                expanded["margin_share"].isna(), "purchaser_industry"
            ].unique()
            raise KeyError(f"Missing purchaser margins for {missing.tolist()}")
        expanded = expanded.loc[~expanded["tax_flag"].astype(bool)].copy()
        expanded["producer_spend_usd"] = (
            expanded["purchaser_spend_usd"] * expanded["margin_share"]
        )
        expanded["local_producer_spend_usd"] = (
            expanded["producer_spend_usd"] * self.calibration.equipment_indirect_local_share
        )
        vector = np.zeros(len(self.industries), dtype=float)
        for row in expanded.itertuples(index=False):
            vector[self.index[str(row.producer_industry)]] += float(
                row.local_producer_spend_usd
            )
        return vector

    def _impact_detail(
        self,
        region: str,
        type_i_output: np.ndarray,
        induced_output: np.ndarray,
        initial_direct_output: np.ndarray | None = None,
    ) -> pd.DataFrame:
        matrices = self.regional_matrices(region)
        direct = (
            np.zeros_like(type_i_output)
            if initial_direct_output is None
            else initial_direct_output
        )
        indirect = type_i_output - direct
        if (indirect < -1e-6).any() or (induced_output < -1e-6).any():
            raise ValueError("Non-negative final demand produced negative impacts")
        frames = [
            _metric_frame(
                self.industries,
                np.maximum(indirect, 0.0),
                matrices,
                "indirect",
                "supplier",
            ),
            _metric_frame(
                self.industries,
                np.maximum(induced_output, 0.0),
                matrices,
                "induced",
                "household",
            ),
        ]
        if initial_direct_output is not None:
            frames.insert(
                0,
                _metric_frame(
                    self.industries,
                    direct,
                    matrices,
                    "direct",
                    "capacity_anchored_producer",
                ),
            )
        return pd.concat(frames, ignore_index=True)

    def construction_impacts(
        self, region: str, direct_job_years: float
    ) -> tuple[dict[str, float | str], pd.DataFrame]:
        direct_job_years = float(direct_job_years)
        if direct_job_years < 0:
            raise ValueError("Direct construction job-years cannot be negative")
        matrices = self.regional_matrices(region)
        construction_code = str(
            self.assets.loc[
                self.assets["asset_component"].eq("building_and_site"),
                "purchaser_industry",
            ].iloc[0]
        )
        construction_idx = self.index[construction_code]
        construction_jobs_per_output = matrices.jobs_per_output[construction_idx]
        if construction_jobs_per_output <= 0:
            raise ValueError(
                f"No construction employment/output coefficient for {region}"
            )
        local_construction_spend = direct_job_years / construction_jobs_per_output
        construction_initial = np.zeros(len(self.industries), dtype=float)
        construction_initial[construction_idx] = local_construction_spend

        tangible = self.assets.loc[
            ~self.assets["excluded"].astype(bool)
            & ~self.assets["asset_component"].eq("building_and_site")
        ].copy()
        gross_capex = (
            local_construction_spend / self.calibration.capex_share_construction
        )
        equipment_spend = gross_capex * self.calibration.capex_share_equipment
        land_spend = gross_capex * self.calibration.capex_share_land
        other_spend = gross_capex * self.calibration.capex_share_other
        tangible["purchaser_spend_usd"] = equipment_spend * tangible["asset_share"]
        equipment_initial = self._equipment_local_final_demand(tangible)
        final_demand = construction_initial + equipment_initial
        type_i_output = matrices.type_i @ final_demand
        type_ii_output = matrices.type_ii @ final_demand
        induced_output = type_ii_output - type_i_output
        detail = self._impact_detail(
            region, type_i_output, induced_output, construction_initial
        )


        detail.loc[detail["effect"].eq("direct"), "jobs"] = 0.0
        detail.loc[
            detail["effect"].eq("direct")
            & detail["bea_industry"].eq(construction_code),
            "jobs",
        ] = direct_job_years
        summary = self._summarize(detail, "construction", direct_job_years)
        summary.update(
            {
                "local_construction_final_demand_usd": local_construction_spend,
                "gross_campus_capex_usd": gross_capex,
                "gross_land_capex_usd": land_spend,
                "gross_construction_capex_usd": local_construction_spend,
                "gross_equipment_capex_usd": equipment_spend,
                "gross_other_capex_usd": other_spend,
                "gross_equipment_purchaser_spend_usd": equipment_spend,
                "equipment_local_final_demand_usd": float(equipment_initial.sum()),
                "land_excluded_from_io": True,
                "other_capex_excluded_from_io": True,
                "economic_method": ECONOMIC_METHOD,
                "economic_method_version": ECONOMIC_METHOD_VERSION,
                "equipment_to_construction_ratio": (
                    self.calibration.equipment_to_construction_ratio
                ),
                "equipment_indirect_local_share": (
                    self.calibration.equipment_indirect_local_share
                ),
                "direct_job_anchor_replaced_io_jobs": True,
                "currency": "USD",
                "price_year": MODEL_PRICE_YEAR,
            }
        )
        return summary, detail

    def operations_impacts(
        self, region: str, direct_jobs: float
    ) -> tuple[dict[str, float | str], pd.DataFrame]:
        direct_jobs = float(direct_jobs)
        if direct_jobs < 0:
            raise ValueError("Direct operating jobs cannot be negative")
        matrices = self.regional_matrices(region)
        coefficients = self.regional[
            self.regional["region"].astype(str).eq(str(region))
            & self.regional["bea_industry"].astype(str).eq("518200")
        ]
        if coefficients.empty:
            raise KeyError(
                f"Missing regional 518200 operating coefficients for {region}"
            )
        op = coefficients.iloc[0]
        compensation_per_job = float(op["earnings_per_output_2024"]) / float(
            op["jobs_per_output_2024"]
        )
        direct_compensation = direct_jobs * compensation_per_job
        vendor = self.operations.loc[
            ~self.operations["electricity_flag"].astype(bool)
        ].copy()
        direct_output = (
            direct_compensation / self.calibration.operating_employee_comp_share
        )
        direct_value_added = (
            direct_output * self.calibration.operating_value_added_share
        )
        vendor_budget = direct_output * self.calibration.operating_non_electric_share
        vendor["purchaser_industry"] = vendor["input_commodity"].astype(str)
        vendor["purchaser_spend_usd"] = (
            vendor_budget * vendor["non_electric_input_share"]
        )
        vendor_initial = self._producer_vector(
            region, vendor[["purchaser_industry", "purchaser_spend_usd"]]
        )
        type_i_vendor = matrices.type_i @ vendor_initial
        type_ii_vendor = matrices.type_ii @ vendor_initial
        supply_induced = type_ii_vendor - type_i_vendor

        pce_rows = self.pce[self.pce["region"].astype(str).eq(str(region))]
        wage_share = float(pce_rows["wage_salary_share"].iloc[0])
        eta = float(pce_rows["employee_compensation_to_consumption"].iloc[0])
        household_initial = np.zeros(len(self.industries), dtype=float)
        for row in pce_rows.itertuples(index=False):
            household_initial[self.index[str(row.bea_industry)]] += (
                direct_compensation * eta * float(row.producer_share)
            )
        household_output = matrices.type_i @ household_initial
        indirect_detail = _metric_frame(
            self.industries,
            type_i_vendor,
            matrices,
            "indirect",
            "operating_vendor",
        )
        induced_detail = _metric_frame(
            self.industries,
            supply_induced + household_output,
            matrices,
            "induced",
            "supply_chain_and_direct_payroll_household",
        )
        direct_detail = pd.DataFrame(
            {
                "bea_industry": ["518200"],
                "industry_title": ["Data processing, hosting, and related services"],
                "effect": ["direct"],
                "classification": ["capacity_anchored_operating"],
                "jobs": [direct_jobs],
                "earnings_usd": [direct_compensation],
                "output_usd": [direct_output],
                "value_added_usd": [direct_value_added],
            }
        )
        detail = pd.concat(
            [direct_detail, indirect_detail, induced_detail], ignore_index=True
        )
        summary = self._summarize(detail, "operating", direct_jobs)
        summary.update(
            {
                "direct_operating_compensation_per_job_2024_usd": compensation_per_job,
                "direct_operating_employee_compensation_usd": direct_compensation,
                "direct_operating_payroll_usd": direct_compensation,
                "direct_operating_output_usd": direct_output,
                "direct_operating_value_added_usd": direct_value_added,
                "vendor_purchaser_spend_usd": float(
                    vendor["purchaser_spend_usd"].sum()
                ),
                "direct_wage_household_demand_usd": direct_compensation * eta,
                "employee_wage_share": wage_share,
                "wage_salary_share": wage_share,
                "dpi_to_personal_income": float(
                    pce_rows["dpi_to_personal_income"].iloc[0]
                ),
                "pce_to_dpi": float(pce_rows["pce_to_dpi"].iloc[0]),
                "employee_compensation_to_consumption": eta,
                "negative_household_saving": bool(
                    pce_rows["negative_household_saving"].iloc[0]
                ),
                "household_factor_fallback": str(
                    pce_rows["household_factor_fallback"].iloc[0]
                ),
                "qcew_fallback_level": str(op["qcew_fallback_level"]),
                "economic_method": ECONOMIC_METHOD,
                "economic_method_version": ECONOMIC_METHOD_VERSION,
                "operating_employee_comp_share": (
                    self.calibration.operating_employee_comp_share
                ),
                "operating_value_added_share": (
                    self.calibration.operating_value_added_share
                ),
                "operating_non_electric_share": (
                    self.calibration.operating_non_electric_share
                ),
                "electricity_spend_excluded_usd": (
                    direct_output * self.calibration.operating_electricity_share
                ),
                "electricity_excluded_from_total": True,
                "currency": "USD",
                "price_year": MODEL_PRICE_YEAR,
            }
        )
        return summary, detail

    @staticmethod
    def _summarize(
        detail: pd.DataFrame, phase: str, anchored_direct_jobs: float
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        job_label = (
            "construction_job_years" if phase == "construction" else "operating_jobs"
        )
        for effect in ("direct", "indirect", "induced"):
            selected = detail[detail["effect"].eq(effect)]
            result[f"{effect}_{job_label}"] = float(selected["jobs"].sum())
            for metric in ("earnings_usd", "output_usd", "value_added_usd"):
                result[f"{effect}_{phase}_{metric}"] = float(selected[metric].sum())
        result[f"direct_{job_label}"] = anchored_direct_jobs
        result[f"total_{job_label}"] = sum(
            result[f"{effect}_{job_label}"]
            for effect in ("direct", "indirect", "induced")
        )
        for metric in ("earnings_usd", "output_usd", "value_added_usd"):
            result[f"total_{phase}_{metric}"] = sum(
                result[f"{effect}_{phase}_{metric}"]
                for effect in ("direct", "indirect", "induced")
            )
        return result


def apply_public_bea_construction(
    annual: pd.DataFrame, model: PublicBEALQIO
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict] = []
    details: list[pd.DataFrame] = []
    templates: dict[str, tuple[dict, pd.DataFrame]] = {}
    for row in annual.itertuples(index=False):
        region = str(row.st)
        if region not in templates:
            templates[region] = model.construction_impacts(region, 1.0)
        template_summary, template_detail = templates[region]
        scale = float(row.direct_construction_job_years)
        summary = dict(template_summary)
        for key in summary:
            if key.endswith("_job_years") or key.endswith("_usd"):
                summary[key] = float(summary[key]) * scale
        detail = template_detail.copy()
        detail[["jobs", "earnings_usd", "output_usd", "value_added_usd"]] *= scale
        record = row._asdict()
        record.update(summary)
        record.update(
            {
                "multiplier_scenario": "BEA2017_QCEW2024_FAF2017",
                "job_multiplier_industry": "BEA detailed construction",
                "job_multiplier_source": ECONOMIC_METHOD,
                "job_multiplier_data_year": 2024,
                "monetary_multiplier_industry": "BEA 2017 detailed",
                "multiplier_data_year": 2017,
                "multiplier_data_vintages": "BEA 2017; QCEW/BEA regional 2024; FAF 2017",
                "impact_method": ECONOMIC_METHOD,
                "spend_profile_id": "dc_calibrated_v2_ACES2017_composition",
                "spend_scale_method": "capacity_job_anchor",
                "construction_spend_scope": "state_bea_lq_faf",
                "construction_io_calibration_status": "capacity_anchor_replaced",
                "source_currency": "USD",
                "multiplier_source_currency": "USD",
                "source_price_year": 2024,
                "fx_source_currency_per_usd": 1.0,
                "price_adjustment_factor": 1.0,
                "deflator_source": "not_required_2024_anchor",
                "fx_application": "not_applicable",
                "monetary_basis_verified": True,
                "currency": "USD",
                "price_year": 2024,
            }
        )
        records.append(record)
        detail = detail.copy()
        for key in ("impact_year", "st", "country", "scenario"):
            detail[key] = record[key]
        detail["phase"] = "construction"
        detail["economic_method"] = ECONOMIC_METHOD
        detail["economic_method_version"] = ECONOMIC_METHOD_VERSION
        detail["currency"] = "USD"
        detail["price_year"] = 2024
        details.append(detail)
    return pd.DataFrame(records), pd.concat(details, ignore_index=True)


def apply_public_bea_operations(
    annual: pd.DataFrame, model: PublicBEALQIO
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[dict] = []
    details: list[pd.DataFrame] = []
    electricity: list[pd.DataFrame] = []
    templates: dict[str, tuple[dict, pd.DataFrame]] = {}
    for row in annual.itertuples(index=False):
        region = str(row.st)
        if region not in templates:
            templates[region] = model.operations_impacts(region, 1.0)
        template_summary, template_detail = templates[region]
        scale = float(row.direct_operating_jobs)
        summary = dict(template_summary)
        for key in summary:
            if key == "direct_operating_compensation_per_job_2024_usd":
                continue
            if key.endswith("_jobs") or key.endswith("_usd"):
                summary[key] = float(summary[key]) * scale
        detail = template_detail.copy()
        detail[["jobs", "earnings_usd", "output_usd", "value_added_usd"]] *= scale
        record = row._asdict()
        record.update(summary)
        record.update(
            {
                "multiplier_scenario": "BEA2017_QCEW2024_FAF2017",
                "job_multiplier_industry": "QCEW 518210",
                "job_multiplier_source": ECONOMIC_METHOD,
                "job_multiplier_data_year": 2024,
                "monetary_multiplier_industry": "BEA 518200",
                "multiplier_data_year": 2017,
                "multiplier_data_vintages": "BEA 2017; QCEW/BEA regional 2024; FAF 2017",
                "impact_method": ECONOMIC_METHOD,
                "spend_profile_id": "dc_calibrated_v2_BEA2017_composition",
                "spend_scale_method": "dc_output_calibrated_non_electric_composition",
                "spend_scope": "non_electric_BEALQ",
                "source_currency": "USD",
                "multiplier_source_currency": "USD",
                "source_price_year": 2024,
                "fx_source_currency_per_usd": 1.0,
                "price_adjustment_factor": 1.0,
                "deflator_source": "not_required_2024_anchor",
                "fx_application": "not_applicable",
                "monetary_basis_verified": True,
                "currency": "USD",
                "price_year": 2024,
            }
        )
        records.append(record)
        detail = detail.copy()
        for key in ("impact_year", "st", "country", "scenario"):
            detail[key] = record[key]
        detail["phase"] = "operations"
        detail["economic_method"] = ECONOMIC_METHOD
        detail["economic_method_version"] = ECONOMIC_METHOD_VERSION
        detail["currency"] = "USD"
        detail["price_year"] = 2024
        details.append(detail)
        electricity.append(
            pd.DataFrame(
                {
                    "impact_year": [record["impact_year"]],
                    "st": [record["st"]],
                    "country": [record["country"]],
                    "scenario": [record["scenario"]],
                    "phase": ["electricity_excluded"],
                    "classification": ["excluded_from_campus_total"],
                    "economic_method": [ECONOMIC_METHOD],
                    "economic_method_version": [ECONOMIC_METHOD_VERSION],
                    "producer_demand_usd": [0.0],
                    "currency": ["USD"],
                    "price_year": [2024],
                }
            )
        )
    return (
        pd.DataFrame(records),
        pd.concat(details, ignore_index=True),
        pd.concat(electricity, ignore_index=True),
    )
