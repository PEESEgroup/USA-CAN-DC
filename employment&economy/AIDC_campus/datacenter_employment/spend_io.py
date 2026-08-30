
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .assumptions import DEFAULT_CAN_GEO_MULT_PATH, DEFAULT_US_GEO_MULT_PATH
from .canada_inputs import CanadaRegionalData, worst_quality
from .calibration import ECONOMIC_METHOD_VERSION, DataCenterCalibration

STANDARD_INDUSTRIES = (
    "agriculture",
    "mining",
    "construction",
    "manufacturing",
    "fabricated_metals",
    "machinery",
    "electrical_equipment",
    "tcpu",
    "wholesale_trade",
    "retail_trade",
    "fire",
    "misc_services",
    "professional_services",
    "government",
)
EFFECTS = ("direct", "indirect", "induced")
METRICS = ("jobs", "earnings", "output", "value_added")

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_SPEND_PROFILE_PATH = MODULE_DIR / "config" / "spend_profiles.csv"
DEFAULT_CONSTRUCTION_LOCALIZATION_PATH = (
    MODULE_DIR / "config" / "construction_localization.csv"
)
DEFAULT_MARGIN_PATH = (
    Path(__file__).resolve().parents[2]
    / "shared_data"
    / "jedi_purchaser_to_producer_margins.csv"
)
DEFAULT_CAN_OPS_PROFILE_AUDIT_PATH = (
    Path(__file__).resolve().parents[2]
    / "shared_data"
    / "canada_datacenter_operating_input_shares.csv"
)
DEFAULT_CAN_CONSTRUCTION_RETENTION_PATH = (
    Path(__file__).resolve().parents[2]
    / "shared_data"
    / "canada_construction_us_border_retention_recalibration.csv"
)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid boolean value in spend profile: {value!r}")


@dataclass(frozen=True)
class SpendProfile:

    data: pd.DataFrame

    @classmethod
    def load(cls, path: Path = DEFAULT_SPEND_PROFILE_PATH) -> "SpendProfile":
        data = pd.read_csv(path)
        required = {
            "profile_id",
            "country",
            "phase",
            "scenario",
            "component",
            "jedi_industry",
            "cost_share",
            "local_share",
            "include_in_campus_total",
            "electricity_flag",
            "source_year",
            "quality_flag",
            "source",
        }
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"Spend profile missing columns: {sorted(missing)}")
        data = data.copy()
        for col in ("include_in_campus_total", "electricity_flag"):
            data[col] = data[col].map(_as_bool)
        for col in ("cost_share", "local_share"):
            data[col] = pd.to_numeric(data[col], errors="raise")
        if not data["local_share"].between(0.0, 1.0).all():
            raise ValueError("Spend-profile local_share must be within [0, 1]")
        unknown = sorted(set(data["jedi_industry"]) - set(STANDARD_INDUSTRIES))
        if unknown:
            raise ValueError(
                f"Spend profile contains unmapped JEDI industries: {unknown}"
            )
        sums = data.groupby(["profile_id", "country", "phase", "scenario"])[
            "cost_share"
        ].sum()
        bad = sums[~np.isclose(sums, 1.0, rtol=0.0, atol=1e-9)]
        if not bad.empty:
            raise ValueError(f"Spend-profile cost_share groups must sum to 1:\n{bad}")
        for (profile_id, country, scenario), group in data.groupby(
            ["profile_id", "country", "scenario"]
        ):
            ops = group[group["phase"].eq("operations")]
            if not ops.empty:
                electricity = ops[ops["electricity_flag"]]
                if (
                    len(electricity) != 1
                    or electricity.iloc[0]["include_in_campus_total"]
                ):
                    raise ValueError(
                        f"{profile_id}/{country}/{scenario}: operations must have exactly one "
                        "electricity row excluded from campus totals"
                    )
            construction = group[group["phase"].eq("construction")]
            land = construction[construction["component"].eq("land")]
            if not land.empty and (
                land["include_in_campus_total"].any() or (land["local_share"] > 0).any()
            ):
                raise ValueError(
                    f"{profile_id}/{country}: land must not create local demand"
                )
        return cls(data)

    def rows(
        self,
        country: str,
        phase: str,
        scenario: str,
        profile_id: str | None = None,
        region: str | None = None,
    ) -> pd.DataFrame:
        data = self.data
        if profile_id is None:
            ids = data["profile_id"].drop_duplicates().tolist()
            if len(ids) != 1:
                raise ValueError(
                    "Multiple spend profiles available; profile_id is required"
                )
            profile_id = str(ids[0])
        selected = data[
            data["profile_id"].eq(profile_id)
            & data["country"].eq(country)
            & data["phase"].eq(phase)
            & data["scenario"].isin([scenario, "all"])
        ].copy()
        if region is not None and "region" in selected.columns:
            regional = selected[
                selected["region"].astype(str).isin([str(region), "all"])
            ]
            if not regional.empty:
                selected = regional
        exact = selected[selected["scenario"].eq(scenario)]
        if not exact.empty:
            selected = exact
        else:
            selected = selected[selected["scenario"].eq("all")]
        if selected.empty:
            raise KeyError(
                f"No spend profile for {profile_id}/{country}/{phase}/{scenario}"
            )
        return selected.reset_index(drop=True)


@dataclass(frozen=True)
class ConstructionLocalization:

    data: pd.DataFrame

    @classmethod
    def load(
        cls, path: Path = DEFAULT_CONSTRUCTION_LOCALIZATION_PATH
    ) -> "ConstructionLocalization":
        data = pd.read_csv(path, keep_default_na=False)
        required = {
            "rule_id",
            "rule_scope",
            "country",
            "region",
            "component",
            "producer_industry",
            "retention_share",
            "treatment",
            "source_year",
            "quality_flag",
            "source",
        }
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(
                f"Construction-localization rules missing columns: {sorted(missing)}"
            )
        data = data.copy()
        data["retention_share"] = pd.to_numeric(data["retention_share"], errors="raise")
        if (
            not np.isfinite(data["retention_share"]).all()
            or not data["retention_share"].between(0.0, 1.0).all()
        ):
            raise ValueError(
                "Construction retention_share must be finite and within [0, 1]"
            )
        allowed_scopes = {"main_result", "audit_only"}
        unexpected_scopes = set(data["rule_scope"]) - allowed_scopes
        if unexpected_scopes:
            raise ValueError(
                f"Unexpected construction-localization rule_scope: {sorted(unexpected_scopes)}"
            )
        keys = [
            "rule_scope",
            "country",
            "region",
            "component",
            "producer_industry",
        ]
        if data.duplicated(keys).any():
            duplicates = data.loc[data.duplicated(keys, keep=False), keys]
            raise ValueError(
                f"Duplicate construction-localization rules:\n{duplicates}"
            )
        if not data["country"].eq("USA").all():
            raise ValueError(
                "Construction-localization rules currently support USA only"
            )
        state_specific_main = data[
            data["rule_scope"].eq("main_result") & ~data["region"].eq("all")
        ]
        if not state_specific_main.empty:
            raise ValueError(
                "US main-result construction localization must be uniform: "
                "state-specific rows are audit_only"
            )
        if (data["rule_scope"].eq("audit_only") & data["region"].eq("all")).any():
            raise ValueError(
                "Audit-only construction localization must name a specific state"
            )
        fallback = data[data["region"].eq("all") & data["rule_scope"].eq("main_result")]
        expected = {
            ("land", "fire"),
            ("building_and_site", "construction"),
            ("computer_equipment", "manufacturing"),
            ("computer_equipment", "wholesale_trade"),
            ("computer_equipment", "tcpu"),
            ("cooling_and_electrical", "electrical_equipment"),
            ("cooling_and_electrical", "wholesale_trade"),
            ("cooling_and_electrical", "tcpu"),
            ("other_capital", "professional_services"),
        }
        available = set(zip(fallback["component"], fallback["producer_industry"]))
        if available != expected:
            raise ValueError(
                "USA fallback localization rules must exactly cover the expanded "
                f"construction profile; missing={sorted(expected - available)}, "
                f"unexpected={sorted(available - expected)}"
            )
        return cls(data)

    def resolve(
        self, region: str, expanded: pd.DataFrame, rule_scope: str = "main_result"
    ) -> pd.DataFrame:
        if rule_scope not in {"main_result", "audit_only"}:
            raise ValueError(f"Unknown construction localization scope: {rule_scope}")
        rows = expanded.copy()
        keys = ["country", "component", "producer_industry"]
        columns = [
            *keys,
            "rule_id",
            "retention_share",
            "treatment",
            "source_year",
            "quality_flag",
            "source",
        ]
        fallback = self.data[
            self.data["region"].eq("all") & self.data["rule_scope"].eq("main_result")
        ][columns].rename(
            columns={
                "rule_id": "fallback_rule_id",
                "retention_share": "fallback_retention_share",
                "treatment": "fallback_treatment",
                "source_year": "fallback_source_year",
                "quality_flag": "fallback_quality_flag",
                "source": "fallback_source",
            }
        )
        exact = self.data[
            self.data["region"].eq(str(region)) & self.data["rule_scope"].eq(rule_scope)
        ][columns].rename(
            columns={
                "rule_id": "exact_rule_id",
                "retention_share": "exact_retention_share",
                "treatment": "exact_treatment",
                "source_year": "exact_source_year",
                "quality_flag": "exact_quality_flag",
                "source": "exact_source",
            }
        )
        rows = rows.merge(fallback, on=keys, how="left", validate="many_to_one")
        rows = rows.merge(exact, on=keys, how="left", validate="many_to_one")
        if rows["fallback_retention_share"].isna().any():
            missing = rows.loc[
                rows["fallback_retention_share"].isna(),
                ["component", "producer_industry"],
            ].drop_duplicates()
            raise KeyError(
                f"Missing USA construction localization fallback:\n{missing}"
            )
        for target in (
            "rule_id",
            "retention_share",
            "treatment",
            "source_year",
            "quality_flag",
            "source",
        ):
            exact_values = rows[f"exact_{target}"]
            use_exact = exact_values.notna()
            if exact_values.dtype == object:
                use_exact &= exact_values.ne("")
            rows[f"resolved_{target}"] = exact_values.where(
                use_exact, rows[f"fallback_{target}"]
            )
        drop = [
            column
            for column in rows.columns
            if column.startswith("exact_") or column.startswith("fallback_")
        ]
        rows = rows.drop(columns=drop).rename(
            columns={
                "resolved_rule_id": "construction_spend_rule_id",
                "resolved_retention_share": "producer_retention_share",
                "resolved_treatment": "localization_treatment",
                "resolved_source_year": "localization_source_year",
                "resolved_quality_flag": "localization_quality_flag",
                "resolved_source": "localization_source",
            }
        )
        return rows

    def policy_id(self, region: str, rule_scope: str = "main_result") -> str:
        if rule_scope not in {"main_result", "audit_only"}:
            raise ValueError(f"Unknown construction localization scope: {rule_scope}")
        exact = self.data.loc[
            self.data["region"].eq(str(region))
            & self.data["rule_scope"].eq(rule_scope),
            "rule_id",
        ].unique()
        selected = (
            exact
            if len(exact)
            else self.data.loc[
                self.data["region"].eq("all")
                & self.data["rule_scope"].eq("main_result"),
                "rule_id",
            ].unique()
        )
        if len(selected) != 1:
            raise ValueError(f"No unique construction localization policy for {region}")
        return str(selected[0])


class IOSystem:

    def __init__(self, multipliers: pd.DataFrame, margins: pd.DataFrame):
        self.data = multipliers.copy()
        self.margins = margins.copy()
        standard = self.data[self.data["industry"].isin(STANDARD_INDUSTRIES)]
        duplicates = standard.duplicated(["region", "industry"])
        if duplicates.any():
            raise ValueError("Duplicate region/industry rows in I-O multiplier data")
        coverage = standard.groupby("region")["industry"].agg(set)
        incomplete = {
            region: sorted(set(STANDARD_INDUSTRIES) - industries)
            for region, industries in coverage.items()
            if industries != set(STANDARD_INDUSTRIES)
        }
        if incomplete:
            raise ValueError(
                f"Incomplete 14-industry multiplier coverage: {incomplete}"
            )
        pce_sums = standard.groupby("region")["pce_share"].sum(min_count=1)
        if not np.allclose(pce_sums, 1.0, rtol=0.0, atol=1e-6):
            raise ValueError(f"Regional PCE shares must sum to 1:\n{pce_sums}")
        margin_sums = self.margins.groupby(["country", "purchaser_industry"])[
            "margin_share"
        ].sum()
        if not np.allclose(margin_sums, 1.0, rtol=0.0, atol=1e-5):
            raise ValueError(f"Purchaser margin rows must sum to 1:\n{margin_sums}")
        self._rows = {
            (str(row.region), str(row.industry)): row
            for row in self.data.itertuples(index=False)
        }

    @classmethod
    def load(
        cls,
        us_path: Path = DEFAULT_US_GEO_MULT_PATH,
        can_path: Path = DEFAULT_CAN_GEO_MULT_PATH,
        margin_path: Path = DEFAULT_MARGIN_PATH,
        include_canada_proxy_margins: bool = False,
    ) -> "IOSystem":
        frames = [pd.read_csv(us_path)]
        if Path(can_path).exists():
            frames.append(pd.read_csv(can_path))
        margins = pd.read_csv(margin_path)
        if not include_canada_proxy_margins:
            margins = margins[~margins["country"].eq("CAN")].copy()
        return cls(pd.concat(frames, ignore_index=True), margins)

    def row(self, region: str, industry: str):
        try:
            return self._rows[(str(region), str(industry))]
        except KeyError as exc:
            raise KeyError(f"No I-O multiplier for {region}/{industry}") from exc

    def target_factor(self, region: str, industry: str) -> float:
        value = float(self.row(region, industry).target_2022_factor)
        return value if np.isfinite(value) and value > 0 else 1.0

    def pce_vector(self, region: str) -> pd.DataFrame:
        data = self.data[
            self.data["region"].astype(str).eq(str(region))
            & self.data["industry"].isin(STANDARD_INDUSTRIES)
        ][["industry", "pce_share"]].copy()
        return data.rename(columns={"industry": "producer_industry"}).reset_index(
            drop=True
        )

    def apply_margins(self, demand: pd.DataFrame) -> pd.DataFrame:
        if demand.empty:
            return demand.assign(
                producer_industry=pd.Series(dtype=str), producer_demand_usd=0.0
            )
        merged = demand.merge(
            self.margins,
            left_on=["country", "jedi_industry"],
            right_on=["country", "purchaser_industry"],
            how="left",
            validate="many_to_many",
        )
        if merged["margin_share"].isna().any():
            bad = merged.loc[
                merged["margin_share"].isna(), ["country", "jedi_industry"]
            ]
            raise KeyError(f"Missing purchaser margins:\n{bad.drop_duplicates()}")
        merged["producer_demand_usd"] = (
            merged["local_purchaser_spend_usd"] * merged["margin_share"]
        )
        return merged

    def apply(self, region: str, producer_demand: pd.DataFrame) -> pd.DataFrame:
        records: list[dict] = []
        for demand in producer_demand.itertuples(index=False):
            industry = str(demand.producer_industry)
            row = self.row(region, industry)
            factor = self.target_factor(region, industry)
            demand_usd = float(demand.producer_demand_usd)
            demand_million = demand_usd / factor / 1_000_000.0
            record = demand._asdict()
            record.update(
                {
                    "region": str(region),
                    "multiplier_data_year": int(row.multiplier_data_year),
                    "target_2022_factor": factor,
                    "multiplier_source": str(row.source),
                }
            )
            for effect in EFFECTS:
                record[f"{effect}_jobs"] = demand_million * float(
                    getattr(row, f"{effect}_mult")
                )
                for metric in ("earnings", "output", "value_added"):
                    multiplier = float(getattr(row, f"{effect}_{metric}_mult"))
                    record[f"{effect}_{metric}_usd"] = (
                        demand_million * multiplier * 1_000_000.0 * factor
                    )
            records.append(record)
        return pd.DataFrame(records)

    def direct_metric_per_job(self, region: str, metric: str) -> float:
        exact = self._rows.get((str(region), "datacenter_operations"))
        if exact is not None:
            jobs = float(exact.direct_mult)
            monetary = float(getattr(exact, f"direct_{metric}_mult"))
            if jobs > 0 and np.isfinite(monetary):
                return (
                    monetary
                    * 1_000_000.0
                    * self.target_factor(region, "datacenter_operations")
                    / jobs
                )
        values = []
        for industry in ("tcpu", "professional_services"):
            row = self.row(region, industry)
            jobs = float(row.direct_mult)
            monetary = float(getattr(row, f"direct_{metric}_mult"))
            if jobs > 0 and np.isfinite(monetary):
                values.append(
                    monetary * 1_000_000.0 * self.target_factor(region, industry) / jobs
                )
        if not values:
            raise ValueError(f"No direct operations {metric}/job value for {region}")
        return float(np.mean(values))


def _quality_flag(rows: pd.DataFrame) -> str:
    flags = set(rows["quality_flag"].astype(str))
    return "proxy" if any("proxy" in flag for flag in flags) else "source"


def _sum_effect(detail: pd.DataFrame, effect: str, metric: str) -> float:
    return float(detail[f"{effect}_{metric}"].sum())


def _component_demand(
    profile_rows: pd.DataFrame,
    total_spend_usd: float,
    include_electricity: bool = False,
) -> pd.DataFrame:
    rows = profile_rows.copy()
    rows = rows[
        rows["include_in_campus_total"]
        | (include_electricity & rows["electricity_flag"])
    ]
    rows = rows[~rows["component"].eq("direct_labor")].copy()
    rows["purchaser_spend_usd"] = total_spend_usd * rows["cost_share"]
    rows["local_purchaser_spend_usd"] = (
        rows["purchaser_spend_usd"] * rows["local_share"]
    )
    return rows


def _us_construction_producer_demand(
    profile_rows: pd.DataFrame,
    io_system: IOSystem,
    localization: ConstructionLocalization,
    region: str,
    rule_scope: str = "main_result",
) -> pd.DataFrame:
    purchaser = profile_rows.copy()
    purchaser["purchaser_spend_usd"] = purchaser["cost_share"]


    purchaser["local_purchaser_spend_usd"] = purchaser["purchaser_spend_usd"]
    expanded = io_system.apply_margins(purchaser)
    expanded["pre_localization_producer_demand_usd"] = expanded["producer_demand_usd"]
    expanded = localization.resolve(region, expanded, rule_scope=rule_scope)
    expanded["profile_legacy_local_share"] = expanded["local_share"]
    expanded["local_share"] = expanded["producer_retention_share"]
    expanded["producer_demand_usd"] = (
        expanded["pre_localization_producer_demand_usd"]
        * expanded["producer_retention_share"]
    )
    expanded["local_purchaser_spend_usd"] = expanded["producer_demand_usd"]
    expanded["excluded_leakage_usd"] = (
        expanded["pre_localization_producer_demand_usd"]
        - expanded["producer_demand_usd"]
    )
    return expanded


def _scaled_detail(template: pd.DataFrame, scale: float) -> pd.DataFrame:
    result = template.copy()
    scale_columns = [
        col
        for col in result.columns
        if col
        in {
            "purchaser_spend_usd",
            "local_purchaser_spend_usd",
            "producer_demand_usd",
            "pre_localization_producer_demand_usd",
            "excluded_leakage_usd",
        }
        or col.endswith("_jobs")
        or col.endswith("_earnings_usd")
        or col.endswith("_output_usd")
        or col.endswith("_value_added_usd")
    ]
    result[scale_columns] = result[scale_columns] * scale
    return result


def _apply_construction_share_profile(
    annual: pd.DataFrame,
    io_system: IOSystem,
    profiles: SpendProfile,
    localization: ConstructionLocalization,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = annual.copy()
    key_columns = ["st", "country", "scenario"]
    templates: list[pd.DataFrame] = []
    coefficients: list[dict] = []
    for key in result[key_columns].drop_duplicates().itertuples(index=False):
        profile = profiles.rows(key.country, "construction", key.scenario)
        if key.country == "USA":
            producer_template = _us_construction_producer_demand(
                profile, io_system, localization, key.st, rule_scope="main_result"
            )
            audit_producer_template = _us_construction_producer_demand(
                profile, io_system, localization, key.st, rule_scope="audit_only"
            )
            construction_scope = "us_uniform_mn_conservative_transport_only"
            calibration_status = "uncalibrated_generic_io"
            policy_id = localization.policy_id(key.st, rule_scope="main_result")
            audit_policy_id = localization.policy_id(key.st, rule_scope="audit_only")
        else:
            purchaser_template = _component_demand(profile, 1.0)
            producer_template = io_system.apply_margins(purchaser_template)
            producer_template["pre_localization_producer_demand_usd"] = (
                producer_template["producer_demand_usd"]
            )
            producer_template["producer_retention_share"] = 1.0
            producer_template["excluded_leakage_usd"] = 0.0
            producer_template["construction_spend_rule_id"] = (
                "canada_us_proxy_sensitivity_legacy"
            )
            producer_template["localization_treatment"] = "legacy_pre_margin_share"
            producer_template["localization_source"] = str(profile.iloc[0]["source"])
            construction_scope = "canada_us_proxy_sensitivity"
            calibration_status = "proxy_sensitivity"
            policy_id = "canada_us_proxy_sensitivity_legacy"
            audit_policy_id = policy_id
            audit_producer_template = producer_template
        anchor_share = float(
            producer_template.loc[
                producer_template["producer_industry"].eq("construction"),
                "producer_demand_usd",
            ].sum()
        )
        multiplier_row = io_system.row(key.st, "construction")
        direct_multiplier = float(multiplier_row.direct_mult)
        if direct_multiplier <= 0 or anchor_share <= 0:
            raise ValueError(f"Cannot anchor construction spending for {key.st}")
        spend_per_direct_job = (
            1_000_000.0
            * io_system.target_factor(key.st, "construction")
            / direct_multiplier
            / anchor_share
        )
        detail = io_system.apply(key.st, producer_template)
        detail["st"] = key.st
        detail["country"] = key.country
        detail["scenario"] = key.scenario
        detail["phase"] = "construction"
        detail["classification"] = np.where(
            detail["producer_industry"].eq("construction"),
            "anchored_direct",
            "indirect",
        )
        templates.append(detail)
        anchor_io_jobs = float(
            detail.loc[
                detail["producer_industry"].eq("construction"), "direct_jobs"
            ].sum()
        )
        non_anchor = detail[~detail["producer_industry"].eq("construction")]
        equipment_direct_jobs = _sum_effect(
            detail[
                detail["component"].isin(
                    ["computer_equipment", "cooling_and_electrical"]
                )
            ],
            "direct",
            "jobs",
        )
        audit_detail = io_system.apply(key.st, audit_producer_template)
        audit_non_anchor = audit_detail[
            ~audit_detail["producer_industry"].eq("construction")
        ]
        audit_equipment_direct_jobs = _sum_effect(
            audit_detail[
                audit_detail["component"].isin(
                    ["computer_equipment", "cooling_and_electrical"]
                )
            ],
            "direct",
            "jobs",
        )
        audit_direct_jobs = anchor_io_jobs
        audit_indirect_jobs = _sum_effect(audit_non_anchor, "direct", "jobs") + (
            _sum_effect(audit_detail, "indirect", "jobs")
        )
        if key.country == "USA" and key.st == "VA":
            audit_direct_jobs += audit_equipment_direct_jobs
            audit_indirect_jobs -= audit_equipment_direct_jobs
        audit_induced_jobs = _sum_effect(audit_detail, "induced", "jobs")
        values = {
            "st": key.st,
            "country": key.country,
            "scenario": key.scenario,
            "spend_per_direct_job": spend_per_direct_job,
            "indirect_construction_job_years": _sum_effect(non_anchor, "direct", "jobs")
            + _sum_effect(detail, "indirect", "jobs"),
            "induced_construction_job_years": _sum_effect(detail, "induced", "jobs"),
            "local_final_demand_usd": float(detail["producer_demand_usd"].sum()),
            "io_implied_direct_jobs": _sum_effect(detail, "direct", "jobs"),
            "anchor_io_jobs_per_dollar": anchor_io_jobs,
            "equipment_direct_construction_job_years": equipment_direct_jobs,
            "audit_modeled_direct_construction_job_years": audit_direct_jobs,
            "audit_modeled_indirect_construction_job_years": audit_indirect_jobs,
            "audit_modeled_induced_construction_job_years": audit_induced_jobs,
            "construction_audit_rule_id": audit_policy_id,
            "impact_method": "spend_io",
            "spend_profile_id": str(profile.iloc[0]["profile_id"]),
            "spend_scale_method": "direct_construction_jobs_anchor",
            "electricity_excluded_from_total": True,
            "spend_profile_quality_flag": _quality_flag(profile),
            "construction_spend_rule_id": policy_id,
            "construction_spend_scope": construction_scope,
            "construction_io_calibration_status": calibration_status,
        }
        for metric in ("earnings_usd", "output_usd", "value_added_usd"):
            values[f"direct_construction_{metric}"] = _sum_effect(
                detail[detail["producer_industry"].eq("construction")], "direct", metric
            )
            values[f"indirect_construction_{metric}"] = _sum_effect(
                non_anchor, "direct", metric
            ) + _sum_effect(detail, "indirect", metric)
            values[f"induced_construction_{metric}"] = _sum_effect(
                detail, "induced", metric
            )
        coefficients.append(values)
    coefficient_frame = pd.DataFrame(coefficients)
    replace_columns = [
        col
        for col in coefficient_frame.columns
        if col in result.columns and col not in key_columns
    ]
    result = result.drop(columns=replace_columns).merge(
        coefficient_frame, on=key_columns, how="left"
    )
    result["implied_total_spend_usd"] = result[
        "direct_construction_job_years"
    ] * result.pop("spend_per_direct_job")
    scale_columns = [
        "indirect_construction_job_years",
        "induced_construction_job_years",
        "local_final_demand_usd",
        "io_implied_direct_jobs",
        "equipment_direct_construction_job_years",
        "audit_modeled_direct_construction_job_years",
        "audit_modeled_indirect_construction_job_years",
        "audit_modeled_induced_construction_job_years",
        *[
            f"{effect}_construction_{metric}"
            for metric in ("earnings_usd", "output_usd", "value_added_usd")
            for effect in ("direct", "indirect", "induced")
        ],
    ]
    result[scale_columns] = result[scale_columns].multiply(
        result["implied_total_spend_usd"], axis=0
    )
    result["direct_reconciliation_ratio"] = np.where(
        result["direct_construction_job_years"].ne(0),
        result.pop("anchor_io_jobs_per_dollar")
        * result["implied_total_spend_usd"]
        / result["direct_construction_job_years"],
        1.0,
    )
    result["audit_comparable_direct_construction_job_years"] = result[
        "audit_modeled_direct_construction_job_years"
    ]
    result["audit_modeled_total_construction_job_years"] = (
        result["audit_modeled_direct_construction_job_years"]
        + result["audit_modeled_indirect_construction_job_years"]
        + result["audit_modeled_induced_construction_job_years"]
    )
    for metric in ("jobs", "earnings_usd", "output_usd", "value_added_usd"):
        suffix = "job_years" if metric == "jobs" else metric
        result[f"total_construction_{suffix}"] = (
            result[f"direct_construction_{suffix}"]
            + result[f"indirect_construction_{suffix}"]
            + result[f"induced_construction_{suffix}"]
        )
    template_data = pd.concat(templates, ignore_index=True)
    detail = result[["impact_year", *key_columns, "implied_total_spend_usd"]].merge(
        template_data, on=key_columns, how="left"
    )
    detail_scale_columns = [
        col
        for col in detail.columns
        if col
        in {
            "purchaser_spend_usd",
            "local_purchaser_spend_usd",
            "producer_demand_usd",
            "pre_localization_producer_demand_usd",
            "excluded_leakage_usd",
        }
        or col.endswith("_jobs")
        or col.endswith("_earnings_usd")
        or col.endswith("_output_usd")
        or col.endswith("_value_added_usd")
    ]
    detail[detail_scale_columns] = detail[detail_scale_columns].multiply(
        detail["implied_total_spend_usd"], axis=0
    )
    return result, detail


def _apply_operations_share_profile(
    annual: pd.DataFrame,
    io_system: IOSystem,
    profiles: SpendProfile,
    payroll_spendable_share: float,
    load_st: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result = annual.copy()
    key_columns = ["st", "country", "scenario"]
    vendor_templates: list[pd.DataFrame] = []
    electricity_templates: list[pd.DataFrame] = []
    coefficients: list[dict] = []
    for key in result[key_columns].drop_duplicates().itertuples(index=False):
        profile = profiles.rows(key.country, "operations", key.scenario)
        labor_share = float(
            profile.loc[profile["component"].eq("direct_labor"), "cost_share"].sum()
        )
        payroll_per_job = io_system.direct_metric_per_job(key.st, "earnings")
        spend_per_direct_job = payroll_per_job / labor_share
        vendor_purchaser = _component_demand(profile, 1.0)
        vendor_producer = io_system.apply_margins(vendor_purchaser)
        vendor = io_system.apply(key.st, vendor_producer)
        vendor["st"] = key.st
        vendor["country"] = key.country
        vendor["scenario"] = key.scenario
        vendor["phase"] = "operations_vendor"
        vendor["classification"] = "indirect"
        pce = io_system.pce_vector(key.st)
        pce["component"] = "direct_payroll_pce"
        pce["country"] = key.country
        pce["purchaser_spend_usd"] = (
            labor_share * payroll_spendable_share * pce["pce_share"]
        )
        pce["local_purchaser_spend_usd"] = pce["purchaser_spend_usd"]
        pce["producer_demand_usd"] = pce["purchaser_spend_usd"]
        pce_detail = io_system.apply(key.st, pce)
        pce_detail["st"] = key.st
        pce_detail["country"] = key.country
        pce_detail["scenario"] = key.scenario
        pce_detail["phase"] = "operations_payroll_pce"
        pce_detail["classification"] = "payroll_induced"
        vendor_templates.extend([vendor, pce_detail])
        electricity_profile = profile[profile["electricity_flag"]].copy()
        electricity_purchaser = _component_demand(
            electricity_profile, 1.0, include_electricity=True
        )
        electricity_producer = io_system.apply_margins(electricity_purchaser)
        electricity = io_system.apply(key.st, electricity_producer)
        electricity["st"] = key.st
        electricity["country"] = key.country
        electricity["scenario"] = key.scenario
        electricity["phase"] = "electricity_excluded"
        electricity["classification"] = "excluded_from_campus_total"
        electricity_templates.append(electricity)
        values = {
            "st": key.st,
            "country": key.country,
            "scenario": key.scenario,
            "spend_per_direct_job": spend_per_direct_job,
            "indirect_operating_jobs": _sum_effect(vendor, "direct", "jobs")
            + _sum_effect(vendor, "indirect", "jobs"),
            "induced_operating_jobs": _sum_effect(vendor, "induced", "jobs")
            + _sum_effect(pce_detail, "direct", "jobs")
            + _sum_effect(pce_detail, "indirect", "jobs"),
            "local_final_demand_usd": float(vendor["producer_demand_usd"].sum())
            + float(pce_detail["producer_demand_usd"].sum()),
            "io_implied_direct_jobs": _sum_effect(vendor, "direct", "jobs"),
            "direct_reconciliation_ratio": 1.0,
            "impact_method": "spend_io",
            "spend_profile_id": str(profile.iloc[0]["profile_id"]),
            "spend_scale_method": "direct_payroll_anchor",
            "electricity_excluded_from_total": True,
            "spend_profile_quality_flag": _quality_flag(profile),
        }
        for metric in ("earnings_usd", "output_usd", "value_added_usd"):
            base_metric = metric.removesuffix("_usd")
            values[f"direct_{metric}_per_job"] = io_system.direct_metric_per_job(
                key.st, base_metric
            )
            values[f"indirect_operating_{metric}"] = _sum_effect(
                vendor, "direct", metric
            ) + _sum_effect(vendor, "indirect", metric)
            values[f"induced_operating_{metric}"] = (
                _sum_effect(vendor, "induced", metric)
                + _sum_effect(pce_detail, "direct", metric)
                + _sum_effect(pce_detail, "indirect", metric)
            )
        coefficients.append(values)
    coefficient_frame = pd.DataFrame(coefficients)
    replace_columns = [
        col
        for col in coefficient_frame.columns
        if col in result.columns and col not in key_columns
    ]
    result = result.drop(columns=replace_columns).merge(
        coefficient_frame, on=key_columns, how="left"
    )
    result["implied_total_spend_usd"] = result["direct_operating_jobs"] * result.pop(
        "spend_per_direct_job"
    )
    scale_columns = [
        "indirect_operating_jobs",
        "induced_operating_jobs",
        "local_final_demand_usd",
        "io_implied_direct_jobs",
        *[
            f"{effect}_operating_{metric}"
            for metric in ("earnings_usd", "output_usd", "value_added_usd")
            for effect in ("indirect", "induced")
        ],
    ]
    result[scale_columns] = result[scale_columns].multiply(
        result["implied_total_spend_usd"], axis=0
    )
    for metric in ("earnings_usd", "output_usd", "value_added_usd"):
        result[f"direct_operating_{metric}"] = result[
            "direct_operating_jobs"
        ] * result.pop(f"direct_{metric}_per_job")
    for metric in ("jobs", "earnings_usd", "output_usd", "value_added_usd"):
        result[f"total_operating_{metric}"] = (
            result[f"direct_operating_{metric}"]
            + result[f"indirect_operating_{metric}"]
            + result[f"induced_operating_{metric}"]
        )
    annual_scale = result[["impact_year", *key_columns, "implied_total_spend_usd"]]
    vendor_detail = annual_scale.merge(
        pd.concat(vendor_templates, ignore_index=True), on=key_columns, how="left"
    )
    electricity_detail = annual_scale.merge(
        pd.concat(electricity_templates, ignore_index=True), on=key_columns, how="left"
    )
    for detail in (vendor_detail, electricity_detail):
        detail_scale_columns = [
            col
            for col in detail.columns
            if col
            in {
                "purchaser_spend_usd",
                "local_purchaser_spend_usd",
                "producer_demand_usd",
            }
            or col.endswith("_jobs")
            or col.endswith("_earnings_usd")
            or col.endswith("_output_usd")
            or col.endswith("_value_added_usd")
        ]
        detail[detail_scale_columns] = detail[detail_scale_columns].multiply(
            detail["implied_total_spend_usd"], axis=0
        )
    if load_st is not None and not load_st.empty:
        loads = (
            load_st.rename(columns={"t": "impact_year", "Value": "electricity_load_mw"})
            .groupby(["st", "impact_year"], as_index=False)["electricity_load_mw"]
            .sum()
        )
        electricity_detail = electricity_detail.merge(
            loads, on=["st", "impact_year"], how="left"
        )
    else:
        electricity_detail["electricity_load_mw"] = np.nan
    electricity_detail["implied_electricity_price_usd_per_mwh"] = np.where(
        electricity_detail["electricity_load_mw"].gt(0),
        electricity_detail["producer_demand_usd"]
        / (electricity_detail["electricity_load_mw"] * 8760.0),
        np.nan,
    )
    return result, vendor_detail, electricity_detail


def _canada_producer_template(
    rows: pd.DataFrame, io_system: IOSystem, region: str
) -> pd.DataFrame:
    usable = rows[
        rows["include_in_campus_total"]
        & rows["producer_industry"].isin(STANDARD_INDUSTRIES)
        & rows["amount_per_anchor"].gt(0)
    ].copy()
    usable["country"] = "CAN"
    usable["jedi_industry"] = usable["producer_industry"]
    usable["purchaser_spend_usd"] = (
        usable["purchaser_amount_per_anchor"] * usable["margin_share"]
    )
    usable["local_purchaser_spend_usd"] = (
        usable["purchaser_spend_usd"] * usable["local_share"]
    )
    usable["producer_demand_usd"] = usable["amount_per_anchor"]
    usable["pre_localization_producer_demand_usd"] = usable["producer_demand_usd"]
    usable["producer_retention_share"] = 1.0
    usable["excluded_leakage_usd"] = 0.0
    usable["construction_spend_rule_id"] = "statcan_regional_v2"
    usable["localization_treatment"] = "precomputed_statcan_local_demand"
    usable["localization_source"] = usable["source"]
    return io_system.apply(region, usable)


def _canada_purchaser_coefficient(rows: pd.DataFrame) -> float:
    return float(
        rows.groupby(["component", "statcan_product"], dropna=False)[
            "purchaser_amount_per_anchor"
        ]
        .first()
        .sum()
    )


def _zero_equipment_local_share(
    rows: pd.DataFrame, calibration: DataCenterCalibration
) -> pd.DataFrame:
    rows = rows.copy()
    equipment = ~rows["component"].eq("building_and_site")
    rows.loc[equipment, "local_share"] = calibration.equipment_indirect_local_share
    rows.loc[equipment, "amount_per_anchor"] = (
        rows.loc[equipment, "purchaser_amount_per_anchor"]
        * rows.loc[equipment, "margin_share"]
        * calibration.equipment_indirect_local_share
    )
    return rows


def _recalibrate_canada_construction_retention(io_system: IOSystem) -> None:
    overrides = pd.read_csv(DEFAULT_CAN_CONSTRUCTION_RETENTION_PATH).set_index("region")
    columns = [
        "indirect_mult",
        "indirect_earnings_mult",
        "indirect_output_mult",
        "indirect_value_added_mult",
    ]
    mask = io_system.data["region"].isin(overrides.index) & io_system.data["industry"].eq(
        "construction"
    )
    for column in columns:
        io_system.data.loc[mask, column] = io_system.data.loc[mask, "region"].map(
            overrides[column]
        )
    io_system._rows = {
        (str(row.region), str(row.industry)): row
        for row in io_system.data.itertuples(index=False)
    }


def _apply_canada_construction_spend_io(
    annual: pd.DataFrame,
    io_system: IOSystem,
    canada_data: CanadaRegionalData,
    calibration: DataCenterCalibration,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _recalibrate_canada_construction_retention(io_system)
    result_records: list[dict] = []
    detail_records: list[pd.DataFrame] = []
    for annual_row in annual.itertuples(index=False):
        region = str(annual_row.st)
        rows = _zero_equipment_local_share(
            canada_data.rows(region, "construction"), calibration
        )
        template = _canada_producer_template(rows, io_system, region)
        template["st"] = region
        template["country"] = "CAN"
        template["scenario"] = str(annual_row.scenario)
        template["phase"] = "construction"
        template["classification"] = np.where(
            template["producer_industry"].eq("construction"),
            "anchored_direct",
            "indirect",
        )
        direct_multiplier = float(io_system.row(region, "construction").direct_mult)
        anchor_per_job = (
            1_000_000.0
            * io_system.target_factor(region, "construction")
            / direct_multiplier
        )
        direct_jobs = float(annual_row.direct_construction_job_years)
        anchor_demand = direct_jobs * anchor_per_job
        detail = _scaled_detail(template, anchor_demand)
        gross_capex = anchor_demand / calibration.capex_share_construction
        equipment_capex = gross_capex * calibration.capex_share_equipment
        land_capex = gross_capex * calibration.capex_share_land
        other_capex = gross_capex * calibration.capex_share_other
        total_purchaser = anchor_demand + equipment_capex
        detail["impact_year"] = int(annual_row.impact_year)
        detail["implied_total_spend_usd"] = total_purchaser
        detail_records.append(detail)
        construction_detail = detail[detail["producer_industry"].eq("construction")]
        non_anchor = detail[~detail["producer_industry"].eq("construction")]
        record = annual_row._asdict()
        record.update(
            {
                "indirect_construction_job_years": _sum_effect(
                    non_anchor, "direct", "jobs"
                )
                + _sum_effect(detail, "indirect", "jobs"),
                "induced_construction_job_years": _sum_effect(
                    detail, "induced", "jobs"
                ),
                "implied_total_spend_usd": total_purchaser,
                "gross_campus_capex_usd": gross_capex,
                "gross_land_capex_usd": land_capex,
                "gross_construction_capex_usd": anchor_demand,
                "gross_equipment_capex_usd": equipment_capex,
                "gross_other_capex_usd": other_capex,
                "gross_equipment_purchaser_spend_usd": equipment_capex,
                "equipment_local_final_demand_usd": float(
                    non_anchor["producer_demand_usd"].sum()
                ),
                "equipment_to_construction_ratio": (
                    calibration.equipment_to_construction_ratio
                ),
                "land_excluded_from_io": True,
                "other_capex_excluded_from_io": True,
                "local_final_demand_usd": float(detail["producer_demand_usd"].sum()),
                "io_implied_direct_jobs": _sum_effect(detail, "direct", "jobs"),
                "direct_reconciliation_ratio": (
                    _sum_effect(construction_detail, "direct", "jobs") / direct_jobs
                    if direct_jobs
                    else 1.0
                ),
                "impact_method": "spend_io",
                "economic_method_version": ECONOMIC_METHOD_VERSION,
                "spend_profile_id": "statcan_regional_v2",
                "spend_scale_method": "direct_construction_jobs_anchor",
                "electricity_excluded_from_total": True,
                "spend_profile_quality_flag": canada_data.quality(
                    region, "construction"
                ),
                "spend_scope": "canada_tangible_capex_ex_land_ip",
                "construction_spend_rule_id": "statcan_regional_v2",
                "construction_audit_rule_id": "not_applicable",
                "construction_spend_scope": "canada_tangible_capex_ex_land_ip",
                "construction_io_calibration_status": "statcan_regional_not_us_calibrated",
                "audit_comparable_direct_construction_job_years": direct_jobs,
                "equipment_direct_construction_job_years": 0.0,
                "excluded_land_spend_usd": np.nan,
                "excluded_ip_spend_usd": np.nan,
            }
        )
        for metric in ("earnings_usd", "output_usd", "value_added_usd"):
            record[f"direct_construction_{metric}"] = _sum_effect(
                construction_detail, "direct", metric
            )
            record[f"indirect_construction_{metric}"] = _sum_effect(
                non_anchor, "direct", metric
            ) + _sum_effect(detail, "indirect", metric)
            record[f"induced_construction_{metric}"] = _sum_effect(
                detail, "induced", metric
            )
        for metric in ("jobs", "earnings_usd", "output_usd", "value_added_usd"):
            suffix = "job_years" if metric == "jobs" else metric
            record[f"total_construction_{suffix}"] = (
                record[f"direct_construction_{suffix}"]
                + record[f"indirect_construction_{suffix}"]
                + record[f"induced_construction_{suffix}"]
            )
        record["audit_modeled_direct_construction_job_years"] = direct_jobs
        record["audit_modeled_indirect_construction_job_years"] = record[
            "indirect_construction_job_years"
        ]
        record["audit_modeled_induced_construction_job_years"] = record[
            "induced_construction_job_years"
        ]
        record["audit_modeled_total_construction_job_years"] = record[
            "total_construction_job_years"
        ]
        result_records.append(record)
    return pd.DataFrame(result_records), pd.concat(detail_records, ignore_index=True)


def _apply_canada_operations_spend_io(
    annual: pd.DataFrame,
    io_system: IOSystem,
    canada_data: CanadaRegionalData,
    load_st: pd.DataFrame | None,
    calibration: DataCenterCalibration,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result_records: list[dict] = []
    vendor_details: list[pd.DataFrame] = []
    electricity_details: list[pd.DataFrame] = []
    if load_st is not None and not load_st.empty:
        loads = (
            load_st.rename(columns={"t": "impact_year", "Value": "electricity_load_mw"})
            .groupby(["st", "impact_year"], as_index=False)["electricity_load_mw"]
            .sum()
        )
        load_lookup = {
            (str(row.st), int(row.impact_year)): float(row.electricity_load_mw)
            for row in loads.itertuples(index=False)
        }
    else:
        load_lookup = {}
    for annual_row in annual.itertuples(index=False):
        region = str(annual_row.st)
        year = int(annual_row.impact_year)
        scenario = str(annual_row.scenario)
        rows = canada_data.rows(region, "operations")
        payroll_per_job = canada_data.employee_compensation_per_job(region)
        direct_jobs = float(annual_row.direct_operating_jobs)
        direct_compensation = direct_jobs * payroll_per_job
        direct_output = direct_compensation / calibration.operating_employee_comp_share
        direct_value_added = direct_output * calibration.operating_value_added_share
        vendor_budget = direct_output * calibration.operating_non_electric_share
        vendor_template = _canada_producer_template(rows, io_system, region)
        vendor = _scaled_detail(vendor_template, vendor_budget)
        vendor["st"] = region
        vendor["country"] = "CAN"
        vendor["scenario"] = scenario
        vendor["impact_year"] = year
        vendor["phase"] = "operations_vendor"
        vendor["classification"] = "indirect"

        household = canada_data.household_parameters(region)
        wage_share = float(household["wage_salary_share"])
        eta = float(household["employee_compensation_to_consumption"])
        pce = io_system.pce_vector(region)
        pce["component"] = "direct_payroll_pce"
        pce["country"] = "CAN"
        pce["purchaser_spend_usd"] = direct_compensation * eta * pce["pce_share"]
        pce["local_share"] = pce["producer_industry"].map(
            lambda industry: canada_data.local_share(region, str(industry))
        )
        pce["local_purchaser_spend_usd"] = (
            pce["purchaser_spend_usd"] * pce["local_share"]
        )
        pce["producer_demand_usd"] = pce["local_purchaser_spend_usd"]
        pce_detail = io_system.apply(region, pce)
        pce_detail["st"] = region
        pce_detail["country"] = "CAN"
        pce_detail["scenario"] = scenario
        pce_detail["impact_year"] = year
        pce_detail["phase"] = "operations_payroll_pce"
        pce_detail["classification"] = "payroll_induced"
        implied_non_electric_spend = direct_compensation + vendor_budget
        combined = pd.concat([vendor, pce_detail], ignore_index=True)
        combined["implied_total_spend_usd"] = implied_non_electric_spend
        vendor_details.append(combined)

        load_mw = load_lookup.get((region, year), 0.0)
        price = canada_data.electricity_price(region)
        tariff_crosscheck = load_mw * 8760.0 * price
        electricity_spend = direct_output * calibration.operating_electricity_share
        electricity_local_share = canada_data.local_share(region, "tcpu", "ENE221100")
        electricity_input = pd.DataFrame(
            {
                "component": ["electricity"],
                "statcan_product": ["ENE221100"],
                "country": ["CAN"],
                "jedi_industry": ["tcpu"],
                "producer_industry": ["tcpu"],
                "purchaser_spend_usd": [electricity_spend],
                "local_share": [electricity_local_share],
                "local_purchaser_spend_usd": [
                    electricity_spend * electricity_local_share
                ],
                "producer_demand_usd": [electricity_spend * electricity_local_share],
            }
        )
        electricity = io_system.apply(region, electricity_input)
        electricity["st"] = region
        electricity["country"] = "CAN"
        electricity["scenario"] = scenario
        electricity["impact_year"] = year
        electricity["phase"] = "electricity_excluded"
        electricity["classification"] = "excluded_from_campus_total"
        electricity["electricity_load_mw"] = load_mw
        electricity["electricity_price_2022_usd_per_mwh"] = price
        electricity["implied_electricity_price_usd_per_mwh"] = price
        electricity["implied_total_spend_usd"] = electricity_spend
        electricity["load_tariff_crosscheck_usd"] = tariff_crosscheck
        electricity_details.append(electricity)

        record = annual_row._asdict()
        record.update(
            {
                "indirect_operating_jobs": _sum_effect(vendor, "direct", "jobs")
                + _sum_effect(vendor, "indirect", "jobs"),
                "induced_operating_jobs": _sum_effect(vendor, "induced", "jobs")
                + _sum_effect(pce_detail, "direct", "jobs")
                + _sum_effect(pce_detail, "indirect", "jobs"),
                "implied_total_spend_usd": implied_non_electric_spend,
                "local_final_demand_usd": float(vendor["producer_demand_usd"].sum())
                + float(pce_detail["producer_demand_usd"].sum()),
                "io_implied_direct_jobs": _sum_effect(vendor, "direct", "jobs"),
                "direct_reconciliation_ratio": 1.0,
                "impact_method": "spend_io",
                "economic_method_version": ECONOMIC_METHOD_VERSION,
                "spend_profile_id": "statcan_regional_v2",
                "spend_scale_method": "dc_output_calibrated_non_electric_composition",
                "electricity_excluded_from_total": True,
                "spend_profile_quality_flag": canada_data.quality(region, "operations"),
                "spend_scope": "canada_payroll_plus_non_electric_inputs",
                "electricity_spend_excluded_usd": electricity_spend,
                "electricity_load_tariff_crosscheck_usd": tariff_crosscheck,
                "electricity_price_2022_usd_per_mwh": price,
                "employee_wage_share": wage_share,
                "wage_salary_share": wage_share,
                "dpi_to_personal_income": household["dpi_to_personal_income"],
                "pce_to_dpi": household["pce_to_dpi"],
                "employee_compensation_to_consumption": eta,
                "negative_household_saving": household["negative_household_saving"],
                "household_factor_fallback": household["household_factor_fallback"],
                "direct_operating_employee_compensation_usd": direct_compensation,
                "direct_operating_payroll_usd": direct_compensation,
                "direct_operating_output_usd": direct_output,
                "direct_operating_value_added_usd": direct_value_added,
                "operating_employee_comp_share": (
                    calibration.operating_employee_comp_share
                ),
                "operating_value_added_share": calibration.operating_value_added_share,
                "operating_non_electric_share": calibration.operating_non_electric_share,
            }
        )
        for metric in ("earnings_usd", "output_usd", "value_added_usd"):
            direct_values = {
                "earnings_usd": direct_compensation,
                "output_usd": direct_output,
                "value_added_usd": direct_value_added,
            }
            record[f"direct_operating_{metric}"] = direct_values[metric]
            record[f"indirect_operating_{metric}"] = _sum_effect(
                vendor, "direct", metric
            ) + _sum_effect(vendor, "indirect", metric)
            record[f"induced_operating_{metric}"] = (
                _sum_effect(vendor, "induced", metric)
                + _sum_effect(pce_detail, "direct", metric)
                + _sum_effect(pce_detail, "indirect", metric)
            )
        for metric in ("jobs", "earnings_usd", "output_usd", "value_added_usd"):
            record[f"total_operating_{metric}"] = (
                record[f"direct_operating_{metric}"]
                + record[f"indirect_operating_{metric}"]
                + record[f"induced_operating_{metric}"]
            )
        result_records.append(record)
    return (
        pd.DataFrame(result_records),
        pd.concat(vendor_details, ignore_index=True),
        pd.concat(electricity_details, ignore_index=True),
    )


def apply_construction_spend_io(
    annual: pd.DataFrame,
    io_system: IOSystem,
    profiles: SpendProfile,
    canada_data: CanadaRegionalData | None = None,
    canada_spend_profile: str = "statcan_regional",
    construction_localization: ConstructionLocalization | None = None,
    calibration: DataCenterCalibration | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    annual = annual.copy()
    annual["_input_order"] = np.arange(len(annual))
    result_frames: list[pd.DataFrame] = []
    detail_frames: list[pd.DataFrame] = []
    localized = (
        annual["country"].eq("CAN")
        if canada_spend_profile == "statcan_regional"
        else pd.Series(False, index=annual.index)
    )
    if localized.any():
        canada_data = canada_data or CanadaRegionalData.load()
        calibration = calibration or DataCenterCalibration.from_assumptions()
        result, detail = _apply_canada_construction_spend_io(
            annual[localized].drop(columns="_input_order"),
            io_system,
            canada_data,
            calibration,
        )
        result = result.merge(
            annual.loc[
                localized, ["impact_year", "st", "country", "scenario", "_input_order"]
            ],
            on=["impact_year", "st", "country", "scenario"],
            how="left",
        )
        result_frames.append(result)
        detail_frames.append(detail)
    remainder = annual[~localized]
    if not remainder.empty:
        construction_localization = (
            construction_localization or ConstructionLocalization.load()
        )
        result, detail = _apply_construction_share_profile(
            remainder.drop(columns="_input_order"),
            io_system,
            profiles,
            construction_localization,
        )
        result = result.merge(
            remainder[["impact_year", "st", "country", "scenario", "_input_order"]],
            on=["impact_year", "st", "country", "scenario"],
            how="left",
        )
        result_frames.append(result)
        detail_frames.append(detail)
    combined = pd.concat(result_frames, ignore_index=True).sort_values("_input_order")
    return combined.drop(columns="_input_order").reset_index(drop=True), pd.concat(
        detail_frames, ignore_index=True
    )


def apply_operations_spend_io(
    annual: pd.DataFrame,
    io_system: IOSystem,
    profiles: SpendProfile,
    payroll_spendable_share: float,
    load_st: pd.DataFrame | None = None,
    canada_data: CanadaRegionalData | None = None,
    canada_spend_profile: str = "statcan_regional",
    calibration: DataCenterCalibration | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    annual = annual.copy()
    annual["_input_order"] = np.arange(len(annual))
    result_frames: list[pd.DataFrame] = []
    vendor_frames: list[pd.DataFrame] = []
    electricity_frames: list[pd.DataFrame] = []
    localized = (
        annual["country"].eq("CAN")
        if canada_spend_profile == "statcan_regional"
        else pd.Series(False, index=annual.index)
    )
    if localized.any():
        canada_data = canada_data or CanadaRegionalData.load()
        calibration = calibration or DataCenterCalibration.from_assumptions()
        result, vendor, electricity = _apply_canada_operations_spend_io(
            annual[localized].drop(columns="_input_order"),
            io_system,
            canada_data,
            load_st,
            calibration,
        )
        result = result.merge(
            annual.loc[
                localized, ["impact_year", "st", "country", "scenario", "_input_order"]
            ],
            on=["impact_year", "st", "country", "scenario"],
            how="left",
        )
        result_frames.append(result)
        vendor_frames.append(vendor)
        electricity_frames.append(electricity)
    remainder = annual[~localized]
    if not remainder.empty:
        result, vendor, electricity = _apply_operations_share_profile(
            remainder.drop(columns="_input_order"),
            io_system,
            profiles,
            payroll_spendable_share,
            load_st,
        )
        result = result.merge(
            remainder[["impact_year", "st", "country", "scenario", "_input_order"]],
            on=["impact_year", "st", "country", "scenario"],
            how="left",
        )
        result_frames.append(result)
        vendor_frames.append(vendor)
        electricity_frames.append(electricity)
    combined = pd.concat(result_frames, ignore_index=True).sort_values("_input_order")
    return (
        combined.drop(columns="_input_order").reset_index(drop=True),
        pd.concat(vendor_frames, ignore_index=True),
        pd.concat(electricity_frames, ignore_index=True),
    )
