
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

DEFAULT_ASSUMPTIONS_PATH = (
    Path(__file__).resolve().parent / "config" / "assumptions.yml"
)
SCENARIOS = ("low", "base", "high")


_SHARED_DATA_DIR = Path(__file__).resolve().parents[2] / "shared_data"
DEFAULT_US_GEO_MULT_PATH = _SHARED_DATA_DIR / "jedi_state_io_multipliers_us.csv"
DEFAULT_CAN_GEO_MULT_PATH = _SHARED_DATA_DIR / "jedi_province_io_multipliers_can.csv"
DEFAULT_CAN_WAGE_INDEX_PATH = _SHARED_DATA_DIR / "canada_provincial_wage_index.csv"



_DEFAULT_OPS_WEIGHTS: dict[str, float] = {"tcpu": 0.5, "professional_services": 0.5}
_DATACENTER_CONSTRUCTION_INDUSTRY = "datacenter_construction"
_DATACENTER_OPERATIONS_INDUSTRY = "datacenter_operations"


@dataclass(frozen=True)
class Triplet:
    low: float | None
    base: float | None
    high: float | None
    citation: str = ""

    def as_dict(self) -> dict[str, float | None]:
        return {"low": self.low, "base": self.base, "high": self.high}

    @property
    def is_missing(self) -> bool:
        return self.base is None


def _entry_to_triplet(entry: dict) -> Triplet:
    if "value" in entry:
        v = entry["value"]
        return Triplet(low=v, base=v, high=v, citation=entry.get("citation", ""))
    return Triplet(
        low=entry.get("low"),
        base=entry.get("base"),
        high=entry.get("high"),
        citation=entry.get("citation", ""),
    )


class AssumptionRegister:

    def __init__(self, raw: dict):
        self._raw = raw

    @classmethod
    def load(cls, path: Path = DEFAULT_ASSUMPTIONS_PATH) -> "AssumptionRegister":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Assumptions file not found: {path}")
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(raw)

    def triplet(self, name: str) -> Triplet:
        if name not in self._raw:
            raise KeyError(f"Unknown assumption parameter: {name!r}")
        return _entry_to_triplet(self._raw[name])







class GeoMultiplierTable:

    def __init__(
        self,
        data: pd.DataFrame,
        ops_industry_weights: dict[str, float] | None = None,
        provincial_wage_data: pd.DataFrame | None = None,
    ) -> None:
        self._data = data.copy()
        self._ops_weights: dict[str, float] = ops_industry_weights or dict(
            _DEFAULT_OPS_WEIGHTS
        )
        self._rows = {
            (str(row.region), str(row.industry)): row
            for row in self._data.itertuples(index=False)
        }
        wage_data = (
            provincial_wage_data.copy()
            if provincial_wage_data is not None
            else pd.DataFrame()
        )
        self._provincial_wage_rows = {
            str(row.province): row for row in wage_data.itertuples(index=False)
        }
        if {"multiplier_data_year", "target_2022_factor"}.issubset(self._data.columns):
            standard_factors = pd.to_numeric(
                self._data.loc[
                    pd.to_numeric(
                        self._data["multiplier_data_year"], errors="coerce"
                    ).eq(2014),
                    "target_2022_factor",
                ],
                errors="coerce",
            ).dropna()
        else:
            standard_factors = pd.Series(dtype=float)
        self._us_standard_target_factor = (
            float(standard_factors.median())
            if not standard_factors.empty
            else 1.1512656891351583
        )





    def constr_triplet(self, region: str) -> Triplet:
        industry = self.job_multiplier_industry(region, "construction")
        base = self._lookup(region, industry)
        if base is None:
            raise KeyError(
                f"GeoMultiplierTable: region {region!r} has no 'construction' row. "
                "Re-run employment/energy_system/extract_io_multipliers.py to update the CSV."
            )
        return _point_triplet(
            base, f"{industry} construction multiplier (public point estimate)"
        )

    def ops_triplet(self, region: str) -> Triplet:
        base = self._ops_base(region)
        if base is None:
            raise KeyError(
                f"GeoMultiplierTable: region {region!r} has no ops-industry rows "
                f"({list(self._ops_weights)}). "
                "Re-run employment/energy_system/extract_io_multipliers.py to update the CSV."
            )
        return _point_triplet(
            base,
            "JEDI IMPLAN operations proxy (TCPU + Prof. Services) point estimate",
        )

    def constr_scalar(self, region: str, scenario: str) -> float:
        return getattr(self.constr_triplet(region), scenario)

    def ops_scalar(self, region: str, scenario: str) -> float:
        return getattr(self.ops_triplet(region), scenario)





    def constr_indirect_scalar(self, region: str, scenario: str) -> float:
        industry = self.job_multiplier_industry(region, "construction")
        base = self._indirect_ratio(region, industry)
        if base is None:
            raise KeyError(
                f"GeoMultiplierTable: region {region!r} has no 'construction' row."
            )
        return getattr(
            _point_triplet(base, "JEDI IMPLAN construction indirect point estimate"),
            scenario,
        )

    def constr_induced_scalar(self, region: str, scenario: str) -> float:
        industry = self.job_multiplier_industry(region, "construction")
        base = self._induced_ratio(region, industry)
        if base is None:
            raise KeyError(
                f"GeoMultiplierTable: region {region!r} has no 'construction' row."
            )
        return getattr(
            _point_triplet(base, "JEDI IMPLAN construction induced point estimate"),
            scenario,
        )





    def ops_indirect_scalar(self, region: str, scenario: str) -> float:
        exact = self._indirect_ratio(region, _DATACENTER_OPERATIONS_INDUSTRY)
        base = (
            exact
            if exact is not None
            else self._ops_weighted_sub_ratio(region, "indirect")
        )
        if base is None:
            raise KeyError(
                f"GeoMultiplierTable: region {region!r} has no ops-industry rows "
                f"({list(self._ops_weights)})."
            )
        return getattr(
            _point_triplet(base, "JEDI IMPLAN ops indirect proxy point estimate"),
            scenario,
        )

    def ops_induced_scalar(self, region: str, scenario: str) -> float:
        exact = self._induced_ratio(region, _DATACENTER_OPERATIONS_INDUSTRY)
        base = (
            exact
            if exact is not None
            else self._ops_weighted_sub_ratio(region, "induced")
        )
        if base is None:
            raise KeyError(
                f"GeoMultiplierTable: region {region!r} has no ops-industry rows "
                f"({list(self._ops_weights)})."
            )
        return getattr(
            _point_triplet(base, "JEDI IMPLAN ops induced proxy point estimate"),
            scenario,
        )





    def constr_earnings_per_job_scalar(
        self, region: str, effect: str, scenario: str
    ) -> float:
        val = self._earnings_per_job(region, "construction", effect)
        if val is None:
            raise KeyError(
                f"GeoMultiplierTable: earnings_per_job missing for "
                f"region={region!r}, industry='construction', effect={effect!r}. "
                "Re-run extract_io_multipliers.py to update the CSV."
            )
        return getattr(
            _point_triplet(val, f"JEDI IMPLAN construction {effect} earnings/job"),
            scenario,
        )

    def constr_output_per_job_scalar(
        self, region: str, effect: str, scenario: str
    ) -> float:
        val = self._output_per_job(region, "construction", effect)
        if val is None:
            raise KeyError(
                f"GeoMultiplierTable: output_per_job missing for "
                f"region={region!r}, industry='construction', effect={effect!r}."
            )
        return getattr(
            _point_triplet(val, f"JEDI IMPLAN construction {effect} output/job"),
            scenario,
        )

    def constr_metric_per_direct_job_scalar(
        self, region: str, metric: str, effect: str, scenario: str
    ) -> float:
        exact = self._metric_per_direct_job(
            region, _DATACENTER_CONSTRUCTION_INDUSTRY, metric, effect
        )
        val = (
            exact
            if exact is not None
            else self._metric_per_direct_job(region, "construction", metric, effect)
        )
        if val is None:
            raise KeyError(
                f"GeoMultiplierTable: {metric} multiplier missing for "
                f"region={region!r}, industry='construction', effect={effect!r}."
            )
        return getattr(
            _point_triplet(val, f"JEDI construction {effect} {metric}"), scenario
        )





    def ops_earnings_per_job_scalar(
        self, region: str, effect: str, scenario: str
    ) -> float:
        val = self._ops_weighted_earnings_per_job(region, effect)
        if val is None:
            raise KeyError(
                f"GeoMultiplierTable: ops earnings_per_job missing for "
                f"region={region!r}, effect={effect!r} ({list(self._ops_weights)})."
            )
        return getattr(
            _point_triplet(val, f"JEDI IMPLAN ops {effect} earnings/job proxy"),
            scenario,
        )

    def ops_output_per_job_scalar(
        self, region: str, effect: str, scenario: str
    ) -> float:
        val = self._ops_weighted_output_per_job(region, effect)
        if val is None:
            raise KeyError(
                f"GeoMultiplierTable: ops output_per_job missing for "
                f"region={region!r}, effect={effect!r} ({list(self._ops_weights)})."
            )
        return getattr(
            _point_triplet(val, f"JEDI IMPLAN ops {effect} output/job proxy"),
            scenario,
        )

    def ops_metric_per_direct_job_scalar(
        self, region: str, metric: str, effect: str, scenario: str
    ) -> float:
        exact = self._metric_per_direct_job(
            region, _DATACENTER_OPERATIONS_INDUSTRY, metric, effect
        )
        val = (
            exact
            if exact is not None
            else self._ops_weighted_metric_per_direct_job(region, metric, effect)
        )
        if val is None:
            raise KeyError(
                f"GeoMultiplierTable: ops {metric} multiplier missing for "
                f"region={region!r}, effect={effect!r}."
            )
        return getattr(_point_triplet(val, f"JEDI ops {effect} {metric}"), scenario)

    @property
    def name(self) -> str:
        return "public_datacenter_industry"

    def job_multiplier_industry(self, region: str, phase: str) -> str:
        if phase == "construction":
            if (str(region), _DATACENTER_CONSTRUCTION_INDUSTRY) in self._rows:
                return _DATACENTER_CONSTRUCTION_INDUSTRY
            return "construction"
        if phase == "operations":
            if (str(region), _DATACENTER_OPERATIONS_INDUSTRY) in self._rows:
                return _DATACENTER_OPERATIONS_INDUSTRY
            return "tcpu+professional_services"
        raise ValueError(f"Unknown multiplier phase: {phase!r}")

    def job_multiplier_source(self, region: str, phase: str) -> str:
        industry = self.job_multiplier_industry(region, phase)
        if industry == "tcpu+professional_services":
            sources = {
                str(self._rows[(str(region), key)].source)
                for key in self._ops_weights
                if (str(region), key) in self._rows
            }
            return "; ".join(sorted(sources))
        return str(self._rows[(str(region), industry)].source)

    def job_multiplier_data_year(self, region: str, phase: str) -> str:
        industry = self.job_multiplier_industry(region, phase)
        if industry == "tcpu+professional_services":
            years = {
                str(int(self._rows[(str(region), key)].multiplier_data_year))
                for key in self._ops_weights
                if (str(region), key) in self._rows
            }
            return ";".join(sorted(years))
        return str(int(self._rows[(str(region), industry)].multiplier_data_year))

    def monetary_multiplier_industry(self, region: str, phase: str) -> str:
        exact = (
            _DATACENTER_CONSTRUCTION_INDUSTRY
            if phase == "construction"
            else _DATACENTER_OPERATIONS_INDUSTRY
        )
        if self._metric_per_direct_job(region, exact, "earnings", "direct") is not None:
            return exact
        return (
            "construction" if phase == "construction" else "tcpu+professional_services"
        )

    def provincial_wage_index(self, region: str) -> float | None:
        row = self._provincial_wage_rows.get(str(region))
        if row is None:
            return None
        return float(row.province_index_to_canada)

    def provincial_wage_source(self, region: str) -> str:
        row = self._provincial_wage_rows.get(str(region))
        if row is None:
            return ""
        return f"{row.source_table}, {int(row.source_year)}"





    @classmethod
    def load(
        cls,
        us_path: Path = DEFAULT_US_GEO_MULT_PATH,
        can_path: Path = DEFAULT_CAN_GEO_MULT_PATH,
        ops_industry_weights: dict[str, float] | None = None,
        can_wage_index_path: Path = DEFAULT_CAN_WAGE_INDEX_PATH,
    ) -> "GeoMultiplierTable":
        if not us_path.exists():
            raise FileNotFoundError(
                f"US geo multiplier CSV not found: {us_path}\n"
                "Run employment/energy_system/extract_io_multipliers.py to generate it."
            )
        frames = [pd.read_csv(us_path)]
        if can_path.exists():
            frames.append(pd.read_csv(can_path))
        data = pd.concat(frames, ignore_index=True)
        wage_data = (
            pd.read_csv(can_wage_index_path)
            if Path(can_wage_index_path).exists()
            else pd.DataFrame()
        )
        return cls(data, ops_industry_weights, wage_data)





    def _lookup(self, region: str, industry: str) -> float | None:
        row = self._rows.get((region, industry))
        if row is None:
            return None
        return float(row.total_to_direct)

    def _lookup_column(self, region: str, industry: str, col: str) -> float | None:
        row = self._rows.get((region, industry))
        if row is None or not hasattr(row, col):
            return None
        return float(getattr(row, col))

    def _indirect_ratio(self, region: str, industry: str) -> float | None:
        indirect = self._lookup_column(region, industry, "indirect_mult")
        direct = self._lookup_column(region, industry, "direct_mult")
        if indirect is None or direct is None or direct == 0.0:
            return None
        if math.isnan(indirect) or math.isnan(direct):
            return None
        return indirect / direct

    def _induced_ratio(self, region: str, industry: str) -> float | None:
        induced = self._lookup_column(region, industry, "induced_mult")
        direct = self._lookup_column(region, industry, "direct_mult")
        if induced is None or direct is None or direct == 0.0:
            return None
        if math.isnan(induced) or math.isnan(direct):
            return None
        return induced / direct

    def _ops_weighted_sub_ratio(self, region: str, sub: str) -> float | None:
        total_weight = 0.0
        weighted_sum = 0.0
        for industry, weight in self._ops_weights.items():
            val = (
                self._indirect_ratio(region, industry)
                if sub == "indirect"
                else self._induced_ratio(region, industry)
            )
            if val is not None:
                weighted_sum += val * weight
                total_weight += weight
        if total_weight == 0.0:
            return None
        return weighted_sum / total_weight

    def _earnings_per_job(
        self, region: str, industry: str, effect: str
    ) -> float | None:
        earnings = self._lookup_column(region, industry, f"{effect}_earnings_mult")
        jobs = self._lookup_column(region, industry, f"{effect}_mult")
        if earnings is None or jobs is None or jobs == 0.0:
            return None
        if math.isnan(earnings) or math.isnan(jobs):
            return None
        return earnings * 1_000_000 / jobs

    def _output_per_job(self, region: str, industry: str, effect: str) -> float | None:
        output = self._lookup_column(region, industry, f"{effect}_output_mult")
        jobs = self._lookup_column(region, industry, f"{effect}_mult")
        if output is None or jobs is None or jobs == 0.0:
            return None
        if math.isnan(output) or math.isnan(jobs):
            return None
        return output * 1_000_000 / jobs

    def _metric_per_direct_job(
        self, region: str, industry: str, metric: str, effect: str
    ) -> float | None:
        monetary = self._lookup_column(region, industry, f"{effect}_{metric}_mult")
        direct_jobs = self._lookup_column(region, industry, "direct_mult")
        if monetary is None or direct_jobs is None or direct_jobs == 0.0:
            return None
        if math.isnan(monetary) or math.isnan(direct_jobs):
            return None
        row = self._rows[(region, industry)]
        target_factor = float(
            getattr(row, "target_2022_factor", self._us_standard_target_factor)
        )


        return monetary * 1_000_000 / direct_jobs * target_factor

    def _ops_weighted_metric_per_direct_job(
        self, region: str, metric: str, effect: str
    ) -> float | None:
        weighted_sum = 0.0
        total_weight = 0.0
        for industry, weight in self._ops_weights.items():
            val = self._metric_per_direct_job(region, industry, metric, effect)
            if val is not None:
                weighted_sum += weight * val
                total_weight += weight
        if total_weight == 0.0:
            return None
        return weighted_sum / total_weight

    def _ops_weighted_earnings_per_job(self, region: str, effect: str) -> float | None:
        total_weight = 0.0
        weighted_sum = 0.0
        for industry, weight in self._ops_weights.items():
            val = self._earnings_per_job(region, industry, effect)
            if val is not None:
                weighted_sum += val * weight
                total_weight += weight
        if total_weight == 0.0:
            return None
        return weighted_sum / total_weight

    def _ops_weighted_output_per_job(self, region: str, effect: str) -> float | None:
        total_weight = 0.0
        weighted_sum = 0.0
        for industry, weight in self._ops_weights.items():
            val = self._output_per_job(region, industry, effect)
            if val is not None:
                weighted_sum += val * weight
                total_weight += weight
        if total_weight == 0.0:
            return None
        return weighted_sum / total_weight

    def _ops_base(self, region: str) -> float | None:
        exact = self._lookup(region, _DATACENTER_OPERATIONS_INDUSTRY)
        if exact is not None:
            return exact
        total_weight = 0.0
        weighted_sum = 0.0
        for industry, weight in self._ops_weights.items():
            val = self._lookup(region, industry)
            if val is not None:
                weighted_sum += val * weight
                total_weight += weight
        if total_weight == 0.0:
            return None
        return weighted_sum / total_weight


def _point_triplet(base: float, citation: str) -> Triplet:
    return Triplet(
        low=base,
        base=base,
        high=base,
        citation=citation,
    )
