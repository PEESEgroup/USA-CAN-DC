
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .assumptions import AssumptionRegister


ECONOMIC_METHOD_VERSION = "dc_calibrated_v2"


@dataclass(frozen=True)
class DataCenterCalibration:
    capex_share_land: float
    capex_share_construction: float
    capex_share_equipment: float
    capex_share_other: float
    equipment_to_construction_ratio: float
    equipment_indirect_local_share: float
    operating_employee_comp_share: float
    operating_electricity_share: float
    operating_value_added_share: float

    @classmethod
    def from_assumptions(
        cls, assumptions: AssumptionRegister | None = None
    ) -> "DataCenterCalibration":
        register = assumptions or AssumptionRegister.load()

        def value(name: str) -> float:
            result = register.triplet(name).base
            if result is None:
                raise ValueError(f"Production calibration {name!r} cannot be null")
            return float(result)

        calibration = cls(
            capex_share_land=value("dc_capex_share_land"),
            capex_share_construction=value("dc_capex_share_construction"),
            capex_share_equipment=value("dc_capex_share_equipment"),
            capex_share_other=value("dc_capex_share_other"),
            equipment_to_construction_ratio=value("dc_equipment_to_construction_ratio"),
            equipment_indirect_local_share=value("dc_equipment_indirect_local_share"),
            operating_employee_comp_share=value("dc_operating_employee_comp_share"),
            operating_electricity_share=value("dc_operating_electricity_share"),
            operating_value_added_share=value("dc_operating_value_added_share"),
        )
        calibration.validate()
        return calibration

    @property
    def operating_intermediate_share(self) -> float:
        return 1.0 - self.operating_value_added_share

    @property
    def operating_non_electric_share(self) -> float:
        return self.operating_intermediate_share - self.operating_electricity_share

    def validate(self) -> None:
        capex = np.array(
            [
                self.capex_share_land,
                self.capex_share_construction,
                self.capex_share_equipment,
                self.capex_share_other,
            ],
            dtype=float,
        )
        if not np.isfinite(capex).all() or (capex < 0).any():
            raise ValueError("Data-center CAPEX shares must be finite and non-negative")
        if not np.isclose(capex.sum(), 1.0, atol=1e-12):
            raise ValueError(f"Data-center CAPEX shares sum to {capex.sum()}, not one")
        implied_ratio = self.capex_share_equipment / self.capex_share_construction
        if not np.isclose(
            self.equipment_to_construction_ratio, implied_ratio, atol=1e-12
        ):
            raise ValueError(
                "Equipment/construction ratio does not equal the CAPEX-share ratio"
            )
        if not np.isfinite(self.equipment_indirect_local_share) or not (
            0.0 <= self.equipment_indirect_local_share <= 1.0
        ):
            raise ValueError(
                "equipment_indirect_local_share must lie within [0, 1]"
            )
        for name, value in {
            "operating_employee_comp_share": self.operating_employee_comp_share,
            "operating_electricity_share": self.operating_electricity_share,
            "operating_value_added_share": self.operating_value_added_share,
        }.items():
            if not np.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie strictly between zero and one")
        if self.operating_employee_comp_share > self.operating_value_added_share:
            raise ValueError(
                "Employee compensation cannot exceed operating value added"
            )
        if self.operating_non_electric_share < 0.0:
            raise ValueError(
                "Calibrated non-electric operating share cannot be negative"
            )
