from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_manifest(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "employment_run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_data_dictionary(output_dir: Path) -> None:
    rows = [
        {
            "field": "case_name",
            "description": "ReEDS case directory name.",
        },
        {
            "field": "impact_year",
            "description": "Calendar year of the reported annual impact.",
        },
        {
            "field": "tech",
            "description": (
                "Technology label. 'AIDC' for data center campus employment; "
                "power-sector technology name (e.g. 'utility pv', 'ng-cc') for JEDI rows."
            ),
        },
        {
            "field": "st",
            "description": "State or province abbreviation.",
        },
        {
            "field": "country",
            "description": "Country for the state/province row ('USA' or 'Canada').",
        },
        {
            "field": "scenario",
            "description": (
                "Complete output scenario: low, base, or high. The JEDI point estimate is "
                "replicated into each complete scenario and combined with the matching AIDC "
                "assumption scenario. Always filter to one scenario before summing."
            ),
        },
        {
            "field": "jobs_direct",
            "description": (
                "Direct annual jobs/job-years. For construction rows this is annualized direct "
                "construction job-years; for operating rows this is annual direct operating jobs."
            ),
        },
        {
            "field": "jobs_indirect",
            "description": (
                "Indirect annual jobs/job-years. Present for AIDC and for JEDI full-breakdown outputs; "
                "NaN for legacy total-only JEDI outputs."
            ),
        },
        {
            "field": "jobs_induced",
            "description": (
                "Induced annual jobs/job-years. Present for AIDC and for JEDI full-breakdown outputs; "
                "NaN for legacy total-only JEDI outputs."
            ),
        },
        {
            "field": "jobs_indirect_induced",
            "description": (
                "Combined indirect + induced jobs. For JEDI full_breakdown this is computed as "
                "indirect + induced. For AIDC it is computed from the separate indirect and induced columns "
                "when present, otherwise read from the legacy combined source column."
            ),
        },
        {
            "field": "jobs_total",
            "description": "Total annual jobs/job-years (direct + indirect + induced).",
        },
        {
            "field": "earnings_total_usd",
            "description": (
                "Total labor income (direct + indirect + induced); `earnings` is retained as "
                "the compatibility field name. JEDI rows use workbook labor income. AIDC "
                "direct operations use employee compensation, while supplier and household "
                "effects use the public BEA-LQ or StatCan regional system."
            ),
        },
        {
            "field": "output_total_usd",
            "description": (
                "Total economic output (direct + indirect + induced). For JEDI rows, read from "
                "JEDI workbook output. AIDC direct operations use the data-center-specific "
                "employee-compensation/output calibration; other effects use the public "
                "BEA-LQ or StatCan regional system."
            ),
        },
        {
            "field": "value_added_total_usd",
            "description": (
                "Total value added (direct + indirect + induced). JEDI rows come from the "
                "workbook; AIDC rows use the same implied final-demand vector as jobs, "
                "earnings, and output. Value added is a separate GDP-contribution metric."
            ),
        },
        {
            "field": "source_currency",
            "description": "Currency of the unnormalized source coefficients or JEDI output.",
        },
        {
            "field": "multiplier_source_currency",
            "description": "Currency of the original I-O multiplier source before any embedded FX adjustment.",
        },
        {
            "field": "source_price_year",
            "description": "Price year of the unnormalized monetary values.",
        },
        {
            "field": "fx_source_currency_per_usd",
            "description": "Source-currency units per USD; Canadian AIDC multiplier conversion is already embedded and is not applied twice.",
        },
        {
            "field": "price_adjustment_factor",
            "description": "Multiplier applied to convert source-price-year money into the target price year.",
        },
        {
            "field": "multiplier_data_year",
            "description": "Single underlying JEDI/IMPLAN or StatCan multiplier vintage; blank when a row combines vintages.",
        },
        {
            "field": "multiplier_data_vintages",
            "description": "Complete semicolon-delimited multiplier vintages, including mixed-source proxy rows.",
        },
        {
            "field": "deflator_source",
            "description": "Named price-year conversion source; ReEDS inputs/financials/deflator.csv for all normalized outputs.",
        },
        {
            "field": "fx_application",
            "description": "Whether currency conversion is absent or already embedded in the jobs multiplier.",
        },
        {
            "field": "monetary_basis_verified",
            "description": "True when currency, source year, deflator, and target price year are explicit.",
        },
        {
            "field": "currency",
            "description": "Common output currency; currently USD.",
        },
        {
            "field": "price_year",
            "description": "Common constant-dollar output price year; currently 2024.",
        },
    ]
    pd.DataFrame(rows).to_csv(
        output_dir / "employment_data_dictionary.csv", index=False
    )


def write_run_log(output_dir: Path, lines: list[str]) -> None:
    body = "\n".join(f"- {line}" for line in lines)
    (output_dir / "employment_run_log.md").write_text(body + "\n", encoding="utf-8")
