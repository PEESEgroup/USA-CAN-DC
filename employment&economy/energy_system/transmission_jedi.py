
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import xlwings as xw

from employment.energy_system.case_construction_jedi import (
    IMPACT_EFFECTS,
    IMPACT_METRICS,
    ImpactBreakdownSpec,
    SupportedWorkbook,
    _impact_column,
    add_scaled_impact_columns,
    create_excel_app,
    read_impact_breakdown,
    save_operating_summaries,
    save_summaries,
)


logger = logging.getLogger(__name__)

WORKBOOK_NAME = "jedi-transmission-line-model.xlsm"



SUPPORTED_TYPES = {"AC", "LCC", "VSC", "B2B"}
REPORTED_TYPES = SUPPORTED_TYPES
JEDI_ROOT = Path(__file__).resolve().parent
US_WORKBOOK = JEDI_ROOT / "jedi_us" / WORKBOOK_NAME
CANADA_WORKBOOK = JEDI_ROOT / "jedi_canada" / WORKBOOK_NAME
US_STATE_LOOKUP = JEDI_ROOT / "auxiliary_data" / "runtime_inputs" / "State_Abbr.xlsx"
CANADA_STATE_LOOKUP = (
    JEDI_ROOT / "auxiliary_data" / "runtime_inputs" / "canada_state_lookup.xlsx"
)


def _line_config(trtype: str, mw: float) -> tuple[str, float]:
    if trtype in {"LCC", "VSC"}:
        return "500 kV HVDC", 3000.0
    for limit, label, reference_mw in (
        (150.0, "115 kV AC", 150.0),
        (400.0, "230 kV AC", 400.0),
        (900.0, "345 kV AC", 900.0),
        (1500.0, "500 kV AC", 1500.0),
        (float("inf"), "765 kV AC", 3000.0),
    ):
        if mw <= limit:
            return label, reference_mw
    raise AssertionError("unreachable")


def _location_names(canada_lookup_path: Path = CANADA_STATE_LOOKUP) -> dict[str, str]:
    us = pd.read_excel(US_STATE_LOOKUP).set_index("Abbr")["State"].to_dict()
    canada = pd.read_excel(canada_lookup_path).set_index("Abbr")["JEDIName"].to_dict()
    overlap = set(us) & set(canada)
    if overlap:
        raise ValueError(
            f"US and Canada JEDI location lookups overlap: {sorted(overlap)}"
        )
    return {**us, **canada}


def build_transmission_events(
    case_dir: Path,
    money_year: int,
    canada_lookup_path: Path = CANADA_STATE_LOOKUP,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    case_dir = Path(case_dir)
    tran = pd.read_csv(case_dir / "outputs" / "tran_out.csv")
    miles = pd.read_csv(case_dir / "inputs_case" / "transmission_miles.csv").rename(
        columns={"*r": "r"}
    )
    hierarchy = pd.read_csv(
        case_dir / "outputs" / "hierarchy.csv", usecols=["r", "st", "country"]
    )
    states = hierarchy.drop_duplicates("r").set_index("r")
    locations = _location_names(canada_lookup_path)

    key = ["r", "rr", "trtype"]
    tran = tran[tran["trtype"].isin(REPORTED_TYPES)].copy()
    tran = (
        tran.groupby(key + ["t"], as_index=False)["Value"]
        .max()
        .sort_values(key + ["t"])
    )
    tran["build_mw"] = tran.groupby(key)["Value"].diff().fillna(0.0).clip(lower=0.0)
    tran = tran[tran["build_mw"].gt(0)].copy()

    miles = miles[miles["trtype"].isin(SUPPORTED_TYPES)].copy()
    miles = miles.groupby(key, as_index=False)["miles"].max()
    coverage = tran.merge(miles, on=key, how="left")
    coverage = coverage.rename(columns={"t": "online_year"})
    coverage["from_st"] = coverage["r"].map(states["st"])
    coverage["to_st"] = coverage["rr"].map(states["st"])
    coverage["from_country"] = coverage["r"].map(states["country"])
    coverage["to_country"] = coverage["rr"].map(states["country"])


    coverage["from_country"] = coverage["from_country"].replace({"CAN": "Canada"})
    coverage["to_country"] = coverage["to_country"].replace({"CAN": "Canada"})
    coverage["jedi_proxy_type"] = coverage["trtype"].replace({"B2B": "AC"})
    coverage["status"] = "supported"
    coverage["status_reason"] = (
        "interregional AC/HVDC expansion mapped to US/Canada Transmission Line JEDI"
    )
    coverage.loc[
        coverage["miles"].isna() | coverage["miles"].le(0), ["status", "status_reason"]
    ] = (
        "unsupported_missing_route_miles",
        "no positive route miles in inputs_case/transmission_miles.csv",
    )
    unsupported_country = ~coverage["from_country"].isin({"USA", "Canada"}) | ~coverage[
        "to_country"
    ].isin({"USA", "Canada"})
    coverage.loc[unsupported_country, ["status", "status_reason"]] = (
        "unsupported_geography",
        "endpoint country is outside the US/Canada JEDI workbook coverage",
    )
    canada_endpoint = coverage["from_country"].eq("Canada") | coverage["to_country"].eq(
        "Canada"
    )
    canada_missing_workbook = canada_endpoint & ~CANADA_WORKBOOK.exists()
    coverage.loc[canada_missing_workbook, ["status", "status_reason"]] = (
        "unsupported_geography",
        "Canada-enhanced Transmission Line JEDI workbook is missing; run Canada_data/merge_canada_into_jedi.py",
    )
    unknown_location = ~coverage["from_st"].isin(locations) | ~coverage["to_st"].isin(
        locations
    )
    coverage.loc[unknown_location, ["status", "status_reason"]] = (
        "unsupported_geography",
        "endpoint is missing from the US/Canada JEDI location lookup",
    )

    rows: list[dict] = []
    for record in coverage.query("status == 'supported'").to_dict("records"):
        endpoints = (
            [(record["from_st"], record["from_country"])]
            if record["from_st"] == record["to_st"]
            else [
                (record["from_st"], record["from_country"]),
                (record["to_st"], record["to_country"]),
            ]
        )
        share = 1.0 / len(endpoints)
        line_type, reference_mw = _line_config(
            str(record["jedi_proxy_type"]), float(record["build_mw"])
        )
        for st, country in endpoints:
            rows.append(
                {
                    **record,
                    "st": st,
                    "country": country,
                    "state_name": locations[st],
                    "jedi_line_type": line_type,
                    "jedi_proxy_assumption": (
                        "B2B treated as AC Transmission Line JEDI proxy"
                        if record["trtype"] == "B2B"
                        else "native transmission type mapping"
                    ),
                    "line_reference_mw": reference_mw,
                    "endpoint_allocation_share": share,
                    "project_scale_factor": float(record["build_mw"])
                    / reference_mw
                    * share,
                    "construction_start_year": int(record["online_year"]) - 1,
                    "construction_start_year_workbook": min(
                        int(record["online_year"]) - 1, 2030
                    ),
                    "workbook_year_is_proxy": int(record["online_year"]) - 1 > 2030,
                    "money_year": min(int(money_year), 2030),
                    "build_years": 1,
                }
            )
    return pd.DataFrame(rows), coverage


def write_transmission_inputs(book: xw.main.Book, event: dict) -> None:
    sheet = book.sheets["ProjectData"]
    sheet.range("B16").value = str(event["state_name"])
    sheet.range("B18").value = int(event["construction_start_year_workbook"])
    sheet.range("B19").value = event["jedi_line_type"]
    sheet.range("B20").value = float(event["miles"])
    sheet.range("B21").value = "Flat w/access"
    sheet.range("B22").value = "Rural"
    sheet.range("B24").value = "Simple"
    sheet.range("B50").value = int(event["money_year"])


def _spec() -> SupportedWorkbook:
    return SupportedWorkbook(
        workbook_name=WORKBOOK_NAME,
        workbook_key="transmission",
        summary_sheet="SummaryResults",
        construction_breakdown=ImpactBreakdownSpec(
            "B34:E34", "B35:E35", "B36:E36", "B37:E37"
        ),
        operating_breakdown=ImpactBreakdownSpec(
            "B42:E42", "B43:E43", "B44:E44", "B45:E45"
        ),
        money_scale=1_000_000.0,
        writer=write_transmission_inputs,
    )


def annualize_transmission_impacts(
    raw: pd.DataFrame, case_end_year: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()
    construction_rows: list[dict] = []
    operating_rows: list[dict] = []
    for event in raw.to_dict("records"):
        common = {
            "tech": "transmission",
            "st": event["st"],
            "country": "CAN" if event["country"] == "Canada" else event["country"],
            "state_name": event["state_name"],
            "online_year": int(event["online_year"]),
            "build_mw": float(event["build_mw"]),
            "money_year": int(event["money_year"]),
            "currency": "USD",
            "price_year": int(event["money_year"]),
            "transmission_event": True,
            "trtype": event["trtype"],
        }
        construction = {**common, "impact_year": int(event["online_year"]) - 1}
        for metric in IMPACT_METRICS:
            for effect in IMPACT_EFFECTS:
                construction[_impact_column("construction", metric, effect)] = float(
                    event[_impact_column("construction", metric, effect)]
                )
            construction[f"construction_{metric}"] = construction[
                _impact_column("construction", metric, "total")
            ]
        construction_rows.append(construction)

        for year in range(int(event["online_year"]), int(case_end_year) + 1):
            operating = {**common, "impact_year": year}
            for metric in IMPACT_METRICS:
                for effect in IMPACT_EFFECTS:
                    operating[_impact_column("operating", metric, effect)] = float(
                        event[_impact_column("operating", metric, effect, annual=True)]
                    )
                operating[f"operating_{metric}"] = operating[
                    _impact_column("operating", metric, "total")
                ]
            operating_rows.append(operating)
    return pd.DataFrame(construction_rows), pd.DataFrame(operating_rows)


def _append_annual_impacts(
    output_dir: Path, transmission_raw: pd.DataFrame, case_dir: Path
) -> None:
    cap = pd.read_csv(Path(case_dir) / "outputs" / "cap.csv", usecols=["t"])
    case_end_year = int(pd.to_numeric(cap["t"], errors="raise").max())
    construction, operations = annualize_transmission_impacts(
        transmission_raw, case_end_year
    )
    construction_path = output_dir / "construction_impacts_annual.csv"
    operations_path = output_dir / "operating_impacts_annual.csv"
    if not construction.empty:
        existing = (
            pd.read_csv(construction_path)
            if construction_path.exists()
            else pd.DataFrame()
        )
        if not existing.empty and "tech" in existing.columns:
            existing = existing[~existing["tech"].astype(str).eq("transmission")]
        save_summaries(
            pd.concat([existing, construction], ignore_index=True), output_dir
        )
    if not operations.empty:
        existing = (
            pd.read_csv(operations_path) if operations_path.exists() else pd.DataFrame()
        )
        if not existing.empty and "tech" in existing.columns:
            existing = existing[~existing["tech"].astype(str).eq("transmission")]
        save_operating_summaries(
            pd.concat([existing, operations], ignore_index=True), output_dir
        )


def write_electricity_system_coverage(output_dir: Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    records: list[dict] = []
    build_path = output_dir / "build_coverage_report.csv"
    if build_path.exists():
        build = pd.read_csv(build_path)
        status_column = "status" if "status" in build.columns else "coverage_status"
        tech_column = "mapped_tech" if "mapped_tech" in build.columns else "tech"
        for (tech, status), rows in build.groupby(
            [tech_column, status_column], dropna=False
        ):
            category = (
                "storage"
                if any(
                    token in str(tech).lower()
                    for token in ("storage", "battery", "pumped")
                )
                else "generation_or_fuel"
            )
            records.append(
                {
                    "category": category,
                    "technology": tech,
                    "coverage_status": status,
                    "boundary": "generation, fuel, and storage JEDI workbooks",
                    "record_count": len(rows),
                }
            )
    transmission_path = output_dir / "transmission_build_coverage_report.csv"
    if transmission_path.exists():
        transmission = pd.read_csv(transmission_path)
        for status, rows in transmission.groupby("status", dropna=False):
            records.append(
                {
                    "category": "transmission",
                    "technology": "interregional AC/HVDC",
                    "coverage_status": status,
                    "boundary": "Transmission Line JEDI; endpoint allocation",
                    "record_count": len(rows),
                }
            )
    for technology, reason in (
        ("battery manufacturing", "no production battery manufacturing workbook"),
        ("nuclear", "no supported nuclear JEDI workbook in this implementation"),
        ("hydrogen", "hydrogen production and delivery are outside current workbooks"),
        ("oil-gas-steam", "generic steam/oil-gas category lacks a dedicated mapping"),
        (
            "distribution facilities",
            "ReEDS tran_out does not report distribution assets",
        ),
    ):
        records.append(
            {
                "category": "known_gap",
                "technology": technology,
                "coverage_status": "unsupported",
                "boundary": reason,
                "record_count": 0,
            }
        )
    result = pd.DataFrame(records)
    result.to_csv(
        output_dir / "electricity_system_employment_coverage.csv", index=False
    )
    return result


def run_transmission_jedi(
    case_dir: Path,
    output_dir: Path,
    money_year: int = 2024,
    max_events: int | None = None,
) -> pd.DataFrame:
    events, coverage = build_transmission_events(case_dir, money_year)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(output_dir / "transmission_build_coverage_report.csv", index=False)
    if events.empty:
        return events
    if max_events is not None:
        events = events.head(max_events).copy()
    events.to_csv(output_dir / "transmission_supported_build_events.csv", index=False)
    spec = _spec()
    app = create_excel_app()
    raw: list[dict] = []
    try:
        for event in events.to_dict("records"):
            workbook = CANADA_WORKBOOK if event["country"] == "Canada" else US_WORKBOOK
            book = app.books.open(str(workbook))
            try:
                spec.writer(book, event)
                app.api.CalculateFull()
                row = dict(event)
                for prefix, breakdown, annual in (
                    (
                        "construction",
                        read_impact_breakdown(book, spec, spec.construction_breakdown),
                        False,
                    ),
                    (
                        "operating",
                        read_impact_breakdown(book, spec, spec.operating_breakdown),
                        True,
                    ),
                ):



                    breakdown["total"] = [
                        sum(
                            float(breakdown[effect][idx])
                            for effect in ("direct", "indirect", "induced")
                        )
                        for idx in range(len(IMPACT_METRICS))
                    ]
                    add_scaled_impact_columns(
                        row,
                        prefix,
                        breakdown,
                        spec.money_scale,
                        event["project_scale_factor"],
                        annual=annual,
                    )
                raw.append(row)
            finally:
                book.close()
    finally:
        app.quit()
    result = pd.DataFrame(raw)
    result.to_csv(output_dir / "transmission_impacts_raw.csv", index=False)
    construction, operations = annualize_transmission_impacts(
        result,
        int(
            pd.to_numeric(
                pd.read_csv(Path(case_dir) / "outputs" / "cap.csv", usecols=["t"])["t"],
                errors="raise",
            ).max()
        ),
    )
    construction.to_csv(
        output_dir / "transmission_construction_impacts_annual.csv", index=False
    )
    operations.to_csv(
        output_dir / "transmission_operating_impacts_annual.csv", index=False
    )
    _append_annual_impacts(output_dir, result, case_dir)
    return result
