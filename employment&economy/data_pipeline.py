
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd
import requests
import yaml

from employment.AIDC_campus.datacenter_employment.canada_inputs import (
    CANADA_REGIONS,
    CanadaRegionalData,
)
from employment.AIDC_campus.datacenter_employment.calibration import (
    ECONOMIC_METHOD_VERSION,
)
from employment.AIDC_campus.datacenter_employment.public_bea_lq_io import (
    MODEL_PRICE_YEAR,
    PublicBEALQIO,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path(__file__).resolve().with_name("data_sources.yml")
STATE_FIPS_PATH = REPO_ROOT / "inputs" / "shapefiles" / "state_fips_codes.csv"
LOCK_FILENAME = "download_lock.json"
CANADA_SOURCE_IDS = {
    "statcan_34100039",
    "statcan_36100226",
    "statcan_36100478",
    "statcan_36100595",
    "statcan_36100709",
}


def _load_manifest(path: Path) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("sources"), dict):
        raise ValueError("Unsupported employment data-source manifest")
    return data


def _paths(manifest: dict) -> tuple[Path, Path]:
    return (
        REPO_ROOT / manifest["raw_directory"],
        REPO_ROOT / manifest["normalized_directory"],
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_lock(raw_dir: Path) -> dict:
    path = raw_dir / LOCK_FILENAME
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_lock(raw_dir: Path, lock: dict) -> None:
    path = raw_dir / LOCK_FILENAME
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(lock, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp, path)


def download_sources(
    manifest: dict,
    selected: set[str] | None = None,
    force: bool = False,
    offline: bool = False,
    lock_missing_hashes: bool = False,
) -> None:
    raw_dir, _ = _paths(manifest)
    raw_dir.mkdir(parents=True, exist_ok=True)
    lock = _read_lock(raw_dir)
    for source_id, source in manifest["sources"].items():
        if selected and source_id not in selected:
            continue
        destination = raw_dir / source["filename"]
        expected = source.get("sha256") or lock.get(source_id, {}).get("sha256")
        if destination.exists() and not force:
            actual = _sha256(destination)
            if expected and actual != expected:
                raise ValueError(
                    f"Cached {source_id} has SHA-256 {actual}, expected {expected}"
                )
            if expected:
                continue
        if offline:
            raise FileNotFoundError(
                f"Offline cache is missing {source_id}: {destination}"
            )
        temporary = destination.with_suffix(destination.suffix + ".part")
        headers = {}
        previous = lock.get(source_id, {})
        if not force and previous.get("etag"):
            headers["If-None-Match"] = previous["etag"]
        if not force and previous.get("last_modified"):
            headers["If-Modified-Since"] = previous["last_modified"]
        response_headers: dict[str, str | None] = {}
        try:
            try:
                with requests.get(
                    source["url"], stream=True, timeout=(30, 300), headers=headers
                ) as response:
                    if response.status_code == 304 and destination.exists():
                        continue
                    response.raise_for_status()
                    with temporary.open("wb") as handle:
                        for block in response.iter_content(1024 * 1024):
                            if block:
                                handle.write(block)
                    response_headers = {
                        "etag": response.headers.get("ETag"),
                        "last_modified": response.headers.get("Last-Modified"),
                    }
            except requests.RequestException:
                curl = shutil.which("curl")
                if not curl:
                    raise

                subprocess.run(
                    [
                        curl,
                        "--http1.1",
                        "--location",
                        "--fail",
                        "--retry",
                        "3",
                        "--output",
                        str(temporary),
                        source["url"],
                    ],
                    check=True,
                )
            actual = _sha256(temporary)
            if expected and actual != expected:
                temporary.unlink(missing_ok=True)
                raise ValueError(
                    f"Downloaded {source_id} has SHA-256 {actual}, expected {expected}"
                )
            if not expected and not lock_missing_hashes:
                temporary.unlink(missing_ok=True)
                raise ValueError(
                    f"{source_id} has no pinned digest. Re-run with "
                    "--lock-missing-hashes after independently confirming the official file."
                )
            os.replace(temporary, destination)
            lock[source_id] = {
                "sha256": actual,
                "url": source["url"],
                "etag": response_headers.get("etag"),
                "last_modified": response_headers.get("last_modified"),
                "source_data_year": source["source_data_year"],
                "source_price_year": source["source_price_year"],
            }
            _write_lock(raw_dir, lock)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise


def verify_downloads(manifest: dict, selected: set[str] | None = None) -> None:
    raw_dir, _ = _paths(manifest)
    lock = _read_lock(raw_dir)
    errors = []
    for source_id, source in manifest["sources"].items():
        if selected and source_id not in selected:
            continue
        path = raw_dir / source["filename"]
        expected = source.get("sha256") or lock.get(source_id, {}).get("sha256")
        if not path.exists():
            errors.append(f"missing {source_id}: {path}")
        elif not expected:
            errors.append(f"unlocked digest for {source_id}")
        else:
            actual = _sha256(path)
            if actual != expected:
                errors.append(f"bad SHA-256 for {source_id}: {actual} != {expected}")
    if errors:
        raise ValueError("Data-source verification failed:\n" + "\n".join(errors))


def _read_bea_sheet(path: Path, sheet: str = "2017") -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet, header=None)


def _bea_codes(table: pd.DataFrame, stop_code: str) -> tuple[list[str], list[int]]:
    codes: list[str] = []
    columns: list[int] = []
    for column in range(2, table.shape[1]):
        code = str(table.iat[5, column]).strip()
        if code == stop_code:
            break
        if code != "nan" and not code.startswith("T"):
            codes.append(code)
            columns.append(column)
    return codes, columns


def _bea_rows(table: pd.DataFrame, stop_code: str) -> tuple[list[str], list[int]]:
    codes: list[str] = []
    rows: list[int] = []
    for row in range(6, table.shape[0]):
        code = str(table.iat[row, 0]).strip()
        if code == stop_code:
            break
        if code != "nan":
            codes.append(code)
            rows.append(row)
    return codes, rows


def _numeric_block(
    table: pd.DataFrame, rows: list[int], columns: list[int]
) -> np.ndarray:
    return (
        table.iloc[rows, columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(float)
    )


def build_domestic_requirements(
    use_path: Path, supply_path: Path
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame]:
    use = _read_bea_sheet(use_path)
    supply = _read_bea_sheet(supply_path)
    industries, use_columns = _bea_codes(use, "F01000")
    supply_industries, supply_columns = _bea_codes(supply, "T007")
    commodities, use_rows = _bea_rows(use, "T005")
    supply_commodities, supply_rows = _bea_rows(supply, "T017")
    if industries != supply_industries or commodities != supply_commodities:
        raise ValueError("BEA detailed Supply and Use codes do not align")
    use_values = _numeric_block(use, use_rows, use_columns)
    supply_values = _numeric_block(supply, supply_rows, supply_columns)
    supply_headers = {
        str(supply.iat[5, col]).strip(): col for col in range(2, supply.shape[1])
    }
    total_domestic = (
        pd.to_numeric(supply.iloc[supply_rows, supply_headers["T007"]], errors="coerce")
        .fillna(0.0)
        .to_numpy(float)
    )
    total_basic_supply = (
        pd.to_numeric(supply.iloc[supply_rows, supply_headers["T013"]], errors="coerce")
        .fillna(0.0)
        .to_numpy(float)
    )
    total_purchaser = (
        pd.to_numeric(supply.iloc[supply_rows, supply_headers["T016"]], errors="coerce")
        .fillna(0.0)
        .to_numpy(float)
    )
    domestic_purchaser_ratio = np.divide(
        total_domestic,
        total_purchaser,
        out=np.zeros_like(total_domestic),
        where=total_purchaser > 0,
    )
    domestic_use = domestic_purchaser_ratio[:, None] * use_values
    market_share = np.divide(
        supply_values.T,
        total_domestic[None, :],
        out=np.zeros_like(supply_values.T),
        where=total_domestic[None, :] > 0,
    )
    output_row = use.index[use.iloc[:, 0].astype(str).eq("T018")]
    if len(output_row) != 1:
        raise ValueError("BEA Use table is missing unique T018 industry output")
    industry_output = (
        pd.to_numeric(use.iloc[output_row[0], use_columns], errors="coerce")
        .fillna(0.0)
        .to_numpy(float)
    )
    industry_use = market_share @ domestic_use
    a = np.divide(
        industry_use,
        industry_output[None, :],
        out=np.zeros_like(industry_use),
        where=industry_output[None, :] > 0,
    )


    a = np.maximum(a, 0.0)
    radius = float(max(abs(np.linalg.eigvals(a))))
    if radius >= 1.0:
        raise ValueError(f"National domestic requirements spectral radius is {radius}")
    titles = [str(use.iat[4, col]).strip() for col in use_columns]

    def row_values(code: str) -> np.ndarray:
        rows = use.index[use.iloc[:, 0].astype(str).eq(code)]
        if len(rows) != 1:
            raise ValueError(f"BEA Use table missing unique {code} row")
        return (
            pd.to_numeric(use.iloc[rows[0], use_columns], errors="coerce")
            .fillna(0.0)
            .to_numpy(float)
            * 1_000_000.0
        )

    output_usd = industry_output * 1_000_000.0
    compensation = row_values("V00100")
    value_added = row_values("VABAS")
    industry_frame = pd.DataFrame(
        {
            "bea_industry": industries,
            "industry_title": titles,
            "national_output_2017_usd": output_usd,
            "national_labor_compensation_per_output": np.divide(
                compensation,
                output_usd,
                out=np.zeros_like(compensation),
                where=output_usd > 0,
            ),
            "national_value_added_per_output_2024": np.divide(
                value_added,
                output_usd,
                out=np.zeros_like(value_added),
                where=output_usd > 0,
            ),
        }
    )
    supply_detail = pd.DataFrame(
        {
            "purchaser_industry": commodities,
            "primary_producer_industry": [
                industries[int(np.argmax(supply_values[row]))]
                for row in range(len(commodities))
            ],
            "basic_share": np.divide(
                total_domestic,
                total_purchaser,
                out=np.zeros_like(total_domestic),
                where=total_purchaser > 0,
            ),
            "trade_share": pd.to_numeric(
                supply.iloc[supply_rows, supply_headers["TRADE"]], errors="coerce"
            )
            .fillna(0.0)
            .to_numpy(float)
            / np.where(total_purchaser > 0, total_purchaser, 1.0),
            "transport_share": pd.to_numeric(
                supply.iloc[supply_rows, supply_headers["TRANS"]], errors="coerce"
            )
            .fillna(0.0)
            .to_numpy(float)
            / np.where(total_purchaser > 0, total_purchaser, 1.0),
            "tax_share": pd.to_numeric(
                supply.iloc[supply_rows, supply_headers["T015"]], errors="coerce"
            )
            .fillna(0.0)
            .to_numpy(float)
            / np.where(total_purchaser > 0, total_purchaser, 1.0),
            "import_share": np.divide(
                total_basic_supply - total_domestic,
                total_purchaser,
                out=np.zeros_like(total_domestic),
                where=total_purchaser > 0,
            ),
        }
    )
    operating_column = industries.index("518200")
    operating_values = np.maximum(use_values[:, operating_column], 0.0)
    eligible = np.array(
        [
            not str(code).startswith("S") and str(code) != "221100"
            for code in commodities
        ]
    )
    non_electric_total = float(operating_values[eligible].sum())
    if non_electric_total <= 0:
        raise ValueError("BEA 518200 has no positive non-electric intermediate inputs")
    operating = pd.DataFrame(
        {
            "input_commodity": commodities,
            "non_electric_input_share": np.where(
                eligible, operating_values / non_electric_total, 0.0
            ),
        }
    )
    operating = operating[
        (
            (operating["non_electric_input_share"] > 0)
            | operating["input_commodity"].eq("221100")
        )
        & ~operating["input_commodity"].str.startswith("S")
    ].copy()
    operating["electricity_flag"] = operating["input_commodity"].eq("221100")
    return industry_frame, a, supply_detail, operating


def aces_asset_shares() -> dict[str, float]:
    information_processing = 105_140.0
    hardware_total = 308_625.0 - 102_116.0
    computer_office = information_processing * (73_232.0 + 12_986.0) / hardware_total
    communications = information_processing * (79_647.0 + 3_277.0) / hardware_total
    other_information = (
        information_processing * (4_280.0 + 3_278.0 + 29_810.0) / hardware_total
    )
    other_machinery = other_information + 6_983.0 + 1_432.0 + 6_726.0 + 103.0
    transportation = 3_811.0
    raw = {
        "computer_and_office_equipment": computer_office,
        "communications_equipment": communications,
        "other_machinery_and_equipment": other_machinery,
        "transportation_equipment": transportation,
    }
    total = sum(raw.values())
    return {key: value / total for key, value in raw.items()}


def _attach_provenance(
    frame: pd.DataFrame, manifest: dict, source_ids: list[str], fallback: str = "none"
) -> pd.DataFrame:
    result = frame.copy()
    sources = [manifest["sources"][source_id] for source_id in source_ids]
    result["download_url"] = ";".join(source["url"] for source in sources)
    result["sha256"] = ";".join(source["sha256"] for source in sources)
    if "currency" not in result:
        result["currency"] = "USD"
    if "price_year" not in result:
        result["price_year"] = MODEL_PRICE_YEAR
    if "fallback_status" not in result:
        result["fallback_status"] = fallback
    if "mapping_status" not in result:
        result["mapping_status"] = "resolved"
    return result


def _extract_statcan_csv(zip_path: Path, table_id: str, directory: Path) -> Path:
    with zipfile.ZipFile(zip_path) as archive:
        candidates = [
            name
            for name in archive.namelist()
            if Path(name).name.lower() in {f"{table_id}-eng.csv", f"{table_id}.csv"}
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one {table_id} data CSV in {zip_path}, found {candidates}"
            )
        archive.extract(candidates[0], directory)
        return directory / candidates[0]


def _attach_canada_provenance(
    frame: pd.DataFrame, manifest: dict, source_ids: list[str]
) -> pd.DataFrame:
    result = frame.copy()
    sources = [manifest["sources"][source_id] for source_id in source_ids]
    result["source_data_year"] = 2022
    result["source_price_year"] = 2022
    result["currency"] = "CAD"
    result["price_year"] = 2022
    result["download_url"] = ";".join(source["url"] for source in sources)
    result["sha256"] = ";".join(source["sha256"] for source in sources)
    quality = result.get(
        "quality_flag", pd.Series("province_exact", index=result.index)
    ).astype(str)
    result["fallback_status"] = quality
    result["mapping_status"] = np.where(
        quality.eq("unresolved"), "unresolved", "resolved"
    )
    return result


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".building")

    frame.to_csv(temporary, index=False, float_format="%.17g")
    os.replace(temporary, path)


def build_canada_normalized(manifest: dict) -> None:
    from employment.energy_system.Canada_data.can_source_data import (
        build_canada_aidc_spend_inputs as builder,
    )

    raw_dir, _ = _paths(manifest)
    verify_downloads(manifest, CANADA_SOURCE_IDS)
    shared = REPO_ROOT / "employment" / "shared_data"
    source_dir = (
        REPO_ROOT
        / "employment"
        / "energy_system"
        / "Canada_data"
        / "can_source_data"
    )
    with tempfile.TemporaryDirectory(prefix="statcan_build_", dir=raw_dir) as temp_name:
        temp = Path(temp_name)
        paths = {
            source_id: _extract_statcan_csv(
                raw_dir / manifest["sources"][source_id]["filename"],
                source_id.removeprefix("statcan_"),
                temp,
            )
            for source_id in CANADA_SOURCE_IDS
        }
        mapping = builder.load_mapping(source_dir / "aidc_statcan_product_mapping.yaml")
        supply, operations = builder.read_supply_use(paths["statcan_36100478"])
        local, gaps = builder.build_local_shares(
            supply, paths["statcan_36100709"], mapping
        )
        margins = builder.build_margins(supply, mapping)
        assets = pd.read_csv(paths["statcan_34100039"], low_memory=False)
        household = pd.read_csv(paths["statcan_36100226"], low_memory=False)
        profiles, payroll, profile_gaps = builder.build_regional_profiles(
            operations, assets, household, local, margins, mapping
        )
        gaps.extend(profile_gaps)




    used_products = set(profiles["statcan_product"].dropna().astype(str)) | {
        "ENE221100"
    }
    local = local[
        (
            local["statcan_product"].astype(str).isin(used_products)
            | local["statcan_product"].astype(str).eq("__JEDI_AGGREGATE__")
        )
        & local["quality_flag"].astype(str).ne("unresolved")
    ].copy()
    margins = margins[margins["statcan_product"].astype(str).isin(used_products)].copy()

    profiles = _attach_canada_provenance(
        profiles,
        manifest,
        [
            "statcan_34100039",
            "statcan_36100478",
            "statcan_36100709",
        ],
    )
    local = _attach_canada_provenance(
        local, manifest, ["statcan_36100478", "statcan_36100709"]
    )
    margins = _attach_canada_provenance(margins, manifest, ["statcan_36100478"])
    payroll = _attach_canada_provenance(
        payroll, manifest, ["statcan_36100478", "statcan_36100226"]
    )
    if any(
        frame["mapping_status"].eq("unresolved").any()
        for frame in (profiles, local, margins, payroll)
    ):
        raise ValueError("Unresolved mappings remain in a Canada main-result table")

    multipliers_path = shared / "jedi_province_io_multipliers_can.csv"
    multipliers = pd.read_csv(multipliers_path, float_precision="round_trip")
    if set(multipliers["region"].astype(str)) != CANADA_REGIONS:
        raise ValueError(
            "Canada multiplier compact input does not cover all ten provinces"
        )
    multipliers = _attach_canada_provenance(
        multipliers, manifest, ["statcan_36100478", "statcan_36100595"]
    )

    multipliers["currency"] = "USD"
    operating_anchors = multipliers[
        multipliers["industry"].eq("datacenter_operations")
    ][["region", "direct_mult", "direct_output_mult"]]
    payroll = payroll.merge(
        operating_anchors, on="region", how="left", validate="one_to_one"
    )
    if payroll[["direct_mult", "direct_output_mult"]].isna().any().any():
        raise ValueError(
            "Canada payroll table lacks BS518000 direct multiplier anchors"
        )
    payroll["employee_compensation_per_job_2022_usd"] = (
        payroll["employee_compensation_per_output"]
        * payroll["direct_output_mult"]
        * 1_000_000.0
        / payroll["direct_mult"]
    )
    payroll = payroll.drop(columns=["direct_mult", "direct_output_mult"])

    candidate = CanadaRegionalData(
        profiles=profiles,
        local_shares=local,
        margins=margins,
        electricity_prices=pd.read_csv(shared / "canada_large_power_prices.csv"),
        payroll=payroll,
    )
    candidate.validate()
    _atomic_csv(profiles, shared / "canada_regional_spend_profiles.csv")
    _atomic_csv(local, shared / "canada_product_local_shares.csv")
    _atomic_csv(margins, shared / "canada_product_margin_factors.csv")
    _atomic_csv(payroll, shared / "canada_payroll_parameters.csv")
    _atomic_csv(multipliers, multipliers_path)
    metadata = {
        "economic_method": "statcan_regional_io",
        "economic_method_version": ECONOMIC_METHOD_VERSION,
        "source_data_year": 2022,
        "source_price_year": 2022,
        "final_result_currency": "USD",
        "final_result_price_year": 2024,
        "unresolved_main_result_mappings": 0,
        "gap_records": len(gaps),
        "sources": {
            source_id: {
                "url": manifest["sources"][source_id]["url"],
                "sha256": manifest["sources"][source_id]["sha256"],
            }
            for source_id in sorted(CANADA_SOURCE_IDS)
        },
    }
    (shared / "canada_source_build_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def _build_asset_profile(raw_dir: Path) -> pd.DataFrame:
    shares = aces_asset_shares()
    rows = [
        {
            "asset_component": "building_and_site",
            "purchaser_industry": "2332D0",
            "asset_share": 0.0,
            "excluded": False,
            "source_data_year": 2017,
            "source_price_year": 2017,
            "mapping_status": "resolved",
        }
    ]
    codes = {
        "computer_and_office_equipment": "334111",
        "communications_equipment": "334220",
        "other_machinery_and_equipment": "333318",
        "transportation_equipment": "336111",
    }
    for component, share in shares.items():
        rows.append(
            {
                "asset_component": component,
                "purchaser_industry": codes[component],
                "asset_share": share,
                "excluded": False,
                "source_data_year": 2017,
                "source_price_year": 2017,
                "mapping_status": "resolved",
            }
        )
    for component in ("software", "land", "intellectual_property"):
        rows.append(
            {
                "asset_component": component,
                "purchaser_industry": "excluded",
                "asset_share": 0.0,
                "excluded": True,
                "source_data_year": 2017,
                "source_price_year": 2017,
                "mapping_status": "excluded_by_method",
            }
        )
    return pd.DataFrame(rows)


def _build_margins(supply_detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source in supply_detail.itertuples(index=False):
        shares = [
            (
                str(source.primary_producer_industry),
                max(float(source.basic_share), 0.0),
                False,
                "basic_price",
            ),
            ("423400", max(float(source.trade_share), 0.0), False, "trade_margin"),
            (
                "484000",
                max(float(source.transport_share), 0.0),
                False,
                "transport_margin",
            ),
            (
                str(source.primary_producer_industry),
                max(float(source.import_share), 0.0),
                True,
                "foreign_import",
            ),
            (
                str(source.primary_producer_industry),
                max(float(source.tax_share), 0.0),
                True,
                "product_tax",
            ),
        ]
        total = sum(value for _, value, _, _ in shares)
        if total <= 0:
            continue
        for producer, value, excluded, kind in shares:
            if value <= 0:
                continue
            rows.append(
                {
                    "purchaser_industry": str(source.purchaser_industry),
                    "producer_industry": producer,
                    "margin_share": value / total,
                    "tax_flag": excluded,
                    "margin_type": kind,
                    "source_data_year": 2017,
                    "source_price_year": 2017,
                    "currency": "USD",
                    "price_year": MODEL_PRICE_YEAR,
                    "mapping_status": "resolved",
                }
            )
    return pd.DataFrame(rows)


def _expand_naics(text: object) -> list[str]:
    value = str(text).replace("*", "")
    result: list[str] = []
    for token in re.findall(r"\d{2,6}(?:-\d{1,6})?", value):
        if "-" not in token:
            result.append(token)
            continue
        start, end = token.split("-", 1)
        if len(end) < len(start):
            end = start[: len(start) - len(end)] + end
        if len(start) == len(end) and int(end) - int(start) <= 100:
            result.extend(
                str(number).zfill(len(start))
                for number in range(int(start), int(end) + 1)
            )
    return sorted(set(result))


def _bea_naics_map(use_path: Path) -> dict[str, list[str]]:
    raw = pd.read_excel(use_path, sheet_name="NAICS Codes", header=None)
    result: dict[str, list[str]] = {}
    for row in raw.iloc[5:].itertuples(index=False, name=None):
        detail = row[3] if len(row) > 3 else None
        if pd.isna(detail):
            continue
        code = str(detail).strip()
        related = row[6] if len(row) > 6 else ""
        parsed = _expand_naics(related)
        if not parsed and code.startswith("23"):
            parsed = ["23"]
        result[code] = parsed
    return result


def _read_qcew(path: Path) -> pd.DataFrame:
    columns = [
        "area_fips",
        "own_code",
        "industry_code",
        "qtr",
        "disclosure_code",
        "annual_avg_emplvl",
        "total_annual_wages",
        "avg_annual_pay",
    ]
    frames = []
    with zipfile.ZipFile(path) as archive:
        name = next(name for name in archive.namelist() if name.endswith(".csv"))
        with archive.open(name) as binary:
            text = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
            for chunk in pd.read_csv(
                text,
                usecols=columns,
                dtype={"area_fips": str, "industry_code": str},
                chunksize=200_000,
            ):
                chunk["area_fips"] = (
                    chunk["area_fips"]
                    .str.strip()
                    .str.replace('"', "", regex=False)
                    .str.zfill(5)
                )
                chunk["industry_code"] = chunk["industry_code"].str.strip()
                selected = chunk[
                    chunk["qtr"].astype(str).eq("A")
                    & chunk["own_code"].isin([0, 5])
                    & (
                        chunk["area_fips"].eq("US000")
                        | chunk["area_fips"].str.fullmatch(r"\d{2}000")
                    )
                ]
                frames.append(selected)
    data = pd.concat(frames, ignore_index=True)
    return data.drop_duplicates(["area_fips", "industry_code"])


def _state_crosswalk() -> pd.DataFrame:
    data = pd.read_csv(STATE_FIPS_PATH, dtype=str)
    columns = {column.lower(): column for column in data.columns}
    result = pd.DataFrame(
        {
            "region": data[columns.get("state_code", "state_code")],
            "area_fips": data[columns.get("state_fips", "state_fips")].str.zfill(2)
            + "000",
        }
    ).drop_duplicates()
    return result[~result["region"].isin({"AS", "GU", "MP", "NP", "PR", "VI"})]


def _state_wage_shares(path: Path) -> dict[str, float]:
    with zipfile.ZipFile(path) as archive:
        with archive.open("SAINC4__ALL_AREAS_1929_2025.csv") as handle:
            data = pd.read_csv(handle, dtype={"GeoFIPS": str})
    data["GeoFIPS"] = (
        data["GeoFIPS"].str.replace('"', "", regex=False).str.strip().str.zfill(5)
    )
    values = data[data["LineCode"].isin([50, 60])].pivot(
        index="GeoFIPS", columns="LineCode", values="2024"
    )
    shares = values[50] / (values[50] + values[60])
    return shares.to_dict()


def _state_income_factors(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        candidates = [
            name
            for name in archive.namelist()
            if name.startswith("SAINC50__ALL_AREAS_") and name.endswith(".csv")
        ]
        if len(candidates) != 1:
            raise ValueError(f"Expected one SAINC50 all-areas file, found {candidates}")
        with archive.open(candidates[0]) as handle:
            data = pd.read_csv(handle, dtype={"GeoFIPS": str})
    data["GeoFIPS"] = (
        data["GeoFIPS"].str.replace('"', "", regex=False).str.strip().str.zfill(5)
    )
    data["2024"] = pd.to_numeric(data["2024"], errors="coerce")
    values = data[data["LineCode"].isin([10, 16])].pivot(
        index="GeoFIPS", columns="LineCode", values="2024"
    )
    values = values.rename(columns={10: "personal_income", 16: "disposable_income"})
    values["dpi_to_personal_income"] = (
        values["disposable_income"] / values["personal_income"]
    )
    return values


def _match_qcew(rows: pd.DataFrame, naics: list[str]) -> tuple[float, float, str]:
    if not naics:
        return np.nan, np.nan, "unmapped"
    lengths = sorted({len(code) for code in naics}, reverse=True)
    for length in lengths:
        prefixes = {code for code in naics if len(code) == length}
        selected = rows[
            rows["industry_code"].str.len().eq(6)
            & rows["industry_code"].str[:length].isin(prefixes)
        ]
        if not selected.empty and selected["annual_avg_emplvl"].sum() > 0:
            return (
                float(selected["annual_avg_emplvl"].sum()),
                float(selected["total_annual_wages"].sum()),
                f"mapped_{length}digit",
            )
    return np.nan, np.nan, "suppressed_or_unmapped"


def _qcew_sector(code: str) -> str:
    prefix = str(code)[:2]
    if prefix in {"31", "32", "33"}:
        return "31-33"
    if prefix in {"44", "45"}:
        return "44-45"
    if prefix in {"48", "49"}:
        return "48-49"
    return prefix


def _direct_ops_qcew(rows: pd.DataFrame, national: pd.DataFrame) -> tuple[float, str]:
    for code in ("518210", "5182", "518", "51"):
        selected = rows[rows["industry_code"].eq(code)]
        nat = national[national["industry_code"].eq(code)]
        if not selected.empty and float(selected.iloc[0]["annual_avg_emplvl"]) > 0:
            return float(selected.iloc[0]["avg_annual_pay"]), code
        if not selected.empty and not nat.empty:
            parent_pay = float(selected.iloc[0]["avg_annual_pay"])
            national_parent = float(nat.iloc[0]["avg_annual_pay"])
            national_518210 = national[national["industry_code"].eq("518210")]
            if national_parent > 0 and not national_518210.empty:
                return (
                    float(national_518210.iloc[0]["avg_annual_pay"])
                    * parent_pay
                    / national_parent,
                    code,
                )
    raise ValueError("QCEW fallback chain cannot recover 518210 annual pay")


def _infer_sctg(code: str, title: str) -> str:
    code = str(code)
    title = title.lower()
    if code.startswith("11"):
        return "01-09"
    if code.startswith("21"):
        return "10-19"
    if code.startswith(("311", "312")):
        return "01-09"
    if code.startswith(("313", "314", "315", "316")):
        return "30"
    if code.startswith(("321",)):
        return "25"
    if code.startswith(("322", "323")):
        return "27-29"
    if code.startswith(("324",)):
        return "17"
    if code.startswith(("325",)):
        return "20-23"
    if code.startswith(("326",)):
        return "24"
    if code.startswith(("327",)):
        return "31"
    if code.startswith(("331", "332")):
        return "32-33"
    if code.startswith("333"):
        return "34"
    if code.startswith("334"):
        return "35"
    if code.startswith("335"):
        return "35"
    if code.startswith("336"):
        return "36-37"
    if code.startswith("337"):
        return "39"
    if code.startswith("339") or "manufactur" in title:
        return "40"
    return ""


def _read_faf_lps(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        name = next(
            name for name in archive.namelist() if name.lower().endswith(".csv")
        )
        with archive.open(name) as handle:
            data = pd.read_csv(handle)
    required = {"sctg2", "dms_origst", "dms_destst", "trade_type"}
    if not required.issubset(data.columns):
        raise ValueError(
            f"FAF state database schema changed; found {list(data.columns)}"
        )
    value_columns = [
        column for column in data.columns if re.fullmatch(r"value_2017", column.lower())
    ]
    if len(value_columns) != 1:
        raise ValueError("FAF state database lacks a unique 2017 value column")
    value = value_columns[0]
    data["origin_state"] = (
        pd.to_numeric(data["dms_origst"], errors="coerce")
        .astype("Int64")
        .astype(str)
        .str.zfill(2)
    )
    data["destination_state"] = (
        pd.to_numeric(data["dms_destst"], errors="coerce")
        .astype("Int64")
        .astype(str)
        .str.zfill(2)
    )
    data["sctg2"] = pd.to_numeric(data["sctg2"], errors="coerce").astype("Int64")
    data[value] = pd.to_numeric(data[value], errors="coerce").fillna(0.0)
    states = sorted(set(data["origin_state"]) | set(data["destination_state"]))
    rows = []
    for state in states:
        for sctg, group in data.groupby("sctg2"):
            local = group[
                group["trade_type"].eq(1)
                & group["origin_state"].eq(state)
                & group["destination_state"].eq(state)
            ][value].sum()
            imports = group[
                group["destination_state"].eq(state)
                & (~group["origin_state"].eq(state) | group["trade_type"].eq(2))
            ][value].sum()
            lps = local / (local + imports) if local + imports > 0 else 0.0
            rows.append({"state_fips": state, "sctg2": int(sctg), "faf_lps": lps})
    return pd.DataFrame(rows)


def _sctg_values(specification: str) -> list[int]:
    values = []
    for token in specification.split(","):
        if "-" in token:
            start, end = map(int, token.split("-"))
            values.extend(range(start, end + 1))
        elif token:
            values.append(int(token))
    return values


def _build_regional(
    industries: pd.DataFrame,
    use_path: Path,
    qcew_path: Path,
    sainc_path: Path,
    faf_path: Path,
) -> pd.DataFrame:
    qcew = _read_qcew(qcew_path)
    national = qcew[qcew["area_fips"].eq("US000")]
    crosswalk = _state_crosswalk()
    wage_shares = _state_wage_shares(sainc_path)
    naics_map = _bea_naics_map(use_path)
    faf = _read_faf_lps(faf_path)
    rows = []
    for state in crosswalk.itertuples(index=False):
        state_qcew = qcew[qcew["area_fips"].eq(state.area_fips)]
        total_state_wages = float(
            state_qcew.loc[
                state_qcew["industry_code"].eq("10"), "total_annual_wages"
            ].iloc[0]
        )
        total_us_wages = float(
            national.loc[national["industry_code"].eq("10"), "total_annual_wages"].iloc[
                0
            ]
        )
        wage_share = float(wage_shares[state.area_fips])
        ops_pay, ops_fallback = _direct_ops_qcew(state_qcew, national)
        for industry in industries.itertuples(index=False):
            codes = naics_map.get(str(industry.bea_industry), [])
            state_jobs, state_wages, mapping = _match_qcew(state_qcew, codes)
            us_jobs, us_wages, _ = _match_qcew(national, codes)
            if (
                not np.isfinite(state_wages)
                or not np.isfinite(us_wages)
                or us_wages <= 0
            ):

                prefix = _qcew_sector(
                    next(
                        (code for code in codes if len(code) >= 2),
                        str(industry.bea_industry),
                    )
                )
                parent_state = state_qcew[state_qcew["industry_code"].eq(prefix)]
                parent_us = national[national["industry_code"].eq(prefix)]
                if parent_state.empty or parent_us.empty:
                    parent_state = state_qcew[state_qcew["industry_code"].eq("10")]
                    parent_us = national[national["industry_code"].eq("10")]
                    prefix = "all_industries"
                if parent_state.empty or parent_us.empty:
                    raise ValueError(
                        f"No QCEW wage fallback for {state.region}/{industry.bea_industry}"
                    )
                state_wages = float(parent_state.iloc[0]["total_annual_wages"])
                us_wages = float(parent_us.iloc[0]["total_annual_wages"])
                state_jobs = float(parent_state.iloc[0]["annual_avg_emplvl"])
                us_jobs = float(parent_us.iloc[0]["annual_avg_emplvl"])
                mapping = f"fallback_{prefix}"
            wage_lq = min(
                (state_wages / total_state_wages) / (us_wages / total_us_wages), 1.0
            )
            sctg = _infer_sctg(str(industry.bea_industry), str(industry.industry_title))
            if sctg:
                selected = faf[
                    faf["state_fips"].eq(state.area_fips[:2])
                    & faf["sctg2"].isin(_sctg_values(sctg))
                ]
                if selected.empty:
                    raise ValueError(f"No FAF LPS for {state.region}/{sctg}")
                lps = float(selected["faf_lps"].mean())
                lps_basis = "faf"
            else:
                lps = wage_lq
                lps_basis = "wage_lq"
            if str(industry.bea_industry) == "518200":
                compensation_per_job = ops_pay / wage_share
                fallback = ops_fallback
            else:
                if state_jobs > 0 and state_wages > 0:
                    average_wage = state_wages / state_jobs
                else:
                    state_all = state_qcew[state_qcew["industry_code"].eq("10")].iloc[0]
                    us_all = national[national["industry_code"].eq("10")].iloc[0]
                    state_all_pay = float(state_all["total_annual_wages"]) / float(
                        state_all["annual_avg_emplvl"]
                    )
                    us_all_pay = float(us_all["total_annual_wages"]) / float(
                        us_all["annual_avg_emplvl"]
                    )
                    national_industry_pay = us_wages / max(us_jobs, 1.0)
                    average_wage = national_industry_pay * state_all_pay / us_all_pay
                    mapping = f"{mapping}_national_wage_index"
                compensation_per_job = average_wage / wage_share
                fallback = mapping
            labor_share = float(industry.national_labor_compensation_per_output)
            jobs_per_output = labor_share / max(compensation_per_job, 1e-12)
            rows.append(
                {
                    "region": state.region,
                    "bea_industry": str(industry.bea_industry),
                    "lps": lps,
                    "lps_basis": lps_basis,
                    "faf_sctg": sctg,
                    "wage_lq": wage_lq,
                    "jobs_per_output_2024": jobs_per_output,
                    "earnings_per_output_2024": labor_share,
                    "value_added_per_output_2024": float(
                        industry.national_value_added_per_output_2024
                    ),
                    "wage_per_output_2024": labor_share * wage_share,
                    "qcew_fallback_level": fallback,
                    "source_data_year": 2024,
                    "source_price_year": 2024,
                    "currency": "USD",
                    "price_year": MODEL_PRICE_YEAR,
                    "mapping_status": "resolved",
                }
            )
    return pd.DataFrame(rows)


def _build_pce(
    pce_path: Path,
    sapce_path: Path,
    sainc_path: Path,
    regional: pd.DataFrame,
    margins: pd.DataFrame,
) -> pd.DataFrame:
    bridge = pd.read_excel(pce_path, sheet_name="2017", header=4)
    bridge.columns = [
        "nipa_line",
        "pce_category",
        "commodity",
        "commodity_title",
        "producer_value",
        "transport",
        "wholesale",
        "retail",
        "purchaser_value",
        "blank",
    ][: len(bridge.columns)]
    bridge = bridge[pd.to_numeric(bridge["nipa_line"], errors="coerce").notna()].copy()
    bridge["nipa_line"] = pd.to_numeric(bridge["nipa_line"]).astype(int)
    bridge["commodity"] = bridge["commodity"].astype(str)
    bridge = bridge[~bridge["commodity"].eq("S00300")].copy()
    bridge["category"] = np.select(
        [bridge["nipa_line"].lt(72), bridge["nipa_line"].lt(150)],
        ["durable", "nondurable"],
        default="services",
    )
    for column in (
        "producer_value",
        "transport",
        "wholesale",
        "retail",
        "purchaser_value",
    ):
        bridge[column] = pd.to_numeric(bridge[column], errors="coerce").fillna(0.0)
    margin_primary = margins[margins["margin_type"].eq("basic_price")][
        ["purchaser_industry", "producer_industry"]
    ].drop_duplicates()
    bridge = bridge.merge(
        margin_primary, left_on="commodity", right_on="purchaser_industry", how="left"
    )
    bridge.loc[
        bridge["commodity"].eq("S00402") & bridge["producer_industry"].isna(),
        "producer_industry",
    ] = "4B0000"
    if bridge["producer_industry"].isna().any():
        raise ValueError("PCE bridge has commodities without a BEA producer mapping")
    national_rows = []
    for row in bridge.itertuples(index=False):
        national_rows.extend(
            [
                (
                    row.category,
                    str(row.producer_industry),
                    max(float(row.producer_value), 0.0),
                ),
                (row.category, "484000", max(float(row.transport), 0.0)),
                (row.category, "423400", max(float(row.wholesale), 0.0)),
                (row.category, "4B0000", max(float(row.retail), 0.0)),
            ]
        )
    national = pd.DataFrame(
        national_rows, columns=["category", "bea_industry", "value"]
    )
    national = national.groupby(["category", "bea_industry"], as_index=False)[
        "value"
    ].sum()
    national["within_category_share"] = national["value"] / national.groupby(
        "category"
    )["value"].transform("sum")
    with zipfile.ZipFile(sapce_path) as archive:
        with archive.open("SAPCE1__ALL_AREAS_1997_2024.csv") as handle:
            state_pce = pd.read_csv(handle, dtype={"GeoFIPS": str})
    state_pce["GeoFIPS"] = (
        state_pce["GeoFIPS"].str.replace('"', "", regex=False).str.strip().str.zfill(5)
    )
    state_pce["2024"] = pd.to_numeric(state_pce["2024"], errors="coerce")
    total_pce = state_pce[state_pce["LineCode"].eq(1)].set_index("GeoFIPS")["2024"]
    category_lines = {3: "durable", 8: "nondurable", 13: "services"}
    state_pce = state_pce[state_pce["LineCode"].isin(category_lines)].copy()
    state_pce["category"] = state_pce["LineCode"].map(category_lines)
    state_pce["category_share"] = state_pce["2024"] / state_pce.groupby("GeoFIPS")[
        "2024"
    ].transform("sum")
    state_lookup = _state_crosswalk().set_index("area_fips")["region"].to_dict()
    wage_shares = _state_wage_shares(sainc_path)
    income_factors = _state_income_factors(sainc_path)
    result = []
    for geo, categories in state_pce.groupby("GeoFIPS"):
        region = state_lookup.get(geo)
        if not region:
            continue
        category_weights = categories.set_index("category")["category_share"]
        state_rows = national.copy()
        state_rows["producer_share"] = state_rows["within_category_share"] * state_rows[
            "category"
        ].map(category_weights)
        lps = regional[regional["region"].eq(region)].set_index("bea_industry")["lps"]
        state_rows["producer_share"] *= state_rows["bea_industry"].map(lps).fillna(0.0)
        state_rows = state_rows.groupby("bea_industry", as_index=False)[
            "producer_share"
        ].sum()
        state_rows["region"] = region
        wage_share = float(wage_shares[geo])
        dpi_to_income = float(income_factors.loc[geo, "dpi_to_personal_income"])

        pce_to_dpi = (
            float(total_pce.loc[geo])
            * 1_000.0
            / float(income_factors.loc[geo, "disposable_income"])
        )
        eta = wage_share * dpi_to_income * pce_to_dpi
        if not (0 < wage_share <= 1 and 0 < dpi_to_income <= 1):
            raise ValueError(f"Invalid wage/income factor for {region}")
        if not (0 < pce_to_dpi <= 1.25 and 0 < eta <= 1):
            raise ValueError(f"Invalid household consumption factor for {region}")
        state_rows["wage_salary_share"] = wage_share
        state_rows["wage_share"] = wage_share
        state_rows["dpi_to_personal_income"] = dpi_to_income
        state_rows["pce_to_dpi"] = pce_to_dpi
        state_rows["employee_compensation_to_consumption"] = eta
        state_rows["negative_household_saving"] = pce_to_dpi > 1.0
        state_rows["household_factor_fallback"] = "none"
        state_rows["source_data_year"] = 2024
        state_rows["source_price_year"] = 2024
        state_rows["currency"] = "USD"
        state_rows["price_year"] = 2024
        state_rows["mapping_status"] = "resolved"
        result.append(state_rows)
    return pd.concat(result, ignore_index=True)


def build_normalized(manifest: dict) -> None:
    raw_dir, output_dir = _paths(manifest)
    required_ids = {
        "bea_supply_2017",
        "bea_use_2017",
        "bea_pce_bridge_2017",
        "bea_state_income_2024",
        "bea_state_pce_2024",
        "qcew_2024",
        "aces_2017_table7a",
        "aces_2017_table8a",
        "faf5_2017_state",
    }
    verify_downloads(manifest, required_ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    industries, matrix, supply_detail, operating = build_domestic_requirements(
        raw_dir / manifest["sources"]["bea_use_2017"]["filename"],
        raw_dir / manifest["sources"]["bea_supply_2017"]["filename"],
    )

    regional = _build_regional(
        industries,
        raw_dir / manifest["sources"]["bea_use_2017"]["filename"],
        raw_dir / manifest["sources"]["qcew_2024"]["filename"],
        raw_dir / manifest["sources"]["bea_state_income_2024"]["filename"],
        raw_dir / manifest["sources"]["faf5_2017_state"]["filename"],
    )
    national = regional.groupby("bea_industry", as_index=True).agg(
        national_jobs_per_output_2024=("jobs_per_output_2024", "mean"),
        national_earnings_per_output_2024=("earnings_per_output_2024", "mean"),
        national_wage_per_output_2024=("wage_per_output_2024", "mean"),
    )
    industries = industries.merge(
        national, on="bea_industry", how="left", validate="one_to_one"
    )
    industries["source_data_year"] = "2017 production; 2024 satellite"
    industries["source_price_year"] = 2024
    industries["currency"] = "USD"
    industries["price_year"] = 2024
    industries = _attach_provenance(
        industries, manifest, ["bea_use_2017", "bea_supply_2017", "qcew_2024"]
    )
    margins = _attach_provenance(
        _build_margins(supply_detail), manifest, ["bea_supply_2017"]
    )
    assets = _attach_provenance(
        _build_asset_profile(raw_dir),
        manifest,
        ["aces_2017_table7a", "aces_2017_table8a"],
    )
    pce = _build_pce(
        raw_dir / manifest["sources"]["bea_pce_bridge_2017"]["filename"],
        raw_dir / manifest["sources"]["bea_state_pce_2024"]["filename"],
        raw_dir / manifest["sources"]["bea_state_income_2024"]["filename"],
        regional,
        margins,
    )
    regional["fallback_status"] = regional["qcew_fallback_level"]
    regional = _attach_provenance(
        regional,
        manifest,
        ["bea_use_2017", "qcew_2024", "bea_state_income_2024", "faf5_2017_state"],
    )
    pce = _attach_provenance(
        pce,
        manifest,
        ["bea_pce_bridge_2017", "bea_state_pce_2024", "bea_state_income_2024"],
    )
    operating["source_data_year"] = 2017
    operating["source_price_year"] = 2017
    operating["currency"] = "USD"
    operating["price_year"] = 2024
    operating["mapping_status"] = "resolved"
    operating = _attach_provenance(operating, manifest, ["bea_use_2017"])
    temporary = output_dir.with_name(output_dir.name + ".building")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    industries.to_csv(temporary / "bea_industries.csv", index=False)
    np.savez_compressed(temporary / "bea_domestic_requirements_2017.npz", a=matrix)
    regional.to_csv(temporary / "us_regional_coefficients_2024.csv", index=False)
    pce.to_csv(temporary / "us_state_pce_producer_shares_2024.csv", index=False)
    assets.to_csv(temporary / "us_aces_asset_profile.csv", index=False)
    operating.to_csv(temporary / "us_datacenter_operating_inputs.csv", index=False)
    margins.to_csv(temporary / "us_purchaser_margins_2017.csv", index=False)
    (temporary / "build_metadata.json").write_text(
        json.dumps(
            {
                "economic_method": "public_bea_lq_io",
                "economic_method_version": ECONOMIC_METHOD_VERSION,
                "currency": "USD",
                "price_year": 2024,
                "manifest_snapshot_date": str(manifest["snapshot_date"]),
                "bea_api_key_used": False,
                "detailed_industries": len(industries),
                "sources": {
                    source_id: {
                        "url": source["url"],
                        "sha256": source["sha256"],
                        "source_data_year": source["source_data_year"],
                        "source_price_year": source["source_price_year"],
                    }
                    for source_id, source in manifest["sources"].items()
                    if source_id in required_ids
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    PublicBEALQIO.load(temporary)
    if output_dir.exists():
        backup = output_dir.with_name(output_dir.name + ".previous")
        if backup.exists():
            shutil.rmtree(backup)
        os.replace(output_dir, backup)
        os.replace(temporary, output_dir)
        shutil.rmtree(backup)
    else:
        os.replace(temporary, output_dir)
    build_canada_normalized(manifest)
    build_harmonized_aidc_responses(output_dir)


def build_harmonized_aidc_responses(output_dir: Path) -> None:
    model = PublicBEALQIO.load(output_dir)
    rows: list[dict[str, object]] = []
    for region in sorted(model.regional["region"].astype(str).unique()):
        matrices = model.regional_matrices(region)
        construction_code = str(
            model.assets.loc[
                model.assets["asset_component"].eq("building_and_site"), "purchaser_industry"
            ].iloc[0]
        )
        construction_idx = model.index[construction_code]
        demand = 1.0
        type_i = matrices.type_i[:, construction_idx] * demand
        type_ii = matrices.type_ii[:, construction_idx] * demand
        direct = np.zeros_like(type_i)
        direct[construction_idx] = demand
        effect_outputs = {
            "direct": direct,
            "indirect": type_i - direct,
            "induced": type_ii - type_i,
        }
        for phase, outputs in {"construction": effect_outputs}.items():
            for effect, vector in outputs.items():
                rows.append({
                    "region": region, "country": "USA", "phase": phase, "effect": effect,
                    "jobs_per_2024_usd": float(vector @ matrices.jobs_per_output),
                    "earnings_usd_per_2024_usd": float(vector @ matrices.earnings_per_output),
                    "value_added_usd_per_2024_usd": float(vector @ matrices.value_added_per_output),
                    "source_data_vintages": "BEA 2017; QCEW/BEA regional 2024; FAF 2017",
                })

        op_idx = model.index["518200"]
        type_i = matrices.type_i[:, op_idx]
        type_ii = matrices.type_ii[:, op_idx]
        direct = np.zeros_like(type_i)
        direct[op_idx] = 1.0
        for effect, vector in {
            "direct": direct, "indirect": type_i - direct, "induced": type_ii - type_i,
        }.items():
            rows.append({
                "region": region, "country": "USA", "phase": "operations", "effect": effect,
                "jobs_per_2024_usd": float(vector @ matrices.jobs_per_output),
                "earnings_usd_per_2024_usd": float(vector @ matrices.earnings_per_output),
                "value_added_usd_per_2024_usd": float(vector @ matrices.value_added_per_output),
                "source_data_vintages": "BEA 2017; QCEW/BEA regional 2024; FAF 2017",
            })
    shared = REPO_ROOT / "employment" / "shared_data"
    canada = pd.read_csv(shared / "jedi_province_io_multipliers_can.csv")
    for phase, industry in (("construction", "datacenter_construction"), ("operations", "datacenter_operations")):
        subset = canada.loc[canada["industry"].eq(industry)]
        for item in subset.itertuples(index=False):
            for effect in ("direct", "indirect", "induced"):
                rows.append({
                    "region": str(item.region), "country": "CAN", "phase": phase, "effect": effect,
                    "jobs_per_2024_usd": float(getattr(item, f"{effect}_mult")) / 1_000_000.0,
                    "earnings_usd_per_2024_usd": float(getattr(item, f"{effect}_earnings_mult")),
                    "value_added_usd_per_2024_usd": float(getattr(item, f"{effect}_value_added_mult")),
                    "source_data_vintages": "Statistics Canada 36-10-0595-01 (2022)",
                })
    response = pd.DataFrame(rows)
    expected = {"region", "country", "phase", "effect", "jobs_per_2024_usd", "earnings_usd_per_2024_usd", "value_added_usd_per_2024_usd", "source_data_vintages"}
    if set(response.columns) != expected or response.duplicated(["region", "phase", "effect"]).any():
        raise ValueError("Invalid harmonized AIDC response table")
    if (response[["jobs_per_2024_usd", "earnings_usd_per_2024_usd", "value_added_usd_per_2024_usd"]] < 0).any().any():
        raise ValueError("Negative harmonized AIDC response coefficient")
    response.to_csv(output_dir / "harmonized_aidc_regional_responses.csv", index=False)


def verify_normalized(manifest: dict) -> None:
    _, output_dir = _paths(manifest)
    model = PublicBEALQIO.load(output_dir)
    if len(model.industries) != 402:
        raise ValueError(
            f"Expected 402 detailed BEA industries, got {len(model.industries)}"
        )
    for region in sorted(model.regional["region"].unique()):
        matrices = model.regional_matrices(region)
        if (
            matrices.spectral_radius_type_i >= 1
            or matrices.spectral_radius_type_ii >= 1
        ):
            raise ValueError(f"Unstable Leontief inverse for {region}")
    shared = REPO_ROOT / "employment" / "shared_data"
    canada = CanadaRegionalData.load()
    canada_frames = {
        "canada_regional_spend_profiles.csv": canada.profiles,
        "canada_product_local_shares.csv": canada.local_shares,
        "canada_product_margin_factors.csv": canada.margins,
        "canada_payroll_parameters.csv": canada.payroll,
        "jedi_province_io_multipliers_can.csv": pd.read_csv(
            shared / "jedi_province_io_multipliers_can.csv"
        ),
    }
    provenance = {
        "source_data_year",
        "source_price_year",
        "currency",
        "price_year",
        "download_url",
        "sha256",
        "fallback_status",
        "mapping_status",
    }
    for name, frame in canada_frames.items():
        missing = provenance - set(frame.columns)
        if missing:
            raise ValueError(
                f"{name} lacks normalized provenance columns: {sorted(missing)}"
            )
        if frame["mapping_status"].astype(str).ne("resolved").any():
            raise ValueError(f"{name} contains unresolved main-result mappings")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "build", "verify"])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--lock-missing-hashes", action="store_true")
    args = parser.parse_args(argv)
    manifest = _load_manifest(args.manifest)
    selected = set(args.source) or None
    if args.command == "download":
        download_sources(
            manifest,
            selected,
            force=args.force,
            offline=args.offline,
            lock_missing_hashes=args.lock_missing_hashes,
        )
    elif args.command == "build":
        build_normalized(manifest)
    else:
        verify_downloads(manifest, selected)
        verify_normalized(manifest)


if __name__ == "__main__":
    main(sys.argv[1:])
