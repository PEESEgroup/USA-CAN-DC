from __future__ import annotations

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
RESULTS_ROOT = PACKAGE_DIR / "results"

JEDI_ROOT = PACKAGE_DIR / "energy_system"
AIDC_ROOT = PACKAGE_DIR / "AIDC_campus" / "datacenter_employment"

JEDI_MAIN = JEDI_ROOT / "case_construction_jedi.py"
JEDI_MONITOR = JEDI_ROOT / "excel_popup_monitor.ps1"
AIDC_MODULE = "employment.AIDC_campus.datacenter_employment.run_case_analysis"


def default_case_root(case_name: str) -> Path:
    return RESULTS_ROOT / case_name


def default_aidc_output_dir(case_name: str) -> Path:
    return default_case_root(case_name) / "aidc"


def default_jedi_output_dir(case_name: str) -> Path:
    return default_case_root(case_name) / "jedi"


