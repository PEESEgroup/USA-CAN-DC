from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .paths import AIDC_MODULE


def run_aidc_case(
    case_dir: Path,
    output_dir: Path,
    assumptions_file: Path | None = None,
    construction_method: str = "lifecycle_intensity",
    dollar_per_mw_construction: float | None = None,
) -> subprocess.CompletedProcess[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        AIDC_MODULE,
        "--case-dir",
        str(case_dir),
        "--output-dir",
        str(output_dir),
        "--construction-method",
        construction_method,
    ]
    if assumptions_file is not None:
        cmd.extend(["--assumptions-file", str(assumptions_file)])
    if dollar_per_mw_construction is not None:
        cmd.extend(["--dollar-per-mw-construction", str(dollar_per_mw_construction)])
    return subprocess.run(cmd, capture_output=True, text=True, check=True)
