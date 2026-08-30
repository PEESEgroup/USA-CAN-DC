from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .paths import JEDI_MONITOR


@dataclass(frozen=True)
class JediRunArtifacts:
    command: list[str]
    monitor_command: list[str] | None
    monitor_pid: int | None
    monitor_log: str | None
    output_dir: str
    staged_output_dir: str | None
    status: str
    stdout_tail: str
    stderr_tail: str


def run_jedi_case(
    case_dir: Path,
    output_dir: Path,
    nameplate_scenario: str = "mid",
    money_year: int = 2024,
    max_events: int | None = None,
    coverage_only: bool = False,
    offshore_direct_runtime: bool = True,
    enable_popup_monitor: bool = True,
    static_workers: int | None = None,
    macro_workers: int | None = None,
    include_transmission: bool = True,
) -> JediRunArtifacts:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f"{output_dir.name}_tmp_", dir=str(output_dir.parent))
    )

    monitor_proc = None
    monitor_command = None
    monitor_log = staging_dir / "excel_popup_monitor.log"
    if enable_popup_monitor and not coverage_only:
        monitor_command = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(JEDI_MONITOR),
            "-LogPath",
            str(monitor_log),
            "-PollSeconds",
            "0.5",
        ]
        monitor_proc = subprocess.Popen(
            monitor_command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    cmd = [
        sys.executable,
        "-m",
        "employment.energy_system.case_construction_jedi",
        "--case-dir",
        str(case_dir),
        "--output-dir",
        str(staging_dir),
        "--nameplate-scenario",
        nameplate_scenario,
        "--money-year",
        str(money_year),
    ]
    if static_workers is not None:
        cmd.extend(["--static-workers", str(static_workers)])
    if macro_workers is not None:
        cmd.extend(["--macro-workers", str(macro_workers)])
    if max_events is not None:
        cmd.extend(["--max-events", str(max_events)])
    if coverage_only:
        cmd.append("--coverage-only")
    if offshore_direct_runtime:
        cmd.append("--offshore-direct-runtime")
    cmd.append(
        "--include-transmission"
        if include_transmission
        else "--no-include-transmission"
    )

    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
        status = "success"
    except subprocess.CalledProcessError as exc:
        completed = exc
        status = "failed"
    finally:
        if monitor_proc is not None:
            monitor_proc.terminate()
            try:
                monitor_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                monitor_proc.kill()
                monitor_proc.wait(timeout=10)

    if status == "success":
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.move(str(staging_dir), str(output_dir))
        staged_output_dir = None
        monitor_log = output_dir / monitor_log.name
    else:
        staged_output_dir = str(staging_dir)

    return JediRunArtifacts(
        command=cmd,
        monitor_command=monitor_command,
        monitor_pid=monitor_proc.pid if monitor_proc is not None else None,
        monitor_log=str(monitor_log) if monitor_command is not None else None,
        output_dir=str(output_dir),
        staged_output_dir=staged_output_dir,
        status=status,
        stdout_tail=(completed.stdout or "")[-4000:],
        stderr_tail=(completed.stderr or "")[-4000:],
    )
