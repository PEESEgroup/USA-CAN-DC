from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import xlwings as xw


REPO_ROOT = Path(__file__).resolve().parents[2]
EMPLOYMENT_ROOT = Path(__file__).resolve().parents[1]
OFFWIND_CLIENT_DIR = (
    EMPLOYMENT_ROOT / "energy_system" / "jedi_us" / "jedi-offwind-model"
)


OFFSHORE_DEFAULT_LOCAL_SHARE_ROWS = {
    19: 6,
    20: 7,
    22: 9,
    23: 10,
    25: 12,
    26: 13,
    27: 14,
    30: 17,
    33: 20,
    36: 23,
    39: 26,
    42: 29,
    46: 33,
    49: 36,
    52: 39,
    57: 44,
    58: 45,
    60: 47,
    61: 48,
    63: 50,
    64: 51,
    66: 53,
    67: 54,
    69: 56,
    70: 57,
    72: 59,
    73: 60,
    75: 62,
    76: 63,
    78: 65,
    79: 66,
    80: 67,
    81: 68,
    82: 69,
    83: 70,
    84: 71,
    86: 75,
    87: 76,
    88: 77,
    89: 78,
    90: 79,
    91: 80,
    94: 82,
    97: 85,
    98: 86,
    99: 87,
    100: 88,
    101: 89,
    102: 90,
    109: 99,
    110: 100,
    111: 101,
    112: 102,
    114: 104,
    115: 105,
    116: 107,
    117: 108,
    118: 109,
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run offshore wind JEDI workbook without calling the Excel VBA macro."
    )
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--analysis-area", required=True)
    parser.add_argument("--project-area", required=True)
    parser.add_argument("--construction-year", type=int, required=True)
    parser.add_argument("--money-year", type=int, required=True)
    parser.add_argument("--data-library", type=Path, default=None)
    parser.add_argument("--save-workbook", action="store_true")
    return parser


def create_excel_app() -> xw.main.App:
    app = xw.App(visible=False, add_book=False)
    app.display_alerts = False
    app.screen_updating = False
    app.api.Visible = False
    app.api.DisplayAlerts = False
    app.api.DisplayStatusBar = False
    app.api.EnableEvents = False
    app.api.ScreenUpdating = False
    app.api.AskToUpdateLinks = False
    return app


def apply_offshore_default_local_shares(book: xw.main.Book) -> None:
    default_sheet = book.sheets["Default Local"]
    local_sheet = book.sheets["Local Share - Step 3"]
    for target_row, source_row in OFFSHORE_DEFAULT_LOCAL_SHARE_ROWS.items():
        local_sheet.range(f"C{target_row}").value = default_sheet.range(
            f"D{source_row}"
        ).value


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if str(OFFWIND_CLIENT_DIR) not in sys.path:
        sys.path.insert(0, str(OFFWIND_CLIENT_DIR))

    if args.data_library is not None:
        os.environ["DATA_LIBRARY"] = str(args.data_library)
    elif "DATA_LIBRARY" not in os.environ:
        default_library = Path(sys.executable).resolve().parent / "data"
        os.environ["DATA_LIBRARY"] = str(default_library)

    import jedi_orbit_client

    app = create_excel_app()
    book = None
    try:
        print(f"[orbit-direct] workbook={args.workbook}", flush=True)
        print(f"[orbit-direct] data_library={os.environ['DATA_LIBRARY']}", flush=True)
        book = app.books.open(str(args.workbook))
        step1 = book.sheets["Project Data - Step 1"]
        step1.range("D19").value = args.analysis_area
        step1.range("D20").value = args.project_area
        step1.range("D21").value = int(args.construction_year)
        step1.range("D22").value = int(args.money_year)
        app.api.CalculateFull()
        print(
            f"[orbit-direct] inputs={step1.range('D19:D22').value}",
            flush=True,
        )

        book.set_mock_caller()
        print("[orbit-direct] calling jedi_orbit_client.main()", flush=True)
        jedi_orbit_client.main()
        print("[orbit-direct] ORBIT run completed", flush=True)

        apply_offshore_default_local_shares(book)
        app.api.CalculateFull()
        result_sheet = book.sheets["Economic Impact Results"]
        construction_breakdown = {
            "direct": result_sheet.range("D42:G42").value,
            "indirect": result_sheet.range("D53:G53").value,
            "induced": result_sheet.range("D52:G52").value,
            "total": result_sheet.range("D54:G54").value,
        }
        operating_breakdown = {
            "direct": result_sheet.range("D59:G59").value,
            "indirect": result_sheet.range("D60:G60").value,
            "induced": result_sheet.range("D61:G61").value,
            "total": result_sheet.range("D62:G62").value,
        }
        print(
            json.dumps(
                {
                    "construction_breakdown": construction_breakdown,
                    "operating_breakdown": operating_breakdown,
                    "workbook": str(args.workbook),
                    "analysis_area": args.analysis_area,
                    "project_area": args.project_area,
                }
            ),
            flush=True,
        )
        if args.save_workbook:
            book.save()
    finally:
        if book is not None:
            try:
                book.close()
            except Exception:
                pass
        try:
            app.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
