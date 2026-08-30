import argparse
import json
import sys
from pathlib import Path

import xlwings as xw


REPO_ROOT = Path(__file__).resolve().parents[2]
EMPLOYMENT_ROOT = Path(__file__).resolve().parents[1]
ONWIND_CLIENT_DIR = (
    EMPLOYMENT_ROOT / "energy_system" / "jedi_us" / "jedi-onwind-model"
)
LARGE_WIND_PROXY_CAP_MW = 1000.0
IMPACT_EFFECTS = ("direct", "indirect", "induced", "total")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run onshore wind JEDI workbook without calling the Excel VBA macro."
    )
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--state-caps")
    parser.add_argument("--nameplate-value", type=float)
    parser.add_argument("--construction-year", type=int)
    parser.add_argument("--money-year", type=int)
    parser.add_argument("--batch-input", type=Path)
    parser.add_argument("--save-workbook", action="store_true")
    return parser


def create_excel_app():
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


def load_jedi_client():
    if str(ONWIND_CLIENT_DIR) not in sys.path:
        sys.path.insert(0, str(ONWIND_CLIENT_DIR))
    import jedi_client

    return jedi_client


def write_inputs(book, event):
    sheet = book.sheets["STEP 1 - Project Information"]
    sheet.range("C6").value = event["state_caps"]
    sheet.range("C8").value = float(event["nameplate_value"])
    sheet.range("C20").value = int(event["construction_year"])
    sheet.range("C21").value = int(event["money_year"])


def extract_outputs(book):
    sheet = book.sheets["SummaryResults"]
    construction_breakdown = {
        "direct": sheet.range("B28:E28").value,
        "indirect": sheet.range("B29:E29").value,
        "induced": sheet.range("B30:E30").value,
        "total": sheet.range("B31:E31").value,
    }
    operating_breakdown = {
        "direct": sheet.range("B34:E34").value,
        "indirect": sheet.range("B35:E35").value,
        "induced": sheet.range("B36:E36").value,
        "total": sheet.range("B37:E37").value,
    }
    return construction_breakdown, operating_breakdown


def run_event(book, app, jedi_client, event):
    write_inputs(book, event)
    app.api.CalculateFull()
    print(
        "[landbosse-direct] inputs="
        f"{event['state_caps']},{event['nameplate_value']},{event['construction_year']},{event['money_year']}",
        flush=True,
    )
    book.set_mock_caller()
    jedi_client.run()
    app.api.CalculateFull()
    construction_breakdown, operating_breakdown = extract_outputs(book)
    return {
        "construction_breakdown": construction_breakdown,
        "operating_breakdown": operating_breakdown,
    }


def run_event_with_fallback(book, app, jedi_client, event):
    if float(event["nameplate_value"]) > LARGE_WIND_PROXY_CAP_MW:
        fallback_event = dict(event)
        fallback_event["nameplate_value"] = LARGE_WIND_PROXY_CAP_MW
        payload = run_event(book, app, jedi_client, fallback_event)
        payload["fallback_proxy_cap_mw"] = LARGE_WIND_PROXY_CAP_MW
        payload["project_scale_factor_override"] = float(
            event.get("project_scale_factor", 1.0)
        ) * (float(event["nameplate_value"]) / LARGE_WIND_PROXY_CAP_MW)
        payload["fallback_trigger_error"] = "proactive_large_event_proxy"
        return payload
    try:
        payload = run_event(book, app, jedi_client, event)
        payload["fallback_proxy_cap_mw"] = None
        payload["project_scale_factor_override"] = None
        return payload
    except Exception as exc:
        if float(event["nameplate_value"]) <= LARGE_WIND_PROXY_CAP_MW:
            raise
        fallback_event = dict(event)
        fallback_event["nameplate_value"] = LARGE_WIND_PROXY_CAP_MW
        payload = run_event(book, app, jedi_client, fallback_event)
        payload["fallback_proxy_cap_mw"] = LARGE_WIND_PROXY_CAP_MW
        payload["project_scale_factor_override"] = float(
            event.get("project_scale_factor", 1.0)
        ) * (float(event["nameplate_value"]) / LARGE_WIND_PROXY_CAP_MW)
        payload["fallback_trigger_error"] = repr(exc)
        return payload


def validate_single_event_args(args):
    required = [
        ("state_caps", args.state_caps),
        ("nameplate_value", args.nameplate_value),
        ("construction_year", args.construction_year),
        ("money_year", args.money_year),
    ]
    missing = [name for name, value in required if value is None]
    if missing:
        raise ValueError(f"Missing required single-event arguments: {missing}")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.batch_input is None:
        validate_single_event_args(args)

    jedi_client = load_jedi_client()

    app = create_excel_app()
    book = None
    try:
        print(f"[landbosse-direct] workbook={args.workbook}", flush=True)
        book = app.books.open(str(args.workbook))
        if args.batch_input is None:
            event = {
                "state_caps": args.state_caps,
                "nameplate_value": float(args.nameplate_value),
                "construction_year": int(args.construction_year),
                "money_year": int(args.money_year),
                "project_scale_factor": 1.0,
            }
            payload = run_event_with_fallback(book, app, jedi_client, event)
            payload["workbook"] = str(args.workbook)
            payload["state_caps"] = args.state_caps
            print(json.dumps(payload), flush=True)
        else:
            try:
                events = json.loads(args.batch_input.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Failed to read/parse batch input {args.batch_input}: {exc}"
                ) from exc
            results = []
            for event in events:
                event_payload = {
                    "event_id": event["event_id"],
                }
                try:
                    payload = run_event_with_fallback(book, app, jedi_client, event)
                    event_payload.update(payload)
                except Exception as exc:
                    event_payload["error"] = repr(exc)
                results.append(event_payload)
            print(
                json.dumps({"results": results, "workbook": str(args.workbook)}),
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
