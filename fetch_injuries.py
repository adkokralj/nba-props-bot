"""Downloads the official NBA injury report and saves injuries.csv.

One-time setup:  1) install Java (https://www.java.com)   2) pip3 install nbainjuries
Run (each day, before run.py):  python3 fetch_injuries.py
"""
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from fetch_lines import NBA_TEAMS


def java_ok() -> bool:
    if not shutil.which("java"):
        return False
    try:   # macOS has a stub "java" that exists even when Java isn't installed
        return subprocess.run(["java", "-version"], capture_output=True, timeout=20).returncode == 0
    except Exception:
        return False


def to_first_last(name: str) -> str:
    """'Brown, Jaylen' -> 'Jaylen Brown'."""
    name = str(name).strip()
    if "," in name:
        last, first = [x.strip() for x in name.split(",", 1)]
        return f"{first} {last}"
    return name


def clean_report(df: pd.DataFrame, stamp) -> pd.DataFrame:
    df = df.copy()
    for col in ("Game Date", "Game Time", "Matchup", "Team"):
        if col in df.columns:
            df[col] = df[col].ffill()      # the PDF only prints these once per block
    df = df[df["Player Name"].notna()]
    df = df[~df["Player Name"].astype(str).str.contains("NOT YET SUBMITTED", case=False)]
    out = pd.DataFrame({
        "player": df["Player Name"].map(to_first_last),
        "team": df["Team"].astype(str).str.lower().map(NBA_TEAMS).fillna(df["Team"]),
        "status": df["Current Status"].astype(str).str.strip(),
        "reason": df.get("Reason", ""),
        "game_date": df.get("Game Date", ""),
        "report_time_et": stamp.strftime("%Y-%m-%d %H:%M"),
    })
    return out.reset_index(drop=True)


LAST_ERROR = []


def latest_report():
    from nbainjuries import injury
    now = datetime.now(ZoneInfo("America/New_York")).replace(tzinfo=None, second=0, microsecond=0)
    t = now.replace(minute=(now.minute // 15) * 15)
    # newest 15-minute snapshots first, then hourly snapshots further back
    stamps = [t - timedelta(minutes=15 * i) for i in range(8)]
    stamps += [t.replace(minute=0) - timedelta(hours=h) for h in range(2, 37)]
    errors = []
    for ts in stamps:
        try:
            df = injury.get_reportdata(ts, return_df=True)
        except Exception as e:
            if not errors:
                import traceback
                errors.append(traceback.format_exc())
            print(".", end="", flush=True)
            continue
        if df is not None and len(df):
            print()
            return df, ts
        print(".", end="", flush=True)
    LAST_ERROR.extend(errors)
    return None, None


def main():
    if not java_ok():
        sys.exit("Java isn't installed (or isn't working). Install it from java.com, then "
                 "close and reopen Terminal and try again.")
    try:
        import nbainjuries  # noqa: F401
    except ImportError:
        sys.exit("Missing add-on. Run:  pip3 install nbainjuries   (add --break-system-packages "
                 "if it complains), then try again.")
    except Exception as e:   # e.g. Java found but not startable
        sys.exit(f"The injury add-on could not start Java: {e}\n"
                 "Make sure Java is installed, then close and reopen this window.")
    print("Looking for the latest official NBA injury report", end="", flush=True)
    raw, stamp = latest_report()
    if raw is None:
        sys.exit("\nCouldn't read a recent injury report. First error seen:\n" + LAST_ERROR[-1][-1500:]
                 if LAST_ERROR else "\nNo recent injury report found.")
    df = clean_report(raw, stamp)
    df.to_csv("injuries.csv", index=False)
    print(f"Report from {stamp:%Y-%m-%d %H:%M} ET saved to injuries.csv "
          f"({len(df)} players listed).")
    print(df["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
