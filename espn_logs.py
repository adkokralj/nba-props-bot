"""Game logs from ESPN's public JSON (fallback when stats.nba.com is unreachable, e.g. from GitHub).
Builds the same table as data.load_game_logs_nba_api and keeps a CSV cache so each day
only the new games are downloaded."""
import datetime as dt
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
CACHE = "logs_cache.csv"
START = dt.date(2025, 10, 20)            # start of the 2025-26 season
FIX = {"GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS", "PHO": "PHX"}
_H = {"User-Agent": "Mozilla/5.0"}


def _get(url, params=None, tries=3):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=_H, timeout=25)
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == tries - 1:
                raise
    return {}


def _ab(x):
    return FIX.get(x, x)


def _pair(s):
    """'5-11' -> (5, 11)"""
    try:
        a, b = str(s).split("-")
        return int(a), int(b)
    except Exception:
        return 0, 0


def parse_summary(js, game_date):
    """One ESPN game summary -> list of per-player rows."""
    comp = js["header"]["competitions"][0]
    side = {}
    for c in comp["competitors"]:
        side[_ab(c["team"]["abbreviation"])] = c["homeAway"] == "home"
    teams = list(side)
    if len(teams) != 2 or not comp.get("status", {}).get("type", {}).get("completed", False):
        return []
    gid = str(js["header"]["id"])
    rows = []
    for tb in js.get("boxscore", {}).get("players", []):
        team = _ab(tb["team"]["abbreviation"])
        opp = [t for t in teams if t != team]
        if not opp:
            continue
        for grp in tb.get("statistics", []):
            labels = [l.upper() for l in grp.get("labels", [])]
            for ath in grp.get("athletes", []):
                st = ath.get("stats") or []
                if ath.get("didNotPlay") or len(st) != len(labels):
                    continue
                d = dict(zip(labels, st))
                try:
                    mins = float(str(d.get("MIN", "0")).split(":")[0] or 0)
                except Exception:
                    mins = 0.0
                if mins <= 0:
                    continue
                fgm, fga = _pair(d.get("FG", "0-0"))
                tpm, tpa = _pair(d.get("3PT", "0-0"))
                _, fta = _pair(d.get("FT", "0-0"))
                gi = lambda k: int(float(d.get(k, 0) or 0))
                rows.append({
                    "player": ath["athlete"]["displayName"], "team": team, "game_id": gid,
                    "game_date": game_date, "home": side[team], "minutes": mins,
                    "pts": gi("PTS"), "reb": gi("REB"), "ast": gi("AST"), "fg3m": tpm,
                    "fg3a": tpa, "fgm": fgm, "fga": fga, "fta": fta, "tov": gi("TO"),
                    "oreb": gi("OREB"), "opp": opp[0],
                })
    return rows


def _day(d):
    try:
        js = _get(f"{BASE}/scoreboard", {"dates": f"{d:%Y%m%d}"})
    except Exception as ex:
        print(f"[espn] scoreboard {d} failed: {type(ex).__name__}")
        return []
    out = []
    for ev in js.get("events", []):
        if ev.get("status", {}).get("type", {}).get("completed"):
            t = pd.Timestamp(ev["date"]).tz_convert(None) - pd.Timedelta(hours=8)  # US date
            out.append((str(ev["id"]), t.normalize()))
    return out


def _events(d0, d1):
    days = [d0 + dt.timedelta(days=i) for i in range((d1 - d0).days + 1)]
    with ThreadPoolExecutor(10) as ex:
        return [e for part in ex.map(_day, days) for e in part]


def load_game_logs_espn(cache=CACHE):
    old = pd.read_csv(cache, parse_dates=["game_date"]) if os.path.exists(cache) else pd.DataFrame()
    have = set(old["game_id"].astype(str)) if len(old) else set()
    last = old["game_date"].max().date() - dt.timedelta(days=2) if len(old) else START
    today = dt.date.today()
    evs = [(i, t) for i, t in _events(max(last, START), today) if i not in have]
    print(f"[espn] {len(evs)} new games to download (cache has {len(have)})")

    def one(e):
        try:
            return parse_summary(_get(f"{BASE}/summary", {"event": e[0]}), e[1])
        except Exception as ex:
            print(f"[espn] game {e[0]} failed: {type(ex).__name__}")
            return []

    with ThreadPoolExecutor(10) as ex:
        new = [r for rows in ex.map(one, evs) for r in rows]
    df = pd.concat([old, pd.DataFrame(new)], ignore_index=True) if new else old
    if df.empty:
        raise RuntimeError("ESPN returned no games")
    df["game_id"] = df["game_id"].astype(str)
    df = df.drop_duplicates(["game_id", "player"])
    df.to_csv(cache, index=False)
    return df
