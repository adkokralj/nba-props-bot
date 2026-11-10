"""Pulls tonight's NBA player props (PTS, REB, AST, 3PM, PRA, RA) from SportsGameOdds and saves lines.csv.

Setup (once):  put your API key in a file called api_key.txt in this folder.
Run:           python3 fetch_lines.py
"""
import os
import sys

import pandas as pd
import requests

API_BASE = "https://api.sportsgameodds.com/v2"

# SportsGameOdds stat name -> our market code
STAT_TO_MARKET = {"points": "pts", "rebounds": "reb", "assists": "ast",
                  "threePointersMade": "fg3m", "points+rebounds+assists": "pra",
                  "rebounds+assists": "ra"}

NBA_TEAMS = {
    "atlanta hawks": "ATL", "boston celtics": "BOS", "brooklyn nets": "BKN",
    "charlotte hornets": "CHA", "chicago bulls": "CHI", "cleveland cavaliers": "CLE",
    "dallas mavericks": "DAL", "denver nuggets": "DEN", "detroit pistons": "DET",
    "golden state warriors": "GSW", "houston rockets": "HOU", "indiana pacers": "IND",
    "los angeles clippers": "LAC", "la clippers": "LAC", "los angeles lakers": "LAL",
    "memphis grizzlies": "MEM", "miami heat": "MIA", "milwaukee bucks": "MIL",
    "minnesota timberwolves": "MIN", "new orleans pelicans": "NOP",
    "new york knicks": "NYK", "oklahoma city thunder": "OKC", "orlando magic": "ORL",
    "philadelphia 76ers": "PHI", "phoenix suns": "PHX", "portland trail blazers": "POR",
    "sacramento kings": "SAC", "san antonio spurs": "SAS", "toronto raptors": "TOR",
    "utah jazz": "UTA", "washington wizards": "WAS",
}


def get_key() -> str:
    key = os.environ.get("SPORTSGAMEODDS_KEY", "")
    if not key and os.path.exists("api_key.txt"):
        key = open("api_key.txt").read()
    key = key.strip()
    if not key:
        sys.exit("No API key found. Save your key in a file named api_key.txt in this folder.")
    return key


def team_abbr(team: dict) -> str:
    names = team.get("names", {}) or {}
    long_name = str(names.get("long", "")).lower().strip()
    if long_name in NBA_TEAMS:
        return NBA_TEAMS[long_name]
    return str(names.get("short") or names.get("medium") or long_name).upper()


def player_name(stat_entity_id: str) -> str:
    """'LEBRON_JAMES_1_NBA' -> 'Lebron James' (matched to nba_api names later)."""
    parts = stat_entity_id.split("_")
    parts = parts[:-2] if len(parts) > 2 else parts
    return " ".join(p.capitalize() for p in parts)


def parse_events(events: list) -> pd.DataFrame:
    """Turns API events into one row per player / book with line + over/under odds."""
    rows = []
    for ev in events:
        teams = ev.get("teams", {}) or {}
        home, away = team_abbr(teams.get("home", {})), team_abbr(teams.get("away", {}))
        start = (ev.get("status") or {}).get("startsAt")
        odds = ev.get("odds") or {}
        for odd_id, over in odds.items():
            # oddID format: {stat}-{player}-{period}-{betType}-{side}
            if not odd_id.endswith("-game-ou-over"):
                continue
            market = STAT_TO_MARKET.get(odd_id.split("-")[0])
            if market is None:
                continue
            ent = over.get("statEntityID") or odd_id.split("-")[1]
            if ent in ("all", "home", "away"):
                continue
            under = odds.get(odd_id[:-len("over")] + "under") or {}
            ob, ub = over.get("byBookmaker") or {}, under.get("byBookmaker") or {}
            for book, o in ob.items():
                u = ub.get(book)
                if not u or not o.get("available", True) or not u.get("available", True):
                    continue
                try:
                    line, uline = float(o["overUnder"]), float(u["overUnder"])
                    o_odds, u_odds = int(float(o["odds"])), int(float(u["odds"]))
                except (KeyError, TypeError, ValueError):
                    continue
                if line != uline or line % 1 != 0.5:   # skip mismatched / whole-number lines
                    continue
                rows.append(dict(player=player_name(ent), game_id=ev.get("eventID"),
                                 game_date=start, home_team=home, away_team=away,
                                 market=market, line=line, book=book,
                                 over_odds=o_odds, under_odds=u_odds))
    return pd.DataFrame(rows)


def fetch_events(key: str, limit: int = 15) -> list:
    r = requests.get(f"{API_BASE}/events", headers={"x-api-key": key}, timeout=60, params={
        "leagueID": "NBA", "finalized": "false", "oddsAvailable": "true", "limit": limit})
    if r.status_code in (401, 403):
        sys.exit(f"The API rejected your key (HTTP {r.status_code}). Check api_key.txt.")
    r.raise_for_status()
    return r.json().get("data", [])


def main():
    events = fetch_events(get_key())
    print(f"Found {len(events)} upcoming NBA games.")
    df = parse_events(events)
    if df.empty:
        sys.exit("No player props came back. Props usually post 12-24h before tip-off, "
                 "and some free plans may not include them. Tell Claude what you see.")
    df.to_csv("lines.csv", index=False)
    print("Rows per market: " + ", ".join(f"{k} {v}" for k, v in df["market"].value_counts().items()))
    print(f"Saved lines.csv: {df['player'].nunique()} players, "
          f"{df['book'].nunique()} books, {len(df)} rows.")


if __name__ == "__main__":
    main()
