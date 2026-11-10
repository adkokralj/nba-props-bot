"""Data loaders. Swap these out without touching the model."""
import re
import unicodedata

import numpy as np
import pandas as pd


def load_game_logs_nba_api(seasons=("2025-26", "2026-27")) -> pd.DataFrame:
    """Player game logs from nba_api (free). Needs internet + `pip install nba_api`.
    Two seasons by default so early in the year players still have enough games."""
    import time
    from nba_api.stats.endpoints import leaguegamelog
    parts = []
    for season in seasons:
        for attempt in range(4):         # stats.nba.com is sometimes slow or flaky; retry
            try:
                df = leaguegamelog.LeagueGameLog(
                    season=season, player_or_team_abbreviation="P", timeout=90
                ).get_data_frames()[0]
                break
            except Exception as e:
                if attempt == 3:
                    raise
                print(f"[data] nba_api attempt {attempt + 1} failed ({type(e).__name__}); retrying ...")
                time.sleep(10 * (attempt + 1))
        out = pd.DataFrame({
            "player": df["PLAYER_NAME"],
            "team": df["TEAM_ABBREVIATION"],
            "game_id": df["GAME_ID"],
            "game_date": pd.to_datetime(df["GAME_DATE"]),
            "home": ~df["MATCHUP"].str.contains("@"),
            "minutes": df["MIN"],
            "pts": df["PTS"], "reb": df["REB"], "ast": df["AST"], "fg3m": df["FG3M"],
            "fg3a": df["FG3A"], "fgm": df["FGM"], "fga": df["FGA"], "fta": df["FTA"],
            "tov": df["TOV"], "oreb": df["OREB"],
        })
        out["opp"] = df["MATCHUP"].str.split().str[-1]
        parts.append(out)
    return finalize_logs(pd.concat(parts, ignore_index=True))


def load_game_logs() -> pd.DataFrame:
    """nba_api first (works from home); falls back to ESPN when stats.nba.com is unreachable.
    Set LOGS_SOURCE=espn to skip nba_api (used on GitHub, where stats.nba.com times out)."""
    import os
    if os.environ.get("LOGS_SOURCE", "").lower() != "espn":
        try:
            return load_game_logs_nba_api()
        except Exception as e:
            print(f"[data] nba_api failed ({type(e).__name__}); using ESPN instead")
    from espn_logs import load_game_logs_espn
    return finalize_logs(load_game_logs_espn())


def finalize_logs(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["minutes"] > 0].sort_values(["player", "game_date"]).copy()
    df["rest_days"] = df.groupby("player")["game_date"].diff().dt.days.fillna(3)
    return df.reset_index(drop=True)


def norm_name(name: str) -> str:
    """'Nikola Jokić' / 'NIKOLA JOKIC JR' -> comparable key."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s.replace("-", " ").replace("'", ""))
    # spaces removed so "Karl-Anthony" and "Karlanthony" produce the same key
    return "".join(t for t in s.split() if t not in {"jr", "sr", "ii", "iii", "iv"})


def resolve_lines(lines: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    """Raw lines (player, home_team, away_team, ...) -> adds team, opp, home using the
    player's most recent team in the game logs. Unmatched players are dropped."""
    lg = logs.sort_values("game_date").copy()
    lg["key"] = lg["player"].map(norm_name)
    last = lg.groupby("key").tail(1).set_index("key")
    rows, dropped = [], set()
    for r in lines.to_dict("records"):
        key = norm_name(r["player"])
        if key not in last.index:
            dropped.add(r["player"])
            continue
        team = last.loc[key, "team"]
        if team == r["home_team"]:
            r["home"], r["opp"] = True, r["away_team"]
        elif team == r["away_team"]:
            r["home"], r["opp"] = False, r["home_team"]
        else:
            dropped.add(r["player"])
            continue
        r["team"], r["player"] = team, last.loc[key, "player"]
        rows.append(r)
    if dropped:
        print(f"[note] {len(dropped)} players had no match in game logs and were skipped: "
              f"{sorted(dropped)[:8]}{' ...' if len(dropped) > 8 else ''}")
    return pd.DataFrame(rows)


def load_lines(path: str, logs: pd.DataFrame = None) -> pd.DataFrame:
    """CSV from fetch_lines.py (or typed by hand). Needed columns: player, game_id,
    game_date, market, line, book, over_odds, under_odds, plus either
    (team, opp, home) or (home_team, away_team) so the bot can work them out.
    Optional: status (active/questionable/out)."""
    df = pd.read_csv(path)
    df["game_date"] = pd.to_datetime(df["game_date"], utc=True).dt.tz_convert(None)
    if "team" not in df.columns:
        if logs is None:
            raise ValueError("lines file has no team/opp/home columns; pass game logs")
        df = resolve_lines(df, logs)
    if "status" not in df.columns:
        df["status"] = "active"
    return df


# ---------- synthetic data so the pipeline runs end to end ----------
def make_demo(seed: int = 7, days: int = 24):
    """Fake league with built-in team effects (so the matchup code has something to find)."""
    rng = np.random.default_rng(seed)
    teams = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]
    def_f = {t: rng.uniform(0.92, 1.08) for t in teams}      # >1 = bad defense
    pace_f = {t: rng.uniform(0.94, 1.06) for t in teams}
    shapes = [  # minutes, pts/min, reb/min, ast/min, 3pa/min, 3p%
        (33, 0.62, 0.12, 0.22, 0.25, 0.37), (30, 0.50, 0.10, 0.20, 0.20, 0.36),
        (29, 0.48, 0.17, 0.10, 0.18, 0.35), (26, 0.40, 0.15, 0.08, 0.14, 0.34),
        (27, 0.45, 0.33, 0.09, 0.05, 0.30)]
    players = []
    for t in teams:
        for i, sh in enumerate(shapes):
            j = rng.normal(1, 0.08)
            players.append((f"{t} Player{i+1}", t, sh[0] * rng.normal(1, .05), *[x * j for x in sh[1:5]], sh[5]))
    d0 = pd.Timestamp("2026-10-10")
    rows = []
    for day in range(days):
        order = list(teams); rng.shuffle(order)
        for a, b in zip(order[0::2], order[1::2]):
            gid = f"G{day}{a}{b}"
            gp = (pace_f[a] + pace_f[b]) / 2
            for tm, op, home in [(a, b, True), (b, a, False)]:
                for name, team, mins, ppm, rpm, apm, a3, p3 in players:
                    if team != tm:
                        continue
                    m = float(np.clip(rng.normal(mins, 3.5), 8, 42))
                    z = rng.normal()
                    df_ = def_f[op] * gp
                    pts = rng.poisson(max(1, m * ppm * df_ * (1 + 0.12 * z)))
                    reb = rng.poisson(max(.5, m * rpm * gp * rng.normal(1, .1)))
                    ast = rng.poisson(max(.3, m * apm * gp * (1 + 0.08 * z)))
                    fg3a = rng.poisson(max(.3, m * a3 * gp))
                    fg3m = rng.binomial(fg3a, float(np.clip(p3 + 0.03 * z, .1, .6)))
                    fga = rng.poisson(m * 0.45 * gp); fgm = rng.binomial(fga, 0.46)
                    rows.append(dict(player=name, team=tm, opp=op, game_id=gid,
                                     game_date=d0 + pd.Timedelta(days=day), home=home, minutes=m,
                                     pts=pts, reb=reb, ast=ast, fg3m=fg3m, fg3a=fg3a, fgm=fgm, fga=fga,
                                     fta=rng.poisson(m * 0.12), tov=rng.poisson(m * 0.07),
                                     oreb=rng.binomial(reb, 0.25)))
    logs = finalize_logs(pd.DataFrame(rows))
    as_of = d0 + pd.Timedelta(days=days)
    order = sorted(teams); games = list(zip(order[0::2], order[1::2]))
    line_rows = []
    for a, b in games:
        gid = f"TONIGHT_{a}{b}"
        for tm, op, home in [(a, b, True), (b, a, False)]:
            for name, team, *_ in players:
                if team != tm:
                    continue
                h = logs[logs.player == name]
                base = {"pts": h.pts.mean(), "reb": h.reb.mean(), "ast": h.ast.mean(),
                        "fg3m": h.fg3m.mean(), "pra": (h.pts + h.reb + h.ast).mean(),
                        "ra": (h.reb + h.ast).mean()}
                for market, mu in base.items():
                    centre = np.floor(mu + rng.normal(0, 0.6)) + 0.5
                    for book in ["BookA", "BookB", "BookC"]:
                        line_rows.append(dict(
                            player=name, team=tm, opp=op, home=home, game_id=gid, game_date=as_of,
                            market=market, line=max(0.5, centre + (rng.choice([0, 0, 0, 1]) if book == "BookC" else 0)),
                            book=book, over_odds=int(rng.choice([-120, -115, -110, -105, 100])),
                            under_odds=int(rng.choice([-120, -115, -110, -105, 100])), status="active"))
    return logs, pd.DataFrame(line_rows), as_of
