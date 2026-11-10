"""Pick tracker: logs every pick the bot makes, grades them after the games, prints the scoreboard.
   python3 track.py grade      # grade yesterday's picks and show results
"""
import datetime as dt
import sys

import numpy as np
import pandas as pd

PICKS, SLIPS = "picks_log.csv", "slips_log.csv"
KEYS = ["game_date", "player", "market", "side", "line"]


def _read(path):
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def log_picks(legs, ok, slips, matchup_on=True):
    """Save every leg with a positive EV (flag 'eligible' for the ones that passed all rules)."""
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    pos = legs[legs["ev"] > 0].copy()
    pos["eligible"] = pos.set_index(["player", "market"]).index.isin(ok.set_index(["player", "market"]).index)
    pos["game_date"] = pd.to_datetime(pos["game_date"]).dt.strftime("%Y-%m-%d")
    pos["logged_at"], pos["matchup_on"], pos["result"] = now, matchup_on, "pending"
    keep = KEYS + ["team", "opp", "book", "odds", "dec", "proj", "minutes", "p_model", "p_market", "edge",
                   "ev", "why", "eligible", "logged_at", "matchup_on", "result"]
    new = pos[[c for c in keep if c in pos.columns]]
    old = _read(PICKS)
    allp = pd.concat([old, new]).drop_duplicates(KEYS + ["matchup_on"], keep="last") if len(old) else new
    allp.to_csv(PICKS, index=False)
    rows = []
    for i, s in enumerate(slips):
        rows.append(dict(logged_at=now, slip=f"{now}#{i+1}", n=s["n"], p_hit=s["p_hit"], dec=s["dec"],
                         legs=" | ".join(f"{l['player']}|{l['market']}|{l['side']}|{l['line']}" for l in s["legs"]),
                         result="pending"))
    if rows:
        o = _read(SLIPS)
        pd.concat([o, pd.DataFrame(rows)]).to_csv(SLIPS, index=False)
    print(f"[tracker] logged {len(new)} picks ({int(new['eligible'].sum())} eligible) and {len(rows)} slips")


def _actual(logs, row):
    d = pd.Timestamp(row["game_date"])
    g = logs[(logs.player == row["player"]) & (logs.opp == row["opp"])
             & (logs.game_date >= d - pd.Timedelta(days=1)) & (logs.game_date <= d + pd.Timedelta(days=1))]
    if g.empty:
        return None
    g = g.iloc[0]
    m = row["market"]
    return {"pts": g.pts, "reb": g.reb, "ast": g.ast, "fg3m": g.fg3m,
            "pra": g.pts + g.reb + g.ast, "ra": g.reb + g.ast}[m]


def grade():
    from data import load_game_logs
    p = _read(PICKS)
    if p.empty:
        print("No picks logged yet. Run the bot first.")
        return
    logs = load_game_logs()
    today = pd.Timestamp(dt.date.today())
    for i in p.index[(p["result"] == "pending") & (pd.to_datetime(p["game_date"]) < today)]:
        a = _actual(logs, p.loc[i])
        if a is None:
            if (today - pd.Timestamp(p.loc[i, "game_date"])).days >= 3:
                p.loc[i, "result"] = "void"
            continue
        win = a > p.loc[i, "line"] if p.loc[i, "side"] == "OVER" else a < p.loc[i, "line"]
        p.loc[i, "result"] = "win" if win else "loss"
        p.loc[i, "actual"] = a
    p.to_csv(PICKS, index=False)
    _grade_slips(p)
    summary(p)


def _grade_slips(p):
    s = _read(SLIPS)
    if s.empty:
        return
    res = p.drop_duplicates(KEYS).set_index(["player", "market", "side", "line"])["result"]
    for i in s.index[s["result"] == "pending"]:
        outs = []
        for leg in s.loc[i, "legs"].split(" | "):
            pl, mk, sd, ln = leg.split("|")
            r = res[(pl, mk, sd, float(ln))] if (pl, mk, sd, float(ln)) in res.index else "pending"
            outs.append(r if isinstance(r, str) else r.iloc[-1])
        if "loss" in outs:
            s.loc[i, "result"] = "loss"
        elif "pending" in outs:
            continue
        else:
            s.loc[i, "result"] = "win"
    s.to_csv(SLIPS, index=False)
    d = s[s["result"].isin(["win", "loss"])]
    if len(d):
        profit = np.where(d["result"] == "win", d["dec"] - 1, -1.0)
        print(f"\nSLIPS: {len(d)} finished, {int((d['result']=='win').sum())} won, "
              f"profit {profit.sum():+.1f} units ({profit.mean():+.1%} per slip)")


def summary(p):
    d = p[p["result"].isin(["win", "loss"])].copy()
    print(f"\nPICKS logged: {len(p)} | graded: {len(d)} | pending: {(p['result']=='pending').sum()} "
          f"| void: {(p['result']=='void').sum()}")
    if d.empty:
        print("Nothing graded yet - come back after tonight's games.")
        return
    d["profit"] = np.where(d["result"] == "win", d["dec"] - 1, -1.0)
    d["lens"] = d["why"].astype(str).str.split().str[0]
    d["edge_bucket"] = pd.cut(d["edge"], [0, .03, .06, .09, .12, 1], labels=["0-3%", "3-6%", "6-9%", "9-12%", "12%+"])
    d["p_bucket"] = pd.cut(d["p_model"], [0, .45, .5, .55, .6, .7, 1])

    def tab(g):
        t = d.groupby(g, observed=True).agg(picks=("profit", "size"), hit=("result", lambda x: (x == "win").mean()),
                                            avg_p=("p_model", "mean"), units=("profit", "sum"),
                                            roi=("profit", "mean"))
        print(t.round(3).to_string())
    for title, g in [("ALL PICKS", lambda x: "all"), ("ELIGIBLE ONLY (what goes on slips)", "eligible"),
                     ("BY MARKET", "market"), ("BY SIDE", "side"), ("BY WHICH LENS DROVE IT", "lens"),
                     ("BY SIZE OF EDGE", "edge_bucket"),
                     ("CALIBRATION (model % vs real hit rate)", "p_bucket")]:
        print(f"\n--- {title} ---")
        if callable(g):
            d["all"] = "all"
            tab("all")
        else:
            tab(g)
    print("\nNote: a few dozen picks prove nothing. Judge it after several hundred.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "grade":
        grade()
    else:
        summary(_read(PICKS))
