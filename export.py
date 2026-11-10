"""Writes everything the website shows into one json file (docs/data.json)."""
import datetime as dt
import json
import os

import numpy as np
import pandas as pd

LEG_COLS = ["player", "team", "opp", "market", "side", "line", "odds", "book", "proj", "minutes",
            "p_model", "p_market", "edge", "ev", "why", "corr", "flag", "status", "n_books_line"]


def _clean(df, cols):
    d = df[[c for c in cols if c in df.columns]].copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].round(4)
    return json.loads(d.to_json(orient="records"))


def _results():
    """Scoreboard from picks_log.csv (graded picks only)."""
    try:
        p = pd.read_csv("picks_log.csv")
    except Exception:
        return {"graded": 0}
    d = p[p["result"].isin(["win", "loss"])].copy()
    out = {"logged": int(len(p)), "graded": int(len(d)), "pending": int((p["result"] == "pending").sum())}
    if d.empty:
        return out
    d["profit"] = np.where(d["result"] == "win", d["dec"] - 1, -1.0)

    def grp(col, data):
        t = data.groupby(col).agg(picks=("profit", "size"), hit=("result", lambda x: (x == "win").mean()),
                                  avg_p=("p_model", "mean"), units=("profit", "sum"), roi=("profit", "mean"))
        t = t.round(3).reset_index().rename(columns={col: "group"})
        return json.loads(t.to_json(orient="records"))
    d["lens"] = d["why"].astype(str).str.split().str[0]
    d["all"] = "all picks"
    out["all"] = grp("all", d)
    out["eligible"] = grp("eligible", d)
    out["market"] = grp("market", d)
    out["lens"] = grp("lens", d)
    out["side"] = grp("side", d)
    return out


def write(path, as_of, legs, ok, slips, ranks, dropped, notes, safe=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    def _sl(slist):
        out = []
        for s in slist:
            out.append(dict(n=s["n"], p_hit=round(s["p_hit"], 4), pays=round(s["dec"], 2), ev=round(s["ev"], 4),
                            legs=[{k: (l[k] if not isinstance(l[k], (np.floating, np.integer)) else float(l[k]))
                                   for k in ("player", "market", "side", "line", "odds", "book", "proj", "edge", "why")}
                                  for l in s["legs"]]))
        return out
    sl = _sl(slips)
    data = dict(
        updated=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        game_day=str(pd.Timestamp(as_of).date()),
        n_legs=int(len(legs)), n_eligible=int(len(ok)),
        board=_clean(legs.head(150), LEG_COLS), eligible=_clean(ok.head(150), LEG_COLS),
        slips=sl, safe_slips=_sl(safe or []), injuries=list(notes),
        dropped=[dict(player=p, side=s, line=float(l), why=w) for p, s, l, w in dropped[:60]],
        ranks=json.loads(ranks.reset_index().to_json(orient="records")) if len(ranks) else [],
        results=_results())
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"[export] wrote {path}")
