"""Injury status + a simple 'teammate is out' boost. Uses only games before as_of."""
import pandas as pd
from config import ModelCfg
from data import norm_name

STATUS_MAP = {"available": "active", "probable": "active", "questionable": "questionable",
              "doubtful": "doubtful", "out": "out"}


def load_injuries(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["key"] = df["player"].map(norm_name)
    s = df["status"].astype(str).str.lower().str.strip()
    df["status_class"] = s.map(STATUS_MAP).fillna("questionable")   # unknown -> be cautious
    return df


def apply_status(lines: pd.DataFrame, inj: pd.DataFrame) -> pd.DataFrame:
    """Sets lines['status'] from the injury report. Not on the report = active."""
    worst = {"out": 3, "doubtful": 2, "questionable": 1, "active": 0}
    inj = inj.assign(rank=inj["status_class"].map(worst)).sort_values("rank")
    by_key = inj.drop_duplicates("key", keep="last").set_index("key")["status_class"]
    lines = lines.copy()
    lines["status"] = lines["player"].map(norm_name).map(by_key).fillna("active")
    return lines


def teammate_multipliers(logs: pd.DataFrame, inj: pd.DataFrame, as_of,
                         cfg: ModelCfg = ModelCfg()):
    """Per-team {pts, reb, ast} multipliers when rotation players are out/doubtful.

    Heuristic: lost production is shared across the remaining rotation, scaled by
    `cfg.teammate_pickup`. A player who already missed most of the team's last 10 games
    counts less, because recent form already reflects his absence."""
    g = logs[logs["game_date"] < as_of].copy()
    g["key"] = g["player"].map(norm_name)
    out_keys = set(inj.loc[inj["status_class"].isin(["out", "doubtful"]), "key"])
    mult, notes = {}, []
    if not out_keys:
        return mult, notes
    for team in g.loc[g["key"].isin(out_keys), "team"].unique():
        tl = g[g["team"] == team]
        ids = tl.drop_duplicates("game_id").sort_values("game_date").tail(10)["game_id"]
        rec = tl[tl["game_id"].isin(ids)]
        stats = rec.groupby("key").agg(gp=("pts", "size"), pts=("pts", "mean"), reb=("reb", "mean"),
                                       ast=("ast", "mean"), mpg=("minutes", "mean"), name=("player", "last"))
        rot = stats[(stats["mpg"] >= 15) & (stats["gp"] >= 3)]
        gone = rot[rot.index.isin(out_keys)]
        stay = rot[~rot.index.isin(out_keys)]
        if gone.empty:
            continue
        m = {}
        for s in ("pts", "reb", "ast"):
            if stay[s].sum() <= 0:
                continue
            lost = (gone[s] * gone["gp"] / len(ids)).sum()
            m[s] = min(1 + cfg.teammate_pickup * lost / stay[s].sum(), cfg.teammate_cap)
        if m and max(m.values()) > 1.005:
            mult[team] = m
            notes.append(f"{team}: teammates x{m.get('pts', 1):.2f} pts / x{m.get('reb', 1):.2f} reb / "
                         f"x{m.get('ast', 1):.2f} ast (out/doubtful: {', '.join(gone['name'])})")
    return mult, notes
