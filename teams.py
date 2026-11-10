"""Team and matchup tables, built ONLY from games before `as_of` (no peeking at the future).

What it knows about each team:
  pace, offensive rating, defensive rating, and what the team ALLOWS per game in
  points / rebounds / assists / threes made / threes attempted, overall AND by the type of
  player (guard / wing / big), plus how many shots both teams miss (rebound opportunity).
Positions in box scores are unreliable, so players are typed from their own stats.
"""
import numpy as np
import pandas as pd

ALLOW_STATS = ["pts", "reb", "ast", "fg3m", "fg3a"]
BOX = ["pts", "reb", "ast", "fg3m", "fg3a", "fgm", "fga", "fta", "tov", "oreb"]


def classify_archetypes(logs: pd.DataFrame, as_of) -> pd.Series:
    """player -> 'guard' | 'wing' | 'big', from rebounds and assists per 36 minutes."""
    g = logs[logs["game_date"] < as_of]
    agg = g.groupby("player").agg(mins=("minutes", "sum"), reb=("reb", "sum"), ast=("ast", "sum"))
    agg = agg[agg["mins"] > 0]
    reb36, ast36 = agg["reb"] / agg["mins"] * 36, agg["ast"] / agg["mins"] * 36
    arch = pd.Series("wing", index=agg.index)
    arch[(ast36 >= 5.0) & (reb36 < 7.0)] = "guard"
    arch[(reb36 >= 8.5) | ((reb36 >= 7.0) & (ast36 < 3.0))] = "big"
    return arch


def league_rates(logs: pd.DataFrame, as_of, arch: pd.Series) -> dict:
    """Per-minute league rates by player type (the prior each player is shrunk toward)."""
    g = logs[logs["game_date"] < as_of].copy()
    g["arch"] = g["player"].map(arch)
    out = {}
    for name, grp in [("all", g)] + [(a, x) for a, x in g.dropna(subset=["arch"]).groupby("arch")]:
        mins = max(grp["minutes"].sum(), 1.0)
        out[name] = {s: grp[s].sum() / mins for s in ["pts", "reb", "ast", "fg3a"]}
        out[name]["p3"] = grp["fg3m"].sum() / max(grp["fg3a"].sum(), 1.0)
    return out


class TeamTables:
    def __init__(self, logs: pd.DataFrame, as_of, arch: pd.Series, k: float = 15, arch_k: float = 60):
        g = logs[logs["game_date"] < as_of]
        self.empty = g.empty
        self.overall, self.arch_f, self.pace_f, self.miss_f = {}, {}, {}, {}
        self.table = pd.DataFrame()
        if self.empty:
            return
        agg = {s: (s, "sum") for s in BOX}
        tg = g.groupby(["game_id", "team"]).agg(opp=("opp", "first"), mins=("minutes", "sum"), **agg).reset_index()
        tg["poss"] = tg["fga"] + 0.44 * tg["fta"] - tg["oreb"] + tg["tov"]
        o = tg[["game_id", "team", "pts", "reb", "ast", "fg3m", "fg3a", "fgm", "fga", "poss"]].copy()
        o.columns = ["game_id", "opp"] + ["o_" + c for c in o.columns[2:]]
        m = tg.merge(o, on=["game_id", "opp"])
        if m.empty:
            self.empty = True
            return
        m["pace"] = (m["poss"] + m["o_poss"]) / 2 * 48 / (m["mins"] / 5).clip(lower=1)
        by = m.groupby("team")
        t = pd.DataFrame({
            "games": by.size(),
            "pace": by["pace"].mean(),
            "ortg": by["pts"].sum() / by["poss"].sum() * 100,
            "drtg": by["o_pts"].sum() / by["o_poss"].sum() * 100,
            "miss_for": (by["fga"].sum() - by["fgm"].sum()) / by.size(),
            **{f"allow_{s}": by[f"o_{s}"].mean() for s in ALLOW_STATS},
        })
        self.table = t
        shr = t["games"] / (t["games"] + k)
        for s in ALLOW_STATS:
            lg = t[f"allow_{s}"].mean()
            self.overall[s] = (1 + (t[f"allow_{s}"] / lg - 1) * shr).to_dict()
        self.pace_f = (1 + (t["pace"] / t["pace"].mean() - 1) * shr).to_dict()
        self.miss_f = (1 + (t["miss_for"] / t["miss_for"].mean() - 1) * shr).to_dict()

        # what each team allows to each TYPE of player, per minute, vs. that type's league rate
        a = g.copy()
        a["arch"] = a["player"].map(arch)
        a = a.dropna(subset=["arch"])
        stats = ["pts", "reb", "ast", "fg3m", "fg3a"]
        grp = a.groupby(["opp", "arch"]).agg(mins=("minutes", "sum"), n=("minutes", "size"),
                                              **{s: (s, "sum") for s in stats})
        lg = a.groupby("arch").agg(mins=("minutes", "sum"), **{s: (s, "sum") for s in stats})
        for (team, ar), row in grp.iterrows():
            w = row["n"] / (row["n"] + arch_k)
            self.arch_f[(team, ar)] = {
                s: 1 + ((row[s] / max(row["mins"], 1)) / max(lg.loc[ar, s] / lg.loc[ar, "mins"], 1e-9) - 1) * w
                for s in stats}

    # ---- lookups (1.0 = league average; >1 = more of that stat) ----
    def def_factor(self, stat: str, opp: str, arch: str) -> float:
        f_o = self.overall.get(stat, {}).get(opp, 1.0)
        f_a = self.arch_f.get((opp, arch), {}).get(stat)
        return f_o if f_a is None else 0.5 * f_o + 0.5 * f_a

    def pace_factor(self, team: str, opp: str) -> float:
        return (self.pace_f.get(team, 1.0) + self.pace_f.get(opp, 1.0)) / 2

    def reb_opp_factor(self, team: str, opp: str) -> float:
        return (self.miss_f.get(team, 1.0) + self.miss_f.get(opp, 1.0)) / 2

    def ranks(self, teams=None) -> pd.DataFrame:
        """Readable ranks (1 = best defence / best offence / fastest)."""
        t = self.table
        if t.empty:
            return t
        r = pd.DataFrame({
            "pace": t["pace"].round(1), "pace#": t["pace"].rank(ascending=False).astype(int),
            "ORtg": t["ortg"].round(1), "ORtg#": t["ortg"].rank(ascending=False).astype(int),
            "DRtg": t["drtg"].round(1), "DRtg#": t["drtg"].rank().astype(int),
            "PTSa": t["allow_pts"].round(1), "PTSa#": t["allow_pts"].rank().astype(int),
            "REBa": t["allow_reb"].round(1), "REBa#": t["allow_reb"].rank().astype(int),
            "ASTa": t["allow_ast"].round(1), "ASTa#": t["allow_ast"].rank().astype(int),
            "3PMa": t["allow_fg3m"].round(1), "3PMa#": t["allow_fg3m"].rank().astype(int),
        })
        return r.loc[[x for x in teams if x in r.index]] if teams is not None else r
