"""Odds math + turning projections and book lines into candidate legs."""
import pandas as pd
from config import EdgeCfg, SafeCfg
import numpy as np


def p_over(samples, line: float) -> float:
    """P(X > line) from simulated games (with a +1/+2 smoothing so it is never exactly 0 or 1)."""
    return float(((samples > line).sum() + 1) / (len(samples) + 2))


def dec(american: float) -> float:
    return 1 + american / 100 if american > 0 else 1 + 100 / abs(american)


def devig(o_over: float, o_under: float):
    pi, pu = 1 / dec(o_over), 1 / dec(o_under)
    s = pi + pu
    return pi / s, pu / s


def build_legs(lines: pd.DataFrame, projections: dict, cfg: EdgeCfg = EdgeCfg(),
               dedupe: bool = True) -> pd.DataFrame:
    """projections: {(player, market): dict(samples, mu, sd, minutes, n, why, ...)}.
    dedupe=False keeps every line/side so safe_legs() can pick the consensus line."""
    rows = []
    counts = lines.groupby(["player", "market", "line"]).size()
    modal = counts.groupby(level=[0, 1]).max()
    for (player, market, line), g in lines.groupby(["player", "market", "line"]):
        proj = projections.get((player, market))
        if proj is None:
            continue
        n_line = len(g)
        is_cons = n_line == modal[(player, market)]
        fair = [devig(r.over_odds, r.under_odds)[0] for r in g.itertuples()]
        fair_over = sum(fair) / len(fair)
        best_over = g.loc[g["over_odds"].map(dec).idxmax()]
        best_under = g.loc[g["under_odds"].map(dec).idxmax()]
        po = p_over(proj["samples"], line)
        first = g.iloc[0]
        for side, p, fp, best, odds_col in [
            ("OVER", po, fair_over, best_over, "over_odds"),
            ("UNDER", 1 - po, 1 - fair_over, best_under, "under_odds"),
        ]:
            d = dec(best[odds_col])
            pb = cfg.model_weight * p + (1 - cfg.model_weight) * fp
            rows.append(dict(
                player=player, team=first.team, opp=first.opp, game_id=first.game_id, game_date=first.game_date,
                market=market, side=side, line=line, book=best["book"],
                odds=int(best[odds_col]), dec=d, proj=round(proj["mu"], 1),
                minutes=round(proj["minutes"], 1), p_model=p, p_market=fp,
                edge=p - fp, ev=p * d - 1, status=first.status,
                p_blend=pb, blend_edge=pb - fp, n_books_line=n_line, is_consensus=is_cons,
                margin=abs(proj["mu"] - line) / max(proj["sd"], 1e-6), why=proj.get("why", ""),
                n_games=proj["n"], min_cv=proj.get("min_cv", 0.0),
                recent_ratio=proj.get("recent_ratio", 1.0),
                team_boost=proj.get("team_boost", 1.0),
                appear_rate=proj.get("appear_rate", 1.0)))
    legs = pd.DataFrame(rows)
    if legs.empty:
        return legs
    legs["flag"] = ""
    legs.loc[legs["edge"] > cfg.max_edge, "flag"] = "CHECK: edge too big (news/bug?)"
    bad = ~legs["status"].isin(cfg.allowed_status)
    legs.loc[bad, "flag"] = "STATUS: " + legs.loc[bad, "status"].astype(str)
    legs = legs.sort_values("ev", ascending=False)
    # correlation note: same player showing up in several markets on the same side
    key = legs.groupby(["player", "side"])["market"].apply(lambda m: ",".join(sorted(set(m))))
    legs["corr"] = [("also " + ",".join(x for x in key[(p, sd)].split(",") if x != mk))
                    if len(key[(p, sd)].split(",")) > 1 else ""
                    for p, sd, mk in zip(legs["player"], legs["side"], legs["market"])]
    if dedupe:   # keep the better side per prop
        legs = legs.drop_duplicates(["player", "market"])
    return legs.reset_index(drop=True)


def safe_legs(all_legs: pd.DataFrame, cfg: SafeCfg = SafeCfg()) -> pd.DataFrame:
    """Legs with the highest chance to hit that also pass every sanity rule.
    Returned with p_model/edge/ev replaced by the blended (market-shrunk) values."""
    if all_legs.empty:
        return all_legs
    d = all_legs.copy()
    ok = (
        d["is_consensus"] & (d["n_books_line"] >= cfg.min_books)
        & (d["status"] == "active")
        & (d["p_blend"] >= cfg.min_p)
        & (d["blend_edge"] >= cfg.min_blend_edge) & (d["blend_edge"] <= cfg.max_blend_edge)
        & (d["minutes"] >= cfg.min_minutes) & (d["min_cv"] <= cfg.max_minutes_cv)
        & d["recent_ratio"].between(*cfg.recent_ratio)
        & (d["n_games"] >= cfg.min_games) & (d["margin"] >= cfg.min_margin)
        & (d["line"] >= cfg.min_line) & (d["odds"] >= cfg.worst_odds)
        & ~((d["side"] == "UNDER") & (d["team_boost"] > cfg.block_under_if_teammates_out))
    )
    d = d[ok].sort_values("p_blend", ascending=False).drop_duplicates(["player", "market"])
    d["p_raw"], d["edge_raw"] = d["p_model"], d["edge"]
    d["p_model"], d["edge"] = d["p_blend"], d["blend_edge"]
    d["ev"] = d["p_blend"] * d["dec"] - 1
    d["flag"] = ""
    return d.reset_index(drop=True)


def minutes_reason(r, cfg: EdgeCfg = EdgeCfg()) -> str:
    """Why this leg's playing time can't be trusted ('' = fine)."""
    why = []
    if r["appear_rate"] < cfg.min_appear:
        why.append(f"missed games lately (played {r['appear_rate']:.0%} of team's last 10)")
    if r["minutes"] < cfg.min_minutes:
        why.append(f"low minutes ({r['minutes']:.0f} proj)")
    if r["min_cv"] > cfg.max_minutes_cv:
        why.append("minutes swing a lot")
    if not (cfg.recent_ratio[0] <= r["recent_ratio"] <= cfg.recent_ratio[1]):
        why.append("recent minutes changed")
    if r["side"] == "UNDER" and r["team_boost"] > cfg.block_under_if_teammates_out:
        why.append("teammates out (usage up)")
    return "; ".join(why)


def eligible(legs: pd.DataFrame, cfg: EdgeCfg = EdgeCfg()) -> pd.DataFrame:
    m = (legs["ev"] >= cfg.min_ev) & (legs["edge"] >= cfg.min_edge) & (legs["flag"] == "")
    base = legs[m]
    if base.empty:
        return base.copy()
    reasons = base.apply(lambda r: minutes_reason(r, cfg), axis=1)
    return base[reasons == ""].copy()


def dropped_by_minutes(legs: pd.DataFrame, cfg: EdgeCfg = EdgeCfg()) -> list:
    """(player, side, line, reason) for legs that had an edge but failed the playing-time gate."""
    m = (legs["ev"] >= cfg.min_ev) & (legs["edge"] >= cfg.min_edge) & (legs["flag"] == "")
    out = []
    for _, r in legs[m].iterrows():
        why = minutes_reason(r, cfg)
        if why:
            out.append((r["player"], r["side"], r["line"], why))
    return out
