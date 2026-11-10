"""Player projection: minutes first -> per-minute rates -> context multipliers.

For every stat (pts, reb, ast, 3-point attempts) the projection is
    minutes x per-minute rate x (home/rest) x (teammates out) x (pace) x (opponent allows)
and every step is logged so the board can say WHY a pick has an edge (minutes / form / splits / matchup).
3PM = attempts x the player's shrunk 3P% (adjusted for how well the opponent defends the arc).
"""
import numpy as np
import pandas as pd
from config import ModelCfg

STATS = ["pts", "reb", "ast", "fg3a"]


def _ewma(x: pd.Series, halflife: float) -> float:
    return float(x.ewm(halflife=halflife).mean().iloc[-1])


def project_player(logs, player, as_of, home, rest_days, team, opp, tables, arch_map, league,
                   cfg: ModelCfg = ModelCfg(), team_mult=None):
    """Returns a dict of per-minute means and context, or None. Only games before `as_of` are used.
    team_mult: {'pts':x,'reb':y,'ast':z} boost from injured teammates (default 1)."""
    g = logs[(logs["player"] == player) & (logs["game_date"] < as_of)]
    n = len(g)
    if n < cfg.min_games:
        return None
    team_mult = team_mult or {}
    arch = arch_map.get(player, "wing")
    lg = league.get(arch, league["all"])

    season_min = g["minutes"].mean()
    mins = cfg.minutes_recent_weight * _ewma(g["minutes"], 4) + (1 - cfg.minutes_recent_weight) * season_min
    sd_min = float(max(g["minutes"].tail(15).std(ddof=1) if n > 2 else 6.0, 3.0))
    last10 = g["minutes"].tail(10)
    min_cv = float(last10.std(ddof=1) / last10.mean()) if len(last10) > 2 else 1.0
    recent_ratio = float(g["minutes"].tail(3).mean() / season_min)
    tg = logs[(logs["team"] == team) & (logs["game_date"] < as_of)]
    last_ids = tg.drop_duplicates("game_id").sort_values("game_date").tail(10)["game_id"]
    appear = float(g["game_id"].isin(last_ids).sum() / max(len(last_ids), 1))

    use_m = cfg.use_matchup and tables is not None and not tables.empty
    hb = cfg.home_boost if home else 1 / cfg.home_boost
    splits_base = hb * (cfg.b2b_penalty if rest_days <= 1 else 1.0)

    out = dict(mins=mins, sd_min=sd_min, n=n, min_cv=min_cv, recent_ratio=recent_ratio,
               appear_rate=appear, rate={}, disp={}, why={}, tm=dict(team_mult))
    boost_any = 1.0
    for s in STATS:
        per_min = g[s] / g["minutes"]
        season = g[s].sum() / g["minutes"].sum()
        k = cfg.shrink_k
        base = (n * season + k * lg[s]) / (n + k)
        recent = _ewma(per_min, cfg.ewma_halflife_games)
        rate = cfg.recent_weight * recent + (1 - cfg.recent_weight) * base
        form = rate / base if base > 0 else 1.0
        tm = team_mult.get("pts" if s == "fg3a" else s, 1.0)
        boost_any = max(boost_any, tm)
        splits = splits_base * tm
        matchup = 1.0
        if use_m:
            pace = 1 + (tables.pace_factor(team, opp) - 1) * cfg.pace_weight
            deff = 1 + (tables.def_factor(s, opp, arch) - 1) * cfg.opp_weight
            matchup = pace * deff
            if s == "reb":
                matchup *= 1 + (tables.reb_opp_factor(team, opp) - 1) * cfg.reb_opp_weight
        out["rate"][s] = base * form * splits * matchup
        out["why"][s] = dict(form=form, splits=splits, matchup=matchup,
                             minutes=mins / season_min if season_min else 1.0)
        # dispersion given minutes: var(stat - rate*min) / mean
        gg = g.tail(40)
        resid = gg[s] - season * gg["minutes"]
        mean_s = max((season * gg["minutes"]).mean(), 0.3)
        emp = float(resid.var(ddof=1) / mean_s) if len(gg) > 5 else cfg.disp_prior[s]
        out["disp"][s] = float(np.clip(0.5 * emp + 0.5 * cfg.disp_prior[s], 1.05, 6.0))

    # three-point accuracy: shrink player's % toward league, then opponent arc-defence effect
    a3 = g["fg3a"].sum()
    p3 = (g["fg3m"].sum() + cfg.p3_k * lg["p3"]) / (a3 + cfg.p3_k)
    pct_mult = 1.0
    if use_m:
        fo = tables.overall
        ratio = fo["fg3m"].get(opp, 1.0) / max(fo["fg3a"].get(opp, 1.0), 1e-9)
        pct_mult = ratio ** cfg.pct_weight
    out["p3"] = float(np.clip(p3 * pct_mult, 0.05, 0.6))
    out["team_boost"] = boost_any
    return out
