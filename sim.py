"""Correlated Monte Carlo: one shared minutes draw + a copula over pts/reb/ast/3PA per player.
Because every market comes from the SAME simulated games, PRA and RA are consistent with PTS/REB/AST.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, nbinom

STATS = ["pts", "reb", "ast", "fg3a"]


def pooled_corr(logs: pd.DataFrame, as_of, min_games: int = 15) -> np.ndarray:
    """Correlation of the four stats AFTER removing each player's average and the effect of minutes."""
    g = logs[logs["game_date"] < as_of]
    parts = []
    for _, p in g.groupby("player"):
        if len(p) < min_games:
            continue
        r = pd.DataFrame({s: p[s] - (p[s].sum() / p["minutes"].sum()) * p["minutes"] for s in STATS})
        parts.append(r / r.std(ddof=1).replace(0, 1))
    if not parts:
        return np.eye(4)
    c = pd.concat(parts).corr().fillna(0).to_numpy()
    w, v = np.linalg.eigh(c)                     # make sure it is a valid correlation matrix
    c = v @ np.diag(np.clip(w, 1e-3, None)) @ v.T
    d = np.sqrt(np.diag(c))
    return c / np.outer(d, d)


def simulate(proj: dict, corr: np.ndarray, n: int = 12000, seed: int = 0) -> dict:
    """proj from model.project_player -> samples for pts, reb, ast, fg3m, pra, ra."""
    rng = np.random.default_rng(seed)
    m = np.clip(rng.normal(proj["mins"], proj["sd_min"], n), 0, 48)
    z = rng.multivariate_normal(np.zeros(4), corr, size=n)
    u = np.clip(norm.cdf(z), 1e-6, 1 - 1e-6)
    out = {}
    for i, s in enumerate(STATS):
        mean = np.clip(m * proj["rate"][s], 1e-3, None)
        d = proj["disp"][s]
        r = mean / (d - 1.0)
        out[s] = nbinom.ppf(u[:, i], r, r / (r + mean))
    out["fg3m"] = rng.binomial(out["fg3a"].astype(int), proj["p3"])
    out["pra"] = out["pts"] + out["reb"] + out["ast"]
    out["ra"] = out["reb"] + out["ast"]
    return out


def why_label(proj: dict, market: str) -> str:
    """Which lens moved this projection most vs. the player's plain season baseline."""
    comps = {"minutes": 0.0, "form": 0.0, "splits": 0.0, "matchup": 0.0}
    parts = {"pts": ["pts"], "reb": ["reb"], "ast": ["ast"], "fg3m": ["fg3a"],
             "pra": ["pts", "reb", "ast"], "ra": ["reb", "ast"]}[market]
    w = {s: proj["rate"][s] * proj["mins"] for s in parts}
    tot = sum(w.values()) or 1.0
    for s in parts:
        for k in comps:
            comps[k] += (w[s] / tot) * np.log(max(proj["why"][s][k], 1e-6))
    k = max(comps, key=lambda x: abs(comps[x]))
    if abs(comps[k]) < 0.02:
        return "baseline"
    return f"{k} {'+' if comps[k] > 0 else '-'}{abs(comps[k]) * 100:.0f}%"
