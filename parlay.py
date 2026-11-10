"""Slip builder. Mostly 1-3 legs, 4-5 only occasionally and only with stronger edges."""
from itertools import combinations
import pandas as pd
from config import ParlayCfg


def _valid(combo, cfg: ParlayCfg) -> bool:
    players = [l["player"] for l in combo]
    if len(set(players)) < len(players):
        return False
    if not cfg.allow_same_game:
        games = [l["game_id"] for l in combo]
        if len(set(games)) < len(games):
            return False
    if len(combo) >= 4 and any(l["edge"] < cfg.min_edge_long for l in combo):
        return False
    return True


def _slip(combo):
    p, d = 1.0, 1.0
    for l in combo:
        p *= l["p_model"]
        d *= l["dec"]
    return dict(legs=combo, n=len(combo), p_hit=p, dec=d, ev=p * d - 1)


def _counts(cfg: ParlayCfg):
    raw = {k: v * cfg.n_slips for k, v in cfg.leg_mix.items() if k <= cfg.max_legs}
    counts = {k: int(v) for k, v in raw.items()}
    # hand out leftover slips to the sizes with the largest remainders,
    # but never let 4-5 leg slips take more than their mix allows
    rem = sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)
    while sum(counts.values()) < cfg.n_slips:
        k = rem.pop(0) if rem else 2
        if k <= 3 or raw[k] - counts[k] >= 0.5:
            counts[k] += 1
        elif not rem:
            counts[2] += 1
    return counts


def build_slips(eligible_legs: pd.DataFrame, cfg: ParlayCfg = ParlayCfg(), sort_by: str = "ev"):
    """sort_by='p_hit' ranks legs and slips by chance to hit instead of EV."""
    pool_key = "p_model" if sort_by == "p_hit" else "ev"
    pool = eligible_legs.sort_values(pool_key, ascending=False).head(cfg.pool_size)
    legs = pool.to_dict("records")
    uses, chosen = {}, []
    for size, want in _counts(cfg).items():
        if want == 0 or len(legs) < size:
            continue
        combos = [_slip(c) for c in combinations(legs, size) if _valid(c, cfg)]
        combos.sort(key=lambda s: s[sort_by], reverse=True)
        got = 0
        for s in combos:
            keys = [(l["player"], l["market"]) for l in s["legs"]]
            if any(uses.get(k, 0) >= cfg.max_leg_reuse for k in keys):
                continue
            for k in keys:
                uses[k] = uses.get(k, 0) + 1
            chosen.append(s)
            got += 1
            if got >= want:
                break
    return sorted(chosen, key=lambda s: (s["n"], -s[sort_by]))


def format_slip(s) -> str:
    lines = [f"{s['n']}-leg | hit {s['p_hit']:.0%} | pays {s['dec']:.2f}x | EV {s['ev']:+.1%}"]
    for l in s["legs"]:
        lines.append(f"   {l['player']} {l['market'].upper()} {l['side']} {l['line']} "
                     f"({l['odds']:+d} {l['book']}, proj {l['proj']}, edge {l['edge']:+.1%})")
    return "\n".join(lines)
