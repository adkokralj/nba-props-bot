"""Usage:
  python3 run.py --demo                     # synthetic data, tests the pipeline
  python3 run.py --lines lines.csv          # real run (uses injuries.csv if it exists)
  options: --no-matchup (switch the matchup lens off, for testing)  --sims 12000
"""
import argparse
import os
import time
import zlib

import numpy as np
import pandas as pd
from config import ModelCfg, EdgeCfg, ParlayCfg
from data import load_game_logs, load_lines, make_demo
from teams import classify_archetypes, league_rates, TeamTables
from model import project_player
from sim import pooled_corr, simulate, why_label
from edge import build_legs, eligible, dropped_by_minutes
from parlay import build_slips, format_slip
from config import ParlayCfg as _PCfg
from injuries import load_injuries, apply_status, teammate_multipliers
import track

MARKETS = ["pts", "reb", "ast", "fg3m", "pra", "ra"]


def project_all(logs, lines, as_of, mcfg, team_mult):
    arch = classify_archetypes(logs, as_of)
    league = league_rates(logs, as_of, arch)
    tables = TeamTables(logs, as_of, arch, mcfg.shrink_k, mcfg.arch_k)
    corr = pooled_corr(logs, as_of)
    proj, skipped = {}, []
    for r in lines.drop_duplicates("player").itertuples():
        last = logs[(logs.player == r.player) & (logs.game_date < as_of)]
        rest = (as_of - last["game_date"].max()).days if len(last) else 3
        p = project_player(logs, r.player, as_of, bool(r.home), rest, r.team, r.opp, tables,
                           arch, league, mcfg, team_mult.get(r.team))
        if p is None:
            skipped.append(r.player)
            continue
        samples = simulate(p, corr, mcfg.n_sims, seed=zlib.crc32(r.player.encode()) % (2**31))
        for m in lines.loc[lines.player == r.player, "market"].unique():
            s = samples[m]
            proj[(r.player, m)] = dict(
                samples=s, mu=float(s.mean()), sd=float(s.std()), minutes=p["mins"], n=p["n"],
                min_cv=p["min_cv"], recent_ratio=p["recent_ratio"], team_boost=p["team_boost"],
                appear_rate=p["appear_rate"], why=why_label(p, m))
    return proj, tables, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--lines")
    ap.add_argument("--injuries", help="injuries.csv (default: injuries.csv if present)")
    ap.add_argument("--out", default="board.csv")
    ap.add_argument("--no-matchup", action="store_true")
    ap.add_argument("--sims", type=int, default=None)
    ap.add_argument("--export", help="write website data to this json file, e.g. docs/data.json")
    ap.add_argument("--no-log", action="store_true", help="do not record picks in picks_log.csv")
    a = ap.parse_args()

    if a.demo:
        logs, lines, as_of = make_demo()
    else:
        logs = load_game_logs()
        lines = load_lines(a.lines, logs)
        as_of = lines["game_date"].min().normalize()

    mcfg = ModelCfg(use_matchup=not a.no_matchup)
    if a.sims:
        mcfg.n_sims = a.sims
    if a.no_matchup:
        print("[matchup lens OFF - ablation run]")
    inj_path = a.injuries or (None if a.demo else ("injuries.csv" if os.path.exists("injuries.csv") else None))
    team_mult = {}
    if inj_path:
        inj = load_injuries(inj_path)
        age_h = (time.time() - os.path.getmtime(inj_path)) / 3600
        lines = apply_status(lines, inj)
        team_mult, notes = teammate_multipliers(logs, inj, as_of, mcfg)
        counts = inj["status_class"].value_counts().to_dict()
        print(f"[injuries] {inj_path} (saved {age_h:.0f}h ago): "
              + ", ".join(f"{v} {k}" for k, v in counts.items() if k != "active"))
        if age_h > 8:
            print("[injuries] WARNING: report is old. Run fetch_injuries.py again before trusting picks.")
        for n in notes:
            print(f"[injuries] {n}")
        flagged = lines.loc[lines["status"] != "active", ["player", "status"]].drop_duplicates()
        if len(flagged):
            print("[injuries] on tonight's board but not confirmed: "
                  + ", ".join(f"{p} ({s})" for p, s in flagged.itertuples(index=False)))
    else:
        print("[injuries] No injuries.csv found. Running WITHOUT injury info; "
              "run fetch_injuries.py first.")

    print(f"[simulating {lines['player'].nunique()} players x {mcfg.n_sims} games ...]")
    projections, tables, skipped = project_all(logs, lines, as_of, mcfg, team_mult)
    if skipped:
        print(f"[data] skipped (too few games): {', '.join(skipped[:12])}{' ...' if len(skipped) > 12 else ''}")

    tonight = sorted(set(lines["team"]) | set(lines["opp"]))
    rk = tables.ranks(tonight)
    if len(rk):
        print("\n=== TEAM RANKS for tonight's teams (1 = best defence / best offence / fastest) ===")
        with pd.option_context("display.width", 220):
            print(rk.to_string())

    legs = build_legs(lines, projections)
    if legs.empty:
        print("No legs could be built (no lines matched a projection).")
        return
    legs.to_csv(a.out, index=False)
    print(f"\n=== RANKED BOARD (top 20 by EV, all six markets) ===")
    cols = ["player", "market", "side", "line", "odds", "book", "proj", "p_model", "p_market",
            "edge", "ev", "why", "corr", "flag"]
    with pd.option_context("display.width", 250, "display.max_colwidth", 34,
                           "display.float_format", "{:.3f}".format):
        print(legs[cols].head(20).to_string(index=False))

    ok = eligible(legs)
    dropped = dropped_by_minutes(legs)
    if dropped:
        print(f"\n[playing time] {len(dropped)} legs had an edge but were kept off the slips:")
        for p, side, line, why in dropped[:25]:
            print(f"   - {p} {side} {line}: {why}")
    print(f"\n{len(ok)} eligible legs -> building slips (mostly 1-3 legs, 4-5 rare)\n")
    slips = build_slips(ok)
    safe = build_slips(ok, _PCfg(n_slips=6, leg_mix={1: 0.5, 2: 0.5}, max_legs=2, pool_size=10),
                       sort_by="p_hit")
    for s in slips:
        s["kind"] = "main"
    for s in safe:
        s["kind"] = "safe"
    for s in slips:
        print(format_slip(s), "\n")
    if slips:
        print(f"avg legs per slip: {sum(s['n'] for s in slips) / len(slips):.2f}")
    if safe:
        print("\n--- safer slips (highest chance to hit first, smaller payouts) ---\n")
        for s in safe:
            print(format_slip(s), "\n")
    if not a.demo and not a.no_log:
        track.log_picks(legs, ok, slips + safe, matchup_on=not a.no_matchup)
    if a.export:
        import export
        export.write(a.export, as_of, legs, ok, slips, rk, dropped, notes if inj_path else [], safe=safe)


if __name__ == "__main__":
    main()
