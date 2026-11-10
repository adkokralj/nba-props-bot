"""All tunable settings in one place."""
from dataclasses import dataclass, field


@dataclass
class ModelCfg:
    min_games: int = 10             # skip players with fewer games
    ewma_halflife_games: float = 6  # recent-form weight decay
    recent_weight: float = 0.35     # blend: recent form vs. shrunk season baseline
    shrink_k: float = 15            # prior strength (in games) toward league average
    minutes_recent_weight: float = 0.5
    home_boost: float = 1.01        # small, tune via backtest
    b2b_penalty: float = 0.97
    # ---- matchup lens (switch off with --no-matchup to test by ablation) ----
    use_matchup: bool = True
    opp_weight: float = 0.5         # share of the opponent's allowed-stat factor applied
    arch_k: float = 60              # player-games of evidence before archetype factor is trusted
    pace_weight: float = 0.6        # share of the pace difference applied
    reb_opp_weight: float = 0.4     # share of rebound-opportunity (missed shots) applied
    pct_weight: float = 0.3         # share of opponent 3P% effect applied
    p3_k: float = 120               # 3PA of prior when shrinking a player's 3P% to league
    # ---- injuries ----
    teammate_pickup: float = 0.6    # share of an injured player's production teammates absorb
    teammate_cap: float = 1.15      # never boost teammates by more than this
    # ---- simulation ----
    n_sims: int = 12000
    # prior "variance / mean" given minutes, by stat; blended 50/50 with the player's own
    disp_prior: dict = field(default_factory=lambda: {"pts": 2.5, "reb": 1.8, "ast": 1.8, "fg3a": 1.6})


@dataclass
class EdgeCfg:
    min_ev: float = 0.03            # min EV per unit to be a candidate leg
    min_edge: float = 0.03          # min (model prob - fair market prob)
    max_edge: float = 0.12          # above this = probably a bug / missing news
    allowed_status: tuple = ("active",)  # questionable/doubtful/out are kept off slips
    model_weight: float = 0.5       # (only used by the optional safe mode)
    # playing-time gate: keeps players whose minutes the model can't trust off the slips
    min_minutes: float = 18.0       # projected minutes
    max_minutes_cv: float = 0.35    # minutes swing over last 10 games (std / mean)
    recent_ratio: tuple = (0.80, 1.30)  # last-3 minutes vs season average
    min_appear: float = 0.80        # played in at least this share of team's last 10 games
    block_under_if_teammates_out: float = 1.03  # teammates out => more production, so no unders


@dataclass
class ParlayCfg:
    # Share of slips at each leg count. Mean here is ~2.35 legs.
    leg_mix: dict = field(default_factory=lambda: {1: 0.25, 2: 0.35, 3: 0.25, 4: 0.10, 5: 0.05})
    n_slips: int = 8
    max_legs: int = 5
    min_edge_long: float = 0.05     # 4-5 leg slips need every leg at/above this edge
    allow_same_game: bool = False   # same-game legs are correlated; off by default
    max_leg_reuse: int = 3          # a single leg can appear in at most this many slips
    pool_size: int = 12             # top-N legs considered for combos


@dataclass
class SafeCfg:
    """Optional 'highest chance to hit' mode (switched off; ranking is by edge)."""
    min_p: float = 0.56
    min_blend_edge: float = 0.02
    max_blend_edge: float = 0.10
    min_books: int = 3
    min_minutes: float = 20.0
    max_minutes_cv: float = 0.30
    recent_ratio: tuple = (0.85, 1.25)
    min_games: int = 12
    min_margin: float = 0.15
    min_line: float = 9.5
    worst_odds: int = -150
    block_under_if_teammates_out: float = 1.03
