"""Maps each decision name to its current implementation. Phase 1 registers lookup tables for everything."""
from __future__ import annotations

import numpy as np

from nflsim.models import buckets as B
from nflsim.models.tables import TableModel, load_tables

DECISIONS = ["play_call", "fourth_down", "air_yards", "dropback_outcome", "throw_outcome", "yac", "sack_yards",
             "scramble_yards", "rush_yards", "fumble_lost", "fumble_return_td", "out_of_bounds", "pick_six", "int_return_yards", "fg_make", "punt_result",
             "punt_net_yards", "punt_retain_spot", "kickoff_result", "kickoff_spot", "free_kick_spot", "pat_choice",
             "xp_result", "two_point_result", "clock_runoff", "kickoff_runoff", "timeout_after", "timeout_runoff", "presnap_penalty", "penalty_yards", "def_penalty_first_down"]


def _tb(s) -> int:
    return B.time_bucket(s.qtr, s.half_seconds_remaining, s.game_seconds_remaining)


KEY_FNS = {
    "play_call": lambda s, e: (min(s.down, 3), B.dist_bucket(s.ydstogo), B.score_bucket(s.score_diff), _tb(s)),
    "fourth_down": lambda s, e: (B.dist_bucket(s.ydstogo), B.yardline5(s.yardline_100),
                                 B.score_time_bucket(s.qtr, s.half_seconds_remaining, s.score_diff)),
    "air_yards": lambda s, e: (s.down, B.dist_bucket(s.ydstogo), B.zone(s.yardline_100)),
    "dropback_outcome": lambda s, e: (B.zone(s.yardline_100), s.down, B.dist_bucket(s.ydstogo)),
    "throw_outcome": lambda s, e: (e["air_bin"], B.zone(s.yardline_100)),
    "yac": lambda s, e: (e["air_bin"], B.zone(s.yardline_100)),
    "sack_yards": lambda s, e: (),
    "scramble_yards": lambda s, e: (B.zone(s.yardline_100),),
    "rush_yards": lambda s, e: (B.dist_bucket(s.ydstogo), B.zone(s.yardline_100)),
    "fumble_lost": lambda s, e: (e["play_type_idx"],),
    "out_of_bounds": lambda s, e: (e["play_type_idx"], B.lead_bucket(s.score_diff),
                                   B.under_two_half(s.half_seconds_remaining, s.qtr)),
    "pick_six": lambda s, e: (),
    "fumble_return_td": lambda s, e: (),
    "int_return_yards": lambda s, e: (),
    "fg_make": lambda s, e: (e["distance"],),
    "punt_result": lambda s, e: (B.yardline5(s.yardline_100),),
    "punt_net_yards": lambda s, e: (B.yardline5(s.yardline_100),),
    "punt_retain_spot": lambda s, e: (),
    "kickoff_result": lambda s, e: (),
    "kickoff_spot": lambda s, e: (),
    "free_kick_spot": lambda s, e: (),
    "pat_choice": lambda s, e: (B.pat_score_key(e["diff_after_td"]), _tb(s)),
    "xp_result": lambda s, e: (),
    "two_point_result": lambda s, e: (),
    "clock_runoff": lambda s, e: (int(e["clock_stopper"]), B.lead_bucket(e["offense_diff"]),
                                  B.under_two_half(s.half_seconds_remaining, s.qtr)),
    "timeout_after": lambda s, e: (int(e["clock_stopper"]), B.lead_bucket(e["offense_diff"]),
                                   B.under_two_half(s.half_seconds_remaining, s.qtr)),
    "kickoff_runoff": lambda s, e: (),
    "timeout_runoff": lambda s, e: (int(e["clock_stopper"]), B.under_two_half(s.half_seconds_remaining, s.qtr)),
    "presnap_penalty": lambda s, e: (s.down, B.dist_bucket(s.ydstogo)),
    "def_penalty_first_down": lambda s, e: (),
    "penalty_yards": lambda s, e: (int(e["side_idx"]),),
}


def table_models() -> dict:
    T = load_tables()
    return {name: TableModel(T[name], KEY_FNS[name]) for name in DECISIONS if name in T}


# Decisions that Phase 2 trains; everything else stays an empirical table (spec step 2.2).
TRAINABLE = ["play_call", "fourth_down", "air_yards", "dropback_outcome", "throw_outcome", "yac", "rush_yards",
             "fumble_lost", "fg_make", "pat_choice", "clock_runoff", "presnap_penalty"]


def build_models(trained: set[str] | None = None, tables: dict | None = None) -> dict:
    """Lookup tables for every decision, with trained models swapped in for the names in `trained`
    (None = every trained model saved in artifacts/models; empty set = tables only)."""
    from nflsim.models.trained import available_trained, load_trained

    M = table_models() if tables is None else {name: TableModel(tables[name], KEY_FNS[name]) for name in DECISIONS if name in tables}
    names = set(available_trained()) & set(TRAINABLE) if trained is None else set(trained)
    for name in sorted(names):
        M[name] = load_trained(name)
    return M


_DEFAULT: dict | None = None


def default_models() -> dict:
    """Trained models where available, tables elsewhere (Phase 2 onward)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = build_models()
    return _DEFAULT


def sample_exact(models: dict, name: str, rng: np.random.Generator, state, extra: dict | None = None) -> int:
    """Sample a class, then the exact value inside it if the model has a within-bin distribution."""
    m = models[name]
    cls = m.sample(rng, state, extra or {})
    within = getattr(m, "within", None)
    if within and cls in within:
        return int(m.sample_within(rng, cls))
    return cls
