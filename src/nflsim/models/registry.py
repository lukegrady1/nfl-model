"""Maps each decision name to its current implementation. Phase 1 registers lookup tables for everything."""
from __future__ import annotations

import numpy as np

from nflsim.models import buckets as B
from nflsim.models.tables import TableModel, load_tables

DECISIONS = ["play_call", "fourth_down", "air_yards", "dropback_outcome", "throw_outcome", "yac", "sack_yards",
             "scramble_yards", "rush_yards", "fumble_lost", "out_of_bounds", "pick_six", "int_return_yards", "fg_make", "punt_result",
             "punt_net_yards", "punt_retain_spot", "kickoff_result", "kickoff_spot", "free_kick_spot", "pat_choice",
             "xp_result", "two_point_result", "clock_runoff", "timeout_after", "timeout_runoff", "presnap_penalty", "def_penalty_first_down"]


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
    "timeout_runoff": lambda s, e: (int(e["clock_stopper"]), B.under_two_half(s.half_seconds_remaining, s.qtr)),
    "presnap_penalty": lambda s, e: (s.down, B.dist_bucket(s.ydstogo)),
    "def_penalty_first_down": lambda s, e: (),
}


def table_models() -> dict:
    T = load_tables()
    return {name: TableModel(T[name], KEY_FNS[name]) for name in DECISIONS if name in T}


_DEFAULT: dict | None = None


def default_models() -> dict:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = table_models()
    return _DEFAULT


def sample_exact(models: dict, name: str, rng: np.random.Generator, state, extra: dict | None = None) -> int:
    """Sample a class, then the exact value inside it if the model has a within-bin distribution."""
    m = models[name]
    cls = m.sample(rng, state, extra or {})
    sw = getattr(m, "sample_within", None)
    if sw is not None and getattr(m, "table", None) is not None and cls in m.table.within:
        return int(sw(rng, cls))
    return cls
