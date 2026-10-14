"""Step 2.1: one feature builder for both training rows and simulation states.

Both paths build a `Situation` (from a GameState + extra dict, or from a plays.parquet row) and call the same
`features(model_name, situation)` function, so training and simulation cannot drift apart.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from nflsim.config import settings

DEFAULT_WIND = 8.0          # mean wind on outdoor plays 2018-2025 (8.2 mph); used when no weather input is given
import os as _os
ERA_REF_SEASON = 2025       # era feature = (season - 2025) / 5, clipped at the last training season for simulation
ERA_MAX_SEASON = int(_os.environ.get("NFLSIM_ERA_MAX", 2025))
PLAY_TYPES = ["completion", "sack", "scramble", "run"]   # fumble model one-hot order (= Phase 1 play_type_idx)


@dataclass(slots=True)
class Situation:
    down: int
    ydstogo: int
    yardline_100: int
    qtr: int
    game_seconds_remaining: int
    half_seconds_remaining: int
    score_diff: int                 # offense minus defense
    off_timeouts: int
    def_timeouts: int
    is_home: int
    # model-specific extras (None when not needed)
    air_yards: float | None = None
    kick_distance: float | None = None
    roof_dome: int | None = None
    wind: float | None = None
    play_type: str | None = None
    yards_gained: float | None = None
    clock_stopper: int | None = None
    play_was_pass: int | None = None
    out_of_bounds: int | None = None
    diff_after_td: int | None = None
    season: int | None = None
    play_kind: str | None = None      # clock model: kickoff / punt / field_goal / penalty / scrimmage
    off_ratings: dict | None = None   # Phase 3 team z-scores for the offense (missing = league average, 0)
    def_ratings: dict | None = None
    player: dict | None = None        # Phase 4 shrunk over-expected effects of the players on this play


# ---- base situation features ---------------------------------------------------------------------------------
BASE_NAMES = ["down_1", "down_2", "down_3", "down_4", "ydstogo_c", "log_ydstogo", "yardline", "yardline_sq",
              "goal_to_go", "red_zone", "inside_5", "score_diff_c", "half_secs", "under_2min_half", "trailing_late",
              "leading_late", "score_x_time", "off_timeouts", "def_timeouts", "is_home"]


def base_features(s: Situation) -> list[float]:
    ydstogo = max(int(s.ydstogo), 1)
    yl = float(s.yardline_100)
    sd = max(min(float(s.score_diff), 28.0), -28.0) / 28.0
    late = (s.qtr == 4 and s.game_seconds_remaining <= 480) or s.qtr >= 5
    gsr = float(s.game_seconds_remaining)
    return [
        1.0 if s.down == 1 else 0.0, 1.0 if s.down == 2 else 0.0, 1.0 if s.down == 3 else 0.0, 1.0 if s.down == 4 else 0.0,
        float(min(ydstogo, 20)), math.log(ydstogo),
        yl / 100.0, (yl / 100.0) ** 2,
        1.0 if ydstogo >= s.yardline_100 else 0.0, 1.0 if yl <= 20 else 0.0, 1.0 if yl <= 5 else 0.0,
        sd, float(s.half_seconds_remaining) / 1800.0, 1.0 if s.half_seconds_remaining <= 120 else 0.0,
        1.0 if (late and s.score_diff < 0) else 0.0, 1.0 if (late and s.score_diff > 0) else 0.0,
        sd * (1.0 - gsr / 3600.0),
        float(s.off_timeouts), float(s.def_timeouts), float(s.is_home),
    ]


# ---- model-specific feature sets ------------------------------------------------------------------------------
def _air(s: Situation) -> list[float]:
    a = max(min(float(s.air_yards or 0.0), 50.0), -10.0)
    return [a, a * a]


def _fg(s: Situation) -> list[float]:
    d = float(s.kick_distance)
    wind = s.wind if s.wind is not None else (0.0 if s.roof_dome else DEFAULT_WIND)
    return [d, d * d, float(s.roof_dome or 0), max(min(wind, 30.0), 0.0)]


def _fumble(s: Situation) -> list[float]:
    y = max(min(float(s.yards_gained or 0.0), 40.0), -20.0)
    return [1.0 if s.play_type == t else 0.0 for t in PLAY_TYPES] + [y]


def _pat(s: Situation) -> list[float]:
    d = int(s.diff_after_td)
    onehot = [1.0 if d == v else 0.0 for v in range(-15, 16)]
    return onehot + [max(min(d, 16), -16) / 16.0, float(s.half_seconds_remaining) / 1800.0,
                     1.0 if s.qtr >= 3 else 0.0, _era(s)]


def _clock(s: Situation) -> list[float]:
    return [float(s.clock_stopper or 0), float(s.play_was_pass or 0), float(s.out_of_bounds or 0)]


def _fourth(s: Situation) -> list[float]:
    d = float(s.yardline_100 + 17)
    return [d, d * d, _era(s)]


def _era(s: Situation) -> float:
    season = s.season if s.season is not None else ERA_MAX_SEASON
    return (min(int(season), ERA_MAX_SEASON) - ERA_REF_SEASON) / 5.0


def _play_call(s: Situation) -> list[float]:
    """Down x distance and score x clock interactions that an additive logit cannot express."""
    ly = math.log(max(int(s.ydstogo), 1))
    short = 1.0 if s.ydstogo <= 2 else 0.0
    d2 = 1.0 if s.down == 2 else 0.0
    d3 = 1.0 if s.down == 3 else 0.0
    late = (s.qtr == 4 and s.game_seconds_remaining <= 480) or s.qtr >= 5
    sd = max(min(float(s.score_diff), 28.0), -28.0) / 28.0
    u2 = 1.0 if s.half_seconds_remaining <= 120 else 0.0
    trail = 1.0 if (late and s.score_diff < 0) else 0.0
    lead = 1.0 if (late and s.score_diff > 0) else 0.0
    return [d2 * ly, d3 * ly, short, d2 * short, d3 * short, sd * u2, trail * u2, lead * u2, sd * sd, _era(s)]


YL_EDGES = [5, 10, 15, 20, 30, 50]      # yardline buckets: <=5, 6-10, 11-15, 16-20, 21-30, 31-50, 51+
YL_NAMES = ["yl_le5", "yl_6_10", "yl_11_15", "yl_16_20", "yl_21_30", "yl_31_50", "yl_51p"]
AIR_NAMES = ["air_le-1", "air_0_4", "air_5_9", "air_10_14", "air_15_19", "air_20_29", "air_30p"]


def _bucket_onehot(value: float, edges: list[int]) -> list[float]:
    out = [0.0] * (len(edges) + 1)
    i = 0
    while i < len(edges) and value > edges[i]:
        i += 1
    out[i] = 1.0
    return out


def _yl_buckets(s: Situation) -> list[float]:
    """Goal-line room: lets a multinomial yardage model zero out bins the field cannot hold."""
    return _bucket_onehot(float(s.yardline_100), YL_EDGES)


def _air_onehot(s: Situation) -> list[float]:
    return _bucket_onehot(float(s.air_yards or 0.0), [-1, 4, 9, 14, 19, 29])


def _yac_extra(s: Situation) -> list[float]:
    room = max(float(s.yardline_100) - float(s.air_yards or 0.0), 0.0)
    return _air(s) + _air_onehot(s) + _bucket_onehot(room, YL_EDGES) + [math.log(room + 1.0)]


def _throw_extra(s: Situation) -> list[float]:
    return _air(s) + _air_onehot(s)


def _fg_extra(s: Situation) -> list[float]:
    d = float(s.kick_distance)
    return _fg(s) + [d ** 3 / 1e5, _era(s)]


def _clock_extra(s: Situation) -> list[float]:
    cs = float(s.clock_stopper or 0)
    u2 = 1.0 if s.half_seconds_remaining <= 120 else 0.0
    late = (s.qtr == 4 and s.game_seconds_remaining <= 480) or s.qtr >= 5
    trail = 1.0 if (late and s.score_diff < 0) else 0.0
    lead = 1.0 if (late and s.score_diff > 0) else 0.0
    second_half = 1.0 if s.qtr >= 3 else 0.0
    kinds = [1.0 if s.play_kind == k else 0.0 for k in CLOCK_KINDS]
    return _clock(s) + [cs * u2, cs * trail, cs * lead, trail * u2, lead * u2, second_half, second_half * u2,
                        float(s.play_was_pass or 0) * cs, _era(s)] + kinds


def _fine_yardline(v: float) -> list[float]:
    """One indicator per yard from 1 to 20 (where the goal line truncates yardage), then 21-30, 31-50, 51+."""
    out = [0.0] * 23
    v = int(v)
    if v <= 20:
        out[max(v, 1) - 1] = 1.0
    elif v <= 30:
        out[20] = 1.0
    elif v <= 50:
        out[21] = 1.0
    else:
        out[22] = 1.0
    return out


FINE_YL_NAMES = [f"yl_{i}" for i in range(1, 21)] + ["yl_21_30", "yl_31_50", "yl_51p"]
FINE_ROOM_NAMES = [f"room_{i}" for i in range(1, 21)] + ["room_21_30", "room_31_50", "room_51p"]
INTER_NAMES = [f"i_{n}" for n in ["d2_logyd", "d3_logyd", "short", "d2_short", "d3_short", "sd_u2", "trail_u2", "lead_u2", "sd_sq", "era"]]


def _air_full(s: Situation) -> list[float]:
    return _play_call(s) + _fine_yardline(s.yardline_100)


def _rush_full(s: Situation) -> list[float]:
    return _play_call(s) + _fine_yardline(s.yardline_100)


def _yac_full(s: Situation) -> list[float]:
    room = max(float(s.yardline_100) - float(s.air_yards or 0.0), 0.0)
    return _yac_extra(s) + _play_call(s) + _fine_yardline(room)


def _throw_full(s: Situation) -> list[float]:
    return _throw_extra(s) + _play_call(s)


CLOCK_KINDS = ["punt", "field_goal", "penalty"]
CLOCK_EXTRA = ["clock_stopper", "play_was_pass", "out_of_bounds", "cs_u2", "cs_trail", "cs_lead", "trail_u2", "lead_u2",
               "second_half", "h2_u2", "pass_cs", "era"] + [f"kind_{k}" for k in CLOCK_KINDS]
PLAY_KIND_FROM_TYPE = {"punt": "punt", "field_goal": "field_goal", "penalty": "penalty"}
PLAY_KIND_FROM_CLASS = {"punt": "punt", "field_goal": "field_goal", "penalty_no_play": "penalty"}
ROOM_NAMES = ["room_le5", "room_6_10", "room_11_15", "room_16_20", "room_21_30", "room_31_50", "room_51p"]
PLAY_CALL_EXTRA = ["d2_logyd", "d3_logyd", "short", "d2_short", "d3_short", "sd_u2", "trail_u2", "lead_u2", "sd_sq", "era"]


SPECS: dict[str, tuple[list[str], object]] = {
    # name: (extra feature names, extra feature function); None = base only; "no_base" sets use only extras
    "play_call": (PLAY_CALL_EXTRA, _play_call),
    "fourth_down": (["fg_distance", "fg_distance_sq", "era"], _fourth),
    "air_yards": (INTER_NAMES + FINE_YL_NAMES, _air_full),
    "dropback_outcome": ([], None),
    "throw_outcome": (["air_yards_c", "air_yards_sq"] + AIR_NAMES + INTER_NAMES, _throw_full),
    "yac": (["air_yards_c", "air_yards_sq"] + AIR_NAMES + ROOM_NAMES + ["log_room"] + INTER_NAMES + FINE_ROOM_NAMES, _yac_full),
    "rush_yards": (INTER_NAMES + FINE_YL_NAMES, _rush_full),
    "presnap_penalty": (PLAY_CALL_EXTRA, _play_call),
    "clock_runoff": (CLOCK_EXTRA, _clock_extra),
    "fumble_lost": ([f"pt_{t}" for t in PLAY_TYPES] + ["yards_gained_c"], _fumble),
    "fg_make": (["kick_distance", "kick_distance_sq", "roof_dome", "wind_c", "kick_distance_cu", "era"], _fg_extra),
    "pat_choice": ([f"sd_{v}" for v in range(-15, 16)] + ["sd_clip", "half_secs", "second_half", "era"], _pat),
}
NO_BASE = {"fumble_lost", "fg_make", "pat_choice"}

# Phase 3 (step 3.2): team ratings per model, (offense ratings, defense ratings). Added only when team=True.
TEAM_FEATURES: dict[str, tuple[list[str], list[str]]] = {
    "play_call": (["proe"], []),
    "fourth_down": (["fourth_aggr"], []),
    "air_yards": (["adot"], ["pass_epa_allowed"]),
    "dropback_outcome": (["sack_rate", "pass_epa"], ["sack_rate_forced", "pass_epa_allowed"]),
    "throw_outcome": (["cpoe", "pass_epa"], ["int_rate_forced", "pass_epa_allowed"]),
    "yac": (["pass_epa", "explosive_rate"], ["pass_epa_allowed"]),
    "rush_yards": (["rush_epa", "explosive_rate"], ["rush_epa_allowed"]),
    "clock_runoff": (["pace"], []),
}
# Phase 3 (step 3.4): weather for the passing models (fg_make already has it).
WEATHER_MODELS = {"throw_outcome", "air_yards"}
# Phase 4 (step 4.4): player efficiency features (shrunk over-expected), added with the team features.
PLAYER_FEATURES: dict[str, list[str]] = {
    "dropback_outcome": ["qb_sack_rate", "qb_scramble_rate"],
    "throw_outcome": ["rec_catch_rate_oe", "qb_cpoe", "qb_int_rate"],
    "yac": ["rec_yac_oe"],
    "rush_yards": ["rush_yards_oe"],
    "air_yards": ["qb_adot"],
    "fg_make": ["kicker_fg_oe"],
}


def player_feature_names(model_name: str) -> list[str]:
    return [f"p_{f}" for f in PLAYER_FEATURES.get(model_name, [])]


def player_features(model_name: str, s: Situation) -> list[float]:
    pf = s.player or {}
    return [float(pf.get(f, 0.0) or 0.0) for f in PLAYER_FEATURES.get(model_name, [])]


def team_feature_names(model_name: str) -> list[str]:
    off, deff = TEAM_FEATURES.get(model_name, ([], []))
    return [f"off_{r}" for r in off] + [f"def_{r}" for r in deff]


def team_features(model_name: str, s: Situation) -> list[float]:
    off, deff = TEAM_FEATURES.get(model_name, ([], []))
    o = s.off_ratings or {}
    d = s.def_ratings or {}
    return [float(o.get(r, 0.0) or 0.0) for r in off] + [float(d.get(r, 0.0) or 0.0) for r in deff]


def weather_features(s: Situation) -> list[float]:
    wind = s.wind if s.wind is not None else (0.0 if s.roof_dome else DEFAULT_WIND)
    return [float(s.roof_dome or 0), max(min(wind, 30.0), 0.0)]


def feature_names(model_name: str, team: bool = False) -> list[str]:
    extra_names, _ = SPECS[model_name]
    names = (extra_names if model_name in NO_BASE else BASE_NAMES + extra_names)
    if team:
        names = names + team_feature_names(model_name)
        if model_name in WEATHER_MODELS:
            names = names + ["roof_dome", "wind_c"]
        names = names + player_feature_names(model_name)
    return names


def features(model_name: str, s: Situation, team: bool = False) -> np.ndarray:
    extra_names, fn = SPECS[model_name]
    vals = [] if model_name in NO_BASE else base_features(s)
    if fn is not None:
        vals = vals + fn(s)
    if team:
        vals = vals + team_features(model_name, s)
        if model_name in WEATHER_MODELS:
            vals = vals + weather_features(s)
        vals = vals + player_features(model_name, s)
    return np.asarray(vals, dtype=float)


# ---- building a Situation from either path -------------------------------------------------------------------
def situation_from_state(state, extra: dict | None, model_name: str) -> Situation:
    extra = extra or {}
    down = int(state.down)
    if model_name == "play_call":
        down = min(down, 3)           # trained on downs 1-3; a 4th-down "go" uses the 3rd-down behavior
    gsr = state.ot_state.seconds_remaining if state.in_ot else state.game_seconds_remaining
    weather = getattr(state.context, "weather", None) or {}
    off = extra.get("offense") or state.possession      # the clock model runs after possession may have flipped
    deff = state.opponent(off)
    return Situation(
        down=down, ydstogo=int(state.ydstogo), yardline_100=int(state.yardline_100), qtr=int(state.qtr),
        game_seconds_remaining=int(gsr), half_seconds_remaining=int(state.half_seconds_remaining),
        score_diff=int(state.score[off] - state.score[deff]), off_timeouts=int(state.timeouts[off]),
        def_timeouts=int(state.timeouts[deff]),
        is_home=0 if state.is_neutral else int(off == state.home),
        air_yards=extra.get("air_yards"), kick_distance=extra.get("distance"),
        roof_dome=weather.get("roof_dome", 0), wind=weather.get("wind"),
        play_type=extra.get("play_type"), yards_gained=extra.get("yards"),
        clock_stopper=extra.get("clock_stopper"), play_was_pass=extra.get("play_was_pass"),
        out_of_bounds=extra.get("out_of_bounds"), diff_after_td=extra.get("diff_after_td"),
        season=getattr(state.context, "season", None),
        play_kind=PLAY_KIND_FROM_TYPE.get(extra.get("play_type_kind"), "scrimmage"),
        off_ratings=_scaled(state.context.home if off == state.home else state.context.away),
        def_ratings=_scaled(state.context.away if off == state.home else state.context.home),
        player=extra.get("player"),
    )


def _scaled(ratings: dict | None) -> dict | None:
    """Simulation-time calibration of the team inputs (Phase 3): the per-play models extract the ratings' play-level
    signal, but actual margins move about 1.4 points per simulated point, so the inputs are scaled by
    `team_strength.input_scale` here (never on the training path)."""
    if not ratings:
        return ratings
    k = float(settings()["team_strength"].get("input_scale", 1.0))
    return ratings if k == 1.0 else {r: (v or 0.0) * k for r, v in ratings.items()}


def situation_from_row(r: dict, model_name: str) -> Situation:
    """r: one plays.parquet row as a dict (needs `neutral` from the schedule join, see train.py)."""
    down = int(r["down"] or 1)
    if model_name == "play_call":
        down = min(down, 3)
    roof = r.get("roof")
    pt = r.get("play_type_fumble")
    return Situation(
        down=down, ydstogo=int(r["ydstogo"] or 10), yardline_100=int(r["yardline_100"]), qtr=int(r["qtr"]),
        game_seconds_remaining=int(r["game_seconds_remaining"]), half_seconds_remaining=int(r["half_seconds_remaining"]),
        score_diff=int(r["score_differential"] or 0),
        off_timeouts=int(r["posteam_timeouts_remaining"] if r.get("posteam_timeouts_remaining") is not None else 3),
        def_timeouts=int(r["defteam_timeouts_remaining"] if r.get("defteam_timeouts_remaining") is not None else 3),
        is_home=0 if r.get("neutral") else int(r["posteam"] == r["home_team"]),
        air_yards=r.get("air_yards"), kick_distance=r.get("kick_distance"),
        roof_dome=1 if roof in ("dome", "closed") else 0, wind=r.get("wind"),
        play_type=pt, yards_gained=r.get("yards_gained"),
        clock_stopper=int(bool(r.get("clock_stopper"))), play_was_pass=int(r.get("play_class") == "pass"),
        out_of_bounds=int(r.get("out_of_bounds") or 0),
        diff_after_td=int(r["score_differential"] or 0) if model_name == "pat_choice" else None,
        season=r.get("season"),
        play_kind=PLAY_KIND_FROM_CLASS.get(r.get("play_class"), "scrimmage"),
        off_ratings={k[4:]: v for k, v in r.items() if k.startswith("off_")},
        def_ratings={k[4:]: v for k, v in r.items() if k.startswith("def_")},
        player={k[2:]: v for k, v in r.items() if k.startswith("p_")},
    )


def feature_matrix(model_name: str, rows: list[dict], team: bool = False) -> np.ndarray:
    return np.stack([features(model_name, situation_from_row(r, model_name), team) for r in rows])
