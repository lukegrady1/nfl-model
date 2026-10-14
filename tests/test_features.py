"""Step 2.1: the feature builder gives identical features from a simulation state and from a training row."""
import numpy as np
import polars as pl

from nflsim.models.features import feature_names, features, situation_from_row, situation_from_state
from tests.conftest import needs_data

MODELS = ["play_call", "fourth_down", "throw_outcome", "yac", "rush_yards", "clock_runoff", "fg_make", "fumble_lost",
          "pat_choice", "presnap_penalty", "air_yards", "dropback_outcome"]


@needs_data
def test_state_and_row_paths_match_on_1000_real_plays():
    from nflsim.models.train import extra_from_row, rows_of, state_from_row, training_plays

    plays = training_plays().filter(pl.col("play_class").is_in(["pass", "run", "field_goal", "extra_point", "two_point"]))
    sample = plays.sample(n=1000, seed=7).with_columns(
        play_type_fumble=pl.when(pl.col("play_class") == "run").then(pl.lit("run"))
        .when(pl.col("pass_outcome") == "complete").then(pl.lit("completion"))
        .when(pl.col("pass_outcome").is_in(["sack", "scramble"])).then(pl.col("pass_outcome")).otherwise(None),
        kick_distance=pl.when(pl.col("play_class") == "field_goal").then(pl.col("kick_distance")).otherwise(pl.lit(40.0)))
    rows = rows_of(sample)
    assert len(rows) == 1000
    for r in rows:
        state = state_from_row(r)
        for name in MODELS:
            a = features(name, situation_from_state(state, extra_from_row(r, name), name))
            b = features(name, situation_from_row(r, name))
            assert a.shape == (len(feature_names(name)),)
            assert np.array_equal(a, b), (name, r["game_id"], r["play_id"])


def test_base_feature_definitions():
    from nflsim.models.features import Situation

    s = Situation(down=3, ydstogo=7, yardline_100=35, qtr=4, game_seconds_remaining=170, half_seconds_remaining=170,
                  score_diff=-10, off_timeouts=2, def_timeouts=3, is_home=1)
    f = dict(zip(feature_names("dropback_outcome"), features("dropback_outcome", s)))
    assert f["down_3"] == 1 and f["down_1"] == 0 and f["ydstogo_c"] == 7 and abs(f["log_ydstogo"] - np.log(7)) < 1e-12
    assert f["yardline"] == 0.35 and f["yardline_sq"] == 0.35 ** 2 and f["goal_to_go"] == 0 and f["red_zone"] == 0
    assert f["score_diff_c"] == -10 / 28 and f["under_2min_half"] == 0 and f["trailing_late"] == 1 and f["leading_late"] == 0
    assert abs(f["score_x_time"] - (-10 / 28) * (1 - 170 / 3600)) < 1e-12
    assert f["off_timeouts"] == 2 and f["def_timeouts"] == 3 and f["is_home"] == 1
    s2 = Situation(down=1, ydstogo=25, yardline_100=3, qtr=2, game_seconds_remaining=1900, half_seconds_remaining=100,
                   score_diff=40, off_timeouts=0, def_timeouts=0, is_home=0)
    f2 = dict(zip(feature_names("dropback_outcome"), features("dropback_outcome", s2)))
    assert f2["ydstogo_c"] == 20 and f2["goal_to_go"] == 1 and f2["inside_5"] == 1 and f2["score_diff_c"] == 1.0
    assert f2["under_2min_half"] == 1 and f2["trailing_late"] == 0 and f2["leading_late"] == 0


@needs_data
def test_team_features_match_from_state_and_row():
    """Phase 3: ratings joined on training rows equal the ratings placed in the simulation context."""
    from nflsim.engine.state import Context
    from nflsim.features.team_strength import RATINGS
    from nflsim.models.features import TEAM_FEATURES, feature_names
    from nflsim.models.train import extra_from_row, rows_of, state_from_row, training_plays

    from nflsim.config import settings

    plays = training_plays(team=True).filter(pl.col("play_class").is_in(["pass", "run"]) & (pl.col("season") >= 2019))
    rows = rows_of(plays.sample(n=300, seed=11))
    saved = settings()["team_strength"]["input_scale"]
    settings()["team_strength"]["input_scale"] = 1.0   # the simulation-time calibration is tested separately
    try:
        _check_rows(rows, Context, RATINGS, TEAM_FEATURES, feature_names, extra_from_row, state_from_row)
    finally:
        settings()["team_strength"]["input_scale"] = saved


def _check_rows(rows, Context, RATINGS, TEAM_FEATURES, feature_names, extra_from_row, state_from_row):
    for r in rows:
        state = state_from_row(r)
        off = {k: r[f"off_{k}"] for k in RATINGS}
        deff = {k: r[f"def_{k}"] for k in RATINGS}
        home_is_off = r["posteam"] == r["home_team"]
        state.context = Context(home=off if home_is_off else deff, away=deff if home_is_off else off,
                                weather={"roof_dome": 1 if r.get("roof") in ("dome", "closed") else 0, "wind": r.get("wind")},
                                season=r["season"])
        for name in TEAM_FEATURES:
            a = features(name, situation_from_state(state, extra_from_row(r, name), name), True)
            b = features(name, situation_from_row(r, name), True)
            assert a.shape == (len(feature_names(name, True)),)
            assert np.array_equal(a, b), (name, r["game_id"], r["play_id"])
        assert any(abs(v) > 0 for v in off.values())  # ratings really are attached
