"""Phase 0 validation checks as tests (step 0.3)."""
import polars as pl
import pytest

from nflsim.data import validate
from nflsim.data.clean import add_derived, load_plays, load_schedules
from tests.conftest import needs_data


@pytest.fixture(scope="module")
def plays():
    return load_plays()


@pytest.fixture(scope="module")
def sched():
    return load_schedules()


@needs_data
def test_scores_match_schedules(plays, sched):
    rate, bad = validate.check_scores(plays, sched)
    assert rate < 0.01, f"{bad.height} games mismatched"


@needs_data
def test_pass_rows_have_one_outcome(plays):
    assert validate.check_pass_outcomes(plays) == 0
    # exactly one: outcome is a single string column, so non-null + valid is enough
    p = plays.filter(pl.col("play_class") == "pass")
    assert p["pass_outcome"].is_in(["complete", "incomplete", "interception", "sack", "scramble"]).all()


@needs_data
def test_air_yards_coverage(plays):
    assert validate.check_air_yards(plays) >= 0.95


@needs_data
def test_yardline_in_range(plays):
    assert validate.check_yardline(plays) == 0


@needs_data
def test_seconds_elapsed_bounds(plays):
    s = plays["seconds_elapsed"].drop_nulls()
    assert s.min() >= 0 and s.max() <= 60


def test_play_class_rules():
    """Unit-level checks of the derivation on a tiny synthetic frame."""
    base = {
        "game_id": ["g"] * 6, "play_id": [1, 2, 3, 4, 5, 6], "qtr": [1] * 6,
        "game_seconds_remaining": [3600, 3560, 3530, 3500, 3480, 3470],
        "half_seconds_remaining": [1800, 1760, 1730, 1700, 1680, 1670],
        "play_type": ["pass", "run", "run", "no_play", "no_play", None],
        "qb_dropback": [1, 1, 0, 0, 0, None], "qb_scramble": [0, 1, 0, 0, 0, None],
        "rush_attempt": [0, 1, 1, 0, 0, None], "pass_attempt": [1, 0, 0, 0, 0, None],
        "sack": [0, 0, 0, 0, 0, None], "complete_pass": [0, 0, 0, 0, 0, None],
        "incomplete_pass": [1, 0, 0, 0, 0, None], "interception": [0] * 5 + [None],
        "two_point_attempt": [0] * 5 + [None], "timeout": [0, 0, 0, 1, 0, None], "penalty": [0, 0, 0, 0, 1, None],
        "desc": ["p", "r", "r", "Timeout", "pen", "END QUARTER 1"],
        "fixed_drive": [1, 1, 1, 1, 1, None], "touchdown": [0] * 5 + [None], "safety": [0] * 5 + [None],
        "fumble_lost": [0] * 5 + [None], "first_down": [0] * 5 + [None], "down": [1, 2, 3, 3, 3, None],
        "out_of_bounds": [0] * 5 + [None], "field_goal_result": [None] * 6, "posteam": ["A"] * 5 + [None],
        "yardline_100": [75, 75, 70, 65, 65, None],
    }
    df = add_derived(pl.DataFrame(base))
    assert df["play_class"].to_list() == ["pass", "pass", "run", "timeout", "penalty_no_play"]
    assert df["pass_outcome"].to_list() == ["incomplete", "scramble", None, None, None]
    assert df["seconds_elapsed"].to_list() == [40, 30, 30, 20, 10]
    assert df["next_play_is_timeout"].to_list() == [False, False, True, False, False]
    assert df["clock_stopper"].to_list()[0] is True   # incomplete
    assert df["clock_stopper"].to_list()[1] is False  # scramble in bounds
    assert df["drive_start"].to_list()[0] is True and df["drive_start"].to_list()[1] is False
