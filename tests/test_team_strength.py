"""Phase 3 leakage tests (required by step 3.1) and rating sanity checks."""
import numpy as np
import polars as pl
import pytest

from nflsim.features.team_strength import RATINGS, build_team_ratings, game_stats, load_team_ratings
from tests.conftest import needs_data, needs_models

pytestmark = [needs_data, needs_models]


@pytest.fixture(scope="module")
def ratings():
    return load_team_ratings()


def test_ratings_cover_every_team_and_week(ratings):
    sched = pl.read_parquet("data/processed/schedules.parquet").filter(pl.col("game_type") != "PRE")
    for season in (2021, 2024, 2025):
        weeks = sched.filter(pl.col("season") == season)["week"].unique().to_list()
        r = ratings.filter(pl.col("season") == season)
        assert r["team"].n_unique() == 32 and set(r["week"].unique().to_list()) == set(weeks)
    for s in RATINGS:
        assert ratings[s].is_nan().sum() == 0 and ratings[s].null_count() == 0
    # standardized ratings: league mean 0 within a week; spread near 1 once teams have played (scaled by last
    # season's final spread, so not exactly 1), and visibly shrunk in week 1 (prior only)
    wk = ratings.filter((pl.col("season") == 2025) & (pl.col("week") == 10))
    for s in RATINGS:
        assert abs(wk[s].mean()) < 1e-9 and 0.6 < wk[s].std() < 1.5, s
    wk1 = ratings.filter((pl.col("season") == 2025) & (pl.col("week") == 1))
    assert wk1["pass_epa"].std() < wk["pass_epa"].std()


def test_leakage_recompute_from_plays_before_asof_date(ratings):
    """Spec: for 200 random (team, week) rows, recompute the rating using only plays with game date < the as-of
    date and assert an exact match. One row does the full rebuild from plays; all 200 rebuild from the per-game
    stats restricted by date (identical, since every play of a game shares the game date)."""
    from nflsim.data.clean import load_plays
    from nflsim.features.team_strength import ratings_from_game_stats

    plays = load_plays()
    sample = ratings.filter(pl.col("games_played") > 0).sample(n=200, seed=3)
    first = sample.row(0, named=True)
    full = build_team_ratings(plays.filter(pl.col("game_date") < first["asof_date"]), write=False)
    row = full.filter((pl.col("team") == first["team"]) & (pl.col("season") == first["season"]) & (pl.col("week") == first["week"]))
    assert row.height == 1
    for s in RATINGS:
        assert abs(first[f"raw_{s}"] - row[f"raw_{s}"][0]) < 1e-12, (first["team"], first["season"], first["week"], s)
    gs = game_stats(plays)
    for r in sample.iter_rows(named=True):
        restricted = gs.filter(pl.col("game_date") < r["asof_date"])
        rebuilt = ratings_from_game_stats(restricted)
        row = rebuilt.filter((pl.col("team") == r["team"]) & (pl.col("season") == r["season"]) & (pl.col("week") == r["week"]))
        assert row.height == 1
        for s in RATINGS:
            a, b = r[f"raw_{s}"], row[f"raw_{s}"][0]
            assert abs(a - b) < 1e-12, (r["team"], r["season"], r["week"], s, a, b)


def test_leakage_changing_a_future_play_does_not_change_earlier_ratings():
    """Spec: changing a play in week W must not change any rating at week <= W."""
    from nflsim.data.clean import load_plays

    plays = load_plays().filter(pl.col("season").is_in([2024, 2025]))
    base = build_team_ratings(plays, write=False)
    W = 9
    target = plays.filter((pl.col("season") == 2025) & (pl.col("week") == W) & (pl.col("play_class") == "pass"))
    gid, pid = target["game_id"][0], target["play_id"][0]
    perturbed = plays.with_columns(
        epa=pl.when((pl.col("game_id") == gid) & (pl.col("play_id") == pid)).then(pl.col("epa") + 50.0).otherwise(pl.col("epa")),
        air_yards=pl.when((pl.col("game_id") == gid) & (pl.col("play_id") == pid)).then(pl.lit(45.0)).otherwise(pl.col("air_yards")),
    )
    pert = build_team_ratings(perturbed, write=False)
    cols = ["team", "season", "week"] + [f"raw_{s}" for s in RATINGS] + RATINGS
    before = base.filter((pl.col("season") < 2025) | (pl.col("week") <= W)).select(cols).sort("season", "week", "team")
    after = pert.filter((pl.col("season") < 2025) | (pl.col("week") <= W)).select(cols).sort("season", "week", "team")
    assert before.equals(after)
    # and the perturbation did reach later weeks (the test has teeth)
    later_b = base.filter((pl.col("season") == 2025) & (pl.col("week") > W)).select(cols).sort("season", "week", "team")
    later_a = pert.filter((pl.col("season") == 2025) & (pl.col("week") > W)).select(cols).sort("season", "week", "team")
    assert not later_b.equals(later_a)


def test_input_scale_applies_only_on_the_simulation_path():
    from nflsim.config import settings
    from nflsim.engine.state import Context, GameState, Phase
    from nflsim.models.features import features, situation_from_row, situation_from_state

    k = settings()["team_strength"]["input_scale"]
    assert k > 1.0
    ctx = Context(home={r: 1.0 for r in RATINGS}, away={r: 0.0 for r in RATINGS})
    s = GameState(home="H", away="A", possession="H", down=1, ydstogo=10, yardline_100=50, qtr=1,
                  game_seconds_remaining=3000, phase=Phase.SCRIMMAGE, context=ctx)
    sit = situation_from_state(s, {}, "rush_yards")
    assert abs(sit.off_ratings["rush_epa"] - k) < 1e-12 and sit.def_ratings["rush_epa_allowed"] == 0.0
    row = {"down": 1, "ydstogo": 10, "yardline_100": 50, "qtr": 1, "game_seconds_remaining": 3000, "half_seconds_remaining": 1200,
           "score_differential": 0, "posteam_timeouts_remaining": 3, "defteam_timeouts_remaining": 3, "neutral": False,
           "posteam": "H", "home_team": "H", "off_rush_epa": 1.0, "def_rush_epa_allowed": 0.0}
    assert situation_from_row(row, "rush_yards").off_ratings["rush_epa"] == 1.0   # training path: unscaled
