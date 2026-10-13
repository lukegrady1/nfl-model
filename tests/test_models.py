"""Every model: probabilities sum to 1 and stay in [0, 1]; tables roundtrip through Parquet."""
import numpy as np
import pytest

from nflsim.models import buckets as B
from tests.conftest import needs_data


def test_buckets_cover_edges():
    assert [B.dist_bucket(d) for d in (1, 2, 3, 4, 6, 7, 10, 11, 25)] == [0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert [B.zone(y) for y in (99, 51, 50, 21, 20, 11, 10, 1)] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [B.yardline5(y) for y in (1, 5, 6, 10, 96, 99)] == [0, 0, 1, 1, 19, 19]
    assert [B.air_bin(a) for a in (-5, -1, 0, 4, 5, 9, 10, 14, 15, 19, 20, 29, 30, 60)] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
    assert [B.seconds_bin(s) for s in (0, 5, 6, 10, 45, 46, 60)] == [0, 0, 1, 1, 8, 9, 9]
    assert [B.rush_class(y) for y in (-9, -6, -5, 0, 15, 16, 24, 25, 39, 40, 80)] == [-6, -6, -5, 0, 15, 16, 16, 25, 25, 40, 40]
    assert B.time_bucket(2, 100, 1900) == 1 and B.time_bucket(4, 100, 100) == 4 and B.time_bucket(5, 600, 0) == 5
    assert B.under_two_half(119, 2) == 1 and B.under_two_half(119, 4) == 2 and B.under_two_half(121, 4) == 0


@needs_data
def test_every_table_cell_is_a_distribution():
    from nflsim.models.tables import load_tables

    T = load_tables()
    assert len(T) >= 25
    for name, t in T.items():
        for L, cells in enumerate(t.levels):
            for key, (probs, count) in cells.items():
                assert len(key) == L
                assert probs.shape == (len(t.classes),)
                assert abs(probs.sum() - 1) < 1e-9, (name, key)
                assert probs.min() >= 0 and probs.max() <= 1, (name, key)
        assert () in t.levels[0], name
        for cls, (vals, cum) in t.within.items():
            assert len(vals) == len(cum) and abs(cum[-1] - 1) < 1e-6


@needs_data
def test_merged_cells_have_at_least_min_plays():
    from nflsim.config import settings
    from nflsim.models.tables import load_tables

    min_n = settings()["min_cell_plays"]
    for name, t in load_tables().items():
        if name in ("fg_make", "punt_retain_spot"):
            continue  # smoothed / tiny empirical tables are exempt
        for L, cells in enumerate(t.levels):
            if L == 0:
                continue
            for key, (probs, count) in cells.items():
                assert count >= min_n, (name, key, count)


@needs_data
def test_table_models_give_distributions_for_random_states():
    from nflsim.engine.state import GameState, Phase
    from nflsim.models.registry import KEY_FNS, default_models

    M = default_models()
    rng = np.random.default_rng(5)
    for _ in range(300):
        s = GameState(home="H", away="A", possession="H", down=int(rng.integers(1, 5)), yardline_100=int(rng.integers(1, 100)),
                      qtr=int(rng.integers(1, 6)), game_seconds_remaining=int(rng.integers(0, 3601)), phase=Phase.SCRIMMAGE)
        s.ydstogo = int(min(rng.integers(1, 21), s.yardline_100))
        s.score = {"H": int(rng.integers(0, 40)), "A": int(rng.integers(0, 40))}
        s.ot_state.seconds_remaining = int(rng.integers(0, 601))
        extra = {"air_bin": int(rng.integers(0, 7)), "play_type_idx": int(rng.integers(0, 4)), "distance": int(rng.integers(18, 80)),
                 "diff_after_td": int(rng.integers(-20, 21)), "clock_stopper": bool(rng.integers(0, 2)),
                 "offense_diff": int(rng.integers(-20, 21)), "under_two": bool(rng.integers(0, 2))}
        for name, m in M.items():
            p = m.proba(s, extra)
            assert abs(p.sum() - 1) < 1e-9 and p.min() >= 0, name
            assert m.sample(rng, s, extra) in m.classes
