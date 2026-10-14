"""Step 2.4: NumPy inference equals scikit-learn predict_proba; Phase 2 sanity scenarios."""
import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LogisticRegression

from nflsim.models.trained import TrainedModel
from tests.conftest import needs_data, needs_models


def _fit_and_compare(name: str, classes: list, n_train: int = 20000, n_val: int = 10000):
    from nflsim.models.train import dataset, encode, feature_matrix, feature_names, rows_of, standardize_params, training_plays

    df, _, _ = dataset(name, training_plays())
    val_season = df["season"].max()
    train = df.filter(pl.col("season") < val_season).sample(n=n_train, seed=1)
    val = df.filter(pl.col("season") == val_season).sample(n=min(n_val, (df["season"] == val_season).sum()), seed=2)
    Xtr, Xva = feature_matrix(name, rows_of(train)), feature_matrix(name, rows_of(val))
    ytr = encode(train, classes)
    mean, std = standardize_params(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=5000).fit((Xtr - mean) / std, ytr)
    spec = {"name": name, "classes": [str(c) for c in classes], "class_type": "int" if isinstance(classes[0], int) else "str",
            "feature_names": feature_names(name), "means": mean.tolist(), "stds": std.tolist(),
            "coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist()}
    tm = TrainedModel(spec)
    ours = tm.proba_from_features(Xva)
    theirs = clf.predict_proba((Xva - mean) / std)
    assert ours.shape == theirs.shape and ours.shape[0] >= 1000
    assert np.abs(ours - theirs).max() < 1e-9
    assert np.allclose(ours.sum(axis=1), 1.0, atol=1e-12)


@needs_data
def test_numpy_inference_matches_sklearn_binary():
    _fit_and_compare("play_call", ["pass", "run"])


@needs_data
def test_numpy_inference_matches_sklearn_multinomial():
    _fit_and_compare("fourth_down", ["go", "punt", "field_goal"])


@needs_models
def test_saved_models_are_valid_distributions():
    from nflsim.engine.state import GameState, Phase
    from nflsim.models.registry import build_models
    from nflsim.models.trained import available_trained

    M = build_models()
    names = [n for n in M if type(M[n]).__name__ == "TrainedModel"]
    assert set(names) == set(available_trained())
    rng = np.random.default_rng(3)
    for _ in range(200):
        s = GameState(home="H", away="A", possession="H", down=int(rng.integers(1, 5)), yardline_100=int(rng.integers(1, 100)),
                      qtr=int(rng.integers(1, 6)), game_seconds_remaining=int(rng.integers(0, 3601)), phase=Phase.SCRIMMAGE)
        s.ydstogo = int(min(rng.integers(1, 21), s.yardline_100))
        s.score = {"H": int(rng.integers(0, 40)), "A": int(rng.integers(0, 40))}
        extra = {"air_yards": int(rng.integers(-5, 40)), "distance": int(rng.integers(18, 70)), "play_type": "run",
                 "yards": int(rng.integers(-10, 30)), "clock_stopper": True, "play_was_pass": False, "out_of_bounds": False,
                 "diff_after_td": int(rng.integers(-20, 21))}
        for n in names:
            p = M[n].proba(s, extra)
            assert abs(p.sum() - 1) < 1e-9 and p.min() >= 0, n
            assert M[n].sample(rng, s, extra) in M[n].classes


@needs_models
def test_sanity_scenarios_down_10_passes_up_10_runs():
    """Phase 2 acceptance: down 10 with 3:00 left in Q4 passes > 80%; up 10 runs > 60% (in simulation)."""
    from nflsim.engine.boxscore import BoxScore
    from nflsim.engine.scrimmage import step
    from nflsim.engine.state import GameState, Phase
    from nflsim.models.registry import build_models

    M = build_models()
    rng = np.random.default_rng(11)

    def drive_mix(diff: int, n: int = 300) -> float:
        passes = runs = 0
        for _ in range(n):
            s = GameState(home="H", away="A", possession="H", down=1, ydstogo=10, yardline_100=75, qtr=4,
                          game_seconds_remaining=180, phase=Phase.SCRIMMAGE, score={"H": 20 + diff, "A": 20})
            box = BoxScore("H", "A")
            while s.phase == Phase.SCRIMMAGE and s.possession == "H" and s.game_seconds_remaining > 0:
                play = step(s, rng, M, box)
                if play and play["type"] == "pass":
                    passes += 1
                elif play and play["type"] in ("run", "kneel"):
                    runs += 1
        return passes / (passes + runs)

    assert drive_mix(-10) > 0.80
    assert 1 - drive_mix(+10) > 0.60


@needs_models
def test_home_logit_offset_direction():
    """Phase 3 step 3.3: with the same situation, the home offense completes more and gains more on the ground."""
    from nflsim.config import settings
    from nflsim.engine.state import GameState, Phase
    from nflsim.models.trained import load_trained

    h = settings()["home_logit_offset"]
    assert h["throw_outcome"] > 0 and h["rush_yards"] > 0
    throw, rush = load_trained("throw_outcome"), load_trained("rush_yards")

    def state(possession: str, neutral: bool = False) -> GameState:
        return GameState(home="H", away="A", possession=possession, down=1, ydstogo=10, yardline_100=60, qtr=2,
                         game_seconds_remaining=2400, phase=Phase.SCRIMMAGE, is_neutral=neutral)

    extra = {"air_yards": 8}
    p_home, p_away = throw.proba(state("H"), extra), throw.proba(state("A"), extra)
    assert p_home[throw.classes.index("complete")] > p_away[throw.classes.index("complete")]
    r_home, r_away = rush.proba(state("H"), {}), rush.proba(state("A"), {})
    ev_home = sum(p * (int(c) if int(c) < 16 else {16: 20, 25: 32, 40: 45}[int(c)]) for p, c in zip(r_home, rush.classes))
    ev_away = sum(p * (int(c) if int(c) < 16 else {16: 20, 25: 32, 40: 45}[int(c)]) for p, c in zip(r_away, rush.classes))
    assert ev_home > ev_away
    # neutral site: no offset, and is_home = 0 for both, so the two offenses are identical
    assert np.allclose(throw.proba(state("H", True), extra), throw.proba(state("A", True), extra))


@needs_models
def test_missing_player_defaults_to_training_mean():
    """A context without players must behave like league-average players, not like zero over-expected players."""
    from nflsim.engine.state import GameState, Phase
    from nflsim.models.trained import load_trained

    m = load_trained("throw_outcome")
    s = GameState(home="H", away="A", possession="H", down=1, ydstogo=10, yardline_100=60, qtr=2, game_seconds_remaining=2400, phase=Phase.SCRIMMAGE)
    p_none = m.proba(s, {"air_yards": 8, "player": None})
    p_zero = m.proba(s, {"air_yards": 8, "player": {}})
    means = {f: float(m.mean[m.feature_names.index(f"p_{f}")]) for f in ("rec_catch_rate_oe", "qb_cpoe", "qb_int_rate")}
    p_mean = m.proba(s, {"air_yards": 8, "player": means})
    assert abs(p_none[0] - p_mean[0]) < 1e-12
    assert p_none[0] > p_zero[0]   # the training-mean receiver catches more than a zero-OE one
