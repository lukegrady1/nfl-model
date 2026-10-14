"""Phase 5: the jitted feature builder equals the Python one; engine equivalence and speed."""
import numpy as np
import pytest

from tests.conftest import needs_data, needs_models

pytestmark = [needs_data, needs_models]


def _scalars(state, extra, name):
    """The engine's mapping from a GameState + extra to build_features inputs (mirrors situation_from_state)."""
    from nflsim.sim.pack import PLAYER_EFF_ORDER, SLOT
    off = extra.get("offense") or state.possession
    deff = state.opponent(off)
    gsr = state.ot_state.seconds_remaining if state.in_ot else state.game_seconds_remaining
    w = state.context.weather or {}
    pf = np.array([float((extra.get("player") or {}).get(f, 0.0) or 0.0) for f in PLAYER_EFF_ORDER])
    kind = {"punt": 1, "field_goal": 2, "penalty": 3}.get(extra.get("play_type_kind"), 0)
    return dict(slot=SLOT[name], down=state.down, ydstogo=state.ydstogo, yl=state.yardline_100, qtr=state.qtr, gsr=gsr,
                half_secs=state.half_seconds_remaining, score_diff=state.score[off] - state.score[deff],
                off_to=state.timeouts[off], def_to=state.timeouts[deff], is_home=0 if state.is_neutral else int(off == state.home),
                season=int(state.context.season or 0), air=int(extra.get("air_yards", 0) or 0), kick_distance=int(extra.get("distance", 0) or 0),
                roof_dome=int(w.get("roof_dome", 0) or 0), wind=-1.0 if w.get("wind") is None else float(w["wind"]),
                pt=int(extra.get("play_type_idx", 0) or 0), yards=int(extra.get("yards", 0) or 0), cs=bool(extra.get("clock_stopper", False)),
                play_was_pass=bool(extra.get("play_was_pass", False)), oob=bool(extra.get("out_of_bounds", False)), play_kind=kind,
                diff_after_td=int(extra.get("diff_after_td", 0) or 0), pf=pf)


def test_jitted_features_equal_python_features():
    from nflsim.config import settings
    from nflsim.engine.state import Context, GameState, Phase
    from nflsim.features.team_strength import RATINGS
    from nflsim.models.features import features, situation_from_state
    from nflsim.models.registry import default_models
    from nflsim.sim.fast_features import MAX_FEATURES, build_features
    from nflsim.sim.pack import SLOTS, pack_models

    M = default_models()
    P = pack_models(M)
    scale = settings()["team_strength"]["input_scale"]
    rng = np.random.default_rng(42)
    out = np.zeros(MAX_FEATURES)
    for _ in range(400):
        ctx = Context(home={r: float(rng.normal()) for r in RATINGS}, away={r: float(rng.normal()) for r in RATINGS},
                      weather={"roof_dome": int(rng.integers(0, 2)), "wind": None if rng.random() < 0.3 else float(rng.integers(0, 25))},
                      season=int(rng.choice([2024, 2025, 2026])))
        s = GameState(home="H", away="A", possession=str(rng.choice(["H", "A"])), down=int(rng.integers(1, 5)), yardline_100=int(rng.integers(1, 100)),
                      qtr=int(rng.integers(1, 6)), game_seconds_remaining=int(rng.integers(0, 3601)), phase=Phase.SCRIMMAGE,
                      is_neutral=bool(rng.random() < 0.1), context=ctx)
        s.ydstogo = int(min(rng.integers(1, 21), s.yardline_100))
        s.score = {"H": int(rng.integers(0, 40)), "A": int(rng.integers(0, 40))}
        s.timeouts = {"H": int(rng.integers(0, 4)), "A": int(rng.integers(0, 4))}
        if s.qtr >= 5:
            s.ot_state.seconds_remaining = int(rng.integers(0, 601))
        pt = int(rng.integers(0, 4))
        extra = {"air_yards": int(rng.integers(-5, 45)), "distance": int(rng.integers(18, 70)), "play_type_idx": pt,
                 "play_type": ["completion", "sack", "scramble", "run"][pt], "yards": int(rng.integers(-10, 30)), "clock_stopper": bool(rng.random() < 0.5), "play_was_pass": bool(rng.random() < 0.5),
                 "out_of_bounds": bool(rng.random() < 0.2), "play_type_kind": str(rng.choice(["pass", "punt", "field_goal", "penalty"])),
                 "diff_after_td": int(rng.integers(-20, 21)), "offense": str(rng.choice(["H", "A"])),
                 "player": {"qb_cpoe": 0.02, "qb_sack_rate": -0.01, "qb_int_rate": 0.003, "qb_scramble_rate": 0.04, "qb_adot": 0.5,
                            "rec_catch_rate_oe": 0.03, "rec_yac_oe": -0.4, "rush_yards_oe": 0.2, "kicker_fg_oe": 0.01}}
        off = extra["offense"]
        off_r = np.array([ctx.home[r] if off == "H" else ctx.away[r] for r in RATINGS]) * scale
        def_r = np.array([ctx.away[r] if off == "H" else ctx.home[r] for r in RATINGS]) * scale
        for i, name in enumerate(SLOTS):
            if P.kind[i] != 1:
                continue
            m = M[name]
            ref = features(name, situation_from_state(s, extra, name), m.team)
            sc = _scalars(s, extra, name)
            n = build_features(sc["slot"], out, sc["down"], sc["ydstogo"], sc["yl"], sc["qtr"], sc["gsr"], sc["half_secs"], sc["score_diff"],
                               sc["off_to"], sc["def_to"], sc["is_home"], sc["season"], sc["air"], sc["kick_distance"], sc["roof_dome"], sc["wind"],
                               sc["pt"], sc["yards"], sc["cs"], sc["play_was_pass"], sc["oob"], sc["play_kind"], sc["diff_after_td"],
                               off_r, def_r, sc["pf"], P.team[i], P.team_off_idx[i], P.team_def_idx[i], P.pf_idx[i], P.weather_flag[i])
            assert n == len(ref) == P.n_features[i], (name, n, len(ref))
            assert np.allclose(out[:n], ref, atol=1e-12, rtol=0), (name, np.abs(out[:n] - ref).argmax())


def test_jitted_probabilities_equal_python_probabilities():
    """trained_probs (with mask + home offset) and dense tables reproduce the Python models' proba."""
    from nflsim.engine.state import Context, GameState, Phase
    from nflsim.features.team_strength import RATINGS
    from nflsim.models.registry import KEY_FNS, default_models
    from nflsim.models.trained import room_for
    from nflsim.sim.fast_features import MAX_FEATURES, build_features, trained_probs
    from nflsim.sim.pack import MAX_CLASSES, SLOTS, pack_models
    from nflsim.config import settings

    M = default_models()
    P = pack_models(M)
    scale = settings()["team_strength"]["input_scale"]
    rng = np.random.default_rng(7)
    out = np.zeros(MAX_FEATURES)
    probs = np.zeros(MAX_CLASSES)
    for _ in range(300):
        ctx = Context(home={r: float(rng.normal()) for r in RATINGS}, away={r: float(rng.normal()) for r in RATINGS}, season=2025)
        s = GameState(home="H", away="A", possession=str(rng.choice(["H", "A"])), down=int(rng.integers(1, 5)), yardline_100=int(rng.integers(1, 100)),
                      qtr=int(rng.integers(1, 5)), game_seconds_remaining=int(rng.integers(1, 3601)), phase=Phase.SCRIMMAGE,
                      is_neutral=bool(rng.random() < 0.2), context=ctx)
        s.ydstogo = int(min(rng.integers(1, 21), s.yardline_100))
        s.score = {"H": int(rng.integers(0, 40)), "A": int(rng.integers(0, 40))}
        pt = int(rng.integers(0, 4))
        extra = {"air_yards": int(rng.integers(-5, 45)), "distance": int(rng.integers(18, 70)), "play_type_idx": pt,
                 "play_type": ["completion", "sack", "scramble", "run"][pt], "yards": 3, "clock_stopper": True, "play_was_pass": True, "out_of_bounds": False, "diff_after_td": 3, "player": {}}
        extra["air_bin"] = max(0, min(6, (extra["air_yards"] + 1) // 5))
        off = s.possession
        off_r = np.array([ctx.home[r] if off == "H" else ctx.away[r] for r in RATINGS]) * scale
        def_r = np.array([ctx.away[r] if off == "H" else ctx.home[r] for r in RATINGS]) * scale
        home_sign = 0.0 if s.is_neutral else (1.0 if off == "H" else -1.0)
        for i, name in enumerate(SLOTS):
            if P.kind[i] != 1:
                continue
            ref = M[name].proba(s, extra)
            sc = _scalars(s, extra, name)
            n = build_features(sc["slot"], out, sc["down"], sc["ydstogo"], sc["yl"], sc["qtr"], sc["gsr"], sc["half_secs"], sc["score_diff"],
                               sc["off_to"], sc["def_to"], sc["is_home"], sc["season"], sc["air"], sc["kick_distance"], sc["roof_dome"], sc["wind"],
                               sc["pt"], sc["yards"], sc["cs"], sc["play_was_pass"], sc["oob"], sc["play_kind"], sc["diff_after_td"],
                               off_r, def_r, sc["pf"], P.team[i], P.team_off_idx[i], P.team_def_idx[i], P.pf_idx[i], P.weather_flag[i])
            room = room_for(name, s.yardline_100, extra.get("air_yards")) if P.mask_kind[i] else 0
            trained_probs(out, n, P.mean[i], P.std[i], np.atleast_2d(P.W[i]), P.b[i], P.binary[i], probs, P.n_classes[i], P.mask_kind[i],
                          P.lower_edge[i], room, P.home_dir[i], 0.5 * P.home_offset[i] * home_sign)
            assert np.allclose(probs[:P.n_classes[i]], ref, atol=1e-10), name
