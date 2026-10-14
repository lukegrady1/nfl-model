"""Phase 4 tests: leakage of player inputs, box-score consistency, injury scenario, actives."""
import json

import numpy as np
import polars as pl
import pytest

from tests.conftest import needs_data, needs_models

pytestmark = [needs_data, needs_models]


@pytest.fixture(scope="module")
def usage():
    from nflsim.features.players import load_player_usage
    return load_player_usage()


def test_player_shares_use_only_games_before_the_week(usage):
    """Spec leakage test: shares and efficiencies for week W use only games before week W (as-of date)."""
    from nflsim.features.players import _asof_rows, player_game_stats, roster_grid
    from nflsim.data.clean import load_plays, load_schedules

    plays = load_plays().filter(pl.col("season").is_in([2024, 2025]))
    stats = player_game_stats(plays)
    sched = load_schedules().filter(pl.col("game_type") != "PRE")
    grid = roster_grid().filter((pl.col("season") == 2025) & (pl.col("week") == 9) & (pl.col("team") == "BUF"))
    full = _asof_rows(stats, sched, grid)
    buf_date = sched.filter((pl.col("season") == 2025) & (pl.col("week") == 9) & ((pl.col("home_team") == "BUF") | (pl.col("away_team") == "BUF")))["gameday"][0]
    restricted = _asof_rows(stats.filter(pl.col("game_date") < buf_date), sched, grid)
    assert full.sort("player_id").equals(restricted.sort("player_id"))
    # perturbing a week-9 BUF game must not change week 9 rows but must change week 10 rows
    gid = sched.filter((pl.col("season") == 2025) & (pl.col("week") == 9) & ((pl.col("home_team") == "BUF") | (pl.col("away_team") == "BUF")))["game_id"][0]
    pert = stats.with_columns(targets=pl.when(pl.col("game_id") == gid).then(pl.col("targets") + 5).otherwise(pl.col("targets")))
    assert _asof_rows(pert, sched, grid).sort("player_id").equals(full.sort("player_id"))
    grid10 = roster_grid().filter((pl.col("season") == 2025) & (pl.col("week") == 10) & (pl.col("team") == "BUF"))
    assert not _asof_rows(pert, sched, grid10).sort("player_id").equals(_asof_rows(stats, sched, grid10).sort("player_id"))


def test_box_score_consistency_over_random_games():
    """Receiving yards sum to the QB's passing yards; rushing yards sum to the team's; sacks never touch QB yards."""
    from nflsim.engine.game import simulate_game
    from nflsim.models.registry import default_models
    from nflsim.sim.game_inputs import WeekInputs

    M = default_models()
    wi = WeekInputs(2025, 11)
    ctx = wi.context("BUF", "KC")
    rng = np.random.default_rng(5)
    for _ in range(60):
        r = simulate_game("BUF", "KC", M, rng, context=ctx)
        for team in ("BUF", "KC"):
            tp = ctx.players[team]
            ps = r.players[team]
            qb = ps.get(tp.qb, {})
            assert sum(s["rec_yards"] for s in ps.values()) == qb.get("pass_yards", 0) == r.team_stats[team]["pass_yards"]
            assert sum(s["receptions"] for s in ps.values()) == qb.get("completions", 0) == r.team_stats[team]["completions"]
            assert sum(s["targets"] for s in ps.values()) == r.team_stats[team]["pass_attempts"]
            assert sum(s["rush_yards"] for s in ps.values()) == r.team_stats[team]["rush_yards"]
            assert sum(s["rush_attempts"] for s in ps.values()) == r.team_stats[team]["rush_attempts"]
            assert sum(s["rec_tds"] for s in ps.values()) == qb.get("pass_tds", 0)
            assert r.team_stats[team]["sack_yards"] >= 0 and qb.get("pass_yards", 0) >= 0
            for s in ps.values():
                assert s["anytime_td"] == (1 if s["rec_tds"] + s["rush_tds"] > 0 else 0)


def test_injury_scenario_wr1_out_redistributes_targets():
    """Removing the WR1 raises the other receivers' targets and keeps the team's total targets within 1."""
    from nflsim.engine.game import simulate_game
    from nflsim.models.registry import default_models
    from nflsim.sim.game_inputs import WeekInputs

    M = default_models()
    wi = WeekInputs(2025, 11)
    base_ctx = wi.context("BUF", "KC")
    tp = base_ctx.players["BUF"]
    wr1 = max((p for p in tp.players if tp.players[p]["position"] == "WR"), key=lambda p: tp.players[p]["target_share"])
    actives2 = {t: dict(v) for t, v in wi.actives.items()}
    actives2["BUF"] = {**actives2["BUF"], "players": [p for p in actives2["BUF"]["players"] if p["player_id"] != wr1]}
    wi2 = WeekInputs(2025, 11, actives=actives2)
    ctx2 = wi2.context("BUF", "KC")
    assert wr1 not in ctx2.players["BUF"].players
    assert abs(sum(p["target_share"] for p in ctx2.players["BUF"].players.values()) - 1.0) < 1e-9

    def targets(ctx, n=400):
        rng = np.random.default_rng(9)
        tot = {}
        team_total = 0.0
        for _ in range(n):
            r = simulate_game("BUF", "KC", M, rng, context=ctx)
            team_total += r.team_stats["BUF"]["pass_attempts"]
            for pid, s in r.players["BUF"].items():
                tot[pid] = tot.get(pid, 0) + s["targets"]
        return {k: v / n for k, v in tot.items()}, team_total / n

    t1, total1 = targets(base_ctx)
    t2, total2 = targets(ctx2)
    others = [p for p in tp.target_ids if p != wr1 and tp.players[p]["position"] in ("WR", "TE", "RB")]
    raised = sum(1 for p in others if t2.get(p, 0) > t1.get(p, 0))
    assert raised >= 0.7 * len(others), (raised, len(others))
    assert abs(total1 - total2) <= 1.0, (total1, total2)
    assert t1[wr1] > 3


def test_actives_json_and_starting_qb(tmp_path):
    from nflsim.features.actives import build_actives
    from nflsim.features.players import load_player_eff

    eff = load_player_eff()
    a = build_actives(2025, 10, eff.filter((pl.col("season") == 2025) & (pl.col("week") == 10)), write=False, quiet=True)
    assert "BUF" in a and a["BUF"]["starting_qb"] == "00-0034857"   # Josh Allen
    for team, v in a.items():
        assert v["starting_qb"] is None or any(p["player_id"] == v["starting_qb"] and p["position"] == "QB" for p in v["players"])
        assert all(p["report_status"] not in ("Out", "Doubtful") for p in v["players"])


def test_bin_fit_and_priors_sane():
    from nflsim.features.players import load_position_priors
    from nflsim.sim.game_inputs import WeekInputs

    pri = load_position_priors()
    wr = pri.filter(pl.col("position") == "WR").sort("slot")
    assert wr["target_share"].to_list() == sorted(wr["target_share"].to_list(), reverse=True)
    wi = WeekInputs(2025, 11)
    tp = wi.context("KC", "BUF").players["KC"]
    for p in tp.players.values():
        assert all(f > 0 for f in p["bin_fit"]) and 0 <= p["target_share"] <= 1 and 0 <= p["carry_share"] <= 1
    assert sum(1 for p in tp.players.values() if p["position"] == "QB" and p["target_share"] > 0) == 0
