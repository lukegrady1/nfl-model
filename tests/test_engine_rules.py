"""Step 1.8: deterministic rule tests with stub models, plus invariants over random games."""
import numpy as np
import pytest

from nflsim.engine import overtime
from nflsim.engine.boxscore import BoxScore
from nflsim.engine.game import simulate_game
from nflsim.engine.scrimmage import step
from nflsim.engine.state import GameState, OTState, Phase
from nflsim.models.base import FixedModel
from tests.conftest import needs_data

QUIET = {  # every decision forced to a boring outcome; tests override what they need
    "presnap_penalty": "none", "fourth_down": "punt", "play_call": "run", "rush_yards": 3,
    "dropback_outcome": "throw", "air_yards": 5, "throw_outcome": "complete", "yac": 0, "sack_yards": -7,
    "scramble_yards": 5, "fumble_lost": "no", "out_of_bounds": "no", "pick_six": "no", "int_return_yards": 0,
    "fg_make": "made", "punt_result": "net", "punt_net_yards": 40, "punt_retain_spot": 50,
    "kickoff_result": "normal", "kickoff_spot": 70, "pat_choice": "kick", "xp_result": "good",
    "two_point_result": "good", "clock_runoff": 30, "timeout_after": "no", "timeout_runoff": 10,
    "def_penalty_first_down": "no",
}


def stub_models(**overrides) -> dict:
    vals = {**QUIET, **overrides}
    return {name: FixedModel(name, [v], fixed=v) for name, v in vals.items()}


def make_state(**kw) -> GameState:
    base = dict(home="H", away="A", possession="H", down=1, ydstogo=10, yardline_100=50, qtr=1,
                game_seconds_remaining=3000, phase=Phase.SCRIMMAGE, second_half_receiver="A", kicking_team="A")
    base.update(kw)
    return GameState(**base)


def run_step(state, M):
    rng = np.random.default_rng(0)
    box = BoxScore(state.home, state.away)
    play = step(state, rng, M, box)
    return play, box


def test_touchdown_adds_six_then_pat_then_scoring_team_kicks():
    s = make_state(yardline_100=5)
    M = stub_models(rush_yards=10)
    run_step(s, M)
    assert s.score["H"] == 6 and s.phase == Phase.PAT and s.pat_team == "H"
    run_step(s, M)                       # the PAT
    assert s.score["H"] == 7 and s.phase == Phase.KICKOFF and s.kicking_team == "H"
    run_step(s, M)                       # kickoff: A receives at the stub spot
    assert s.possession == "A" and s.yardline_100 == 70 and s.down == 1 and s.ydstogo == 10


def test_safety_adds_two_to_defense_and_conceding_team_free_kicks():
    s = make_state(yardline_100=98)
    M = stub_models(play_call="pass", dropback_outcome="sack", sack_yards=-7)
    run_step(s, M)
    assert s.score["A"] == 2 and s.score["H"] == 0
    assert s.phase == Phase.FREE_KICK and s.kicking_team == "H"
    run_step(s, M)                       # free kick: A gets the ball at the default spot
    assert s.possession == "A" and s.phase == Phase.SCRIMMAGE and s.yardline_100 == 60


def test_turnover_on_downs_flips_possession_at_the_spot():
    s = make_state(down=4, ydstogo=5, yardline_100=40)
    M = stub_models(fourth_down="go", play_call="run", rush_yards=3)
    run_step(s, M)
    assert s.possession == "A" and s.yardline_100 == 63 and s.down == 1 and s.ydstogo == 10


def test_missed_field_goal_spots():
    s = make_state(down=4, ydstogo=8, yardline_100=30)
    M = stub_models(fourth_down="field_goal", fg_make="missed")
    play, _ = run_step(s, M)
    assert play["distance"] == 47
    assert s.possession == "A" and s.yardline_100 == 63        # spot of the kick: 100 - (30 + 7)
    s = make_state(down=4, ydstogo=8, yardline_100=10)
    run_step(s, M)
    assert s.possession == "A" and s.yardline_100 == 80        # spot of the kick inside the 20: ball at the 20
    s = make_state(down=4, ydstogo=8, yardline_100=20)
    M = stub_models(fourth_down="field_goal", fg_make="made")
    run_step(s, M)
    assert s.score["H"] == 3 and s.phase == Phase.KICKOFF and s.kicking_team == "H"


def test_interception_flips_the_field():
    s = make_state(yardline_100=60)
    M = stub_models(play_call="pass", air_yards=10, throw_outcome="interception", int_return_yards=5)
    run_step(s, M)
    # caught at offense-frame spot 50, returned 5 -> defense is 45 from its target end zone
    assert s.possession == "A" and s.yardline_100 == 45 and s.down == 1
    s = make_state(yardline_100=8)                              # end-zone interception: touchback
    M = stub_models(play_call="pass", air_yards=12, throw_outcome="interception", int_return_yards=0)
    run_step(s, M)
    assert s.possession == "A" and s.yardline_100 == 80
    s = make_state(yardline_100=60)
    M = stub_models(play_call="pass", air_yards=10, throw_outcome="interception", pick_six="yes")
    run_step(s, M)
    assert s.score["A"] == 6 and s.phase == Phase.PAT and s.pat_team == "A"


def test_goal_to_go_sets_ydstogo_to_yardline():
    s = make_state(yardline_100=18)
    run_step(s, stub_models(rush_yards=10))
    assert s.yardline_100 == 8 and s.down == 1 and s.ydstogo == 8 and s.goal_to_go
    s = make_state(yardline_100=30, ydstogo=4, down=2)
    run_step(s, stub_models(rush_yards=2))                      # short of the sticks
    assert s.down == 3 and s.ydstogo == 2 and s.yardline_100 == 28


def test_second_half_kickoff_goes_to_the_opening_kicker_and_timeouts_reset():
    s = make_state(qtr=2, game_seconds_remaining=1802, second_half_receiver="A", timeouts={"H": 1, "A": 0})
    M = stub_models(clock_runoff=30)
    run_step(s, M)                        # play ends the half
    assert s.game_seconds_remaining == 1800 and s.period_expired
    run_step(s, M)                        # halftime transition + second-half kickoff
    assert s.qtr == 3 and s.timeouts == {"H": 3, "A": 3}
    assert s.possession == "A" and s.phase == Phase.SCRIMMAGE   # A (the opening kicker) received
    assert s.game_seconds_remaining == 1800 - 30


def test_touchdown_on_last_play_of_half_still_gets_the_try():
    s = make_state(qtr=2, game_seconds_remaining=1803, yardline_100=5)
    M = stub_models(rush_yards=10, clock_runoff=30)
    run_step(s, M)
    assert s.phase == Phase.PAT and s.period_expired
    run_step(s, M)                        # PAT still happens before halftime
    assert s.score["H"] == 7 and s.phase == Phase.KICKOFF
    run_step(s, M)
    assert s.qtr == 3


def test_two_minute_warning_stops_clock_at_120_once_per_half():
    s = make_state(qtr=4, game_seconds_remaining=130)
    M = stub_models(clock_runoff=30)
    run_step(s, M)
    assert s.game_seconds_remaining == 120 and s.two_minute_warning_used[2]
    run_step(s, M)
    assert s.game_seconds_remaining == 90
    s = make_state(qtr=2, game_seconds_remaining=1800 + 125)
    run_step(s, M)
    assert s.game_seconds_remaining == 1800 + 120 and s.two_minute_warning_used[1]


def test_quarter_boundary_continues_the_drive():
    s = make_state(qtr=1, game_seconds_remaining=2710, down=2, ydstogo=7, yardline_100=40)
    run_step(s, stub_models(rush_yards=3, clock_runoff=30))
    assert s.qtr == 2 and s.game_seconds_remaining == 2700
    assert s.possession == "H" and s.down == 3 and s.ydstogo == 4 and s.yardline_100 == 37


def test_game_ends_at_zero_when_not_tied_and_goes_to_ot_when_tied():
    s = make_state(qtr=4, game_seconds_remaining=10, score={"H": 10, "A": 7})
    M = stub_models(clock_runoff=30)
    run_step(s, M)
    assert s.game_seconds_remaining == 0 and s.period_expired
    run_step(s, M)
    assert s.phase == Phase.GAME_OVER
    s = make_state(qtr=4, game_seconds_remaining=10, score={"H": 10, "A": 10})
    run_step(s, M)
    run_step(s, M)                        # transition to OT and the OT kickoff happen in this step
    assert s.qtr == 5 and s.ot_state.active and s.ot_state.possession_count == 1 and s.phase == Phase.SCRIMMAGE


# ---- overtime, one test per numbered rule in step 1.6 --------------------------------------------------------
def ot_state(receiver="H", **kw) -> GameState:
    s = make_state(qtr=5, possession=receiver, **kw)
    s.ot_state = OTState(active=True, first_team=receiver, possession_count=1, period=1, seconds_remaining=600)
    s.timeouts = {"H": 2, "A": 2}
    return s


def test_ot_rule1_coin_toss_random_receiver_kicks_off():
    receivers = set()
    for seed in range(20):
        s = make_state(qtr=4, game_seconds_remaining=0, score={"H": 3, "A": 3})
        overtime.start_overtime(s, np.random.default_rng(seed), BoxScore("H", "A"))
        assert s.qtr == 5 and s.phase == Phase.KICKOFF
        assert s.kicking_team == s.opponent(s.ot_state.first_team) and s.possession == s.ot_state.first_team
        assert s.ot_state.seconds_remaining == 600 and s.timeouts == {"H": 2, "A": 2}
        receivers.add(s.ot_state.first_team)
    assert receivers == {"H", "A"}


def test_ot_rule2_both_teams_possess_unless_defensive_td_or_safety_on_first_possession():
    # (a) first team scores a TD on its first possession: game continues, other team gets the ball
    s = ot_state(yardline_100=5)
    M = stub_models(rush_yards=10)
    run_step(s, M)
    run_step(s, M)                        # PAT
    assert s.phase == Phase.KICKOFF and s.score["H"] == 7
    run_step(s, M)                        # kickoff to A
    assert s.possession == "A" and s.ot_state.possession_count == 2 and s.phase == Phase.SCRIMMAGE
    # (b) pick-six on the first possession ends the game
    s = ot_state(yardline_100=60)
    M = stub_models(play_call="pass", air_yards=10, throw_outcome="interception", pick_six="yes")
    run_step(s, M)
    run_step(s, M)                        # try is skipped, game over
    assert s.phase == Phase.GAME_OVER and s.score["A"] == 6
    # (c) safety on the first possession ends the game
    s = ot_state(yardline_100=98)
    M = stub_models(play_call="pass", dropback_outcome="sack", sack_yards=-7)
    run_step(s, M)
    assert s.phase == Phase.GAME_OVER and s.score["A"] == 2


def test_ot_rule3_after_both_possessions_next_score_wins():
    s = ot_state(down=4, yardline_100=60)
    M = stub_models(fourth_down="punt", punt_net_yards=40)
    run_step(s, M)                        # H punts: A's possession (count 2)
    assert s.possession == "A" and s.ot_state.possession_count == 2
    s.down = 4
    run_step(s, M)                        # A punts: H's possession (count 3), still tied -> play on
    assert s.possession == "H" and s.ot_state.possession_count == 3 and s.phase == Phase.SCRIMMAGE
    s.down, s.yardline_100 = 4, 20
    run_step(s, stub_models(fourth_down="field_goal", fg_make="made"))
    assert s.phase == Phase.GAME_OVER and s.score["H"] == 3
    # a lead held when the second possession ends also ends it
    s = ot_state(down=4, yardline_100=20)
    run_step(s, stub_models(fourth_down="field_goal", fg_make="made"))     # H FG on possession 1
    run_step(s, stub_models())                                           # kickoff -> A (count 2)
    s.down = 4
    run_step(s, stub_models(fourth_down="punt"))                         # A punts without scoring
    assert s.phase == Phase.GAME_OVER and s.score == {"H": 3, "A": 0}


def test_ot_rule4_tied_after_two_possessions_then_next_score_wins():
    s = ot_state(down=4, yardline_100=20)
    fg = stub_models(fourth_down="field_goal", fg_make="made")
    run_step(s, fg)                       # H FG
    run_step(s, stub_models())            # kickoff to A
    s.down, s.yardline_100 = 4, 20
    run_step(s, fg)                       # A FG: tied, game continues
    assert s.phase == Phase.KICKOFF and s.tied
    run_step(s, stub_models())            # kickoff to H (count 3)
    assert s.phase == Phase.SCRIMMAGE and s.ot_state.possession_count == 3
    s.down, s.yardline_100 = 4, 20
    run_step(s, fg)                       # next score wins
    assert s.phase == Phase.GAME_OVER and s.score == {"H": 6, "A": 3}
    # second team trailing by 7 answers with a TD: the try decides (2-pt good wins, XP kick ties -> sudden death)
    s = ot_state(yardline_100=5, score={"H": 0, "A": 7})
    s.ot_state.possession_count = 2
    s.ot_state.first_team = "A"
    run_step(s, stub_models(rush_yards=10))
    run_step(s, stub_models(pat_choice="two", two_point_result="good"))
    assert s.phase == Phase.GAME_OVER and s.score == {"H": 8, "A": 7}


def test_ot_rule5_regular_season_tie_and_playoff_extra_period():
    s = ot_state()
    s.ot_state.seconds_remaining = 10
    M = stub_models(clock_runoff=30)
    run_step(s, M)
    run_step(s, M)
    assert s.phase == Phase.GAME_OVER and s.tied
    s = ot_state(playoff=True)
    s.ot_state.seconds_remaining = 10
    run_step(s, M)
    run_step(s, M)
    assert s.phase != Phase.GAME_OVER and s.qtr == 6 and s.ot_state.period == 2
    assert s.ot_state.seconds_remaining == 900 - 30 and s.possession == "H"


def test_presnap_penalties():
    s = make_state(yardline_100=50, down=2, ydstogo=6)
    run_step(s, stub_models(presnap_penalty="offense"))
    assert (s.down, s.ydstogo, s.yardline_100) == (2, 11, 55)
    s = make_state(yardline_100=50, down=2, ydstogo=6)
    run_step(s, stub_models(presnap_penalty="defense"))
    assert (s.down, s.ydstogo, s.yardline_100) == (2, 1, 45)           # down is replayed
    s = make_state(yardline_100=4, down=2, ydstogo=4)
    run_step(s, stub_models(presnap_penalty="defense"))                 # half the distance
    assert (s.down, s.ydstogo, s.yardline_100) == (2, 2, 2)
    s = make_state(yardline_100=1, down=2, ydstogo=1)
    run_step(s, stub_models(presnap_penalty="defense"))                 # from the 1: no yardage to give
    assert (s.down, s.ydstogo, s.yardline_100) == (2, 1, 1)
    s = make_state(yardline_100=50, down=2, ydstogo=6)
    run_step(s, stub_models(presnap_penalty="defense", def_penalty_first_down="yes"))
    assert (s.down, s.ydstogo, s.yardline_100) == (1, 10, 45)


def test_kneel_rule_runs_out_the_clock():
    s = make_state(qtr=4, game_seconds_remaining=70, score={"H": 10, "A": 7}, timeouts={"H": 3, "A": 0})
    play, _ = run_step(s, stub_models())
    assert play["type"] == "kneel" and s.game_seconds_remaining == 30 and s.down == 2
    s = make_state(qtr=4, game_seconds_remaining=70, score={"H": 10, "A": 7}, timeouts={"H": 3, "A": 3})
    play, _ = run_step(s, stub_models())
    assert play["type"] != "kneel"        # opponent can stop the clock


def test_fumble_lost_flips_possession_at_the_spot():
    s = make_state(yardline_100=50)
    run_step(s, stub_models(rush_yards=4, fumble_lost="yes"))
    assert s.possession == "A" and s.yardline_100 == 54
    s = make_state(yardline_100=50)
    run_step(s, stub_models(rush_yards=4, fumble_lost="yes_td"))
    assert s.score["A"] == 6 and s.pat_team == "A"


# ---- invariants over random games with the real tables -----------------------------------------------------
@needs_data
def test_invariants_hold_on_every_snap_across_1000_random_games():
    from nflsim.models.registry import default_models

    M = default_models()
    rng = np.random.default_rng(123)
    from nflsim.engine.game import coin_toss
    from nflsim.engine.state import new_game

    for g in range(1000):
        s = new_game("H", "A", playoff=(g % 10 == 0))
        box = BoxScore("H", "A")
        coin_toss(s, rng)
        n = 0
        while s.phase != Phase.GAME_OVER:
            step(s, rng, M, box)
            n += 1
            assert n < 600
            assert s.half_seconds_remaining >= 0 and s.game_seconds_remaining >= 0
            assert all(0 <= t <= 3 for t in s.timeouts.values())
            if s.phase == Phase.SCRIMMAGE:
                assert 1 <= s.yardline_100 <= 99, s
                assert 1 <= s.down <= 4, s
                assert 1 <= s.ydstogo <= s.yardline_100, s
        assert s.score["H"] >= 0 and s.score["A"] >= 0


@needs_data
def test_same_seed_gives_identical_games():
    from nflsim.models.registry import default_models

    M = default_models()
    a = [simulate_game("KC", "BUF", M, np.random.default_rng(99)).team_stats for _ in range(1)]
    r1 = [simulate_game("KC", "BUF", M, seed=1000 + i) for i in range(30)]
    r2 = [simulate_game("KC", "BUF", M, seed=1000 + i) for i in range(30)]
    assert [(r.home_score, r.away_score, r.team_stats) for r in r1] == [(r.home_score, r.away_score, r.team_stats) for r in r2]
    assert a  # silence unused warning
