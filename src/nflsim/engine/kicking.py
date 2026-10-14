"""Kickoffs, free kicks, punts, field goals and PATs."""
from __future__ import annotations

import numpy as np

from nflsim.config import settings
from nflsim.engine import overtime
from nflsim.engine.state import GameState, Phase
from nflsim.models.registry import sample_exact

ONSIDE_RECOVERY_YARDLINE = 55   # kicking team recovers its own kick around its 45


def start_possession(state: GameState, team: str, yardline_100: int, box) -> None:
    """New drive for `team` with 1st and 10 (or goal) at yardline_100. Counts OT possessions."""
    state.possession = team
    state.down = 1
    state.yardline_100 = int(min(max(yardline_100, 1), 99))
    state.ydstogo = min(10, state.yardline_100)
    state.phase = Phase.SCRIMMAGE
    box.add(team, "drives")
    state.drive += 1
    if state.ot_state.active:
        state.ot_state.possession_count += 1
        overtime.check_ot_end(state, None)


def touchdown(state: GameState, team: str, box) -> None:
    state.score[team] += 6
    box.add(team, "touchdowns")
    state.phase = Phase.PAT
    state.pat_team = team


def safety(state: GameState, conceding_team: str, box) -> None:
    scorer = state.opponent(conceding_team)
    state.score[scorer] += 2
    state.phase = Phase.FREE_KICK
    state.kicking_team = conceding_team
    state.possession = scorer
    overtime.check_ot_end(state, scoring_team=scorer)


def kickoff(state: GameState, rng: np.random.Generator, M: dict, box, free_kick: bool = False) -> dict:
    kicker = state.kicking_team
    receiver = state.opponent(kicker)
    state.possession = receiver
    play = {"type": "free_kick" if free_kick else "kickoff", "clock_stopper": True, "offense": receiver}
    if free_kick:
        if "free_kick_spot" in M:
            spot = sample_exact(M, "free_kick_spot", rng, state)
        else:
            spot = settings()["engine"]["free_kick_default_yardline"]
        start_possession(state, receiver, spot, box)
        return play
    res = M["kickoff_result"].sample(rng, state, {})
    if res == "return_td":
        touchdown(state, receiver, box)
        play["result"] = "return_td"
    elif res == "kicking_team_ball":
        start_possession(state, kicker, ONSIDE_RECOVERY_YARDLINE, box)
        play["result"] = "kicking_team_ball"
    else:
        spot = sample_exact(M, "kickoff_spot", rng, state)
        start_possession(state, receiver, spot, box)
        play["result"] = spot
    return play


def pat(state: GameState, rng: np.random.Generator, M: dict, box) -> dict:
    team = state.pat_team
    opp = state.opponent(team)
    state.possession = team
    play = {"type": "pat", "clock_stopper": True, "no_clock": True, "offense": team}
    diff_after = state.score[team] - state.score[opp]
    decided = overtime.ot_decided_before_try(state, team) or \
        (state.period_expired and state.qtr == 4 and diff_after > 0)
    if not decided:
        choice = M["pat_choice"].sample(rng, state, {"diff_after_td": diff_after})
        if choice == "two":
            box.add(team, "two_point_attempts")
            good = M["two_point_result"].sample(rng, state, {}) == "good"
            if good:
                state.score[team] += 2
            play["result"] = f"two_point_{'good' if good else 'no_good'}"
        else:
            box.add(team, "xp_attempts")
            good = M["xp_result"].sample(rng, state, {}) == "good"
            if good:
                state.score[team] += 1
            play["result"] = f"xp_{'good' if good else 'no_good'}"
    else:
        play["result"] = "no_try"
    state.phase = Phase.KICKOFF
    state.kicking_team = team
    overtime.check_ot_end(state, scoring_team=team)
    return play


def punt(state: GameState, rng: np.random.Generator, M: dict, box) -> dict:
    off, deff = state.possession, state.defense
    y = state.yardline_100
    box.add(off, "punts")
    play = {"type": "punt", "clock_stopper": True, "offense": off}
    res = M["punt_result"].sample(rng, state, {})
    play["result"] = res
    if res == "touchback":
        start_possession(state, deff, 80, box)
    elif res == "return_td":
        touchdown(state, deff, box)
    elif res == "retain":
        start_possession(state, off, sample_exact(M, "punt_retain_spot", rng, state), box)
    else:
        net = sample_exact(M, "punt_net_yards", rng, state)
        r = 100 - y + net
        if r <= 0:
            r = 80  # punted into the end zone
        start_possession(state, deff, r, box)
        play["net"] = net
    return play


def field_goal(state: GameState, rng: np.random.Generator, M: dict, box) -> dict:
    off, deff = state.possession, state.defense
    dist = state.yardline_100 + 17
    box.add(off, "fg_attempts")
    play = {"type": "field_goal", "clock_stopper": True, "offense": off, "distance": dist}
    tp = state.context.players.get(off) if state.context.players else None
    made = M["fg_make"].sample(rng, state, {"distance": min(dist, 79), "player": tp.kicker_eff() if tp else None}) == "made"
    if made:
        state.score[off] += 3
        box.add(off, "fg_made")
        state.phase = Phase.KICKOFF
        state.kicking_team = off
        overtime.check_ot_end(state, scoring_team=off)
        play["result"] = "made"
    else:
        opp_yl = min(100 - state.yardline_100 - 7, 80)
        start_possession(state, deff, opp_yl, box)
        play["result"] = "missed"
    return play
