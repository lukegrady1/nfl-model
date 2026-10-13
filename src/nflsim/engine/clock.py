"""Step 1.5: clock rules. Called after every play."""
from __future__ import annotations

import numpy as np

from nflsim.config import settings
from nflsim.engine.state import GameState, Phase
from nflsim.models.registry import sample_exact


def advance(state: GameState, play: dict, rng: np.random.Generator, M: dict, box) -> int:
    """Run the clock from the end of `play` to the next snap. Returns the seconds elapsed."""
    if play.get("no_clock") or state.phase == Phase.GAME_OVER:
        return 0
    half_secs = state.half_seconds_remaining
    if half_secs <= 0:
        state.period_expired = True
        return 0
    offense = play.get("offense", state.possession)
    opp = state.opponent(offense)
    diff = state.score[offense] - state.score[opp]
    under_two = half_secs <= 120
    stopper = bool(play["clock_stopper"])

    extra = {"clock_stopper": stopper, "offense_diff": diff, "under_two": under_two, "offense": offense,
             "play_was_pass": play.get("type") == "pass", "out_of_bounds": bool(play.get("oob", False)),
             "play_type_kind": play.get("type")}
    # D16b: does a timeout follow this play? Under 2:00 the trailing team calls it, otherwise the offense.
    if under_two and not state.tied:
        candidate = state.opponent(state.leader)
    else:
        candidate = offense
    timeout_by = None
    if state.timeouts[candidate] > 0 and M["timeout_after"].sample(rng, state, extra) == "yes":
        timeout_by = candidate
    if timeout_by is not None:
        elapsed = sample_exact(M, "timeout_runoff", rng, state, extra)
    elif play.get("forced_runoff") is not None:
        elapsed = int(play["forced_runoff"])
    elif play.get("type") in ("kickoff", "free_kick") and "kickoff_runoff" in M:
        elapsed = sample_exact(M, "kickoff_runoff", rng, state, extra)
    else:
        elapsed = sample_exact(M, "clock_runoff", rng, state, extra)
        # Spec 1.5.2: the offense takes a timeout rather than let the clock expire before a field goal.
        if under_two and not stopper and state.phase == Phase.SCRIMMAGE and state.possession == offense \
                and -3 <= diff <= 0 and state.timeouts[offense] > 0 and state.yardline_100 <= 40 \
                and half_secs - elapsed < 8 and half_secs > 8:
            timeout_by = offense
            lo, hi = settings()["engine"]["timeout_play_seconds"]
            elapsed = int(rng.integers(lo, hi + 1))
    if timeout_by is not None:
        state.timeouts[timeout_by] -= 1
        play["timeout_by"] = timeout_by

    elapsed = max(0, min(elapsed, half_secs))
    box.add(offense, "time_of_possession", elapsed)
    new = half_secs - elapsed

    # Two-minute warning: once per half (and once in OT), the clock stops at exactly 2:00.
    h = state.half
    if half_secs > 120 and new <= 120 and not state.two_minute_warning_used[h]:
        new = 120
        state.two_minute_warning_used[h] = True
    # End of Q1 / Q3: clock set to the boundary, drive continues.
    if not state.in_ot and state.qtr in (1, 3) and half_secs > 900 and new <= 900:
        new = 900
        state.qtr += 1
    if new <= 0:
        new = 0
        state.period_expired = True
    state.set_clock(new)
    play["elapsed"] = elapsed
    return elapsed
