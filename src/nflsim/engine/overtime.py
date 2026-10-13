"""Step 1.6: overtime, halftime and end-of-period transitions."""
from __future__ import annotations

import numpy as np

from nflsim.config import settings
from nflsim.engine.state import GameState, OTState, Phase


def ot_config(state: GameState) -> dict:
    return settings()["playoff_ot"] if state.playoff else settings()["regular_season_ot"]


def halftime(state: GameState) -> None:
    state.qtr = 3
    state.game_seconds_remaining = 1800
    state.timeouts = {state.home: 3, state.away: 3}
    state.period_expired = False
    state.kicking_team = state.opponent(state.second_half_receiver)
    state.possession = state.second_half_receiver
    state.phase = Phase.KICKOFF


def start_overtime(state: GameState, rng: np.random.Generator, box) -> None:
    cfg = ot_config(state)
    box.reg_score = dict(state.score)
    receiver = state.home if rng.random() < 0.5 else state.away   # rule 1: coin toss
    state.qtr = 5
    state.ot_state = OTState(active=True, first_team=receiver, possession_count=0, period=1,
                             seconds_remaining=int(cfg["period_seconds"]))
    state.timeouts = {state.home: int(cfg["timeouts"]), state.away: int(cfg["timeouts"])}
    state.two_minute_warning_used[3] = False
    state.period_expired = False
    state.kicking_team = state.opponent(receiver)
    state.possession = receiver
    state.phase = Phase.KICKOFF


def new_ot_period(state: GameState) -> None:
    """Playoffs only: another full period, possession and OT possession rules carry over."""
    cfg = ot_config(state)
    state.qtr += 1
    state.ot_state.period += 1
    state.ot_state.seconds_remaining = int(cfg["period_seconds"])
    state.timeouts = {state.home: int(cfg["timeouts"]), state.away: int(cfg["timeouts"])}
    state.two_minute_warning_used[3] = False
    state.period_expired = False


def on_period_expired(state: GameState, rng: np.random.Generator, box) -> None:
    """Apply the transition after the clock hit 0:00 (called once any pending PAT has been resolved)."""
    if not state.in_ot:
        if state.qtr <= 2:
            halftime(state)
        elif state.tied:
            start_overtime(state, rng, box)
        else:
            state.phase = Phase.GAME_OVER
        return
    cfg = ot_config(state)
    if state.tied and not cfg["ties_allowed"]:
        new_ot_period(state)          # rule 5 (playoffs): keep playing
    else:
        state.phase = Phase.GAME_OVER  # rule 5 (regular season): tie, or a lead at the end of the period


def check_ot_end(state: GameState, scoring_team: str | None = None) -> None:
    """Rules 2-4. Call after a score fully resolves (incl. the PAT) or after a possession change."""
    ot = state.ot_state
    if not ot.active or state.phase == Phase.GAME_OVER:
        return
    n = ot.possession_count
    if scoring_team is not None:
        if n <= 1:
            if scoring_team != ot.first_team:      # rule 2: defensive score on the first possession ends it
                state.phase = Phase.GAME_OVER
        elif n == 2:
            if not state.tied:                     # rule 3/4: both have possessed (or will have) and someone leads
                state.phase = Phase.GAME_OVER
        else:
            state.phase = Phase.GAME_OVER          # rule 3: sudden death, any score wins
    else:
        if n >= 3 and not state.tied:              # second possession ended with a lead on the board
            state.phase = Phase.GAME_OVER


def ot_decided_before_try(state: GameState, td_team: str) -> bool:
    """A TD that ends the OT game is not followed by a try."""
    ot = state.ot_state
    if not ot.active:
        return False
    n = ot.possession_count
    lead = state.score[td_team] - state.score[state.opponent(td_team)]
    if n <= 1:
        return td_team != ot.first_team
    if n == 2:
        return lead > 0
    return True
