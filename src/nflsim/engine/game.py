"""Simulate one full game with the reference (pure Python) engine."""
from __future__ import annotations

import numpy as np

from nflsim.engine.boxscore import BoxScore, GameResult
from nflsim.engine.scrimmage import step
from nflsim.engine.state import Context, GameState, Phase, new_game

MAX_PLAYS = 600


def coin_toss(state: GameState, rng: np.random.Generator) -> None:
    receiver = state.home if rng.random() < 0.5 else state.away
    state.kicking_team = state.opponent(receiver)
    state.second_half_receiver = state.kicking_team
    state.possession = receiver
    state.phase = Phase.KICKOFF


def simulate_game(home: str, away: str, M: dict | None = None, rng: np.random.Generator | None = None,
                  seed: int | None = None, neutral: bool = False, playoff: bool = False,
                  context: Context | None = None, debug: bool = False) -> GameResult:
    if M is None:
        from nflsim.models.registry import default_models
        M = default_models()
    if home == away:
        raise ValueError("home and away must be distinct labels (use e.g. AVG_H / AVG_A)")
    rng = rng if rng is not None else np.random.default_rng(seed)
    state = new_game(home, away, neutral=neutral, playoff=playoff, context=context)
    box = BoxScore(home, away, debug=debug)
    coin_toss(state, rng)
    while state.phase != Phase.GAME_OVER and state.plays < MAX_PLAYS:
        step(state, rng, M, box)
    if state.phase != Phase.GAME_OVER:
        raise RuntimeError(f"game did not finish within {MAX_PLAYS} plays")
    return box.result(state)
