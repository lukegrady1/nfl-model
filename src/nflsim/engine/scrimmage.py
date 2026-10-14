"""Step 1.4: one play from scrimmage, plus `step()` which dispatches on the phase."""
from __future__ import annotations

import numpy as np

from nflsim.config import settings
from nflsim.engine import clock, kicking, overtime
from nflsim.engine.state import GameState, Phase
from nflsim.models import buckets as B
from nflsim.models.features import PLAY_TYPES
from nflsim.models.registry import sample_exact

PT_COMPLETION, PT_SACK, PT_SCRAMBLE, PT_RUN = 0, 1, 2, 3


def step(state: GameState, rng: np.random.Generator, M: dict, box) -> dict | None:
    """Advance the game by one play (or one transition). Returns the play record."""
    if state.period_expired and state.phase != Phase.PAT:
        overtime.on_period_expired(state, rng, box)
        if state.phase == Phase.GAME_OVER:
            return None
    pre = box.snapshot(state)
    if state.phase == Phase.KICKOFF:
        play = kicking.kickoff(state, rng, M, box)
    elif state.phase == Phase.FREE_KICK:
        play = kicking.kickoff(state, rng, M, box, free_kick=True)
    elif state.phase == Phase.PAT:
        play = kicking.pat(state, rng, M, box)
    elif state.phase == Phase.SCRIMMAGE:
        play = scrimmage_play(state, rng, M, box)
    else:
        return None
    clock.advance(state, play, rng, M, box)
    state.plays += 1
    box.record_play(pre, play)
    return play


# ---- scrimmage ---------------------------------------------------------------------------------------------
def should_kneel(state: GameState) -> bool:
    """Victory formation: leading in Q4 and the remaining time can be run out; or pinned deep at the end of H1."""
    secs = state.half_seconds_remaining
    if state.qtr == 4 and state.score_diff > 0:
        opp_to = state.timeouts[state.defense]
        downs_left = 4 - state.down + 1
        kneel_secs = settings()["engine"]["kneel_seconds"]
        killable = kneel_secs * max(0, downs_left - opp_to) + 2 * min(downs_left, opp_to)
        if state.down == 4:
            return opp_to == 0 and secs <= kneel_secs
        return secs <= killable
    if state.qtr == 2 and secs <= 20 and state.yardline_100 >= 50:
        return True
    return False


def kneel(state: GameState, box) -> dict:
    off = state.possession
    box.add(off, "plays")
    box.add(off, "kneels")
    play = {"type": "kneel", "clock_stopper": False, "offense": off,
            "forced_runoff": settings()["engine"]["kneel_seconds"]}
    yards = -1 if state.yardline_100 < 99 else 0
    apply_result(state, yards, box, play)
    return play


def presnap_penalty(state: GameState, side: str, rng: np.random.Generator, M: dict, box) -> dict:
    off = state.possession
    box.add(off, "penalties")
    yds = int(sample_exact(M, "penalty_yards", rng, state, {"side_idx": 1 if side == "defense" else 0})) if "penalty_yards" in M else 5
    play = {"type": "penalty", "clock_stopper": True, "offense": off, "side": side}
    if side == "offense":
        yds = min(yds, (100 - state.yardline_100) // 2)   # half the distance to the goal line
        state.yardline_100 += yds
        state.ydstogo += yds
    else:
        yds = min(yds, state.yardline_100 // 2)
        state.yardline_100 -= yds
        auto_fd = M["def_penalty_first_down"].sample(rng, state, {}) == "yes"
        if auto_fd or yds >= state.ydstogo:
            state.down = 1
            state.ydstogo = min(10, state.yardline_100)
            box.add(off, "first_downs")
        else:
            state.ydstogo -= yds
    play["yards"] = yds
    return play


def scrimmage_play(state: GameState, rng: np.random.Generator, M: dict, box) -> dict:
    off = state.possession
    # 1. pre-snap / nullified-play penalty
    side = M["presnap_penalty"].sample(rng, state, {})
    if side != "none":
        return presnap_penalty(state, side, rng, M, box)
    # 2. fourth down
    if state.down == 4:
        choice = M["fourth_down"].sample(rng, state, {})
        if choice == "punt":
            return kicking.punt(state, rng, M, box)
        if choice == "field_goal":
            return kicking.field_goal(state, rng, M, box)
    # 3. kneel rule
    if should_kneel(state):
        return kneel(state, box)
    # 4. play call
    box.add(off, "plays")
    call = M["play_call"].sample(rng, state, {})
    if call == "pass":
        return pass_play(state, rng, M, box)
    return run_play(state, rng, M, box)


def team_players(state: GameState, team: str):
    return state.context.players.get(team) if state.context.players else None


def pass_play(state: GameState, rng: np.random.Generator, M: dict, box) -> dict:
    off, deff = state.possession, state.defense
    play = {"type": "pass", "clock_stopper": False, "offense": off}
    tp = team_players(state, off)
    qb = tp.qb if tp else None
    qb_eff = tp.qb_eff() if tp else None
    pre = M["dropback_outcome"].sample(rng, state, {"player": qb_eff})
    if pre == "sack":
        yards = int(sample_exact(M, "sack_yards", rng, state))
        box.add(off, "sacks")
        box.add(off, "sack_yards", -yards)
        play["outcome"], play["yards"] = "sack", yards
        if fumble_check(state, PT_SACK, yards, rng, M, box, play, carrier=qb):
            return play
        apply_result(state, yards, box, play)
        return play
    if pre == "scramble":
        yards = min(int(sample_exact(M, "scramble_yards", rng, state)), state.yardline_100)
        box.add(off, "scrambles")
        box.add(off, "rush_attempts")
        box.add(off, "rush_yards", yards)
        box.padd(off, qb, "rush_attempts")
        box.padd(off, qb, "rush_yards", yards)
        box.padd(off, qb, "longest_rush", yards)
        play["outcome"], play["yards"] = "scramble", yards
        if fumble_check(state, PT_SCRAMBLE, yards, rng, M, box, play, carrier=qb):
            return play
        if out_of_bounds(state, PT_SCRAMBLE, rng, M):
            play["clock_stopper"] = True
            play["oob"] = True
        apply_result(state, yards, box, play, scorer=qb, kind="rush")
        return play
    # a throw: depth first, then the target (step 4.3), then the outcome with the receiver's and QB's effects
    air = min(int(sample_exact(M, "air_yards", rng, state, {"player": qb_eff})), state.yardline_100)
    air_bin = B.air_bin(air)
    receiver = tp.pick_target(rng, air, state.yardline_100) if tp else None
    rec_eff = tp.eff(receiver) if tp else {}
    pfe = {**qb_eff, "rec_catch_rate_oe": rec_eff.get("catch_rate_oe", 0.0), "rec_yac_oe": rec_eff.get("yac_oe", 0.0)} if tp else None
    box.add(off, "pass_attempts")
    box.padd(off, qb, "pass_attempts")
    box.padd(off, receiver, "targets")
    extra = {"air_bin": air_bin, "air_yards": air, "player": pfe}
    outcome = M["throw_outcome"].sample(rng, state, extra)
    play["outcome"], play["air"], play["receiver"] = outcome, air, receiver
    if outcome == "complete":
        yac = int(sample_exact(M, "yac", rng, state, extra))
        yards = min(air + yac, state.yardline_100)
        box.add(off, "completions")
        box.add(off, "pass_yards", yards)
        box.padd(off, qb, "completions")
        box.padd(off, qb, "pass_yards", yards)
        box.padd(off, receiver, "receptions")
        box.padd(off, receiver, "rec_yards", yards)
        box.padd(off, receiver, "longest_rec", yards)
        play["yards"] = yards
        if fumble_check(state, PT_COMPLETION, yards, rng, M, box, play, carrier=receiver):
            return play
        if out_of_bounds(state, PT_COMPLETION, rng, M):
            play["clock_stopper"] = True
            play["oob"] = True
        apply_result(state, yards, box, play, scorer=receiver, kind="pass", passer=qb)
        return play
    if outcome == "incomplete":
        play["yards"] = 0
        play["clock_stopper"] = True
        apply_result(state, 0, box, play)
        return play
    # interception: possession flips at the catch spot plus the return
    box.add(off, "interceptions")
    box.add(off, "turnovers")
    box.padd(off, qb, "interceptions")
    play["clock_stopper"] = True
    play["yards"] = 0
    spot = state.yardline_100 - air     # offense frame; 0 = caught in the end zone
    if M["pick_six"].sample(rng, state, {}) == "yes":
        kicking.touchdown(state, deff, box)
        play["result"] = "pick_six"
        return play
    ret = int(sample_exact(M, "int_return_yards", rng, state))
    new_yl = 80 if spot <= 0 else min(max(100 - spot - ret, 1), 99)
    kicking.start_possession(state, deff, new_yl, box)
    play["result"] = f"int_at_{new_yl}"
    return play


def run_play(state: GameState, rng: np.random.Generator, M: dict, box) -> dict:
    off = state.possession
    play = {"type": "run", "clock_stopper": False, "offense": off}
    tp = team_players(state, off)
    carrier = tp.pick_carrier(rng, state.yardline_100) if tp else None
    yards = min(int(sample_exact(M, "rush_yards", rng, state, {"player": {"rush_yards_oe": tp.eff(carrier).get("rush_yards_oe", 0.0)} if tp else None})), state.yardline_100)
    box.add(off, "rush_attempts")
    box.add(off, "rush_yards", yards)
    box.padd(off, carrier, "rush_attempts")
    box.padd(off, carrier, "rush_yards", yards)
    box.padd(off, carrier, "longest_rush", yards)
    play["yards"], play["carrier"] = yards, carrier
    if fumble_check(state, PT_RUN, yards, rng, M, box, play, carrier=carrier):
        return play
    if out_of_bounds(state, PT_RUN, rng, M):
        play["clock_stopper"] = True
        play["oob"] = True
    apply_result(state, yards, box, play, scorer=carrier, kind="rush")
    return play


def out_of_bounds(state: GameState, pt: int, rng: np.random.Generator, M: dict) -> bool:
    """Out of bounds stops the clock only inside the last 2:00 of H1 / last 5:00 of H2 (and OT)."""
    secs = state.half_seconds_remaining
    window = secs <= 120 if state.half == 1 else secs <= 300
    if not window or "out_of_bounds" not in M:
        return False
    return M["out_of_bounds"].sample(rng, state, {"play_type_idx": pt, "play_type": PLAY_TYPES[pt]}) == "yes"


def fumble_check(state: GameState, pt: int, yards: int, rng: np.random.Generator, M: dict, box, play: dict,
                 carrier: str | None = None) -> bool:
    """D9. A lost fumble flips possession at the spot (v1: no return). Returns True if possession changed."""
    off, deff = state.possession, state.defense
    res = M["fumble_lost"].sample(rng, state, {"play_type_idx": pt, "play_type": PLAY_TYPES[pt], "yards": yards})
    if res == "no":
        return False
    if res == "yes" and "yes_td" not in M["fumble_lost"].classes and "fumble_return_td" in M:
        if M["fumble_return_td"].sample(rng, state, {}) == "yes":
            res = "yes_td"
    box.add(off, "fumbles_lost")
    box.add(off, "turnovers")
    box.padd(off, carrier, "fumbles_lost")
    play["clock_stopper"] = True
    play["result"] = "fumble_lost"
    if res == "yes_td":
        kicking.touchdown(state, deff, box)
        play["result"] = "fumble_return_td"
        return True
    new_spot = state.yardline_100 - yards
    if new_spot <= 0:
        kicking.start_possession(state, deff, 80, box)       # fumbled into / through the end zone: touchback
    elif new_spot >= 100:
        kicking.start_possession(state, deff, 1, box)        # recovered at the goal line (defensive TD not modeled)
    else:
        kicking.start_possession(state, deff, 100 - new_spot, box)
    return True


def apply_result(state: GameState, yards: int, box, play: dict, scorer: str | None = None, kind: str | None = None,
                 passer: str | None = None) -> None:
    """Step 1.4.8: touchdown, safety, first down, next down, turnover on downs."""
    off, deff = state.possession, state.defense
    new_spot = state.yardline_100 - yards
    if new_spot <= 0:
        kicking.touchdown(state, off, box)
        play["clock_stopper"] = True
        play["result"] = "touchdown"
        if scorer is not None:
            box.padd(off, scorer, "rec_tds" if kind == "pass" else "rush_tds")
            box.padd(off, scorer, "anytime_td")
            if kind == "pass":
                box.padd(off, passer, "pass_tds")
        return
    if new_spot >= 100:
        kicking.safety(state, off, box)
        play["clock_stopper"] = True
        play["result"] = "safety"
        return
    if yards >= state.ydstogo:
        state.down = 1
        state.yardline_100 = new_spot
        state.ydstogo = min(10, new_spot)
        box.add(off, "first_downs")
        play["result"] = "first_down"
        return
    if state.down >= 4:
        kicking.start_possession(state, deff, 100 - new_spot, box)
        play["clock_stopper"] = True
        play["result"] = "turnover_on_downs"
        return
    state.down += 1
    state.yardline_100 = new_spot
    state.ydstogo = min(state.ydstogo - yards, new_spot)
