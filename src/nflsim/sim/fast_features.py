"""Numba versions of the bucket functions, the feature builder and the decision sampling used by the fast engine.
Mirrors models/buckets.py and models/features.py exactly; tests/test_fast_engine.py checks the equality."""
from __future__ import annotations

import math

import numpy as np
from numba import njit
from numba.typed import List

from nflsim.sim.pack import MAX_CLASSES, SLOT, PackedModels

MAX_FEATURES = 160
S_PLAY_CALL, S_FOURTH, S_AIR, S_DROPBACK, S_THROW, S_YAC = SLOT["play_call"], SLOT["fourth_down"], SLOT["air_yards"], SLOT["dropback_outcome"], SLOT["throw_outcome"], SLOT["yac"]
S_RUSH, S_FUMBLE, S_FG, S_PAT, S_CLOCK, S_PENALTY = SLOT["rush_yards"], SLOT["fumble_lost"], SLOT["fg_make"], SLOT["pat_choice"], SLOT["clock_runoff"], SLOT["presnap_penalty"]
ERA_REF, ERA_MAX, DEFAULT_WIND = 2025.0, 2025.0, 8.0


# ---- RNG: PCG32-style 64-bit LCG with xorshift output; one state per simulation --------------------------------
@njit(cache=True, inline="always")
def rand_u(st):
    """Uniform in [0, 1) from a 1-element uint64 state array (splitmix64 step)."""
    st[0] += np.uint64(0x9E3779B97F4A7C15)
    z = st[0]
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)) * (1.0 / 9007199254740992.0)


# ---- buckets (models/buckets.py) ---------------------------------------------------------------------------------
@njit(cache=True)
def dist_bucket(ydstogo):
    if ydstogo <= 1:
        return 0
    if ydstogo <= 3:
        return 1
    if ydstogo <= 6:
        return 2
    if ydstogo <= 10:
        return 3
    return 4


@njit(cache=True)
def zone(yl):
    if yl >= 51:
        return 0
    if yl >= 21:
        return 1
    if yl >= 11:
        return 2
    return 3


@njit(cache=True)
def yardline5(yl):
    v = (yl - 1) // 5
    return min(max(v, 0), 19)


@njit(cache=True)
def score_bucket(diff):
    if diff <= -9:
        return 0
    if diff <= -4:
        return 1
    if diff <= -1:
        return 2
    if diff == 0:
        return 3
    if diff <= 3:
        return 4
    if diff <= 8:
        return 5
    return 6


@njit(cache=True)
def lead_bucket(diff):
    return 0 if diff < 0 else (1 if diff == 0 else 2)


@njit(cache=True)
def time_bucket(qtr, half_secs, gsr):
    if qtr >= 5:
        return 5
    if qtr <= 2:
        return 1 if half_secs <= 120 else 0
    if gsr > 480:
        return 2
    if gsr > 120:
        return 3
    return 4


@njit(cache=True)
def score_time_bucket(qtr, half_secs, diff):
    if qtr >= 5:
        return 7
    if qtr <= 2:
        return 1 if half_secs <= 120 else 0
    if qtr == 3:
        return 2
    if diff > 0:
        return 3
    if diff == 0:
        return 4
    if diff >= -8:
        return 5
    return 6


@njit(cache=True)
def under_two_half(half_secs, qtr):
    if half_secs > 120:
        return 0
    return 1 if qtr <= 2 else 2


@njit(cache=True)
def air_bin(air):
    if air <= -1:
        return 0
    if air <= 4:
        return 1
    if air <= 9:
        return 2
    if air <= 14:
        return 3
    if air <= 19:
        return 4
    if air <= 29:
        return 5
    return 6


@njit(cache=True)
def seconds_bin(s):
    if s <= 5:
        return 0
    if s <= 10:
        return 1
    if s <= 15:
        return 2
    if s <= 20:
        return 3
    if s <= 25:
        return 4
    if s <= 30:
        return 5
    if s <= 35:
        return 6
    if s <= 40:
        return 7
    if s <= 45:
        return 8
    return 9


@njit(cache=True)
def pat_score_key(d):
    return min(max(d, -16), 16)


# ---- feature builder (models/features.py) -----------------------------------------------------------------------
@njit(cache=True)
def _onehot_edges(out, k, value, e0, e1, e2, e3, e4, e5):
    """7 one-hots: value <= e0, <= e1, ..., <= e5, else last."""
    for j in range(7):
        out[k + j] = 0.0
    if value <= e0:
        out[k] = 1.0
    elif value <= e1:
        out[k + 1] = 1.0
    elif value <= e2:
        out[k + 2] = 1.0
    elif value <= e3:
        out[k + 3] = 1.0
    elif value <= e4:
        out[k + 4] = 1.0
    elif value <= e5:
        out[k + 5] = 1.0
    else:
        out[k + 6] = 1.0
    return k + 7


@njit(cache=True)
def _fine_yardline(out, k, v):
    for j in range(23):
        out[k + j] = 0.0
    if v <= 20:
        out[k + max(v, 1) - 1] = 1.0
    elif v <= 30:
        out[k + 20] = 1.0
    elif v <= 50:
        out[k + 21] = 1.0
    else:
        out[k + 22] = 1.0
    return k + 23


@njit(cache=True)
def _era(season):
    s = season if season > 0 else ERA_MAX
    return (min(s, ERA_MAX) - ERA_REF) / 5.0


@njit(cache=True)
def _play_call_extra(out, k, ydstogo, down, qtr, gsr, half_secs, score_diff, season):
    ly = math.log(max(ydstogo, 1))
    short = 1.0 if ydstogo <= 2 else 0.0
    d2 = 1.0 if down == 2 else 0.0
    d3 = 1.0 if down == 3 else 0.0
    late = (qtr == 4 and gsr <= 480) or qtr >= 5
    sd = max(min(float(score_diff), 28.0), -28.0) / 28.0
    u2 = 1.0 if half_secs <= 120 else 0.0
    trail = 1.0 if (late and score_diff < 0) else 0.0
    lead = 1.0 if (late and score_diff > 0) else 0.0
    out[k] = d2 * ly; out[k + 1] = d3 * ly; out[k + 2] = short; out[k + 3] = d2 * short; out[k + 4] = d3 * short
    out[k + 5] = sd * u2; out[k + 6] = trail * u2; out[k + 7] = lead * u2; out[k + 8] = sd * sd; out[k + 9] = _era(season)
    return k + 10


@njit(cache=True)
def _air_pair(out, k, air):
    a = max(min(float(air), 50.0), -10.0)
    out[k] = a; out[k + 1] = a * a
    return k + 2


@njit(cache=True)
def build_features(slot, out, down, ydstogo, yl, qtr, gsr, half_secs, score_diff, off_to, def_to, is_home, season,
                   air, kick_distance, roof_dome, wind, pt, yards, cs, play_was_pass, oob, play_kind, diff_after_td,
                   off_r, def_r, pf, team_flag, team_off_idx, team_def_idx, pf_idx, weather_flag, pf_mean, has_players):
    """Fills `out` with the model's features; returns the count. Inputs follow situation_from_state.
    When has_players is false the player features take their training means (TrainedModel.proba does the same)."""
    if slot == S_PLAY_CALL:
        down = min(down, 3)
    k = 0
    if slot != S_FUMBLE and slot != S_FG and slot != S_PAT:
        yd = max(ydstogo, 1)
        ylf = float(yl)
        sd = max(min(float(score_diff), 28.0), -28.0) / 28.0
        late = (qtr == 4 and gsr <= 480) or qtr >= 5
        out[0] = 1.0 if down == 1 else 0.0
        out[1] = 1.0 if down == 2 else 0.0
        out[2] = 1.0 if down == 3 else 0.0
        out[3] = 1.0 if down == 4 else 0.0
        out[4] = float(min(yd, 20))
        out[5] = math.log(yd)
        out[6] = ylf / 100.0
        out[7] = (ylf / 100.0) ** 2
        out[8] = 1.0 if yd >= yl else 0.0
        out[9] = 1.0 if ylf <= 20 else 0.0
        out[10] = 1.0 if ylf <= 5 else 0.0
        out[11] = sd
        out[12] = float(half_secs) / 1800.0
        out[13] = 1.0 if half_secs <= 120 else 0.0
        out[14] = 1.0 if (late and score_diff < 0) else 0.0
        out[15] = 1.0 if (late and score_diff > 0) else 0.0
        out[16] = sd * (1.0 - float(gsr) / 3600.0)
        out[17] = float(off_to)
        out[18] = float(def_to)
        out[19] = float(is_home)
        k = 20
    if slot == S_PLAY_CALL or slot == S_PENALTY:
        k = _play_call_extra(out, k, ydstogo, down, qtr, gsr, half_secs, score_diff, season)
    elif slot == S_FOURTH:
        d = float(yl + 17)
        out[k] = d; out[k + 1] = d * d; out[k + 2] = _era(season)
        k += 3
    elif slot == S_AIR or slot == S_RUSH:
        k = _play_call_extra(out, k, ydstogo, down, qtr, gsr, half_secs, score_diff, season)
        k = _fine_yardline(out, k, yl)
    elif slot == S_THROW:
        k = _air_pair(out, k, air)
        k = _onehot_edges(out, k, air, -1, 4, 9, 14, 19, 29)
        k = _play_call_extra(out, k, ydstogo, down, qtr, gsr, half_secs, score_diff, season)
    elif slot == S_YAC:
        room = max(yl - air, 0)
        k = _air_pair(out, k, air)
        k = _onehot_edges(out, k, air, -1, 4, 9, 14, 19, 29)
        k = _onehot_edges(out, k, room, 5, 10, 15, 20, 30, 50)
        out[k] = math.log(float(room) + 1.0); k += 1
        k = _play_call_extra(out, k, ydstogo, down, qtr, gsr, half_secs, score_diff, season)
        k = _fine_yardline(out, k, room)
    elif slot == S_CLOCK:
        csf = 1.0 if cs else 0.0
        pwp = 1.0 if play_was_pass else 0.0
        u2 = 1.0 if half_secs <= 120 else 0.0
        late = (qtr == 4 and gsr <= 480) or qtr >= 5
        trail = 1.0 if (late and score_diff < 0) else 0.0
        lead = 1.0 if (late and score_diff > 0) else 0.0
        second_half = 1.0 if qtr >= 3 else 0.0
        out[k] = csf; out[k + 1] = pwp; out[k + 2] = 1.0 if oob else 0.0
        out[k + 3] = csf * u2; out[k + 4] = csf * trail; out[k + 5] = csf * lead; out[k + 6] = trail * u2; out[k + 7] = lead * u2
        out[k + 8] = second_half; out[k + 9] = second_half * u2; out[k + 10] = pwp * csf; out[k + 11] = _era(season)
        out[k + 12] = 1.0 if play_kind == 1 else 0.0   # punt
        out[k + 13] = 1.0 if play_kind == 2 else 0.0   # field goal
        out[k + 14] = 1.0 if play_kind == 3 else 0.0   # penalty
        k += 15
    elif slot == S_FUMBLE:
        for j in range(4):
            out[j] = 1.0 if pt == j else 0.0
        out[4] = max(min(float(yards), 40.0), -20.0)
        k = 5
    elif slot == S_FG:
        d = float(kick_distance)
        w = wind if wind >= 0 else (0.0 if roof_dome else DEFAULT_WIND)
        out[0] = d; out[1] = d * d; out[2] = float(roof_dome); out[3] = max(min(w, 30.0), 0.0); out[4] = d ** 3 / 1e5; out[5] = _era(season)
        k = 6
    elif slot == S_PAT:
        for j in range(31):
            out[j] = 1.0 if diff_after_td == j - 15 else 0.0
        out[31] = max(min(diff_after_td, 16), -16) / 16.0
        out[32] = float(half_secs) / 1800.0
        out[33] = 1.0 if qtr >= 3 else 0.0
        out[34] = _era(season)
        k = 35
    if team_flag:
        for j in range(8):
            if team_off_idx[j] >= 0:
                out[k] = off_r[team_off_idx[j]]; k += 1
        for j in range(8):
            if team_def_idx[j] >= 0:
                out[k] = def_r[team_def_idx[j]]; k += 1
        if weather_flag:
            w = wind if wind >= 0 else (0.0 if roof_dome else DEFAULT_WIND)
            out[k] = float(roof_dome); out[k + 1] = max(min(w, 30.0), 0.0); k += 2
        for j in range(8):
            if pf_idx[j] >= 0:
                out[k] = pf[pf_idx[j]] if has_players else pf_mean[j]
                k += 1
    return k


# ---- decision evaluation -------------------------------------------------------------------------------------------
@njit(cache=True)
def trained_probs(x, n, mean, std, W, b, binary, probs, K, mask_kind, lower_edge, room, home_dir, home_shift):
    """probs[:K] from standardized logits (+ home shift along home_dir), softmax / sigmoid, then the yardage mask."""
    if binary:
        z = b[0]
        for j in range(n):
            z += W[0, j] * (x[j] - mean[j]) / std[j]
        p1 = 1.0 / (1.0 + math.exp(-z))
        probs[0] = 1.0 - p1
        probs[1] = p1
    else:
        mx = -1e300
        for c in range(K):
            z = b[c]
            for j in range(n):
                z += W[c, j] * (x[j] - mean[j]) / std[j]
            if home_shift != 0.0:
                z += home_dir[c] * home_shift
            probs[c] = z
            if z > mx:
                mx = z
        tot = 0.0
        for c in range(K):
            probs[c] = math.exp(probs[c] - mx)
            tot += probs[c]
        for c in range(K):
            probs[c] /= tot
    if mask_kind > 0:
        tot = 0.0
        for c in range(K):
            if lower_edge[c] > room:
                probs[c] = 0.0
            tot += probs[c]
        if tot > 0.0:
            for c in range(K):
                probs[c] /= tot


@njit(cache=True)
def sample_probs(probs, K, u):
    """Index of the class drawn from probs[:K] (searchsorted on the cumulative sum, like TrainedModel.sample)."""
    acc = 0.0
    for c in range(K):
        acc += probs[c]
    target = u * acc
    acc = 0.0
    for c in range(K):
        acc += probs[c]
        if target < acc:
            return c
    return K - 1


@njit(cache=True)
def sample_table(cum, flat, K, u):
    base = flat * K
    for c in range(K):
        if u < cum[base + c]:
            return c
    return K - 1


@njit(cache=True)
def sample_within(w_off, w_vals, w_cum, c, u):
    lo = w_off[c]
    hi = w_off[c + 1]
    for i in range(lo, hi):
        if u < w_cum[i]:
            return w_vals[i]
    return w_vals[hi - 1]


def typed_lists(P: PackedModels):
    """Numba typed lists for the per-slot arrays of varying shape."""
    def tl(arrs, dtype, ndim):
        L = List()
        for a in arrs:
            L.append(np.ascontiguousarray(a, dtype=dtype).reshape((-1,) if ndim == 1 else a.shape))
        return L
    return (tl(P.cum, np.float64, 1), tl(P.dims, np.int64, 1), tl(P.w_off, np.int64, 1), tl(P.w_vals, np.int64, 1), tl(P.w_cum, np.float64, 1),
            tl(P.mean, np.float64, 1), tl(P.std, np.float64, 1), tl([np.atleast_2d(w) for w in P.W], np.float64, 2), tl(P.b, np.float64, 1))
