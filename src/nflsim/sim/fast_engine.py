"""Phase 5, step 5.1: the Numba port of the reference engine (engine/*.py). Plain arrays only.

`simulate_games(n, seed, packed_models, packed_game)` runs n simulations in a prange loop with one RNG stream per
simulation (seeded from `seed` and the simulation index), returning home/away scores, OT flags, team stats and the
player stat array [n, 2 * n_max, 16] (side 0 = home players, side 1 = away players, stats in PLAYER_STATS order).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from nflsim.sim import fast_features as F
from nflsim.sim.fast_features import (MAX_FEATURES, air_bin, build_features, dist_bucket, lead_bucket, pat_score_key,
                                      rand_u, sample_probs, sample_table, sample_within, score_bucket, score_time_bucket,
                                      time_bucket, trained_probs, typed_lists, under_two_half, yardline5, zone)
from nflsim.sim.pack import MAX_CLASSES, SLOT

# slots
S = SLOT
PLAY_CALL, FOURTH, AIR, DROPBACK, THROW, YAC, SACK_Y, SCR_Y, RUSH = S["play_call"], S["fourth_down"], S["air_yards"], S["dropback_outcome"], S["throw_outcome"], S["yac"], S["sack_yards"], S["scramble_yards"], S["rush_yards"]
FUMBLE, FUM_TD, OOB, PICK6, INT_RET, FG, PUNT_RES, PUNT_NET, PUNT_RET = S["fumble_lost"], S["fumble_return_td"], S["out_of_bounds"], S["pick_six"], S["int_return_yards"], S["fg_make"], S["punt_result"], S["punt_net_yards"], S["punt_retain_spot"]
KO_RES, KO_SPOT, FK_SPOT, PAT_CH, XP, TWO, CLOCK, KO_RUN, TO_AFTER, TO_RUN, PRESNAP, PEN_Y, DEF_FD = S["kickoff_result"], S["kickoff_spot"], S["free_kick_spot"], S["pat_choice"], S["xp_result"], S["two_point_result"], S["clock_runoff"], S["kickoff_runoff"], S["timeout_after"], S["timeout_runoff"], S["presnap_penalty"], S["penalty_yards"], S["def_penalty_first_down"]
# phases
KICKOFF, SCRIMMAGE, PAT, FREE_KICK, GAME_OVER = 0, 1, 2, 3, 4
# player stats (engine/boxscore.PLAYER_STATS order)
P_PASS_ATT, P_COMP, P_PASS_YDS, P_PASS_TD, P_INT, P_RUSH_ATT, P_RUSH_YDS, P_RUSH_TD, P_TGT, P_REC, P_REC_YDS, P_REC_TD, P_LONG_REC, P_LONG_RUSH, P_FUM, P_ANY_TD = range(16)
# team stats
T_PLAYS, T_PASS_ATT, T_COMP, T_PASS_YDS, T_SACKS, T_INT, T_RUSH_ATT, T_RUSH_YDS, T_PUNTS, T_FGA, T_FGM, T_TURNOVERS, T_SACK_YDS, T_SCRAMBLES, T_FUM_LOST, T_DRIVES, T_FIRST_DOWNS, T_PENALTIES, T_KNEELS, T_TOP, T_XP_ATT, T_TWO_ATT = range(22)
N_TSTATS = 22
# state indices
ST_POS, ST_DOWN, ST_YTG, ST_YL, ST_QTR, ST_GSR, ST_OTS, ST_PHASE, ST_KICK, ST_PATT, ST_EXP, ST_SHR, ST_OTA, ST_OTF, ST_OTC, ST_OTP, ST_PLAYS, ST_REGH, ST_REGA = range(19)
ONSIDE_YL = 55


@njit(cache=True)
def _half(st):
    q = st[ST_QTR]
    return 1 if q <= 2 else (2 if q <= 4 else 3)


@njit(cache=True)
def _half_secs(st):
    if st[ST_QTR] >= 5:
        return st[ST_OTS]
    return st[ST_GSR] - 1800 if st[ST_QTR] <= 2 else st[ST_GSR]


@njit(cache=True)
def _set_clock(st, hs):
    if st[ST_QTR] >= 5:
        st[ST_OTS] = hs
    elif st[ST_QTR] <= 2:
        st[ST_GSR] = hs + 1800
    else:
        st[ST_GSR] = hs


@njit(cache=True)
def _table_flat(slot, dims, down, ytg, yl, qtr, hs, gsr, diff, air, pt, cs, u2h, distance, sd, side):
    """Flat dense-table index for the slot's keys (mirrors models/registry.KEY_FNS)."""
    if slot == PLAY_CALL:
        k0, k1, k2, k3 = min(down, 3), dist_bucket(ytg), score_bucket(diff), time_bucket(qtr, hs, gsr)
        return ((k0 * dims[1] + k1) * dims[2] + k2) * dims[3] + k3
    if slot == FOURTH:
        k0, k1, k2 = dist_bucket(ytg), yardline5(yl), score_time_bucket(qtr, hs, diff)
        return (k0 * dims[1] + k1) * dims[2] + k2
    if slot == AIR:
        return (down * dims[1] + dist_bucket(ytg)) * dims[2] + zone(yl)
    if slot == DROPBACK:
        return (zone(yl) * dims[1] + down) * dims[2] + dist_bucket(ytg)
    if slot == THROW or slot == YAC:
        return air_bin(air) * dims[1] + zone(yl)
    if slot == SCR_Y:
        return zone(yl)
    if slot == RUSH:
        return dist_bucket(ytg) * dims[1] + zone(yl)
    if slot == FUMBLE:
        return pt
    if slot == OOB:
        return (pt * dims[1] + lead_bucket(diff)) * dims[2] + u2h
    if slot == FG:
        return min(max(distance, 18), 79) - 18
    if slot == PUNT_RES or slot == PUNT_NET:
        return yardline5(yl)
    if slot == PAT_CH:
        return (pat_score_key(sd) + 16) * dims[1] + time_bucket(qtr, hs, gsr)
    if slot == CLOCK or slot == TO_AFTER:
        return ((1 if cs else 0) * dims[1] + lead_bucket(diff)) * dims[2] + u2h
    if slot == TO_RUN:
        return (1 if cs else 0) * dims[1] + u2h
    if slot == PRESNAP:
        return down * dims[1] + dist_bucket(ytg)
    if slot == PEN_Y:
        return side
    return 0


@njit(cache=True)
def decide(slot, rs, st, score, timeouts, offense, home_side, neutral, season, ratings, weather, pf, kind, n_classes, class_values,
           cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset,
           team_off_idx, team_def_idx, pf_idx, weather_flag, feat, probs, pf_mean, has_players,
           air, distance, pt, yards, cs, play_was_pass, oob, play_kind, diff_after_td, side_idx, exact_room):
    """Sample a decision. Returns (class index, value): value = exact within-class draw when available, else class value."""
    deff = 1 - offense
    diff = score[offense] - score[deff]
    hs = _half_secs(st)
    gsr = st[ST_OTS] if st[ST_QTR] >= 5 else st[ST_GSR]
    u = rand_u(rs)
    K = n_classes[slot]
    if kind[slot] == 1:
        is_home = 0
        if not neutral and offense == home_side:
            is_home = 1
        n = build_features(slot, feat, st[ST_DOWN], st[ST_YTG], st[ST_YL], st[ST_QTR], gsr, hs, diff, timeouts[offense], timeouts[deff],
                           is_home, season, air, distance, weather[0], weather[1], pt, yards,
                           cs, play_was_pass, oob, play_kind, diff_after_td, ratings[offense], ratings[deff], pf, team[slot],
                           team_off_idx[slot], team_def_idx[slot], pf_idx[slot], weather_flag[slot], pf_mean[slot], has_players[offense])
        room = 0
        if mask_kind[slot] == 2:
            room = max(st[ST_YL] - air, 0)
        elif mask_kind[slot] > 0:
            room = st[ST_YL]
        shift = 0.0
        if home_offset[slot] != 0.0 and not neutral:
            shift = 0.5 * home_offset[slot] if offense == home_side else -0.5 * home_offset[slot]
        trained_probs(feat, n, mean[slot], std[slot], W[slot], b[slot], binary[slot], probs, K, mask_kind[slot], lower_edge[slot], room,
                      home_dir[slot], shift)
        c = sample_probs(probs, K, u)
    else:
        u2h = under_two_half(hs, st[ST_QTR])
        flat = _table_flat(slot, dims[slot], st[ST_DOWN], st[ST_YTG], st[ST_YL], st[ST_QTR], hs, gsr, diff, air, pt, cs, u2h, distance,
                           diff_after_td, side_idx)
        c = sample_table(cum[slot], flat, K, u)
    value = class_values[slot, c]
    wo = w_off[slot]
    if wo.shape[0] > c + 1 and wo[c + 1] > wo[c]:
        value = sample_within(wo, w_vals[slot], w_cum[slot], c, rand_u(rs))
    return c, value


@njit(cache=True)
def simulate_one(seed_state, home_side, neutral, playoff, season, ratings, weather, cfg,
                 n_players, qb, kicker, target_cum, rz_target_cum, carrier_cum, rz_carrier_cum, n_targets, n_carriers, target_idx, carrier_idx, eff,
                 kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge,
                 home_dir, home_offset, present, team_off_idx, team_def_idx, pf_idx, weather_flag, pf_mean, has_players,
                 score, timeouts, tstats, pstats):
    rs = seed_state
    feat = np.zeros(MAX_FEATURES)
    probs = np.zeros(MAX_CLASSES)
    pf = np.zeros(9)
    st = np.zeros(19, dtype=np.int64)
    tmw = np.zeros(4, dtype=np.int64)
    kneel_secs = int(cfg[0]); to_lo = int(cfg[1]); to_hi = int(cfg[2]); fk_default = int(cfg[3])
    ot_secs_cfg = int(cfg[6]) if playoff else int(cfg[4])
    ot_to_cfg = int(cfg[7]) if playoff else int(cfg[5])
    ties_allowed = (cfg[9] > 0.5) if playoff else (cfg[8] > 0.5)
    score[0] = 0; score[1] = 0
    timeouts[0] = 3; timeouts[1] = 3
    st[ST_QTR] = 1; st[ST_GSR] = 3600; st[ST_DOWN] = 1; st[ST_YTG] = 10; st[ST_YL] = 75
    # coin toss
    receiver = 0 if rand_u(rs) < 0.5 else 1
    st[ST_KICK] = 1 - receiver
    st[ST_SHR] = 1 - receiver
    st[ST_POS] = receiver
    st[ST_PHASE] = KICKOFF
    st[ST_OTA] = 0; st[ST_OTF] = -1; st[ST_OTC] = 0; st[ST_OTP] = 0; st[ST_EXP] = 0; st[ST_PLAYS] = 0
    st[ST_REGH] = -1
    went_ot = 0
    while st[ST_PHASE] != GAME_OVER and st[ST_PLAYS] < 600:
        # ---- period transitions
        if st[ST_EXP] == 1 and st[ST_PHASE] != PAT:
            if st[ST_QTR] < 5:
                if st[ST_QTR] <= 2:
                    st[ST_QTR] = 3; st[ST_GSR] = 1800; timeouts[0] = 3; timeouts[1] = 3; st[ST_EXP] = 0
                    st[ST_KICK] = 1 - st[ST_SHR]; st[ST_POS] = st[ST_SHR]; st[ST_PHASE] = KICKOFF
                elif score[0] == score[1]:
                    st[ST_REGH] = score[0]; st[ST_REGA] = score[1]
                    went_ot = 1
                    rec = 0 if rand_u(rs) < 0.5 else 1
                    st[ST_QTR] = 5; st[ST_OTA] = 1; st[ST_OTF] = rec; st[ST_OTC] = 0; st[ST_OTP] = 1; st[ST_OTS] = ot_secs_cfg
                    timeouts[0] = ot_to_cfg; timeouts[1] = ot_to_cfg; tmw[3] = 0; st[ST_EXP] = 0
                    st[ST_KICK] = 1 - rec; st[ST_POS] = rec; st[ST_PHASE] = KICKOFF
                else:
                    st[ST_PHASE] = GAME_OVER
                    break
            else:
                if score[0] == score[1] and not ties_allowed:
                    st[ST_QTR] += 1; st[ST_OTP] += 1; st[ST_OTS] = ot_secs_cfg
                    timeouts[0] = ot_to_cfg; timeouts[1] = ot_to_cfg; tmw[3] = 0; st[ST_EXP] = 0
                else:
                    st[ST_PHASE] = GAME_OVER
                    break
        phase = st[ST_PHASE]
        # play record scalars
        p_stopper = True; p_no_clock = False; p_forced = -1; p_kind = 0; p_offense = st[ST_POS]; p_oob = False; p_was_pass = False
        # ------------------------------------------------------------------ kickoffs
        if phase == KICKOFF or phase == FREE_KICK:
            kicker_side = st[ST_KICK]; rcv = 1 - kicker_side
            st[ST_POS] = rcv; p_offense = rcv
            if phase == FREE_KICK:
                if present[FK_SPOT]:
                    c, spot = decide(FK_SPOT, rs, st, score, timeouts, rcv, home_side, neutral, season, ratings, weather, pf, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, team_off_idx, team_def_idx, pf_idx, weather_flag, feat, probs, pf_mean, has_players, 0, 0, 0, 0, False, False, False, 0, 0, 0, 0)
                else:
                    spot = fk_default
                _start_possession(st, score, timeouts, tstats, rcv, spot)
            else:
                c, v = decide(KO_RES, rs, st, score, timeouts, rcv, home_side, neutral, season, ratings, weather, pf, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, team_off_idx, team_def_idx, pf_idx, weather_flag, feat, probs, pf_mean, has_players, 0, 0, 0, 0, False, False, False, 0, 0, 0, 0)
                if v == 1:      # return_td
                    _touchdown(st, score, rcv)
                elif v == 2:    # kicking_team_ball
                    _start_possession(st, score, timeouts, tstats, kicker_side, ONSIDE_YL)
                else:
                    c, spot = decide(KO_SPOT, rs, st, score, timeouts, rcv, home_side, neutral, season, ratings, weather, pf, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, team_off_idx, team_def_idx, pf_idx, weather_flag, feat, probs, pf_mean, has_players, 0, 0, 0, 0, False, False, False, 0, 0, 0, 0)
                    _start_possession(st, score, timeouts, tstats, rcv, spot)
            p_kind = 4   # kickoff (uses kickoff_runoff)
        # ------------------------------------------------------------------ PAT
        elif phase == PAT:
            tm = st[ST_PATT]; opp = 1 - tm
            st[ST_POS] = tm; p_offense = tm
            diff_after = score[tm] - score[opp]
            decided = _ot_decided_before_try(st, score, tm) or (st[ST_EXP] == 1 and st[ST_QTR] == 4 and diff_after > 0)
            if not decided:
                c, v = decide(PAT_CH, rs, st, score, timeouts, tm, home_side, neutral, season, ratings, weather, pf, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, team_off_idx, team_def_idx, pf_idx, weather_flag, feat, probs, pf_mean, has_players, 0, 0, 0, 0, False, False, False, 0, diff_after, 0, 0)
                if v == 1:   # two
                    tstats[tm, T_TWO_ATT] += 1
                    c2, g = decide(TWO, rs, st, score, timeouts, tm, home_side, neutral, season, ratings, weather, pf, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, team_off_idx, team_def_idx, pf_idx, weather_flag, feat, probs, pf_mean, has_players, 0, 0, 0, 0, False, False, False, 0, 0, 0, 0)
                    if g == 0:
                        score[tm] += 2
                else:
                    tstats[tm, T_XP_ATT] += 1
                    c2, g = decide(XP, rs, st, score, timeouts, tm, home_side, neutral, season, ratings, weather, pf, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, team_off_idx, team_def_idx, pf_idx, weather_flag, feat, probs, pf_mean, has_players, 0, 0, 0, 0, False, False, False, 0, 0, 0, 0)
                    if g == 0:
                        score[tm] += 1
            st[ST_PHASE] = KICKOFF; st[ST_KICK] = tm
            _check_ot_end(st, score, tm)
            p_no_clock = True
        # ------------------------------------------------------------------ scrimmage
        else:
            off = st[ST_POS]; deff = 1 - off
            p_offense = off
            # 1. pre-snap penalty
            c, side = decide(PRESNAP, rs, st, score, timeouts, off, home_side, neutral, season, ratings, weather, pf, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, team_off_idx, team_def_idx, pf_idx, weather_flag, feat, probs, pf_mean, has_players, 0, 0, 0, 0, False, False, False, 0, 0, 0, 0)
            if side != 0:
                tstats[off, T_PENALTIES] += 1
                sidx = 1 if side == 2 else 0
                yds = 5
                if present[PEN_Y]:
                    c, yds = decide(PEN_Y, rs, st, score, timeouts, off, home_side, neutral, season, ratings, weather, pf, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, team_off_idx, team_def_idx, pf_idx, weather_flag, feat, probs, pf_mean, has_players, 0, 0, 0, 0, False, False, False, 0, 0, sidx, 0)
                if side == 1:
                    yds = min(yds, (100 - st[ST_YL]) // 2)
                    st[ST_YL] += yds; st[ST_YTG] += yds
                else:
                    yds = min(yds, st[ST_YL] // 2)
                    st[ST_YL] -= yds
                    c, afd = decide(DEF_FD, rs, st, score, timeouts, off, home_side, neutral, season, ratings, weather, pf, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, team_off_idx, team_def_idx, pf_idx, weather_flag, feat, probs, pf_mean, has_players, 0, 0, 0, 0, False, False, False, 0, 0, 0, 0)
                    if afd == 0 or yds >= st[ST_YTG]:
                        st[ST_DOWN] = 1; st[ST_YTG] = min(10, st[ST_YL]); tstats[off, T_FIRST_DOWNS] += 1
                    else:
                        st[ST_YTG] -= yds
                p_kind = 3
            else:
                done = False
                # 2. fourth down
                if st[ST_DOWN] == 4:
                    c, choice = decide(FOURTH, rs, st, score, timeouts, off, home_side, neutral, season, ratings, weather, pf, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, team_off_idx, team_def_idx, pf_idx, weather_flag, feat, probs, pf_mean, has_players, 0, 0, 0, 0, False, False, False, 0, 0, 0, 0)
                    if choice == 1:   # punt
                        tstats[off, T_PUNTS] += 1
                        y = st[ST_YL]
                        c, res = decide(PUNT_RES, rs, st, score, timeouts, off, home_side, neutral, season, ratings, weather, pf, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, team_off_idx, team_def_idx, pf_idx, weather_flag, feat, probs, pf_mean, has_players, 0, 0, 0, 0, False, False, False, 0, 0, 0, 0)
                        if res == 1:
                            _start_possession(st, score, timeouts, tstats, deff, 80)
                        elif res == 3:
                            _touchdown(st, score, deff)
                        elif res == 2:
                            c, spot = decide(PUNT_RET, rs, st, score, timeouts, off, home_side, neutral, season, ratings, weather, pf, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, team_off_idx, team_def_idx, pf_idx, weather_flag, feat, probs, pf_mean, has_players, 0, 0, 0, 0, False, False, False, 0, 0, 0, 0)
                            _start_possession(st, score, timeouts, tstats, off, spot)
                        else:
                            c, net = decide(PUNT_NET, rs, st, score, timeouts, off, home_side, neutral, season, ratings, weather, pf, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, team_off_idx, team_def_idx, pf_idx, weather_flag, feat, probs, pf_mean, has_players, 0, 0, 0, 0, False, False, False, 0, 0, 0, 0)
                            r = 100 - y + net
                            if r <= 0:
                                r = 80
                            _start_possession(st, score, timeouts, tstats, deff, r)
                        p_kind = 1; done = True
                    elif choice == 2:  # field goal
                        dist = st[ST_YL] + 17
                        tstats[off, T_FGA] += 1
                        for j in range(9):
                            pf[j] = 0.0
                        if kicker[off] >= 0:
                            pf[8] = eff[off, kicker[off], 8]
                        c, made = decide(FG, rs, st, score, timeouts, off, home_side, neutral, season, ratings, weather, pf, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, team_off_idx, team_def_idx, pf_idx, weather_flag, feat, probs, pf_mean, has_players, 0, min(dist, 79), 0, 0, False, False, False, 0, 0, 0, 0)
                        if made == 0:
                            score[off] += 3; tstats[off, T_FGM] += 1
                            st[ST_PHASE] = KICKOFF; st[ST_KICK] = off
                            _check_ot_end(st, score, off)
                        else:
                            _start_possession(st, score, timeouts, tstats, deff, min(100 - st[ST_YL] - 7, 80))
                        p_kind = 2; done = True
                if not done:
                    # 3. kneel rule
                    hs = _half_secs(st)
                    diff = score[off] - score[deff]
                    kneel = False
                    if st[ST_QTR] == 4 and diff > 0:
                        opp_to = timeouts[deff]
                        downs_left = 4 - st[ST_DOWN] + 1
                        killable = kneel_secs * max(0, downs_left - opp_to) + 2 * min(downs_left, opp_to)
                        if st[ST_DOWN] == 4:
                            kneel = opp_to == 0 and hs <= kneel_secs
                        else:
                            kneel = hs <= killable
                    elif st[ST_QTR] == 2 and hs <= 20 and st[ST_YL] >= 50:
                        kneel = True
                    if kneel:
                        tstats[off, T_PLAYS] += 1; tstats[off, T_KNEELS] += 1
                        yards = -1 if st[ST_YL] < 99 else 0
                        p_forced = kneel_secs
                        p_stopper = _apply_result(st, score, timeouts, tstats, pstats, yards, -1, 0, -1, False)
                    else:
                        tstats[off, T_PLAYS] += 1
                        c, call = decide(PLAY_CALL, rs, st, score, timeouts, off, home_side, neutral, season, ratings, weather, pf, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, team_off_idx, team_def_idx, pf_idx, weather_flag, feat, probs, pf_mean, has_players, 0, 0, 0, 0, False, False, False, 0, 0, 0, 0)
                        if call == 0:
                            p_was_pass = True
                            p_stopper, p_oob = _pass_play(rs, st, score, timeouts, tstats, pstats, off, home_side, neutral, season, ratings, weather, pf, feat, probs,
                                                          n_players, qb, target_cum, rz_target_cum, n_targets, target_idx, eff,
                                                          kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, present, team_off_idx, team_def_idx, pf_idx, weather_flag, pf_mean, has_players)
                        else:
                            p_stopper, p_oob = _run_play(rs, st, score, timeouts, tstats, pstats, off, home_side, neutral, season, ratings, weather, pf, feat, probs,
                                                         carrier_cum, rz_carrier_cum, n_carriers, carrier_idx, eff,
                                                         kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, present, team_off_idx, team_def_idx, pf_idx, weather_flag, pf_mean, has_players)
        # ------------------------------------------------------------------ clock
        if not p_no_clock and st[ST_PHASE] != GAME_OVER:
            _advance_clock(rs, st, score, timeouts, tstats, tmw, p_offense, p_stopper, p_forced, p_kind, p_was_pass, p_oob, home_side, neutral, season, ratings, weather, pf, feat, probs,
                           kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, present, team_off_idx, team_def_idx, pf_idx, weather_flag, pf_mean, has_players, to_lo, to_hi)
        st[ST_PLAYS] += 1
    if st[ST_REGH] < 0:
        st[ST_REGH] = score[0]; st[ST_REGA] = score[1]
    return went_ot


@njit(cache=True)
def _start_possession(st, score, timeouts, tstats, tm, yl):
    st[ST_POS] = tm; st[ST_DOWN] = 1
    st[ST_YL] = min(max(yl, 1), 99)
    st[ST_YTG] = min(10, st[ST_YL])
    st[ST_PHASE] = SCRIMMAGE
    tstats[tm, T_DRIVES] += 1
    if st[ST_OTA] == 1:
        st[ST_OTC] += 1
        _check_ot_end(st, score, -1)


@njit(cache=True)
def _touchdown(st, score, tm):
    score[tm] += 6
    st[ST_PHASE] = PAT
    st[ST_PATT] = tm


@njit(cache=True)
def _safety(st, score, conceding):
    scorer = 1 - conceding
    score[scorer] += 2
    st[ST_PHASE] = FREE_KICK
    st[ST_KICK] = conceding
    st[ST_POS] = scorer
    _check_ot_end(st, score, scorer)


@njit(cache=True)
def _check_ot_end(st, score, scoring_team):
    if st[ST_OTA] == 0 or st[ST_PHASE] == GAME_OVER:
        return
    n = st[ST_OTC]
    tied = score[0] == score[1]
    if scoring_team >= 0:
        if n <= 1:
            if scoring_team != st[ST_OTF]:
                st[ST_PHASE] = GAME_OVER
        elif n == 2:
            if not tied:
                st[ST_PHASE] = GAME_OVER
        else:
            st[ST_PHASE] = GAME_OVER
    else:
        if n >= 3 and not tied:
            st[ST_PHASE] = GAME_OVER


@njit(cache=True)
def _ot_decided_before_try(st, score, td_team):
    if st[ST_OTA] == 0:
        return False
    n = st[ST_OTC]
    lead = score[td_team] - score[1 - td_team]
    if n <= 1:
        return td_team != st[ST_OTF]
    if n == 2:
        return lead > 0
    return True


@njit(cache=True)
def _apply_result(st, score, timeouts, tstats, pstats, yards, scorer, kind_code, passer, is_pass):
    """Returns the clock-stopper flag. kind_code: 0 rush, 1 pass (for TD credit)."""
    off = st[ST_POS]; deff = 1 - off
    new_spot = st[ST_YL] - yards
    if new_spot <= 0:
        _touchdown(st, score, off)
        if scorer >= 0:
            pstats[off, scorer, P_REC_TD if is_pass else P_RUSH_TD] += 1
            pstats[off, scorer, P_ANY_TD] = 1
            if is_pass and passer >= 0:
                pstats[off, passer, P_PASS_TD] += 1
        return True
    if new_spot >= 100:
        _safety(st, score, off)
        return True
    if yards >= st[ST_YTG]:
        st[ST_DOWN] = 1; st[ST_YL] = new_spot; st[ST_YTG] = min(10, new_spot)
        tstats[off, T_FIRST_DOWNS] += 1
        return False
    if st[ST_DOWN] >= 4:
        _start_possession(st, score, timeouts, tstats, deff, 100 - new_spot)
        return True
    st[ST_DOWN] += 1
    st[ST_YL] = new_spot
    st[ST_YTG] = min(st[ST_YTG] - yards, new_spot)
    return False


@njit(cache=True)
def _fumble(rs, st, score, timeouts, tstats, pstats, off, pt, yards, carrier, home_side, neutral, season, ratings, weather, pf, feat, probs,
            kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, present, team_off_idx, team_def_idx, pf_idx, weather_flag, pf_mean, has_players):
    """Returns True if possession changed."""
    deff = 1 - off
    c, res = decide(FUMBLE, rs, st, score, timeouts, off, home_side, neutral, season, ratings, weather, pf, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, team_off_idx, team_def_idx, pf_idx, weather_flag, feat, probs, pf_mean, has_players, 0, 0, pt, yards, False, False, False, 0, 0, 0, 0)
    if res == 0:
        return False
    if res == 1 and n_classes[FUMBLE] == 2 and present[FUM_TD]:
        c2, td = decide(FUM_TD, rs, st, score, timeouts, off, home_side, neutral, season, ratings, weather, pf, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, team_off_idx, team_def_idx, pf_idx, weather_flag, feat, probs, pf_mean, has_players, 0, 0, 0, 0, False, False, False, 0, 0, 0, 0)
        if td == 0:
            res = 2
    tstats[off, T_FUM_LOST] += 1; tstats[off, T_TURNOVERS] += 1
    if carrier >= 0:
        pstats[off, carrier, P_FUM] += 1
    if res == 2:
        _touchdown(st, score, deff)
        return True
    new_spot = st[ST_YL] - yards
    if new_spot <= 0:
        _start_possession(st, score, timeouts, tstats, deff, 80)
    elif new_spot >= 100:
        _start_possession(st, score, timeouts, tstats, deff, 1)
    else:
        _start_possession(st, score, timeouts, tstats, deff, 100 - new_spot)
    return True


@njit(cache=True)
def _oob(rs, st, score, timeouts, off, pt, home_side, neutral, season, ratings, weather, pf, feat, probs,
         kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, present, team_off_idx, team_def_idx, pf_idx, weather_flag, pf_mean, has_players):
    hs = _half_secs(st)
    window = hs <= 120 if _half(st) == 1 else hs <= 300
    if not window or not present[OOB]:
        return False
    c, v = decide(OOB, rs, st, score, timeouts, off, home_side, neutral, season, ratings, weather, pf, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, team_off_idx, team_def_idx, pf_idx, weather_flag, feat, probs, pf_mean, has_players, 0, 0, pt, 0, False, False, False, 0, 0, 0, 0)
    return v == 0


@njit(cache=True)
def _pick(cum_row, n, u):
    if n == 0:
        return -1
    for i in range(n):
        if u < cum_row[i]:
            return i
    return n - 1


@njit(cache=True)
def _pass_play(rs, st, score, timeouts, tstats, pstats, off, home_side, neutral, season, ratings, weather, pf, feat, probs,
               n_players, qb, target_cum, rz_target_cum, n_targets, target_idx, eff,
               kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, present, team_off_idx, team_def_idx, pf_idx, weather_flag, pf_mean, has_players):
    deff = 1 - off
    q = qb[off]
    for j in range(9):
        pf[j] = 0.0
    if q >= 0:
        for j in range(5):
            pf[j] = eff[off, q, j]
    c, pre = decide(DROPBACK, rs, st, score, timeouts, off, home_side, neutral, season, ratings, weather, pf, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, team_off_idx, team_def_idx, pf_idx, weather_flag, feat, probs, pf_mean, has_players, 0, 0, 0, 0, False, False, False, 0, 0, 0, 0)
    if pre == 1:   # sack
        c, yards = decide(SACK_Y, rs, st, score, timeouts, off, home_side, neutral, season, ratings, weather, pf, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, team_off_idx, team_def_idx, pf_idx, weather_flag, feat, probs, pf_mean, has_players, 0, 0, 0, 0, False, False, False, 0, 0, 0, 0)
        tstats[off, T_SACKS] += 1; tstats[off, T_SACK_YDS] += -yards
        if _fumble(rs, st, score, timeouts, tstats, pstats, off, 1, yards, q, home_side, neutral, season, ratings, weather, pf, feat, probs, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, present, team_off_idx, team_def_idx, pf_idx, weather_flag, pf_mean, has_players):
            return True, False
        return _apply_result(st, score, timeouts, tstats, pstats, yards, -1, 0, -1, False), False
    if pre == 2:   # scramble
        c, yards = decide(SCR_Y, rs, st, score, timeouts, off, home_side, neutral, season, ratings, weather, pf, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, team_off_idx, team_def_idx, pf_idx, weather_flag, feat, probs, pf_mean, has_players, 0, 0, 0, 0, False, False, False, 0, 0, 0, 0)
        yards = min(yards, st[ST_YL])
        tstats[off, T_SCRAMBLES] += 1; tstats[off, T_RUSH_ATT] += 1; tstats[off, T_RUSH_YDS] += yards
        if q >= 0:
            pstats[off, q, P_RUSH_ATT] += 1; pstats[off, q, P_RUSH_YDS] += yards
            if yards > pstats[off, q, P_LONG_RUSH]:
                pstats[off, q, P_LONG_RUSH] = yards
        if _fumble(rs, st, score, timeouts, tstats, pstats, off, 2, yards, q, home_side, neutral, season, ratings, weather, pf, feat, probs, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, present, team_off_idx, team_def_idx, pf_idx, weather_flag, pf_mean, has_players):
            return True, False
        oob = _oob(rs, st, score, timeouts, off, 2, home_side, neutral, season, ratings, weather, pf, feat, probs, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, present, team_off_idx, team_def_idx, pf_idx, weather_flag, pf_mean, has_players)
        stopper = _apply_result(st, score, timeouts, tstats, pstats, yards, q, 0, -1, False)
        return (stopper or oob), oob
    # throw
    c, air = decide(AIR, rs, st, score, timeouts, off, home_side, neutral, season, ratings, weather, pf, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, team_off_idx, team_def_idx, pf_idx, weather_flag, feat, probs, pf_mean, has_players, 0, 0, 0, 0, False, False, False, 0, 0, 0, 0)
    air = min(air, st[ST_YL])
    band = 0 if air < 5 else (1 if air < 15 else 2)
    rcv = -1
    if n_targets[off] > 0:
        row = rz_target_cum[off, band] if st[ST_YL] <= 20 else target_cum[off, band]
        k = _pick(row, n_targets[off], rand_u(rs))
        if k >= 0:
            rcv = target_idx[off, k]
    if rcv >= 0:
        pf[5] = eff[off, rcv, 5]; pf[6] = eff[off, rcv, 6]
    tstats[off, T_PASS_ATT] += 1
    if q >= 0:
        pstats[off, q, P_PASS_ATT] += 1
    if rcv >= 0:
        pstats[off, rcv, P_TGT] += 1
    c, outcome = decide(THROW, rs, st, score, timeouts, off, home_side, neutral, season, ratings, weather, pf, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, team_off_idx, team_def_idx, pf_idx, weather_flag, feat, probs, pf_mean, has_players, air, 0, 0, 0, False, False, False, 0, 0, 0, 0)
    if outcome == 0:   # complete
        c, yac = decide(YAC, rs, st, score, timeouts, off, home_side, neutral, season, ratings, weather, pf, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, team_off_idx, team_def_idx, pf_idx, weather_flag, feat, probs, pf_mean, has_players, air, 0, 0, 0, False, False, False, 0, 0, 0, 0)
        yards = min(air + yac, st[ST_YL])
        tstats[off, T_COMP] += 1; tstats[off, T_PASS_YDS] += yards
        if q >= 0:
            pstats[off, q, P_COMP] += 1; pstats[off, q, P_PASS_YDS] += yards
        if rcv >= 0:
            pstats[off, rcv, P_REC] += 1; pstats[off, rcv, P_REC_YDS] += yards
            if yards > pstats[off, rcv, P_LONG_REC]:
                pstats[off, rcv, P_LONG_REC] = yards
        if _fumble(rs, st, score, timeouts, tstats, pstats, off, 0, yards, rcv, home_side, neutral, season, ratings, weather, pf, feat, probs, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, present, team_off_idx, team_def_idx, pf_idx, weather_flag, pf_mean, has_players):
            return True, False
        oob = _oob(rs, st, score, timeouts, off, 0, home_side, neutral, season, ratings, weather, pf, feat, probs, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, present, team_off_idx, team_def_idx, pf_idx, weather_flag, pf_mean, has_players)
        stopper = _apply_result(st, score, timeouts, tstats, pstats, yards, rcv, 1, q, True)
        return (stopper or oob), oob
    if outcome == 1:   # incomplete
        _apply_result(st, score, timeouts, tstats, pstats, 0, -1, 0, -1, False)
        return True, False
    # interception
    tstats[off, T_INT] += 1; tstats[off, T_TURNOVERS] += 1
    if q >= 0:
        pstats[off, q, P_INT] += 1
    spot = st[ST_YL] - air
    c, six = decide(PICK6, rs, st, score, timeouts, off, home_side, neutral, season, ratings, weather, pf, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, team_off_idx, team_def_idx, pf_idx, weather_flag, feat, probs, pf_mean, has_players, 0, 0, 0, 0, False, False, False, 0, 0, 0, 0)
    if six == 0:
        _touchdown(st, score, deff)
        return True, False
    c, ret = decide(INT_RET, rs, st, score, timeouts, off, home_side, neutral, season, ratings, weather, pf, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, team_off_idx, team_def_idx, pf_idx, weather_flag, feat, probs, pf_mean, has_players, 0, 0, 0, 0, False, False, False, 0, 0, 0, 0)
    new_yl = 80 if spot <= 0 else min(max(100 - spot - ret, 1), 99)
    _start_possession(st, score, timeouts, tstats, deff, new_yl)
    return True, False


@njit(cache=True)
def _run_play(rs, st, score, timeouts, tstats, pstats, off, home_side, neutral, season, ratings, weather, pf, feat, probs,
              carrier_cum, rz_carrier_cum, n_carriers, carrier_idx, eff,
              kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, present, team_off_idx, team_def_idx, pf_idx, weather_flag, pf_mean, has_players):
    carrier = -1
    if n_carriers[off] > 0:
        row = rz_carrier_cum[off] if st[ST_YL] <= 20 else carrier_cum[off]
        k = _pick(row, n_carriers[off], rand_u(rs))
        if k >= 0:
            carrier = carrier_idx[off, k]
    for j in range(9):
        pf[j] = 0.0
    if carrier >= 0:
        pf[7] = eff[off, carrier, 7]
    c, yards = decide(RUSH, rs, st, score, timeouts, off, home_side, neutral, season, ratings, weather, pf, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, team_off_idx, team_def_idx, pf_idx, weather_flag, feat, probs, pf_mean, has_players, 0, 0, 0, 0, False, False, False, 0, 0, 0, 0)
    yards = min(yards, st[ST_YL])
    tstats[off, T_RUSH_ATT] += 1; tstats[off, T_RUSH_YDS] += yards
    if carrier >= 0:
        pstats[off, carrier, P_RUSH_ATT] += 1; pstats[off, carrier, P_RUSH_YDS] += yards
        if yards > pstats[off, carrier, P_LONG_RUSH]:
            pstats[off, carrier, P_LONG_RUSH] = yards
    if _fumble(rs, st, score, timeouts, tstats, pstats, off, 3, yards, carrier, home_side, neutral, season, ratings, weather, pf, feat, probs, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, present, team_off_idx, team_def_idx, pf_idx, weather_flag, pf_mean, has_players):
        return True, False
    oob = _oob(rs, st, score, timeouts, off, 3, home_side, neutral, season, ratings, weather, pf, feat, probs, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, present, team_off_idx, team_def_idx, pf_idx, weather_flag, pf_mean, has_players)
    stopper = _apply_result(st, score, timeouts, tstats, pstats, yards, carrier, 0, -1, False)
    return (stopper or oob), oob


@njit(cache=True)
def _advance_clock(rs, st, score, timeouts, tstats, tmw, offense, stopper, forced, p_kind, was_pass, oob, home_side, neutral, season, ratings, weather, pf, feat, probs,
                   kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, present, team_off_idx, team_def_idx, pf_idx, weather_flag, pf_mean, has_players, to_lo, to_hi):
    hs = _half_secs(st)
    if hs <= 0:
        st[ST_EXP] = 1
        return
    opp = 1 - offense
    diff = score[offense] - score[opp]
    under_two = hs <= 120
    tied = score[0] == score[1]
    leader = 0 if score[0] > score[1] else 1
    cand = (1 - leader) if (under_two and not tied) else offense
    timeout_by = -1
    for j in range(9):
        pf[j] = 0.0
    if timeouts[cand] > 0:
        c, v = decide(TO_AFTER, rs, st, score, timeouts, offense, home_side, neutral, season, ratings, weather, pf, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, team_off_idx, team_def_idx, pf_idx, weather_flag, feat, probs, pf_mean, has_players, 0, 0, 0, 0, stopper, was_pass, oob, 0, 0, 0, 0)
        if v == 0:
            timeout_by = cand
    if timeout_by >= 0:
        c, elapsed = decide(TO_RUN, rs, st, score, timeouts, offense, home_side, neutral, season, ratings, weather, pf, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, team_off_idx, team_def_idx, pf_idx, weather_flag, feat, probs, pf_mean, has_players, 0, 0, 0, 0, stopper, was_pass, oob, 0, 0, 0, 0)
    elif forced >= 0:
        elapsed = forced
    elif p_kind == 4 and present[KO_RUN]:
        c, elapsed = decide(KO_RUN, rs, st, score, timeouts, offense, home_side, neutral, season, ratings, weather, pf, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, team_off_idx, team_def_idx, pf_idx, weather_flag, feat, probs, pf_mean, has_players, 0, 0, 0, 0, stopper, was_pass, oob, 0, 0, 0, 0)
    else:
        kk = p_kind if p_kind <= 3 else 0
        c, elapsed = decide(CLOCK, rs, st, score, timeouts, offense, home_side, neutral, season, ratings, weather, pf, kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge, home_dir, home_offset, team_off_idx, team_def_idx, pf_idx, weather_flag, feat, probs, pf_mean, has_players, 0, 0, 0, 0, stopper, was_pass, oob, kk, 0, 0, 0)
        if under_two and not stopper and st[ST_PHASE] == SCRIMMAGE and st[ST_POS] == offense and diff >= -3 and diff <= 0 \
                and timeouts[offense] > 0 and st[ST_YL] <= 40 and hs - elapsed < 8 and hs > 8:
            timeout_by = offense
            elapsed = to_lo + int(rand_u(rs) * (to_hi - to_lo + 1))
    if timeout_by >= 0:
        timeouts[timeout_by] -= 1
    elapsed = max(0, min(elapsed, hs))
    tstats[offense, T_TOP] += elapsed
    new = hs - elapsed
    h = _half(st)
    if hs > 120 and new <= 120 and tmw[h] == 0:
        new = 120
        tmw[h] = 1
    if st[ST_QTR] < 5 and (st[ST_QTR] == 1 or st[ST_QTR] == 3) and hs > 900 and new <= 900:
        new = 900
        st[ST_QTR] += 1
    if new <= 0:
        new = 0
        st[ST_EXP] = 1
    _set_clock(st, new)


@njit(parallel=True, cache=True)
def _simulate_batch(n, seed, home_side, neutral, playoff, season, ratings, weather, cfg,
                    n_players, qb, kicker, target_cum, rz_target_cum, carrier_cum, rz_carrier_cum, n_targets, n_carriers, target_idx, carrier_idx, eff,
                    kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge,
                    home_dir, home_offset, present, team_off_idx, team_def_idx, pf_idx, weather_flag, pf_mean, n_max,
                    home_score, away_score, went_ot, tstats_out, pstats_out):
    has_players = np.zeros(2, dtype=np.int64)
    for s in range(2):
        has_players[s] = 1 if n_players[s] > 0 else 0
    for i in prange(n):
        rs = np.zeros(1, dtype=np.uint64)
        rs[0] = np.uint64(seed) * np.uint64(6364136223846793005) + np.uint64(i + 1) * np.uint64(1442695040888963407)
        score = np.zeros(2, dtype=np.int64)
        timeouts = np.zeros(2, dtype=np.int64)
        tstats = np.zeros((2, N_TSTATS), dtype=np.int64)
        pstats = np.zeros((2, n_max, 16), dtype=np.int64)
        ot = simulate_one(rs, home_side, neutral, playoff, season, ratings, weather, cfg,
                          n_players, qb, kicker, target_cum, rz_target_cum, carrier_cum, rz_carrier_cum, n_targets, n_carriers, target_idx, carrier_idx, eff,
                          kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, mask_kind, lower_edge,
                          home_dir, home_offset, present, team_off_idx, team_def_idx, pf_idx, weather_flag, pf_mean, has_players, score, timeouts, tstats, pstats)
        home_score[i] = score[0]
        away_score[i] = score[1]
        went_ot[i] = ot
        for s in range(2):
            for k in range(N_TSTATS):
                tstats_out[i, s, k] = tstats[s, k]
            for p in range(n_max):
                for k in range(16):
                    pstats_out[i, s * n_max + p, k] = pstats[s, p, k]


class FastModels:
    """Typed-list views of a PackedModels, built once per process."""

    def __init__(self, P):
        self.P = P
        (self.cum, self.dims, self.w_off, self.w_vals, self.w_cum, self.mean, self.std, self.W, self.b) = typed_lists(P)


def simulate_games(n: int, seed: int, FM: FastModels, G, playoff: bool = False):
    P = FM.P
    n_max = int(G.eff.shape[1])
    home_score = np.zeros(n, dtype=np.int64)
    away_score = np.zeros(n, dtype=np.int64)
    went_ot = np.zeros(n, dtype=np.int64)
    tstats = np.zeros((n, 2, N_TSTATS), dtype=np.int64)
    pstats = np.zeros((n, 2 * n_max, 16), dtype=np.int64)
    _simulate_batch(n, int(seed), 0, int(G.neutral), int(playoff or G.playoff), int(G.season), G.ratings, G.weather, P.engine_cfg,
                    G.n_players, G.qb, G.kicker, G.target_cum, G.rz_target_cum, G.carrier_cum, G.rz_carrier_cum, G.n_targets, G.n_carriers,
                    G.target_idx, G.carrier_idx, G.eff,
                    P.kind, P.n_classes, P.class_values, FM.cum, FM.dims, FM.w_off, FM.w_vals, FM.w_cum, FM.mean, FM.std, FM.W, FM.b,
                    P.binary, P.team, P.mask_kind, P.lower_edge, P.home_dir, P.home_offset, P.present, P.team_off_idx, P.team_def_idx,
                    P.pf_idx, P.weather_flag, P.pf_mean, n_max, home_score, away_score, went_ot, tstats, pstats)
    return home_score, away_score, went_ot, tstats, pstats
