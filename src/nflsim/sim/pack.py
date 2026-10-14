"""Phase 5, step 5.1: pack models, tables and per-game context into plain NumPy arrays for the jitted engine.

Every decision becomes a slot with a kind: 0 = dense lookup table (cumulative probabilities over all key combinations,
with the Phase 1 back-off baked in), 1 = trained logistic model (standardization, coefficients, masks, home offset).
Classes are integers (exact values for yardage tables, class indices otherwise); optional within-class draws are
packed as flat (values, cumulative probabilities) arrays.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nflsim.config import settings
from nflsim.engine.boxscore import PLAYER_STATS
from nflsim.features.players import EFFICIENCIES
from nflsim.features.team_strength import RATINGS
from nflsim.models import buckets as B
from nflsim.models.features import (BASE_NAMES, PLAYER_FEATURES, SPECS, TEAM_FEATURES, WEATHER_MODELS, feature_names)
from nflsim.models.trained import TrainedModel, YARD_LOWER_EDGE

# decision slots (order fixed; the engine indexes by these constants)
SLOTS = ["play_call", "fourth_down", "air_yards", "dropback_outcome", "throw_outcome", "yac", "sack_yards",
         "scramble_yards", "rush_yards", "fumble_lost", "fumble_return_td", "out_of_bounds", "pick_six",
         "int_return_yards", "fg_make", "punt_result", "punt_net_yards", "punt_retain_spot", "kickoff_result",
         "kickoff_spot", "free_kick_spot", "pat_choice", "xp_result", "two_point_result", "clock_runoff",
         "kickoff_runoff", "timeout_after", "timeout_runoff", "presnap_penalty", "penalty_yards",
         "def_penalty_first_down"]
SLOT = {name: i for i, name in enumerate(SLOTS)}
# key domains for dense tables: (key names, sizes). Keys are produced by fast_features.key_index().
KEY_DOMAINS: dict[str, list[tuple[str, int]]] = {
    "play_call": [("down3", 5), ("dist", 5), ("sb", 7), ("tb", 6)],
    "fourth_down": [("dist", 5), ("yl5", 20), ("st", 8)],
    "air_yards": [("down", 5), ("dist", 5), ("zone", 4)],
    "dropback_outcome": [("zone", 4), ("down", 5), ("dist", 5)],
    "throw_outcome": [("air_bin", 7), ("zone", 4)],
    "yac": [("air_bin", 7), ("zone", 4)],
    "sack_yards": [], "scramble_yards": [("zone", 4)], "rush_yards": [("dist", 5), ("zone", 4)],
    "fumble_lost": [("pt", 4)], "fumble_return_td": [], "out_of_bounds": [("pt", 4), ("lead", 3), ("u2h", 3)],
    "pick_six": [], "int_return_yards": [], "fg_make": [("distance", 62)],
    "punt_result": [("yl5", 20)], "punt_net_yards": [("yl5", 20)], "punt_retain_spot": [],
    "kickoff_result": [], "kickoff_spot": [], "free_kick_spot": [],
    "pat_choice": [("sd", 33), ("tb", 6)], "xp_result": [], "two_point_result": [],
    "clock_runoff": [("cs", 2), ("lead", 3), ("u2h", 3)], "kickoff_runoff": [],
    "timeout_after": [("cs", 2), ("lead", 3), ("u2h", 3)], "timeout_runoff": [("cs", 2), ("u2h", 3)],
    "presnap_penalty": [("down", 5), ("dist", 5)], "penalty_yards": [("side", 2)], "def_penalty_first_down": [],
}
# string classes -> integer codes, per decision (int-class tables keep their values)
CLASS_CODES: dict[str, list[str]] = {
    "play_call": ["pass", "run"], "fourth_down": ["go", "punt", "field_goal"],
    "dropback_outcome": ["throw", "sack", "scramble"], "throw_outcome": ["complete", "incomplete", "interception"],
    "fumble_lost": ["no", "yes", "yes_td"], "fumble_return_td": ["yes", "no"], "out_of_bounds": ["yes", "no"],
    "pick_six": ["yes", "no"], "fg_make": ["made", "missed"], "punt_result": ["net", "touchback", "retain", "return_td"],
    "kickoff_result": ["normal", "return_td", "kicking_team_ball"], "pat_choice": ["kick", "two"],
    "xp_result": ["good", "no_good"], "two_point_result": ["good", "no_good"], "timeout_after": ["yes", "no"],
    "presnap_penalty": ["none", "offense", "defense"], "def_penalty_first_down": ["yes", "no"],
}
MAX_CLASSES = 128
PLAYER_EFF_ORDER = ["qb_cpoe", "qb_sack_rate", "qb_int_rate", "qb_scramble_rate", "qb_adot", "rec_catch_rate_oe",
                    "rec_yac_oe", "rush_yards_oe", "kicker_fg_oe"]
N_PSTATS = len(PLAYER_STATS)


def _key_tuple_from_index(domains: list[tuple[str, int]], idx: tuple[int, ...]) -> tuple:
    """Map dense indices back to the Table's own key values (same except offsets for sd and distance)."""
    out = []
    for (name, _), i in zip(domains, idx):
        if name == "sd":
            out.append(i - 16)
        elif name == "distance":
            out.append(i + 18)
        else:
            out.append(i)
    return tuple(out)


@dataclass
class PackedDecision:
    kind: int                               # 0 table, 1 trained
    n_classes: int
    class_values: np.ndarray                # int64 [n_classes]: exact value for int classes, code index otherwise
    cum: np.ndarray = field(default_factory=lambda: np.zeros(0))       # table: flat [prod(dims) * n_classes]
    dims: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    # within-class draws: for class c, values[w_off[c]:w_off[c+1]] with cumulative probs w_cum[...]
    w_off: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int64))
    w_vals: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    w_cum: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # trained
    mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    std: np.ndarray = field(default_factory=lambda: np.ones(0))
    W: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    b: np.ndarray = field(default_factory=lambda: np.zeros(0))
    binary: int = 0
    team: int = 0
    n_features: int = 0
    mask_kind: int = 0                      # 0 none, 1 air, 2 yac, 3 rush
    lower_edge: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    home_dir: np.ndarray = field(default_factory=lambda: np.zeros(0))
    home_offset: float = 0.0


def _pack_within(within: dict, classes: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offs = [0]
    vals: list[int] = []
    cums: list[float] = []
    for c in classes:
        if c in within:
            v, cu = within[c]
            vals.extend(int(x) for x in v)
            cums.extend(float(x) for x in cu)
        offs.append(len(vals))
    return np.array(offs, dtype=np.int64), np.array(vals, dtype=np.int64), np.array(cums, dtype=float)


def pack_decision(name: str, model) -> PackedDecision:
    classes = list(model.classes)
    if all(isinstance(c, (int, np.integer)) for c in classes):
        class_values = np.array([int(c) for c in classes], dtype=np.int64)
    else:
        codes = CLASS_CODES[name]
        class_values = np.array([codes.index(str(c)) for c in classes], dtype=np.int64)
    within = getattr(model, "within", {}) or {}
    w_off, w_vals, w_cum = _pack_within(within, classes)
    if isinstance(model, TrainedModel):
        pd = PackedDecision(kind=1, n_classes=len(classes), class_values=class_values, w_off=w_off, w_vals=w_vals, w_cum=w_cum,
                            mean=model.mean.astype(float), std=model.std.astype(float), W=model.W.astype(float), b=model.b.astype(float),
                            binary=int(model.binary), team=int(model.team), n_features=len(model.feature_names))
        if name in YARD_LOWER_EDGE:
            pd.mask_kind = {"air_yards": 1, "yac": 2, "rush_yards": 3}[name]
            pd.lower_edge = np.array([YARD_LOWER_EDGE[name](c) for c in classes], dtype=np.int64)
        if model.home_dir is not None:
            pd.home_dir = model.home_dir.astype(float)
            pd.home_offset = float(model.home_offset)
        return pd
    # dense table: iterate every key combination; the Table lookup applies the back-off
    table = model.table
    domains = KEY_DOMAINS[name]
    dims = np.array([n for _, n in domains], dtype=np.int64)
    total = int(np.prod(dims)) if len(dims) else 1
    cum = np.zeros(total * len(classes))
    for flat in range(total):
        idx = np.unravel_index(flat, dims) if len(dims) else ()
        key = _key_tuple_from_index(domains, tuple(int(i) for i in idx))
        try:
            p = table.proba(key)
        except KeyError:
            p = table.proba(())
        c = np.cumsum(p)
        c[-1] = 1.0 + 1e-12
        cum[flat * len(classes):(flat + 1) * len(classes)] = c
    return PackedDecision(kind=0, n_classes=len(classes), class_values=class_values, cum=cum, dims=dims,
                          w_off=w_off, w_vals=w_vals, w_cum=w_cum)


@dataclass
class PackedModels:
    """Arrays indexed by slot; lists hold per-slot arrays of different lengths (Numba typed lists are built in the engine)."""
    kind: np.ndarray
    n_classes: np.ndarray
    class_values: np.ndarray          # [n_slots, MAX_CLASSES]
    cum: list[np.ndarray]
    dims: list[np.ndarray]
    w_off: list[np.ndarray]
    w_vals: list[np.ndarray]
    w_cum: list[np.ndarray]
    mean: list[np.ndarray]
    std: list[np.ndarray]
    W: list[np.ndarray]
    b: list[np.ndarray]
    binary: np.ndarray
    team: np.ndarray
    n_features: np.ndarray
    mask_kind: np.ndarray
    lower_edge: np.ndarray            # [n_slots, MAX_CLASSES]
    home_dir: np.ndarray              # [n_slots, MAX_CLASSES]
    home_offset: np.ndarray
    present: np.ndarray
    engine_cfg: np.ndarray            # [kneel_seconds, timeout_lo, timeout_hi, free_kick_default, ot_reg_secs, ot_reg_to, ot_po_secs, ot_po_to, reg_ties, po_ties, input_scale]
    team_off_idx: np.ndarray = None   # [n_slots, 8] rating indices for the offense team features (-1 padded)
    team_def_idx: np.ndarray = None
    pf_idx: np.ndarray = None         # [n_slots, 8] PLAYER_EFF_ORDER indices for the player features
    weather_flag: np.ndarray = None   # [n_slots]
    pf_mean: np.ndarray = None        # [n_slots, 8] training means of the player features (used when no players are tracked)


def pack_models(M: dict) -> PackedModels:
    n = len(SLOTS)
    kind = np.zeros(n, dtype=np.int64)
    n_classes = np.zeros(n, dtype=np.int64)
    class_values = np.zeros((n, MAX_CLASSES), dtype=np.int64)
    lower_edge = np.zeros((n, MAX_CLASSES), dtype=np.int64)
    home_dir = np.zeros((n, MAX_CLASSES))
    binary = np.zeros(n, dtype=np.int64)
    team = np.zeros(n, dtype=np.int64)
    n_features = np.zeros(n, dtype=np.int64)
    mask_kind = np.zeros(n, dtype=np.int64)
    home_offset = np.zeros(n)
    present = np.zeros(n, dtype=np.int64)
    cum, dims, w_off, w_vals, w_cum, mean, std, W, b = ([] for _ in range(9))
    for i, name in enumerate(SLOTS):
        if name not in M:
            for lst, default in ((cum, np.zeros(1)), (dims, np.zeros(0, dtype=np.int64)), (w_off, np.zeros(1, dtype=np.int64)),
                                 (w_vals, np.zeros(0, dtype=np.int64)), (w_cum, np.zeros(0)), (mean, np.zeros(1)), (std, np.ones(1)),
                                 (W, np.zeros((1, 1))), (b, np.zeros(1))):
                lst.append(default)
            continue
        pd = pack_decision(name, M[name])
        present[i] = 1
        kind[i], n_classes[i] = pd.kind, pd.n_classes
        assert pd.n_classes <= MAX_CLASSES
        class_values[i, :pd.n_classes] = pd.class_values
        cum.append(pd.cum if pd.cum.size else np.zeros(1))
        dims.append(pd.dims)
        w_off.append(pd.w_off)
        w_vals.append(pd.w_vals)
        w_cum.append(pd.w_cum)
        mean.append(pd.mean if pd.mean.size else np.zeros(1))
        std.append(pd.std if pd.std.size else np.ones(1))
        W.append(pd.W if pd.W.size else np.zeros((1, 1)))
        b.append(pd.b if pd.b.size else np.zeros(1))
        binary[i], team[i], n_features[i], mask_kind[i], home_offset[i] = pd.binary, pd.team, pd.n_features, pd.mask_kind, pd.home_offset
        if pd.lower_edge.size:
            lower_edge[i, :pd.n_classes] = pd.lower_edge
        if pd.home_dir.size:
            home_dir[i, :pd.n_classes] = pd.home_dir
    s = settings()
    eng = s["engine"]
    cfg = np.array([eng["kneel_seconds"], eng["timeout_play_seconds"][0], eng["timeout_play_seconds"][1], eng["free_kick_default_yardline"],
                    s["regular_season_ot"]["period_seconds"], s["regular_season_ot"]["timeouts"], s["playoff_ot"]["period_seconds"],
                    s["playoff_ot"]["timeouts"], 1.0 if s["regular_season_ot"]["ties_allowed"] else 0.0,
                    1.0 if s["playoff_ot"]["ties_allowed"] else 0.0, s["team_strength"].get("input_scale", 1.0)], dtype=float)
    pm = PackedModels(kind, n_classes, class_values, cum, dims, w_off, w_vals, w_cum, mean, std, W, b, binary, team, n_features,
                      mask_kind, lower_edge, home_dir, home_offset, present, cfg)
    pm.team_off_idx = -np.ones((n, 8), dtype=np.int64)
    pm.team_def_idx = -np.ones((n, 8), dtype=np.int64)
    pm.pf_idx = -np.ones((n, 8), dtype=np.int64)
    pm.weather_flag = np.zeros(n, dtype=np.int64)
    pm.pf_mean = np.zeros((n, 8))
    for i, name in enumerate(SLOTS):
        if name in M and isinstance(M[name], TrainedModel):
            m = M[name]
            for k, f in enumerate(PLAYER_FEATURES.get(name, [])):
                j = m.feature_names.index(f"p_{f}") if f"p_{f}" in m.feature_names else -1
                if j >= 0:
                    pm.pf_mean[i, k] = float(m.mean[j])
        off, deff = TEAM_FEATURES.get(name, ([], []))
        for k, r in enumerate(off):
            pm.team_off_idx[i, k] = RATINGS.index(r)
        for k, r in enumerate(deff):
            pm.team_def_idx[i, k] = RATINGS.index(r)
        for k, f in enumerate(PLAYER_FEATURES.get(name, [])):
            pm.pf_idx[i, k] = PLAYER_EFF_ORDER.index(f)
        pm.weather_flag[i] = 1 if name in WEATHER_MODELS else 0
    return pm


# ---- game context ------------------------------------------------------------------------------------------------
@dataclass
class PackedGame:
    ratings: np.ndarray        # [2, n_ratings] home, away (already multiplied by input_scale)
    weather: np.ndarray        # [roof_dome, wind or -1 for missing]
    season: int
    neutral: int
    playoff: int
    # players: side 0 = home, 1 = away; arrays padded to n_max
    n_players: np.ndarray      # [2]
    qb: np.ndarray             # [2] index into side's players or -1
    kicker: np.ndarray         # [2]
    target_cum: np.ndarray     # [2, 3, n_max] cumulative by band (0 = none)
    rz_target_cum: np.ndarray  # [2, 3, n_max]
    carrier_cum: np.ndarray    # [2, n_max]
    rz_carrier_cum: np.ndarray # [2, n_max]
    n_targets: np.ndarray      # [2]
    n_carriers: np.ndarray     # [2]
    target_idx: np.ndarray     # [2, n_max] player index for each target slot
    carrier_idx: np.ndarray    # [2, n_max]
    eff: np.ndarray            # [2, n_max, 9] in PLAYER_EFF_ORDER (qb_* taken from the QB row, rec_* from the receiver, ...)
    player_ids: list[list[str]]


def pack_game(ctx, home: str, away: str, neutral: bool = False, playoff: bool = False) -> PackedGame:
    scale = float(settings()["team_strength"].get("input_scale", 1.0))
    ratings = np.zeros((2, len(RATINGS)))
    for side, team_r in enumerate((ctx.home, ctx.away)):
        for j, r in enumerate(RATINGS):
            ratings[side, j] = float((team_r or {}).get(r, 0.0) or 0.0) * scale
    weather = np.array([float((ctx.weather or {}).get("roof_dome", 0) or 0), -1.0 if (ctx.weather or {}).get("wind") is None else float(ctx.weather["wind"])])
    tps = [ctx.players.get(home) if ctx.players else None, ctx.players.get(away) if ctx.players else None]
    n_max = max([len(tp.players) for tp in tps if tp is not None] + [1])
    n_players = np.zeros(2, dtype=np.int64)
    qb = -np.ones(2, dtype=np.int64)
    kicker = -np.ones(2, dtype=np.int64)
    target_cum = np.zeros((2, 3, n_max))
    rz_target_cum = np.zeros((2, 3, n_max))
    carrier_cum = np.zeros((2, n_max))
    rz_carrier_cum = np.zeros((2, n_max))
    n_targets = np.zeros(2, dtype=np.int64)
    n_carriers = np.zeros(2, dtype=np.int64)
    target_idx = np.zeros((2, n_max), dtype=np.int64)
    carrier_idx = np.zeros((2, n_max), dtype=np.int64)
    eff = np.zeros((2, n_max, len(PLAYER_EFF_ORDER)))
    ids: list[list[str]] = [[], []]
    eff_map = {"qb_cpoe": "qb_cpoe", "qb_sack_rate": "qb_sack_rate", "qb_int_rate": "qb_int_rate", "qb_scramble_rate": "qb_scramble_rate",
               "qb_adot": "qb_adot", "rec_catch_rate_oe": "catch_rate_oe", "rec_yac_oe": "yac_oe", "rush_yards_oe": "rush_yards_oe",
               "kicker_fg_oe": "kicker_fg_oe"}
    for side, tp in enumerate(tps):
        if tp is None:
            continue
        pids = list(tp.players)
        ids[side] = pids
        index = {pid: i for i, pid in enumerate(pids)}
        n_players[side] = len(pids)
        qb[side] = index.get(tp.qb, -1)
        kicker[side] = index.get(tp.kicker, -1)
        for i, pid in enumerate(pids):
            e = tp.players[pid].get("eff", {})
            for j, name in enumerate(PLAYER_EFF_ORDER):
                eff[side, i, j] = float(e.get(eff_map[name], 0.0) or 0.0)
        n_targets[side] = len(tp.target_ids)
        for k, pid in enumerate(tp.target_ids):
            target_idx[side, k] = index[pid]
        for band in range(3):
            if tp.target_cum[band]:
                target_cum[side, band, :len(tp.target_cum[band])] = tp.target_cum[band]
            if tp.rz_target_cum[band]:
                rz_target_cum[side, band, :len(tp.rz_target_cum[band])] = tp.rz_target_cum[band]
        n_carriers[side] = len(tp.carrier_ids)
        for k, pid in enumerate(tp.carrier_ids):
            carrier_idx[side, k] = index[pid]
        if tp.carrier_cum:
            carrier_cum[side, :len(tp.carrier_cum)] = tp.carrier_cum
        if tp.rz_carrier_cum:
            rz_carrier_cum[side, :len(tp.rz_carrier_cum)] = tp.rz_carrier_cum
    return PackedGame(ratings, weather, int(getattr(ctx, "season", None) or 0), int(neutral), int(playoff), n_players, qb, kicker,
                      target_cum, rz_target_cum, carrier_cum, rz_carrier_cum, n_targets, n_carriers, target_idx, carrier_idx, eff, ids)
