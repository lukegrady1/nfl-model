"""Bucketing functions shared by table building (from data rows) and simulation (from GameState).

Every function is a pure scalar function so the two paths cannot drift. `vectorize` wraps them for building.
"""
from __future__ import annotations

import numpy as np

DIST_LABELS = ["1", "2-3", "4-6", "7-10", "11+"]
ZONE_LABELS = ["own", "opp21-50", "red11-20", "inside10"]
SCORE_LABELS = ["<=-9", "-8..-4", "-3..-1", "0", "1..3", "4..8", ">=9"]
TIME_LABELS = ["H1 >2:00", "H1 <=2:00", "Q3/Q4 >8:00", "Q4 8:00-2:00", "Q4 <=2:00", "OT"]
ST_LABELS = ["H1 >2:00", "H1 <=2:00", "Q3", "Q4 lead", "Q4 tied", "Q4 trail 1-8", "Q4 trail 9+", "OT"]
AIR_BIN_EDGES = [-1, 4, 9, 14, 19, 29]           # bin i = value <= edge[i]; last bin 30+
AIR_LABELS = ["<=-1", "0-4", "5-9", "10-14", "15-19", "20-29", "30+"]
YAC_BIN_EDGES = [0, 1, 2, 4, 6, 9, 14, 24]       # last bin 25+
YAC_LABELS = ["0", "1", "2", "3-4", "5-6", "7-9", "10-14", "15-24", "25+"]
SECONDS_BIN_EDGES = [5, 10, 15, 20, 25, 30, 35, 40, 45]  # last bin 46+
SECONDS_LABELS = ["0-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46+"]
RUSH_CLASSES = [-6] + list(range(-5, 16)) + [16, 25, 40]  # -6 = "<=-6", 16 = "16-24", 25 = "25-39", 40 = "40+"


def dist_bucket(ydstogo: int) -> int:
    if ydstogo <= 1:
        return 0
    if ydstogo <= 3:
        return 1
    if ydstogo <= 6:
        return 2
    if ydstogo <= 10:
        return 3
    return 4


def zone(yardline_100: int) -> int:
    if yardline_100 >= 51:
        return 0
    if yardline_100 >= 21:
        return 1
    if yardline_100 >= 11:
        return 2
    return 3


def yardline5(yardline_100: int) -> int:
    """Every 5 yards: 1-5 -> 0, 6-10 -> 1, ..., 96-99 -> 19."""
    return min(max((int(yardline_100) - 1) // 5, 0), 19)


def score_bucket(diff: int) -> int:
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


def lead_bucket(diff: int) -> int:
    """0 trailing, 1 tied, 2 leading."""
    return 0 if diff < 0 else (1 if diff == 0 else 2)


def time_bucket(qtr: int, half_seconds_remaining: int, game_seconds_remaining: int) -> int:
    if qtr >= 5:
        return 5
    if qtr <= 2:
        return 1 if half_seconds_remaining <= 120 else 0
    if game_seconds_remaining > 480:
        return 2
    if game_seconds_remaining > 120:
        return 3
    return 4


def score_time_bucket(qtr: int, half_seconds_remaining: int, diff: int) -> int:
    """4th-down situation bucket (D2)."""
    if qtr >= 5:
        return 7
    if qtr <= 2:
        return 1 if half_seconds_remaining <= 120 else 0
    if qtr == 3:
        return 2
    if diff > 0:
        return 3
    if diff == 0:
        return 4
    if diff >= -8:
        return 5
    return 6


def under_two(half_seconds_remaining: int) -> int:
    return 1 if half_seconds_remaining <= 120 else 0


def under_two_half(half_seconds_remaining: int, qtr: int) -> int:
    """0 = not under 2:00; 1 = under 2:00 of the first half; 2 = under 2:00 of the second half or OT."""
    if half_seconds_remaining > 120:
        return 0
    return 1 if qtr <= 2 else 2


def _edge_bin(value: int, edges: list[int]) -> int:
    for i, e in enumerate(edges):
        if value <= e:
            return i
    return len(edges)


def air_bin(air_yards: int) -> int:
    return _edge_bin(int(air_yards), AIR_BIN_EDGES)


def yac_bin(yac: int) -> int:
    return _edge_bin(int(yac), YAC_BIN_EDGES)


def seconds_bin(seconds: int) -> int:
    return _edge_bin(int(seconds), SECONDS_BIN_EDGES)


def rush_class(yards: int) -> int:
    y = int(yards)
    if y <= -6:
        return -6
    if y <= 15:
        return y
    if y <= 24:
        return 16
    if y <= 39:
        return 25
    return 40


def pat_score_key(diff_after_td: int) -> int:
    return int(min(max(diff_after_td, -16), 16))


def vectorize(fn):
    return np.vectorize(fn, otypes=[np.int64])
