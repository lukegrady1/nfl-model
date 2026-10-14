"""Phase 6, step 6.3: implied probabilities and vig removal."""
from __future__ import annotations

import math

import polars as pl

from nflsim.config import settings


def implied(american: int | float) -> float:
    o = float(american)
    return -o / (-o + 100.0) if o < 0 else 100.0 / (o + 100.0)


def decimal(american: int | float) -> float:
    o = float(american)
    return 1 + o / 100.0 if o > 0 else 1 + 100.0 / -o


def multiplicative(q_a: float, q_b: float) -> tuple[float, float, float]:
    """Fair probabilities for a two-sided market and the hold."""
    tot = q_a + q_b
    return q_a / tot, q_b / tot, tot - 1.0


def power(q_a: float, q_b: float, tol: float = 1e-12) -> tuple[float, float, float]:
    """Find k with q_a^k + q_b^k = 1 (bisection); fair = q^k."""
    lo, hi = 0.5, 4.0
    f = lambda k: q_a ** k + q_b ** k - 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    k = 0.5 * (lo + hi)
    return q_a ** k, q_b ** k, k


def one_sided(q: float, assumed_hold: float | None = None) -> float:
    h = settings()["assumed_hold_one_sided"] if assumed_hold is None else assumed_hold
    return q / (1.0 + h)


def devig_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Adds q (implied), p_market (multiplicative), p_market_power, hold, one_sided per selection.
    Two-sided = the other side of the same FanDuel market at the same line (over/under, home/away)."""
    df = df.with_columns(q=pl.col("american_odds").map_elements(implied, return_dtype=pl.Float64))
    other = {"over": "under", "under": "over", "home": "away", "away": "home"}
    key = ["fd_market_id", "line"]
    pairs: dict = {}
    for r in df.select(key + ["selection", "q"]).iter_rows(named=True):
        pairs[(r["fd_market_id"], r["line"], r["selection"])] = r["q"]
    p_mult, p_pow, hold, one = [], [], [], []
    for r in df.iter_rows(named=True):
        q = r["q"]
        o = other.get(r["selection"])
        # a spread pair has opposite lines (home -3.5 vs away +3.5)
        partner_line = -r["line"] if (r["market"] == "spread" and r["line"] is not None) else r["line"]
        q2 = pairs.get((r["fd_market_id"], partner_line, o)) if o else None
        if q2 is not None and not r["one_sided"]:
            a, b, h = multiplicative(q, q2)
            pa, pb, k = power(q, q2)
            p_mult.append(a); p_pow.append(pa); hold.append(h); one.append(False)
        else:
            p_mult.append(one_sided(q)); p_pow.append(one_sided(q)); hold.append(None); one.append(True)
    return df.with_columns(pl.Series("p_market", p_mult), pl.Series("p_market_power", p_pow), pl.Series("hold", hold, dtype=pl.Float64),
                           pl.Series("one_sided", one))
