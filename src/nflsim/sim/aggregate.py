"""Phase 5, steps 5.3-5.4: fair probabilities and fair American odds from simulation arrays."""
from __future__ import annotations

import numpy as np
import polars as pl

from nflsim.config import settings
from nflsim.engine.boxscore import PLAYER_STATS

PROP_STATS = ["pass_yards", "pass_tds", "interceptions", "rush_yards", "rush_attempts", "rec_yards", "receptions", "longest_rec", "longest_rush"]


def american(p: float) -> int:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))


def se(p: float, n: int) -> float:
    return float(np.sqrt(max(p * (1 - p), 0.0) / n))


def _lines(cfg: dict) -> np.ndarray:
    return np.round(np.arange(cfg["min"], cfg["max"] + 1e-9, cfg["step"]), 1)


def game_markets(game_id: str, home: str, away: str, home_score: np.ndarray, away_score: np.ndarray, went_ot: np.ndarray) -> list[dict]:
    s = settings()
    n = len(home_score)
    margin = home_score - away_score
    total = home_score + away_score
    rows = []

    def add(market, selection, team, line, p, push=0.0):
        p_adj = p / (1 - push) if push < 1 else p
        rows.append({"game_id": game_id, "market": market, "selection": selection, "team": team, "line": line,
                     "p": float(p), "p_push": float(push), "p_no_push": float(p_adj), "se": se(p_adj, n), "fair_american": american(p_adj), "n": n})

    p_home, p_away, p_tie = float((margin > 0).mean()), float((margin < 0).mean()), float((margin == 0).mean())
    add("moneyline", "home", home, None, p_home, p_tie)
    add("moneyline", "away", away, None, p_away, p_tie)
    rows.append({"game_id": game_id, "market": "moneyline", "selection": "tie", "team": None, "line": None, "p": p_tie, "p_push": 0.0,
                 "p_no_push": p_tie, "se": se(p_tie, n), "fair_american": american(p_tie), "n": n})
    for L in _lines(s["spread_lines"]):            # home line L: home covers when margin + L > 0
        push = float((margin + L == 0).mean())
        add("spread", "home", home, float(L), float((margin + L > 0).mean()), push)
        add("spread", "away", away, float(-L), float((margin + L < 0).mean()), push)
    for T in _lines(s["total_lines"]):
        push = float((total == T).mean())
        add("total", "over", None, float(T), float((total > T).mean()), push)
        add("total", "under", None, float(T), float((total < T).mean()), push)
    for team, sc in ((home, home_score), (away, away_score)):
        for T in _lines(s.get("team_total_lines", {"min": 10.5, "max": 44.5, "step": 0.5})):
            push = float((sc == T).mean())
            add("team_total", "over", team, float(T), float((sc > T).mean()), push)
            add("team_total", "under", team, float(T), float((sc < T).mean()), push)
    rows.append({"game_id": game_id, "market": "overtime", "selection": "yes", "team": None, "line": None, "p": float(went_ot.mean()),
                 "p_push": 0.0, "p_no_push": float(went_ot.mean()), "se": se(float(went_ot.mean()), n), "fair_american": american(float(went_ot.mean())), "n": n})
    return rows


def player_markets(game_id: str, team: str, player_id: str, name: str, position: str, stats: np.ndarray) -> list[dict]:
    """stats: [n, 16] in PLAYER_STATS order. Percentiles 5-95 and P(over) at every 0.5 line inside that range."""
    n = stats.shape[0]
    rows = []
    for stat in PROP_STATS:
        col = stats[:, PLAYER_STATS.index(stat)].astype(float)
        if col.max() == 0:
            continue
        q = np.percentile(col, list(range(5, 100, 5)))
        base = {"game_id": game_id, "team": team, "player_id": player_id, "name": name, "position": position, "stat": stat,
                "mean": float(col.mean()), "p_any": float((col > 0).mean()), "n": n, **{f"q{p}": float(v) for p, v in zip(range(5, 100, 5), q)}}
        lo, hi = np.floor(q[0]) + 0.5, np.ceil(q[-1]) - 0.5    # half-point lines only (how props are posted)
        lines = []
        for X in np.arange(lo, hi + 1e-9, 1.0):
            p_over = float((col > X).mean())
            lines.append({"line": float(X), "p_over": p_over, "se": se(p_over, n), "fair_american_over": american(p_over), "fair_american_under": american(1 - p_over)})
        base["lines"] = lines
        rows.append(base)
    td = stats[:, PLAYER_STATS.index("anytime_td")].astype(float)
    if td.max() > 0:
        p = float(td.mean())
        rows.append({"game_id": game_id, "team": team, "player_id": player_id, "name": name, "position": position, "stat": "anytime_td",
                     "mean": p, "p_any": p, "n": n, "lines": [{"line": 0.5, "p_over": p, "se": se(p, n), "fair_american_over": american(p), "fair_american_under": american(1 - p)}]})
    return rows


def player_markets_frame(rows: list[dict]) -> pl.DataFrame:
    """Long form: one row per (player, stat, line)."""
    out = []
    for r in rows:
        base = {k: v for k, v in r.items() if k != "lines"}
        for ln in r["lines"]:
            out.append({**base, **ln})
    return pl.DataFrame(out, infer_schema_length=None) if out else pl.DataFrame()
