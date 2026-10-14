"""Phase 7, step 7.2: backtest metrics against the closing line."""
from __future__ import annotations

import math

import numpy as np
import polars as pl
from scipy.stats import kstest

from nflsim.odds.devig import implied, multiplicative


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def win_probability_metrics(g: pl.DataFrame) -> dict:
    """Model P(home win | no tie) vs the no-vig closing moneyline, ties excluded."""
    d = g.filter((pl.col("result") != 0) & pl.col("home_moneyline").is_not_null())
    y = (d["result"] > 0).to_numpy().astype(float)
    p_model = d["p_home_no_push"].to_numpy()
    qh = np.array([implied(o) for o in d["home_moneyline"]])
    qa = np.array([implied(o) for o in d["away_moneyline"]])
    p_market = qh / (qh + qa)
    return {"n": int(len(y)), "log_loss_model": _log_loss(y, p_model), "log_loss_market": _log_loss(y, p_market),
            "brier_model": float(np.mean((p_model - y) ** 2)), "brier_market": float(np.mean((p_market - y) ** 2)),
            "home_win_rate": float(y.mean())}


def error_metrics(g: pl.DataFrame) -> dict:
    d = g.filter(pl.col("spread_line").is_not_null() & pl.col("total_line").is_not_null())
    res, tot = d["result"].to_numpy().astype(float), d["total"].to_numpy().astype(float)
    return {"n": d.height,
            "margin_mae_model": float(np.abs(d["sim_margin"].to_numpy() - res).mean()), "margin_mae_market": float(np.abs(d["spread_line"].to_numpy() - res).mean()),
            "total_mae_model": float(np.abs(d["sim_total"].to_numpy() - tot).mean()), "total_mae_market": float(np.abs(d["total_line"].to_numpy() - tot).mean()),
            "margin_bias_model": float((d["sim_margin"].to_numpy() - res).mean()), "total_bias_model": float((d["sim_total"].to_numpy() - tot).mean()),
            "total_bias_market": float((d["total_line"].to_numpy() - tot).mean()),
            "corr_margin_model": float(np.corrcoef(d["sim_margin"], res)[0, 1]), "corr_margin_market": float(np.corrcoef(d["spread_line"], res)[0, 1]),
            "corr_model_market_margin": float(np.corrcoef(d["sim_margin"], d["spread_line"])[0, 1]),
            "corr_model_market_total": float(np.corrcoef(d["sim_total"], d["total_line"])[0, 1])}


def reliability(g: pl.DataFrame, bins: int = 10) -> list[dict]:
    d = g.filter(pl.col("result") != 0)
    p = d["p_home_no_push"].to_numpy(); y = (d["result"] > 0).to_numpy().astype(float)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] + (1e-9 if i == bins - 1 else 0))
        out.append({"bucket": f"{edges[i]:.1f}-{edges[i + 1]:.1f}", "n": int(m.sum()), "pred": float(p[m].mean()) if m.any() else None,
                    "obs": float(y[m].mean()) if m.any() else None})
    return out


def disagreement_buckets(g: pl.DataFrame, kind: str, thresholds=(1.5, 2.5, 3.5)) -> list[dict]:
    """ATS (or over/under) record when the model's number differs from the close by at least t points.
    Spread: sim_margin (expected home margin) vs spread_line (home expected margin). Totals: sim_total vs total_line."""
    rows = []
    for t in thresholds:
        if kind == "spread":
            d = g.filter(pl.col("spread_line").is_not_null())
            diff = d["sim_margin"].to_numpy() - d["spread_line"].to_numpy()
            pick_home = diff >= t; pick_away = diff <= -t
            cover = d["result"].to_numpy() - d["spread_line"].to_numpy()      # > 0 home covers, 0 push
            picked = pick_home | pick_away
            won = (pick_home & (cover > 0)) | (pick_away & (cover < 0))
            push = picked & (cover == 0)
        else:
            d = g.filter(pl.col("total_line").is_not_null())
            diff = d["sim_total"].to_numpy() - d["total_line"].to_numpy()
            pick_over = diff >= t; pick_under = diff <= -t
            res = d["total"].to_numpy() - d["total_line"].to_numpy()
            picked = pick_over | pick_under
            won = (pick_over & (res > 0)) | (pick_under & (res < 0))
            push = picked & (res == 0)
        n_bets = int(picked.sum()); n_push = int(push.sum()); n_dec = n_bets - n_push; k = int(won.sum())
        lo, hi = wilson(k, n_dec)
        rows.append({"threshold": t, "bets": n_bets, "pushes": n_push, "wins": k, "losses": n_dec - k,
                     "win_rate": k / n_dec if n_dec else None, "ci_low": lo, "ci_high": hi, "breakeven": 0.524})
    return rows


def pit_summary(values: np.ndarray, bins: int = 10) -> dict:
    v = values[~np.isnan(values)]
    hist, _ = np.histogram(v, bins=np.linspace(0, 1, bins + 1))
    ks = kstest(v, "uniform")
    return {"n": int(len(v)), "hist": hist.tolist(), "ks_p": float(ks.pvalue), "max_dev": float(np.abs(hist / len(v) - 1 / bins).max()),
            "mean": float(v.mean())}


def prop_calibration(players: pl.DataFrame, min_games: int = 8) -> dict:
    d = players.filter(pl.col("actual").is_not_null())
    counts = d.group_by("player_id", "stat").len()
    d = d.join(counts.filter(pl.col("len") >= min_games).select("player_id", "stat"), on=["player_id", "stat"], how="inner")
    out = {}
    for stat in ["pass_yards", "rush_yards", "rec_yards", "receptions", "all"]:
        s = d if stat == "all" else d.filter(pl.col("stat") == stat)
        if s.height == 0:
            continue
        out[stat] = {"n": s.height, "below_median": float((s["actual"] < s["q50"]).mean() + 0.5 * (s["actual"] == s["q50"]).mean()),
                     "inside_10_90": float(((s["actual"] >= s["q10"]) & (s["actual"] <= s["q90"])).mean()),
                     "mean_actual": float(s["actual"].mean()), "mean_sim": float(s["mean"].mean()), "pit": pit_summary(s["pct"].to_numpy())}
    return out


def season_metrics(g: pl.DataFrame, players: pl.DataFrame | None) -> dict:
    return {"win_prob": win_probability_metrics(g), "errors": error_metrics(g), "reliability": reliability(g),
            "ats": disagreement_buckets(g, "spread"), "ou": disagreement_buckets(g, "total"),
            "pit_margin": pit_summary(g["pit_margin"].to_numpy()), "pit_total": pit_summary(g["pit_total"].to_numpy()),
            "props": prop_calibration(players) if players is not None and players.height else {}}
