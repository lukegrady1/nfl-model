"""Phase 3 acceptance: simulate a past season week by week with as-of team ratings (multiprocess)."""
from __future__ import annotations

import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl
import typer

from nflsim.config import OUTPUTS_DIR, settings
from nflsim.data.clean import load_schedules
from nflsim.engine.game import simulate_game
from nflsim.engine.state import Context
from nflsim.features.team_strength import RATINGS, load_team_ratings

_M = None


def game_seed(seed: int, game_id: str) -> int:
    return (int(seed) + zlib.crc32(game_id.encode())) % (2**32)


def _init():
    global _M
    from nflsim.models.registry import default_models

    _M = default_models()


def _run_game(args: tuple) -> dict:
    game_id, home, away, n, seed, neutral, playoff, season, ratings_home, ratings_away, weather = args
    ctx = Context(home=ratings_home, away=ratings_away, season=season, weather=weather)
    rng = np.random.default_rng(game_seed(seed, game_id))
    margins = np.empty(n)
    totals = np.empty(n)
    for i in range(n):
        r = simulate_game(home, away, _M, rng, neutral=neutral, playoff=playoff, context=ctx)
        margins[i] = r.margin
        totals[i] = r.total
    return {"game_id": game_id, "sim_margin": float(margins.mean()), "sim_total": float(totals.mean()),
            "sim_margin_sd": float(margins.std()), "p_home": float((margins > 0).mean() + 0.5 * (margins == 0).mean())}


def simulate_season(season: int, n: int = 2000, weeks: list[int] | None = None, seed: int | None = None,
                    workers: int | None = None, game_type: str = "REG") -> pl.DataFrame:
    seed = seed if seed is not None else settings()["seed"]
    sched = load_schedules().filter((pl.col("season") == season) & (pl.col("game_type") == game_type) & pl.col("result").is_not_null())
    if weeks:
        sched = sched.filter(pl.col("week").is_in(weeks))
    ratings = load_team_ratings().filter(pl.col("season") == season)
    by_week = {w: {r["team"]: {s: r[s] for s in RATINGS} for r in ratings.filter(pl.col("week") == w).iter_rows(named=True)}
               for w in sched["week"].unique().to_list()}
    jobs = []
    for g in sched.iter_rows(named=True):
        rw = by_week.get(g["week"], {})
        weather = {"roof_dome": 1 if g.get("roof") in ("dome", "closed") else 0, "wind": g.get("wind")}
        jobs.append((g["game_id"], g["home_team"], g["away_team"], n, seed, g.get("location") == "Neutral",
                     game_type != "REG", season, rw.get(g["home_team"], {}), rw.get(g["away_team"], {}), weather))
    t0 = time.perf_counter()
    typer.echo(f"simulating {len(jobs)} games x {n} sims for {season} ({game_type}) ...")
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as ex:
        results = list(ex.map(_run_game, jobs, chunksize=2))
    typer.echo(f"done in {time.perf_counter() - t0:.0f}s")
    out = sched.select("game_id", "season", "week", "home_team", "away_team", "result", "total", "spread_line", "total_line",
                       "location").join(pl.DataFrame(results), on="game_id", how="inner")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUTPUTS_DIR / f"season_sim_{season}_{game_type}_{n}.parquet")
    return out


def phase3_metrics(out: pl.DataFrame) -> tuple[dict, str]:
    home = out.filter(pl.col("location") == "Home")
    m = {
        "games": out.height,
        "corr_sim_actual": float(np.corrcoef(out["sim_margin"], out["result"])[0, 1]),
        "corr_market_actual": float(np.corrcoef(out["spread_line"], out["result"])[0, 1]),
        "corr_sim_market": float(np.corrcoef(out["sim_margin"], out["spread_line"])[0, 1]),
        "sim_home_margin": float(home["sim_margin"].mean()),
        "real_home_margin": float(home["result"].mean()),
        "sim_margin_std": float(out["sim_margin"].std()),
        "spread_std": float(out["spread_line"].std()),
        "mae_sim": float((out["sim_margin"] - out["result"]).abs().mean()),
        "mae_market": float((out["spread_line"] - out["result"]).abs().mean()),
        "corr_sim_total_market": float(np.corrcoef(out["sim_total"], out["total_line"])[0, 1]),
        "mean_sim_total": float(out["sim_total"].mean()), "mean_total_line": float(out["total_line"].mean()),
        "mean_actual_total": float(out["total"].mean()),
    }
    ratio = m["sim_margin_std"] / m["spread_std"]
    checks = [
        ("corr(sim margin, actual margin) >= 0.25", m["corr_sim_actual"], m["corr_sim_actual"] >= 0.25),
        ("corr(sim margin, closing spread) >= 0.7", m["corr_sim_market"], m["corr_sim_market"] >= 0.7),
        ("sim home margin within real range (see text)", m["sim_home_margin"], None),
        ("std of sim margins within +/-25% of spread std", ratio, 0.75 <= ratio <= 1.25),
    ]
    lines = [f"| Check | Value | Pass |", "|---|---|---|"]
    for label, v, ok in checks:
        lines.append(f"| {label} | {v:.3f} | {'yes' if ok else ('see text' if ok is None else 'NO')} |")
    lines += ["", "| Metric | Value |", "|---|---|"]
    for k, v in m.items():
        lines.append(f"| {k} | {v:.3f} |" if isinstance(v, float) else f"| {k} | {v} |")
    return m, "\n".join(lines)
