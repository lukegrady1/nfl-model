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
from nflsim.features.team_strength import RATINGS, load_team_ratings  # noqa: F401

_M = None


def game_seed(seed: int, game_id: str) -> int:
    return (int(seed) + zlib.crc32(game_id.encode())) % (2**32)


def _init():
    global _M
    from nflsim.models.registry import default_models

    _M = default_models()


PLAYER_CAL_STATS = ["pass_yards", "rush_yards", "rec_yards", "receptions"]
PCTS = list(range(5, 100, 5))


def _run_game(args: tuple) -> dict:
    game_id, home, away, n, seed, neutral, playoff, season, week, weather, with_players, actuals = args
    if with_players:
        from nflsim.sim.game_inputs import WeekInputs
        wi = _WEEK.get((season, week))
        if wi is None:
            wi = _WEEK[(season, week)] = WeekInputs(season, week, use_saved_actives=False)
        ctx = wi.context(home, away, weather)
    else:
        from nflsim.features.team_strength import ratings_for_week
        rw = _RATINGS.get((season, week))
        if rw is None:
            rw = _RATINGS[(season, week)] = ratings_for_week(season, week)
        ctx = Context(home=rw.get(home, {}), away=rw.get(away, {}), season=season, weather=weather)
    rng = np.random.default_rng(game_seed(seed, game_id))
    margins = np.empty(n)
    totals = np.empty(n)
    samples: dict = {}
    for i in range(n):
        r = simulate_game(home, away, _M, rng, neutral=neutral, playoff=playoff, context=ctx)
        margins[i] = r.margin
        totals[i] = r.total
        if with_players:
            for team in (home, away):
                for pid, st in r.players.get(team, {}).items():
                    arr = samples.get((team, pid))
                    if arr is None:
                        arr = samples[(team, pid)] = np.zeros((n, len(PLAYER_CAL_STATS)))
                    arr[i] = [st[k] for k in PLAYER_CAL_STATS]
    out = {"game_id": game_id, "sim_margin": float(margins.mean()), "sim_total": float(totals.mean()),
           "sim_margin_sd": float(margins.std()), "p_home": float((margins > 0).mean() + 0.5 * (margins == 0).mean())}
    if with_players:
        rows = []
        for (team, pid), arr in samples.items():
            tp = ctx.players.get(team)
            for j, stat in enumerate(PLAYER_CAL_STATS):
                col = arr[:, j]
                row = {"game_id": game_id, "team": team, "player_id": pid, "name": tp.names.get(pid, pid) if tp else pid,
                       "position": tp.players[pid]["position"] if tp and pid in tp.players else None, "stat": stat,
                       "mean": float(col.mean()), "p_any": float((col > 0).mean())}
                qs = np.percentile(col, PCTS)
                row.update({f"q{p}": float(q) for p, q in zip(PCTS, qs)})
                actual = actuals.get((team, pid, stat))
                row["actual"] = actual
                row["pct"] = float((col < actual).mean() + 0.5 * (col == actual).mean()) if actual is not None else None
                rows.append(row)
        out["players"] = rows
        out["blend_log"] = list(getattr(_WEEK.get((season, week)), "blend_log", []))
    return out


_WEEK: dict = {}
_RATINGS: dict = {}


def actual_player_stats(season: int) -> dict:
    """{(team, player_id, stat): value} from the play-by-play, for the calibration check."""
    from nflsim.data.clean import load_plays

    p = load_plays().filter((pl.col("season") == season) & pl.col("play_class").is_in(["pass", "run"]))
    att = pl.col("pass_outcome").is_in(["complete", "incomplete", "interception"])
    comp = pl.col("pass_outcome") == "complete"
    out: dict = {}
    qb = p.filter(comp & pl.col("passer_player_id").is_not_null()).group_by("game_id", "posteam", "passer_player_id").agg(pl.col("yards_gained").sum().alias("v"))  # passing yards: completions only
    for g, t, pid, v in qb.iter_rows():
        out[(g, t, pid, "pass_yards")] = float(v)
    rec = p.filter(att & pl.col("receiver_player_id").is_not_null()).group_by("game_id", "posteam", "receiver_player_id").agg(
        pl.col("yards_gained").filter(comp).sum().alias("y"), comp.sum().alias("r"))
    for g, t, pid, y, r in rec.iter_rows():
        out[(g, t, pid, "rec_yards")] = float(y)
        out[(g, t, pid, "receptions")] = float(r)
    ru = p.filter((pl.col("play_class") == "run") | (pl.col("pass_outcome") == "scramble")).filter(pl.col("rusher_player_id").is_not_null()).group_by(
        "game_id", "posteam", "rusher_player_id").agg(pl.col("yards_gained").sum().alias("v"))
    for g, t, pid, v in ru.iter_rows():
        out[(g, t, pid, "rush_yards")] = float(v)
    return out


def simulate_season(season: int, n: int = 2000, weeks: list[int] | None = None, seed: int | None = None,
                    workers: int | None = None, game_type: str = "REG", with_players: bool = False) -> pl.DataFrame:
    seed = seed if seed is not None else settings()["seed"]
    sched = load_schedules().filter((pl.col("season") == season) & (pl.col("game_type") == game_type) & pl.col("result").is_not_null())
    if weeks:
        sched = sched.filter(pl.col("week").is_in(weeks))
    actual = actual_player_stats(season) if with_players else {}
    jobs = []
    for g in sched.iter_rows(named=True):
        weather = {"roof_dome": 1 if g.get("roof") in ("dome", "closed") else 0, "wind": g.get("wind")}
        acts = {(t, pid, st): v for (gid, t, pid, st), v in actual.items() if gid == g["game_id"]} if with_players else {}
        jobs.append((g["game_id"], g["home_team"], g["away_team"], n, seed, g.get("location") == "Neutral",
                     game_type != "REG", season, g["week"], weather, with_players, acts))
    jobs.sort(key=lambda j: j[8])   # group by week so each worker reuses the week's inputs
    t0 = time.perf_counter()
    typer.echo(f"simulating {len(jobs)} games x {n} sims for {season} ({game_type}){' with players' if with_players else ''} ...")
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as ex:
        results = list(ex.map(_run_game, jobs, chunksize=1))
    typer.echo(f"done in {time.perf_counter() - t0:.0f}s")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{season}_{game_type}_{n}" + ("_players" if with_players else "")
    if with_players:
        prow = [r for res in results for r in res.pop("players", [])]
        pl.DataFrame(prow, infer_schema_length=None).write_parquet(OUTPUTS_DIR / f"player_sim_{tag}.parquet")
        logs = sorted({line for res in results for line in res.pop("blend_log", [])})
        for line in logs:
            typer.echo(f"  qb blend: {line}")
    out = sched.select("game_id", "season", "week", "home_team", "away_team", "result", "total", "spread_line", "total_line",
                       "location").join(pl.DataFrame(results), on="game_id", how="inner")
    out.write_parquet(OUTPUTS_DIR / f"season_sim_{tag}.parquet")
    return out


def phase4_metrics(players: pl.DataFrame, min_games: int = 8) -> tuple[dict, str]:
    """Calibration of player stat distributions: percentile of the actual stat, pooled over players with enough games."""
    d = players.filter(pl.col("actual").is_not_null())
    counts = d.group_by("player_id", "stat").len()
    d = d.join(counts.filter(pl.col("len") >= min_games).select("player_id", "stat"), on=["player_id", "stat"], how="inner")
    lines = ["| Stat | player-games | below median | inside 10-90 band | mean actual | mean sim |", "|---|---|---|---|---|---|"]
    m = {}
    for stat in PLAYER_CAL_STATS + ["all"]:
        s = d if stat == "all" else d.filter(pl.col("stat") == stat)
        if s.height == 0:
            continue
        below = float((s["actual"] < s["q50"]).mean() + 0.5 * (s["actual"] == s["q50"]).mean())
        inside = float(((s["actual"] >= s["q10"]) & (s["actual"] <= s["q90"])).mean())
        m[stat] = {"n": s.height, "below_median": below, "inside_10_90": inside, "mean_actual": float(s["actual"].mean()), "mean_sim": float(s["mean"].mean())}
        lines.append(f"| {stat} | {s.height} | {below:.1%} | {inside:.1%} | {s['actual'].mean():.2f} | {s['mean'].mean():.2f} |")
    ok = 0.40 <= m["all"]["below_median"] <= 0.60 and 0.70 <= m["all"]["inside_10_90"] <= 0.90
    lines.append("")
    lines.append(f"Pooled: {m['all']['below_median']:.1%} below the median (target 40-60%), {m['all']['inside_10_90']:.1%} inside the 10th-90th band (target 70-90%): {'PASS' if ok else 'FAIL'}")
    return m, "\n".join(lines)


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
