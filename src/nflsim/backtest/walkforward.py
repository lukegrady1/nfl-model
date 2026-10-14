"""Phase 7, step 7.1: walk-forward backtest.

For each test season S: everything the simulator touches (lookup tables, situation-only models, full models, team
ratings, player tables, era clip, kickoff seasons, the rating input scale) is rebuilt from seasons before S, in a
subprocess whose environment points the code at artifacts/models/backtest_<S>, artifacts/tables/backtest_<S> and
data/processed/backtest_<S>. The season is then simulated week by week with as-of inputs, in two variants:
'played' (actives from play-by-play participation) and 'thursday' (the week's injury report).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import typer

from nflsim.config import ARTIFACTS_DIR, OUTPUTS_DIR, PROCESSED_DIR, ROOT, settings

BT_DIR = OUTPUTS_DIR / "backtest"
# Phase 3 per-unit margin effects of each rating in the simulation (points per standardized unit, home side)
SIM_POINTS_PER_UNIT = {"pass_epa": 0.80, "rush_epa": 0.71, "proe": 0.23, "adot": 0.39, "sack_rate": -0.32, "cpoe": 0.92,
                       "explosive_rate": 0.44, "pace": -0.14, "pass_epa_allowed": -0.61, "rush_epa_allowed": -0.39,
                       "sack_rate_forced": 0.26, "int_rate_forced": 0.28, "fourth_aggr": 0.17}


def season_dirs(S: int) -> dict[str, Path]:
    return {"models": ARTIFACTS_DIR / "models" / f"backtest_{S}", "tables": ARTIFACTS_DIR / "tables" / f"backtest_{S}",
            "derived": PROCESSED_DIR / f"backtest_{S}", "out": BT_DIR / str(S)}


def fit_input_scale(S: int) -> float:
    """Slope of actual margin on the sim-implied linear margin, fitted on 2019..S-1 only (Phase 3 calibration, walk-forward)."""
    from nflsim.data.clean import load_schedules
    from nflsim.features.team_strength import RATINGS

    r = pl.read_parquet(PROCESSED_DIR / "team_ratings.parquet").select(["team", "season", "week"] + RATINGS)
    s = load_schedules().filter((pl.col("game_type") == "REG") & pl.col("result").is_not_null() & pl.col("season").is_between(2019, S - 1))
    g = s.join(r.rename({"team": "home_team", **{k: f"h_{k}" for k in RATINGS}}), on=["home_team", "season", "week"]).join(
        r.rename({"team": "away_team", **{k: f"a_{k}" for k in RATINGS}}), on=["away_team", "season", "week"])
    lin = sum(SIM_POINTS_PER_UNIT[k] * (g[f"h_{k}"] - g[f"a_{k}"]).to_numpy() for k in RATINGS)
    X = np.column_stack([np.ones(g.height), lin, (g["location"] == "Home").cast(pl.Float64).to_numpy()])
    beta, *_ = np.linalg.lstsq(X, g["result"].to_numpy().astype(float), rcond=None)
    return float(beta[1])


def season_env(S: int) -> dict[str, str]:
    d = season_dirs(S)
    env = dict(os.environ)
    env.update({"NFLSIM_MODELS_DIR": str(d["models"]), "NFLSIM_TABLES_DIR": str(d["tables"]), "NFLSIM_DERIVED_DIR": str(d["derived"]),
                "NFLSIM_TRAINING_SEASONS": f"2018-{S - 1}", "NFLSIM_TABLE_SEASONS": f"{max(2021, S - 4)}-{S - 1}",
                "NFLSIM_KICKOFF_SEASONS": str(S - 1), "NFLSIM_ERA_MAX": str(S - 1), "NFLSIM_INPUT_SCALE": f"{fit_input_scale(S):.4f}"})
    return env


def run_step(S: int, step: str, extra: list[str] | None = None) -> None:
    env = season_env(S)
    cmd = [sys.executable, "-m", "nflsim.backtest.walkforward", step, str(S)] + (extra or [])
    typer.echo(f"[{S}] {step} (input_scale {env['NFLSIM_INPUT_SCALE']}, training {env['NFLSIM_TRAINING_SEASONS']})")
    t0 = time.perf_counter()
    subprocess.run(cmd, env=env, cwd=ROOT, check=True)
    typer.echo(f"[{S}] {step} done in {time.perf_counter() - t0:.0f}s")


# ---- subprocess entry points (run with the season's environment) -----------------------------------------------
def _prepare(S: int) -> None:
    from nflsim.config import MODELS_DIR, TABLES_DIR
    from nflsim.features.players import build_player_tables, build_position_priors
    from nflsim.features.team_strength import build_team_ratings
    from nflsim.models.registry import TRAINABLE
    from nflsim.models.tables import build_all_tables
    from nflsim.models.train import train_all

    main_models = ARTIFACTS_DIR / "models"
    fixed_C = {n: json.load(open(main_models / f"{n}.json"))["C"] for n in TRAINABLE if (main_models / f"{n}.json").exists()}
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    build_all_tables(write=True)
    (MODELS_DIR / "situation").mkdir(parents=True, exist_ok=True)
    train_all(team=False, out_dir=MODELS_DIR / "situation", fixed_C=fixed_C, report=False)
    build_team_ratings()
    usage, _ = build_player_tables()
    build_position_priors(usage)
    train_all(team=True, out_dir=MODELS_DIR, fixed_C=fixed_C, report=False)
    meta = {"season": S, "training_seasons": settings()["training_seasons"], "table_seasons": settings()["table_seasons"],
            "input_scale": settings()["team_strength"]["input_scale"], "fixed_C": fixed_C}
    (MODELS_DIR / "backtest_meta.json").write_text(json.dumps(meta, indent=1))


def _simulate(S: int, sims_played: int, sims_thursday: int) -> None:
    from nflsim.data.clean import load_plays, load_schedules
    from nflsim.engine.state import Context
    from nflsim.features.actives import build_actives, played_actives
    from nflsim.features.players import load_player_eff
    from nflsim.models.registry import default_models
    from nflsim.report.dashboard import fair_spread
    from nflsim.sim.aggregate import game_markets
    from nflsim.sim.fast_engine import FastModels, simulate_games
    from nflsim.sim.game_inputs import WeekInputs
    from nflsim.sim.pack import pack_game, pack_models
    from nflsim.sim.season import PCTS, PLAYER_CAL_STATS, actual_player_stats, game_seed

    out = season_dirs(S)["out"]
    out.mkdir(parents=True, exist_ok=True)
    FM = FastModels(pack_models(default_models()))
    sched = load_schedules().filter((pl.col("season") == S) & (pl.col("game_type") == "REG") & pl.col("result").is_not_null()).sort("week", "gameday")
    plays = load_plays()
    eff = load_player_eff()
    played = played_actives(S, plays, eff)
    actuals = actual_player_stats(S)
    seed = settings()["seed"]
    rows: dict[str, list] = {"played": [], "thursday": []}
    dists: dict[str, dict] = {"played": {}, "thursday": {}}
    prow = []
    t0 = time.perf_counter()
    for week in sorted(sched["week"].unique().to_list()):
        wsched = sched.filter(pl.col("week") == week)
        thu_actives = build_actives(S, week, eff.filter((pl.col("season") == S) & (pl.col("week") == week)), write=False, quiet=True)
        inputs = {"thursday": WeekInputs(S, week, actives=thu_actives), "played": WeekInputs(S, week, actives=played.get(week, {}))}
        for g in wsched.iter_rows(named=True):
            gid, home, away = g["game_id"], g["home_team"], g["away_team"]
            weather = {"roof_dome": 1 if g.get("roof") in ("dome", "closed") else 0, "wind": g.get("wind")}
            for variant, n in (("played", sims_played), ("thursday", sims_thursday)):
                ctx = inputs[variant].context(home, away, weather)
                G = pack_game(ctx, home, away, neutral=g.get("location") == "Neutral")
                h, a, ot, ts, ps = simulate_games(n, game_seed(seed, gid), FM, G)
                margin, total = h - a, h + a
                fair = pl.DataFrame(game_markets(gid, home, away, h, a, ot))
                p_home = float((margin > 0).mean()); p_tie = float((margin == 0).mean())
                rows[variant].append({"game_id": gid, "season": S, "week": week, "home_team": home, "away_team": away, "result": g["result"],
                                      "total": g["total"], "spread_line": g["spread_line"], "total_line": g["total_line"],
                                      "home_moneyline": g["home_moneyline"], "away_moneyline": g["away_moneyline"], "location": g["location"],
                                      "p_home": p_home, "p_tie": p_tie, "p_home_no_push": p_home / (1 - p_tie) if p_tie < 1 else p_home,
                                      "fair_spread": fair_spread(fair), "fair_total": float(np.median(total)), "sim_margin": float(margin.mean()),
                                      "sim_total": float(total.mean()), "sim_margin_sd": float(margin.std()), "sim_total_sd": float(total.std()),
                                      "pit_margin": float((margin < g["result"]).mean() + 0.5 * (margin == g["result"]).mean()),
                                      "pit_total": float((total < g["total"]).mean() + 0.5 * (total == g["total"]).mean()), "n_sims": n})
                dists[variant][gid] = np.stack([margin, total]).astype(np.int16)
                if variant == "played":
                    n_max = G.eff.shape[1]
                    for side, team in ((0, home), (1, away)):
                        tp = ctx.players.get(team)
                        for k, pid in enumerate(G.player_ids[side]):
                            arr = ps[:, side * n_max + k, :]
                            for stat in PLAYER_CAL_STATS:
                                col = arr[:, {"pass_yards": 2, "rush_yards": 6, "rec_yards": 10, "receptions": 9}[stat]].astype(float)
                                if col.max() == 0:
                                    continue
                                actual = actuals.get((gid, team, pid, stat))
                                q = np.percentile(col, [10, 50, 90])
                                prow.append({"game_id": gid, "season": S, "week": week, "team": team, "player_id": pid, "stat": stat,
                                             "name": tp.names.get(pid, pid) if tp else pid, "mean": float(col.mean()), "q10": float(q[0]),
                                             "q50": float(q[1]), "q90": float(q[2]), "actual": actual,
                                             "pct": float((col < actual).mean() + 0.5 * (col == actual).mean()) if actual is not None else None})
        typer.echo(f"[{S}] week {week} done ({time.perf_counter() - t0:.0f}s)")
    for variant in rows:
        pl.DataFrame(rows[variant], infer_schema_length=None).write_parquet(out / f"games_{variant}.parquet")
        np.savez_compressed(out / f"dists_{variant}.npz", **dists[variant])
    pl.DataFrame(prow, infer_schema_length=None).write_parquet(out / "players_played.parquet")
    typer.echo(f"[{S}] {len(rows['played'])} games x 2 variants in {time.perf_counter() - t0:.0f}s -> {out}")


def walk_forward(seasons: list[int], sims_played: int = 5000, sims_thursday: int = 2000, prepare: bool = True, simulate: bool = True) -> None:
    for S in seasons:
        if prepare:
            run_step(S, "prepare")
        if simulate:
            run_step(S, "simulate", [str(sims_played), str(sims_thursday)])


if __name__ == "__main__":
    step, S = sys.argv[1], int(sys.argv[2])
    if step == "prepare":
        _prepare(S)
    elif step == "simulate":
        _simulate(S, int(sys.argv[3]), int(sys.argv[4]))
