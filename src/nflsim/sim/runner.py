"""Run many games with the reference engine and summarize against the Phase 0 targets."""
from __future__ import annotations

import time

import numpy as np
import polars as pl
import typer

from nflsim.config import OUTPUTS_DIR
from nflsim.engine.game import simulate_game
from nflsim.engine.boxscore import GameResult

ACCEPTANCE = [  # (metric, allowed gap) from the Phase 1 acceptance table
    ("points_per_team_game", 1.5), ("plays_per_team_game", 3.0), ("pass_call_rate", 0.02),
    ("completion_rate", 0.02), ("sack_rate_per_dropback", 0.01), ("int_rate_per_attempt", 0.005),
    ("punts_per_team_game", 0.5), ("fg_att_per_team_game", 0.4), ("margin_std", 1.5),
]


def simulate_many(home: str, away: str, n: int, seed: int, playoff: bool = False, neutral: bool = False,
                  M: dict | None = None) -> list[GameResult]:
    if M is None:
        from nflsim.models.registry import default_models
        M = default_models()
    rng = np.random.default_rng(seed)
    return [simulate_game(home, away, M, rng, neutral=neutral, playoff=playoff) for _ in range(n)]


def summarize(results: list[GameResult]) -> dict:
    n = len(results)
    teams = [results[0].home, results[0].away]
    agg = {k: 0 for k in results[0].team_stats[teams[0]]}
    for r in results:
        for t in teams:
            for k, v in r.team_stats[t].items():
                agg[k] += v
    tg = 2 * n
    dropbacks = agg["pass_attempts"] + agg["sacks"] + agg["scrambles"]
    designed_runs = agg["rush_attempts"] - agg["scrambles"]
    margins = np.array([r.margin for r in results], dtype=float)
    totals = np.array([r.total for r in results], dtype=float)
    return {
        "games": n,
        "points_per_team_game": float(totals.mean() / 2),
        "plays_per_team_game": agg["plays"] / tg,
        "pass_call_rate": dropbacks / (dropbacks + designed_runs),
        "completion_rate": agg["completions"] / agg["pass_attempts"],
        "sack_rate_per_dropback": agg["sacks"] / dropbacks,
        "scramble_rate_per_dropback": agg["scrambles"] / dropbacks,
        "int_rate_per_attempt": agg["interceptions"] / agg["pass_attempts"],
        "punts_per_team_game": agg["punts"] / tg,
        "fg_att_per_team_game": agg["fg_attempts"] / tg,
        "fg_make_rate": agg["fg_made"] / max(agg["fg_attempts"], 1),
        "margin_std": float(margins.std(ddof=1)),
        "home_margin_mean": float(margins.mean()),
        "total_mean": float(totals.mean()),
        "total_std": float(totals.std(ddof=1)),
        "ot_share": float(np.mean([r.went_ot for r in results])),
        "tie_share": float(np.mean([r.tie for r in results])),
        "drives_per_team_game": agg["drives"] / tg,
        "fumbles_lost_per_team_game": agg["fumbles_lost"] / tg,
        "penalties_per_team_game": agg["penalties"] / tg,
        "yards_per_dropback": (agg["pass_yards"] - agg["sack_yards"]) / dropbacks,
        "yards_per_rush": (agg["rush_yards"]) / max(agg["rush_attempts"], 1),
        "kneels_per_team_game": agg["kneels"] / tg,
        "two_point_share": agg["two_point_attempts"] / max(agg["two_point_attempts"] + agg["xp_attempts"], 1),
        "home_win_rate": float(np.mean(margins > 0)),
    }


def targets_2021_2025() -> dict:
    from nflsim.data.clean import load_schedules
    from nflsim.data.validate import load_summary

    row = load_summary().filter(pl.col("season") == 0).to_dicts()[0]
    # Equal-teams benchmark: real margins net of the closing spread remove the between-team quality spread
    # that an AVG-vs-AVG simulation cannot contain before Phase 3.
    s = load_schedules().filter(pl.col("season").is_between(2021, 2025) & (pl.col("game_type") == "REG")
                                & pl.col("home_score").is_not_null() & pl.col("spread_line").is_not_null())
    margin = s["home_score"] - s["away_score"]
    row["margin_std_net_of_spread"] = float((margin - s["spread_line"]).std())
    row["spread_line_std"] = float(s["spread_line"].std())
    return row


def acceptance_table(sim: dict, real: dict) -> tuple[str, bool]:
    lines = ["| Metric | Simulated | Real 2021-25 | Gap | Allowed | Pass |", "|---|---|---|---|---|---|"]
    all_ok = True
    for metric, tol in ACCEPTANCE:
        s, r = sim[metric], real[metric]
        gap = s - r
        ok = abs(gap) <= tol
        all_ok &= ok
        lines.append(f"| {metric} | {s:.3f} | {r:.3f} | {gap:+.3f} | ±{tol} | {'yes' if ok else 'NO'} |")
    net = real["margin_std_net_of_spread"]
    net_gap = sim["margin_std"] - net
    net_ok = abs(net_gap) <= 1.5
    lines.append(f"| margin_std vs real net of closing spread | {sim['margin_std']:.3f} | {net:.3f} | {net_gap:+.3f} | ±1.5 | {'yes' if net_ok else 'NO'} |")
    ot_ok = 0.03 <= sim["ot_share"] <= 0.09
    all_ok &= ot_ok
    lines.append(f"| ot_share | {sim['ot_share']:.3f} | {real['ot_share']:.3f} | | 3%-9% | {'yes' if ot_ok else 'NO'} |")
    extra = ["drives_per_team_game", "fg_make_rate", "scramble_rate_per_dropback", "fumbles_lost_per_team_game",
             "two_point_share", "home_margin_mean", "yards_per_dropback", "yards_per_rush", "tie_share"]
    lines.append("| *(not graded)* | | | | | |")
    for m in extra:
        r = real.get(m)
        lines.append(f"| {m} | {sim[m]:.3f} | {'' if r is None else f'{r:.3f}'} | | | |")
    return "\n".join(lines), all_ok


def sim_matchup_summary(home: str, away: str, n: int, seed: int, playoff: bool = False, neutral: bool = False,
                        tables_only: bool = False, M: dict | None = None, label: str = "") -> dict:
    if home == away:
        home, away = f"{home}_H", f"{away}_A"
    if M is None:
        from nflsim.models.registry import build_models, default_models
        M = build_models(set()) if tables_only else default_models()
    trained = sorted(k for k, m in M.items() if type(m).__name__ == "TrainedModel")
    typer.echo(f"models: {len(trained)} trained ({', '.join(trained) or 'none'}), rest tables {label}")
    t0 = time.perf_counter()
    results = simulate_many(home, away, n, seed, playoff=playoff, neutral=neutral, M=M)
    secs = time.perf_counter() - t0
    sim = summarize(results)
    typer.echo(f"{away} @ {home}: {n} games in {secs:.1f}s ({1000 * secs / n:.1f} ms/game), seed {seed}")
    typer.echo(f"  {home} {sim['home_win_rate']:.1%} win, mean margin {sim['home_margin_mean']:+.2f}, "
               f"mean total {sim['total_mean']:.1f} (sd {sim['total_std']:.1f}), OT {sim['ot_share']:.1%}, ties {sim['tie_share']:.2%}")
    real = targets_2021_2025()
    table, ok = acceptance_table(sim, real)
    typer.echo(table)
    typer.echo("Phase 1 acceptance (literal table): " + ("PASS" if ok else "FAIL")
               + "  (margin_std graded against the raw real std, which includes team-quality spread; see the net-of-spread row)")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS_DIR / f"sim_check_{away}_at_{home}_{n}.md"
    out.write_text(f"# {away} @ {home}, {n} sims, seed {seed}\n\n{table}\n\nruntime {secs:.1f}s\n")
    return {"sim": sim, "real": real, "pass": ok, "seconds": secs, "table": table}


def swap_check(n: int = 2000, seed: int | None = None, names: list[str] | None = None) -> str:
    """Phase 2: swap trained models in one at a time and re-run the Phase 1 league-average check after each."""
    from nflsim.config import settings as _settings
    from nflsim.models.registry import TRAINABLE, build_models
    from nflsim.models.trained import available_trained

    seed = seed if seed is not None else _settings()["seed"]
    names = names or [m for m in TRAINABLE if m in available_trained()]
    real = targets_2021_2025()
    metrics = [m for m, _ in ACCEPTANCE] + ["ot_share"]
    header = "| Swapped in | " + " | ".join(metrics) + " | Pass |\n|" + "---|" * (len(metrics) + 2)
    lines = [header]
    active: set[str] = set()
    steps = [("tables only", None)] + [(name, name) for name in names]
    for label, name in steps:
        if name:
            active.add(name)
        M = build_models(set(active))
        t0 = time.perf_counter()
        results = simulate_many("AVG_H", "AVG_A", n, seed, M=M)
        sim = summarize(results)
        _, ok = acceptance_table(sim, real)
        cells = []
        for m, tol in ACCEPTANCE:
            gap = sim[m] - real[m]
            cells.append(f"{sim[m]:.3f} ({gap:+.2f}){'' if abs(gap) <= tol else ' !'}")
        cells.append(f"{sim['ot_share']:.3f}")
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {'yes' if ok else 'NO'} |")
        typer.echo(f"{label:18s} {time.perf_counter() - t0:5.1f}s  " + ("PASS" if ok else "FAIL") +
                   "  pts %.2f plays %.2f pass %.3f cmp %.3f sack %.3f int %.3f punts %.2f fga %.2f mstd %.2f ot %.3f" % tuple(sim[m] for m in metrics))
    table = "\n".join(lines)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / f"phase2_swap_check_{n}.md").write_text(table + "\n")
    return table


# ---- Phase 5, step 5.2: simulate a week with the fast engine and write the fair-odds tables -----------------------
def simulate_week(season: int, week: int, sims: int | None = None, seed: int | None = None, include_played: bool = False,
                  out_dir=None) -> "pl.DataFrame":
    import json

    from nflsim.config import OUTPUTS_DIR, settings as _settings
    from nflsim.data.clean import load_schedules
    from nflsim.features.weather import load_weather
    from nflsim.models.registry import default_models
    from nflsim.sim.aggregate import game_markets, player_markets, player_markets_frame
    from nflsim.sim.fast_engine import FastModels, simulate_games
    from nflsim.sim.game_inputs import WeekInputs
    from nflsim.sim.pack import pack_game, pack_models
    from nflsim.sim.season import game_seed

    s = _settings()
    sims = sims or s["sims_per_game"]
    seed = seed if seed is not None else s["seed"]
    out_dir = out_dir or (OUTPUTS_DIR / f"{season}_wk{week}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sched = load_schedules().filter((pl.col("season") == season) & (pl.col("week") == week) & (pl.col("game_type") != "PRE"))
    if not include_played:
        sched = sched.filter(pl.col("result").is_null())
    if sched.height == 0:
        typer.echo(f"no games to simulate for {season} week {week}" + ("" if include_played else " (all played; use --include-played)"))
        return pl.DataFrame()
    FM = FastModels(pack_models(default_models()))
    wi = WeekInputs(season, week)
    weather = load_weather(season, week)
    game_rows, prop_rows = [], []
    t0 = time.perf_counter()
    for g in sched.iter_rows(named=True):
        gid, home, away = g["game_id"], g["home_team"], g["away_team"]
        w = weather.get(gid) or {"roof_dome": 1 if g.get("roof") in ("dome", "closed") else 0, "wind": g.get("wind")}
        ctx = wi.context(home, away, w)
        G = pack_game(ctx, home, away, neutral=g.get("location") == "Neutral", playoff=g["game_type"] != "REG")
        gs = game_seed(seed, gid)
        h, a, ot, ts, ps = simulate_games(sims, gs, FM, G)
        n_max = G.eff.shape[1]
        ids = [f"{home}:{pid}" for pid in G.player_ids[0]] + [""] * (n_max - len(G.player_ids[0])) + \
              [f"{away}:{pid}" for pid in G.player_ids[1]] + [""] * (n_max - len(G.player_ids[1]))
        np.savez_compressed(out_dir / f"sims_{gid}.npz", home_score=h, away_score=a, went_ot=ot, team_stats=ts, player_stats=ps,
                            player_ids=np.array(ids), seed=gs)
        game_rows += game_markets(gid, home, away, h, a, ot)
        for side, team in ((0, home), (1, away)):
            tp = ctx.players.get(team)
            for k, pid in enumerate(G.player_ids[side]):
                prop_rows += player_markets(gid, team, pid, tp.names.get(pid, pid), tp.players[pid]["position"], ps[:, side * n_max + k, :])
        typer.echo(f"  {gid}: P(home) {float((h > a).mean()):.3f}  fair spread {np.median(h - a):+.1f}  total {float((h + a).mean()):.1f}")
    for line in wi.blend_log:
        typer.echo(f"  qb blend: {line}")
    fair = pl.DataFrame(game_rows, infer_schema_length=None)
    fair.write_parquet(out_dir / "fair_odds.parquet")
    props = player_markets_frame(prop_rows)
    if props.height:
        props.write_parquet(out_dir / "player_props.parquet")
    meta = {"season": season, "week": week, "sims": sims, "seed": seed, "games": sched["game_id"].to_list(),
            "elapsed_s": round(time.perf_counter() - t0, 1), "qb_blends": wi.blend_log}
    (out_dir / "run.json").write_text(json.dumps(meta, indent=1))
    typer.echo(f"{sched.height} games x {sims} sims in {time.perf_counter() - t0:.1f}s -> {out_dir}")
    return fair
