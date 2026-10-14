"""Phase 3, step 3.1: leak-free team ratings as of the morning of each game.

Every rating is a ratio of exponentially weighted per-game numerators and denominators (half-life in games), with
last season's final rating, regressed toward the league mean, as a prior worth six games. Ratings are stored as
z-scores across the league for each (season, week). Rows exist for every team and every week of the schedule,
using only games whose date is strictly before the team's as-of date for that week.
"""
from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import numpy as np
import polars as pl
import typer

from nflsim.config import DERIVED_DIR, MODELS_DIR, settings
from nflsim.data.clean import load_plays, load_schedules

RATINGS_PATH = DERIVED_DIR / "team_ratings.parquet"
SITUATION_MODELS_DIR = MODELS_DIR / "situation"

OFFENSE = ["pass_epa", "rush_epa", "proe", "adot", "sack_rate", "cpoe", "explosive_rate", "pace"]
DEFENSE = ["pass_epa_allowed", "rush_epa_allowed", "sack_rate_forced", "int_rate_forced"]
COACH = ["fourth_aggr"]
RATINGS = OFFENSE + DEFENSE + COACH
PRIOR_GAMES = 6


# ---- situation-only model predictions used by proe, cpoe and fourth_aggr ----------------------------------------
def _situation_predictions(plays: pl.DataFrame) -> pl.DataFrame:
    """Adds p_pass (play_call), p_complete (throw_outcome) and p_go (fourth_down) from the Phase 2 situation-only models."""
    from nflsim.models.features import feature_matrix
    from nflsim.models.train import rows_of
    from nflsim.models.trained import load_trained

    def predict(name: str, mask: pl.Expr, class_label) -> pl.Series:
        m = load_trained(name, SITUATION_MODELS_DIR)
        sub = plays.filter(mask)
        X = feature_matrix(name, rows_of(sub))
        P = m.proba_from_features(X)[:, m.classes.index(class_label)]
        pred = sub.select("game_id", "play_id").with_columns(pl.Series("_p", P))
        return plays.select("game_id", "play_id").join(pred, on=["game_id", "play_id"], how="left")["_p"]

    calls = pl.col("play_class").is_in(["pass", "run"]) & pl.col("down").is_between(1, 3) & pl.col("yardline_100").is_not_null()
    att = pl.col("pass_outcome").is_in(["complete", "incomplete", "interception"]) & pl.col("air_yards").is_not_null() & pl.col("yardline_100").is_not_null()
    fourth = (pl.col("down") == 4) & pl.col("play_class").is_in(["pass", "run", "punt", "field_goal"]) & pl.col("yardline_100").is_not_null()
    return plays.with_columns(
        p_pass=predict("play_call", calls, "pass"),
        p_complete=predict("throw_outcome", att, "complete"),
        p_go=predict("fourth_down", fourth, "go"),
    )


# ---- per-game, per-team numerators and denominators --------------------------------------------------------------
def game_stats(plays: pl.DataFrame) -> pl.DataFrame:
    """One row per (game, team) with num_<rating> / den_<rating> columns, plus game_date, season, week."""
    p = plays.filter(pl.col("season_type").is_in(["REG", "POST"]) & pl.col("posteam").is_not_null())
    loc = load_schedules().select("game_id", neutral=(pl.col("location") == "Neutral"))
    p = p.join(loc, on="game_id", how="left").with_columns(pl.col("neutral").fill_null(False))
    p = _situation_predictions(p)
    is_pass = pl.col("play_class") == "pass"
    is_run = pl.col("play_class") == "run"
    is_att = pl.col("pass_outcome").is_in(["complete", "incomplete", "interception"])
    neutral = (pl.col("wp") > 0.2) & (pl.col("wp") < 0.8) & (pl.col("qtr") <= 3)
    call = pl.col("play_class").is_in(["pass", "run"]) & pl.col("down").is_between(1, 3) & pl.col("p_pass").is_not_null()
    fourth = (pl.col("down") == 4) & pl.col("play_class").is_in(["pass", "run", "punt", "field_goal"]) & pl.col("p_go").is_not_null()
    off_snap = pl.col("play_class").is_in(["pass", "run"])
    epa = pl.col("epa").fill_null(0.0)
    sack = (pl.col("pass_outcome") == "sack").cast(pl.Float64)
    interception = (pl.col("pass_outcome") == "interception").cast(pl.Float64)
    explosive = (pl.col("yards_gained").fill_null(0) >= 20).cast(pl.Float64)

    def agg(team_col: str) -> pl.DataFrame:
        base = [pl.col("season").first(), pl.col("week").first(), pl.col("game_date").first()]
        if team_col == "posteam":
            exprs = [
                epa.filter(is_pass).sum().alias("num_pass_epa"), is_pass.sum().cast(pl.Float64).alias("den_pass_epa"),
                epa.filter(is_run).sum().alias("num_rush_epa"), is_run.sum().cast(pl.Float64).alias("den_rush_epa"),
                (is_pass.cast(pl.Float64) - pl.col("p_pass")).filter(call & neutral).sum().alias("num_proe"),
                (call & neutral).sum().cast(pl.Float64).alias("den_proe"),
                pl.col("air_yards").filter(is_att).sum().alias("num_adot"), is_att.sum().cast(pl.Float64).alias("den_adot"),
                sack.filter(is_pass).sum().alias("num_sack_rate"), is_pass.sum().cast(pl.Float64).alias("den_sack_rate"),
                ((pl.col("pass_outcome") == "complete").cast(pl.Float64) - pl.col("p_complete")).filter(is_att & pl.col("p_complete").is_not_null()).sum().alias("num_cpoe"),
                (is_att & pl.col("p_complete").is_not_null()).sum().cast(pl.Float64).alias("den_cpoe"),
                explosive.filter(off_snap).sum().alias("num_explosive_rate"), off_snap.sum().cast(pl.Float64).alias("den_explosive_rate"),
                (pl.col("seconds_elapsed").filter(off_snap & neutral & pl.col("seconds_elapsed").is_not_null()).median().fill_null(0.0)
                 * (off_snap & neutral & pl.col("seconds_elapsed").is_not_null()).sum()).alias("num_pace"),
                (off_snap & neutral & pl.col("seconds_elapsed").is_not_null()).sum().cast(pl.Float64).alias("den_pace"),
                (pl.col("play_class").is_in(["pass", "run"]).cast(pl.Float64) - pl.col("p_go")).filter(fourth).sum().alias("num_fourth_aggr"),
                fourth.sum().cast(pl.Float64).alias("den_fourth_aggr"),
            ]
        else:
            exprs = [
                epa.filter(is_pass).sum().alias("num_pass_epa_allowed"), is_pass.sum().cast(pl.Float64).alias("den_pass_epa_allowed"),
                epa.filter(is_run).sum().alias("num_rush_epa_allowed"), is_run.sum().cast(pl.Float64).alias("den_rush_epa_allowed"),
                sack.filter(is_pass).sum().alias("num_sack_rate_forced"), is_pass.sum().cast(pl.Float64).alias("den_sack_rate_forced"),
                interception.filter(is_att).sum().alias("num_int_rate_forced"), is_att.sum().cast(pl.Float64).alias("den_int_rate_forced"),
            ]
        return p.group_by("game_id", pl.col(team_col).alias("team")).agg(base + exprs)

    off = agg("posteam")
    deff = agg("defteam").drop("season", "week", "game_date")
    out = off.join(deff, on=["game_id", "team"], how="full", coalesce=True)
    numden = [c for c in out.columns if c.startswith("num_") or c.startswith("den_")]
    return out.with_columns([pl.col(c).fill_null(0.0) for c in numden]).sort("game_date", "game_id", "team")


# ---- exponentially weighted, as-of ratings -------------------------------------------------------------------------
def _raw_rating(prior_num: float, prior_den: float, nums: list[float], dens: list[float], lam: float) -> float:
    """nums/dens ordered oldest to newest; the newest game has weight 1, older ones decay by lam per game."""
    n, d = prior_num, prior_den
    k = len(nums)
    for i in range(k):
        w = lam ** (k - 1 - i)
        n += w * nums[i]
        d += w * dens[i]
    return n / d if d > 0 else float("nan")


def build_team_ratings(plays: pl.DataFrame | None = None, write: bool = True) -> pl.DataFrame:
    plays = load_plays() if plays is None else plays
    return ratings_from_game_stats(game_stats(plays), write=write)


def ratings_from_game_stats(gs: pl.DataFrame, write: bool = False, sched: pl.DataFrame | None = None) -> pl.DataFrame:
    """The exponential-weighting / prior / z-score step, separated from the play aggregation so a date-restricted
    recompute (the leakage test) is cheap. `gs` is the output of `game_stats`."""
    cfg = settings()["team_strength"]
    lam = 0.5 ** (1.0 / cfg["half_life_games"])
    regress = cfg["prior_season_regression"]
    prior_games = cfg.get("prior_games", PRIOR_GAMES)
    sched = (load_schedules() if sched is None else sched).filter(pl.col("game_type") != "PRE")   # REG, WC, DIV, CON, SB
    seasons = sorted(gs["season"].unique().to_list())
    teams = sorted(set(gs["team"].to_list()))
    # per team-season game lists (ordered by date)
    by_team: dict[tuple[str, int], list[dict]] = {}
    for r in gs.iter_rows(named=True):
        by_team.setdefault((r["team"], r["season"]), []).append(r)
    rows = []
    priors: dict[tuple[str, int], dict[str, float]] = {}       # (team, season) -> prior rating per stat (or nan)
    season_scale: dict[int, dict[str, float]] = {}             # season -> std of last season's final raw ratings per stat
    prior_den: dict[int, dict[str, float]] = {}                # season -> prior denominator (6 games' worth) per stat
    for season in seasons:
        weeks = sorted(sched.filter(pl.col("season") == season)["week"].unique().to_list())
        week_dates = {w: sched.filter((pl.col("season") == season) & (pl.col("week") == w))["gameday"].min() for w in weeks}
        if season - 1 in seasons:
            finals = {}
            for t in teams:
                games = by_team.get((t, season - 1), [])
                if not games:
                    continue
                pn, pd_ = priors.get((t, season - 1), {}), prior_den.get(season - 1, {})
                finals[t] = {s: _raw_rating(pn.get(s, float("nan")) * pd_.get(s, 0.0) if pd_.get(s, 0.0) > 0 and not math.isnan(pn.get(s, float("nan"))) else 0.0,
                                            pd_.get(s, 0.0) if not math.isnan(pn.get(s, float("nan"))) else 0.0,
                                            [g[f"num_{s}"] for g in games], [g[f"den_{s}"] for g in games], lam) for s in RATINGS}
            league_mean = {s: float(np.nanmean([finals[t][s] for t in finals])) for s in RATINGS}
            season_scale[season] = {s: float(np.nanstd([finals[t][s] for t in finals])) for s in RATINGS}
            prev_games = [g for t in teams for g in by_team.get((t, season - 1), [])]
            prior_den[season] = {s: prior_games * float(np.mean([g[f"den_{s}"] for g in prev_games])) for s in RATINGS}
            for t in teams:
                if t in finals:
                    priors[(t, season)] = {s: league_mean[s] + (1 - regress) * (finals[t][s] - league_mean[s]) if not math.isnan(finals[t][s]) else league_mean[s] for s in RATINGS}
                else:
                    priors[(t, season)] = dict(league_mean)
        else:
            prior_den[season] = {s: 0.0 for s in RATINGS}
            for t in teams:
                priors[(t, season)] = {s: float("nan") for s in RATINGS}
        for t in teams:
            games = by_team.get((t, season), [])
            if not games and season - 1 not in seasons:
                continue
            own_dates = {g["week"]: g["game_date"] for g in games}
            for w in weeks:
                asof = own_dates.get(w, week_dates[w])
                if asof is None:
                    continue
                hist = [g for g in games if g["game_date"] < asof]
                row = {"team": t, "season": season, "week": w, "asof_date": asof, "games_played": len(hist)}
                for s in RATINGS:
                    row[f"scale_{s}"] = season_scale.get(season, {}).get(s)
                for s in RATINGS:
                    pr = priors[(t, season)][s]
                    pd_ = prior_den[season][s] if not math.isnan(pr) else 0.0
                    pn = pr * pd_ if pd_ > 0 else 0.0
                    row[f"raw_{s}"] = _raw_rating(pn, pd_, [g[f"num_{s}"] for g in hist], [g[f"den_{s}"] for g in hist], lam)
                rows.append(row)
    df = pl.DataFrame(rows, infer_schema_length=None).with_columns([pl.col(f"raw_{s}").fill_nan(None) for s in RATINGS])
    # Standardized ratings: centered on the league mean for that week, scaled by the spread of last season's final
    # ratings (a per-week std would re-inflate the shrunken early-season ratings to the same size as sharp late-season
    # ones). The first season, with no previous finals, falls back to the within-week std. Null raw = z 0.
    zexprs = []
    for s in RATINGS:
        raw = pl.col(f"raw_{s}")
        mean = raw.mean().over("season", "week")
        wk_std = raw.std().over("season", "week")
        scale = pl.coalesce([pl.col(f"scale_{s}"), wk_std])
        zexprs.append(pl.when(raw.is_null() | scale.is_null() | (scale == 0)).then(0.0).otherwise((raw - mean) / scale).alias(s))
    df = df.with_columns(zexprs).sort("season", "week", "team")
    if write:
        RATINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(RATINGS_PATH)
        typer.echo(f"team ratings: {df.height} rows ({df['season'].min()}-{df['season'].max()}) -> {RATINGS_PATH}")
    return df


def load_team_ratings() -> pl.DataFrame:
    return pl.read_parquet(RATINGS_PATH)


def ratings_for_week(season: int, week: int, ratings: pl.DataFrame | None = None) -> dict[str, dict[str, float]]:
    """{team: {rating: z}} as of the given week."""
    df = (ratings if ratings is not None else load_team_ratings()).filter((pl.col("season") == season) & (pl.col("week") == week))
    return {r["team"]: {s: r[s] for s in RATINGS} for r in df.iter_rows(named=True)}


def ratings_markdown(season: int, week: int, ratings: pl.DataFrame | None = None) -> str:
    df = (ratings if ratings is not None else load_team_ratings()).filter((pl.col("season") == season) & (pl.col("week") == week))
    cols = ["team", "games_played"] + RATINGS
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = "\n".join("| " + " | ".join(str(r[c]) if c in ("team", "games_played") else f"{r[c]:+.2f}" for c in cols) + " |"
                     for r in df.sort("pass_epa", descending=True).iter_rows(named=True))
    return head + body + "\n"
