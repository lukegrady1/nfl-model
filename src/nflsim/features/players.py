"""Phase 4, steps 4.2 and 4.4: per-player usage shares and efficiency over expected, as of before each game.

Shares use the Phase 3 exponential weighting with half-life `usage_half_life_games` (4); efficiencies use the team
half-life (8) and are shrunk toward 0 (the position average of an over-expected quantity) with `player_shrinkage_k`
pseudo-plays (kickers: 100 attempts). Rows exist for every (player, season, week) a player is on an active roster,
using only games dated strictly before the team's game that week.
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl
import typer

from nflsim.config import MODELS_DIR, PROCESSED_DIR, RAW_DIR, settings
from nflsim.data.clean import load_plays, load_schedules

USAGE_PATH = PROCESSED_DIR / "player_usage.parquet"
EFF_PATH = PROCESSED_DIR / "player_eff.parquet"
SITUATION_MODELS_DIR = MODELS_DIR / "situation"
SKILL_POSITIONS = ["QB", "RB", "WR", "TE", "K"]
BANDS = ["short", "intermediate", "deep"]       # air yards < 5, 5-14, >= 15
USAGE_COUNTS = ["targets", "team_targets", "carries", "team_carries", "deep_targets", "team_deep_targets",
                "short_targets", "team_short_targets", "int_targets", "team_int_targets",
                "rz_targets", "team_rz_targets", "rz_carries", "team_rz_carries", "off_snaps", "team_snaps"]
EFF_COUNTS = {  # efficiency: (numerator over expected, denominator)
    "catch_rate_oe": ("n_catch_oe", "n_targets_eff"), "yac_oe": ("n_yac_oe", "n_catches"),
    "rush_yards_oe": ("n_rush_oe", "n_rushes"), "qb_cpoe": ("n_qb_cpoe", "n_qb_att"),
    "qb_sack_rate": ("n_qb_sack_oe", "n_qb_db"), "qb_int_rate": ("n_qb_int_oe", "n_qb_att"),
    "qb_scramble_rate": ("n_qb_scr_oe", "n_qb_db"), "qb_adot": ("n_qb_adot_oe", "n_qb_att"),
    "kicker_fg_oe": ("n_fg_oe", "n_fg"),
}
EFFICIENCIES = list(EFF_COUNTS)
QB_EFF = ["qb_cpoe", "qb_sack_rate", "qb_int_rate", "qb_scramble_rate", "qb_adot"]


# ---- situation-only expectations per play ----------------------------------------------------------------------
def _expectations(plays: pl.DataFrame) -> pl.DataFrame:
    """Adds x_complete, x_yac, x_rush, x_sack, x_scramble, x_int, x_air, x_fg from the situation-only models."""
    from nflsim.models.features import feature_matrix
    from nflsim.models.train import rows_of
    from nflsim.models.trained import apply_mask, feasible_mask, load_trained, room_for

    def predict(name: str, mask: pl.Expr, fn) -> pl.Series:
        m = load_trained(name, SITUATION_MODELS_DIR)
        sub = plays.filter(mask)
        rows = rows_of(sub)
        P = m.proba_from_features(feature_matrix(name, rows))
        if name in ("air_yards", "yac", "rush_yards"):
            for i, r in enumerate(rows):
                P[i] = apply_mask(P[i], feasible_mask(name, m.classes, room_for(name, r["yardline_100"], r.get("air_yards"))))
        vals = fn(m, P)
        pred = sub.select("game_id", "play_id").with_columns(pl.Series("_p", vals))
        return plays.select("game_id", "play_id").join(pred, on=["game_id", "play_id"], how="left")["_p"]

    def ev(m, P, mapping=None):
        v = np.array([mapping(c) for c in m.classes]) if mapping else None
        return P @ v

    att = pl.col("pass_outcome").is_in(["complete", "incomplete", "interception"]) & pl.col("air_yards").is_not_null() & pl.col("yardline_100").is_not_null()
    comp = (pl.col("pass_outcome") == "complete") & pl.col("yards_after_catch").is_not_null() & pl.col("air_yards").is_not_null() & pl.col("yardline_100").is_not_null()
    run = (pl.col("play_class") == "run") & pl.col("yardline_100").is_not_null()
    db = (pl.col("play_class") == "pass") & pl.col("yardline_100").is_not_null()
    fg = (pl.col("play_class") == "field_goal") & pl.col("kick_distance").is_not_null()
    yac_val = lambda c: int(c)
    rush_val = lambda c: {-6: -8.0, 16: 20.0, 25: 32.0, 40: 45.0}.get(int(c), float(c))
    return plays.with_columns(
        x_complete=predict("throw_outcome", att, lambda m, P: P[:, m.classes.index("complete")]),
        x_int=predict("throw_outcome", att, lambda m, P: P[:, m.classes.index("interception")]),
        x_yac=predict("yac", comp, lambda m, P: ev(m, P, yac_val)),
        x_rush=predict("rush_yards", run, lambda m, P: ev(m, P, rush_val)),
        x_sack=predict("dropback_outcome", db, lambda m, P: P[:, m.classes.index("sack")]),
        x_scramble=predict("dropback_outcome", db, lambda m, P: P[:, m.classes.index("scramble")]),
        x_air=predict("air_yards", att, lambda m, P: ev(m, P, lambda c: {0: -3.0, 1: 2.0, 2: 7.0, 3: 12.0, 4: 17.0, 5: 24.0, 6: 36.0}[int(c)])),
        x_fg=predict("fg_make", fg, lambda m, P: P[:, m.classes.index("made")]),
    )


def qb_id_expr() -> pl.Expr:
    """The quarterback on a dropback: nflfastR puts the scrambling QB in rusher_player_id with no passer id."""
    return pl.when(pl.col("pass_outcome") == "scramble").then(pl.col("rusher_player_id")).otherwise(pl.col("passer_player_id"))


# ---- per-game, per-player counts ---------------------------------------------------------------------------------
def player_crosswalk() -> pl.DataFrame:
    p = pl.read_parquet(RAW_DIR / "players.parquet").select("gsis_id", "pfr_id", "display_name", "position")
    return p.filter(pl.col("gsis_id").is_not_null())


def player_game_stats(plays: pl.DataFrame | None = None) -> pl.DataFrame:
    """One row per (game, team, player) with usage counts and efficiency numerators/denominators. Players appear
    when they had an offensive snap (snap counts) or any recorded touch; team totals are repeated on each row."""
    plays = load_plays() if plays is None else plays
    p = plays.filter(pl.col("season_type").is_in(["REG", "POST"]) & pl.col("posteam").is_not_null())
    p = p.with_columns(qb_player_id=qb_id_expr())
    p = _expectations(p)
    is_att = pl.col("pass_outcome").is_in(["complete", "incomplete", "interception"])
    is_run = pl.col("play_class") == "run"
    is_db = pl.col("play_class") == "pass"
    air = pl.col("air_yards")
    rz = pl.col("yardline_100") <= 20
    key = ["game_id", "season", "week", "game_date", "posteam"]
    pkey = ["game_id", "posteam"]
    # receivers
    rec = p.filter(is_att & pl.col("receiver_player_id").is_not_null()).group_by(pkey + [pl.col("receiver_player_id").alias("player_id")]).agg(
        pl.len().alias("targets"), (air >= 15).sum().alias("deep_targets"), (air < 5).sum().alias("short_targets"),
        ((air >= 5) & (air < 15)).sum().alias("int_targets"), rz.sum().alias("rz_targets"),
        ((pl.col("pass_outcome") == "complete").cast(pl.Float64) - pl.col("x_complete")).filter(pl.col("x_complete").is_not_null()).sum().alias("n_catch_oe"),
        pl.col("x_complete").is_not_null().sum().alias("n_targets_eff"),
        (pl.col("yards_after_catch") - pl.col("x_yac")).filter(pl.col("x_yac").is_not_null()).sum().alias("n_yac_oe"),
        pl.col("x_yac").is_not_null().sum().alias("n_catches"))
    rush = p.filter(is_run & pl.col("rusher_player_id").is_not_null()).group_by(pkey + [pl.col("rusher_player_id").alias("player_id")]).agg(
        pl.len().alias("carries"), rz.sum().alias("rz_carries"),
        (pl.col("yards_gained") - pl.col("x_rush")).filter(pl.col("x_rush").is_not_null()).sum().alias("n_rush_oe"),
        pl.col("x_rush").is_not_null().sum().alias("n_rushes"))
    qb = p.filter(is_db & pl.col("qb_player_id").is_not_null()).group_by(pkey + [pl.col("qb_player_id").alias("player_id")]).agg(
        pl.len().alias("n_qb_db"), is_att.sum().alias("n_qb_att"),
        ((pl.col("pass_outcome") == "complete").cast(pl.Float64) - pl.col("x_complete")).filter(is_att & pl.col("x_complete").is_not_null()).sum().alias("n_qb_cpoe"),
        ((pl.col("pass_outcome") == "sack").cast(pl.Float64) - pl.col("x_sack")).filter(pl.col("x_sack").is_not_null()).sum().alias("n_qb_sack_oe"),
        ((pl.col("pass_outcome") == "scramble").cast(pl.Float64) - pl.col("x_scramble")).filter(pl.col("x_scramble").is_not_null()).sum().alias("n_qb_scr_oe"),
        ((pl.col("pass_outcome") == "interception").cast(pl.Float64) - pl.col("x_int")).filter(is_att & pl.col("x_int").is_not_null()).sum().alias("n_qb_int_oe"),
        (air - pl.col("x_air")).filter(is_att & pl.col("x_air").is_not_null()).sum().alias("n_qb_adot_oe"))
    kick = p.filter((pl.col("play_class") == "field_goal") & pl.col("kicker_player_id").is_not_null() & pl.col("x_fg").is_not_null()).group_by(
        pkey + [pl.col("kicker_player_id").alias("player_id")]).agg(
        pl.len().alias("n_fg"), ((pl.col("field_goal_result") == "made").cast(pl.Float64) - pl.col("x_fg")).sum().alias("n_fg_oe"))
    team = p.group_by(key).agg(
        is_att.sum().alias("team_targets"), is_run.sum().alias("team_carries"), (is_att & (air >= 15)).sum().alias("team_deep_targets"),
        (is_att & (air < 5)).sum().alias("team_short_targets"), (is_att & (air >= 5) & (air < 15)).sum().alias("team_int_targets"),
        (is_att & rz).sum().alias("team_rz_targets"), (is_run & rz).sum().alias("team_rz_carries"),
        (is_att | is_run).sum().alias("team_snaps"))
    # snap counts (PFR ids -> gsis) give participation for players without a touch
    xw = player_crosswalk().select("gsis_id", "pfr_id").filter(pl.col("pfr_id").is_not_null()).unique(subset=["pfr_id"])
    seasons = sorted(p["season"].unique().to_list())
    snaps = pl.concat([pl.read_parquet(RAW_DIR / f"snap_counts_{s}.parquet") for s in seasons if (RAW_DIR / f"snap_counts_{s}.parquet").exists()], how="diagonal_relaxed")
    snaps = snaps.filter(pl.col("offense_snaps") > 0).join(xw, left_on="pfr_player_id", right_on="pfr_id", how="inner")
    snaps = snaps.select("game_id", pl.col("team").alias("posteam"), pl.col("gsis_id").alias("player_id"), pl.col("position").alias("snap_position"),
                         pl.col("offense_snaps").alias("off_snaps"))
    ids = pl.concat([d.select("game_id", "posteam", "player_id") for d in (rec, rush, qb, kick, snaps)]).unique()
    out = ids.join(rec, on=["game_id", "posteam", "player_id"], how="left").join(rush, on=["game_id", "posteam", "player_id"], how="left") \
        .join(qb, on=["game_id", "posteam", "player_id"], how="left").join(kick, on=["game_id", "posteam", "player_id"], how="left") \
        .join(snaps, on=["game_id", "posteam", "player_id"], how="left")
    gmeta = p.select("game_id", "season", "week", "game_date").unique(subset=["game_id"])
    out = out.join(gmeta, on="game_id", how="left").join(team, on=["game_id", "season", "week", "game_date", "posteam"], how="left")
    numcols = list(dict.fromkeys(USAGE_COUNTS + [v for pair in EFF_COUNTS.values() for v in pair]))
    out = out.with_columns([pl.col(c).cast(pl.Float64).fill_null(0.0) for c in numcols if c in out.columns])
    pos = player_crosswalk().select("gsis_id", "position").unique(subset=["gsis_id"]).rename({"gsis_id": "player_id", "position": "roster_position"})
    out = out.join(pos, on="player_id", how="left").with_columns(position=pl.coalesce([pl.col("snap_position"), pl.col("roster_position")]))
    return out.rename({"posteam": "team"}).sort("game_date", "game_id", "team", "player_id")


# ---- as-of exponential weighting --------------------------------------------------------------------------------
def _asof_rows(stats: pl.DataFrame, sched: pl.DataFrame, roster_grid: pl.DataFrame) -> pl.DataFrame:
    """For every (player, season, week) in roster_grid, EW sums of every count over the player's games dated before the
    team's game that week (or the week's first kickoff on a bye). Usage uses the usage half-life, efficiency the team one."""
    cfg = settings()
    lam_u = 0.5 ** (1.0 / cfg["player_usage"]["half_life_games"])
    lam_e = 0.5 ** (1.0 / cfg["team_strength"]["half_life_games"])
    counts = [c for c in USAGE_COUNTS if c in stats.columns]
    effs = list(dict.fromkeys(v for pair in EFF_COUNTS.values() for v in pair if v in stats.columns))
    by_player: dict[str, list[dict]] = {}
    for r in stats.select(["player_id", "season", "game_date", "team", "position"] + counts + effs).iter_rows(named=True):
        by_player.setdefault(r["player_id"], []).append(r)
    week_dates = {(s, w): d for s, w, d in sched.group_by("season", "week").agg(pl.col("gameday").min()).iter_rows()}
    team_dates = {}
    for g in sched.iter_rows(named=True):
        team_dates[(g["home_team"], g["season"], g["week"])] = g["gameday"]
        team_dates[(g["away_team"], g["season"], g["week"])] = g["gameday"]
    rows = []
    for pid, season, week, team, position in roster_grid.select("player_id", "season", "week", "team", "position").iter_rows():
        asof = team_dates.get((team, season, week), week_dates.get((season, week)))
        if asof is None:
            continue
        games = [g for g in by_player.get(pid, []) if g["game_date"] < asof]
        # same-season games count fully; earlier seasons carry over through the decay (usage: at most last season)
        games = [g for g in games if g["season"] >= season - 1]
        row = {"player_id": pid, "season": season, "week": week, "team": team, "position": position,
               "games_played": len([g for g in games if g["season"] == season])}
        n = len(games)
        for c in counts:
            row[f"ew_{c}"] = sum(lam_u ** (n - 1 - i) * games[i][c] for i in range(n))
        for c in effs:
            row[f"ew_{c}"] = sum(lam_e ** (n - 1 - i) * games[i][c] for i in range(n))
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def roster_grid() -> pl.DataFrame:
    """(player, season, week, team, position) for every skill player on a weekly roster (any status), 2018-2026."""
    frames = []
    for path in sorted(RAW_DIR.glob("rosters_weekly_*.parquet")):
        r = pl.read_parquet(path, columns=["season", "week", "team", "position", "gsis_id", "status", "game_type"])
        frames.append(r.filter(pl.col("position").is_in(SKILL_POSITIONS) & pl.col("gsis_id").is_not_null()))
    g = pl.concat(frames).rename({"gsis_id": "player_id"}).unique(subset=["player_id", "season", "week"]).select(
        "player_id", "season", "week", "team", "position", "status")
    return g.with_columns(pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64))


def build_player_tables(plays: pl.DataFrame | None = None, write: bool = True) -> tuple[pl.DataFrame, pl.DataFrame]:
    cfg = settings()
    k = float(cfg["player_shrinkage_k"])
    k_kick = float(cfg["player_usage"].get("kicker_shrinkage_k", 100))
    stats = player_game_stats(plays)
    sched = load_schedules().filter(pl.col("game_type") != "PRE")
    grid = roster_grid()
    ew = _asof_rows(stats, sched, grid)
    # ---- usage shares (ratios of EW sums; renormalization across actives happens at game-build time)
    def share(num: str, den: str) -> pl.Expr:
        return pl.when(pl.col(f"ew_{den}") > 0).then(pl.col(f"ew_{num}") / pl.col(f"ew_{den}")).otherwise(None)

    usage = ew.with_columns(
        target_share=share("targets", "team_targets"), carry_share=share("carries", "team_carries"),
        deep_target_share=share("deep_targets", "team_deep_targets"), short_target_share=share("short_targets", "team_short_targets"),
        int_target_share=share("int_targets", "team_int_targets"),
        rz_target_share=share("rz_targets", "team_rz_targets"), rz_carry_share=share("rz_carries", "team_rz_carries"),
        snap_share=share("off_snaps", "team_snaps"),
        team_plays=pl.col("ew_team_snaps"),
    )
    # ---- efficiencies: EW over-expected per play, shrunk toward 0 with k pseudo-plays
    effs = usage
    for name, (num, den) in EFF_COUNTS.items():
        kk = k_kick if name == "kicker_fg_oe" else k
        effs = effs.with_columns(
            pl.when(pl.col(f"ew_{den}") > 0).then(pl.col(f"ew_{num}") / (pl.col(f"ew_{den}") + kk)).otherwise(0.0).alias(name),
            pl.col(f"ew_{den}").alias(f"n_{name}"))
    usage_cols = ["player_id", "season", "week", "team", "position", "games_played", "team_plays", "target_share", "carry_share",
                  "deep_target_share", "short_target_share", "int_target_share", "rz_target_share", "rz_carry_share", "snap_share",
                  "ew_targets", "ew_team_targets", "ew_carries", "ew_team_carries", "ew_deep_targets", "ew_team_deep_targets",
                  "ew_short_targets", "ew_team_short_targets", "ew_int_targets", "ew_team_int_targets", "ew_rz_targets",
                  "ew_rz_carries", "ew_off_snaps"]
    eff_cols = ["player_id", "season", "week", "team", "position"] + EFFICIENCIES + [f"n_{e}" for e in EFFICIENCIES]
    usage_df = effs.select([c for c in usage_cols if c in effs.columns]).sort("season", "week", "team", "player_id")
    eff_df = effs.select(eff_cols).sort("season", "week", "team", "player_id")
    if write:
        usage_df.write_parquet(USAGE_PATH)
        eff_df.write_parquet(EFF_PATH)
        typer.echo(f"player usage: {usage_df.height} rows -> {USAGE_PATH}; efficiency -> {EFF_PATH}")
    return usage_df, eff_df


def load_player_usage() -> pl.DataFrame:
    return pl.read_parquet(USAGE_PATH)


def load_player_eff() -> pl.DataFrame:
    return pl.read_parquet(EFF_PATH)


# ---- position-slot priors (step 4.2: rookies / thin history get the slot average) -------------------------------
PRIORS_PATH = PROCESSED_DIR / "position_priors.parquet"


def build_position_priors(usage: pl.DataFrame | None = None, write: bool = True) -> pl.DataFrame:
    """Average target / carry share by (position, depth slot), slot = rank by snap share within the team-week,
    over 2021-2025 rows with real history (>= min_team_plays team plays)."""
    u = usage if usage is not None else load_player_usage()
    cfg = settings()["player_usage"]
    u = u.filter(pl.col("season").is_between(2021, 2025) & (pl.col("team_plays") >= cfg["min_team_plays"]) & pl.col("snap_share").is_not_null())
    u = u.with_columns(slot=pl.col("snap_share").rank(method="ordinal", descending=True).over("season", "week", "team", "position").clip(1, 5))
    pri = u.group_by("position", "slot").agg(
        pl.col("target_share").fill_null(0).mean(), pl.col("carry_share").fill_null(0).mean(),
        pl.col("deep_target_share").fill_null(0).mean(), pl.col("short_target_share").fill_null(0).mean(),
        pl.col("int_target_share").fill_null(0).mean(), pl.col("rz_target_share").fill_null(0).mean(),
        pl.col("rz_carry_share").fill_null(0).mean(), pl.len().alias("n")).sort("position", "slot")
    if write:
        pri.write_parquet(PRIORS_PATH)
    return pri


def load_position_priors() -> pl.DataFrame:
    return pl.read_parquet(PRIORS_PATH)
