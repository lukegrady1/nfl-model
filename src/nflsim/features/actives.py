"""Phase 4, step 4.1: active skill players per team for a week, saved to data/inputs/actives_<season>_wk<week>.json."""
from __future__ import annotations

import json

import polars as pl
import typer

from nflsim.config import INPUTS_DIR, RAW_DIR
from nflsim.features.players import SKILL_POSITIONS

INACTIVE_REPORT = {"Out", "Doubtful"}
INACTIVE_ROSTER = {"RES", "CUT", "RET", "EXE", "SUS", "PUP", "NON", "DEV", "INA"}   # IR, practice squad, inactive list ...


def weekly_roster(season: int, week: int) -> pl.DataFrame:
    """The week's roster, or the latest published week before it (nflverse posts rosters as each week is played)."""
    r = pl.read_parquet(RAW_DIR / f"rosters_weekly_{season}.parquet", columns=["season", "week", "team", "position", "gsis_id", "full_name", "status"])
    weeks = [w for w in r["week"].unique().to_list() if w <= week]
    if not weeks:
        return r.clear()
    use = max(weeks)
    if use != week:
        typer.echo(f"warning: no weekly roster for {season} week {week} yet; using week {use}")
    return r.filter((pl.col("week") == use) & pl.col("position").is_in(SKILL_POSITIONS) & pl.col("gsis_id").is_not_null())


def injury_report(season: int, week: int) -> pl.DataFrame:
    path = RAW_DIR / f"injuries_{season}.parquet"
    if not path.exists():
        return pl.DataFrame({"gsis_id": [], "report_status": []}, schema={"gsis_id": pl.String, "report_status": pl.String})
    i = pl.read_parquet(path, columns=["week", "gsis_id", "report_status", "team"])
    return i.filter((pl.col("week") == week) & pl.col("gsis_id").is_not_null()).select("gsis_id", "report_status")


def load_overrides(season: int, week: int) -> pl.DataFrame | None:
    path = INPUTS_DIR / f"overrides_{season}_wk{week}.csv"
    if not path.exists():
        return None
    return pl.read_csv(path, schema_overrides={"starter_qb": pl.String})


def recent_dropbacks(season: int, week: int, team: str, usage: pl.DataFrame) -> dict[str, float]:
    """Dropbacks per QB over the team's recent games as of this week, from the as-of usage table (EW n_qb_db)."""
    sub = usage.filter((pl.col("season") == season) & (pl.col("week") == week) & (pl.col("team") == team) & (pl.col("position") == "QB"))
    return {r["player_id"]: float(r["n_qb_cpoe"] or 0.0) for r in sub.iter_rows(named=True)}


def build_actives(season: int, week: int, eff: pl.DataFrame, write: bool = True, quiet: bool = False) -> dict:
    """{team: {"players": [{player_id, name, position, status}], "starting_qb": id, "warnings": [...]}}"""
    roster = weekly_roster(season, week)
    inj = {r["gsis_id"]: r["report_status"] for r in injury_report(season, week).iter_rows(named=True)}
    ov = load_overrides(season, week)
    out: dict = {}
    for (team,), rows in roster.group_by(["team"], maintain_order=True):
        players, warnings = [], []
        forced_out, forced_active, starter_override = set(), set(), None
        if ov is not None:
            for o in ov.filter(pl.col("team") == team).iter_rows(named=True):
                if str(o.get("status", "")).lower() == "out":
                    forced_out.add(o["player_id"])
                elif str(o.get("status", "")).lower() == "active":
                    forced_active.add(o["player_id"])
                if str(o.get("starter_qb", "")).lower() in ("true", "1", "yes"):
                    starter_override = o["player_id"]
        for r in rows.iter_rows(named=True):
            pid, status = r["gsis_id"], r["status"]
            report = inj.get(pid)
            active = status == "ACT" and report not in INACTIVE_REPORT
            if pid in forced_out:
                active = False
            if pid in forced_active:
                active = True
            if not active:
                continue
            players.append({"player_id": pid, "name": r["full_name"], "position": r["position"], "report_status": report})
            if report == "Questionable":
                warnings.append(f"{r['full_name']} ({r['position']}) is Questionable")
        qbs = [p for p in players if p["position"] == "QB"]
        starter = None
        if starter_override and any(p["player_id"] == starter_override for p in players):
            starter = starter_override
        elif qbs:
            db = recent_dropbacks(season, week, team, eff)
            starter = max(qbs, key=lambda p: db.get(p["player_id"], 0.0))["player_id"]
        out[team] = {"players": players, "starting_qb": starter, "warnings": warnings}
    if write:
        INPUTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(INPUTS_DIR / f"actives_{season}_wk{week}.json", "w") as f:
            json.dump(out, f, indent=1)
    if not quiet:
        n = sum(len(v["players"]) for v in out.values())
        typer.echo(f"actives {season} week {week}: {len(out)} teams, {n} players")
        for team, v in out.items():
            for w in v["warnings"]:
                typer.echo(f"  warning {team}: {w}")
    return out


def load_actives(season: int, week: int) -> dict:
    with open(INPUTS_DIR / f"actives_{season}_wk{week}.json") as f:
        return json.load(f)


# ---- Phase 7: actives from actual participation (isolates model quality from injury guessing) ------------------
def played_actives(season: int, plays, eff: pl.DataFrame, week: int | None = None) -> dict[int, dict]:
    """{week: {team: {"players": [...], "starting_qb": id}}} from players who touched the ball or took an offensive
    snap in that game (snap counts via the players crosswalk), with the game's most-used passer as the starting QB."""
    from nflsim.features.players import SKILL_POSITIONS, player_crosswalk, qb_id_expr

    p = plays.filter((pl.col("season") == season) & (pl.col("season_type") == "REG"))
    if week is not None:
        p = p.filter(pl.col("week") == week)
    p = p.with_columns(qb_player_id=qb_id_expr())
    xw = player_crosswalk().select("gsis_id", "pfr_id", "display_name", "position").unique(subset=["gsis_id"])
    names = {r["gsis_id"]: (r["display_name"], r["position"]) for r in xw.iter_rows(named=True)}
    touched = pl.concat([
        p.filter(pl.col("receiver_player_id").is_not_null()).select("week", "posteam", pl.col("receiver_player_id").alias("pid")),
        p.filter(pl.col("rusher_player_id").is_not_null()).select("week", "posteam", pl.col("rusher_player_id").alias("pid")),
        p.filter(pl.col("qb_player_id").is_not_null()).select("week", "posteam", pl.col("qb_player_id").alias("pid")),
        p.filter(pl.col("kicker_player_id").is_not_null()).select("week", "posteam", pl.col("kicker_player_id").alias("pid")),
    ]).unique()
    snaps_path = RAW_DIR / f"snap_counts_{season}.parquet"
    if snaps_path.exists():
        sc = pl.read_parquet(snaps_path).filter((pl.col("game_type") == "REG") & (pl.col("offense_snaps") > 0))
        if week is not None:
            sc = sc.filter(pl.col("week") == week)
        sc = sc.join(xw.filter(pl.col("pfr_id").is_not_null()).select("gsis_id", "pfr_id"), left_on="pfr_player_id", right_on="pfr_id", how="inner")
        touched = pl.concat([touched, sc.select(pl.col("week").cast(touched.schema["week"]), pl.col("team").alias("posteam"), pl.col("gsis_id").alias("pid"))]).unique()
    qbs = p.filter(pl.col("qb_player_id").is_not_null()).group_by("week", "posteam", "qb_player_id").len().sort("len", descending=True)
    starters = {(r["week"], r["posteam"]): r["qb_player_id"] for r in qbs.unique(subset=["week", "posteam"], keep="first").iter_rows(named=True)}
    out: dict[int, dict] = {}
    for r in touched.iter_rows(named=True):
        name, pos = names.get(r["pid"], (r["pid"], None))
        if pos not in SKILL_POSITIONS:
            continue
        w = int(r["week"])
        out.setdefault(w, {}).setdefault(r["posteam"], {"players": [], "starting_qb": starters.get((w, r["posteam"])), "warnings": []})
        out[w][r["posteam"]]["players"].append({"player_id": r["pid"], "name": name, "position": pos, "report_status": None})
    return out
