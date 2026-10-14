"""Phase 6, step 6.2: map FanDuel teams, games and players to nflverse ids. Never guess below the threshold."""
from __future__ import annotations

import difflib
import re
from pathlib import Path

import polars as pl
import typer
import yaml

from nflsim.config import CONFIG_DIR, team_aliases
from nflsim.data.clean import load_schedules

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(name: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", "", name.lower().replace("-", " ").replace(".", ""))
    parts = [p for p in s.split() if p not in SUFFIXES]
    return " ".join(parts)


def player_aliases() -> dict[str, str]:
    path = CONFIG_DIR / "player_aliases.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def map_team(name: str) -> str:
    aliases = team_aliases()
    if name in aliases:
        return aliases[name]
    if name in aliases.values():
        return name
    raise KeyError(f"unknown FanDuel team name: {name!r}; add it to config/team_aliases.yaml")


def match_games(df: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    """Adds nflverse game_id and team abbreviations; games match on both teams and a date within +/-1 day."""
    sched = load_schedules().filter((pl.col("season") == season) & (pl.col("week") == week))
    games = {(g["home_team"], g["away_team"]): (g["game_id"], g["gameday"]) for g in sched.iter_rows(named=True)}
    home_abbr, away_abbr, game_ids, team_abbr = [], [], [], []
    for r in df.iter_rows(named=True):
        h, a = map_team(r["home_name"]), map_team(r["away_name"])
        gid = None
        if (h, a) in games:
            gid_c, day = games[(h, a)]
            ev_day = r["event_date"][:10] if r["event_date"] else None
            if ev_day is None or abs((_date(ev_day) - _date(day)).days) <= 1:
                gid = gid_c
        home_abbr.append(h)
        away_abbr.append(a)
        game_ids.append(gid)
        t = r["team"]
        team_abbr.append(map_team(t) if t else None)
    return df.with_columns(pl.Series("home_team", home_abbr, dtype=pl.String), pl.Series("away_team", away_abbr, dtype=pl.String),
                           pl.Series("game_id", game_ids, dtype=pl.String), pl.Series("team", team_abbr, dtype=pl.String))


def _date(s: str):
    import datetime as dt
    return dt.date.fromisoformat(s[:10])


def match_players(df: pl.DataFrame, actives: dict, threshold: float = 0.9) -> tuple[pl.DataFrame, list[dict]]:
    """Players match within the game's two active rosters: exact normalized name, then fuzzy >= threshold, then
    the manual alias file. Returns the frame with player_id/team filled and a list of unmatched rows."""
    aliases = player_aliases()
    rosters: dict[str, dict[str, tuple[str, str]]] = {}   # team -> normalized name -> (player_id, position)
    for team, v in actives.items():
        rosters[team] = {normalize_name(p["name"]): (p["player_id"], p["position"]) for p in v["players"]}
    ids, teams, unmatched = [], [], []
    for r in df.iter_rows(named=True):
        name = r["player_name"]
        if not name:
            ids.append(r["player_id"])
            teams.append(r["team"])
            continue
        key = normalize_name(name)
        found = None
        for team in (r["home_team"], r["away_team"]):
            ros = rosters.get(team, {})
            if key in ros:
                found = (ros[key][0], team)
                break
            close = difflib.get_close_matches(key, list(ros), n=1, cutoff=threshold)
            if close:
                found = (ros[close[0]][0], team)
                break
        if found is None and name in aliases:
            pid = aliases[name]
            for team in (r["home_team"], r["away_team"]):
                if any(p[0] == pid for p in rosters.get(team, {}).values()):
                    found = (pid, team)
        if found is None:
            unmatched.append({**{k: r[k] for k in ("market", "player_name", "home_team", "away_team", "fd_market_id")}})
            ids.append(None)
            teams.append(r["team"])
        else:
            ids.append(found[0])
            teams.append(found[1])
    return df.with_columns(pl.Series("player_id", ids, dtype=pl.String), pl.Series("team", teams, dtype=pl.String)), unmatched


def write_unmatched(unmatched: list[dict], game_unmatched: pl.DataFrame, out_dir: Path) -> None:
    rows = [{"kind": "player", **u} for u in unmatched]
    for r in game_unmatched.select("event_name", "event_date").unique().iter_rows(named=True):
        rows.append({"kind": "game", "market": None, "player_name": None, "home_team": r["event_name"], "away_team": r["event_date"], "fd_market_id": None})
    pl.DataFrame(rows, infer_schema_length=None).write_csv(out_dir / "unmatched.csv") if rows else (out_dir / "unmatched.csv").write_text("kind,market,player_name,home_team,away_team,fd_market_id\n")
    typer.echo(f"unmatched: {len(unmatched)} player selections, {game_unmatched['fd_event_id'].n_unique() if game_unmatched.height else 0} events -> unmatched.csv")
