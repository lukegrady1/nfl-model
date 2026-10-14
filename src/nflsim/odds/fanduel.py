"""Phase 6, step 6.1: FanDuel odds client and normalizer.

The repo had no FanDuel client before this phase (see DECISIONS), so this one is written here around the
content-managed-page endpoint Luke supplied (NFL page: events + main markets) and the event-page endpoint with its
tab parameter (props). Read-only: it never logs in or places anything. Every pull is saved to
data/outputs/<season>_wk<week>/fanduel_<timestamp>.parquet so line movement can be studied later.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import time
from pathlib import Path

import polars as pl
import requests
import typer

from nflsim.config import OUTPUTS_DIR, settings, team_aliases

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
COLUMNS = ["pulled_at", "game_id", "market", "selection", "team", "player_id", "player_name", "line", "american_odds",
           "one_sided", "fd_event_id", "fd_market_id", "fd_selection_id", "fd_market_type", "fd_market_name", "event_name",
           "event_date", "home_name", "away_name"]
PLAYER_STAT = {"PASSING_YARDS": "player_pass_yds", "PASSING_TOUCHDOWNS": "player_pass_tds", "RECEIVING_YARDS": "player_rec_yds",
               "RECEPTIONS": "player_receptions", "RUSHING_YARDS": "player_rush_yds", "RUSHING_ATTEMPTS": "player_rush_att",
               "INTERCEPTIONS": "player_interceptions", "LONGEST_RECEPTION": "player_longest_rec", "LONGEST_RUSH": "player_longest_rush"}
PLAYER_RE = re.compile(r"^PLAYER_X_(ALT_)?([A-Z_+]+?)_(HIGH|MEDIUM|LOW)$")
OU_NAME_RE = re.compile(r"\b(Over|Under)\s*\(?\s*(-?\d+(?:\.\d+)?)\s*\)?", re.I)
PAREN_LINE_RE = re.compile(r"^(?P<name>.+?)\s*\((?P<line>[+-]?\d+(?:\.\d+)?)\)\s*$")
ALT_RUNNER_RE = re.compile(r"^(?P<name>.+?)\s+(?P<n>\d+)\+\s+(?P<unit>Yards|Receptions|Touchdowns|TDs|Rushing Attempts|Attempts)$", re.I)


class FanDuelClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, delay: float | None = None):
        cfg = settings()["fanduel"]
        self.base_url = base_url or cfg["base_url"]
        self.api_key = api_key or cfg["api_key"]
        self.delay = cfg["request_delay_s"] if delay is None else delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": UA, "Accept": "application/json"})

    def get(self, path: str, **params) -> dict:
        params["_ak"] = self.api_key
        r = self.session.get(f"{self.base_url}/{path}", params=params, timeout=30)
        r.raise_for_status()
        time.sleep(self.delay)
        return r.json()

    def nfl_page(self) -> dict:
        return self.get("content-managed-page", page="CUSTOM", customPageId=settings()["fanduel"]["page_id"])

    def event_page(self, event_id: int, tab: str) -> dict:
        return self.get("event-page", eventId=event_id, tab=tab)


# ---- normalization -------------------------------------------------------------------------------------------------
def _odds(runner: dict) -> int | None:
    try:
        return int(runner["winRunnerOdds"]["americanDisplayOdds"]["americanOdds"])
    except (KeyError, TypeError, ValueError):
        return None


def _split_event(name: str) -> tuple[str, str] | None:
    if " @ " not in name:
        return None
    away, home = name.split(" @ ", 1)
    return away.strip(), home.strip()


def normalize_markets(markets: dict, events: dict, pulled_at: str) -> list[dict]:
    """FanDuel market/runner dicts -> normalized rows (team names still FanDuel's; ids unmatched)."""
    rows = []
    for m in markets.values():
        ev = events.get(str(m.get("eventId"))) or events.get(m.get("eventId"))
        if not ev:
            continue
        teams = _split_event(ev.get("name", ""))
        if not teams:
            continue
        away_name, home_name = teams
        if m.get("marketStatus") not in ("OPEN", None):
            continue
        mt = m.get("marketType", "")
        base = {"pulled_at": pulled_at, "game_id": None, "player_id": None, "fd_event_id": str(m.get("eventId")),
                "fd_market_id": str(m.get("marketId")), "fd_market_type": mt, "fd_market_name": m.get("marketName"),
                "event_name": ev.get("name"), "event_date": ev.get("openDate"), "home_name": home_name, "away_name": away_name,
                "one_sided": False, "player_name": None}
        for r in m.get("runners", []):
            if r.get("runnerStatus") not in ("ACTIVE", None):
                continue
            odds = _odds(r)
            if odds is None:
                continue
            rn = r.get("runnerName", "")
            hc = r.get("handicap")
            # alternate totals and team totals carry the line in the runner name ("Over 44.5") with handicap 0
            ou = OU_NAME_RE.search(rn)
            if ou and (hc is None or float(hc) == 0.0) and mt != "MONEY_LINE":
                hc = float(ou.group(2))
            row = {**base, "fd_selection_id": str(r.get("selectionId")), "american_odds": odds, "line": None, "team": None, "selection": None}
            if mt == "MONEY_LINE":
                row["market"] = "moneyline"
                row["selection"], row["team"] = ("home", home_name) if rn == home_name else ("away", away_name)
            elif mt in ("MATCH_HANDICAP_(2-WAY)", "ALTERNATE_HANDICAP"):
                pm = PAREN_LINE_RE.match(rn.strip())
                team_name = rn.strip()
                if pm and (hc is None or float(hc) == 0.0):     # "Seattle Seahawks (+23.5)"
                    team_name, hc = pm.group("name").strip(), float(pm.group("line"))
                if team_name not in (home_name, away_name):
                    continue
                row["market"] = "spread"
                row["selection"], row["team"] = ("home", home_name) if team_name == home_name else ("away", away_name)
                row["line"] = float(hc)
            elif mt in ("TOTAL_POINTS_(OVER/UNDER)", "ALTERNATE_TOTAL"):
                row["market"] = "total"
                row["selection"] = "over" if "over" in rn.lower() else "under"
                row["line"] = float(hc)
            elif mt in ("HOME_TEAM_TOTAL_POINTS", "HOME_TEAM_ALTERNATE_TOTAL", "AWAY_TEAM_TOTAL_POINTS", "AWAY_TEAM_ALTERNATE_TOTAL"):
                row["market"] = "team_total"
                row["team"] = home_name if mt.startswith("HOME") else away_name
                row["selection"] = "over" if rn.lower().startswith("over") or " over" in rn.lower() else "under"
                row["line"] = float(hc)
            elif mt == "ANY_TIME_TOUCHDOWN_SCORER":
                if rn.strip().endswith(("Defense", "D/ST")):
                    continue   # team defense runners have no player
                row["market"] = "player_anytime_td"
                row["selection"] = "yes"
                row["player_name"] = rn.strip()
                row["line"] = 0.5
                row["one_sided"] = True
            else:
                pm = PLAYER_RE.match(mt)
                if not pm:
                    continue
                alt, stat = pm.group(1), pm.group(2)
                market = PLAYER_STAT.get(stat)
                if market is None:
                    continue
                row["market"] = market
                if alt:
                    am = ALT_RUNNER_RE.match(rn.strip())
                    if not am:
                        continue
                    row["player_name"] = am.group("name").strip()
                    row["selection"] = "over"
                    row["line"] = float(am.group("n")) - 0.5
                    row["one_sided"] = True
                else:
                    low = rn.strip()
                    if low.endswith(" Over"):
                        row["player_name"], row["selection"] = low[:-5].strip(), "over"
                    elif low.endswith(" Under"):
                        row["player_name"], row["selection"] = low[:-6].strip(), "under"
                    else:
                        continue
                    row["line"] = float(hc)
            if row.get("market") and row.get("selection"):
                rows.append(row)
    return rows


def pull_week(season: int, week: int, client: FanDuelClient | None = None, save: bool = True, out_dir: Path | None = None,
              event_filter=None) -> pl.DataFrame:
    """Pull the NFL page, then each game's prop tabs, and save the normalized table."""
    client = client or FanDuelClient()
    pulled_at = dt.datetime.now().isoformat(timespec="seconds")
    page = client.nfl_page()
    events = page["attachments"].get("events", {})
    games = {k: e for k, e in events.items() if _split_event(e.get("name", "")) and (event_filter is None or event_filter(e))}
    typer.echo(f"fanduel: {len(games)} game events on the NFL page")
    rows = normalize_markets({k: m for k, m in page["attachments"].get("markets", {}).items() if str(m.get("eventId")) in games}, events, pulled_at)
    tabs = settings()["fanduel"]["tabs"]
    for eid, e in games.items():
        seen: set[str] = set()
        for tab in tabs:
            try:
                ep = client.event_page(int(eid), tab)
            except requests.RequestException as exc:
                typer.echo(f"  warning: {e['name']} tab {tab}: {exc}")
                continue
            mk = {k: m for k, m in ep.get("attachments", {}).get("markets", {}).items() if k not in seen}
            seen.update(mk)
            rows += normalize_markets(mk, {eid: e, **ep.get("attachments", {}).get("events", {})}, pulled_at)
    df = pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame(schema={c: pl.String for c in COLUMNS})
    df = df.unique(subset=["fd_market_id", "fd_selection_id", "selection", "line"])
    if save:
        out_dir = out_dir or (OUTPUTS_DIR / f"{season}_wk{week}")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"fanduel_{pulled_at.replace(':', '').replace('-', '')}.parquet"
        df.write_parquet(path)
        typer.echo(f"fanduel: {df.height} selections across {df['fd_event_id'].n_unique()} events -> {path.name}")
    return df


def latest_pull(season: int, week: int) -> pl.DataFrame | None:
    out_dir = OUTPUTS_DIR / f"{season}_wk{week}"
    files = sorted(out_dir.glob("fanduel_*.parquet"))
    return pl.read_parquet(files[-1]) if files else None
