"""Phase 6, step 6.4: line up FanDuel's no-vig probabilities with the model's fair probabilities at the exact line."""
from __future__ import annotations

from pathlib import Path

import polars as pl
import typer

from nflsim.config import OUTPUTS_DIR, settings
from nflsim.odds.devig import decimal
from nflsim.sim.aggregate import american

PROP_STAT = {"player_pass_yds": "pass_yards", "player_pass_tds": "pass_tds", "player_rec_yds": "rec_yards", "player_receptions": "receptions",
             "player_rush_yds": "rush_yards", "player_rush_att": "rush_attempts", "player_interceptions": "interceptions",
             "player_longest_rec": "longest_rec", "player_longest_rush": "longest_rush", "player_anytime_td": "anytime_td"}


def model_lookup(fair: pl.DataFrame, props: pl.DataFrame | None) -> tuple[dict, dict]:
    """(game_id, market, selection, team, line) -> (p_no_push, p_push, se); props: (game_id, player_id, stat, line) -> (p_over, se)."""
    games: dict = {}
    for r in fair.iter_rows(named=True):
        games[(r["game_id"], r["market"], r["selection"], r["team"], r["line"])] = (r["p_no_push"], r["p_push"], r["se"])
    plays: dict = {}
    if props is not None and props.height:
        for r in props.iter_rows(named=True):
            plays[(r["game_id"], r["player_id"], r["stat"], r["line"])] = (r["p_over"], r["se"], r["q50"], r["mean"])
    return games, plays


def compare(odds: pl.DataFrame, fair: pl.DataFrame, props: pl.DataFrame | None, actives: dict, out_dir: Path | None = None) -> pl.DataFrame:
    s = settings()
    flag_thr = float(s["edge_flag_threshold"])
    games, plays = model_lookup(fair, props)
    questionable: dict[str, set] = {}   # team -> set of (player_id, position) Questionable
    pos_of: dict[str, str] = {}
    for team, v in actives.items():
        for p in v["players"]:
            pos_of[p["player_id"]] = p["position"]
            if p.get("report_status") == "Questionable":
                questionable.setdefault(team, set()).add((p["player_id"], p["position"]))
    rows = []
    for r in odds.iter_rows(named=True):
        if r["game_id"] is None:
            continue
        m = r["market"]
        p_model = p_push = mc_se = None
        median = mean = None
        if m in ("moneyline", "spread", "total", "team_total"):
            team = r["team"] if m in ("moneyline", "spread", "team_total") else None
            hit = games.get((r["game_id"], m, r["selection"], team, r["line"]))
            if hit is None:
                continue
            p_model, p_push, mc_se = hit
        elif m in PROP_STAT:
            if r["player_id"] is None:
                continue
            hit = plays.get((r["game_id"], r["player_id"], PROP_STAT[m], r["line"]))
            if hit is None:
                continue
            p_over, mc_se, median, mean = hit
            p_model = p_over if r["selection"] in ("over", "yes") else 1.0 - p_over
            p_push = 0.0
        else:
            continue
        p_market = r["p_market"]
        edge = p_model - p_market
        dec = decimal(r["american_odds"])
        ev = p_model * (dec - 1.0) - (1.0 - p_model)
        flags = []
        if abs(edge) > flag_thr:
            flags.append("CHECK_INPUTS")
        if mc_se is not None and abs(edge) < 2 * mc_se:
            flags.append("NOISE")
        if r["player_id"] is not None:
            pos = pos_of.get(r["player_id"])
            qset = questionable.get(r["team"], set())
            if any(pid == r["player_id"] or (ppos == pos) for pid, ppos in qset):
                flags.append("QUESTIONABLE_PLAYER")
        if r["one_sided"]:
            flags.append("ONE_SIDED")
        rows.append({**{k: r[k] for k in ("game_id", "market", "selection", "team", "player_id", "player_name", "line", "american_odds",
                                          "fd_market_id", "fd_selection_id", "pulled_at", "hold", "p_market_power", "one_sided")},
                     "p_model": p_model, "p_push": p_push, "p_market": p_market, "edge": edge, "decimal_odds": dec, "ev_per_dollar": ev,
                     "model_fair_american": american(p_model), "mc_se": mc_se, "model_median": median, "model_mean": mean,
                     "flag": ",".join(flags)})
    out = pl.DataFrame(rows, infer_schema_length=None).sort("ev_per_dollar", descending=True) if rows else pl.DataFrame()
    if out_dir is not None and out.height:
        out.write_parquet(out_dir / "comparison.parquet")
    return out


def match_rates(odds: pl.DataFrame, comparison: pl.DataFrame) -> dict:
    """Share of FanDuel game markets and player props that found a model probability at the exact line."""
    game_mk = odds.filter(pl.col("market").is_in(["moneyline", "spread", "total", "team_total"]) & pl.col("game_id").is_not_null())
    prop_mk = odds.filter(pl.col("market").str.starts_with("player_") & pl.col("game_id").is_not_null())
    cg = comparison.filter(pl.col("market").is_in(["moneyline", "spread", "total", "team_total"])).height if comparison.height else 0
    cp = comparison.filter(pl.col("market").str.starts_with("player_")).height if comparison.height else 0
    return {"game_markets": game_mk.height, "game_matched": cg, "game_rate": cg / max(game_mk.height, 1),
            "props": prop_mk.height, "props_matched": cp, "prop_rate": cp / max(prop_mk.height, 1)}
