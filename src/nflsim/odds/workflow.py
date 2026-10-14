"""Phase 6, step 6.6: the weekly pipeline and the compare step."""
from __future__ import annotations

import polars as pl
import typer

from nflsim.config import OUTPUTS_DIR, settings
from nflsim.odds.compare import compare, match_rates
from nflsim.odds.devig import devig_frame
from nflsim.odds.fanduel import latest_pull, pull_week
from nflsim.odds.matching import match_games, match_players, write_unmatched


def compare_week(season: int, week: int, pull: bool = True) -> pl.DataFrame:
    from nflsim.features.actives import build_actives, load_actives
    from nflsim.features.players import load_player_eff

    out_dir = OUTPUTS_DIR / f"{season}_wk{week}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fair_path = out_dir / "fair_odds.parquet"
    if not fair_path.exists():
        raise typer.BadParameter(f"{fair_path} missing: run `nflsim simulate --season {season} --week {week}` first")
    odds = pull_week(season, week) if pull else latest_pull(season, week)
    if odds is None or odds.height == 0:
        raise typer.BadParameter("no FanDuel odds available")
    odds = match_games(odds, season, week)
    try:
        actives = load_actives(season, week)
    except FileNotFoundError:
        eff = load_player_eff()
        actives = build_actives(season, week, eff.filter((pl.col("season") == season) & (pl.col("week") == week)), quiet=True)
    odds, unmatched_players = match_players(odds, actives)
    write_unmatched(unmatched_players, odds.filter(pl.col("game_id").is_null()), out_dir)
    odds = devig_frame(odds)
    fair = pl.read_parquet(fair_path)
    props = pl.read_parquet(out_dir / "player_props.parquet") if (out_dir / "player_props.parquet").exists() else None
    comp = compare(odds, fair, props, actives, out_dir)
    rates = match_rates(odds, comp)
    typer.echo(f"matched {rates['game_matched']}/{rates['game_markets']} game selections ({rates['game_rate']:.1%}) and "
               f"{rates['props_matched']}/{rates['props']} player prop selections ({rates['prop_rate']:.1%}) -> comparison.parquet ({comp.height} rows)")
    (out_dir / "match_rates.json").write_text(pl.DataFrame([rates]).write_json())
    return comp


def run_week(season: int, week: int, sims: int | None = None, pull: bool = True) -> None:
    from nflsim.data.ingest import current_season, ingest_all
    from nflsim.features.actives import build_actives
    from nflsim.features.players import build_player_tables, build_position_priors
    from nflsim.features.team_strength import build_team_ratings
    from nflsim.report.dashboard import build_report
    from nflsim.sim.runner import simulate_week

    typer.echo(f"== ingest (refresh {current_season()})")
    ingest_all([current_season()])
    from nflsim.data.clean import build_plays
    from nflsim.data.validate import run_validation
    build_plays(list(range(2018, current_season() + 1)))
    typer.echo("== features")
    build_team_ratings()
    usage, eff = build_player_tables()
    build_position_priors(usage)
    build_actives(season, week, eff.filter((pl.col("season") == season) & (pl.col("week") == week)))
    typer.echo("== simulate")
    simulate_week(season, week, sims)
    typer.echo("== compare")
    comp = compare_week(season, week, pull=pull)
    typer.echo("== report")
    path = build_report(season, week)
    typer.echo(f"report: {path}")
    if comp.height:
        top = comp.filter(~pl.col("flag").str.contains("CHECK_INPUTS") & ~pl.col("flag").str.contains("NOISE")).head(15)
        typer.echo("top 15 by EV per dollar without CHECK_INPUTS or NOISE flags:")
        for r in top.iter_rows(named=True):
            who = r["player_name"] or r["team"] or ""
            typer.echo(f"  {r['game_id']:18s} {r['market']:20s} {who:22s} {r['selection']:6s} {str(r['line']):6s} {r['american_odds']:>5d}  "
                       f"model {r['p_model']:.3f} market {r['p_market']:.3f} edge {r['edge']*100:+.1f} EV {r['ev_per_dollar']:+.3f} {r['flag']}")
