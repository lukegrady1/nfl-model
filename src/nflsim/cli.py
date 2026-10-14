"""nflsim command-line interface. Each phase fills in one command."""
from __future__ import annotations

import typer

from nflsim.config import parse_seasons, settings

app = typer.Typer(help="NFL play-by-play Monte Carlo simulator and fair-odds engine.", no_args_is_help=True)


def _not_implemented(phase: str) -> None:
    typer.echo(f"Not implemented yet (Phase {phase}).")
    raise typer.Exit(code=2)


@app.command()
def ingest(
    seasons: str = typer.Option("2018-2026", help="Season range, e.g. 2018-2026"),
    refresh: bool = typer.Option(False, help="Re-download seasons already cached"),
) -> None:
    """Download and cache raw nflverse data (Phase 0)."""
    from nflsim.data.ingest import ingest_all

    ingest_all(parse_seasons(seasons), refresh=refresh)


@app.command()
def clean(
    seasons: str = typer.Option("2018-2026", help="Season range to include in the processed play table"),
) -> None:
    """Build processed tables from the raw cache (Phase 0)."""
    from nflsim.data.clean import build_plays
    from nflsim.data.validate import run_validation

    build_plays(parse_seasons(seasons))
    run_validation()


@app.command("build-tables")
def build_tables() -> None:
    """Build league-average lookup tables (Phase 1)."""
    from nflsim.models.tables import build_all_tables

    build_all_tables()


@app.command("sim-game")
def sim_game(
    home: str = typer.Option("AVG", help="Home team abbreviation (AVG = league average)"),
    away: str = typer.Option("AVG", help="Away team abbreviation (AVG = league average)"),
    n: int = typer.Option(2000, help="Number of simulations"),
    seed: int | None = typer.Option(None, help="RNG seed (defaults to settings.yaml seed)"),
    playoff: bool = typer.Option(False, help="Use playoff overtime rules"),
    neutral: bool = typer.Option(False, help="Neutral site"),
    tables_only: bool = typer.Option(False, help="Use the Phase 1 lookup tables for every decision"),
) -> None:
    """Simulate one matchup with the reference engine and print a summary (Phase 1)."""
    from nflsim.sim.runner import sim_matchup_summary

    sim_matchup_summary(home, away, n, seed=seed if seed is not None else settings()["seed"], playoff=playoff,
                        neutral=neutral, tables_only=tables_only)


@app.command()
def train(
    models: str = typer.Option("", help="Comma-separated model names (default: all trainable)"),
    evaluate_only: bool = typer.Option(False, help="Rebuild the report from saved models without retraining"),
    situation_only: bool = typer.Option(False, help="Train without team ratings (the Phase 2 reference models)"),
) -> None:
    """Train all decision models and write artifacts/models/report.md (Phase 2)."""
    from nflsim.models.train import train_all

    train_all([m.strip() for m in models.split(",") if m.strip()] or None, evaluate_only=evaluate_only, team=not situation_only)


@app.command("swap-check")
def swap_check_cmd(
    n: int = typer.Option(2000, help="Simulations per step"),
    seed: int | None = typer.Option(None),
) -> None:
    """Phase 2: re-run the Phase 1 league-average check after swapping in each trained model."""
    from nflsim.sim.runner import swap_check

    typer.echo(swap_check(n, seed))


@app.command()
def features(season: int = typer.Option(...), week: int = typer.Option(...)) -> None:
    """Build team ratings, player usage/efficiency tables and the week's actives (Phases 3-4)."""
    from nflsim.features.actives import build_actives
    from nflsim.features.players import build_player_tables, build_position_priors
    from nflsim.features.team_strength import build_team_ratings, ratings_markdown

    df = build_team_ratings()
    typer.echo(ratings_markdown(season, week, df))
    usage, eff = build_player_tables()
    build_position_priors(usage)
    build_actives(season, week, eff.filter((eff["season"] == season) & (eff["week"] == week)))


@app.command("phase4-check")
def phase4_check(
    season: int = typer.Option(2025),
    weeks: str = typer.Option("4-18"),
    n: int = typer.Option(2000, help="Simulations per game"),
    min_games: int = typer.Option(8, help="Minimum game appearances per player for the calibration check"),
    workers: int | None = typer.Option(None),
) -> None:
    """Phase 4 acceptance: backfill a season with as-of player inputs and check player stat calibration."""
    import polars as pl

    from nflsim.config import OUTPUTS_DIR
    from nflsim.sim.season import phase3_metrics, phase4_metrics, simulate_season

    out = simulate_season(season, n, parse_seasons(weeks), workers=workers, with_players=True)
    m3, t3 = phase3_metrics(out)
    typer.echo(t3)
    players = pl.read_parquet(OUTPUTS_DIR / f"player_sim_{season}_REG_{n}_players.parquet")
    m4, t4 = phase4_metrics(players, min_games)
    typer.echo(t4)
    (OUTPUTS_DIR / f"phase4_check_{season}_{n}.md").write_text(t3 + "\n\n" + t4 + "\n")


@app.command("phase3-check")
def phase3_check(
    season: int = typer.Option(2025),
    n: int = typer.Option(2000, help="Simulations per game"),
    weeks: str = typer.Option("", help="Optional week range, e.g. 1-18"),
    workers: int | None = typer.Option(None),
) -> None:
    """Phase 3 acceptance: simulate a season week by week with as-of ratings and compare to results and the market."""
    from nflsim.config import OUTPUTS_DIR
    from nflsim.sim.season import phase3_metrics, simulate_season

    out = simulate_season(season, n, parse_seasons(weeks) if weeks else None, workers=workers)
    m, table = phase3_metrics(out)
    typer.echo(table)
    (OUTPUTS_DIR / f"phase3_check_{season}_{n}.md").write_text(table + "\n")


@app.command()
def simulate(season: int = typer.Option(...), week: int = typer.Option(...)) -> None:
    """Simulate all games in a week and write fair odds (Phase 5)."""
    _not_implemented("5")


@app.command()
def compare(season: int = typer.Option(...), week: int = typer.Option(...)) -> None:
    """Pull FanDuel odds and compute edges (Phase 6)."""
    _not_implemented("6")


@app.command()
def week(season: int = typer.Option(...), week: int = typer.Option(...)) -> None:
    """Run features, simulate, compare and report in one go (Phase 6)."""
    _not_implemented("6")


@app.command()
def backtest(seasons: str = typer.Option("2023-2025")) -> None:
    """Walk-forward evaluation vs closing lines (Phase 7)."""
    _not_implemented("7")


if __name__ == "__main__":
    app()
