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
) -> None:
    """Simulate one matchup with the reference engine and print a summary (Phase 1)."""
    from nflsim.sim.runner import sim_matchup_summary

    sim_matchup_summary(home, away, n, seed=seed if seed is not None else settings()["seed"], playoff=playoff, neutral=neutral)


@app.command()
def train() -> None:
    """Train all decision models (Phase 2)."""
    _not_implemented("2")


@app.command()
def features(season: int = typer.Option(...), week: int = typer.Option(...)) -> None:
    """Build team and player inputs for a week (Phases 3-4)."""
    _not_implemented("3-4")


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
