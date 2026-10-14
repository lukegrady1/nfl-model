"""Step 3.4: manual weather input for upcoming games: data/inputs/weather_<season>_wk<week>.csv
with columns game_id, temp, wind, roof. Missing file = neutral weather with a warning."""
from __future__ import annotations

import polars as pl
import typer

from nflsim.config import INPUTS_DIR


def load_weather(season: int, week: int) -> dict[str, dict]:
    path = INPUTS_DIR / f"weather_{season}_wk{week}.csv"
    if not path.exists():
        typer.echo(f"warning: no weather file {path.name}; assuming neutral weather (outdoors, {8.0} mph wind) for every game")
        return {}
    df = pl.read_csv(path)
    out = {}
    for r in df.iter_rows(named=True):
        out[r["game_id"]] = {"roof_dome": 1 if str(r.get("roof", "")).lower() in ("dome", "closed") else 0,
                             "wind": float(r["wind"]) if r.get("wind") is not None else None, "temp": r.get("temp")}
    return out
