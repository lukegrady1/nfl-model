"""Phase 7: leakage tests in backtest mode (artifacts for a test season use only earlier seasons)."""
import json
import os

import polars as pl
import pytest

from nflsim.backtest.walkforward import season_dirs
from tests.conftest import needs_data

S = 2023
d = season_dirs(S)
needs_backtest = pytest.mark.skipif(not (d["models"] / "backtest_meta.json").exists(), reason="backtest artifacts for 2023 not built")
pytestmark = [needs_data, needs_backtest]


def test_backtest_models_trained_on_earlier_seasons_only():
    meta = json.loads((d["models"] / "backtest_meta.json").read_text())
    assert max(meta["training_seasons"]) == S - 1 and max(meta["table_seasons"]) == S - 1
    for p in list(d["models"].glob("*.json")) + list((d["models"] / "situation").glob("*.json")):
        if p.name in ("backtest_meta.json", "report.json"):
            continue
        spec = json.loads(p.read_text())
        assert max(spec["training_seasons"]) <= S - 1, p
        assert spec["validation_season"] <= S - 1, p


def test_backtest_tables_built_from_earlier_seasons_only():
    meta = json.loads((d["tables"] / "tables.json").read_text())
    assert max(meta["table_seasons"]) <= S - 1


def test_backtest_ratings_are_as_of_and_use_backtest_models():
    """The backtest season's ratings table is rebuilt in that environment; its week-W rows use only games before W
    (same exact recompute as the Phase 3 test, on the backtest table)."""
    from nflsim.data.clean import load_plays
    from nflsim.features.team_strength import RATINGS

    r = pl.read_parquet(d["derived"] / "team_ratings.parquet")
    assert r.filter(pl.col("season") == S).height > 0
    sample = r.filter((pl.col("season") == S) & (pl.col("games_played") > 0)).sample(n=20, seed=1)
    env = {"NFLSIM_MODELS_DIR": str(d["models"]), "NFLSIM_TABLES_DIR": str(d["tables"]), "NFLSIM_DERIVED_DIR": str(d["derived"]),
           "NFLSIM_TRAINING_SEASONS": f"2018-{S - 1}", "NFLSIM_TABLE_SEASONS": f"2021-{S - 1}", "NFLSIM_KICKOFF_SEASONS": str(S - 1), "NFLSIM_ERA_MAX": str(S - 1)}
    import subprocess, sys, tempfile
    rows = sample.select("team", "season", "week", "asof_date", *[f"raw_{s}" for s in RATINGS]).to_dicts()
    code = f'''
import json, polars as pl
from nflsim.data.clean import load_plays
from nflsim.features.team_strength import game_stats, ratings_from_game_stats, RATINGS
rows = json.loads(open("{tempfile.gettempdir()}/bt_rows.json").read())
gs = game_stats(load_plays())
bad = []
for r in rows:
    rebuilt = ratings_from_game_stats(gs.filter(pl.col("game_date") < r["asof_date"]))
    row = rebuilt.filter((pl.col("team") == r["team"]) & (pl.col("season") == r["season"]) & (pl.col("week") == r["week"]))
    for s in RATINGS:
        if abs(r["raw_" + s] - row["raw_" + s][0]) > 1e-9: bad.append((r["team"], r["week"], s))
print("BAD", len(bad))
'''
    with open(f"{tempfile.gettempdir()}/bt_rows.json", "w") as f:
        json.dump(rows, f)
    out = subprocess.run([sys.executable, "-c", code], env={**os.environ, **env}, capture_output=True, text=True, cwd=str(d["models"].parents[2]))
    assert "BAD 0" in out.stdout, out.stdout[-500:] + out.stderr[-1500:]


def test_backtest_games_have_closing_lines_and_no_future_weeks():
    g = pl.read_parquet(d["out"] / "games_played.parquet") if (d["out"] / "games_played.parquet").exists() else None
    if g is None:
        pytest.skip("season not simulated yet")
    assert g["spread_line"].null_count() == 0 and g["home_moneyline"].null_count() < 5
    assert g["season"].unique().to_list() == [S]
