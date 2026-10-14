"""Phase 6, step 6.5: one self-contained HTML report per week."""
from __future__ import annotations

import base64
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
from jinja2 import Environment, FileSystemLoader, select_autoescape

from nflsim.config import OUTPUTS_DIR, ROOT, settings
from nflsim.sim.aggregate import american

TEMPLATES = Path(__file__).parent / "templates"
GAME_MARKETS = ["moneyline", "spread", "total", "team_total"]


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def fair_spread(fair_game: pl.DataFrame) -> float | None:
    """The home line where the cover probability crosses 50% (linear interpolation on half-point lines)."""
    sp = fair_game.filter((pl.col("market") == "spread") & (pl.col("selection") == "home")).sort("line")
    if sp.height == 0:
        return None
    lines, p = sp["line"].to_numpy(), sp["p_no_push"].to_numpy()
    idx = np.where(p >= 0.5)[0]
    if len(idx) == 0:
        return float(lines[-1])
    i = idx[0]
    if i == 0:
        return float(lines[0])
    x0, x1, y0, y1 = lines[i - 1], lines[i], p[i - 1], p[i]
    return float(x0 + (0.5 - y0) / (y1 - y0) * (x1 - x0)) if y1 != y0 else float(x1)


def histogram_svg(values: np.ndarray, marker: float | None, lo: int, hi: int, width: int = 320, height: int = 90, label: str = "") -> str:
    bins = np.arange(lo, hi + 2) - 0.5
    counts, _ = np.histogram(values, bins=bins)
    mx = counts.max() if counts.max() > 0 else 1
    n = len(counts)
    bw = width / n
    bars = []
    for i, c in enumerate(counts):
        h = height * c / mx
        bars.append(f'<rect x="{i * bw:.1f}" y="{height - h:.1f}" width="{bw:.2f}" height="{h:.1f}" fill="#4c78a8"/>')
    mk = ""
    if marker is not None and lo <= marker <= hi:
        x = (marker - lo + 0.5) * bw
        mk = f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{height}" stroke="#e45756" stroke-width="2"/>'
    return (f'<svg width="{width}" height="{height + 14}" viewBox="0 0 {width} {height + 14}">' + "".join(bars) + mk +
            f'<text x="2" y="{height + 12}" font-size="10" fill="#555">{label}</text></svg>')


def build_report(season: int, week: int, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or (OUTPUTS_DIR / f"{season}_wk{week}")
    fair = pl.read_parquet(out_dir / "fair_odds.parquet")
    props = pl.read_parquet(out_dir / "player_props.parquet") if (out_dir / "player_props.parquet").exists() else None
    comp = pl.read_parquet(out_dir / "comparison.parquet") if (out_dir / "comparison.parquet").exists() else pl.DataFrame()
    run = json.loads((out_dir / "run.json").read_text()) if (out_dir / "run.json").exists() else {}
    unmatched = 0
    if (out_dir / "unmatched.csv").exists():
        unmatched = max(pl.read_csv(out_dir / "unmatched.csv").height, 0)
    pulled_at = comp["pulled_at"][0] if comp.height else None
    sched = pl.read_parquet(ROOT / "data/processed/schedules.parquet").filter((pl.col("season") == season) & (pl.col("week") == week))
    # ---- games table
    games = []
    for g in sched.filter(pl.col("game_id").is_in(fair["game_id"].unique().to_list())).sort("gameday", "gametime").iter_rows(named=True):
        gid = g["game_id"]
        fg = fair.filter(pl.col("game_id") == gid)
        ml = fg.filter((pl.col("market") == "moneyline") & (pl.col("selection") == "home"))
        p_home = float(ml["p_no_push"][0]) if ml.height else None
        tot = fg.filter((pl.col("market") == "total") & (pl.col("selection") == "over")).sort("line")
        fair_tot = None
        if tot.height:
            lines, p = tot["line"].to_numpy(), tot["p_no_push"].to_numpy()
            idx = np.where(p <= 0.5)[0]
            fair_tot = float(lines[idx[0]] - 0.5) if len(idx) else float(lines[-1])
        cg = comp.filter(pl.col("game_id") == gid) if comp.height else pl.DataFrame()
        def best(market, selection):
            if not cg.height:
                return None
            x = cg.filter((pl.col("market") == market) & (pl.col("selection") == selection) & ~pl.col("one_sided"))
            if x.height == 0:
                return None
            # main line = the pair closest to -110/-110
            x = x.with_columns((pl.col("american_odds") + 110).abs().alias("_d")).sort("_d")
            r = x.row(0, named=True)
            return {"line": r["line"], "odds": r["american_odds"], "edge": r["edge"], "p_market": r["p_market"]}
        fd_sp, fd_tot, fd_ml = best("spread", "home"), best("total", "over"), best("moneyline", "home")
        npz = out_dir / f"sims_{gid}.npz"
        svg_m = svg_t = ""
        if npz.exists():
            z = np.load(npz)
            m = z["home_score"] - z["away_score"]
            t = z["home_score"] + z["away_score"]
            svg_m = histogram_svg(m, -fd_sp["line"] if fd_sp else None, -35, 35, label=f"margin (home - away){'; red = FanDuel spread' if fd_sp else ''}")
            svg_t = histogram_svg(t, fd_tot["line"] if fd_tot else None, 15, 80, label=f"total{'; red = FanDuel total' if fd_tot else ''}")
        games.append({"game_id": gid, "home": g["home_team"], "away": g["away_team"], "kickoff": f"{g['gameday']} {g.get('gametime') or ''}",
                      "p_home": p_home, "fair_spread": fair_spread(fg), "fair_total": fair_tot, "fair_ml_home": american(p_home) if p_home else None,
                      "fd_spread": fd_sp, "fd_total": fd_tot, "fd_ml": fd_ml, "svg_margin": svg_m, "svg_total": svg_t})
    # ---- disagreements table (all comparison rows)
    rows = comp.sort("ev_per_dollar", descending=True).to_dicts() if comp.height else []
    for r in rows:
        r["market_kind"] = "game" if r["market"] in GAME_MARKETS else "prop"
    # ---- props tab grouped by game
    prop_rows = [r for r in rows if r["market_kind"] == "prop"]
    props_by_game: dict[str, list] = {}
    for r in prop_rows:
        props_by_game.setdefault(r["game_id"], []).append(r)
    top = [r for r in rows if "CHECK_INPUTS" not in r["flag"] and "NOISE" not in r["flag"]][:15]
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=select_autoescape(["html"]))
    html = env.get_template("week.html.j2").render(
        season=season, week=week, pulled_at=pulled_at, sims=run.get("sims"), version=git_hash(), unmatched=unmatched,
        games=games, rows=rows, props_by_game=props_by_game, top=top, n_rows=len(rows), qb_blends=run.get("qb_blends", []),
        markets=sorted({r["market"] for r in rows}), flags=sorted({f for r in rows for f in r["flag"].split(",") if f}),
        game_ids=[g["game_id"] for g in games])
    path = out_dir / "report.html"
    path.write_text(html)
    return path
