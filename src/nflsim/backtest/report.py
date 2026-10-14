"""Phase 7, step 7.3: the backtest report (data/outputs/backtest/report.html)."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl
from jinja2 import Environment, FileSystemLoader, select_autoescape

from nflsim.backtest.metrics import season_metrics
from nflsim.backtest.walkforward import BT_DIR
from nflsim.config import OUTPUTS_DIR

TEMPLATES = Path(__file__).resolve().parents[1] / "report" / "templates"
RULES = [
    "The model is not expected to beat the closing line overall. Being close to it (log loss within about 0.01, margin MAE within about 0.3 points) means it is a sound engine.",
    "Edges are only worth studying in a market or situation where the backtest shows a consistent, sizable, well-sampled advantage.",
    "Never tune the model on the test season. Any change made after looking at backtest results must be re-tested on a season not used to make that change.",
    "Report bad results as clearly as good ones.",
]


def bar_svg(values: list[float], labels: list[str], width: int = 420, height: int = 120, ref: float | None = None, color: str = "#4c78a8") -> str:
    n = len(values)
    mx = max(values + [ref or 0]) or 1
    bw = width / n
    parts = []
    for i, v in enumerate(values):
        h = height * v / mx
        parts.append(f'<rect x="{i * bw + 2:.1f}" y="{height - h:.1f}" width="{bw - 4:.1f}" height="{h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{i * bw + bw / 2:.1f}" y="{height + 12}" font-size="9" text-anchor="middle" fill="#555">{labels[i]}</text>')
    if ref is not None:
        y = height - height * ref / mx
        parts.append(f'<line x1="0" y1="{y:.1f}" x2="{width}" y2="{y:.1f}" stroke="#e45756" stroke-dasharray="4 3"/>')
    return f'<svg width="{width}" height="{height + 16}" viewBox="0 0 {width} {height + 16}">' + "".join(parts) + "</svg>"


def reliability_svg(rel: list[dict], size: int = 220) -> str:
    parts = [f'<line x1="0" y1="{size}" x2="{size}" y2="0" stroke="#aaa" stroke-dasharray="4 3"/>']
    for r in rel:
        if r["pred"] is None:
            continue
        x, y = r["pred"] * size, size - r["obs"] * size
        rad = 3 + min(r["n"], 200) / 25
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rad:.1f}" fill="#4c78a8" fill-opacity="0.7"><title>n={r["n"]} pred={r["pred"]:.2f} obs={r["obs"]:.2f}</title></circle>')
    return f'<svg width="{size + 30}" height="{size + 30}" viewBox="-5 -5 {size + 30} {size + 30}"><rect x="0" y="0" width="{size}" height="{size}" fill="none" stroke="#ccc"/>' + "".join(parts) + \
        f'<text x="{size / 2}" y="{size + 18}" font-size="10" text-anchor="middle" fill="#555">predicted P(home win)</text></svg>'


def summary_paragraph(all_m: dict, per_season: dict) -> str:
    wp, er = all_m["win_prob"], all_m["errors"]
    dll = wp["log_loss_model"] - wp["log_loss_market"]
    dmae = er["margin_mae_model"] - er["margin_mae_market"]
    dtot = er["total_mae_model"] - er["total_mae_market"]
    close_wp = abs(dll) <= 0.01
    close_m = abs(dmae) <= 0.3
    verdict = []
    verdict.append(f"On win probability the model is {'close to' if close_wp else 'worse than' if dll > 0 else 'better than'} the closing moneyline "
                   f"(log loss {wp['log_loss_model']:.4f} vs {wp['log_loss_market']:.4f}, a gap of {dll:+.4f}).")
    verdict.append(f"On spreads it is {'close to' if close_m else 'worse than' if dmae > 0 else 'better than'} the closing line "
                   f"(margin MAE {er['margin_mae_model']:.2f} vs {er['margin_mae_market']:.2f}, {dmae:+.2f} points).")
    verdict.append(f"On totals it is {'close to' if abs(dtot) <= 0.3 else 'worse than' if dtot > 0 else 'better than'} the market "
                   f"(total MAE {er['total_mae_model']:.2f} vs {er['total_mae_market']:.2f}, {dtot:+.2f}; the model's totals run {er['total_bias_model']:+.1f} points vs actual, the market's {er['total_bias_market']:+.1f}).")
    best = None
    for kind, rows in (("spreads", all_m["ats"]), ("totals", all_m["ou"])):
        for r in rows:
            if r["win_rate"] is not None and r["ci_low"] > 0.524:
                best = (kind, r)
    if best:
        kind, r = best
        verdict.append(f"The only disagreement bucket whose 95% interval clears the 52.4% break-even is {kind} at >= {r['threshold']} points "
                       f"({r['wins']}-{r['losses']}, {r['win_rate']:.1%}, CI {r['ci_low']:.1%}-{r['ci_high']:.1%}, {r['bets']} bets); treat it as a lead to re-test, not a result.")
    else:
        verdict.append("No disagreement bucket, for spreads or totals, has a 95% interval that clears the 52.4% break-even, so the backtest shows no exploitable edge yet.")
    return " ".join(verdict)


def build_backtest_report(seasons: list[int] | None = None, variant: str = "played") -> Path:
    seasons = seasons or sorted(int(p.name) for p in BT_DIR.iterdir() if p.is_dir() and p.name.isdigit())
    frames, pframes, per_season = [], [], {}
    for S in seasons:
        d = BT_DIR / str(S)
        if not (d / f"games_{variant}.parquet").exists():
            continue
        g = pl.read_parquet(d / f"games_{variant}.parquet")
        p = pl.read_parquet(d / "players_played.parquet") if (d / "players_played.parquet").exists() else None
        frames.append(g); pframes.append(p) if p is not None else None
        per_season[S] = season_metrics(g, p)
    g_all = pl.concat(frames)
    p_all = pl.concat(pframes) if pframes else None
    all_m = season_metrics(g_all, p_all)
    other = {}
    alt = "thursday" if variant == "played" else "played"
    alt_frames = [pl.read_parquet(BT_DIR / str(S) / f"games_{alt}.parquet") for S in seasons if (BT_DIR / str(S) / f"games_{alt}.parquet").exists()]
    if alt_frames:
        other = season_metrics(pl.concat(alt_frames), None)
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=select_autoescape(["html"]))
    html = env.get_template("backtest.html.j2").render(
        rules=RULES, seasons=list(per_season), variant=variant, alt=alt, all_m=all_m, per_season=per_season, other=other,
        generated=dt.datetime.now().isoformat(timespec="minutes"), n_games=g_all.height,
        reliability_svg=reliability_svg(all_m["reliability"]),
        pit_margin_svg=bar_svg([h / all_m["pit_margin"]["n"] for h in all_m["pit_margin"]["hist"]], [f"{i / 10:.1f}" for i in range(10)], ref=0.1),
        pit_total_svg=bar_svg([h / all_m["pit_total"]["n"] for h in all_m["pit_total"]["hist"]], [f"{i / 10:.1f}" for i in range(10)], ref=0.1),
        prop_svgs={k: bar_svg([h / v["pit"]["n"] for h in v["pit"]["hist"]], [f"{i / 10:.1f}" for i in range(10)], ref=0.1, width=300, height=90)
                   for k, v in all_m["props"].items()},
        summary=summary_paragraph(all_m, per_season))
    BT_DIR.mkdir(parents=True, exist_ok=True)
    path = BT_DIR / "report.html"
    path.write_text(html)
    (BT_DIR / "metrics.json").write_text(json.dumps({"all": all_m, "per_season": {str(k): v for k, v in per_season.items()}, "thursday_variant": other}, indent=1, default=float))
    return path
