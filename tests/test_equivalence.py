"""Phase 5 equivalence test: reference engine vs fast engine on 5 matchups, 20,000 sims each (slow; ~10 min)."""
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest
from scipy.stats import ks_2samp

from tests.conftest import needs_data, needs_models

pytestmark = [needs_data, needs_models, pytest.mark.slow]
MATCHUPS = [(2025, 11, "BUF", "TB"), (2025, 11, "DEN", "KC"), (2025, 12, "HOU", "BUF"), (2025, 12, "KC", "IND"), (2025, 9, "LA", "NO")]
N = 20000
_M = None


def _ref_chunk(args):
    global _M
    season, week, home, away, n, seed = args
    from nflsim.engine.game import simulate_game
    from nflsim.models.registry import default_models
    from nflsim.sim.game_inputs import WeekInputs

    if _M is None:
        _M = default_models()
    ctx = WeekInputs(season, week).context(home, away)
    rng = np.random.default_rng(seed)
    out = []
    tp_h, tp_a = ctx.players[home], ctx.players[away]
    for _ in range(n):
        r = simulate_game(home, away, _M, rng, context=ctx)
        qb = r.players[home].get(tp_h.qb, {})
        aqb = r.players[away].get(tp_a.qb, {})
        top_wr = max(tp_h.target_ids, key=lambda p: tp_h.players[p]["target_share"])
        top_rb = max(tp_h.carrier_ids, key=lambda p: tp_h.players[p]["carry_share"])
        wr, rb = r.players[home].get(top_wr, {}), r.players[home].get(top_rb, {})
        out.append((r.home_score, r.away_score, qb.get("pass_yards", 0), qb.get("pass_attempts", 0), aqb.get("pass_yards", 0),
                    wr.get("rec_yards", 0), rb.get("rush_yards", 0)))
    return np.array(out, dtype=float)


def run_matchup(season, week, home, away, workers=10):
    from nflsim.models.registry import default_models
    from nflsim.sim.fast_engine import FastModels, simulate_games
    from nflsim.sim.game_inputs import WeekInputs
    from nflsim.sim.pack import pack_game, pack_models

    wi = WeekInputs(season, week)
    ctx = wi.context(home, away)
    G = pack_game(ctx, home, away)
    FM = FastModels(pack_models(default_models()))
    t = time.perf_counter()
    h, a, ot, ts, ps = simulate_games(N, 123, FM, G)
    fast_secs = time.perf_counter() - t
    n_max = G.eff.shape[1]
    tp_h, tp_a = ctx.players[home], ctx.players[away]
    idx_h = {pid: i for i, pid in enumerate(G.player_ids[0])}
    idx_a = {pid: i for i, pid in enumerate(G.player_ids[1])}
    top_wr = max(tp_h.target_ids, key=lambda p: tp_h.players[p]["target_share"])
    top_rb = max(tp_h.carrier_ids, key=lambda p: tp_h.players[p]["carry_share"])
    fast = np.column_stack([h, a, ps[:, idx_h[tp_h.qb], 2], ps[:, idx_h[tp_h.qb], 0], ps[:, n_max + idx_a[tp_a.qb], 2],
                            ps[:, idx_h[top_wr], 10], ps[:, idx_h[top_rb], 6]]).astype(float)
    chunk = N // workers
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(_ref_chunk, [(season, week, home, away, chunk, 1000 + k) for k in range(workers)]))
    ref = np.concatenate(parts)
    names = ["home_score", "away_score", "margin", "total", "home_qb_pass_yds", "home_qb_att", "away_qb_pass_yds", "home_wr1_rec_yds", "home_rb1_rush_yds"]
    ref_s = np.column_stack([ref[:, 0], ref[:, 1], ref[:, 0] - ref[:, 1], ref[:, 0] + ref[:, 1], ref[:, 2], ref[:, 3], ref[:, 4], ref[:, 5], ref[:, 6]])
    fast_s = np.column_stack([fast[:, 0], fast[:, 1], fast[:, 0] - fast[:, 1], fast[:, 0] + fast[:, 1], fast[:, 2], fast[:, 3], fast[:, 4], fast[:, 5], fast[:, 6]])
    rows = []
    for j, nm in enumerate(names):
        d = fast_s[:, j].mean() - ref_s[:, j].mean()
        se = np.sqrt(fast_s[:, j].var() / len(fast_s) + ref_s[:, j].var() / len(ref_s))
        rows.append((nm, ref_s[:, j].mean(), fast_s[:, j].mean(), d, se, abs(d) <= 2 * se))
    ks_m = ks_2samp(ref_s[:, 2], fast_s[:, 2]).pvalue
    ks_t = ks_2samp(ref_s[:, 3], fast_s[:, 3]).pvalue
    return rows, ks_m, ks_t, fast_secs


def test_fast_engine_matches_reference_on_five_matchups():
    from nflsim.config import OUTPUTS_DIR

    lines = ["| Matchup | Stat | Reference mean | Fast mean | Diff | SE | Within 2 SE |", "|---|---|---|---|---|---|---|"]
    all_ok = True
    ks_lines = ["", "| Matchup | KS p (margin) | KS p (total) | Fast engine 20,000 sims |", "|---|---|---|---|"]
    for season, week, home, away in MATCHUPS:
        rows, ks_m, ks_t, secs = run_matchup(season, week, home, away)
        for nm, rm, fm, d, se, ok in rows:
            all_ok &= ok
            lines.append(f"| {away}@{home} wk{week} | {nm} | {rm:.3f} | {fm:.3f} | {d:+.3f} | {se:.3f} | {'yes' if ok else 'NO'} |")
        ks_lines.append(f"| {away}@{home} wk{week} | {ks_m:.3f} | {ks_t:.3f} | {secs:.1f}s |")
        all_ok &= ks_m > 0.01 and ks_t > 0.01
    (OUTPUTS_DIR / "phase5_equivalence.md").write_text("\n".join(lines + ks_lines) + "\n")
    assert all_ok, "\n".join(lines + ks_lines)
