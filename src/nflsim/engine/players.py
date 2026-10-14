"""Phase 4, steps 4.2-4.3 at game time: a team's active players with renormalized shares, air-yard band fits and
efficiencies, plus the sampling of the target / ball carrier on each play."""
from __future__ import annotations

import bisect

import numpy as np
import polars as pl

from nflsim.config import settings
from nflsim.features.players import EFFICIENCIES, QB_EFF

BAND_EDGES = (5, 15)   # short < 5, intermediate 5-14, deep >= 15


def band_of(air: int) -> int:
    return 0 if air < BAND_EDGES[0] else (1 if air < BAND_EDGES[1] else 2)


class TeamPlayers:
    """Built once per game per team. `players` maps player_id -> dict(name, position, shares..., eff dict)."""

    def __init__(self, team: str, players: dict[str, dict], qb: str | None, kicker: str | None = None):
        self.team = team
        self.players = players
        self.qb = qb
        self.kicker = kicker
        self.names = {pid: p.get("name", pid) for pid, p in players.items()}
        self._build_tables()

    def _build_tables(self) -> None:
        ids = [pid for pid, p in self.players.items() if p.get("target_share", 0) > 0]
        self.target_ids = ids
        w = np.array([[self.players[pid]["target_share"] * self.players[pid]["bin_fit"][b] for pid in ids] for b in range(3)])
        rz = np.array([[0.5 * self.players[pid]["target_share"] * self.players[pid]["bin_fit"][b]
                        + 0.5 * self.players[pid]["rz_target_share"] * self.players[pid]["bin_fit"][b] for pid in ids] for b in range(3)])
        self.target_cum = [self._cum(w[b]) for b in range(3)]
        self.rz_target_cum = [self._cum(rz[b]) for b in range(3)]
        cids = [pid for pid, p in self.players.items() if p.get("carry_share", 0) > 0]
        self.carrier_ids = cids
        cw = np.array([self.players[pid]["carry_share"] for pid in cids])
        crz = np.array([0.5 * self.players[pid]["carry_share"] + 0.5 * self.players[pid]["rz_carry_share"] for pid in cids])
        self.carrier_cum = self._cum(cw)
        self.rz_carrier_cum = self._cum(crz)

    @staticmethod
    def _cum(w: np.ndarray) -> list[float]:
        if len(w) == 0 or w.sum() <= 0:
            return []
        c = np.cumsum(w / w.sum()).tolist()
        c[-1] = 1.0 + 1e-12
        return c

    def pick_target(self, rng: np.random.Generator, air: int, yardline_100: int) -> str | None:
        if not self.target_ids:
            return None
        cum = (self.rz_target_cum if yardline_100 <= 20 else self.target_cum)[band_of(air)]
        if not cum:
            return None
        return self.target_ids[min(bisect.bisect_right(cum, rng.random()), len(self.target_ids) - 1)]

    def pick_carrier(self, rng: np.random.Generator, yardline_100: int) -> str | None:
        if not self.carrier_ids:
            return None
        cum = self.rz_carrier_cum if yardline_100 <= 20 else self.carrier_cum
        if not cum:
            return None
        return self.carrier_ids[min(bisect.bisect_right(cum, rng.random()), len(self.carrier_ids) - 1)]

    def eff(self, pid: str | None) -> dict:
        if pid is None or pid not in self.players:
            return {}
        return self.players[pid].get("eff", {})

    def qb_eff(self) -> dict:
        return {k: v for k, v in self.eff(self.qb).items() if k in QB_EFF}

    def kicker_eff(self) -> dict:
        return {"kicker_fg_oe": self.eff(self.kicker).get("kicker_fg_oe", 0.0)}


def build_team_players(team: str, actives: list[dict], qb: str | None, usage: pl.DataFrame, eff: pl.DataFrame,
                       priors: pl.DataFrame, season: int, week: int, blend_log: list | None = None) -> TeamPlayers:
    """Shares as of (season, week) for the active players, renormalized to the actives (step 4.2)."""
    cfg = settings()["player_usage"]
    k_fit = float(cfg["bin_fit_pseudo_targets"])
    blend = float(cfg["prior_blend"])
    min_plays = float(cfg["min_team_plays"])
    u = {r["player_id"]: r for r in usage.filter((pl.col("season") == season) & (pl.col("week") == week) & (pl.col("team") == team)).iter_rows(named=True)}
    e = {r["player_id"]: r for r in eff.filter((pl.col("season") == season) & (pl.col("week") == week)).iter_rows(named=True)}
    pri = {(r["position"], int(r["slot"])): r for r in priors.iter_rows(named=True)}
    # depth slots among the actives by snap share
    by_pos: dict[str, list] = {}
    for a in actives:
        by_pos.setdefault(a["position"], []).append(a)
    slot_of: dict[str, int] = {}
    for pos, lst in by_pos.items():
        lst.sort(key=lambda a: -(u.get(a["player_id"], {}).get("snap_share") or 0.0))
        for i, a in enumerate(lst):
            slot_of[a["player_id"]] = min(i + 1, 5)
    players: dict[str, dict] = {}
    kicker = None
    for a in actives:
        pid, pos = a["player_id"], a["position"]
        row = u.get(pid, {})
        prior = pri.get((pos, slot_of[pid]), {})
        thin = (row.get("team_plays") or 0.0) < min_plays
        def get(name: str) -> float:
            own = row.get(name)
            if thin or own is None:
                return float(prior.get(name, 0.0) or 0.0)
            return float(own)
        p = {"name": a.get("name", pid), "position": pos, "slot": slot_of[pid],
             "target_share": get("target_share"), "carry_share": get("carry_share"),
             "rz_target_share": get("rz_target_share"), "rz_carry_share": get("rz_carry_share"),
             "prior_target_share": float(prior.get("target_share", 0.0) or 0.0), "prior_carry_share": float(prior.get("carry_share", 0.0) or 0.0),
             "thin": thin}
        # air-yard band fit: share of the team's targets in the band relative to the overall share, shrunk toward 1
        fits = []
        for band_num, band_den in (("ew_short_targets", "ew_team_short_targets"), ("ew_int_targets", "ew_team_int_targets"), ("ew_deep_targets", "ew_team_deep_targets")):
            n_b, N_b, s = row.get(band_num) or 0.0, row.get(band_den) or 0.0, p["target_share"]
            if thin or s <= 0 or N_b <= 0:
                fits.append(1.0)
            else:
                share_b = (n_b + k_fit * s) / (N_b + k_fit)
                fits.append(max(share_b / s, 0.05))
        p["bin_fit"] = fits
        er = e.get(pid, {})
        p["eff"] = {name: float(er.get(name) or 0.0) for name in EFFICIENCIES}
        if pos == "QB":
            p["target_share"] = p["rz_target_share"] = 0.0   # QBs are passers, not targets; designed QB runs stay via carry_share
        players[pid] = p
        if pos == "K" and kicker is None:
            kicker = pid
    # renormalize: the missing share (players not active) is redistributed in proportion to a blend of own share and prior
    for share_key, prior_key, eligible in (("target_share", "prior_target_share", lambda q: q["position"] != "K"),
                                           ("carry_share", "prior_carry_share", lambda q: q["position"] != "K"),
                                           ("rz_target_share", "prior_target_share", lambda q: q["position"] != "K"),
                                           ("rz_carry_share", "prior_carry_share", lambda q: q["position"] != "K")):
        pool = [q for q in players.values() if eligible(q)]
        total = sum(q[share_key] for q in pool)
        if total <= 0:
            continue
        missing = 1.0 - total
        if missing > 0:
            weights = [blend * q[share_key] + (1 - blend) * q[prior_key] for q in pool]
            wsum = sum(weights)
            for q, w in zip(pool, weights):
                q[share_key] += missing * (w / wsum if wsum > 0 else 1 / len(pool))
        else:
            for q in pool:
                q[share_key] /= total
    return TeamPlayers(team, players, qb, kicker)
