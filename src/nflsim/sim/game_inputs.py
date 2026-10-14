"""Phase 4: assemble the per-game Context (team ratings, weather, actives, player shares and efficiencies)."""
from __future__ import annotations

import polars as pl
import typer

from nflsim.config import settings
from nflsim.engine.players import TeamPlayers, build_team_players
from nflsim.engine.state import Context
from nflsim.features.actives import build_actives, load_actives
from nflsim.features.players import QB_EFF, load_player_eff, load_player_usage, load_position_priors
from nflsim.features.team_strength import RATINGS, load_team_ratings

BACKUP_QB_PRIOR = {"pass_epa": -0.8, "cpoe": -0.6, "sack_rate": 0.4, "adot": 0.0}   # league z of non-primary QBs (see DECISIONS)


class WeekInputs:
    """Loads everything once for a (season, week) and builds game contexts."""

    def __init__(self, season: int, week: int, actives: dict | None = None, use_saved_actives: bool = False):
        self.season, self.week = season, week
        self.usage = load_player_usage().filter((pl.col("season") == season) & (pl.col("week") == week))
        self.eff = load_player_eff().filter((pl.col("season") == season) & (pl.col("week") == week))
        self.priors = load_position_priors()
        self.ratings = {r["team"]: {s: r[s] for s in RATINGS}
                        for r in load_team_ratings().filter((pl.col("season") == season) & (pl.col("week") == week)).iter_rows(named=True)}
        if actives is not None:
            self.actives = actives
        elif use_saved_actives:
            self.actives = load_actives(season, week)
        else:
            self.actives = build_actives(season, week, self.eff, write=False, quiet=True)
        self.blend_log: list[str] = []

    def team_players(self, team: str) -> TeamPlayers | None:
        a = self.actives.get(team)
        if not a:
            return None
        return build_team_players(team, a["players"], a["starting_qb"], self.usage, self.eff, self.priors, self.season, self.week)

    def team_ratings(self, team: str, tp: TeamPlayers | None) -> dict:
        """Step 4.4: if the starting QB is not the QB behind the team's recent passing, blend the passing ratings 50/50
        with a rating from the new QB's own history (or a backup prior under 100 dropbacks)."""
        r = dict(self.ratings.get(team, {s: 0.0 for s in RATINGS}))
        if tp is None or tp.qb is None:
            return r
        qbs = self.eff.filter((pl.col("team") == team) & (pl.col("position") == "QB")).select("player_id", "n_qb_cpoe", "qb_cpoe", "qb_sack_rate", "qb_adot")
        if qbs.height == 0:
            return r
        primary = qbs.sort("n_qb_cpoe", descending=True).row(0, named=True)
        if primary["player_id"] == tp.qb or primary["n_qb_cpoe"] <= 0:
            return r
        me = qbs.filter(pl.col("player_id") == tp.qb)
        me = me.row(0, named=True) if me.height else {"n_qb_cpoe": 0.0, "qb_cpoe": 0.0, "qb_sack_rate": 0.0, "qb_adot": 0.0}
        n = float(me["n_qb_cpoe"] or 0.0)
        if n >= 100:
            # map the QB's own shrunk over-expected rates to the team z scale (cpoe: 1 z ~ 2.5 points; sack: 1 z ~ 1.5 points)
            own = {"pass_epa": me["qb_cpoe"] / 0.025 * 0.6, "cpoe": me["qb_cpoe"] / 0.025, "sack_rate": me["qb_sack_rate"] / 0.015, "adot": me["qb_adot"] / 1.0}
            src = f"own history ({n:.0f} dropbacks)"
        else:
            own = dict(BACKUP_QB_PRIOR)
            src = f"backup prior ({n:.0f} dropbacks)"
        for k, v in own.items():
            r[k] = 0.5 * r[k] + 0.5 * v
        self.blend_log.append(f"{team}: starting QB {tp.names.get(tp.qb, tp.qb)} differs from primary passer; passing ratings blended 50/50 with {src}")
        return r

    def context(self, home: str, away: str, weather: dict | None = None) -> Context:
        tph, tpa = self.team_players(home), self.team_players(away)
        players = {t: tp for t, tp in ((home, tph), (away, tpa)) if tp is not None}
        return Context(home=self.team_ratings(home, tph), away=self.team_ratings(away, tpa), players=players,
                       weather=weather or {}, season=self.season)
