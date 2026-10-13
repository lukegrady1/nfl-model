"""Step 1.7: compact per-game result."""
from __future__ import annotations

from dataclasses import dataclass, field

TEAM_STATS = ["plays", "pass_attempts", "completions", "pass_yards", "sacks", "sack_yards", "interceptions",
              "rush_attempts", "rush_yards", "scrambles", "punts", "fg_attempts", "fg_made", "turnovers",
              "fumbles_lost", "time_of_possession", "first_downs", "drives", "penalties", "kneels",
              "touchdowns", "two_point_attempts", "xp_attempts"]


def empty_stats() -> dict:
    return {k: 0 for k in TEAM_STATS}


@dataclass
class GameResult:
    home: str
    away: str
    home_score: int
    away_score: int
    reg_home_score: int
    reg_away_score: int
    went_ot: bool
    tie: bool
    team_stats: dict               # {team: {stat: value}}
    players: dict = field(default_factory=dict)   # empty until Phase 4
    play_log: list | None = None   # only when debug=True

    @property
    def margin(self) -> int:
        return self.home_score - self.away_score

    @property
    def total(self) -> int:
        return self.home_score + self.away_score


class BoxScore:
    def __init__(self, home: str, away: str, debug: bool = False):
        self.home, self.away = home, away
        self.stats = {home: empty_stats(), away: empty_stats()}
        self.reg_score: dict | None = None
        self.log: list | None = [] if debug else None

    def add(self, team: str, stat: str, value: int = 1) -> None:
        self.stats[team][stat] += value

    def snapshot(self, state) -> dict | None:
        if self.log is None:
            return None
        return {"qtr": state.qtr, "clock": state.half_seconds_remaining, "pos": state.possession,
                "down": state.down, "ydstogo": state.ydstogo, "yl": state.yardline_100,
                "score": dict(state.score), "timeouts": dict(state.timeouts), "phase": state.phase.value, "drive": state.drive}

    def record_play(self, pre: dict | None, play: dict) -> None:
        if self.log is not None:
            self.log.append({**pre, **play})

    def result(self, state) -> GameResult:
        reg = self.reg_score or dict(state.score)
        return GameResult(
            home=self.home, away=self.away, home_score=state.score[self.home], away_score=state.score[self.away],
            reg_home_score=reg[self.home], reg_away_score=reg[self.away], went_ot=state.ot_state.period > 0,
            tie=state.score[self.home] == state.score[self.away], team_stats=self.stats, players={},
            play_log=self.log,
        )
