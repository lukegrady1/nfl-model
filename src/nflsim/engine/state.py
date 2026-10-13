"""Step 1.1: the game state."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Phase(Enum):
    KICKOFF = "kickoff"
    SCRIMMAGE = "scrimmage"
    PAT = "pat"
    FREE_KICK = "free_kick"
    GAME_OVER = "game_over"


@dataclass
class OTState:
    active: bool = False
    first_team: str | None = None      # team that received the OT kickoff
    possession_count: int = 0          # 1 = first team's possession, 2 = second team's, 3+ = sudden death
    period: int = 0                    # OT periods started
    seconds_remaining: int = 0


@dataclass
class Context:
    """Team and player inputs. Empty in Phase 1; filled in Phases 3-4."""
    home: dict = field(default_factory=dict)
    away: dict = field(default_factory=dict)
    players: dict = field(default_factory=dict)
    weather: dict = field(default_factory=dict)   # {"roof_dome": 0/1, "wind": mph}; empty = neutral defaults
    season: int | None = None                      # season being simulated (era feature); None = last training season


@dataclass
class GameState:
    home: str
    away: str
    possession: str = ""
    down: int = 0
    ydstogo: int = 10
    yardline_100: int = 75
    qtr: int = 1
    game_seconds_remaining: int = 3600
    score: dict = field(default_factory=dict)
    timeouts: dict = field(default_factory=dict)
    second_half_receiver: str = ""     # the team that kicked off to start the game
    phase: Phase = Phase.KICKOFF
    two_minute_warning_used: dict = field(default_factory=lambda: {1: False, 2: False, 3: False})
    ot_state: OTState = field(default_factory=OTState)
    is_neutral: bool = False
    playoff: bool = False
    context: Context = field(default_factory=Context)
    kicking_team: str = ""             # team kicking in KICKOFF / FREE_KICK
    pat_team: str = ""                 # team attempting the PAT
    period_expired: bool = False       # clock hit 0:00; transition applied once any pending PAT resolves
    plays: int = 0
    drive: int = 0                     # drive counter (increments on every new possession)

    def __post_init__(self) -> None:
        if not self.score:
            self.score = {self.home: 0, self.away: 0}
        if not self.timeouts:
            self.timeouts = {self.home: 3, self.away: 3}

    # ---- helpers ------------------------------------------------------------------------------------------
    def opponent(self, team: str) -> str:
        return self.away if team == self.home else self.home

    @property
    def defense(self) -> str:
        return self.opponent(self.possession)

    @property
    def score_diff(self) -> int:
        """Offense minus defense."""
        return self.score[self.possession] - self.score[self.defense]

    @property
    def in_ot(self) -> bool:
        return self.qtr >= 5

    @property
    def half(self) -> int:
        """1, 2, or 3 for OT."""
        return 1 if self.qtr <= 2 else (2 if self.qtr <= 4 else 3)

    @property
    def half_seconds_remaining(self) -> int:
        if self.in_ot:
            return self.ot_state.seconds_remaining
        return self.game_seconds_remaining - 1800 if self.qtr <= 2 else self.game_seconds_remaining

    @property
    def quarter_seconds_remaining(self) -> int:
        if self.in_ot:
            return self.ot_state.seconds_remaining
        return self.game_seconds_remaining - 900 * (4 - self.qtr)

    @property
    def tied(self) -> bool:
        return self.score[self.home] == self.score[self.away]

    @property
    def leader(self) -> str | None:
        if self.tied:
            return None
        return self.home if self.score[self.home] > self.score[self.away] else self.away

    @property
    def goal_to_go(self) -> bool:
        return self.ydstogo >= self.yardline_100

    def set_clock(self, half_seconds: int) -> None:
        """Set the clock from seconds remaining in the current half (or OT period)."""
        if self.in_ot:
            self.ot_state.seconds_remaining = half_seconds
        elif self.qtr <= 2:
            self.game_seconds_remaining = half_seconds + 1800
        else:
            self.game_seconds_remaining = half_seconds


def new_game(home: str, away: str, neutral: bool = False, playoff: bool = False, context: Context | None = None) -> GameState:
    return GameState(home=home, away=away, is_neutral=neutral, playoff=playoff, context=context or Context())
