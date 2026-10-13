# DECISIONS.md

One line per assumption or deviation from the build spec, with the reason. Newest at the bottom of each section.

## Setup

- **Repo state before starting (2026-10-13):** `~/nfl-model` contained only the build spec markdown and an empty git repo (no commits). There was no FanDuel odds client, no FantasyPoints scraping code, and no other code anywhere under `~`. `odds/fanduel.py` will therefore need a client written in Phase 6; nothing existed to wrap.
- **Phase scope for this run:** Luke asked for "Phase 1", but Setup and Phase 0 had not been done and Phase 1 depends on them (lookup tables and targets come from Phase 0 data). Built Setup, Phase 0 and Phase 1 in order with one commit per phase, and stopped before Phase 2.
- **Python 3.12** (not 3.13+) pinned in `pyproject.toml` because Numba 0.68 wheels are most mature there. Managed with `uv`.
- **nflreadpy 0.1.5**: every function named in Phase 0 exists with the exact name (`load_pbp`, `load_schedules`, `load_rosters_weekly`, `load_snap_counts`, `load_injuries`, `load_players`). No renames were needed.
- **Extra settings keys** added to `config/settings.yaml` beyond the spec's list: `table_seasons` (2021-2025, the Phase 1 table window), `min_cell_plays` (200), OT `timeouts` per period type, `assumed_hold_one_sided` (0.08, from Phase 6 text), and an `engine:` block for the kneel/timeout/free-kick constants the spec mentions in prose. Reason: ground rule "every number someone might tune lives in config".
- **Overtime rules check:** see the Phase 1 section below.

## Phase 0

- **Extra columns kept** beyond the spec list: `drive`, `fixed_drive` (spec asks for them), `game_date`, `home_score`, `away_score`, `wp`, `vegas_wp` (Phase 2 filters on pre-snap win probability), `desc`, `td_team`, `defensive_two_point_conv`, `return_team`, kick/punt detail flags. Reason: validation and the Phase 1 tables need them; nothing was removed from the spec list.
- **Marker rows** (`play_type` null: GAME, END QUARTER, END GAME, Two-Minute Warning) are used to compute `seconds_elapsed` and `next_is_two_minute_warning`, then dropped from `plays.parquet`.
- **`seconds_elapsed`** = this row's `game_seconds_remaining` minus the next row's (any row type, including timeout rows, so a play followed by a timeout records only the play's own duration). Null across halftime, the end of regulation and OT period boundaries. Clipped to [0, 60].
- **Clock-stopper approximation:** out of bounds counts as a stop only when the clock at the snap is under 2:00 of the first half or 5:00 of the second half (NFL rule: otherwise the clock restarts on the ready-for-play signal). Also treated as stops: incomplete passes, spikes, scores, changes of possession (interception, lost fumble, punt, kickoff, failed 4th down), accepted penalties, and plays followed by the two-minute warning.
- **nflfastR codes the receiving team as `posteam` on kickoff rows.** Start-spot tables use `next_scrim_posteam` / `next_scrim_yardline` (the next scrimmage snap), so kickoffs and punts are handled identically.
- **Two-point tries** (`two_point_attempt == 1`, play_type pass/run) are classified `two_point` before the pass/run rules so they never enter the play-call or outcome tables.
- **Score check mismatches** (3 games): all are 2-point swaps where a safety was scored by the original offense after a same-play turnover. The validation attributes safeties to `defteam`; the real play has two possession changes. Left as logged mismatches (0.13% < 1%).
- **Summary definitions:** offensive plays per team-game = pass + run + kneel + spike; pass call rate = pass / (pass + run) excluding kneels and spikes; sack and scramble rates per dropback; interception and completion rates per attempt (complete + incomplete + interception). Extra target columns were added (`margin_std`, `ot_share`, `tie_share`, `home_margin_mean`, `fg_make_rate`, `drives_per_team_game`, `two_point_share`) because the Phase 1 acceptance needs OT share and margin std.
- **2026 is in progress** (79 regular-season games through week 6 in the cache). It is always refreshed on ingest and is included in `plays.parquet`, but the Phase 1 tables use 2021-2025 only (`table_seasons`).

## Phase 1

(filled in during Phase 1)
