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

(filled in during Phase 0)

## Phase 1

(filled in during Phase 1)
