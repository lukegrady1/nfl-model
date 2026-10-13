# PROGRESS.md

| Phase | Status | Commit |
| --- | --- | --- |
| Setup | done | |
| 0 Data | done | |
| 1 Engine | not started | |
| 2 Models | not started | |
| 3 Team strength | not started | |
| 4 Players | not started | |
| 5 Scale + fair odds | not started | |
| 6 FanDuel compare | not started | |
| 7 Backtest | not started | |

## Setup acceptance (2026-10-13)

- [x] `uv run nflsim --help` lists every command (ingest, clean, build-tables, sim-game, train, features, simulate, compare, week, backtest).
- [x] `uv run pytest` runs ("no tests ran").
- [x] `data/` is in `.gitignore`.
- [x] `DECISIONS.md` notes what existed in the repo before starting (only the spec).

## Phase 0 acceptance (2026-10-13)

- [x] Raw Parquet files exist for 2018-2026 for pbp, schedules, rosters_weekly, snap_counts, injuries, plus players (`data/raw/`, 113 MB). Processed `plays.parquet` (391,375 rows, 2018-2026) and `schedules.parquet` exist.
- [x] Validation checks pass:
  - Score check: 3 of 2,316 completed games mismatch (0.13%, under the 1% limit), logged to `data/processed/score_mismatches.csv`. All three are 2-point swaps from safeties scored after a same-play turnover.
  - Every `play_class == "pass"` row has exactly one `pass_outcome` (0 bad rows).
  - Air yards present on 99.99% of attempts (limit 95%).
  - `yardline_100` within 1-99 on all scrimmage plays (0 bad rows).
- [x] Re-running `nflsim ingest --seasons 2018-2026` only re-downloads the current season (2026); 2018-2025 report `cached`.
- [x] `uv run pytest`: 6 passed.

### Season summary (regular season; Phase 1 targets)

Definitions: offensive plays = pass + run + kneel + spike snaps; pass call rate = pass calls (incl. scrambles) / (pass + run calls); median seconds elapsed is over offensive snaps. The last row pools 2021-2025, the Phase 1 table window and acceptance target.

| season | games | plays_per_team_game | pass_call_rate | sack_rate_per_dropback | int_rate_per_attempt | completion_rate | points_per_team_game | punts_per_team_game | fg_att_per_team_game | median_seconds_elapsed | margin_std | ot_share | home_margin_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2018 | 256 | 62.930 | 0.619 | 0.065 | 0.024 | 0.651 | 23.344 | 4.324 | 1.848 | 33.000 | 14.364 | 0.059 | 2.203 |
| 2019 | 256 | 63.508 | 0.619 | 0.064 | 0.023 | 0.637 | 22.812 | 4.217 | 1.920 | 33.000 | 14.840 | 0.035 | -0.141 |
| 2020 | 256 | 64.346 | 0.614 | 0.057 | 0.022 | 0.656 | 24.789 | 3.713 | 1.875 | 33.000 | 14.235 | 0.039 | 0.055 |
| 2021 | 272 | 63.325 | 0.612 | 0.060 | 0.024 | 0.650 | 22.982 | 3.820 | 1.888 | 33.000 | 15.432 | 0.074 | 1.713 |
| 2022 | 271 | 62.982 | 0.601 | 0.064 | 0.023 | 0.644 | 21.882 | 4.059 | 1.959 | 33.000 | 12.348 | 0.074 | 1.978 |
| 2023 | 272 | 63.108 | 0.612 | 0.068 | 0.024 | 0.647 | 21.768 | 4.160 | 1.949 | 34.000 | 14.423 | 0.048 | 2.684 |
| 2024 | 272 | 62.153 | 0.603 | 0.065 | 0.022 | 0.656 | 22.912 | 3.761 | 2.050 | 34.000 | 14.458 | 0.059 | 1.868 |
| 2025 | 272 | 61.259 | 0.601 | 0.065 | 0.022 | 0.646 | 23.013 | 3.553 | 2.000 | 35.000 | 14.177 | 0.051 | 2.070 |
| 2026 | 79 | 62.070 | 0.608 | 0.061 | 0.021 | 0.646 | 23.316 | 3.804 | 2.044 | 34.000 | 13.115 | 0.076 | 0.278 |
| 2021-25 pooled | 1359 | 62.565 | 0.606 | 0.065 | 0.023 | 0.649 | 22.512 | 3.870 | 1.969 | 34.000 | 14.188 | 0.061 | 2.063 |

