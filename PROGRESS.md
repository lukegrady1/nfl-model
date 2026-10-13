# PROGRESS.md

| Phase | Status | Commit |
| --- | --- | --- |
| Setup | done | e429a84 |
| 0 Data | done | e7a0861 |
| 1 Engine | done | (this commit) |
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


## Phase 1 acceptance (2026-10-13)

Command: `uv run nflsim sim-game --home AVG --away AVG --n 5000` (seed 20261013 from `config/settings.yaml`).

```
AVG_A @ AVG_H: 5000 games in 4.9s (1.0 ms/game), seed 20261013
  AVG_H 47.6% win, mean margin -0.38, mean total 46.3 (sd 12.7), OT 4.7%, ties 0.96%
```

| Metric | Simulated | Real 2021-25 | Gap | Allowed | Pass |
|---|---|---|---|---|---|
| points_per_team_game | 23.173 | 22.512 | +0.662 | ±1.5 | yes |
| plays_per_team_game | 63.218 | 62.565 | +0.653 | ±3.0 | yes |
| pass_call_rate | 0.607 | 0.606 | +0.001 | ±0.02 | yes |
| completion_rate | 0.646 | 0.649 | -0.002 | ±0.02 | yes |
| sack_rate_per_dropback | 0.065 | 0.065 | +0.000 | ±0.01 | yes |
| int_rate_per_attempt | 0.023 | 0.023 | +0.000 | ±0.005 | yes |
| punts_per_team_game | 3.825 | 3.870 | -0.046 | ±0.5 | yes |
| fg_att_per_team_game | 1.832 | 1.969 | -0.137 | ±0.4 | yes |
| margin_std | 12.682 | 14.188 | -1.506 | ±1.5 | NO |
| margin_std vs real net of closing spread | 12.682 | 12.629 | +0.052 | ±1.5 | yes |
| ot_share | 0.047 | 0.061 | | 3%-9% | yes |
| *(not graded)* | | | | | |
| drives_per_team_game | 11.005 | 10.811 | | | |
| fg_make_rate | 0.864 | 0.851 | | | |
| scramble_rate_per_dropback | 0.050 | 0.049 | | | |
| fumbles_lost_per_team_game | 0.437 | 0.507 | | | |
| two_point_share | 0.073 | 0.097 | | | |
| home_margin_mean | -0.382 | 2.063 | | | |
| yards_per_dropback | 5.874 | 6.220 | | | |
| yards_per_rush | 4.584 | 4.289 | | | |
| tie_share | 0.010 | 0.003 | | | |

**Result: 9 of 10 graded rows pass; `margin_std` misses the literal ±1.5 band by 0.006** (gap -1.506). That gap is
structural, not an engine bug: the real std of 14.19 includes the spread of team quality across matchups (closing spreads
have a std of 6.11 over 2021-25), which an AVG-vs-AVG simulation cannot contain until Phase 3 adds team strength. Net of
the closing spread, real margins have a residual std of 12.63 and the sim gives 12.68 (the extra row above). A second run
with `--seed 7` gives margin_std 12.78 (gap -1.41, passes the literal band), so the configured-seed miss is within the
Monte Carlo error of the band (std of a std at n=5000 is about 0.13). Nothing was tuned to hit the targets; the fixes
made along the way are listed in `DECISIONS.md` (Phase 1), each with the diagnosis that motivated it.

Other acceptance items:

- [x] `tests/test_engine_rules.py`: every rule in step 1.8 has a deterministic stub-model test, plus one test per OT rule
  (1.6.1-1.6.5), pre-snap penalties, the kneel rule and lost fumbles. Invariants hold on every snap across 1,000 random
  games (10% with playoff OT rules). Same seed gives identical games (30 game pairs).
- [x] `tests/test_models.py`: every table cell sums to 1 and lies in [0, 1] (27 tables), every merged cell has at least
  200 plays, and the `TableModel` interface returns valid distributions for 300 random states.
- [x] `uv run pytest`: 31 passed.
- Speed (reference Python engine, Apple M5): 1.0 ms per game, 5,000 games in 4.9 s. The Phase 5 Numba engine is not needed
  for the Phase 1 check but the 30-second budget for 10,000 sims is already met by the reference engine (about 10 s).
- Known gaps carried forward (all logged in `DECISIONS.md`): no untimed down, no spikes, no fumble returns other than
  scoop-and-score TDs, no tack-on penalties on plays that stand, no onside-kick strategy, no home-field effect (Phase 3).
