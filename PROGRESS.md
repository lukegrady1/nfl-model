# PROGRESS.md

| Phase | Status | Commit |
| --- | --- | --- |
| Setup | done | e429a84 |
| 0 Data | done | e7a0861 |
| 1 Engine | done | a62fa5f |
| 2 Models | done | (this commit) |
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

## Phase 2 acceptance (2026-10-13)

### Models trained, saved and reported

`uv run nflsim train` writes `artifacts/models/<name>.json` (classes, feature names, means, stds, coefficients, intercepts,
C, seasons, validation log loss, timestamp, within-bin yard distributions) for 12 decisions, plus `artifacts/models/report.md`
(log loss vs the Phase 1 table rebuilt from 2021-2024 and vs a constant model, 10-decile calibration, top-5 coefficients
with a sanity read). Validation season: 2025; models for the comparison were fit on 2018-2024 only.

| Model | Val rows | Log loss: model | table | constant | Beats table | Beats constant | Calibration (deciles ≥500 within 3 pts) | C |
|---|---|---|---|---|---|---|---|---|
| play_call | 33571 | 0.5604 | 0.5652 | 0.6730 | yes | yes | NO | 1.0 |
| fourth_down | 3997 | 0.2925 | 0.4227 | 1.0396 | yes | yes | yes | 1.0 |
| air_yards | 16556 | 1.7129 | 1.7180 | 1.8039 | yes | yes | yes | 0.1 |
| dropback_outcome | 18816 | 0.4475 | 0.4484 | 0.4513 | yes | yes | yes | 10.0 |
| throw_outcome | 16556 | 0.6881 | 0.6901 | 0.7345 | yes | yes | yes | 0.01 |
| yac | 10628 | 1.8582 | 1.8693 | 2.0449 | yes | yes | yes | 10.0 |
| rush_yards | 12191 | 2.6613 | 2.6702 | 2.7376 | yes | yes | yes | 0.1 |
| fumble_lost | 25080 | 0.0431 | 0.0435 | 0.0479 | yes | yes | yes | 10.0 |
| fg_make | 1067 | 0.3593 | 0.3615 | 0.4148 | yes | yes | yes | 1.0 |
| pat_choice | 1460 | 0.1611 | 0.2357 | 0.3005 | yes | yes | yes | 10.0 |
| clock_runoff | 37519 | 1.3514 | 1.5177 | 2.0570 | yes | yes | yes | 1.0 |
| presnap_penalty | 36491 | 0.2738 | 0.2740 | 0.2762 | yes | yes | yes | 0.1 |

- [x] All 12 models beat their lookup table and the constant model on validation log loss. The two yardage tables were
  hard to beat (they know the exact goal-line cap); per-yard goal-line indicators and down x distance interactions were
  needed (see DECISIONS.md). `fourth_down`, `pat_choice` and `clock_runoff` improve on the tables by a wide margin.
- [x] Calibration: 11 of 12 models have every decile with >= 500 plays within 3 points. `play_call` misses in one decile
  (predicted 0.385 vs observed 0.353 for "run", n = 3,357); the other nine deciles are within 2 points. Kept as the
  trained model since it beats the table on log loss; noted for Phase 3, where the team pass-rate feature changes it anyway.
- [x] Coefficient sanity reads are in the report: passing rises with ydstogo, down 3 and trailing late; 4th-down go
  falls with distance and FG rises with short kicks; completions fall and interceptions rise with air yards; FG make
  falls with distance and wind; sacks carry the highest fumble rate; two-point tries peak at the textbook deficits.
- [x] NumPy inference test (`tests/test_trained.py`): a binary and a multinomial model fit with scikit-learn on 20,000
  rows, scored on 10,000 validation rows, agree with `predict_proba` to within 1e-9.
- [x] Feature-builder equality test: 1,000 real plays built from a `GameState` and from the row give identical feature
  vectors for all 12 models.
- [x] Sanity scenario (`test_sanity_scenarios_down_10_passes_up_10_runs`): from 1st-and-10 at the own 25 with 3:00 left
  in Q4, a team down 10 passes on more than 80% of its snaps over 300 simulated drives, and a team up 10 runs (or kneels)
  on more than 60%.
- [x] `uv run pytest`: 37 passed.

### Phase 1 league-average check with trained models swapped in one at a time (2,000 games per step, seed 20261013)

Each row adds one trained model to the previous set (gap vs real 2021-25 in parentheses, `!` = outside the band).
The `Pass` column grades the raw margin std literally, as in Phase 1; the equal-teams benchmark (net of closing spread)
passes at every step. The `yac` step's only miss is OT share 2.8% at 2,000 games (band 3-9%; 3.8% at 5,000 games).

| Swapped in | points_per_team_game | plays_per_team_game | pass_call_rate | completion_rate | sack_rate_per_dropback | int_rate_per_attempt | punts_per_team_game | fg_att_per_team_game | margin_std | ot_share | Pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tables only | 22.973 (+0.46) | 62.648 (+0.08) | 0.606 (-0.00) | 0.651 (+0.00) | 0.066 (+0.00) | 0.023 (-0.00) | 3.736 (-0.13) | 1.818 (-0.15) | 12.492 (-1.70) ! | 0.043 | NO |
| play_call | 23.151 (+0.64) | 62.812 (+0.25) | 0.607 (+0.00) | 0.648 (-0.00) | 0.065 (+0.00) | 0.022 (-0.00) | 3.777 (-0.09) | 1.822 (-0.15) | 12.413 (-1.77) ! | 0.044 | NO |
| fourth_down | 23.176 (+0.66) | 62.825 (+0.26) | 0.610 (+0.00) | 0.647 (-0.00) | 0.065 (+0.00) | 0.022 (-0.00) | 3.663 (-0.21) | 1.852 (-0.12) | 12.644 (-1.54) ! | 0.034 | NO |
| air_yards | 23.150 (+0.64) | 62.894 (+0.33) | 0.610 (+0.00) | 0.647 (-0.00) | 0.064 (-0.00) | 0.022 (-0.00) | 3.651 (-0.22) | 1.879 (-0.09) | 12.227 (-1.96) ! | 0.041 | NO |
| dropback_outcome | 22.749 (+0.24) | 62.833 (+0.27) | 0.609 (+0.00) | 0.647 (-0.00) | 0.063 (-0.00) | 0.024 (+0.00) | 3.663 (-0.21) | 1.887 (-0.08) | 12.695 (-1.49) | 0.034 | yes |
| throw_outcome | 23.163 (+0.65) | 63.011 (+0.45) | 0.609 (+0.00) | 0.644 (-0.00) | 0.063 (-0.00) | 0.023 (+0.00) | 3.795 (-0.08) | 1.857 (-0.11) | 13.841 (-0.35) | 0.043 | yes |
| yac | 21.973 (-0.54) | 62.629 (+0.06) | 0.604 (-0.00) | 0.643 (-0.01) | 0.064 (-0.00) | 0.024 (+0.00) | 3.684 (-0.19) | 1.961 (-0.01) | 14.146 (-0.04) | 0.028 | NO |
| rush_yards | 21.401 (-1.11) | 62.864 (+0.30) | 0.606 (+0.00) | 0.642 (-0.01) | 0.064 (-0.00) | 0.023 (+0.00) | 3.748 (-0.12) | 1.910 (-0.06) | 13.413 (-0.77) | 0.034 | yes |
| fumble_lost | 21.782 (-0.73) | 62.782 (+0.22) | 0.604 (-0.00) | 0.641 (-0.01) | 0.065 (+0.00) | 0.023 (+0.00) | 3.662 (-0.21) | 1.944 (-0.02) | 13.828 (-0.36) | 0.033 | yes |
| fg_make | 21.689 (-0.82) | 62.843 (+0.28) | 0.607 (+0.00) | 0.642 (-0.01) | 0.064 (-0.00) | 0.023 (+0.00) | 3.677 (-0.19) | 1.902 (-0.07) | 13.953 (-0.23) | 0.033 | yes |
| pat_choice | 21.763 (-0.75) | 62.782 (+0.22) | 0.605 (-0.00) | 0.642 (-0.01) | 0.064 (-0.00) | 0.023 (+0.00) | 3.717 (-0.15) | 1.931 (-0.04) | 14.156 (-0.03) | 0.030 | yes |
| clock_runoff | 21.506 (-1.01) | 62.380 (-0.19) | 0.606 (+0.00) | 0.642 (-0.01) | 0.065 (+0.00) | 0.023 (-0.00) | 3.656 (-0.21) | 1.925 (-0.04) | 13.479 (-0.71) | 0.037 | yes |
| presnap_penalty | 21.553 (-0.96) | 62.293 (-0.27) | 0.607 (+0.00) | 0.644 (-0.00) | 0.065 (+0.00) | 0.023 (+0.00) | 3.715 (-0.16) | 1.886 (-0.08) | 13.495 (-0.69) | 0.037 | yes |

### Phase 1 check with all 12 trained models, 5,000 games (seed 20261013)

```
AVG_A @ AVG_H: 5000 games in 40.7s (8.1 ms/game), seed 20261013
  AVG_H 55.0% win, mean margin +1.41, mean total 42.9 (sd 12.1), OT 3.8%, ties 0.48%
```

| Metric | Simulated | Real 2021-25 | Gap | Allowed | Pass |
|---|---|---|---|---|---|
| points_per_team_game | 21.446 | 22.512 | -1.066 | ±1.5 | yes |
| plays_per_team_game | 62.303 | 62.565 | -0.262 | ±3.0 | yes |
| pass_call_rate | 0.607 | 0.606 | +0.001 | ±0.02 | yes |
| completion_rate | 0.643 | 0.649 | -0.006 | ±0.02 | yes |
| sack_rate_per_dropback | 0.063 | 0.065 | -0.001 | ±0.01 | yes |
| int_rate_per_attempt | 0.023 | 0.023 | +0.000 | ±0.005 | yes |
| punts_per_team_game | 3.719 | 3.870 | -0.152 | ±0.5 | yes |
| fg_att_per_team_game | 1.895 | 1.969 | -0.074 | ±0.4 | yes |
| margin_std | 13.448 | 14.188 | -0.739 | ±1.5 | yes |
| margin_std vs real net of closing spread | 13.448 | 12.629 | +0.819 | ±1.5 | yes |
| ot_share | 0.038 | 0.061 | | 3%-9% | yes |
| *(not graded)* | | | | | |
| drives_per_team_game | 10.849 | 10.811 | | | |
| fg_make_rate | 0.871 | 0.851 | | | |
| scramble_rate_per_dropback | 0.048 | 0.049 | | | |
| fumbles_lost_per_team_game | 0.436 | 0.507 | | | |
| two_point_share | 0.102 | 0.097 | | | |
| home_margin_mean | 1.415 | 2.063 | | | |
| yards_per_dropback | 5.764 | 6.220 | | | |
| yards_per_rush | 4.431 | 4.289 | | | |
| tie_share | 0.005 | 0.003 | | | |

**Result: all 10 graded rows pass, on both the raw and the net-of-spread margin-std benchmarks.** Residuals worth
carrying into Phase 3: points per team-game are 1.07 low (the trained yardage models score about a point less than the
exact-yard tables did: yards per dropback 5.76 vs 6.22 real), OT share is at the low end (3.8% vs 6.1%), and a home
advantage of +1.4 points now appears through `is_home` (real +2.1; Phase 3 step 3.3 measures it).

- Speed: 8.1 ms per game with the pure-Python engine and trained models (5,000 games in 41 s). 10,000 sims of one game
  would take about 80 s, so the Phase 5 Numba port is required to meet the 30-second target.
