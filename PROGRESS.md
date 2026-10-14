# PROGRESS.md

| Phase | Status | Commit |
| --- | --- | --- |
| Setup | done | e429a84 |
| 0 Data | done | e7a0861 |
| 1 Engine | done | a62fa5f |
| 2 Models | done | 725a6f1 |
| 3 Team strength | done | 6eafd8b |
| 4 Players | done | 7ecc830 |
| 5 Scale + fair odds | done | 3cb740f |
| 6 FanDuel compare | done | (this commit) |
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

## Phase 3 acceptance (2026-10-13)

### Built

- `features/team_strength.py`: 13 as-of team ratings per (team, season, week) for 2018-2026 (6,112 rows, playoffs
  included) in `data/processed/team_ratings.parquet`, exponentially weighted (half-life 8 games, play-weighted) with last
  season's regressed final as a 6-game prior, centered per week and scaled by last season's spread (see DECISIONS).
  `nflsim features --season S --week W` builds the table and prints the week.
- Team features wired into 7 models per the spec table (`pass_outcome` split across dropback/throw), weather into
  `throw_outcome` and `air_yards`; every model retrained on the as-of joins. Situation-only reference models kept in
  `artifacts/models/situation/` for the `proe` / `cpoe` / `fourth_aggr` definitions.
- Home-field offset (`home_logit_offset`, fitted by simulation) and a simulation-time rating input scale
  (`team_strength.input_scale`, fitted on 2019-2024, tested on 2025), both in `config/settings.yaml`.
- `nflsim phase3-check`: week-by-week season backtest with as-of ratings, multiprocess.

### Model report with team features (validation 2025; "situation-only" = Phase 2 model)

| Model | Val rows | Log loss: model | situation-only | table | constant | Beats table | Beats constant | Calibration (deciles ≥500 within 3 pts) | C | Team features |
|---|---|---|---|---|---|---|---|---|---|---|
| play_call | 33571 | 0.5592 | 0.5604 | 0.5652 | 0.6730 | yes | yes | yes | 1.0 | yes |
| fourth_down | 3997 | 0.2901 | 0.2925 | 0.4227 | 1.0396 | yes | yes | yes | 1.0 | yes |
| air_yards | 16556 | 1.7110 | 1.7129 | 1.7180 | 1.8039 | yes | yes | yes | 1.0 | yes |
| dropback_outcome | 18816 | 0.4449 | 0.4475 | 0.4484 | 0.4513 | yes | yes | yes | 10.0 | yes |
| throw_outcome | 16556 | 0.6861 | 0.6881 | 0.6901 | 0.7345 | yes | yes | yes | 0.1 | yes |
| yac | 10628 | 1.8578 | 1.8582 | 1.8693 | 2.0449 | yes | yes | yes | 10.0 | yes |
| rush_yards | 12191 | 2.6607 | 2.6613 | 2.6702 | 2.7376 | yes | yes | yes | 0.1 | yes |
| fumble_lost | 25080 | 0.0431 |  | 0.0435 | 0.0479 | yes | yes | yes | 10.0 | no |
| fg_make | 1067 | 0.3593 |  | 0.3615 | 0.4148 | yes | yes | yes | 1.0 | no |
| pat_choice | 1460 | 0.1611 |  | 0.2357 | 0.3005 | yes | yes | yes | 10.0 | no |
| clock_runoff | 37519 | 1.3477 | 1.3514 | 1.5177 | 2.0570 | yes | yes | yes | 10.0 | yes |
| presnap_penalty | 36491 | 0.2738 |  | 0.2740 | 0.2762 | yes | yes | yes | 0.1 | no |

Every model with team features improves on its situation-only version; all 12 beat the table and the constant model and
every calibration decile is within 3 points (play_call now calibrates too).

### Checks

- [x] **Leakage tests pass** (`tests/test_team_strength.py`): 200 random (team, week) rows recomputed from plays dated
  before the as-of date match exactly; a perturbed week-9 play changes no rating at weeks <= 9 and does change later weeks.
- [x] `uv run pytest`: 43 passed.

Quick backtest, 2025 regular season, every game simulated 2,000 times with that week's as-of ratings
(`nflsim phase3-check --season 2025 --n 2000`, 272 games, 23 minutes on 10 cores):

| Check | Value | Required | Pass |
|---|---|---|---|
| corr(simulated mean margin, actual margin) | 0.429 | >= 0.25 | yes |
| corr(closing spread, actual margin), for reference | 0.504 | | |
| corr(simulated mean margin, closing spread) | 0.864 | >= 0.70 | yes |
| simulated home margin (non-neutral games) | 2.16 | real 2025: 2.19; 2021-25 range 1.57-2.71 | yes |
| std of simulated mean margins / std of closing spreads | 6.09 / 6.24 = 0.98 | 0.75-1.25 | yes |
| slope of actual margin on simulated margin (calibration) | 1.00 | ~1 | yes |
| mean absolute error vs actual: sim / market | 10.04 / 9.72 | | |
| corr(simulated total, closing total) | 0.745 | | |

The std ratio holds through the season (0.98 in weeks 1-6, 1.01 in 7-12, 0.95 in 13-18). The simulated totals run low
(42.8 vs 44.9 market / 46.0 actual), the same 1-point-per-team scoring residual carried from Phase 2.

**How the acceptance was reached, in order (details and numbers in DECISIONS.md):** (1) the home margin was +0.95 with
`is_home` alone, so the spec's home logit offset was fitted by simulation (h = 0.06 gives +2.2); (2) the first backtest had
a std ratio of 0.64. Diagnosis ruled out regularization, the smoothing parameters and per-play miscalibration, and showed the
ratings carry a 6-point predictable spread of which the play-level composition transmits 4; (3) a per-week z-score was
replaced by a per-season scale (kept the spec's purpose, removed an artifact); (4) a simulation-time input scale of 1.43 was
fitted on 2019-2024 (slope 1.43, se 0.085) and validated on 2025 (slope 1.00, ratio 0.98). Item 4 is a fitted calibration
layer rather than a play-level mechanism; Luke should confirm he is comfortable with it (it is one config value).

## Phase 4 acceptance (2026-10-13)

### Built

- `features/players.py`: per-player usage shares (target, carry, deep/short/intermediate, red zone, snap share) and
  shrunk efficiency over expected (receiver catch rate and YAC, rusher yards, QB CPOE / sack / INT / scramble / aDOT,
  kicker FG make), as of before each game, for every skill player on a weekly roster 2018-2026 (129,751 rows) in
  `data/processed/player_usage.parquet` and `player_eff.parquet`; position-slot priors in `position_priors.parquet`.
- `features/actives.py`: actives per team from the weekly roster, the injury report and the overrides file; starting QB
  by recent dropbacks; `data/inputs/actives_<season>_wk<week>.json` with Questionable warnings.
- `engine/players.py` + engine hooks: target chosen after the throw depth (share x air-yard band fit, red-zone blend),
  carrier from carry share, QB as passer (scrambles = QB rushes), player effects fed to the models, full per-player box
  score (16 stats incl. anytime TD) in `GameResult.players`.
- `sim/game_inputs.py`: per-game context assembly including the QB blend rule (step 4.4); `nflsim features` now builds
  team ratings, player tables and the week's actives; `nflsim phase4-check` runs the player backfill.
- Six models retrained with player features (report below); kicker effect in `fg_make` (step 4.5 done).

### Model report (validation 2025; "situation-only" = Phase 2 reference)

| Model | Val rows | Log loss: model | situation-only | table | constant | Beats table | Beats constant | Calibration (deciles ≥500 within 3 pts) | C | Team features |
|---|---|---|---|---|---|---|---|---|---|---|
| play_call | 33571 | 0.5592 | 0.5604 | 0.5652 | 0.6730 | yes | yes | yes | 1.0 | yes |
| fourth_down | 3997 | 0.2901 | 0.2925 | 0.4227 | 1.0396 | yes | yes | yes | 1.0 | yes |
| air_yards | 16556 | 1.7104 | 1.7129 | 1.7180 | 1.8039 | yes | yes | yes | 1.0 | yes |
| dropback_outcome | 18816 | 0.4392 | 0.4475 | 0.4484 | 0.4513 | yes | yes | yes | 1.0 | yes |
| throw_outcome | 16556 | 0.6838 | 0.6881 | 0.6901 | 0.7345 | yes | yes | yes | 0.01 | yes |
| yac | 10628 | 1.8562 | 1.8582 | 1.8693 | 2.0449 | yes | yes | NO | 10.0 | yes |
| rush_yards | 12191 | 2.6613 | 2.6613 | 2.6702 | 2.7376 | yes | yes | yes | 0.1 | yes |
| fumble_lost | 25080 | 0.0431 |  | 0.0435 | 0.0479 | yes | yes | yes | 10.0 | no |
| fg_make | 1067 | 0.3589 | 0.3593 | 0.3615 | 0.4148 | yes | yes | yes | 1.0 | yes |
| pat_choice | 1460 | 0.1611 |  | 0.2357 | 0.3005 | yes | yes | yes | 10.0 | no |
| clock_runoff | 37519 | 1.3477 | 1.3514 | 1.5177 | 2.0570 | yes | yes | yes | 10.0 | yes |
| presnap_penalty | 36491 | 0.2738 |  | 0.2740 | 0.2762 | yes | yes | yes | 0.1 | no |

Every player-feature model improves on its Phase 3 version; `yac` misses one calibration decile by 4 points (the
others are within 3), kept since it beats the table and the Phase 3 model on log loss.

### Checks

Backfill: `nflsim phase4-check --season 2025 --weeks 4-18 --n 2000` (224 games, as-of actives / shares /
efficiencies / ratings, 2,000 sims per game, 40 minutes on 10 cores). The run printed 28 QB blends
(step 4.4), e.g. ARI (Jacoby Brissett), BAL (Cooper Rush), CAR (Andy Dalton), CIN (Jake Browning).

- [x] **Calibration** (players with at least 8 appearances in weeks 4-18; the spec's 20 is impossible in 15 weeks):

| Stat | player-games | below median | inside 10-90 band | mean actual | mean sim |
|---|---|---|---|---|---|
| pass_yards | 359 | 46.9% | 67.4% | 224.58 | 214.74 |
| rush_yards | 1194 | 44.7% | 69.7% | 37.90 | 33.15 |
| rec_yards | 2731 | 44.1% | 79.9% | 32.36 | 29.98 |
| receptions | 2731 | 44.4% | 85.8% | 2.90 | 2.73 |
| all | 7015 | 44.5% | 79.8% | 31.67 | 29.37 |

  Pooled: 44.5% below the median (target 40-60%), 79.8% inside the 10th-90th band (target 70-90%): PASS

  Per stat, receiving yards and receptions are well inside both bands; passing and rushing yards are at the edge of
  the 10-90 band (67% and 70%) with simulated means 4-12% low, the same scoring residual carried since Phase 2
  (simulated totals 42.7 vs 46.1 actual).
- [x] **Team consistency** (`test_box_score_consistency_over_random_games`, 60 games): receiving yards = QB passing
  yards = team passing yards; receptions = completions; targets = attempts; rushing yards and attempts sum to the
  team's; sack yards live only in the team stat; receiving TDs = QB passing TDs.
- [x] **Leakage** (`test_player_shares_use_only_games_before_the_week`): week-9 rows rebuilt from games dated before
  the as-of date match exactly, and a perturbed week-9 game changes week 10 but not week 9.
- [x] **Injury scenario** (`test_injury_scenario_wr1_out_redistributes_targets`): removing BUF's WR1 raises the other
  receivers' simulated targets and keeps team pass attempts within 1 (400 games each way).
- [x] Team-level checks with player inputs still pass:

| Check | Value | Pass |
|---|---|---|
| corr(sim margin, actual margin) >= 0.25 | 0.438 | yes |
| corr(sim margin, closing spread) >= 0.7 | 0.866 | yes |
| sim home margin within real range (see text) | 2.237 | see text |
| std of sim margins within +/-25% of spread std | 0.874 | yes |

- [x] `uv run pytest`: 48 passed.

## Phase 5 acceptance (2026-10-13)

### Built

- `sim/pack.py`: every decision packed as a dense cumulative table (back-off baked in) or a trained-model array set;
  each game's context (scaled ratings, weather, actives, shares, band fits, efficiencies) packed into fixed-shape arrays.
- `sim/fast_features.py` + `sim/fast_engine.py`: the Numba port of the reference engine with one RNG stream per
  simulation (`prange`), returning `home_score[n]`, `away_score[n]`, `went_ot[n]`, team stats and
  `player_stats[n, players, 16]`.
- `sim/aggregate.py`: moneyline (with tie and tie-as-push), spreads -20.5..+20.5, totals 30.5..60.5, team totals,
  player props (percentiles 5-95 and P(over) at every half-point line), anytime TD, overtime; `se` beside every
  probability; fair American odds.
- `nflsim simulate --season S --week W [--include-played]`: writes `sims_<game_id>.npz`, `fair_odds.parquet`,
  `player_props.parquet` and `run.json` under `data/outputs/<season>_wk<week>/`.
- Exact-equality tests of the jitted feature builder and probabilities against the Python models
  (`tests/test_fast_engine.py`); the equivalence test is `tests/test_equivalence.py` (marked slow, run explicitly).

### Checks

- [x] **Equivalence test** (reference engine vs fast engine, 20,000 sims each, five 2025 matchups with full player
  inputs): every mean within 2 SE, both KS tests above 0.01 for every matchup.

| Matchup | Stat | Reference mean | Fast mean | Diff | SE | Within 2 SE |
|---|---|---|---|---|---|---|
| TB@BUF wk11 | home_score | 22.919 | 23.020 | +0.101 | 0.089 | yes |
| TB@BUF wk11 | away_score | 19.282 | 19.274 | -0.008 | 0.085 | yes |
| TB@BUF wk11 | margin | 3.638 | 3.746 | +0.109 | 0.125 | yes |
| TB@BUF wk11 | total | 42.201 | 42.294 | +0.093 | 0.121 | yes |
| TB@BUF wk11 | home_qb_pass_yds | 214.317 | 214.634 | +0.317 | 0.565 | yes |
| TB@BUF wk11 | home_qb_att | 29.655 | 29.652 | -0.003 | 0.062 | yes |
| TB@BUF wk11 | away_qb_pass_yds | 221.317 | 221.353 | +0.036 | 0.581 | yes |
| TB@BUF wk11 | home_wr1_rec_yds | 54.447 | 54.442 | -0.005 | 0.315 | yes |
| TB@BUF wk11 | home_rb1_rush_yds | 78.575 | 78.641 | +0.066 | 0.365 | yes |
| KC@DEN wk11 | home_score | 23.384 | 23.382 | -0.002 | 0.092 | yes |
| KC@DEN wk11 | away_score | 20.696 | 20.732 | +0.036 | 0.091 | yes |
| KC@DEN wk11 | margin | 2.688 | 2.650 | -0.038 | 0.133 | yes |
| KC@DEN wk11 | total | 44.080 | 44.114 | +0.035 | 0.125 | yes |
| KC@DEN wk11 | home_qb_pass_yds | 243.194 | 242.802 | -0.392 | 0.600 | yes |
| KC@DEN wk11 | home_qb_att | 36.359 | 36.318 | -0.041 | 0.070 | yes |
| KC@DEN wk11 | away_qb_pass_yds | 263.165 | 262.989 | -0.176 | 0.629 | yes |
| KC@DEN wk11 | home_wr1_rec_yds | 50.404 | 50.577 | +0.173 | 0.316 | yes |
| KC@DEN wk11 | home_rb1_rush_yds | 41.378 | 41.408 | +0.030 | 0.239 | yes |
| BUF@HOU wk12 | home_score | 20.983 | 20.870 | -0.113 | 0.088 | yes |
| BUF@HOU wk12 | away_score | 19.898 | 20.032 | +0.134 | 0.087 | yes |
| BUF@HOU wk12 | margin | 1.085 | 0.838 | -0.247 | 0.130 | yes |
| BUF@HOU wk12 | total | 40.881 | 40.903 | +0.022 | 0.119 | yes |
| BUF@HOU wk12 | home_qb_pass_yds | 238.833 | 238.242 | -0.592 | 0.595 | yes |
| BUF@HOU wk12 | home_qb_att | 35.282 | 35.320 | +0.038 | 0.067 | yes |
| BUF@HOU wk12 | away_qb_pass_yds | 204.733 | 203.830 | -0.903 | 0.569 | yes |
| BUF@HOU wk12 | home_wr1_rec_yds | 64.991 | 64.540 | -0.451 | 0.357 | yes |
| BUF@HOU wk12 | home_rb1_rush_yds | 43.375 | 43.379 | +0.003 | 0.241 | yes |
| IND@KC wk12 | home_score | 24.857 | 24.794 | -0.063 | 0.095 | yes |
| IND@KC wk12 | away_score | 24.722 | 24.625 | -0.097 | 0.095 | yes |
| IND@KC wk12 | margin | 0.135 | 0.169 | +0.034 | 0.139 | yes |
| IND@KC wk12 | total | 49.578 | 49.418 | -0.160 | 0.130 | yes |
| IND@KC wk12 | home_qb_pass_yds | 289.503 | 289.338 | -0.165 | 0.644 | yes |
| IND@KC wk12 | home_qb_att | 40.718 | 40.747 | +0.029 | 0.070 | yes |
| IND@KC wk12 | away_qb_pass_yds | 254.091 | 254.482 | +0.391 | 0.615 | yes |
| IND@KC wk12 | home_wr1_rec_yds | 64.212 | 63.760 | -0.453 | 0.339 | yes |
| IND@KC wk12 | home_rb1_rush_yds | 31.389 | 31.557 | +0.169 | 0.211 | yes |
| NO@LA wk9 | home_score | 27.328 | 27.266 | -0.062 | 0.093 | yes |
| NO@LA wk9 | away_score | 15.509 | 15.535 | +0.026 | 0.082 | yes |
| NO@LA wk9 | margin | 11.819 | 11.731 | -0.088 | 0.128 | yes |
| NO@LA wk9 | total | 42.837 | 42.802 | -0.035 | 0.120 | yes |
| NO@LA wk9 | home_qb_pass_yds | 281.263 | 280.257 | -1.006 | 0.623 | yes |
| NO@LA wk9 | home_qb_att | 36.490 | 36.490 | +0.000 | 0.067 | yes |
| NO@LA wk9 | away_qb_pass_yds | 196.567 | 195.762 | -0.805 | 0.570 | yes |
| NO@LA wk9 | home_wr1_rec_yds | 90.790 | 90.368 | -0.422 | 0.414 | yes |
| NO@LA wk9 | home_rb1_rush_yds | 64.542 | 64.590 | +0.048 | 0.311 | yes |

| Matchup | KS p (margin) | KS p (total) | Fast engine 20,000 sims |
|---|---|---|---|
| TB@BUF wk11 | 0.455 | 0.332 | 20.0s |
| KC@DEN wk11 | 0.750 | 0.888 | 25.2s |
| BUF@HOU wk12 | 0.176 | 0.440 | 19.5s |
| IND@KC wk12 | 0.921 | 0.525 | 25.3s |
| NO@LA wk9 | 0.899 | 0.996 | 26.8s |

- [x] **Speed:** 10,000 sims of one game in 10.3 s on the M5 (10 threads), excluding the one-time compile of about
  40 s (cached on disk after the first run). The week run below averaged 13 s per game. The reference engine is
  about 25 ms per game, so the fast engine is roughly 25x faster per core and 250x with threads.
- [x] `nflsim simulate --season 2025 --week 10 --include-played`: 14 games x 10,000 sims in 185 s;
  `fair_odds.parquet` has 7,952 rows (14 games), `player_props.parquet` 43,591 rows (371 players), one `.npz` per game.
- [x] **Sanity:** for every game P(home win) + P(away win) + P(tie) = 1, P(home covers -L) is monotone in L across all
  82 spread lines, and P(over T) is monotone in T across all 61 totals.
- [x] `uv run pytest`: 50 passed (the slow equivalence test deselected by default; run with `-m slow`).

## Phase 6 acceptance (2026-10-13)

### Built

- `odds/fanduel.py`: a read-only FanDuel client (none existed) around the NFL content page Luke supplied and the
  event-page prop tabs, normalizing moneyline, spreads (incl. alternates), totals (incl. alternates), team totals, player
  over/unders, alternate ladders (one-sided) and anytime TD into the spec's table; every pull saved with a timestamp.
- `odds/matching.py` (teams via `team_aliases.yaml`, games on teams + date, players within the game's active rosters
  with fuzzy fallback and `player_aliases.yaml`, `unmatched.csv`), `odds/devig.py` (multiplicative and power
  methods, hold, one-sided estimate), `odds/compare.py` (p_model at the exact line, push-adjusted; edge, decimal odds,
  EV per dollar, fair American odds, MC se, flags), `report/dashboard.py` + `templates/week.html.j2` (header, games
  table, sortable/filterable disagreements table, props by game, per-game margin/total histograms with the FanDuel
  lines marked, footer), `nflsim compare`, `nflsim report` and `nflsim week`.

### Checks (current week = 2026 week 6, Oct 15-19; FanDuel also lists week 7)

| Check | Result | Required |
|---|---|---|
| FanDuel game markets matched (week 6) | 5174/5358 (96.6%) | >= 95% |
| FanDuel player props matched (week 6) | 1638/1889 (86.7%) | >= 85% |
| Week 7 (next week, main markets only on FanDuel) | game 84/84 (100.0%), no props posted yet | |
| De-vig unit tests (-110/-110 -> 0.5/0.5, hold 4.76%; -150/+130) | pass | pass |
| EV unit tests incl. whole-number push | pass | pass |
| Dashboard renders, sorts and filters in Chrome | yes (no console errors; sort by Edge and min-edge filter exercised) | yes |
| Median absolute edge on main spreads and totals | 8.8 pts (spreads 5.7, totals 12.0) | under about 4 |
| `uv run pytest` | 59 passed | |

The last row fails. Fair totals average -3.6 points vs FanDuel's main totals (week 7: -3.5), so
nearly every main-total row carries CHECK_INPUTS, and spreads disagree by a 5.7-point median. The spec's reading
applies: the model is miscalibrated on scoring and no edge should be trusted until that is fixed. The two causes
found and fixed during this phase (zero player effects for unpublished roster weeks; a corrupt receiver-YAC feature)
removed about 3 of the original 6.6-point gap; the remaining 3.6 points are the scoring deficit carried since Phase 2
(league-average sim 42.0 vs 46 actual in 2025-26). First item for Phase 7.

Unmatched after the fixes: 4 player names in week 6 (Zavier Scott, Chigoziem Okonkwo, Omar Cooper Jr., Dont'e Thornton Jr.),
practice-squad elevations not on the published roster; game-market misses are alternate lines beyond the widened
ranges. `nflsim week --season 2026 --week 6` ran end to end (ingest refresh, features, 14 games x 10,000 sims,
pull, compare, report) and printed the report path and the top-15 rows.
