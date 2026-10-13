# NFL Monte Carlo Simulation Engine — Build Spec for Claude Code

Oct 13, 2026 · @Luke

## How to use this doc (read first, Claude Code)

You are building a play-by-play Monte Carlo simulator for NFL games. It must output fair probabilities for moneylines, spreads, totals and player props, and then rank where FanDuel's odds disagree most with the model. This doc is the full spec. Build it in the phase order below, and do not start a phase until the previous phase's acceptance criteria pass.

### The end goal

For any upcoming NFL week, one command (`nflsim week --season 2026 --week 7`) does four things:

1. It simulates every game 10,000+ times, play by play.
2. It writes the fair probabilities: win/tie for each team, cover probability for every spread from -20.5 to +20.5, over/under for every total from 30.5 to 60.5, and full distributions for player stats.
3. It pulls FanDuel odds, removes the vig, and computes edge and expected value for every market it can match.
4. It writes an HTML dashboard sorted by edge.

### Ground rules (follow all of these)

1. **Work one phase at a time.** Each phase ends with acceptance criteria. Run them, show the results, then stop and summarize for Luke before starting the next phase.
2. **Inspect the existing repo first.** `~/nfl-model` may already contain code, including a FanDuel odds client. Read it before writing anything. Reuse the FanDuel client. Do not delete or rewrite existing working code without asking.
3. **Never leak the future.** Any feature used to simulate a game in week W may only use data from games that finished before week W. This applies to training, team strength, player shares and backtests. This is the most important correctness rule in the project.
4. **Make it reproducible.** Every simulation run takes a seed. The same seed and the same inputs must give identical output.
5. **Test as you go.** Every rule of football you implement gets a unit test (`pytest`). Every model gets a test that its probabilities sum to 1 and stay in \[0, 1\].
6. **Do not invent data.** If a column, endpoint or function name in this doc does not exist, inspect the real data (`df.columns`, `dir(module)`) and adapt. Write what you changed in `DECISIONS.md`.
7. **Log decisions and progress.** Keep `DECISIONS.md` (every assumption or deviation from this doc, one line each, with the reason) and `PROGRESS.md` (phase status and the acceptance results).
8. **Ask when truly blocked.** If a choice would change results in a way this doc does not cover, ask Luke instead of guessing. Do not ask about things this doc already decides.
9. **No betting automation.** The system only reads odds and reports. It never logs into FanDuel to place bets and never sends any wager.
10. **Commit after each phase** with a message like `phase 2: trained decision models`.

### Phases at a glance

| Phase | Delivers | Done when |
| --- | --- | --- |
| Setup | Repo layout, deps, config, CLI skeleton | `nflsim --help` works and tests run |
| 0 Data | Cached, cleaned play-by-play and schedules | Clean tables saved, validation checks pass |
| 1 Engine | Full game simulator using league-average lookup tables | Simulated league averages match reality |
| 2 Models | Trained logistic/multinomial models swapped in | Models beat lookup tables on held-out log loss; sim still matches reality |
| 3 Team strength | Leak-free team features feeding the models | Simulated margins correlate with real results |
| 4 Players | Who gets each touch, player efficiency, actives | Player stat distributions are calibrated |
| 5 Scale + fair odds | Fast batch sims and the fair-odds output files | 10,000 sims of one game in under 30 seconds |
| 6 FanDuel compare | De-vigged comparison and weekly dashboard | Dashboard renders for a real week |
| 7 Backtest | Walk-forward evaluation vs closing lines | Backtest report generated |

### Key conventions used everywhere

- **`yardline_100`** is the distance from the offense to the opponent's end zone (1 to 99), the nflfastR convention. On a change of possession, the new value is `100 - spot`.
- **Time** is `game_seconds_remaining`: 3600 at kickoff, 0 at the end of regulation. Quarters are 900 seconds.
- **Probabilities** are stored as floats in \[0, 1\]. They are only converted to American odds for display.
- **Team abbreviations** follow nflverse (`LA` for the Rams, `LAC`, `JAX`, `WAS`, and so on). All other sources get mapped to these.

## Setup: stack, repo layout, config

Use Python 3.11+ managed with `uv`. Use Polars for data, NumPy for math, scikit-learn for training, and Numba for the fast engine.

### Step 1: Inspect, then scaffold

1. Run `ls -la ~/nfl-model` and read every existing file. Note in `DECISIONS.md` what already exists, especially the FanDuel odds code and any FantasyPoints scraping code.
2. Create the new package beside the existing code. Do not move existing files unless needed. If needed, ask first.
3. Create `pyproject.toml` with a console script `nflsim = "nflsim.cli:app"`.

### Step 2: Dependencies

```
uv add nflreadpy polars pyarrow numpy scikit-learn numba typer pyyaml jinja2 matplotlib requests
uv add --dev pytest pytest-cov
```

`nflreadpy` is the current nflverse Python package. It returns Polars DataFrames. If any function named in Phase 0 is missing, run `python -c "import nflreadpy as nfl; print(dir(nfl))"` and use the closest equivalent.

### Step 3: Repo layout

```
nfl-model/
  pyproject.toml
  DECISIONS.md
  PROGRESS.md
  config/
    settings.yaml            # all tunable numbers live here
    team_aliases.yaml        # FanDuel/other names -> nflverse abbreviations
  data/
    raw/                     # untouched downloads (parquet), gitignored
    processed/               # cleaned tables (parquet), gitignored
    inputs/                  # weekly actives, manual overrides
    outputs/                 # sim results, comparisons, reports
  artifacts/
    models/                  # trained model coefficients (json)
    tables/                  # Phase 1 lookup tables (parquet)
  src/nflsim/
    cli.py
    config.py
    data/{ingest.py, clean.py, validate.py}
    engine/{state.py, game.py, scrimmage.py, kicking.py, clock.py, overtime.py, boxscore.py}
    models/{base.py, tables.py, features.py, train.py, registry.py}
    features/{team_strength.py, players.py, actives.py}
    sim/{runner.py, fast_engine.py, aggregate.py}
    odds/{fanduel.py, devig.py, matching.py, compare.py}
    report/{dashboard.py, templates/week.html.j2}
    backtest/{walkforward.py, metrics.py}
  tests/
```

`odds/fanduel.py` should wrap the existing FanDuel client, not replace it.

### Step 4: Config (`config/settings.yaml`)

Put every number someone might tune here, not in code. Start with:

```yaml
seed: 20261013
training_seasons: [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
kickoff_seasons: [2025]          # kickoff rules changed recently; only use current-rule seasons
sims_per_game: 10000
regular_season_ot:
  period_seconds: 600
  both_teams_possess: true
  ties_allowed: true
playoff_ot:
  period_seconds: 900
  both_teams_possess: true
  ties_allowed: false
team_strength:
  half_life_games: 8
  prior_season_regression: 0.33
player_shrinkage_k: 50            # pseudo-plays toward position average
spread_lines: {min: -20.5, max: 20.5, step: 0.5}
total_lines: {min: 30.5, max: 60.5, step: 0.5}
edge_flag_threshold: 0.10         # flag model-vs-market gaps larger than 10 points for review
```

The overtime values reflect current NFL rules as understood. Verify them against the official rulebook and record the check in `DECISIONS.md`.

### Step 5: CLI skeleton (`typer`)

Create these commands now as stubs. Each phase fills one in.

| Command | Phase | What it does |
| --- | --- | --- |
| `nflsim ingest --seasons 2018-2026` | 0 | Download and cache raw data |
| `nflsim clean` | 0 | Build processed tables |
| `nflsim build-tables` | 1 | Build league-average lookup tables |
| `nflsim sim-game --home KC --away BUF --n 2000` | 1 | Simulate one matchup and print a summary |
| `nflsim train` | 2 | Train all decision models |
| `nflsim features --season 2026 --week 7` | 3-4 | Build team and player inputs for a week |
| `nflsim simulate --season 2026 --week 7` | 5 | Simulate all games and write fair odds |
| `nflsim compare --season 2026 --week 7` | 6 | Pull FanDuel odds and compute edges |
| `nflsim week --season 2026 --week 7` | 6 | Run features, simulate, compare and report in one go |
| `nflsim backtest --seasons 2023-2025` | 7 | Walk-forward evaluation |

### Setup acceptance

- [ ] `uv run nflsim --help` lists every command.
- [ ] `uv run pytest` runs (no tests yet is fine).
- [ ] `data/` is in `.gitignore`.
- [ ] `DECISIONS.md` notes what existed in the repo before you started.

## Phase 0: Data ingestion

This phase downloads nflverse data once, caches it as Parquet, and builds one clean play table that every later phase trains on.

### Step 0.1: Download (`nflsim ingest`)

Pull each dataset per season and save it to `data/raw/<dataset>_<season>.parquet`. Skip seasons already cached unless `--refresh` is passed. Always refresh the current season.

| Dataset | nflreadpy call | Used for |
| --- | --- | --- |
| Play-by-play | `nfl.load_pbp(seasons)` | Training every model, team strength, player shares |
| Schedules | `nfl.load_schedules(seasons)` | Game list, home/away, results, closing `spread_line` / `total_line` / moneylines for backtests |
| Weekly rosters | `nfl.load_rosters_weekly(seasons)` | Player positions and IDs |
| Snap counts | `nfl.load_snap_counts(seasons)` | Player role / playing time |
| Injuries | `nfl.load_injuries(seasons)` | Game status (Out, Doubtful, Questionable) |
| Players | `nfl.load_players()` | Player ID cross-reference and names |

If a call name differs in the installed version, use the real one and note it in `DECISIONS.md`.

### Step 0.2: Clean plays (`nflsim clean` → `data/processed/plays.parquet`)

Keep these columns from play-by-play. Check each exists first.

- **Identity:** `game_id, play_id, season, week, season_type, home_team, away_team, posteam, defteam`
- **Situation:** `qtr, down, ydstogo, yardline_100, goal_to_go, game_seconds_remaining, half_seconds_remaining, quarter_seconds_remaining, score_differential, posteam_score, defteam_score, posteam_timeouts_remaining, defteam_timeouts_remaining, roof, temp, wind`
- **Play type:** `play_type, pass_attempt, rush_attempt, qb_dropback, qb_scramble, qb_kneel, qb_spike, sack`
- **Pass details:** `air_yards, yards_after_catch, complete_pass, incomplete_pass, interception, pass_location, passer_player_id, receiver_player_id`
- **Run details:** `rusher_player_id, run_location`
- **Results:** `yards_gained, first_down, touchdown, pass_touchdown, rush_touchdown, return_touchdown, safety, fumble, fumble_lost, out_of_bounds, epa, return_yards`
- **Kicking:** `field_goal_result, kick_distance, extra_point_result, two_point_attempt, two_point_conv_result, punt_blocked, touchback`
- **Penalties and timeouts:** `penalty, penalty_team, penalty_yards, penalty_type, timeout, timeout_team`

Then add these derived columns:

1. **`play_class`**, one of: `pass` (`qb_dropback == 1`, including sacks and scrambles), `run` (designed run: `rush_attempt == 1` and `qb_scramble == 0`), `punt`, `field_goal`, `kickoff`, `extra_point`, `two_point`, `kneel`, `spike`, `penalty_no_play`, `timeout`, `other`. Scrambles count as **pass calls** with outcome `scramble`. This matters because the play-call model predicts the call, not the result.
2. **`pass_outcome`** for pass plays: `complete`, `incomplete`, `interception`, `sack`, `scramble`.
3. **`seconds_elapsed`**: this play's `game_seconds_remaining` minus the next row's, within the same game. Set it to null across halftime and at the end of regulation. Clip it to \[0, 60\].
4. **`next_play_is_timeout`**: whether a timeout follows before the next snap. This is used to train the clock model.
5. **`clock_stopper`**: whether the play stops the clock. True for an incomplete pass, out of bounds outside the stated windows, a score, a change of possession, a penalty, or a two-minute warning. Note the rule (out of bounds stops the clock only in the last 2:00 of the first half and the last 5:00 of the second half, otherwise the clock restarts on the snap signal) in `DECISIONS.md` and approximate it.
6. **`drive_start`**: whether the play is the first snap of a drive. Keep `drive` and `fixed_drive` for drive analysis.

### Step 0.3: Validate (`data/validate.py`, tests in `tests/test_data.py`)

- [ ] For each completed season, the sum of `posteam_score` changes per game matches final scores from schedules (allow under 1% of games mismatched; log them).
- [ ] Every row with `play_class == "pass"` has exactly one `pass_outcome`.
- [ ] Air yards are present on at least 95% of non-sack, non-scramble pass attempts.
- [ ] `yardline_100` is within 1–99 on all scrimmage plays.
- [ ] For each season, print and save to `PROGRESS.md`: plays per team-game, pass call rate, sack rate per dropback, interception rate per attempt, completion rate, points per team-game, punts per team-game, FG attempts per team-game, and median `seconds_elapsed`. **These numbers become the targets for Phase 1.**

### Phase 0 acceptance

- [ ] Raw and processed Parquet files exist for 2018 through the current season.
- [ ] All validation checks pass, and the season summary table is in `PROGRESS.md`.
- [ ] Re-running `nflsim ingest` with no new data is a no-op apart from the current season.

## Phase 1: Game engine with league-average tables

This phase builds a complete, rule-correct game simulator in plain Python. Every decision samples from simple league-average lookup tables. Simulated league averages must match the Phase 0 targets before any ML is added.

### Step 1.1: Game state (`engine/state.py`)

Use a `@dataclass` called `GameState`:

| Field | Type | Meaning |
| --- | --- | --- |
| `home, away` | str | Team abbreviations |
| `possession` | str | Team with the ball |
| `down` | int | 1–4 (0 during kickoffs/PATs) |
| `ydstogo` | int | Yards for a first down. Equals `yardline_100` when goal to go |
| `yardline_100` | int | Distance to the opponent's end zone |
| `qtr` | int | 1–4, 5 = OT |
| `game_seconds_remaining` | int | Regulation clock; OT tracked separately in `ot_seconds_remaining` |
| `score` | dict | `{home: int, away: int}` |
| `timeouts` | dict | `{home: 3, away: 3}`, reset at halftime; 2 each in regular-season OT |
| `second_half_receiver` | str | The team that kicked off to start the game |
| `phase` | enum | `KICKOFF`, `SCRIMMAGE`, `PAT`, `FREE_KICK`, `GAME_OVER` |
| `two_minute_warning_used` | dict | Per half, so it fires once |
| `ot_state` | object | First-possession tracking for OT |

Also give it an **`is_neutral`** flag, plus a **`context`** object that holds team and player inputs. Context is empty in Phase 1 and filled in Phases 3–4.

### Step 1.2: The model interface (`models/base.py`)

Every decision point uses the same interface, so lookup tables (Phase 1) and trained models (Phase 2) can be swapped without touching the engine:

```python
class DecisionModel(Protocol):
    name: str
    classes: list            # e.g. ["pass", "run"] or yard bins
    def proba(self, state: GameState, extra: dict) -> np.ndarray: ...   # sums to 1
    def sample(self, rng: np.random.Generator, state, extra) -> object: ...
```

`models/registry.py` maps each decision name to its current implementation. Phase 1 registers `tables.*`, and Phase 2 swaps in `trained.*` one model at a time.

### Step 1.3: The decision points

Build one lookup table per decision from 2021–2025 regular-season data (kickoffs: `kickoff_seasons` only). Save them to `artifacts/tables/`. Bucket sparse situations so every cell has at least 200 plays. If a cell is under 200, merge it with the neighboring bucket.

| # | Decision | Classes | Table keyed by |
| --- | --- | --- | --- |
| D1 | Play call (downs 1–3) | pass, run | down × distance bucket (1, 2–3, 4–6, 7–10, 11+) × score-diff bucket × time bucket |
| D2 | 4th-down choice | go, punt, field\_goal | distance bucket × yardline bucket (every 5 yards) × score/time bucket |
| D3 | Air yards | bins: ≤-1, 0–4, 5–9, 10–14, 15–19, 20–29, 30+ | down × distance bucket × field zone |
| D4 | Pass outcome | complete, incomplete, interception, sack, scramble | air-yards bin × field zone |
| D5 | Yards after catch | bins: 0, 1, 2, 3–4, 5–6, 7–9, 10–14, 15–24, 25+ | air-yards bin |
| D6 | Sack yards lost | empirical distribution | none |
| D7 | Scramble yards | empirical distribution | field zone |
| D8 | Rush yards | integers -5 to 15, then bins 16–24, 25–39, 40+ | distance bucket × field zone |
| D9 | Fumble lost | yes, no | play type (completion, sack, scramble, run) |
| D10 | Interception return | return yards distribution + pick-six rate | none |
| D11 | Field goal made | made, missed (blocked counts as missed) | kick distance (per yard, smoothed) |
| D12 | Punt result | net-yards distribution + touchback rate | yardline bucket |
| D13 | Kickoff result | receiving team's starting `yardline_100` distribution | none |
| D14 | PAT choice | kick, go for two | score diff after the TD × time bucket |
| D15 | Extra point / two-point result | good, no good | none |
| D16 | Clock runoff | seconds bins: 0–5, 6–10, …, 41–45, 46+ | `clock_stopper` × trailing/leading × under 2:00 of a half |
| D17 | Pre-snap penalty | none, offense, defense | down × distance bucket |

When a bin is sampled, draw the exact yardage uniformly within the bin, or from the empirical within-bin distribution (preferred).

**Field zones:** own territory (`yardline_100` 51–99), opponent 21–50, red zone 11–20, inside 10.

### Step 1.4: The play flow (`engine/scrimmage.py`)

Each call to `step(state, rng)` advances one play:

1. **Kickoff** (`phase == KICKOFF`): sample D13. Give the receiver 1st and 10 at that spot. Kickoff return touchdowns: use the empirical return-TD rate from the data.
2. **PAT** (`phase == PAT`): sample D14, then D15. Add 1 or 2 points. Set `phase = KICKOFF` with the scoring team kicking.
3. **Pre-snap penalty:** sample D17. Offense: move back 5 yards and replay the down. Defense: move forward 5 yards (half the distance if inside the 10), with a first down if that reaches the line to gain. Run the clock D16 with `clock_stopper=True`, then return.
4. **4th down:** sample D2. A punt goes to step 9, a field goal to step 10, and go continues to step 5.
5. **Play call:** sample D1. Apply the **kneel rule** first: if the offense leads, the opponent cannot stop the clock, and the remaining time is less than about 40 seconds × downs left plus timeouts left, kneel (−1 yard, the clock runs 40 seconds).
6. **Pass:** sample D3 for air yards, capped at `yardline_100`. Then sample D4.
   - `complete`: sample D5 for yards after catch, capped so total yards ≤ `yardline_100`. Then check D9.
   - `incomplete`: 0 yards, the clock stops.
   - `interception`: possession flips at the catch spot, `100 - (yardline_100 - air_yards)`, plus the D10 return. A pick-six goes to the PAT for the defense.
   - `sack`: sample D6, then D9.
   - `scramble`: sample D7, then D9.
7. **Run:** sample D8, capped at `yardline_100`, then D9.
8. **Apply the result:**
   - The new spot is `yardline_100 - yards`. If that is ≤ 0, it's a **touchdown**: +6, `phase = PAT`.
   - If it is ≥ 100 after a loss, it's a **safety**: +2 to the defense, then a free kick by the team that conceded. Use `phase = FREE_KICK` and give the receiver an empirical start (default `yardline_100 = 60` if data is thin).
   - If yards ≥ `ydstogo`, it's a first down: 1st and 10, or 1st and goal when `yardline_100 ≤ 10`.
   - Otherwise the down goes up by one. A failed 4th down turns the ball over on downs at the spot, `100 - new_spot`.
   - A lost fumble flips possession at the spot (v1: no return).
9. **Punt:** sample D12 for net yards. A touchback puts the receiver at `yardline_100 = 80`. A blocked punt uses the empirical rate and the spot.
10. **Field goal:** distance is `yardline_100 + 17`. Sample D11. A make is +3 followed by a kickoff. A miss gives the opponent the ball at the spot of the kick, `100 - (yardline_100 + 7)`, or at their 20 (`yardline_100 = 80`) if the kick was from inside the 20.
11. **Clock:** after every play, run `clock.advance()` (Step 1.5).
12. **Record:** update the box score (Step 1.7) and append the play to the play log if `debug=True`.

### Step 1.5: Clock rules (`engine/clock.py`)

1. Sample D16 for the seconds elapsed until the next snap.
2. **Timeout heuristic:** inside the last 2:00 of either half, if the play didn't stop the clock, the team that trails (on either side of the ball) and has timeouts left calls one. Elapsed is then just the play's duration, sampled from a 4–8 second uniform, and its timeout count drops by 1. The offense also calls a timeout before a last-second field goal when it needs one.
3. **Two-minute warning:** if the clock crosses 120 seconds left in Q2 or Q4, stop it at exactly 120.
4. **End of quarter:** if the clock reaches a quarter boundary, set it to the boundary. Q1→Q2 and Q3→Q4 continue the drive with the same down and distance. End of Q2: halftime. Reset timeouts to 3, and `second_half_receiver` receives the kickoff. End of Q4: if not tied, `GAME_OVER`; if tied, go to OT. A play that started before 0:00 always finishes, and a score on it counts.
5. **Untimed down:** a defensive penalty on the final play of a half gives one untimed down. Implement it only if simple; otherwise note it in `DECISIONS.md` as a known gap.
6. The clock must never go negative.

### Step 1.6: Overtime (`engine/overtime.py`)

Use the settings in `config/settings.yaml`:

1. Coin toss: random receiver, who kicks off.
2. Both teams get a possession unless the first possession ends with a defensive touchdown or a safety.
3. After both teams have possessed, the next score of any kind wins.
4. If the first two possessions leave the score tied, the next score wins.
5. Regular season: if the period expires while tied, the game is a **tie**. Playoffs: keep playing new periods until there is a winner.

Write one unit test for each numbered rule.

### Step 1.7: Box score and output (`engine/boxscore.py`)

Each simulated game returns a compact result:

- Final score per team, the regulation score, and whether OT was played.
- Team stats: plays, pass attempts, completions, passing yards, sacks, interceptions, rush attempts, rushing yards, punts, FG attempts and makes, turnovers, time of possession.
- Phase 4 adds player stat lines. For now, keep the player dictionary empty.

A game should never store its play log unless `debug=True`. 10,000 play logs per game would use far too much memory.

### Step 1.8: Tests (`tests/test_engine_rules.py`)

Write a deterministic test for each rule. Force outcomes with a stub model that returns a fixed class:

- [ ] A touchdown adds 6, moves to PAT, and the scoring team kicks off.
- [ ] A safety adds 2 to the defense, and the team that conceded kicks.
- [ ] Turnover on downs flips possession at the correct spot.
- [ ] A missed FG from the opponent's 30 gives the ball at the correct spot. A miss from inside the 20 gives the ball at the 20.
- [ ] An interception flips the field correctly.
- [ ] Goal to go sets `ydstogo = yardline_100`.
- [ ] The second-half kickoff goes to the right team, and timeouts reset.
- [ ] The two-minute warning stops the clock at 120 seconds, once per half.
- [ ] The game ends at 0:00 of Q4 when not tied. OT rules match Step 1.6.
- [ ] Invariants hold on every snap across 1,000 random games: `1 ≤ yardline_100 ≤ 99`, `1 ≤ down ≤ 4`, `ydstogo ≥ 1`, clock ≥ 0, timeouts in 0–3.

### Phase 1 acceptance

Simulate 5,000 games between two league-average teams (`nflsim sim-game --home AVG --away AVG --n 5000`). Compare the simulated averages to the Phase 0 targets for 2021–2025 and save the table in `PROGRESS.md`.

| Metric | Allowed gap vs real |
| --- | --- |
| Points per team-game | ±1.5 |
| Offensive plays per team-game | ±3 |
| Pass call rate | ±2 points |
| Completion rate | ±2 points |
| Sack rate per dropback | ±1 point |
| Interception rate per attempt | ±0.5 points |
| Punts per team-game | ±0.5 |
| FG attempts per team-game | ±0.4 |
| Share of games going to OT | between 3% and 9% |
| Standard deviation of the final margin | ±1.5 points of real |

If a metric misses, find the cause, usually the clock (too many or too few plays) or the 4th-down table. Do not tune the numbers by hand to hit the target. Record each fix in `DECISIONS.md`.

## Phase 2: Trained decision models

This phase replaces the lookup tables with logistic regression models (binary or multinomial) trained on play-by-play data. They use continuous situation features, so they handle rare situations smoothly. Swap them in one at a time, and re-run the Phase 1 acceptance check after each swap.

### Why logistic regression

Logistic models output probabilities that always sit between 0 and 1 and always sum to 1. Adjustments added in logit space (a team effect, a player effect) can never push a probability out of bounds. The model is also just a dot product plus a softmax, so the fast engine in Phase 5 can evaluate it in a few lines of NumPy or Numba. Do not use gradient-boosted trees or neural nets in v1.

### Step 2.1: Shared feature builder (`models/features.py`)

One function builds the situation features from a state (simulation) or from a row (training). Both paths must use the **same code**, so training and simulation cannot drift apart. Write a test that builds features from 1,000 real plays both ways and checks they are identical.

| Feature | Definition |
| --- | --- |
| `down_1, down_2, down_3, down_4` | One-hot down |
| `ydstogo_c` | `min(ydstogo, 20)` |
| `log_ydstogo` | `log(ydstogo)` |
| `yardline` | `yardline_100 / 100` |
| `yardline_sq` | its square |
| `goal_to_go` | 0/1 |
| `red_zone, inside_5` | 0/1 |
| `score_diff_c` | offense score minus defense score, clipped to ±28, divided by 28 |
| `half_secs` | `half_seconds_remaining / 1800` |
| `under_2min_half` | 0/1 |
| `trailing_late` | 0/1: trailing and under 8:00 of Q4 |
| `leading_late` | 0/1: leading and under 8:00 of Q4 |
| `score_x_time` | `score_diff_c × (1 - game_secs/3600)`, the key late-game interaction |
| `off_timeouts, def_timeouts` | 0–3 |
| `is_home` | 0/1 (0 at neutral sites) |

Model-specific extras (for example `air_yards` for the pass-outcome model) are added per model. Phases 3 and 4 add team and player features to this same builder.

### Step 2.2: The models

Train on regular-season and playoff plays from `training_seasons`. Exclude plays where the win probability before the snap is below 2% or above 98%, unless the model is about late-game behavior (D1, D2, D14, D16).

| Model | Replaces | Type | Target | Extra features |
| --- | --- | --- | --- | --- |
| `play_call` | D1 | binary | pass (incl. scramble) vs run | base |
| `fourth_down` | D2 | multinomial | go / punt / field\_goal | base + `fg_distance`, `fg_distance_sq` |
| `air_yards` | D3 | multinomial | the 7 air-yard bins | base |
| `pass_outcome` | D4 | multinomial | the 5 outcomes | base + `air_yards_c` (clip -10 to 50) + its square |
| `yac` | D5 | multinomial | the 9 YAC bins | base + `air_yards_c` |
| `rush_yards` | D8 | multinomial | the rush-yard classes | base |
| `fumble_lost` | D9 | binary | lost fumble | one-hot play type + `yards_gained_c` |
| `fg_make` | D11 | binary | made | `kick_distance`, its square, `roof_dome`, `wind_c` |
| `pat_choice` | D14 | binary | go for two | score diff after the TD (one-hot -15..+15, else clipped), `half_secs` |
| `clock_runoff` | D16 | multinomial | the seconds bins | base + `clock_stopper` + `play_was_pass` + `out_of_bounds` |
| `presnap_penalty` | D17 | multinomial | none / offense / defense | base |

Keep D6, D7, D10, D12, D13 and D15 as empirical tables. They are simple distributions, and a model adds little.

### Step 2.3: Training procedure (`models/train.py`)

For each model:

1. Build X and y with the feature builder. Standardize features (save the mean and std).
2. Split by **season**: train on all training seasons except the last, and validate on the last. Never split randomly by play, because plays within a game are correlated.
3. Fit `LogisticRegression` with `max_iter=5000`. Use a binary model for 2 classes and a multinomial model (the default `lbfgs` solver) for more. Tune `C` over \[0.01, 0.1, 1, 10\] on validation log loss.
4. Refit on all training seasons with the best `C`.
5. Save to `artifacts/models/<name>.json`: classes, feature names, means, stds, coefficient matrix, intercepts, `C`, training seasons, validation log loss, and a training timestamp.

### Step 2.4: Inference without scikit-learn

Write `TrainedModel.proba()` with NumPy only: standardize, compute `logits = X @ W.T + b`, then a softmax (a sigmoid for binary). **Test:** on 10,000 validation rows, this must equal scikit-learn's `predict_proba` within 1e-9. Phase 5 depends on this.

### Step 2.5: Evaluation report (`artifacts/models/report.md`)

For each model, record:

- Validation log loss vs the Phase 1 lookup table, and vs a constant (base-rate) model. The trained model must beat both. If it does not beat the table, keep the table for that decision and log why.
- A calibration table: bucket predictions into 10 deciles and compare the mean prediction to the observed rate. Every decile with at least 500 plays must be within 3 points.
- The 5 largest coefficients, with a one-line sanity read. For example, `play_call` should show passing rising with `ydstogo` and with `trailing_late`. If a sign looks backwards, investigate before moving on.

### Phase 2 acceptance

- [ ] All models are trained, saved and listed in the report, with log loss and calibration.
- [ ] Each trained model beats its lookup table on validation, or the table is kept with a logged reason.
- [ ] The NumPy inference test passes.
- [ ] The Phase 1 league-average check still passes with all trained models swapped in.
- [ ] A sanity scenario run: a team down 10 with 3:00 left in Q4 passes more than 80% of the time in simulation, and a team up 10 runs more than 60% of the time.

## Phase 3: Team strength features

This phase makes the Chiefs different from the Panthers. The design is **one league-wide model per decision, with team-quality features added as inputs**. Do not train a separate model per team: about 1,000 plays per team-season is far too little, and per-team models will overfit. The models learn how much team quality matters, and each team gets its own behavior through its own feature values.

### Step 3.1: Team ratings (`features/team_strength.py`)

For each team and each game, compute the ratings **as of the morning of that game**, using only earlier games. Store them in `data/processed/team_ratings.parquet` keyed by `(team, season, week)`.

| Rating | Side | Definition (per play, from pbp) |
| --- | --- | --- |
| `pass_epa` | offense | mean EPA on dropbacks |
| `rush_epa` | offense | mean EPA on designed runs |
| `proe` | offense | pass rate over expected: actual pass calls minus the `play_call` model's prediction (situation-only version), in neutral situations (win prob 20–80%, Q1–Q3) |
| `adot` | offense | mean air yards on attempts |
| `sack_rate` | offense | sacks per dropback |
| `cpoe` | offense | completion rate minus the `pass_outcome` model's expected rate |
| `explosive_rate` | offense | share of plays gaining 20+ yards |
| `pace` | offense | median `seconds_elapsed` in neutral situations |
| `pass_epa_allowed` | defense | mean EPA allowed on opponent dropbacks |
| `rush_epa_allowed` | defense | mean EPA allowed on opponent runs |
| `sack_rate_forced` | defense | sacks per opponent dropback |
| `int_rate_forced` | defense | interceptions per opponent attempt |
| `fourth_aggr` | coach | go rate on 4th down minus the situation-only `fourth_down` model's expected go rate |

**Weighting:**

1. Within a season, use an exponentially weighted mean over games, with half-life `half_life_games` (default 8). Weight each game by its number of plays.
2. At the start of a season, the prior is the team's final rating from last season, regressed toward the league mean by `prior_season_regression` (default 0.33). Blend the prior with this season's games as if the prior were worth 6 games. The prior's weight then shrinks as real games accumulate.
3. Store every rating as a z-score relative to the league for that week. This keeps the scales comparable across seasons.

**Leakage test (required):** for a random sample of 200 `(team, week)` rows, recompute the rating using only plays with `game date < that week's date`, and assert an exact match. Add a second test that changing a play in week W does not change any rating at week ≤ W.

### Step 3.2: Wire ratings into the models

Each model gets the offense's and the defense's relevant ratings as extra features. Retrain every model in `models/train.py` with them.

| Model | Offense features | Defense features |
| --- | --- | --- |
| `play_call` | `proe` | none |
| `fourth_down` | `fourth_aggr` | none |
| `air_yards` | `adot` | `pass_epa_allowed` |
| `pass_outcome` | `cpoe`, `sack_rate`, `pass_epa` | `sack_rate_forced`, `int_rate_forced`, `pass_epa_allowed` |
| `yac` | `pass_epa`, `explosive_rate` | `pass_epa_allowed` |
| `rush_yards` | `rush_epa`, `explosive_rate` | `rush_epa_allowed` |
| `clock_runoff` | `pace` | none |

The training rows must use the ratings as they were **before** each play's game. Join on `(posteam, season, week)` and `(defteam, season, week)` from the as-of table. Never join on end-of-season ratings.

### Step 3.3: Home field

`is_home` is already in the feature builder. Check that the trained models produce a home advantage in simulation. The expected range is 1 to 2.5 points of average margin in recent seasons. Measure the real value from the data rather than assuming it. If it falls outside the real range, add a small home logit offset to `pass_outcome` and `rush_yards` and fit it to the real home margin. Log that in `DECISIONS.md`.

### Step 3.4: Weather and venue (optional in v1)

Add `wind_c` and `roof_dome` to `pass_outcome`, `air_yards` and `fg_make`. Upcoming-game weather comes from a manual input file, `data/inputs/weather_<season>_wk<week>.csv`, with columns `game_id, temp, wind, roof`. If the file is missing, assume neutral weather and print a warning.

### Phase 3 acceptance

Run a quick backtest on the 2025 season. Simulate every game 2,000 times with week-appropriate ratings.

- [ ] The leakage tests pass.
- [ ] The correlation between the simulated mean margin and the actual margin is at least 0.25. For reference, the market's closing spread correlates at roughly 0.4–0.45, so record the market's correlation from `spread_line` too.
- [ ] The correlation between the simulated mean margin and the closing spread is at least 0.7. Very low correlation means team strength isn't flowing through.
- [ ] The simulated home margin is within the real range.
- [ ] The spread of team quality is realistic: the standard deviation of simulated mean margins across games is within ±25% of the closing spreads' standard deviation.

## Phase 4: Players and props

This phase decides **who** gets each touch and how efficient each player is, so every simulated game produces a full player box score. Props come straight from those box scores.

### Step 4.1: Actives (`features/actives.py`)

For each game, build the list of active players per team. Save it to `data/inputs/actives_<season>_wk<week>.json`.

1. Start from the weekly roster for the team (QB, RB, WR, TE, K).
2. Remove anyone listed **Out**, **IR**, or **Doubtful** in the injury data. Keep **Questionable** players by default.
3. If the existing FanDuel client exposes who is playing, use it as a cross-check. If a player has FanDuel props posted, treat him as active.
4. Apply manual overrides last from `data/inputs/overrides_<season>_wk<week>.csv`. Columns: `team, player_id, status (active/out), starter_qb (true/false), note`. Overrides always win.
5. Exactly one starting QB per team: the override if given, else the QB with the most dropbacks in the team's last 3 games who is active.
6. Print a warning listing every player whose status is Questionable or whose share changed by more than 10 points from last week.

### Step 4.2: Usage shares (`features/players.py`)

For each active player, compute shares **as of before the game**, using the Phase 3 exponential weighting (half-life 4 games for usage, since roles change fast):

| Share | Definition |
| --- | --- |
| `target_share` | player's targets / team targets |
| `carry_share` | player's designed runs / team designed runs (QB designed runs included) |
| `deep_target_share` | share of the team's targets with air yards ≥ 15 |
| `short_target_share` | share of the team's targets with air yards < 5 |
| `rz_target_share`, `rz_carry_share` | same, inside the 20 |

**Renormalize** across actives. If the WR1 is out, his share is redistributed to the others in proportion to their own shares, blended 50/50 with a position-based prior (WR / TE / RB). Rookies and players with fewer than 20 team plays get the position-average share for their depth-chart slot, from the snap-count data.

### Step 4.3: Choosing the player on each play

- **Pass:** after air yards are sampled, pick the target with probability proportional to `target_share × bin_fit`. `bin_fit` comes from the player's share of targets in that air-yard band (deep, intermediate, short) relative to his overall share, shrunk toward 1. This makes deep shots go to deep threats and short throws to slot WRs, TEs and RBs.
- **Red zone:** inside the 20, use `rz_target_share` / `rz_carry_share` blended 50/50 with the normal shares.
- **Run:** pick the ball carrier from `carry_share`.
- **Passer:** always the starting QB. Scrambles credit the QB with rushing yards.
- The player is chosen **before** the outcome and yards are sampled, so his efficiency can feed into them.

### Step 4.4: Player efficiency

Shrink each player's efficiency toward his position average with `player_shrinkage_k` pseudo-plays (default 50). For example, a WR with 40 targets keeps 40/90 of his own rate.

| Effect | Player | Feeds model as feature |
| --- | --- | --- |
| `catch_rate_oe` | receiver | `pass_outcome` (complete vs incomplete) |
| `yac_oe` | receiver | `yac` |
| `rush_yards_oe` | rusher | `rush_yards` |
| `qb_cpoe`, `qb_sack_rate`, `qb_int_rate`, `qb_scramble_rate` | QB | `pass_outcome` |
| `qb_adot` | QB | `air_yards` |

"OE" means over expected: actual minus the situation-only model's prediction. Retrain the affected models with these features added. Training rows use as-of-before-the-game values, the same leakage rule as Phase 3.

If the starting QB differs from the QB who generated most of the team's recent `pass_epa`, the team passing ratings must not silently credit the backup with the starter's quality. Blend the team ratings 50/50 with a rating that uses the new QB's own history, or a backup-QB prior if he has fewer than 100 dropbacks. Log the blend in the run output.

### Step 4.5: Kicker (optional)

Add a shrunk kicker make rate over expected to `fg_make`. Shrink heavily (k = 100 attempts).

### Step 4.6: Player box score

Per simulated game, per player, track: pass attempts, completions, passing yards, passing TDs, interceptions, rush attempts, rushing yards, rushing TDs, targets, receptions, receiving yards, receiving TDs, longest reception, longest rush, and fumbles lost. Also track the **anytime TD** flag (any rushing or receiving TD).

In the fast engine (Phase 5), store these as fixed arrays `[n_sims, n_players, n_stats]`, not dictionaries.

### Phase 4 acceptance

Backfill 2025 weeks 4–18 with as-of player inputs, and simulate 2,000 times per game.

- [ ] **Calibration:** for every player with at least 20 game appearances, record where the real stat fell in his simulated distribution (the percentile). Across all players and stats (passing, rushing, receiving yards, receptions), the percentiles should be roughly uniform: 40–60% below the simulated median, and 70–90% inside the 10th–90th percentile band.
- [ ] **Team consistency:** the sum of player receiving yards equals the QB's passing yards in every simulated game. Sack yards are subtracted only from team net passing yards, never from the QB's passing yards, which matches how sportsbooks settle passing-yard props. The sum of player rushing yards equals team rushing yards.
- [ ] **Leakage test:** shares and efficiencies for week W use only games before week W.
- [ ] **Injury scenario:** removing a team's WR1 visibly raises the other receivers' simulated targets, and the team's total targets stay within 1 of before.

## Phase 5: Fast simulation and fair odds

This phase makes the engine fast enough to run 10,000 simulations of every game each week, then turns the results into fair probabilities and fair American odds.

### Step 5.1: Fast engine (`sim/fast_engine.py`)

The Python engine from Phases 1–4 stays as the **reference implementation**. Do not delete it. Build a second engine for speed:

1. Port the engine to **Numba** (`@njit`). Use plain arrays only: game state as scalars inside the jitted loop, model coefficients as NumPy arrays packed by `models/registry.py`, and lookup tables as arrays. No dicts, dataclasses or Python objects inside jitted code.
2. Pack each game's context (team ratings, player shares, player efficiencies, actives) into fixed-shape arrays before the jitted call. Players get integer indices per game, mapped back to IDs afterward.
3. Run the simulations in a `prange` loop over sims with one RNG stream per sim, derived from the seed, so results are reproducible regardless of thread count.
4. Output arrays: `home_score[n]`, `away_score[n]`, `went_ot[n]`, plus `player_stats[n, n_players, n_stats]`.

**Equivalence test (required):** run the reference engine and the fast engine with 20,000 sims each on 5 different matchups. For each matchup, the means of home score, away score, margin, total and five player stats must agree within 2 standard errors. Margin and total distributions must also pass a two-sample KS test (p > 0.01). Any mismatch is a bug in one engine. Find it, don't loosen the test.

**Speed target:** 10,000 sims of one game in under 30 seconds on Luke's Mac, excluding the one-time Numba compile. Report the actual time in `PROGRESS.md`.

### Step 5.2: Run a week (`sim/runner.py`, `nflsim simulate`)

1. Load the schedule for the season and week. Skip games already played.
2. Build or load team ratings, actives and player inputs as of that week.
3. Simulate each game `sims_per_game` times with seed `seed + hash(game_id)`.
4. Save raw results to `data/outputs/<season>_wk<week>/sims_<game_id>.npz`.
5. Save the fair-odds table (Step 5.3) to `data/outputs/<season>_wk<week>/fair_odds.parquet`.

### Step 5.3: Fair probabilities (`sim/aggregate.py`)

With margin = home score − away score and total = home score + away score over n sims:

| Market | Formula |
| --- | --- |
| Moneyline (home) | `P(home win) = mean(margin > 0)`; also store `P(tie) = mean(margin == 0)` |
| Moneyline, ties as push | \`P(home win |
| Spread, home line L (e.g. −3.5) | `P(home covers) = mean(margin + L > 0)`; `P(push) = mean(margin + L == 0)` for whole-number lines |
| Total, line T | `P(over) = mean(total > T)`; `P(push) = mean(total == T)` |
| Team total | same as total, using one team's score |
| Player over/under, line X | `P(over) = mean(stat > X)` |
| Anytime TD | `mean(td_flag)` |
| First half | only if half scores are tracked; otherwise skip |

For whole-number lines with a push, compare using the probability **given no push**: `P(cover) / (1 - P(push))`.

Compute spreads for every line from −20.5 to +20.5 in steps of 0.5, and totals from 30.5 to 60.5. For each player and stat, store the 5th–95th percentiles in steps of 5, and `P(over)` at every 0.5 line within that range.

**Monte Carlo error:** with 10,000 sims, a 50% probability has a standard error of about 0.5 points. Store `se = sqrt(p(1-p)/n)` beside every probability so Phase 6 can ignore edges smaller than the noise.

### Step 5.4: Fair American odds

```
if p >= 0.5: american = -100 * p / (1 - p)
else:        american = 100 * (1 - p) / p
```

Round to the nearest whole number for display. Keep the raw probability for every calculation.

### Phase 5 acceptance

- [ ] The equivalence test passes on 5 matchups.
- [ ] Speed target met, and timing recorded.
- [ ] `nflsim simulate` for a past week writes `fair_odds.parquet` for every game.
- [ ] Sanity: each game's `P(home covers −L)` decreases monotonically as L grows, and `P(over T)` decreases monotonically as T grows. P(home win) + P(away win) + P(tie) = 1.

## Phase 6: FanDuel comparison and weekly dashboard

This phase pulls FanDuel's odds, removes the vig, lines them up against the model's fair probabilities, and ranks every market by how much the two disagree.

### Step 6.1: Pull odds (`odds/fanduel.py`)

Wrap the **existing** FanDuel client in the repo. Do not write a new scraper. The wrapper returns one normalized table, saved to `data/outputs/<season>_wk<week>/fanduel_<timestamp>.parquet`:

| Column | Example |
| --- | --- |
| `pulled_at` | 2026-10-15T14:02:00 |
| `game_id` | nflverse game ID, after matching |
| `market` | `moneyline`, `spread`, `total`, `team_total`, `player_pass_yds`, `player_rush_yds`, `player_rec_yds`, `player_receptions`, `player_pass_tds`, `player_anytime_td`, … |
| `selection` | `home`, `away`, `over`, `under`, `yes` |
| `team` | nflverse abbreviation |
| `player_id` | nflverse `gsis_id`, after matching (null for game markets) |
| `line` | −3.5, 47.5, 64.5, … |
| `american_odds` | −110 |
| `fd_market_id`, `fd_selection_id` | raw FanDuel IDs, for debugging |

Keep every pull, so line movement can be studied later. The latest pull is used by default.

### Step 6.2: Matching (`odds/matching.py`)

1. **Teams:** map FanDuel names to nflverse abbreviations with `config/team_aliases.yaml`. Fail loudly on any unknown name.
2. **Games:** match on the two teams plus a game date within ±1 day.
3. **Players:** normalize names (lowercase, remove punctuation and suffixes like Jr., Sr., II, III), then match within the same team against the active roster. Fall back to fuzzy matching (similarity ≥ 0.9) only within the same team. Keep a manual mapping file, `config/player_aliases.yaml`, for stubborn cases.
4. Write every unmatched market to `unmatched.csv` and print a count. Never guess a match below the threshold.

### Step 6.3: Remove the vig (`odds/devig.py`)

1. Convert American odds to implied probability. For negative odds, `q = -odds / (-odds + 100)`. For positive odds, `q = 100 / (odds + 100)`.
2. **Two-sided markets** (both sides posted at the same line): `fair = q_side / (q_over + q_under)`. This is the multiplicative method. Also compute the power method for reference: find k such that `q_over^k + q_under^k = 1`, then `fair = q_side^k`. Store both, and use the multiplicative one by default.
3. Store the **hold**: `q_over + q_under - 1`.
4. **One-sided markets** (for example anytime TD, where only "yes" is posted): you cannot de-vig them exactly. Estimate with `fair = q / (1 + assumed_hold)`, where `assumed_hold` is in config (default 0.08). Mark the row `one_sided = true` so the dashboard shows it separately.
5. Moneylines: compare with the model's tie-as-push probability (Phase 5).

### Step 6.4: Compare (`odds/compare.py`)

For each matched selection, look up the model's probability for **that exact line**. Then compute:

| Column | Formula |
| --- | --- |
| `p_model` | the model's probability for this selection at this line, push-adjusted |
| `p_market` | FanDuel no-vig probability |
| `edge` | `p_model - p_market` |
| `decimal_odds` | from American: `1 + odds/100` if positive, `1 + 100/-odds` if negative |
| `ev_per_dollar` | `p_win × (decimal_odds - 1) - p_lose`, where p\_win and p\_lose exclude pushes |
| `model_fair_american` | Phase 5 formula |
| `mc_se` | the model's Monte Carlo standard error |
| `flag` | see below |

**Flags:** these mark rows to double-check, not bets.

- `CHECK_INPUTS` when the absolute edge exceeds `edge_flag_threshold` (default 10 points). Gaps that big are usually a model input problem: a wrong active list, a QB change or stale ratings.
- `NOISE` when the absolute edge is below 2 × `mc_se`.
- `QUESTIONABLE_PLAYER` when the player, or a key teammate at his position, is Questionable.
- `ONE_SIDED` for de-vig estimates.

Save the result to `data/outputs/<season>_wk<week>/comparison.parquet`.

### Step 6.5: Dashboard (`report/dashboard.py`)

Write one self-contained HTML file, `data/outputs/<season>_wk<week>/report.html`, using the Jinja template. No server is needed. Open it in a browser.

1. **Header:** season, week, the time of the odds pull, sims per game, model version (git commit hash), and the count of unmatched markets.
2. **Games table:** one row per game with the model's win probability, fair spread (the line where cover probability is 50%), fair total, and FanDuel's spread, total and moneyline, plus each one's edge.
3. **Biggest disagreements:** every comparison row, sorted by EV per dollar descending, with filters for market type, game and flag. Use plain inline JavaScript for sorting and filtering; no external libraries are required.
4. **Player props tab:** grouped by game, showing each prop's model median, FanDuel line, model `P(over)`, FanDuel no-vig `P(over)`, and edge.
5. **Per-game detail:** a small histogram of margin and total (inline SVG or base64 matplotlib PNG), with the FanDuel spread and total marked.
6. A footer stating that these are model estimates, not guarantees, and that flagged rows need a manual check.

### Step 6.6: One command for the whole week

`nflsim week --season 2026 --week 7` runs, in order: ingest (current season refresh) → features → simulate → compare → report. It prints the report path at the end, plus the top 15 rows by EV that have no `CHECK_INPUTS` or `NOISE` flag.

### Phase 6 acceptance

- [ ] For a current week, at least 95% of FanDuel game markets and at least 85% of player props match.
- [ ] De-vig unit tests pass: −110/−110 gives 0.5/0.5 with a hold of about 4.76%; −150/+130 gives the correct fair split.
- [ ] EV unit tests pass, including a whole-number spread push case.
- [ ] The dashboard renders, sorts and filters in a browser.
- [ ] The median absolute edge on main spreads and totals is small (under about 4 points). If the model disagrees with the market on almost everything, something is miscalibrated. Investigate before trusting any edge.

## Phase 7: Backtesting and validation

This phase answers one question: would this model's disagreements with the market have been right in the past? The benchmark is the **closing line**. It is the sharpest public estimate available, so a model that cannot get close to it on average has no real edges yet.

### Step 7.1: Walk-forward backtest (`backtest/walkforward.py`)

For each test season S in `--seasons` (default 2023, 2024, 2025):

1. **Retrain** every Phase 2–4 model on seasons before S only. Save to `artifacts/models/backtest_<S>/`.
2. For each week W of S, build team ratings and player inputs as of before W. Use the actives that actually played (from play-by-play participation) to isolate model quality from injury-guessing. Run a second variant with the injury report as of Thursday of that week, to measure realistic performance.
3. Simulate every game (5,000 sims is enough for backtests).
4. Record the model's win probability, fair spread, fair total, and the full margin and total distributions.

The closing lines come from the schedules table: `spread_line`, `total_line`, `home_moneyline`, `away_moneyline`. Check the sign convention of `spread_line` (in nflverse it is typically the home team's expected margin, positive when home is favored), verify it on a few known games, and record it in `DECISIONS.md`.

### Step 7.2: Metrics (`backtest/metrics.py`)

| Metric | Model | Compare to |
| --- | --- | --- |
| Win probability log loss | model P(home win) vs results | the no-vig closing moneyline's log loss |
| Win probability Brier score | same | same |
| Margin mean absolute error | model mean margin vs actual | closing spread vs actual |
| Total mean absolute error | model mean total vs actual | closing total vs actual |
| Calibration | 10-bucket reliability table for win probability | perfect calibration |
| Spread disagreement test | ATS record when the model's fair spread differs from the close by ≥ 1.5, ≥ 2.5, ≥ 3.5 points | 50% after removing pushes; also report the 52.4% break-even at −110 |
| Total disagreement test | the same for totals | same |
| Margin distribution check | PIT histogram of actual margins in the model's distribution | uniform |
| Prop calibration | from Phase 4, for the backtest seasons | uniform PIT |

For each disagreement bucket, report the **sample size** and a 95% confidence interval for the win rate. With 100 bets, the interval is about ±10 points. Say so plainly in the report rather than celebrating a small-sample result.

Historical player prop lines are not in nflverse. Evaluate props only through calibration, unless Luke supplies historical prop odds later.

### Step 7.3: Report (`data/outputs/backtest/report.html`)

Include every metric table, a reliability plot, PIT histograms, and the ATS results per bucket with sample sizes and intervals. End with one plain-English summary paragraph: whether the model is close to the market, better or worse, and in which markets.

### Step 7.4: Rules for interpreting results

Put these rules at the top of the backtest report and follow them in any summary:

1. The model is not expected to beat the closing line overall. Being close to it (log loss within about 0.01, margin MAE within about 0.3 points) means it is a sound engine.
2. Edges are only worth studying in a market or situation where the backtest shows a consistent, sizable, well-sampled advantage.
3. Never tune the model on the test season. Any change made after looking at backtest results must be re-tested on a season not used to make that change.
4. Report bad results as clearly as good ones.

### Phase 7 acceptance

- [ ] Walk-forward runs for 3 seasons with retraining per season, and the leakage tests pass in backtest mode.
- [ ] The report contains every metric in Step 7.2, with sample sizes and intervals.
- [ ] `PROGRESS.md` has a 5-line summary of where the model stands against the closing line.

## Appendix: formulas, glossary, and the per-phase prompt

### A. Formulas in one place

```latex
\text{logit}(p) = \ln\frac{p}{1-p}, \qquad \sigma(z) = \frac{1}{1+e^{-z}}
```

```latex
P(\text{class } k) = \frac{e^{z_k}}{\sum_j e^{z_j}}, \qquad z_k = \mathbf{w}_k \cdot \mathbf{x} + b_k
```

```latex
q = \begin{cases} \frac{-o}{-o+100} & o < 0 \\ \frac{100}{o+100} & o > 0 \end{cases} \qquad \text{fair}_i = \frac{q_i}{\sum_j q_j}
```

```latex
\text{EV per dollar} = p_{win}(d-1) - p_{lose}, \qquad SE = \sqrt{\frac{p(1-p)}{n}}
```

Here o is the American odds, d is the decimal odds, and n is the number of simulations.

### B. Glossary

| Term | Meaning |
| --- | --- |
| Monte Carlo | Running a random simulation many times and counting outcomes to get probabilities |
| EPA | Expected points added: how much a play changed the offense's expected points |
| PROE | Pass rate over expected, given the situation |
| CPOE | Completion percentage over expected, given air yards and situation |
| aDOT | Average depth of target (mean air yards) |
| Vig / hold | The sportsbook's margin: implied probabilities add up to more than 100% |
| No-vig probability | The book's implied probability with the margin removed |
| Closing line | The final odds before kickoff, the market's sharpest estimate |
| PIT | Probability integral transform: where the real result fell in the predicted distribution. Uniform PIT means well-calibrated |
| Leakage | Using information from the future (relative to the game being predicted) during training or features |
| Shrinkage | Pulling a small-sample estimate toward an average, in proportion to how little data there is |

### C. Common mistakes to avoid

1. Joining end-of-season team or player stats onto training plays. This is leakage and will make every metric look great and every live prediction worse.
2. Splitting train and validation randomly by play instead of by season.
3. Letting the training feature builder and the simulation feature builder diverge.
4. Forgetting to cap yards at the end zone, which produces impossible plays and inflated scoring.
5. Getting the clock wrong. This is the most common reason for simulated scoring being too high or too low.
6. Treating scrambles as designed runs. This distorts both the play-call and the rushing models.
7. Comparing the model to FanDuel's raw implied probability instead of the no-vig one.
8. Comparing at a different line than FanDuel's (for example, the model's −3 against FanDuel's −3.5).
9. Trusting a large edge without checking actives, the starting QB and recent news first.
10. Hand-tuning numbers to pass an acceptance check instead of finding the bug.

### D. Prompt to start each phase in Claude Code

Paste this at the start of each phase, filling in the phase number:

```
Read the full build spec (NFL Monte Carlo Simulation Engine — Build Spec), plus PROGRESS.md and DECISIONS.md.
We are on Phase <N>. Follow its steps in order and do not start Phase <N+1>.
Follow every ground rule in "How to use this doc", especially the no-leakage rule.
When done, run every Phase <N> acceptance check, put the results in PROGRESS.md,
commit with message "phase <N>: <short description>", and give me a summary of
what you built, the acceptance results, and anything you logged in DECISIONS.md.
```

### E. Sources

- [Modeling Football: Combining ML models and Monte Carlo simulations](https://medium.com/draftkings-engineering/modeling-football-combining-ml-models-and-monte-carlo-simulations-170fd58cbe11) (DraftKings Engineering): the play-by-play design this spec is based on.
- [nflreadpy on PyPI](https://pypi.org/project/nflreadpy/): the nflverse data package used in Phase 0.
