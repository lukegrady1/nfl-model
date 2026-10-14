"""Step 2.3 / 2.5: train the logistic decision models and write the evaluation report."""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import polars as pl
import typer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from nflsim.config import MODELS_DIR, settings
from nflsim.data.clean import load_plays, load_schedules
from nflsim.engine.state import Context, GameState, OTState, Phase
from nflsim.models import buckets as B
from nflsim.models.features import PLAY_TYPES, feature_matrix, feature_names, features, situation_from_state
from nflsim.models.registry import KEY_FNS, TRAINABLE
from nflsim.models.tables import TableModel, build_all_tables
from nflsim.models.trained import TrainedModel, apply_mask, feasible_mask, model_path, room_for

C_GRID = [0.01, 0.1, 1.0, 10.0]
LATE_GAME_MODELS = {"play_call", "fourth_down", "pat_choice", "clock_runoff"}   # no win-probability filter
CLASSES = {
    "play_call": ["pass", "run"], "fourth_down": ["go", "punt", "field_goal"],
    "dropback_outcome": ["throw", "sack", "scramble"], "throw_outcome": ["complete", "incomplete", "interception"],
    "air_yards": list(range(7)), "yac": list(range(9)), "rush_yards": B.RUSH_CLASSES,
    "fumble_lost": ["no", "yes"], "fg_make": ["missed", "made"], "pat_choice": ["kick", "two"],
    "clock_runoff": list(range(10)), "presnap_penalty": ["none", "offense", "defense"],
}
ROW_COLS = ["receiver_player_id", "rusher_player_id", "passer_player_id", "kicker_player_id", "game_id", "play_id", "season", "season_type", "home_team", "away_team", "posteam", "defteam", "down",
            "ydstogo", "yardline_100", "qtr", "game_seconds_remaining", "half_seconds_remaining", "score_differential",
            "posteam_score", "defteam_score", "posteam_timeouts_remaining", "defteam_timeouts_remaining", "roof", "wind",
            "air_yards", "yards_after_catch", "kick_distance", "yards_gained", "clock_stopper", "play_class",
            "pass_outcome", "out_of_bounds", "wp", "field_goal_result", "fumble_lost", "seconds_elapsed",
            "next_play_is_timeout", "next_is_marker", "next_is_two_minute_warning", "next_qtr", "penalty_team",
            "penalty_yards", "neutral", "touchdown"]


def _f(col: str) -> pl.Expr:
    return pl.col(col).fill_null(0)


def training_plays(team: bool = False) -> pl.DataFrame:
    """Training rows; with team=True the as-of team ratings are joined on (posteam|defteam, season, week)."""
    s = settings()
    plays = load_plays().filter(pl.col("season").is_in(s["training_seasons"]) & pl.col("season_type").is_in(["REG", "POST"]))
    loc = load_schedules().select("game_id", neutral=(pl.col("location") == "Neutral"))
    plays = plays.join(loc, on="game_id", how="left").with_columns(pl.col("neutral").fill_null(False))
    if team:
        from nflsim.features.team_strength import RATINGS, load_team_ratings

        tr = load_team_ratings().select(["team", "season", "week"] + RATINGS)
        off = tr.rename({"team": "posteam", **{r: f"off_{r}" for r in RATINGS}})
        deff = tr.rename({"team": "defteam", **{r: f"def_{r}" for r in RATINGS}})
        plays = plays.join(off, on=["posteam", "season", "week"], how="left").join(deff, on=["defteam", "season", "week"], how="left")
        plays = plays.with_columns([pl.col(f"off_{r}").fill_null(0.0) for r in RATINGS] + [pl.col(f"def_{r}").fill_null(0.0) for r in RATINGS])
        plays = join_player_effects(plays)
    return plays


def join_player_effects(plays: pl.DataFrame) -> pl.DataFrame:
    """Phase 4: as-of shrunk player efficiencies joined by the receiver, rusher, passer and kicker of each play."""
    from nflsim.features.players import EFF_PATH, load_player_eff

    if not EFF_PATH.exists():
        return plays
    from nflsim.features.players import qb_id_expr

    eff = load_player_eff()
    plays = plays.with_columns(qb_player_id=qb_id_expr())
    specs = [("receiver_player_id", {"catch_rate_oe": "p_rec_catch_rate_oe", "yac_oe": "p_rec_yac_oe"}),
             ("rusher_player_id", {"rush_yards_oe": "p_rush_yards_oe"}),
             ("qb_player_id", {k: f"p_{k}" for k in ("qb_cpoe", "qb_sack_rate", "qb_int_rate", "qb_scramble_rate", "qb_adot")}),
             ("kicker_player_id", {"kicker_fg_oe": "p_kicker_fg_oe"})]
    for id_col, mapping in specs:
        if id_col not in plays.columns:
            continue
        sub = eff.select(["player_id", "season", "week"] + list(mapping)).rename({"player_id": id_col, **mapping})
        sub = sub.with_columns(pl.col("season").cast(plays.schema["season"]), pl.col("week").cast(plays.schema["week"]))
        plays = plays.join(sub, on=[id_col, "season", "week"], how="left")
        plays = plays.with_columns([pl.col(c).fill_null(0.0) for c in mapping.values()])
    return plays


def wp_filter(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter((pl.col("wp") > 0.02) & (pl.col("wp") < 0.98))


def dataset(name: str, plays: pl.DataFrame) -> tuple[pl.DataFrame, str, str | None]:
    """Rows for one model with a `target` column (class label) and optional `exact` column for within-bin draws."""
    p = plays
    exact = None
    if name == "play_call":
        d = p.filter(pl.col("play_class").is_in(["pass", "run"]) & pl.col("down").is_between(1, 3)).with_columns(target=pl.col("play_class"))
    elif name == "fourth_down":
        d = p.filter((pl.col("down") == 4) & pl.col("play_class").is_in(["pass", "run", "punt", "field_goal"])).with_columns(
            target=pl.when(pl.col("play_class").is_in(["pass", "run"])).then(pl.lit("go")).otherwise(pl.col("play_class")))
    elif name == "dropback_outcome":
        d = p.filter(pl.col("play_class") == "pass").with_columns(
            target=pl.when(pl.col("pass_outcome").is_in(["sack", "scramble"])).then(pl.col("pass_outcome")).otherwise(pl.lit("throw")))
    elif name == "throw_outcome":
        d = p.filter(pl.col("pass_outcome").is_in(["complete", "incomplete", "interception"]) & pl.col("air_yards").is_not_null()).with_columns(target=pl.col("pass_outcome"))
    elif name == "air_yards":
        d = p.filter(pl.col("pass_outcome").is_in(["complete", "incomplete", "interception"]) & pl.col("air_yards").is_not_null()).with_columns(
            exact=pl.col("air_yards").cast(pl.Int64).clip(-15, 99))
        d = d.with_columns(target=pl.col("exact").map_batches(lambda x: pl.Series(B.vectorize(B.air_bin)(x.to_numpy()))))
        exact = "exact"
    elif name == "yac":
        d = p.filter((pl.col("pass_outcome") == "complete") & pl.col("yards_after_catch").is_not_null() & pl.col("air_yards").is_not_null()).with_columns(
            exact=pl.col("yards_after_catch").cast(pl.Int64).clip(-20, 99))
        d = d.with_columns(target=pl.col("exact").map_batches(lambda x: pl.Series(B.vectorize(B.yac_bin)(x.to_numpy()))))
        exact = "exact"
    elif name == "rush_yards":
        d = p.filter(pl.col("play_class") == "run").with_columns(exact=pl.col("yards_gained").cast(pl.Int64).clip(-25, 99))
        d = d.with_columns(target=pl.col("exact").map_batches(lambda x: pl.Series(B.vectorize(B.rush_class)(x.to_numpy()))))
        exact = "exact"
    elif name == "fumble_lost":
        d = p.filter(pl.col("play_class").is_in(["pass", "run"])
                     & ~pl.col("pass_outcome").is_in(["incomplete", "interception"]).fill_null(False)).with_columns(
            play_type_fumble=pl.when(pl.col("play_class") == "run").then(pl.lit("run"))
            .when(pl.col("pass_outcome") == "complete").then(pl.lit("completion")).otherwise(pl.col("pass_outcome")),
            target=pl.when(_f("fumble_lost") == 1).then(pl.lit("yes")).otherwise(pl.lit("no")))
    elif name == "fg_make":
        d = p.filter((pl.col("play_class") == "field_goal") & pl.col("kick_distance").is_not_null()
                     & pl.col("field_goal_result").is_in(["made", "missed", "blocked"])).with_columns(
            target=pl.when(pl.col("field_goal_result") == "made").then(pl.lit("made")).otherwise(pl.lit("missed")))
    elif name == "pat_choice":
        d = p.filter(pl.col("play_class").is_in(["extra_point", "two_point"])).with_columns(
            target=pl.when(pl.col("play_class") == "two_point").then(pl.lit("two")).otherwise(pl.lit("kick")))
    elif name == "clock_runoff":
        d = p.filter(pl.col("play_class").is_in(["pass", "run", "punt", "field_goal", "penalty_no_play", "spike"])
                     & pl.col("seconds_elapsed").is_not_null() & ~pl.col("next_play_is_timeout") & ~pl.col("next_is_marker")
                     & ~pl.col("next_is_two_minute_warning") & (pl.col("next_qtr") == pl.col("qtr"))).with_columns(
            exact=pl.col("seconds_elapsed").cast(pl.Int64))
        d = d.with_columns(target=pl.col("exact").map_batches(lambda x: pl.Series(B.vectorize(B.seconds_bin)(x.to_numpy()))))
        exact = "exact"
    elif name == "presnap_penalty":
        d = p.filter(pl.col("play_class").is_in(["pass", "run", "punt", "field_goal", "penalty_no_play", "kneel", "spike"])
                     & pl.col("down").is_between(1, 4)).with_columns(
            target=pl.when(pl.col("play_class") != "penalty_no_play").then(pl.lit("none"))
            .when(pl.col("penalty_team") == pl.col("posteam")).then(pl.lit("offense")).otherwise(pl.lit("defense")))
    else:
        raise KeyError(name)
    if name not in LATE_GAME_MODELS:
        d = wp_filter(d)
    d = d.filter(pl.col("yardline_100").is_not_null() & pl.col("game_seconds_remaining").is_not_null())
    return d, "target", exact


def rows_of(df: pl.DataFrame) -> list[dict]:
    cols = [c for c in ROW_COLS + ["play_type_fumble", "target", "exact"] if c in df.columns]
    cols += [c for c in df.columns if c.startswith("off_") or c.startswith("def_") or c.startswith("p_")]
    return df.select(cols).to_dicts()


def encode(df: pl.DataFrame, classes: list) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    return np.array([idx[v] for v in df["target"].to_list()], dtype=int)


def fit_logreg(X: np.ndarray, y: np.ndarray, C: float, n_classes: int) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=5000).fit(X, y)


def standardize_params(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-12] = 1.0
    return mean, std


def within_distributions(df: pl.DataFrame, exact_col: str, classes: list, bin_fn) -> dict:
    out = {}
    for cls in classes:
        vals = df.filter(pl.col("target") == cls)[exact_col].drop_nulls()
        if vals.len() == 0:
            continue
        vc = vals.value_counts().sort(exact_col)
        values = vc[exact_col].to_list()
        if len(values) == 1:
            continue  # exact class, nothing to draw
        cnt = vc["count"].to_numpy().astype(float)
        out[str(cls)] = {"values": [int(v) for v in values], "probs": (cnt / cnt.sum()).tolist()}
    return out


def train_model(name: str, plays: pl.DataFrame, write: bool = True, team: bool = False, out_dir=None, fixed_C: float | None = None) -> dict:
    classes = CLASSES[name]
    df, _, exact = dataset(name, plays)
    seasons = sorted(df["season"].unique().to_list())
    val_season = seasons[-1]
    names = feature_names(name, team)
    typer.echo(f"{name}: {df.height} rows, seasons {seasons[0]}-{seasons[-1]}, validate on {val_season}, {len(names)} features"
               + (" (with team ratings)" if team else ""))
    X = feature_matrix(name, rows_of(df), team)
    y = encode(df, classes)
    val = (df["season"] == val_season).to_numpy()
    mean, std = standardize_params(X[~val])
    Z = (X - mean) / std
    best = None
    for C in ([fixed_C] if fixed_C is not None else C_GRID):
        clf = fit_logreg(Z[~val], y[~val], C, len(classes))
        ll = log_loss(y[val], clf.predict_proba(Z[val]), labels=list(range(len(classes))))
        typer.echo(f"  C={C:<5} val log loss {ll:.5f}")
        if best is None or ll < best[1]:
            best = (C, ll, clf)
    C, val_ll, val_clf = best
    # refit on everything with the best C (standardization from all rows)
    mean_all, std_all = standardize_params(X)
    clf = fit_logreg((X - mean_all) / std_all, y, C, len(classes))
    spec = {
        "name": name, "classes": [str(c) for c in classes],
        "class_type": "int" if isinstance(classes[0], (int, np.integer)) else "str",
        "feature_names": names, "means": mean_all.tolist(), "stds": std_all.tolist(),
        "coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist(), "C": C,
        "training_seasons": seasons, "validation_season": val_season, "val_log_loss": float(val_ll),
        "n_rows": int(df.height), "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
        "team_features": bool(team),
    }
    if exact:
        spec["within"] = within_distributions(df, exact, classes, None)
    # validation-split artefacts for the report (model fit on train seasons only)
    val_spec = {**spec, "means": mean.tolist(), "stds": std.tolist(), "coef": val_clf.coef_.tolist(),
                "intercept": val_clf.intercept_.tolist()}
    if write:
        out_dir = out_dir or MODELS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "val").mkdir(exist_ok=True)
        with open(model_path(name, out_dir), "w") as f:
            json.dump(spec, f)
        with open(model_path(name, out_dir / "val"), "w") as f:
            json.dump(val_spec, f)
    return {"spec": spec, "val_model": TrainedModel(val_spec), "val_df": df.filter(val), "X_val": X[val], "y_val": y[val],
            "y_train": y[~val], "sklearn_val": val_clf, "mean": mean, "std": std}


def reload_fit(name: str, plays: pl.DataFrame, directory=None) -> dict:
    """Rebuild the evaluation inputs from the saved specs without retraining."""
    directory = directory or MODELS_DIR
    with open(model_path(name, directory)) as f:
        spec = json.load(f)
    with open(model_path(name, directory / "val")) as f:
        val_spec = json.load(f)
    classes = CLASSES[name]
    df, _, _ = dataset(name, plays)
    val = (df["season"] == spec["validation_season"]).to_numpy()
    X_val = feature_matrix(name, rows_of(df.filter(val)), bool(spec.get("team_features", False)))
    y = encode(df, classes)
    return {"spec": spec, "val_model": TrainedModel(val_spec), "val_df": df.filter(val), "X_val": X_val, "y_val": y[val],
            "y_train": y[~val]}


# ---- evaluation against the Phase 1 tables and a constant model -------------------------------------------------
def state_from_row(r: dict) -> GameState:
    home, away, pos = r["home_team"], r["away_team"], r["posteam"]
    qtr = int(r["qtr"])
    s = GameState(home=home, away=away, possession=pos, down=int(r["down"] or 1), ydstogo=int(r["ydstogo"] or 10),
                  yardline_100=int(r["yardline_100"]), qtr=qtr, phase=Phase.SCRIMMAGE, is_neutral=bool(r.get("neutral")),
                  game_seconds_remaining=int(r["game_seconds_remaining"]) if qtr <= 4 else 0)
    ps, ds = int(r["posteam_score"] or 0), int(r["defteam_score"] or 0)
    s.score = {pos: ps, s.opponent(pos): ds}
    pt = r["posteam_timeouts_remaining"] if r.get("posteam_timeouts_remaining") is not None else 3
    dtm = r["defteam_timeouts_remaining"] if r.get("defteam_timeouts_remaining") is not None else 3
    s.timeouts = {pos: int(pt), s.opponent(pos): int(dtm)}
    if qtr >= 5:
        s.ot_state = OTState(active=True, first_team=pos, possession_count=1, period=1,
                             seconds_remaining=int(r["game_seconds_remaining"]))
    roof = r.get("roof")
    s.context = Context(weather={"roof_dome": 1 if roof in ("dome", "closed") else 0, "wind": r.get("wind")}, season=r.get("season"))
    return s


def extra_from_row(r: dict, name: str) -> dict:
    e: dict = {}
    if r.get("air_yards") is not None:
        e["air_yards"] = r["air_yards"]
        e["air_bin"] = B.air_bin(int(r["air_yards"]))
    if r.get("kick_distance") is not None:
        e["distance"] = int(min(r["kick_distance"], 79))
    if r.get("play_type_fumble"):
        e["play_type"] = r["play_type_fumble"]
        e["play_type_idx"] = PLAY_TYPES.index(r["play_type_fumble"])
    e["yards"] = r.get("yards_gained")
    e["clock_stopper"] = bool(r.get("clock_stopper"))
    e["play_type_kind"] = {"punt": "punt", "field_goal": "field_goal", "penalty_no_play": "penalty"}.get(r.get("play_class"))
    e["play_was_pass"] = r.get("play_class") == "pass"
    e["out_of_bounds"] = bool(r.get("out_of_bounds") or 0)
    e["offense_diff"] = int(r.get("score_differential") or 0)
    e["under_two"] = int(r["half_seconds_remaining"]) <= 120
    if name == "pat_choice":
        e["diff_after_td"] = int(r.get("score_differential") or 0)
    e["player"] = {k[2:]: v for k, v in r.items() if k.startswith("p_")}
    return e


def table_probs(name: str, tm: TableModel, state: GameState, extra: dict, classes: list) -> np.ndarray:
    """Phase 1 table probabilities mapped onto the trained model's classes (exact yards binned, capped at the goal line)."""
    p = tm.proba(state, extra)
    tcls = tm.classes
    if name in ("air_yards", "yac", "rush_yards"):
        cap = state.yardline_100 if name != "yac" else max(state.yardline_100 - int(extra.get("air_yards", 0)), 0)
        bin_fn = {"air_yards": B.air_bin, "yac": B.yac_bin, "rush_yards": B.rush_class}[name]
        out = np.zeros(len(classes))
        idx = {c: i for i, c in enumerate(classes)}
        for v, pv in zip(tcls, p):
            out[idx[bin_fn(min(int(v), cap))]] += pv
        return out
    if name == "fumble_lost":
        d = dict(zip(tcls, p))
        return np.array([d["no"], d["yes"] + d["yes_td"]])
    if name == "fg_make":
        d = dict(zip(tcls, p))
        return np.array([d["missed"], d["made"]])
    d = dict(zip(tcls, p))
    return np.array([d[c] for c in classes])


def evaluate(name: str, fit: dict, tables: dict) -> dict:
    classes = CLASSES[name]
    K = len(classes)
    rows = rows_of(fit["val_df"])
    y = fit["y_val"]
    P_model = fit["val_model"].proba_from_features(fit["X_val"])
    if name in ("air_yards", "yac", "rush_yards"):   # same goal-line mask the engine applies
        for i, r in enumerate(rows):
            P_model[i] = apply_mask(P_model[i], feasible_mask(name, classes, room_for(name, r["yardline_100"], r.get("air_yards"))))
    # constant (base-rate) model from the training seasons
    base = np.bincount(fit["y_train"], minlength=K) / len(fit["y_train"])
    P_const = np.tile(base, (len(y), 1))
    # Phase 1 table built from training seasons only
    tm = TableModel(tables[name if name != "fumble_lost" else "fumble_lost"], KEY_FNS[name])
    P_table = np.zeros((len(rows), K))
    for i, r in enumerate(rows):
        st = state_from_row(r)
        P_table[i] = table_probs(name, tm, st, extra_from_row(r, name), classes)
    eps = 1e-9
    P_table = np.clip(P_table, eps, 1)
    P_table /= P_table.sum(axis=1, keepdims=True)
    labels = list(range(K))
    res = {
        "n_val": int(len(y)),
        "ll_model": float(log_loss(y, P_model, labels=labels)),
        "ll_table": float(log_loss(y, P_table, labels=labels)),
        "ll_const": float(log_loss(y, P_const, labels=labels)),
    }
    sit_dir = MODELS_DIR / "situation"
    if fit["spec"].get("team_features") and model_path(name, sit_dir / "val").exists():
        with open(model_path(name, sit_dir / "val")) as f:
            sit = TrainedModel(json.load(f))
        P_sit = sit.proba_from_features(feature_matrix(name, rows, False))
        if name in ("air_yards", "yac", "rush_yards"):
            for i, r in enumerate(rows):
                P_sit[i] = apply_mask(P_sit[i], feasible_mask(name, classes, room_for(name, r["yardline_100"], r.get("air_yards"))))
        res["ll_situation"] = float(log_loss(y, P_sit, labels=labels))
    # calibration: binary -> P(class 1); multinomial -> probability of the predicted (modal) class
    if K == 2:
        conf, hit = P_model[:, 1], (y == 1).astype(float)
    else:
        modal = P_model.argmax(axis=1)
        conf, hit = P_model[np.arange(len(y)), modal], (y == modal).astype(float)
    order = np.argsort(conf)
    deciles = []
    for chunk in np.array_split(order, 10):
        deciles.append({"n": int(len(chunk)), "pred": float(conf[chunk].mean()), "obs": float(hit[chunk].mean())})
    res["calibration"] = deciles
    res["calibration_ok"] = all(abs(d["pred"] - d["obs"]) <= 0.03 for d in deciles if d["n"] >= 500)
    # top coefficients (standardized scale)
    W = np.asarray(fit["spec"]["coef"])
    names = fit["spec"]["feature_names"]
    flat = [(abs(W[k, j]), W[k, j], names[j], classes[k] if W.shape[0] > 1 else classes[1]) for k in range(W.shape[0]) for j in range(W.shape[1])]
    flat.sort(reverse=True)
    res["top_coef"] = [{"feature": f, "class": str(c), "coef": float(v)} for _, v, f, c in flat[:5]]
    return res


SANITY_NOTES = {
    "play_call": "passing should rise with ydstogo, trailing_late and down_3; fall with leading_late",
    "fourth_down": "go rate should fall with ydstogo; punt should rise with yardline (own territory); FG with short fg_distance",
    "dropback_outcome": "sacks and scrambles should rise with ydstogo / down_3 (long-developing plays)",
    "throw_outcome": "completions fall and interceptions rise with air_yards_c",
    "air_yards": "deep bins rise with ydstogo and trailing_late; fall near the goal line (yardline small)",
    "yac": "long YAC bins fall with air_yards_c (screens/short passes get YAC) and near the goal line",
    "rush_yards": "long gains fall with red_zone/inside_5 and with short-yardage situations",
    "fumble_lost": "sacks should carry the highest fumble-lost rate; completions the lowest",
    "fg_make": "make probability falls with kick_distance and wind, rises with roof_dome",
    "pat_choice": "two-point tries peak at specific deficits after the TD (-2, -5, -10, +1, +4, +5) and late in the game",
    "clock_runoff": "long runoffs with leading_late; short with clock_stopper, under_2min_half and trailing_late",
    "presnap_penalty": "penalties rise with ydstogo (obvious passing downs) and are rarer on 4th down",
}


def write_report(results: dict[str, dict], fits: dict[str, dict]) -> str:
    s = settings()
    lines = ["# Phase 2 model report", "",
             f"Trained {dt.date.today().isoformat()} on seasons {s['training_seasons'][0]}-{s['training_seasons'][-1]} "
             "(regular season + playoffs), validated on the last season with models fit on the earlier seasons. "
             "Lookup tables for the comparison were rebuilt from 2021 to the season before validation, so no validation "
             "play touches either competitor. Win-probability filter (2%-98%) applied except to the late-game models.", "",
             "## Summary", "", "| Model | Val rows | Log loss: model | situation-only | table | constant | Beats table | Beats constant | Calibration (deciles ≥500 within 3 pts) | C | Team features |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        sit = f"{r['ll_situation']:.4f}" if "ll_situation" in r else ""
        lines.append(f"| {name} | {r['n_val']} | {r['ll_model']:.4f} | {sit} | {r['ll_table']:.4f} | {r['ll_const']:.4f} | "
                     f"{'yes' if r['ll_model'] < r['ll_table'] else 'NO'} | {'yes' if r['ll_model'] < r['ll_const'] else 'NO'} | "
                     f"{'yes' if r['calibration_ok'] else 'NO'} | {fits[name]['spec']['C']} | {'yes' if fits[name]['spec'].get('team_features') else 'no'} |")
    lines.append("")
    lines.append("Spec note: the spec's single `pass_outcome` model (5 classes) is implemented as `dropback_outcome` "
                 "(throw / sack / scramble) followed by `throw_outcome` (complete / incomplete / interception given a throw), "
                 "because sacks and scrambles have no air yards to condition on. The two multiply to the five-class distribution.")
    for name, r in results.items():
        lines += ["", f"## {name}", "", f"Rows: {fits[name]['spec']['n_rows']} total, {r['n_val']} validation. "
                  f"Features: {len(fits[name]['spec']['feature_names'])}. Best C = {fits[name]['spec']['C']}.", "",
                  f"Log loss on validation: model {r['ll_model']:.4f}, Phase 1 table {r['ll_table']:.4f}, constant {r['ll_const']:.4f}.", "",
                  "Calibration (" + ("P(class 1)" if len(CLASSES[name]) == 2 else "probability of the modal class") + "):", "",
                  "| Decile | n | Mean predicted | Observed | Gap |", "|---|---|---|---|---|"]
        for i, d in enumerate(r["calibration"], 1):
            lines.append(f"| {i} | {d['n']} | {d['pred']:.3f} | {d['obs']:.3f} | {d['pred'] - d['obs']:+.3f} |")
        lines += ["", "Largest standardized coefficients:", "", "| Feature | Class | Coef |", "|---|---|---|"]
        for c in r["top_coef"]:
            lines.append(f"| {c['feature']} | {c['class']} | {c['coef']:+.3f} |")
        lines += ["", f"Sanity read: {SANITY_NOTES.get(name, '')}"]
    text = "\n".join(lines) + "\n"
    (MODELS_DIR / "report.md").write_text(text)
    return text


def train_all(names: list[str] | None = None, evaluate_only: bool = False, team: bool = True, out_dir=None,
              fixed_C: dict | None = None, report: bool = True) -> dict:
    """Train `names` (default all), then evaluate and report every model that has saved specs.
    team=True (Phase 3 onward) joins the as-of team ratings and adds the team features to the models in TEAM_FEATURES.
    Backtests pass out_dir (artifacts/models/backtest_<S>) and fixed_C (the C chosen in the main run)."""
    from nflsim.models.features import PLAYER_FEATURES, TEAM_FEATURES, WEATHER_MODELS

    to_train = [] if evaluate_only else (names or TRAINABLE)
    plays = training_plays(team=team)
    fits = {name: train_model(name, plays, team=team and (name in TEAM_FEATURES or name in WEATHER_MODELS or name in PLAYER_FEATURES),
                              out_dir=out_dir, fixed_C=(fixed_C or {}).get(name)) for name in to_train}
    if not report:
        return {name: fit["spec"] for name, fit in fits.items()}
    for name in TRAINABLE:
        if name not in fits and model_path(name).exists() and model_path(name, MODELS_DIR / "val").exists():
            fits[name] = reload_fit(name, plays)
    names = [n for n in TRAINABLE if n in fits]
    val_season = max(fits[n]["spec"]["validation_season"] for n in names)
    table_seasons = [y for y in settings()["table_seasons"] if y < val_season]
    typer.echo(f"rebuilding comparison tables from {table_seasons}")
    tables = build_all_tables(write=False, seasons=table_seasons)
    results = {name: evaluate(name, fits[name], tables) for name in names}
    for name, r in results.items():
        typer.echo(f"{name:18s} model {r['ll_model']:.4f}  table {r['ll_table']:.4f}  const {r['ll_const']:.4f}  calib {'ok' if r['calibration_ok'] else 'MISS'}")
    write_report(results, fits)
    with open(MODELS_DIR / "report.json", "w") as f:
        json.dump(results, f, indent=1)
    typer.echo(f"report written to {MODELS_DIR / 'report.md'}")
    return results
