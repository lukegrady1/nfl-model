"""Step 1.3: league-average lookup tables built from 2021-2025 regular-season plays.

A `Table` is a hierarchy of key levels. Level n holds cells keyed by all n keys; sparse cells (fewer than
`min_cell_plays`) are merged with the neighboring bucket along the last key, and prefix groups that are sparse
even after pooling are dropped so a lookup backs off to level n-1, ..., level 0 (the marginal). Some tables
also carry a within-class empirical distribution (e.g. exact yards inside a yard bin).

Storage: artifacts/tables/<name>.parquet (long format) + artifacts/tables/<name>_within.parquet + tables.json.
"""
from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import polars as pl
import typer

from nflsim.config import TABLES_DIR, settings
from nflsim.models import buckets as B

META_PATH = TABLES_DIR / "tables.json"


@dataclass
class Table:
    name: str
    key_names: list[str]
    classes: list
    levels: list[dict[tuple, tuple[np.ndarray, int]]]   # levels[L][key[:L]] = (probs, count)
    within: dict[object, tuple[list[int], list[float]]] = field(default_factory=dict)  # class -> (values, cum probs)
    _cum_cache: dict = field(default_factory=dict, repr=False)

    # ---- lookup ---------------------------------------------------------------------------------------------
    def lookup(self, key: tuple) -> tuple[np.ndarray, int]:
        key = tuple(int(k) for k in key)
        for L in range(len(self.key_names), -1, -1):
            cell = self.levels[L].get(key[:L])
            if cell is not None:
                return cell
        raise KeyError(f"{self.name}: no cell for {key}")

    def proba(self, key: tuple) -> np.ndarray:
        return self.lookup(key)[0]

    def _cum(self, key: tuple) -> list[float]:
        key = tuple(int(k) for k in key)
        c = self._cum_cache.get(key)
        if c is None:
            c = np.cumsum(self.proba(key)).tolist()
            c[-1] = 1.0 + 1e-12
            self._cum_cache[key] = c
        return c

    def sample(self, rng: np.random.Generator, key: tuple):
        i = bisect.bisect_right(self._cum(key), rng.random())
        return self.classes[min(i, len(self.classes) - 1)]

    def sample_within(self, rng: np.random.Generator, cls) -> int:
        vals, cum = self.within[cls]
        i = bisect.bisect_right(cum, rng.random())
        return vals[min(i, len(vals) - 1)]

    def cell_count(self, key: tuple) -> int:
        return self.lookup(key)[1]

    # ---- serialization --------------------------------------------------------------------------------------
    def to_frame(self) -> pl.DataFrame:
        rows = []
        for L, cells in enumerate(self.levels):
            for key, (probs, count) in cells.items():
                for cls, p in zip(self.classes, probs):
                    row = {"level": L, "class": str(cls), "prob": float(p), "count": int(count)}
                    for i, kn in enumerate(self.key_names):
                        row[kn] = int(key[i]) if i < L else None
                    rows.append(row)
        schema = {"level": pl.Int32, "class": pl.String, "prob": pl.Float64, "count": pl.Int64}
        schema.update({kn: pl.Int32 for kn in self.key_names})
        return pl.DataFrame(rows, schema=schema)

    def within_frame(self) -> pl.DataFrame | None:
        if not self.within:
            return None
        rows = []
        for cls, (vals, cum) in self.within.items():
            prev = 0.0
            for v, c in zip(vals, cum):
                rows.append({"class": str(cls), "value": int(v), "prob": float(min(c, 1.0) - prev)})
                prev = min(c, 1.0)
        return pl.DataFrame(rows, schema={"class": pl.String, "value": pl.Int32, "prob": pl.Float64})

    @classmethod
    def from_frames(cls, name: str, key_names: list[str], classes: list, class_type: str,
                    frame: pl.DataFrame, within_frame: pl.DataFrame | None) -> "Table":
        caster = int if class_type == "int" else str
        classes = [caster(c) for c in classes]
        idx = {str(c): i for i, c in enumerate(classes)}
        levels: list[dict] = [dict() for _ in range(len(key_names) + 1)]
        for (L, *keyvals), grp in frame.group_by(["level"] + key_names, maintain_order=True):
            key = tuple(int(k) for k in keyvals[:L])
            probs = np.zeros(len(classes))
            for c, p in zip(grp["class"], grp["prob"]):
                probs[idx[c]] = p
            levels[L][key] = (probs, int(grp["count"][0]))
        within = {}
        if within_frame is not None:
            for (c,), grp in within_frame.group_by(["class"], maintain_order=True):
                g = grp.sort("value")
                cum = np.cumsum(g["prob"].to_numpy()).tolist()
                cum[-1] = 1.0 + 1e-12
                within[caster(c)] = (g["value"].to_list(), cum)
        return cls(name, key_names, classes, levels, within)


# ---- building -----------------------------------------------------------------------------------------------
def _merge_level(counts: dict[tuple, np.ndarray], n_keys: int, L: int, min_n: int) -> dict[tuple, tuple[np.ndarray, int]]:
    """Cells at level L (first L keys). Sparse cells merge with their ordinal neighbor along key L-1."""
    agg: dict[tuple, np.ndarray] = {}
    for key, cnt in counts.items():
        k = key[:L]
        agg[k] = agg.get(k, 0) + cnt
    if L == 0:
        total = agg[()]
        return {(): (total / total.sum(), int(round(total.sum())))} if total.sum() > 0 else {}
    out: dict[tuple, tuple[np.ndarray, int]] = {}
    prefixes = sorted({k[:-1] for k in agg})
    for pre in prefixes:
        last_vals = sorted(k[-1] for k in agg if k[:-1] == pre)
        total = sum(float(agg[pre + (v,)].sum()) for v in last_vals)
        if round(total) < min_n:
            continue  # whole prefix group sparse: lookups back off to level L-1
        groups: list[list[int]] = []
        cur: list[int] = []
        cur_n = 0
        for v in last_vals:
            cur.append(v)
            cur_n += float(agg[pre + (v,)].sum())
            if round(cur_n) >= min_n:
                groups.append(cur)
                cur, cur_n = [], 0
        if cur:
            if groups:
                groups[-1].extend(cur)
            else:
                groups.append(cur)
        for grp in groups:
            cnt = sum(agg[pre + (v,)] for v in grp)
            n = float(cnt.sum())
            probs = cnt / n
            for v in grp:
                out[pre + (v,)] = (probs, int(round(n)))
    return out


def km_weights(df: pl.DataFrame, keys: list[str], value_col: str, censored_col: str, tail_value: int) -> pl.DataFrame:
    """Kaplan-Meier estimate, per key group, of the yardage distribution when plays that reached the end zone are
    right-censored (the play could have gone further on a longer field). Returns keys + value + weight, where the
    weights of a group sum to its number of plays, so the table merging rules still see real play counts."""
    rows = []
    groups = df.group_by(keys, maintain_order=True) if keys else [((), df)]
    for keyvals, grp in groups:
        y = grp[value_col].to_numpy().astype(int)
        c = grp[censored_col].to_numpy().astype(bool)
        n = len(y)
        S = 1.0
        probs: dict[int, float] = {}
        for v in np.unique(y):
            at_risk = int((y >= v).sum())
            d = int(((y == v) & ~c).sum())
            if at_risk == 0:
                break
            h = d / at_risk
            probs[int(v)] = S * h
            S *= 1 - h
        if S > 1e-12:
            probs[tail_value] = probs.get(tail_value, 0.0) + S
        for v, pr in probs.items():
            if pr > 0:
                rows.append((*[int(k) for k in keyvals], v, pr * n))
    return pl.DataFrame(rows, schema=[*keys, value_col, "weight"], orient="row")


def build_table(name: str, df: pl.DataFrame, key_names: list[str], class_col: str, classes: list,
                min_n: int | None = None, within_col: str | None = None, weight_col: str | None = None) -> Table:
    """df must contain the key columns (ints) and class_col. Rows with classes not in `classes` are dropped."""
    min_n = settings()["min_cell_plays"] if min_n is None else min_n
    cls_index = {c: i for i, c in enumerate(classes)}
    df = df.filter(pl.col(class_col).is_in(classes))
    counts: dict[tuple, np.ndarray] = {}
    keys = df.select(key_names).to_numpy().astype(int) if key_names else np.zeros((df.height, 0), dtype=int)
    cls = df[class_col].to_list()
    w = df[weight_col].to_numpy().astype(float) if weight_col else np.ones(df.height)
    for i in range(df.height):
        k = tuple(keys[i].tolist())
        arr = counts.get(k)
        if arr is None:
            arr = np.zeros(len(classes))
            counts[k] = arr
        arr[cls_index[cls[i]]] += w[i]
    levels = [_merge_level(counts, len(key_names), L, min_n) for L in range(len(key_names) + 1)]
    within: dict = {}
    if within_col is not None:
        for (c,), grp in df.group_by([class_col], maintain_order=True):
            vc = grp[within_col].drop_nulls().cast(pl.Int64).value_counts().sort(within_col)
            vals = vc[within_col].to_list()
            cnt = vc["count"].to_numpy().astype(float)
            cum = np.cumsum(cnt / cnt.sum()).tolist()
            cum[-1] = 1.0 + 1e-12
            within[c] = (vals, cum)
    t = Table(name, key_names, list(classes), levels, within)
    typer.echo(f"  {name:18s} keys={key_names} classes={len(classes)} plays={int(round(w.sum()))} "
               f"cells@top={len(levels[-1])}")
    return t


def _f(col: str) -> pl.Expr:
    return pl.col(col).fill_null(0)


def table_plays() -> pl.DataFrame:
    """Regular-season plays from table_seasons with the bucket columns attached."""
    from nflsim.data.clean import load_plays

    s = settings()
    p = load_plays().filter(pl.col("season").is_in(s["table_seasons"]) & (pl.col("season_type") == "REG"))
    diff = pl.col("score_differential").fill_null(0)
    p = p.with_columns(
        dist=pl.col("ydstogo").fill_null(10).map_batches(lambda x: pl.Series(B.vectorize(B.dist_bucket)(x.to_numpy()))),
        zone=pl.col("yardline_100").fill_null(50).map_batches(lambda x: pl.Series(B.vectorize(B.zone)(x.to_numpy()))),
        yl5=pl.col("yardline_100").fill_null(50).map_batches(lambda x: pl.Series(B.vectorize(B.yardline5)(x.to_numpy()))),
        sb=diff.map_batches(lambda x: pl.Series(B.vectorize(B.score_bucket)(x.to_numpy()))),
        lead=diff.map_batches(lambda x: pl.Series(B.vectorize(B.lead_bucket)(x.to_numpy()))),
    )
    u2h = B.vectorize(B.under_two_half)
    p = p.with_columns(u2=pl.Series(u2h(p["half_seconds_remaining"].fill_null(900).to_numpy(), p["qtr"].fill_null(1).to_numpy())))
    tb = B.vectorize(B.time_bucket)
    st = B.vectorize(B.score_time_bucket)
    q = p["qtr"].fill_null(1).to_numpy()
    hs = p["half_seconds_remaining"].fill_null(900).to_numpy()
    gs = p["game_seconds_remaining"].fill_null(1800).to_numpy()
    d = p["score_differential"].fill_null(0).to_numpy()
    p = p.with_columns(tb=pl.Series(tb(q, hs, gs)), st=pl.Series(st(q, hs, d)))
    p = p.sort("game_id", "play_id").with_columns(next_elapsed=pl.col("seconds_elapsed").shift(-1).over("game_id"))
    return p


def _fg_smoothed(fg: pl.DataFrame) -> Table:
    """D11: make rate per yard of kick distance, triangular-kernel smoothed (bandwidth 3 yards)."""
    d = fg["kick_distance"].to_numpy().astype(int)
    made = (fg["field_goal_result"] == "made").to_numpy().astype(float)
    levels: list[dict] = [dict(), dict()]
    levels[0][()] = (np.array([made.mean(), 1 - made.mean()]), len(d))
    for dist in range(18, 71):
        w = np.clip(4 - np.abs(d - dist), 0, None).astype(float)
        if w.sum() < 5:
            continue
        rate = float((w * made).sum() / w.sum())
        levels[1][(dist,)] = (np.array([rate, 1 - rate]), int((w > 0).sum()))
    # extrapolate beyond the data with a decaying tail so very long attempts are not undefined
    last = 70
    while (last,) not in levels[1]:
        last -= 1
    r = levels[1][(last,)][0][0]
    for dist in range(last + 1, 80):
        r *= 0.8
        levels[1][(dist,)] = (np.array([r, 1 - r]), 0)
    return Table("fg_make", ["distance"], ["made", "missed"], levels)


def build_all_tables(write: bool = True) -> dict[str, Table]:
    s = settings()
    min_n = s["min_cell_plays"]
    p = table_plays()
    typer.echo(f"building tables from {p.height} regular-season plays, seasons {s['table_seasons']}")
    T: dict[str, Table] = {}

    calls = p.filter(pl.col("play_class").is_in(["pass", "run"]) & pl.col("down").is_between(1, 3))
    T["play_call"] = build_table("play_call", calls, ["down", "dist", "sb", "tb"], "play_class", ["pass", "run"], min_n)

    fourth = p.filter((pl.col("down") == 4) & pl.col("play_class").is_in(["pass", "run", "punt", "field_goal"]))
    fourth = fourth.with_columns(choice=pl.when(pl.col("play_class").is_in(["pass", "run"])).then(pl.lit("go"))
                                 .otherwise(pl.col("play_class")))
    T["fourth_down"] = build_table("fourth_down", fourth, ["dist", "yl5", "st"], "choice", ["go", "punt", "field_goal"], min_n)

    att = p.filter(pl.col("pass_outcome").is_in(["complete", "incomplete", "interception"]) & pl.col("air_yards").is_not_null())
    att = att.with_columns(air_bin=pl.col("air_yards").map_batches(lambda x: pl.Series(B.vectorize(B.air_bin)(x.to_numpy()))),
                           air_i=pl.col("air_yards").cast(pl.Int64).clip(-15, 99),
                           air_cens=(pl.col("air_yards") >= pl.col("yardline_100")))
    air_km = km_weights(att, ["down", "dist", "zone"], "air_i", "air_cens", 99)
    T["air_yards"] = build_table("air_yards", air_km, ["down", "dist", "zone"], "air_i", list(range(-15, 100)), min_n, weight_col="weight")

    # D4 split into (a) dropback pre-throw: sack / scramble / throw by zone, (b) throw outcome by air bin x zone
    db = p.filter(pl.col("play_class") == "pass").with_columns(
        pre=pl.when(pl.col("pass_outcome").is_in(["sack", "scramble"])).then(pl.col("pass_outcome")).otherwise(pl.lit("throw")))
    T["dropback_outcome"] = build_table("dropback_outcome", db, ["zone", "down", "dist"], "pre", ["throw", "sack", "scramble"], min_n)
    T["throw_outcome"] = build_table("throw_outcome", att, ["air_bin", "zone"], "pass_outcome",
                                     ["complete", "incomplete", "interception"], min_n)

    comp = att.filter((pl.col("pass_outcome") == "complete") & pl.col("yards_after_catch").is_not_null())
    comp = comp.with_columns(yac_i=pl.col("yards_after_catch").cast(pl.Int64).clip(-20, 99),
                             yac_cens=(pl.col("yards_gained") >= pl.col("yardline_100")))
    yac_km = km_weights(comp, ["air_bin", "zone"], "yac_i", "yac_cens", 99)
    T["yac"] = build_table("yac", yac_km, ["air_bin", "zone"], "yac_i", list(range(-20, 100)), min_n, weight_col="weight")

    sacks = p.filter(pl.col("pass_outcome") == "sack").with_columns(yds=pl.col("yards_gained").cast(pl.Int64).clip(-30, 0))
    T["sack_yards"] = build_table("sack_yards", sacks, [], "yds", sorted(sacks["yds"].unique().to_list()), min_n)

    scr = p.filter(pl.col("pass_outcome") == "scramble").with_columns(
        yds=pl.col("yards_gained").cast(pl.Int64).clip(-15, 99), cens=(pl.col("yards_gained") >= pl.col("yardline_100")))
    T["scramble_yards"] = build_table("scramble_yards", km_weights(scr, ["zone"], "yds", "cens", 99), ["zone"], "yds",
                                      list(range(-15, 100)), min_n, weight_col="weight")

    runs = p.filter(pl.col("play_class") == "run").with_columns(
        yds=pl.col("yards_gained").cast(pl.Int64).clip(-25, 99), cens=(pl.col("yards_gained") >= pl.col("yardline_100")))
    T["rush_yards"] = build_table("rush_yards", km_weights(runs, ["dist", "zone"], "yds", "cens", 99), ["dist", "zone"], "yds",
                                  list(range(-25, 100)), min_n, weight_col="weight")

    fum = p.filter(pl.col("play_class").is_in(["pass", "run"])).with_columns(
        ptype=pl.when(pl.col("play_class") == "run").then(pl.lit("run"))
        .when(pl.col("pass_outcome") == "complete").then(pl.lit("completion"))
        .when(pl.col("pass_outcome") == "sack").then(pl.lit("sack"))
        .when(pl.col("pass_outcome") == "scramble").then(pl.lit("scramble")).otherwise(pl.lit("none")),
        lost=pl.when((_f("fumble_lost") == 1) & (_f("return_touchdown") == 1)).then(pl.lit("yes_td"))
        .when(_f("fumble_lost") == 1).then(pl.lit("yes")).otherwise(pl.lit("no")))
    fum = fum.filter(pl.col("ptype") != "none").with_columns(
        pt=pl.col("ptype").replace_strict({"completion": 0, "sack": 1, "scramble": 2, "run": 3}, return_dtype=pl.Int64))
    T["fumble_lost"] = build_table("fumble_lost", fum, ["pt"], "lost", ["no", "yes", "yes_td"], min_n)
    oob = fum.with_columns(ob=pl.when(_f("out_of_bounds") == 1).then(pl.lit("yes")).otherwise(pl.lit("no")))
    T["out_of_bounds"] = build_table("out_of_bounds", oob, ["pt", "lead", "u2"], "ob", ["yes", "no"], min_n)

    ints = p.filter(pl.col("pass_outcome") == "interception").with_columns(
        six=pl.when(_f("return_touchdown") == 1).then(pl.lit("yes")).otherwise(pl.lit("no")),
        ry=pl.col("return_yards").fill_null(0).cast(pl.Int64).clip(-10, 99))
    T["pick_six"] = build_table("pick_six", ints, [], "six", ["yes", "no"], min_n)
    T["int_return_yards"] = build_table("int_return_yards", ints.filter(pl.col("six") == "no"), [], "ry",
                                        sorted(ints.filter(pl.col("six") == "no")["ry"].unique().to_list()), min_n)

    fg = p.filter((pl.col("play_class") == "field_goal") & pl.col("kick_distance").is_not_null()
                  & pl.col("field_goal_result").is_in(["made", "missed", "blocked"]))
    T["fg_make"] = _fg_smoothed(fg)
    typer.echo(f"  {'fg_make':18s} keys=['distance'] smoothed from {fg.height} attempts")

    # D12 punt: result class by yardline bucket, then net yards (receiver spot) for normal punts
    punts = p.filter((pl.col("play_class") == "punt") & pl.col("seconds_elapsed").is_not_null()
                     & pl.col("next_scrim_posteam").is_not_null())
    punts = punts.with_columns(
        res=pl.when(_f("return_touchdown") == 1).then(pl.lit("return_td"))
        .when(pl.col("next_scrim_posteam") == pl.col("posteam")).then(pl.lit("retain"))
        .when(_f("touchback") == 1).then(pl.lit("touchback")).otherwise(pl.lit("net")),
        net=(pl.col("yardline_100") - (100 - pl.col("next_scrim_yardline"))).cast(pl.Int64).clip(-30, 90),
        rspot=pl.col("next_scrim_yardline").cast(pl.Int64))
    T["punt_result"] = build_table("punt_result", punts, ["yl5"], "res", ["net", "touchback", "retain", "return_td"], min_n)
    pn = punts.filter(pl.col("res") == "net")
    T["punt_net_yards"] = build_table("punt_net_yards", pn, ["yl5"], "net", sorted(pn["net"].unique().to_list()), min_n)
    pr = punts.filter(pl.col("res") == "retain")
    T["punt_retain_spot"] = build_table("punt_retain_spot", pr, [], "rspot", sorted(pr["rspot"].unique().to_list()), 1)

    # D13 kickoff (kickoff_seasons only): receiving team's starting yardline_100; nflfastR posteam = receiver
    ko_all = p.filter((pl.col("play_class") == "kickoff") & pl.col("seconds_elapsed").is_not_null()
                      & pl.col("next_scrim_posteam").is_not_null())
    prev_safety = pl.col("safety").shift(1).over("game_id").fill_null(0)
    ko_all = ko_all.with_columns(after_safety=prev_safety)  # NB: shift within the filtered frame; refined below
    ko = p.with_columns(prev_safety=pl.col("safety").fill_null(0).shift(1).over("game_id")).filter(
        (pl.col("play_class") == "kickoff") & pl.col("seconds_elapsed").is_not_null() & pl.col("next_scrim_posteam").is_not_null())
    ko = ko.with_columns(
        res=pl.when(_f("return_touchdown") == 1).then(pl.lit("return_td"))
        .when(pl.col("next_scrim_posteam") != pl.col("posteam")).then(pl.lit("kicking_team_ball"))
        .otherwise(pl.lit("normal")),
        rspot=pl.col("next_scrim_yardline").cast(pl.Int64))
    ko_cur = ko.filter(pl.col("season").is_in(s["kickoff_seasons"]) & (pl.col("prev_safety") == 0))
    T["kickoff_result"] = build_table("kickoff_result", ko_cur, [], "res", ["normal", "return_td", "kicking_team_ball"], min_n)
    kn = ko_cur.filter(pl.col("res") == "normal")
    T["kickoff_spot"] = build_table("kickoff_spot", kn, [], "rspot", sorted(kn["rspot"].unique().to_list()), min_n)
    fk = ko.filter((pl.col("prev_safety") == 1) & (pl.col("res") == "normal"))
    if fk.height >= min_n:
        T["free_kick_spot"] = build_table("free_kick_spot", fk, [], "rspot", sorted(fk["rspot"].unique().to_list()), min_n)
    else:
        typer.echo(f"  free_kick_spot: only {fk.height} free kicks; engine uses the default yardline from settings")

    # D14 PAT choice keyed by score diff after the TD x time bucket; nflfastR PAT rows already include the TD points
    pats = p.filter(pl.col("play_class").is_in(["extra_point", "two_point"])).with_columns(
        sd=pl.col("score_differential").fill_null(0).map_batches(lambda x: pl.Series(B.vectorize(B.pat_score_key)(x.to_numpy()))),
        choice=pl.when(pl.col("play_class") == "two_point").then(pl.lit("two")).otherwise(pl.lit("kick")))
    T["pat_choice"] = build_table("pat_choice", pats, ["sd", "tb"], "choice", ["kick", "two"], min_n)
    xp = pats.filter(pl.col("play_class") == "extra_point").with_columns(
        r=pl.when(pl.col("extra_point_result") == "good").then(pl.lit("good")).otherwise(pl.lit("no_good")))
    T["xp_result"] = build_table("xp_result", xp, [], "r", ["good", "no_good"], min_n)
    two = pats.filter(pl.col("play_class") == "two_point").with_columns(
        r=pl.when(pl.col("two_point_conv_result") == "success").then(pl.lit("good")).otherwise(pl.lit("no_good")))
    T["two_point_result"] = build_table("two_point_result", two, [], "r", ["good", "no_good"], min_n)

    # D16 clock runoff: seconds until the next snap. Exclude rows followed by a timeout or a period marker.
    clk = p.filter(pl.col("play_class").is_in(["pass", "run", "punt", "field_goal", "kickoff", "penalty_no_play", "spike"])
                   & pl.col("seconds_elapsed").is_not_null() & ~pl.col("next_play_is_timeout")
                   & ~pl.col("next_is_marker") & ~pl.col("next_is_two_minute_warning")
                   & (pl.col("next_qtr") == pl.col("qtr")))
    clk = clk.with_columns(cs=pl.col("clock_stopper").cast(pl.Int64),
                           sbin=pl.col("seconds_elapsed").cast(pl.Int64).map_batches(lambda x: pl.Series(B.vectorize(B.seconds_bin)(x.to_numpy()))),
                           secs=pl.col("seconds_elapsed").cast(pl.Int64))
    T["clock_runoff"] = build_table("clock_runoff", clk, ["cs", "lead", "u2"], "sbin", list(range(10)), min_n, within_col="secs")

    # D16b timeouts: whether a timeout follows the play, and the snap-to-snap seconds when one does
    # (nflfastR puts the play's duration on the timeout row, so the two rows' elapsed are summed).
    tob = p.filter(pl.col("play_class").is_in(["pass", "run", "kneel", "punt", "field_goal", "kickoff", "penalty_no_play", "spike"])
                   & pl.col("seconds_elapsed").is_not_null() & ~pl.col("next_is_marker")
                   & ~pl.col("next_is_two_minute_warning") & (pl.col("next_qtr") == pl.col("qtr")))
    tob = tob.with_columns(cs=pl.col("clock_stopper").cast(pl.Int64),
                           to=pl.when(pl.col("next_play_is_timeout")).then(pl.lit("yes")).otherwise(pl.lit("no")))
    T["timeout_after"] = build_table("timeout_after", tob, ["cs", "lead", "u2"], "to", ["yes", "no"], min_n)
    tor = tob.filter(pl.col("next_play_is_timeout") & pl.col("next_elapsed").is_not_null()).with_columns(
        secs=(pl.col("seconds_elapsed") + pl.col("next_elapsed")).clip(0, 60).cast(pl.Int64))
    tor = tor.with_columns(sbin=pl.col("secs").map_batches(lambda x: pl.Series(B.vectorize(B.seconds_bin)(x.to_numpy()))))
    T["timeout_runoff"] = build_table("timeout_runoff", tor, ["cs", "u2"], "sbin", list(range(10)), min_n, within_col="secs")

    # D17 pre-snap / nullified-play penalty: none / offense / defense per snap, with the assessed yards
    snaps = p.filter(pl.col("play_class").is_in(["pass", "run", "punt", "field_goal", "penalty_no_play", "kneel", "spike"])
                     & pl.col("down").is_between(1, 4))
    snaps = snaps.with_columns(
        side=pl.when(pl.col("play_class") != "penalty_no_play").then(pl.lit("none"))
        .when(pl.col("penalty_team") == pl.col("posteam")).then(pl.lit("offense")).otherwise(pl.lit("defense")),
        pyds=pl.col("penalty_yards").fill_null(5).cast(pl.Int64).clip(1, 50))
    T["presnap_penalty"] = build_table("presnap_penalty", snaps, ["down", "dist"], "side", ["none", "offense", "defense"], min_n, within_col="pyds")
    dpen = snaps.filter(pl.col("side") == "defense").with_columns(
        afd=pl.when(_f("first_down") == 1).then(pl.lit("yes")).otherwise(pl.lit("no")))
    T["def_penalty_first_down"] = build_table("def_penalty_first_down", dpen, [], "afd", ["yes", "no"], min_n)

    if write:
        save_tables(T)
    return T


def save_tables(T: dict[str, Table]) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    meta = {}
    for name, t in T.items():
        t.to_frame().write_parquet(TABLES_DIR / f"{name}.parquet")
        wf = t.within_frame()
        if wf is not None:
            wf.write_parquet(TABLES_DIR / f"{name}_within.parquet")
        meta[name] = {"key_names": t.key_names, "classes": [str(c) for c in t.classes],
                      "class_type": "int" if all(isinstance(c, (int, np.integer)) for c in t.classes) else "str",
                      "has_within": wf is not None}
    with open(META_PATH, "w") as f:
        json.dump({"table_seasons": settings()["table_seasons"], "tables": meta}, f, indent=1)
    typer.echo(f"saved {len(T)} tables to {TABLES_DIR}")


_TABLE_CACHE: dict[str, Table] | None = None


def load_tables() -> dict[str, Table]:
    global _TABLE_CACHE
    if _TABLE_CACHE is not None:
        return _TABLE_CACHE
    with open(META_PATH) as f:
        meta = json.load(f)["tables"]
    T = {}
    for name, m in meta.items():
        frame = pl.read_parquet(TABLES_DIR / f"{name}.parquet")
        wf = pl.read_parquet(TABLES_DIR / f"{name}_within.parquet") if m["has_within"] else None
        T[name] = Table.from_frames(name, m["key_names"], m["classes"], m["class_type"], frame, wf)
    _TABLE_CACHE = T
    return T


class TableModel:
    """DecisionModel backed by a lookup Table. `key_fn(state, extra)` builds the key tuple."""

    def __init__(self, table: Table, key_fn: Callable):
        self.table = table
        self.name = table.name
        self.classes = table.classes
        self.key_fn = key_fn

    def proba(self, state, extra: dict) -> np.ndarray:
        return self.table.proba(self.key_fn(state, extra))

    def sample(self, rng, state, extra: dict):
        return self.table.sample(rng, self.key_fn(state, extra))

    def sample_within(self, rng, cls) -> int:
        return self.table.sample_within(rng, cls)
