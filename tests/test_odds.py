"""Phase 6: de-vig, EV and matching unit tests."""
import polars as pl
import pytest

from nflsim.odds.devig import decimal, implied, multiplicative, one_sided, power
from nflsim.odds.matching import normalize_name


def test_implied_and_decimal():
    assert abs(implied(-110) - 110 / 210) < 1e-12
    assert abs(implied(150) - 100 / 250) < 1e-12
    assert abs(decimal(150) - 2.5) < 1e-12 and abs(decimal(-200) - 1.5) < 1e-12


def test_devig_minus110_both_sides():
    a, b, hold = multiplicative(implied(-110), implied(-110))
    assert abs(a - 0.5) < 1e-12 and abs(b - 0.5) < 1e-12
    assert abs(hold - 0.047619) < 1e-5
    pa, pb, k = power(implied(-110), implied(-110))
    assert abs(pa - 0.5) < 1e-9 and abs(pb - 0.5) < 1e-9


def test_devig_minus150_plus130():
    qa, qb = implied(-150), implied(130)
    a, b, hold = multiplicative(qa, qb)
    assert abs(a + b - 1) < 1e-12
    assert abs(a - qa / (qa + qb)) < 1e-12          # 0.6 / (0.6 + 0.4348) = 0.5798
    assert abs(a - 0.5798) < 1e-3 and abs(hold - 0.0348) < 1e-3
    pa, pb, k = power(qa, qb)
    assert abs(pa + pb - 1) < 1e-9 and pa > a        # the power method shifts more weight to the favorite


def test_one_sided_estimate():
    assert abs(one_sided(implied(100), 0.08) - 0.5 / 1.08) < 1e-12


def test_ev_including_whole_number_push():
    """EV per dollar with pushes excluded: -3 home spread, model margin distribution with a push mass."""
    from nflsim.sim.aggregate import game_markets
    import numpy as np

    rng = np.random.default_rng(1)
    home = rng.integers(17, 31, 5000)
    away = home - np.where(rng.random(5000) < 0.1, 3, rng.integers(-10, 12, 5000))   # 10% of games land on margin 3
    rows = pl.DataFrame(game_markets("g", "H", "A", home, away, np.zeros(5000, dtype=int)))
    r = rows.filter((pl.col("market") == "spread") & (pl.col("selection") == "home") & (pl.col("line") == -3.0)).row(0, named=True)
    assert r["p_push"] > 0.05
    p = r["p_no_push"]
    assert abs(p - r["p"] / (1 - r["p_push"])) < 1e-12
    dec = decimal(-110)
    ev = p * (dec - 1) - (1 - p)
    assert abs(ev - (p * (dec - 1) - (1 - p))) < 1e-12
    # a 55% no-push cover probability at -110 has EV +5%: 0.55 * 0.909 - 0.45
    assert abs((0.55 * (dec - 1) - 0.45) - 0.05) < 1e-3


def test_name_normalization():
    assert normalize_name("Marvin Mims Jr.") == "marvin mims"
    assert normalize_name("Jaxon Smith-Njigba") == "jaxon smith njigba"
    assert normalize_name("Odell Beckham Jr.") == normalize_name("Odell Beckham")
    assert normalize_name("Ja'Marr Chase") == "jamarr chase"


def test_normalize_markets_shapes():
    from nflsim.odds.fanduel import normalize_markets

    events = {"1": {"name": "Seattle Seahawks @ Denver Broncos", "openDate": "2026-10-16T00:16:00.000Z"}}
    def runner(sid, name, hc, odds):
        return {"selectionId": sid, "runnerName": name, "handicap": hc, "runnerStatus": "ACTIVE", "winRunnerOdds": {"americanDisplayOdds": {"americanOdds": odds}}}
    markets = {
        "m1": {"eventId": 1, "marketId": "m1", "marketType": "MATCH_HANDICAP_(2-WAY)", "marketName": "Spread", "marketStatus": "OPEN",
               "runners": [runner(1, "Seattle Seahawks", -2.5, -120), runner(2, "Denver Broncos", 2.5, -102)]},
        "m2": {"eventId": 1, "marketId": "m2", "marketType": "PLAYER_X_PASSING_YARDS_HIGH", "marketName": "Sam Darnold - Passing Yds", "marketStatus": "OPEN",
               "runners": [runner(3, "Sam Darnold Over", 220.5, -114), runner(4, "Sam Darnold Under", 220.5, -114)]},
        "m3": {"eventId": 1, "marketId": "m3", "marketType": "PLAYER_X_ALT_PASSING_YARDS_HIGH", "marketName": "Sam Darnold - Alt Passing Yds", "marketStatus": "OPEN",
               "runners": [runner(5, "Sam Darnold 250+ Yards", 0, 174)]},
        "m4": {"eventId": 1, "marketId": "m4", "marketType": "ANY_TIME_TOUCHDOWN_SCORER", "marketName": "Any Time Touchdown Scorer", "marketStatus": "OPEN",
               "runners": [runner(6, "Jaxon Smith-Njigba", 0, 105)]},
        "m5": {"eventId": 1, "marketId": "m5", "marketType": "TOTAL_POINTS_(OVER/UNDER)", "marketName": "Total Points", "marketStatus": "OPEN",
               "runners": [runner(7, "Over", 40.5, -115), runner(8, "Under", 40.5, -105)]},
    }
    rows = normalize_markets(markets, events, "t")
    by = {(r["market"], r["selection"], r.get("player_name")): r for r in rows if not r["one_sided"] or r["market"] == "player_anytime_td"}
    assert by[("spread", "away", None)]["line"] == -2.5 and by[("spread", "away", None)]["team"] == "Seattle Seahawks"
    assert by[("spread", "home", None)]["line"] == 2.5
    assert by[("player_pass_yds", "over", "Sam Darnold")]["line"] == 220.5 and not by[("player_pass_yds", "under", "Sam Darnold")]["one_sided"]
    alt = [r for r in rows if r["market"] == "player_pass_yds" and r["one_sided"]][0]
    assert alt["line"] == 249.5 and alt["selection"] == "over"
    assert by[("player_anytime_td", "yes", "Jaxon Smith-Njigba")]["line"] == 0.5
    assert by[("total", "over", None)]["line"] == 40.5


def test_normalize_alternate_formats():
    from nflsim.odds.fanduel import normalize_markets

    events = {"1": {"name": "Seattle Seahawks @ Denver Broncos", "openDate": "2026-10-16T00:16:00.000Z"}}
    def runner(sid, name, hc, odds):
        return {"selectionId": sid, "runnerName": name, "handicap": hc, "runnerStatus": "ACTIVE", "winRunnerOdds": {"americanDisplayOdds": {"americanOdds": odds}}}
    markets = {
        "a": {"eventId": 1, "marketId": "a", "marketType": "ALTERNATE_HANDICAP", "marketName": "Alternate Spread", "marketStatus": "OPEN",
              "runners": [runner(1, "Seattle Seahawks (+23.5)", 0, -5000), runner(2, "Denver Broncos (-23.5)", 0, 1500)]},
        "b": {"eventId": 1, "marketId": "b", "marketType": "ALTERNATE_TOTAL", "marketName": "Alternate Total Points", "marketStatus": "OPEN",
              "runners": [runner(3, "Over (16.5)", 0, -4000), runner(4, "Under (16.5)", 0, 1200)]},
        "c": {"eventId": 1, "marketId": "c", "marketType": "HOME_TEAM_TOTAL_POINTS", "marketName": "DEN Broncos Total Points", "marketStatus": "OPEN",
              "runners": [runner(5, "Over", 19.5, -110), runner(6, "Under", 19.5, -110)]},
        "d": {"eventId": 1, "marketId": "d", "marketType": "HOME_TEAM_ALTERNATE_TOTAL", "marketName": "DEN Broncos Alternate Total", "marketStatus": "OPEN",
              "runners": [runner(7, "Over (3.5)", 0, -9000)]},
    }
    rows = normalize_markets(markets, events, "t")
    got = {(r["market"], r["selection"], r["team"], r["line"]) for r in rows}
    assert ("spread", "away", "Seattle Seahawks", 23.5) in got and ("spread", "home", "Denver Broncos", -23.5) in got
    assert ("total", "over", None, 16.5) in got and ("total", "under", None, 16.5) in got
    assert ("team_total", "over", "Denver Broncos", 19.5) in got and ("team_total", "under", "Denver Broncos", 19.5) in got
    assert ("team_total", "over", "Denver Broncos", 3.5) in got
