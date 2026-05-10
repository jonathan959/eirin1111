"""Leaderboard service (Round 3)."""
from unittest.mock import patch

from services.leaderboard import build_leaderboard, _composite_score, CANONICAL_STRATEGIES


def test_composite_score_deterministic():
    a = _composite_score(0.55, 1.4, 0.8, 0.7, 0.10)
    b = _composite_score(0.55, 1.4, 0.8, 0.7, 0.10)
    assert a == b


def test_build_leaderboard_returns_nine_rows_inline_fallback():
    with (
        patch("services.leaderboard._fetch_journal_rows", return_value=[]),
        patch("services.leaderboard._latest_backtest_row", return_value=None),
        patch(
            "services.leaderboard._run_inline_backtest",
            return_value={
                "trades": 12,
                "win_rate": 0.5,
                "profit_factor": 1.2,
                "expectancy": 1.0,
                "sharpe": 0.4,
                "sortino": 0.3,
                "max_dd": 0.08,
                "avg_hold_hours": 18.0,
                "regime_fit": {k: 0.5 for k in ("TREND_UP", "TREND_DOWN", "RANGE", "RISK_OFF")},
            },
        ),
    ):
        rows = build_leaderboard(window_days=90, min_live_trades=10)
    assert len(rows) == len(CANONICAL_STRATEGIES)
    assert {r["strategy"] for r in rows} == set(CANONICAL_STRATEGIES)
    for r in rows:
        assert r["source"] == "backtest"


def test_fifteen_live_rows_force_live_source():
    import sqlite3
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    rows = []
    t0 = 1_700_000_000
    for i in range(15):
        rows.append(
            (
                "smart_dca",
                1.0,
                None,
                t0 + i * 1000,
                t0 + i * 1000 + 500,
                "TREND_UP",
            )
        )
    cur = con.cursor()
    cur.execute(
        "CREATE TABLE j (strategy text, pnl_quote real, deal_id int, entry_ts int, exit_ts int, entry_regime text)"
    )
    cur.executemany("INSERT INTO j VALUES (?,?,?,?,?,?)", rows)
    fake_sql_rows = list(cur.execute("SELECT * FROM j"))

    def _fake_fetch(_since):
        return fake_sql_rows

    bt_metrics = {
        "trades": 80,
        "win_rate": 0.5,
        "profit_factor": 1.1,
        "expectancy": 0.0,
        "sharpe": 0.2,
        "sortino": 0.2,
        "max_dd": 0.1,
        "avg_hold_hours": 12.0,
        "regime_fit": {k: 0.5 for k in ("TREND_UP", "TREND_DOWN", "RANGE", "RISK_OFF")},
    }

    with (
        patch("services.leaderboard._fetch_journal_rows", side_effect=_fake_fetch),
        patch("services.leaderboard._latest_backtest_row", return_value={"metrics": "{}"}),
        patch("services.leaderboard._metrics_from_backtest_row", return_value=bt_metrics),
        patch(
            "services.leaderboard._run_inline_backtest",
            return_value=bt_metrics,
        ),
    ):
        out = build_leaderboard(window_days=365, min_live_trades=10)
    smart = next(r for r in out if r["strategy"] == "smart_dca")
    assert smart["source"] == "live"


def test_blended_smart_dca_seven_trades():
    import sqlite3

    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute(
        "CREATE TABLE j (strategy text, pnl_quote real, deal_id int, entry_ts int, exit_ts int, entry_regime text)"
    )
    t0 = 1_700_000_000
    for i in range(7):
        cur.execute(
            "INSERT INTO j VALUES (?,?,?,?,?,?)",
            ("smart_dca", 0.5, None, t0 + i * 1000, t0 + i * 1000 + 400, "RANGE"),
        )
    fake_sql_rows = list(cur.execute("SELECT * FROM j"))

    bt = {
        "trades": 40,
        "win_rate": 0.6,
        "profit_factor": 1.3,
        "expectancy": 0.5,
        "sharpe": 0.9,
        "sortino": 0.8,
        "max_dd": 0.09,
        "avg_hold_hours": 20.0,
        "regime_fit": {k: 0.55 for k in ("TREND_UP", "TREND_DOWN", "RANGE", "RISK_OFF")},
    }

    with (
        patch("services.leaderboard._fetch_journal_rows", return_value=fake_sql_rows),
        patch("services.leaderboard._latest_backtest_row", return_value={"metrics": "{}"}),
        patch("services.leaderboard._metrics_from_backtest_row", return_value=bt),
        patch("services.leaderboard._run_inline_backtest", return_value=bt),
    ):
        out = build_leaderboard(window_days=400, min_live_trades=10)
    smart = next(r for r in out if r["strategy"] == "smart_dca")
    assert smart["source"] == "blended"
