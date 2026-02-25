# db.py  (REPLACE ENTIRE FILE)
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

# Allow override via env; default keeps compatibility with your existing file
DB_NAME = os.getenv("BOT_DB_PATH", "botdb.sqlite3")

# Whitelist for dynamic SQL (prevents SQL injection)
_ALLOWED_TABLES = frozenset({
    "bots", "bot_logs", "deals", "settings", "strategy_decisions", "regime_snapshots",
    "autopilot_config",
    "order_events", "perf_metrics", "backtest_runs", "recommendations_snapshots",
    "recommendations_latest", "strategy_perf_trades", "intelligence_decisions",
    "recommendation_performance", "scoring_calibration_log", "intraday_patterns",
    "sector_performance_history", "dividend_events", "tax_harvest_suggestions", "market_events",
    "insider_transactions", "ml_predictions", "ml_model_versions", "execution_quality",
    "data_quality_log", "error_log", "trade_journal", "portfolio_snapshots",
    "autopilot_audit_log",  # LIVE-HARDENED: autopilot decision traceability
})
_ALLOWED_COLUMNS = frozenset({"bot_id", "id"})


def _conn() -> sqlite3.Connection:
    """
    Stability-focused SQLite connection:
    - WAL mode reduces 'database is locked' under concurrent reads/writes
    - busy_timeout makes SQLite wait briefly instead of failing immediately
    - foreign_keys ON for future-proofing (even if you don't use FKs yet)
    """
    con = sqlite3.connect(DB_NAME, check_same_thread=False, timeout=30.0)
    con.row_factory = sqlite3.Row

    try:
        con.execute("PRAGMA journal_mode=WAL;")
    except Exception:
        pass
    try:
        con.execute("PRAGMA synchronous=NORMAL;")
    except Exception:
        pass
    try:
        con.execute("PRAGMA foreign_keys=ON;")
    except Exception:
        pass
    try:
        con.execute("PRAGMA busy_timeout=5000;")  # ms
    except Exception:
        pass

    return con


def now_ts() -> int:
    return int(time.time())


def _table_columns(con: sqlite3.Connection, table: str) -> List[str]:
    if table not in _ALLOWED_TABLES:
        raise ValueError(f"Invalid table name: {table}")
    rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    return [r["name"] for r in rows]


def _ensure_column(con: sqlite3.Connection, table: str, col: str, col_def_sql: str) -> None:
    if table not in _ALLOWED_TABLES:
        raise ValueError(f"Invalid table name: {table}")
    cols = _table_columns(con, table)
    if col not in cols:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_def_sql}")


def init_db() -> None:
    """
    Creates tables if missing + performs lightweight migrations to keep older DBs compatible.
    Safe to call multiple times.
    """
    con = _conn()
    cur = con.cursor()

    # --- bots
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS bots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            symbol TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 0,
            dry_run INTEGER NOT NULL DEFAULT 1,

            base_quote REAL NOT NULL DEFAULT 0,
            safety_quote REAL NOT NULL DEFAULT 0,
            max_safety INTEGER NOT NULL DEFAULT 0,
            first_dev REAL NOT NULL DEFAULT 0.01,
            step_mult REAL NOT NULL DEFAULT 1.2,
            tp REAL NOT NULL DEFAULT 0.01,

            trend_filter INTEGER NOT NULL DEFAULT 0,
            trend_sma INTEGER NOT NULL DEFAULT 200,

            max_spend_quote REAL NOT NULL DEFAULT 0,
            poll_seconds INTEGER NOT NULL DEFAULT 10,
            strategy_mode TEXT NOT NULL DEFAULT 'classic',
            forced_strategy TEXT NOT NULL DEFAULT '',
            max_open_orders INTEGER NOT NULL DEFAULT 6,
            vol_gap_mult REAL NOT NULL DEFAULT 1.0,
            tp_vol_mult REAL NOT NULL DEFAULT 1.0,
            min_gap_pct REAL NOT NULL DEFAULT 0.003,
            max_gap_pct REAL NOT NULL DEFAULT 0.06,
            regime_hold_candles INTEGER NOT NULL DEFAULT 2,
            regime_switch_ticks INTEGER NOT NULL DEFAULT 2,
            regime_switch_threshold REAL NOT NULL DEFAULT 0.6,
            max_total_exposure_pct REAL NOT NULL DEFAULT 0.50,
            per_symbol_exposure_pct REAL NOT NULL DEFAULT 0.15,
            min_free_cash_pct REAL NOT NULL DEFAULT 0.1,
            max_concurrent_deals INTEGER NOT NULL DEFAULT 6,
            spread_guard_pct REAL NOT NULL DEFAULT 0.003,
            limit_timeout_sec INTEGER NOT NULL DEFAULT 8,
            daily_loss_limit_pct REAL NOT NULL DEFAULT 0.06,
            pause_hours INTEGER NOT NULL DEFAULT 6,
            auto_restart INTEGER NOT NULL DEFAULT 0,
            last_running INTEGER NOT NULL DEFAULT 0,
            
            market_type TEXT NOT NULL DEFAULT 'crypto',
            alpaca_mode TEXT NOT NULL DEFAULT 'paper',

            created_at INTEGER NOT NULL
        );
        """
    )

    # --- logs
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS bot_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id INTEGER NOT NULL,
            ts INTEGER NOT NULL,
            level TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'SYSTEM',
            message TEXT NOT NULL,
            count INTEGER NOT NULL DEFAULT 1
        );
        """
    )

    # --- deals
    # --- strategy decisions
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS strategy_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id INTEGER NOT NULL,
            ts INTEGER NOT NULL,
            strategy TEXT NOT NULL,
            action TEXT NOT NULL,
            reason TEXT NOT NULL,
            regime TEXT,
            confidence REAL,
            payload TEXT
        );
        """
    )

    # --- regime snapshots
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS regime_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id INTEGER NOT NULL,
            ts INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            regime TEXT NOT NULL,
            confidence REAL,
            why TEXT,
            snapshot TEXT
        );
        """
    )

    # --- order events
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS order_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id INTEGER NOT NULL,
            ts INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            ord_type TEXT NOT NULL,
            price REAL,
            amount REAL,
            order_id TEXT,
            tag TEXT,
            status TEXT,
            reason TEXT
        );
        """
    )

    # --- performance metrics (strategy-level)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS perf_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id INTEGER NOT NULL,
            ts INTEGER NOT NULL,
            strategy TEXT NOT NULL,
            payload TEXT
        );
        """
    )

    # --- backtest runs
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            strategy TEXT NOT NULL,
            params TEXT,
            metrics TEXT,
            equity TEXT
        );
        """
    )
    # --- recommendations
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS recommendations_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            horizon TEXT NOT NULL,
            score REAL NOT NULL,
            regime_json TEXT,
            metrics_json TEXT,
            reasons_json TEXT,
            risk_flags_json TEXT,
            created_ts INTEGER NOT NULL
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS recommendations_latest (
            symbol TEXT NOT NULL,
            horizon TEXT NOT NULL,
            snapshot_id INTEGER NOT NULL,
            created_ts INTEGER NOT NULL,
            PRIMARY KEY(symbol, horizon)
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS deals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id INTEGER NOT NULL,
            state TEXT NOT NULL,
            opened_at INTEGER NOT NULL,
            closed_at INTEGER,
            symbol TEXT NOT NULL,

            entry_avg REAL,
            exit_avg REAL,
            base_amount REAL,
            realized_pnl_quote REAL,
            entry_regime TEXT,
            exit_regime TEXT,
            entry_strategy TEXT,
            exit_strategy TEXT,
            mae REAL,
            mfe REAL,
            hold_sec INTEGER,
            safety_count INTEGER
        );
        """
    )

    # --- strategy performance trades (rolling stats)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS strategy_perf_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id INTEGER NOT NULL,
            symbol TEXT,
            regime TEXT,
            strategy TEXT NOT NULL,
            pnl REAL NOT NULL,
            pnl_pct REAL,
            ts INTEGER NOT NULL
        );
        """
    )

    # --- settings (global app flags)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    
    # --- intelligence_decisions (Intelligence Layer decision log)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS intelligence_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id INTEGER NOT NULL,
            ts INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            allowed_actions TEXT NOT NULL,
            final_action TEXT NOT NULL,
            final_reason TEXT,
            data_ok INTEGER NOT NULL,
            data_reasons TEXT,
            safety_allowed TEXT,
            safety_reasons TEXT,
            regime TEXT,
            regime_confidence REAL,
            strategy_mode TEXT,
            entry_style TEXT,
            exit_style TEXT,
            base_size REAL,
            order_type TEXT,
            manage_actions TEXT,
            proposed_orders TEXT,
            debug_json TEXT,
            execution_result TEXT,
            realized_slippage REAL,
            fill_quality TEXT
        );
        """
    )

    # --- recommendation_performance (track recommendation accuracy vs actual trades)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS recommendation_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            recommendation_date INTEGER NOT NULL,
            score_at_recommendation REAL NOT NULL,
            regime_at_recommendation TEXT,
            bot_id INTEGER,
            deal_id INTEGER,
            entry_price REAL,
            exit_price REAL,
            pnl_realized REAL,
            days_held REAL,
            outcome TEXT NOT NULL DEFAULT 'active',
            notes TEXT,
            technical_patterns_json TEXT,
            snapshot_id INTEGER,
            created_at INTEGER NOT NULL
        );
        """
    )

    # --- scoring_calibration_log (audit trail for adaptive scoring)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS scoring_calibration_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            scoring_version TEXT NOT NULL,
            changes_json TEXT,
            analysis_window_days INTEGER,
            notes TEXT
        );
        """
    )

    # --- intraday_patterns (day-trading: opening range breaks, VWAP crosses, volume spikes)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS intraday_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id INTEGER,
            symbol TEXT NOT NULL,
            pattern_type TEXT NOT NULL,
            ts INTEGER NOT NULL,
            price REAL,
            vwap REAL,
            or_high REAL,
            or_low REAL,
            volume_spike_ratio REAL,
            payload_json TEXT
        );
        """
    )

    # --- sector_performance_history (for sector rotation strategy)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sector_performance_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sector TEXT NOT NULL,
            quarter_ts INTEGER NOT NULL,
            return_pct REAL,
            momentum_score REAL,
            rank INTEGER,
            payload_json TEXT
        );
        """
    )

    # --- dividend_events (dividend tracking)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS dividend_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            ex_date INTEGER NOT NULL,
            payment_date INTEGER,
            amount REAL NOT NULL,
            dividend_yield_pct REAL,
            recorded_at INTEGER NOT NULL
        );
        """
    )

    # --- tax_harvest_suggestions (tax-loss harvesting suggestions)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS tax_harvest_suggestions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            unrealized_loss_pct REAL NOT NULL,
            suggest_sell_ts INTEGER NOT NULL,
            wash_sale_until_ts INTEGER,
            alternate_symbol TEXT,
            recorded_at INTEGER NOT NULL
        );
        """
    )

    # --- autopilot_config (Master Upgrade Part 4 - full autopilot)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS autopilot_config (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            enabled INTEGER DEFAULT 0,
            total_capital_allocated REAL DEFAULT 10000.0,
            max_positions INTEGER DEFAULT 10,
            position_size_mode TEXT DEFAULT 'conviction_based',
            asset_types TEXT DEFAULT 'both',
            min_score_threshold INTEGER DEFAULT 75,
            min_conviction_threshold INTEGER DEFAULT 5,
            excluded_sectors TEXT,
            max_loss_per_day_pct REAL DEFAULT 5.0,
            max_loss_per_week_pct REAL DEFAULT 10.0,
            max_correlated_exposure_pct REAL DEFAULT 50.0,
            max_sector_exposure_pct REAL DEFAULT 50.0,
            scan_frequency_hours INTEGER DEFAULT 4,
            auto_create_bots INTEGER DEFAULT 1,
            auto_start_bots INTEGER DEFAULT 1,
            auto_close_underperformers INTEGER DEFAULT 1,
            underperformer_threshold_score INTEGER DEFAULT 60,
            rebalance_enabled INTEGER DEFAULT 1,
            rebalance_frequency_days INTEGER DEFAULT 7,
            notify_on_new_bot INTEGER DEFAULT 1,
            notify_on_close INTEGER DEFAULT 1,
            notify_daily_summary INTEGER DEFAULT 1,
            last_scan INTEGER,
            last_rebalance INTEGER
        );
        """
    )
    cur.execute("INSERT OR IGNORE INTO autopilot_config (id, enabled) VALUES (1, 0)")
    # Migration: bump exposure limits and reduce defaults (11.md critical fixes)
    _ensure_column(con, "autopilot_config", "capital_per_bot", "REAL DEFAULT 500.0")
    cur.execute("""
        UPDATE autopilot_config SET
            max_correlated_exposure_pct = 50.0,
            max_sector_exposure_pct = 50.0
        WHERE id = 1 AND (max_correlated_exposure_pct < 50 OR max_sector_exposure_pct < 50)
    """)
    con.commit()

    # --- portfolio_snapshots (for charts - 11.md)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_value REAL,
            total_pnl REAL,
            total_pnl_pct REAL,
            active_positions INTEGER,
            realized_pnl REAL,
            unrealized_pnl REAL
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_ts ON portfolio_snapshots(timestamp)"
    )

    # --- market_events (earnings, Fed, etc. - avoid entries day before)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS market_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_date INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            symbol TEXT,
            impact_level INTEGER NOT NULL DEFAULT 2,
            description TEXT,
            recorded_at INTEGER NOT NULL
        );
        """
    )

    # --- insider_transactions (SEC Form 4 - CEO/CFO buys/sells)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS insider_transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            transaction_date INTEGER NOT NULL,
            transaction_type TEXT NOT NULL,
            shares REAL NOT NULL DEFAULT 0,
            value_usd REAL,
            insider_title TEXT,
            filing_url TEXT,
            recorded_at INTEGER NOT NULL
        );
        """
    )

    # --- ml_predictions (log every ML prediction, track outcomes 7d/30d)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ml_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            prediction_date INTEGER NOT NULL,
            predicted_direction TEXT NOT NULL,
            predicted_price REAL,
            confidence REAL NOT NULL DEFAULT 0,
            price_at_prediction REAL,
            actual_outcome_7d REAL,
            actual_outcome_30d REAL,
            model_version TEXT,
            model_used TEXT,
            regime_at_prediction TEXT,
            recorded_at INTEGER NOT NULL
        );
        """
    )

    # --- ml_model_versions (track deployed model versions, validation accuracy)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ml_model_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_type TEXT NOT NULL,
            version TEXT NOT NULL,
            validation_accuracy REAL,
            trained_at INTEGER NOT NULL,
            deployed INTEGER NOT NULL DEFAULT 0
        );
        """
    )

    # --- execution_quality (slippage tracking, post-trade analysis)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS execution_quality (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id TEXT,
            bot_id INTEGER,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            strategy TEXT,
            intended_price REAL NOT NULL,
            executed_price REAL,
            slippage_pct REAL,
            slippage_dollars REAL,
            vwap_at_execution REAL,
            twap_at_execution REAL,
            execution_quality_score INTEGER,
            created_at INTEGER NOT NULL
        );
        """
    )

    # --- data_quality_log (missing candles, stale prices, spreads, volume anomalies)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS data_quality_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            source TEXT NOT NULL,
            issue_type TEXT NOT NULL,
            severity TEXT NOT NULL DEFAULT 'info',
            details_json TEXT
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_data_quality_ts ON data_quality_log(ts);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_data_quality_severity ON data_quality_log(severity, ts);")

    # --- error_log (API errors, circuit breaker events, recovery logs)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS error_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            source TEXT NOT NULL,
            error_type TEXT NOT NULL,
            message TEXT,
            bot_id INTEGER,
            details_json TEXT
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_error_log_ts ON error_log(ts);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_error_log_source ON error_log(source, ts);")

    # --- autopilot_audit_log (LIVE-HARDENED: every autopilot decision with reason)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS autopilot_audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            action TEXT NOT NULL,
            symbol TEXT,
            reason TEXT,
            details_json TEXT
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_autopilot_audit_ts ON autopilot_audit_log(ts);")

    # --- trade_journal (entry/exit reasons, lessons, screenshots for closed deals)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trade_journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            deal_id INTEGER NOT NULL UNIQUE,
            entry_reason TEXT,
            exit_reason TEXT,
            lessons_learned TEXT,
            screenshot_data TEXT,
            updated_at INTEGER NOT NULL
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_trade_journal_deal ON trade_journal(deal_id);")

    # Helpful indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_logs_bot_ts ON bot_logs(bot_id, ts);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_bots_symbol ON bots(symbol);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_bots_enabled ON bots(enabled);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_deals_bot_state ON deals(bot_id, state);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_deals_closed_at ON deals(closed_at);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_intel_bot_ts ON intelligence_decisions(bot_id, ts);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_reco_perf_bot_outcome ON recommendation_performance(bot_id, outcome);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_reco_perf_symbol ON recommendation_performance(symbol, outcome);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_intraday_symbol_ts ON intraday_patterns(symbol, ts);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sector_perf_sector_ts ON sector_performance_history(sector, quarter_ts);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_dividend_symbol_ex ON dividend_events(symbol, ex_date);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_market_events_date ON market_events(event_date, event_type);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_insider_symbol_date ON insider_transactions(symbol, transaction_date);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ml_predictions_symbol_date ON ml_predictions(symbol, prediction_date);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ml_predictions_recorded ON ml_predictions(recorded_at);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_execution_quality_symbol ON execution_quality(symbol, created_at);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_execution_quality_strategy ON execution_quality(strategy, created_at);")

    # Lightweight migrations (in case older DB exists with missing columns)
    try:
        _ensure_column(con, "bots", "enabled", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(con, "bots", "dry_run", "INTEGER NOT NULL DEFAULT 1")

        _ensure_column(con, "bots", "trend_filter", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(con, "bots", "trend_sma", "INTEGER NOT NULL DEFAULT 200")
        _ensure_column(con, "bots", "max_spend_quote", "REAL NOT NULL DEFAULT 0")
        _ensure_column(con, "bots", "poll_seconds", "INTEGER NOT NULL DEFAULT 10")
        _ensure_column(con, "bots", "strategy_mode", "TEXT NOT NULL DEFAULT 'classic'")
        _ensure_column(con, "bots", "forced_strategy", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(con, "bots", "max_open_orders", "INTEGER NOT NULL DEFAULT 6")
        _ensure_column(con, "bots", "vol_gap_mult", "REAL NOT NULL DEFAULT 1.0")
        _ensure_column(con, "bots", "tp_vol_mult", "REAL NOT NULL DEFAULT 1.0")
        _ensure_column(con, "bots", "min_gap_pct", "REAL NOT NULL DEFAULT 0.003")
        _ensure_column(con, "bots", "max_gap_pct", "REAL NOT NULL DEFAULT 0.06")
        _ensure_column(con, "bots", "regime_hold_candles", "INTEGER NOT NULL DEFAULT 2")
        _ensure_column(con, "bots", "regime_switch_ticks", "INTEGER NOT NULL DEFAULT 2")
        _ensure_column(con, "bots", "regime_switch_threshold", "REAL NOT NULL DEFAULT 0.6")
        _ensure_column(con, "bots", "max_total_exposure_pct", "REAL NOT NULL DEFAULT 0.50")
        _ensure_column(con, "bots", "per_symbol_exposure_pct", "REAL NOT NULL DEFAULT 0.15")
        _ensure_column(con, "bots", "min_free_cash_pct", "REAL NOT NULL DEFAULT 0.1")
        _ensure_column(con, "bots", "max_concurrent_deals", "INTEGER NOT NULL DEFAULT 6")
        _ensure_column(con, "bots", "spread_guard_pct", "REAL NOT NULL DEFAULT 0.003")
        _ensure_column(con, "bots", "limit_timeout_sec", "INTEGER NOT NULL DEFAULT 8")
        _ensure_column(con, "bots", "daily_loss_limit_pct", "REAL NOT NULL DEFAULT 0.06")
        _ensure_column(con, "bots", "pause_hours", "INTEGER NOT NULL DEFAULT 6")
        _ensure_column(con, "bots", "auto_restart", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(con, "bots", "last_running", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(con, "bots", "market_type", "TEXT NOT NULL DEFAULT 'crypto'")
        _ensure_column(con, "bots", "alpaca_mode", "TEXT NOT NULL DEFAULT 'paper'")

        _ensure_column(con, "order_events", "is_live", "INTEGER DEFAULT 0")

        # Phase 1: Quick Wins - Trailing Stop Loss
        _ensure_column(con, "bots", "trailing_stop_enabled", "INTEGER NOT NULL DEFAULT 1")
        _ensure_column(con, "bots", "trailing_activation_pct", "REAL NOT NULL DEFAULT 0.02")
        _ensure_column(con, "bots", "trailing_distance_pct", "REAL NOT NULL DEFAULT 0.01")
        
        # Phase 1: Cooldown After Stop Loss
        _ensure_column(con, "bots", "stop_loss_cooldown_sec", "INTEGER NOT NULL DEFAULT 3600")
        _ensure_column(con, "bots", "last_stop_loss_at", "INTEGER")
        
        # Phase 1: Volatility-Based TP Scaling
        _ensure_column(con, "bots", "adaptive_tp_enabled", "INTEGER NOT NULL DEFAULT 1")
        _ensure_column(con, "bots", "tp_volatility_mult", "REAL NOT NULL DEFAULT 1.5")
        
        # Phase 1: BTC Correlation Guard
        _ensure_column(con, "bots", "btc_correlation_guard", "INTEGER NOT NULL DEFAULT 1")
        _ensure_column(con, "bots", "btc_dump_threshold_pct", "REAL NOT NULL DEFAULT 0.05")
        
        # Phase 1: Time-Based Filters
        _ensure_column(con, "bots", "time_filter_enabled", "INTEGER NOT NULL DEFAULT 1")
        _ensure_column(con, "bots", "skip_first_30min", "INTEGER NOT NULL DEFAULT 1")
        _ensure_column(con, "bots", "skip_last_30min", "INTEGER NOT NULL DEFAULT 1")
        
        # Phase 1: Adaptive Volume & Spread
        _ensure_column(con, "bots", "min_volume_ratio", "REAL NOT NULL DEFAULT 1.5")
        _ensure_column(con, "bots", "adaptive_spread_enabled", "INTEGER NOT NULL DEFAULT 1")
        _ensure_column(con, "bots", "max_drawdown_pct", "REAL NOT NULL DEFAULT 0")
        
        # Phase A: Kelly Criterion position sizing
        _ensure_column(con, "bots", "use_kelly_sizing", "INTEGER NOT NULL DEFAULT 1")
        _ensure_column(con, "bots", "kelly_fraction", "REAL NOT NULL DEFAULT 0.25")
        _ensure_column(con, "bots", "max_position_pct", "REAL NOT NULL DEFAULT 0.10")
        # Live trading gate: require explicit confirmation for real orders
        _ensure_column(con, "bots", "live_confirmed", "INTEGER NOT NULL DEFAULT 1")
        # Day-trading / scalping
        _ensure_column(con, "bots", "day_trading_mode", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(con, "bots", "scalp_trailing_activation_pct", "REAL NOT NULL DEFAULT 0.05")
        _ensure_column(con, "bots", "scalp_trailing_distance_pct", "REAL NOT NULL DEFAULT 0.002")
        _ensure_column(con, "bots", "auto_close_eod", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(con, "bots", "long_term_mode", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(con, "bots", "long_term_trailing_pct", "REAL NOT NULL DEFAULT 0.15")

        # Master Upgrade: Trading mode system (Part 1)
        _ensure_column(con, "bots", "trading_mode", "TEXT DEFAULT 'swing_trade'")
        _ensure_column(con, "bots", "intended_hold_days", "INTEGER DEFAULT 14")
        _ensure_column(con, "bots", "conviction_level", "INTEGER DEFAULT 5")
        _ensure_column(con, "bots", "auto_dip_buy", "INTEGER DEFAULT 0")
        _ensure_column(con, "bots", "fundamental_exit_only", "INTEGER DEFAULT 0")
        _ensure_column(con, "bots", "rebalance_enabled", "INTEGER DEFAULT 0")
        _ensure_column(con, "bots", "bot_type", "TEXT DEFAULT ''")

        # Deals table - tracking for trailing stops
        _ensure_column(con, "deals", "highest_price", "REAL")
        _ensure_column(con, "deals", "trailing_stop_price", "REAL")
        _ensure_column(con, "deals", "trailing_stop_active", "INTEGER NOT NULL DEFAULT 0")
        
        _ensure_column(con, "bot_logs", "category", "TEXT NOT NULL DEFAULT 'SYSTEM'")
        _ensure_column(con, "bot_logs", "count", "INTEGER NOT NULL DEFAULT 1")
        _ensure_column(con, "deals", "entry_regime", "TEXT")
        _ensure_column(con, "deals", "exit_regime", "TEXT")
        _ensure_column(con, "deals", "entry_strategy", "TEXT")
        _ensure_column(con, "deals", "exit_strategy", "TEXT")
        _ensure_column(con, "deals", "mae", "REAL")
        _ensure_column(con, "deals", "mfe", "REAL")
        _ensure_column(con, "deals", "hold_sec", "INTEGER")
        _ensure_column(con, "deals", "safety_count", "INTEGER")
        _ensure_column(con, "strategy_perf_trades", "symbol", "TEXT")
        _ensure_column(con, "strategy_perf_trades", "regime", "TEXT")
        _ensure_column(con, "strategy_perf_trades", "pnl_pct", "REAL")
        _ensure_column(con, "recommendations_snapshots", "scoring_version", "TEXT NOT NULL DEFAULT 'v1'")
        # Indexes that depend on migrated columns
        cur.execute("CREATE INDEX IF NOT EXISTS idx_strategy_perf_bot_ts ON strategy_perf_trades(bot_id, ts);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_strategy_perf_sym ON strategy_perf_trades(symbol, ts);")
    except Exception:
        # If migration fails, do not crash app; tables still usable.
        pass

    con.commit()
    con.close()


# =========================================================
# Logs
# =========================================================
def _normalize_message(msg: str) -> str:
    s = str(msg or "").strip()
    if s.startswith("[") and "]" in s[:32]:
        s = s.split("]", 1)[-1].strip()
    for sep in ("AM ", "PM "):
        if sep in s[:32]:
            s = s.split(sep, 1)[-1].strip()
    return s


def add_log(bot_id: int, level: str, message: str, category: str = "SYSTEM") -> None:
    con = _conn()
    norm = _normalize_message(message)
    row = con.execute(
        "SELECT id, level, category, message, count FROM bot_logs WHERE bot_id=? ORDER BY id DESC LIMIT 1",
        (int(bot_id),),
    ).fetchone()
    if row:
        last_norm = _normalize_message(row["message"])
        if str(row["level"]) == str(level) and str(row["category"]) == str(category) and last_norm == norm:
            con.execute(
                "UPDATE bot_logs SET ts=?, count=? WHERE id=?",
                (now_ts(), int(row["count"] or 1) + 1, int(row["id"])),
            )
            con.commit()
            con.close()
            return
    con.execute(
        "INSERT INTO bot_logs(bot_id, ts, level, category, message, count) VALUES (?,?,?,?,?,?)",
        (int(bot_id), now_ts(), str(level), str(category), str(message), 1),
    )
    con.commit()
    con.close()


def list_logs(bot_id: int, limit: int = 200) -> List[Dict[str, Any]]:
    con = _conn()
    rows = con.execute(
        "SELECT ts, level, category, message, count FROM bot_logs WHERE bot_id=? ORDER BY ts DESC LIMIT ?",
        (int(bot_id), int(limit)),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def list_logs_window(bot_id: int, start_ts: int, end_ts: int, limit: int = 200) -> List[Dict[str, Any]]:
    con = _conn()
    rows = con.execute(
        """
        SELECT ts, level, category, message, count
        FROM bot_logs
        WHERE bot_id=? AND ts>=? AND ts<=?
        ORDER BY ts ASC
        LIMIT ?
        """,
        (int(bot_id), int(start_ts), int(end_ts), int(limit)),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def log_data_quality(source: str, issue_type: str, severity: str, details: Optional[Dict] = None) -> None:
    """Log data quality issue to data_quality_log."""
    con = _conn()
    try:
        con.execute(
            "INSERT INTO data_quality_log(ts, source, issue_type, severity, details_json) VALUES (?,?,?,?,?)",
            (now_ts(), str(source)[:64], str(issue_type)[:64], str(severity)[:32],
             str(__import__("json").dumps(details)) if details else None),
        )
        con.commit()
    finally:
        con.close()


def get_recent_data_quality_count(minutes: int = 15, min_severity: str = "warning") -> int:
    """Count data quality issues in last N minutes. Severity order: critical > error > warning > info."""
    sev_order = {"critical": 0, "error": 1, "warning": 2, "info": 3}
    min_level = sev_order.get(min_severity.lower(), 2)
    con = _conn()
    try:
        since = now_ts() - minutes * 60
        rows = con.execute(
            "SELECT severity FROM data_quality_log WHERE ts >= ?",
            (since,),
        ).fetchall()
        count = 0
        for r in rows:
            if sev_order.get(str(r["severity"] or "").lower(), 99) <= min_level:
                count += 1
        return count
    finally:
        con.close()


def log_error(source: str, error_type: str, message: Optional[str] = None, bot_id: Optional[int] = None, details: Optional[Dict] = None) -> None:
    """Log error to error_log table."""
    con = _conn()
    try:
        con.execute(
            "INSERT INTO error_log(ts, source, error_type, message, bot_id, details_json) VALUES (?,?,?,?,?,?)",
            (now_ts(), str(source)[:64], str(error_type)[:64], (message or "")[:1024],
             int(bot_id) if bot_id else None,
             str(__import__("json").dumps(details)) if details else None),
        )
        con.commit()
    finally:
        con.close()


def add_autopilot_audit_log(action: str, symbol: Optional[str] = None, reason: Optional[str] = None, details: Optional[Dict] = None) -> None:
    """LIVE-HARDENED: Log every autopilot decision for traceability."""
    con = _conn()
    try:
        con.execute(
            "INSERT INTO autopilot_audit_log(ts, action, symbol, reason, details_json) VALUES (?,?,?,?,?)",
            (now_ts(), str(action)[:64], (symbol or "")[:32], (reason or "")[:512],
             str(__import__("json").dumps(details)) if details else None),
        )
        con.commit()
    finally:
        con.close()


def list_autopilot_audit_log(limit: int = 50) -> List[Dict[str, Any]]:
    """Return latest autopilot audit entries (newest first)."""
    con = _conn()
    try:
        rows = con.execute(
            "SELECT id, ts, action, symbol, reason, details_json FROM autopilot_audit_log ORDER BY ts DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


# =========================================================
# Bots CRUD
# =========================================================

def list_logs_since(bot_id: int, last_id: int, limit: int = 200) -> List[Dict[str, Any]]:
    """Return logs newer than last_id (ascending order)."""
    con = _conn()
    rows = con.execute(
        "SELECT id, ts, level, category, message, count FROM bot_logs WHERE bot_id=? AND id>? ORDER BY id ASC LIMIT ?",
        (int(bot_id), int(last_id), int(limit)),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]

def create_bot(data: Dict[str, Any]) -> int:
    con = _conn()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO bots(
            name, symbol, enabled, dry_run,
            base_quote, safety_quote, max_safety, first_dev, step_mult, tp,
            trend_filter, trend_sma,
            max_spend_quote, poll_seconds, strategy_mode, forced_strategy, max_open_orders,
            vol_gap_mult, tp_vol_mult, min_gap_pct, max_gap_pct,
            regime_hold_candles, regime_switch_ticks, regime_switch_threshold,
            max_total_exposure_pct, per_symbol_exposure_pct, min_free_cash_pct, max_concurrent_deals,
            spread_guard_pct, limit_timeout_sec, daily_loss_limit_pct, pause_hours,
            auto_restart, last_running, market_type, alpaca_mode, bot_type, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            str(data["name"]),
            str(data["symbol"]),
            int(data.get("enabled", 0)),
            int(data.get("dry_run", 1)),
            float(data["base_quote"]),
            float(data["safety_quote"]),
            int(data["max_safety"]),
            float(data["first_dev"]),
            float(data["step_mult"]),
            float(data["tp"]),
            int(data.get("trend_filter", 0)),
            int(data.get("trend_sma", 200)),
            float(data["max_spend_quote"]),
            int(data.get("poll_seconds", 10)),
            str(data.get("strategy_mode", "classic")),
            str(data.get("forced_strategy", "")),
            int(data.get("max_open_orders", 6)),
            float(data.get("vol_gap_mult", 1.0)),
            float(data.get("tp_vol_mult", 1.0)),
            float(data.get("min_gap_pct", 0.003)),
            float(data.get("max_gap_pct", 0.06)),
            int(data.get("regime_hold_candles", 2)),
            int(data.get("regime_switch_ticks", 2)),
            float(data.get("regime_switch_threshold", 0.6)),
            float(data.get("max_total_exposure_pct", 0.50)),
            float(data.get("per_symbol_exposure_pct", 0.15)),
            float(data.get("min_free_cash_pct", 0.1)),
            int(data.get("max_concurrent_deals", 6)),
            float(data.get("spread_guard_pct", 0.003)),
            int(data.get("limit_timeout_sec", 8)),
            float(data.get("daily_loss_limit_pct", 0.06)),
            int(data.get("pause_hours", 6)),
            int(data.get("auto_restart", 0)),
            int(data.get("last_running", 0)),
            str(data.get("market_type", "crypto")),
            str(data.get("alpaca_mode", "paper")),
            str(data.get("bot_type", "")),
            now_ts(),
        ),
    )
    con.commit()
    bot_id = int(cur.lastrowid)
    con.close()
    return bot_id


def update_bot(bot_id: int, data: Dict[str, Any]) -> None:
    con = _conn()
    con.execute(
        """
        UPDATE bots SET
            name=?,
            symbol=?,
            enabled=?,
            dry_run=?,
            base_quote=?,
            safety_quote=?,
            max_safety=?,
            first_dev=?,
            step_mult=?,
            tp=?,
            trend_filter=?,
            trend_sma=?,
            max_spend_quote=?,
            poll_seconds=?,
            strategy_mode=?,
            forced_strategy=?,
            max_open_orders=?,
            vol_gap_mult=?,
            tp_vol_mult=?,
            min_gap_pct=?,
            max_gap_pct=?,
            regime_hold_candles=?,
            regime_switch_ticks=?,
            regime_switch_threshold=?,
            max_total_exposure_pct=?,
            per_symbol_exposure_pct=?,
            min_free_cash_pct=?,
            max_concurrent_deals=?,
            spread_guard_pct=?,
            limit_timeout_sec=?,
            daily_loss_limit_pct=?,
            pause_hours=?,
            auto_restart=?,
            market_type=?,
            alpaca_mode=?,
            max_drawdown_pct=?,
            trading_mode=?,
            intended_hold_days=?,
            conviction_level=?,
            auto_dip_buy=?,
            fundamental_exit_only=?,
            rebalance_enabled=?
        WHERE id=?
        """,
        (
            str(data["name"]),
            str(data["symbol"]),
            int(data.get("enabled", 0)),
            int(data.get("dry_run", 1)),
            float(data["base_quote"]),
            float(data["safety_quote"]),
            int(data["max_safety"]),
            float(data["first_dev"]),
            float(data["step_mult"]),
            float(data["tp"]),
            int(data.get("trend_filter", 0)),
            int(data.get("trend_sma", 200)),
            float(data["max_spend_quote"]),
            int(data.get("poll_seconds", 10)),
            str(data.get("strategy_mode", "classic")),
            str(data.get("forced_strategy", "")),
            int(data.get("max_open_orders", 6)),
            float(data.get("vol_gap_mult", 1.0)),
            float(data.get("tp_vol_mult", 1.0)),
            float(data.get("min_gap_pct", 0.003)),
            float(data.get("max_gap_pct", 0.06)),
            int(data.get("regime_hold_candles", 2)),
            int(data.get("regime_switch_ticks", 2)),
            float(data.get("regime_switch_threshold", 0.6)),
            float(data.get("max_total_exposure_pct", 0.50)),
            float(data.get("per_symbol_exposure_pct", 0.15)),
            float(data.get("min_free_cash_pct", 0.1)),
            int(data.get("max_concurrent_deals", 6)),
            float(data.get("spread_guard_pct", 0.003)),
            int(data.get("limit_timeout_sec", 8)),
            float(data.get("daily_loss_limit_pct", 0.06)),
            int(data.get("pause_hours", 6)),
            int(data.get("auto_restart", 0)),
            str(data.get("market_type", "crypto")),
            str(data.get("alpaca_mode", "paper")),
            float(data.get("max_drawdown_pct", 0.0)),
            str(data.get("trading_mode", "swing_trade")),
            int(data.get("intended_hold_days", 14)),
            int(data.get("conviction_level", 5)),
            int(data.get("auto_dip_buy", 0)),
            int(data.get("fundamental_exit_only", 0)),
            int(data.get("rebalance_enabled", 0)),
            int(bot_id),
        ),
    )
    con.commit()
    con.close()


def update_bots_by_type(bot_type: str, enabled: int) -> int:
    """Update enabled status for all bots with given bot_type. Returns count updated."""
    con = _conn()
    try:
        cur = con.execute(
            "UPDATE bots SET enabled=? WHERE LOWER(TRIM(COALESCE(bot_type,''))) = LOWER(TRIM(?))",
            (int(enabled), str(bot_type or "").strip()),
        )
        con.commit()
        return cur.rowcount
    finally:
        con.close()


def delete_bot(bot_id: int) -> None:
    """Delete bot and all related rows. Child tables first, then bots."""
    import logging
    logger = logging.getLogger(__name__)
    bid = int(bot_id)
    con = _conn()
    try:
        allowed_pairs = [
            ("order_events", "bot_id"),
            ("strategy_decisions", "bot_id"),
            ("regime_snapshots", "bot_id"),
            ("perf_metrics", "bot_id"),
            ("intelligence_decisions", "bot_id"),
            ("bot_logs", "bot_id"),
            ("deals", "bot_id"),
        ]
        for table, col in allowed_pairs:
            if table in _ALLOWED_TABLES and col in _ALLOWED_COLUMNS:
                try:
                    con.execute(f"DELETE FROM {table} WHERE {col}=?", (bid,))
                except sqlite3.OperationalError as e:
                    if "no such table" in str(e).lower():
                        logger.debug("delete_bot: skip %s (no such table)", table)
                    else:
                        raise
        con.execute("DELETE FROM bots WHERE id=?", (bid,))
        con.commit()
    finally:
        con.close()


def set_bot_enabled(bot_id: int, enabled: bool) -> None:
    con = _conn()
    con.execute(
        "UPDATE bots SET enabled=? WHERE id=?",
        (1 if enabled else 0, int(bot_id)),
    )
    con.commit()
    con.close()


def set_bot_running(bot_id: int, running: bool) -> None:
    con = _conn()
    con.execute(
        "UPDATE bots SET last_running=? WHERE id=?",
        (1 if running else 0, int(bot_id)),
    )
    con.commit()
    con.close()


def get_bot(bot_id: int) -> Optional[Dict[str, Any]]:
    con = _conn()
    row = con.execute("SELECT * FROM bots WHERE id=?", (int(bot_id),)).fetchone()
    con.close()
    return dict(row) if row else None


def list_bots() -> List[Dict[str, Any]]:
    con = _conn()
    rows = con.execute("SELECT * FROM bots ORDER BY id DESC").fetchall()
    con.close()
    return [dict(r) for r in rows]


def set_setting(key: str, value: Any) -> None:
    con = _conn()
    con.execute(
        "INSERT INTO settings(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(key), str(value)),
    )
    con.commit()
    con.close()


def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    con = _conn()
    row = con.execute("SELECT value FROM settings WHERE key=?", (str(key),)).fetchone()
    con.close()
    if not row:
        return default
    try:
        return str(row["value"])
    except Exception:
        return default


# =========================================================
# Autopilot config (Master Upgrade Part 4)
# =========================================================
def get_autopilot_config_row() -> Optional[Dict[str, Any]]:
    """Get autopilot_config row (id=1). Returns dict or None."""
    con = _conn()
    try:
        row = con.execute("SELECT * FROM autopilot_config WHERE id = 1").fetchone()
        return dict(row) if row else None
    except Exception:
        return None
    finally:
        con.close()


def save_autopilot_config(data: Dict[str, Any]) -> None:
    """Upsert autopilot_config row (id=1)."""
    con = _conn()
    try:
        con.execute(
            """
            INSERT INTO autopilot_config (id, enabled, total_capital_allocated, max_positions,
                asset_types, min_score_threshold, min_conviction_threshold,
                max_loss_per_day_pct, max_loss_per_week_pct,
                scan_frequency_hours, auto_create_bots, auto_start_bots,
                auto_close_underperformers, underperformer_threshold_score,
                rebalance_enabled, rebalance_frequency_days,
                notify_on_new_bot, notify_on_close, notify_daily_summary, last_scan, last_rebalance)
            VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                enabled=excluded.enabled, total_capital_allocated=excluded.total_capital_allocated,
                max_positions=excluded.max_positions, asset_types=excluded.asset_types,
                min_score_threshold=excluded.min_score_threshold,
                min_conviction_threshold=excluded.min_conviction_threshold,
                max_loss_per_day_pct=excluded.max_loss_per_day_pct,
                max_loss_per_week_pct=excluded.max_loss_per_week_pct,
                scan_frequency_hours=excluded.scan_frequency_hours,
                auto_create_bots=excluded.auto_create_bots, auto_start_bots=excluded.auto_start_bots,
                auto_close_underperformers=excluded.auto_close_underperformers,
                underperformer_threshold_score=excluded.underperformer_threshold_score,
                rebalance_enabled=excluded.rebalance_enabled,
                rebalance_frequency_days=excluded.rebalance_frequency_days,
                notify_on_new_bot=excluded.notify_on_new_bot,
                notify_on_close=excluded.notify_on_close,
                notify_daily_summary=excluded.notify_daily_summary,
                last_scan=excluded.last_scan, last_rebalance=excluded.last_rebalance
            """,
            (
                int(data.get("enabled", 0)),
                float(data.get("total_capital_allocated", 10000)),
                int(data.get("max_positions", 10)),
                str(data.get("asset_types", "both")),
                int(data.get("min_score_threshold", 75)),
                int(data.get("min_conviction_threshold", 5)),
                float(data.get("max_loss_per_day_pct", 5.0)),
                float(data.get("max_loss_per_week_pct", 10.0)),
                int(data.get("scan_frequency_hours", 4)),
                int(data.get("auto_create_bots", 1)),
                int(data.get("auto_start_bots", 1)),
                int(data.get("auto_close_underperformers", 1)),
                int(data.get("underperformer_threshold_score", 60)),
                int(data.get("rebalance_enabled", 1)),
                int(data.get("rebalance_frequency_days", 7)),
                int(data.get("notify_on_new_bot", 1)),
                int(data.get("notify_on_close", 1)),
                int(data.get("notify_daily_summary", 1)),
                data.get("last_scan"),
                data.get("last_rebalance"),
            ),
        )
        con.commit()
    finally:
        con.close()


# =========================================================
# Deals
# =========================================================
def open_deal(bot_id: int, symbol: str, state: str = "OPEN") -> int:
    con = _conn()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO deals(bot_id, state, opened_at, symbol) VALUES (?,?,?,?)",
        (int(bot_id), str(state), now_ts(), str(symbol)),
    )
    con.commit()
    deal_id = int(cur.lastrowid)
    con.close()
    return deal_id


def close_deal(
    deal_id: int,
    entry_avg: float,
    exit_avg: float,
    base_amount: float,
    realized_pnl_quote: float,
    entry_regime: Optional[str] = None,
    exit_regime: Optional[str] = None,
    entry_strategy: Optional[str] = None,
    exit_strategy: Optional[str] = None,
    mae: Optional[float] = None,
    mfe: Optional[float] = None,
    hold_sec: Optional[int] = None,
    safety_count: Optional[int] = None,
) -> None:
    con = _conn()
    # Fetch bot_id and opened_at before update (for recommendation_performance)
    row = con.execute(
        "SELECT bot_id, opened_at FROM deals WHERE id=?",
        (int(deal_id),),
    ).fetchone()
    closed_ts = now_ts()
    opened_ts = int(row["opened_at"] or 0) if row else 0
    bot_id_val = int(row["bot_id"]) if row else None

    con.execute(
        """
        UPDATE deals SET
            state=?,
            closed_at=?,
            entry_avg=?,
            exit_avg=?,
            base_amount=?,
            realized_pnl_quote=?,
            entry_regime=?,
            exit_regime=?,
            entry_strategy=?,
            exit_strategy=?,
            mae=?,
            mfe=?,
            hold_sec=?,
            safety_count=?
        WHERE id=?
        """,
        (
            "CLOSED",
            closed_ts,
            float(entry_avg) if entry_avg is not None else None,
            float(exit_avg) if exit_avg is not None else None,
            float(base_amount) if base_amount is not None else None,
            float(realized_pnl_quote) if realized_pnl_quote is not None else None,
            str(entry_regime) if entry_regime is not None else None,
            str(exit_regime) if exit_regime is not None else None,
            str(entry_strategy) if entry_strategy is not None else None,
            str(exit_strategy) if exit_strategy is not None else None,
            float(mae) if mae is not None else None,
            float(mfe) if mfe is not None else None,
            int(hold_sec) if hold_sec is not None else None,
            int(safety_count) if safety_count is not None else None,
            int(deal_id),
        ),
    )
    con.commit()

    # Record outcome for recommendation performance tracking (if bot was created from recommendation)
    if bot_id_val and entry_avg is not None and exit_avg is not None and realized_pnl_quote is not None:
        try:
            _record_recommendation_outcome(
                con, bot_id_val, deal_id,
                float(entry_avg), float(exit_avg), float(realized_pnl_quote),
                closed_ts, opened_ts,
            )
            con.commit()
        except Exception:
            pass  # Do not fail deal close if performance tracking fails
    con.close()


def get_symbols_with_open_deals() -> List[str]:
    """Return distinct symbols from all open deals."""
    con = _conn()
    rows = con.execute(
        """
        SELECT DISTINCT symbol FROM deals
        WHERE state='OPEN' AND symbol IS NOT NULL AND symbol != ''
        """
    ).fetchall()
    con.close()
    return [str(r[0]) for r in rows]


def get_symbols_with_open_deals_excluding(bot_id: int) -> List[str]:
    """Return distinct symbols from open deals of other bots (exclude bot_id)."""
    con = _conn()
    rows = con.execute(
        """
        SELECT DISTINCT symbol FROM deals
        WHERE state='OPEN' AND bot_id != ? AND symbol IS NOT NULL AND symbol != ''
        """,
        (int(bot_id),),
    ).fetchall()
    con.close()
    return [str(r[0]) for r in rows]


def latest_open_deal(bot_id: int) -> Optional[Dict[str, Any]]:
    con = _conn()
    row = con.execute(
        """
        SELECT * FROM deals
        WHERE bot_id=? AND state='OPEN'
        ORDER BY opened_at DESC
        LIMIT 1
        """,
        (int(bot_id),),
    ).fetchone()
    con.close()
    return dict(row) if row else None


def list_deals(bot_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    con = _conn()
    rows = con.execute(
        """
        SELECT
            id, state, opened_at, closed_at, symbol,
            entry_avg, exit_avg, base_amount, realized_pnl_quote
        FROM deals
        WHERE bot_id=?
        ORDER BY opened_at DESC
        LIMIT ?
        """,
        (int(bot_id), int(limit)),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def list_all_deals(state: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
    con = _conn()
    if state:
        rows = con.execute(
            """
            SELECT
                id, bot_id, state, opened_at, closed_at, symbol,
                entry_avg, exit_avg, base_amount, realized_pnl_quote
            FROM deals
            WHERE state=?
            ORDER BY opened_at DESC
            LIMIT ?
            """,
            (str(state), int(limit)),
        ).fetchall()
    else:
        rows = con.execute(
            """
            SELECT
                id, bot_id, state, opened_at, closed_at, symbol,
                entry_avg, exit_avg, base_amount, realized_pnl_quote
            FROM deals
            ORDER BY opened_at DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def list_closed_deals_for_journal(since_ts: Optional[int] = None, limit: int = 200) -> List[Dict[str, Any]]:
    """List closed deals with full columns for journal. since_ts = only deals closed after."""
    con = _conn()
    if since_ts:
        rows = con.execute(
            """
            SELECT id, bot_id, state, opened_at, closed_at, symbol,
                   entry_avg, exit_avg, base_amount, realized_pnl_quote,
                   entry_regime, entry_strategy, exit_regime, exit_strategy
            FROM deals
            WHERE state='CLOSED' AND closed_at IS NOT NULL AND closed_at >= ?
            ORDER BY closed_at DESC
            LIMIT ?
            """,
            (int(since_ts), int(limit)),
        ).fetchall()
    else:
        rows = con.execute(
            """
            SELECT id, bot_id, state, opened_at, closed_at, symbol,
                   entry_avg, exit_avg, base_amount, realized_pnl_quote,
                   entry_regime, entry_strategy, exit_regime, exit_strategy
            FROM deals
            WHERE state='CLOSED' AND closed_at IS NOT NULL
            ORDER BY closed_at DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def get_deal(deal_id: int, full: bool = False) -> Optional[Dict[str, Any]]:
    """Get deal. full=True includes entry_regime, entry_strategy, exit_regime, exit_strategy."""
    con = _conn()
    cols = "id, bot_id, state, opened_at, closed_at, symbol, entry_avg, exit_avg, base_amount, realized_pnl_quote"
    if full:
        cols += ", entry_regime, entry_strategy, exit_regime, exit_strategy"
    row = con.execute(f"SELECT {cols} FROM deals WHERE id=?", (int(deal_id),)).fetchone()
    con.close()
    return dict(row) if row else None


def bot_pnl_series(bot_id: int, limit: int = 500) -> List[Dict[str, Any]]:
    con = _conn()
    rows = con.execute(
        """
        SELECT closed_at, realized_pnl_quote
        FROM deals
        WHERE bot_id=? AND state='CLOSED' AND closed_at IS NOT NULL
        ORDER BY closed_at ASC
        LIMIT ?
        """,
        (int(bot_id), int(limit)),
    ).fetchall()
    con.close()
    out: List[Dict[str, Any]] = []
    cum = 0.0
    for r in rows:
        try:
            pnl = float(r["realized_pnl_quote"] or 0.0)
            ts = int(r["closed_at"] or 0)
            cum += pnl
            out.append({"time": ts, "value": float(cum)})
        except Exception:
            continue
    return out


def bot_drawdown_series(bot_id: int, limit: int = 500) -> List[Dict[str, Any]]:
    con = _conn()
    rows = con.execute(
        """
        SELECT closed_at, realized_pnl_quote
        FROM deals
        WHERE bot_id=? AND state='CLOSED' AND closed_at IS NOT NULL
        ORDER BY closed_at ASC
        LIMIT ?
        """,
        (int(bot_id), int(limit)),
    ).fetchall()
    con.close()
    out: List[Dict[str, Any]] = []
    cum = 0.0
    peak = 0.0
    for r in rows:
        try:
            pnl = float(r["realized_pnl_quote"] or 0.0)
            ts = int(r["closed_at"] or 0)
            cum += pnl
            if cum > peak:
                peak = cum
            drawdown = max(0.0, peak - cum)
            # negative value for chart readability
            out.append({"time": ts, "value": float(-drawdown)})
        except Exception:
            continue
    return out


def bot_performance_stats(bot_id: int) -> Dict[str, Any]:
    con = _conn()
    rows = con.execute(
        """
        SELECT opened_at, closed_at, realized_pnl_quote, entry_avg, base_amount
        FROM deals
        WHERE bot_id=? AND state='CLOSED' AND closed_at IS NOT NULL
        ORDER BY closed_at ASC
        """,
        (int(bot_id),),
    ).fetchall()
    con.close()
    pnls: List[float] = []
    pnl_pcts: List[float] = []
    durations: List[int] = []
    for r in rows:
        try:
            pnl = float(r["realized_pnl_quote"] or 0.0)
            entry_avg = float(r["entry_avg"] or 0.0)
            base_amount = float(r["base_amount"] or 0.0)
            opened = int(r["opened_at"] or 0)
            closed = int(r["closed_at"] or 0)
            if closed > 0 and opened > 0:
                durations.append(max(0, closed - opened))
            pnls.append(pnl)
            cost_basis = entry_avg * base_amount if entry_avg > 0 and base_amount > 0 else 0.0
            if cost_basis > 0:
                pnl_pcts.append(pnl / cost_basis)
        except Exception:
            continue
    wins = sum(1 for x in pnls if x > 0)
    losses = sum(1 for x in pnls if x < 0)
    total = len(pnls)
    win_rate = (wins / total) if total else 0.0
    avg_duration = (sum(durations) / len(durations)) if durations else 0.0
    win_pcts = [p for p in pnl_pcts if p > 0]
    loss_pcts = [abs(p) for p in pnl_pcts if p < 0]
    avg_profit_pct = (sum(win_pcts) / len(win_pcts)) if win_pcts else 0.02
    avg_loss_pct = (sum(loss_pcts) / len(loss_pcts)) if loss_pcts else 0.01
    return {
        "wins": int(wins),
        "losses": int(losses),
        "total": int(total),
        "win_rate": float(win_rate),
        "avg_duration_sec": float(avg_duration),
        "avg_profit_pct": float(avg_profit_pct),
        "avg_loss_pct": float(avg_loss_pct),
        "total_trades": int(total),
        "winning_trades": int(wins),
    }


def get_bot_recent_streak(bot_id: int, n: int = 5) -> int:
    """
    Last N deal outcomes: positive = wins, negative = losses.
    Returns streak: 3 means 3 consecutive wins, -3 means 3 consecutive losses.
    """
    con = _conn()
    rows = con.execute(
        """
        SELECT realized_pnl_quote
        FROM deals
        WHERE bot_id=? AND state='CLOSED' AND closed_at IS NOT NULL
        ORDER BY closed_at DESC
        LIMIT ?
        """,
        (int(bot_id), int(n)),
    ).fetchall()
    con.close()
    outcomes = []
    for r in rows:
        pnl = float(r["realized_pnl_quote"] or 0)
        outcomes.append(1 if pnl > 0 else -1)
    streak = 0
    for o in outcomes:
        if streak == 0:
            streak = o
        elif (streak > 0 and o > 0) or (streak < 0 and o < 0):
            streak += o
        else:
            break
    return streak


def bot_deal_stats(bot_id: int) -> Dict[str, Any]:
    """
    Aggregate counts + realized PnL for a bot.
    """
    con = _conn()
    row = con.execute(
        """
        SELECT
            SUM(CASE WHEN state='OPEN' THEN 1 ELSE 0 END) AS open_count,
            SUM(CASE WHEN state='CLOSED' THEN 1 ELSE 0 END) AS closed_count,
            COALESCE(SUM(CASE WHEN state='CLOSED' THEN realized_pnl_quote ELSE 0 END), 0) AS realized_total
        FROM deals
        WHERE bot_id=?
        """,
        (int(bot_id),),
    ).fetchone()
    con.close()
    if not row:
        return {"open_count": 0, "closed_count": 0, "realized_total": 0.0}
    return {
        "open_count": int(row["open_count"] or 0),
        "closed_count": int(row["closed_count"] or 0),
        "realized_total": float(row["realized_total"] or 0.0),
    }


def all_deal_stats() -> Dict[str, Any]:
    """
    Aggregate counts + realized PnL across all bots.
    """
    con = _conn()
    row = con.execute(
        """
        SELECT
            SUM(CASE WHEN state='OPEN' THEN 1 ELSE 0 END) AS open_count,
            SUM(CASE WHEN state='CLOSED' THEN 1 ELSE 0 END) AS closed_count,
            COALESCE(SUM(CASE WHEN state='CLOSED' THEN realized_pnl_quote ELSE 0 END), 0) AS realized_total
        FROM deals
        """
    ).fetchone()
    con.close()
    if not row:
        return {"open_count": 0, "closed_count": 0, "realized_total": 0.0}
    return {
        "open_count": int(row["open_count"] or 0),
        "closed_count": int(row["closed_count"] or 0),
        "realized_total": float(row["realized_total"] or 0.0),
    }


# =========================================================
# PnL summary (stable schema used by app + bot logic)
# =========================================================
def pnl_summary(since_ts: int) -> Dict[str, Any]:
    """
    Summarize realized PnL for CLOSED deals closed_at >= since_ts.
    Returns:
      - realized (float)
      - deals_closed (int)
      - wins (int)
      - losses (int)
      - since_ts (int)
    """
    con = _conn()
    rows = con.execute(
        """
        SELECT realized_pnl_quote
        FROM deals
        WHERE state='CLOSED' AND closed_at IS NOT NULL AND closed_at >= ?
        """,
        (int(since_ts),),
    ).fetchall()
    con.close()

    pnls: List[float] = []
    for r in rows:
        try:
            v = r["realized_pnl_quote"]
            if v is None:
                continue
            pnls.append(float(v))
        except Exception:
            continue

    realized = float(sum(pnls)) if pnls else 0.0
    wins = sum(1 for x in pnls if x > 0)
    losses = sum(1 for x in pnls if x < 0)

    return {
        "since_ts": int(since_ts),
        "realized": float(realized),
        "deals_closed": int(len(pnls)),
        "wins": int(wins),
        "losses": int(losses),
    }


# =========================================================
# Strategy/regime persistence
# =========================================================
def add_regime_snapshot(bot_id: int, symbol: str, regime: str, confidence: float, why: str, snapshot: str) -> None:
    con = _conn()
    con.execute(
        "INSERT INTO regime_snapshots(bot_id, ts, symbol, regime, confidence, why, snapshot) VALUES (?,?,?,?,?,?,?)",
        (int(bot_id), now_ts(), str(symbol), str(regime), float(confidence), str(why), str(snapshot)),
    )
    con.commit()
    con.close()


def add_strategy_decision(bot_id: int, strategy: str, action: str, reason: str, regime: str, confidence: float, payload: str) -> None:
    con = _conn()
    con.execute(
        """
        INSERT INTO strategy_decisions(bot_id, ts, strategy, action, reason, regime, confidence, payload)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (int(bot_id), now_ts(), str(strategy), str(action), str(reason), str(regime), float(confidence), str(payload)),
    )
    con.commit()
    con.close()


def add_strategy_trade(
    bot_id: int,
    strategy: str,
    pnl: float,
    symbol: Optional[str] = None,
    regime: Optional[str] = None,
    pnl_pct: Optional[float] = None,
    ts: Optional[int] = None,
) -> None:
    con = _conn()
    con.execute(
        "INSERT INTO strategy_perf_trades(bot_id, symbol, regime, strategy, pnl, pnl_pct, ts) VALUES (?,?,?,?,?,?,?)",
        (int(bot_id), str(symbol or ""), str(regime or ""), str(strategy), float(pnl), float(pnl_pct) if pnl_pct is not None else None, int(ts or now_ts())),
    )
    con.commit()
    con.close()


def get_strategy_perf(bot_id: int, strategy: str, window: int = 30) -> Dict[str, Any]:
    con = _conn()
    rows = con.execute(
        "SELECT pnl, pnl_pct FROM strategy_perf_trades WHERE bot_id=? AND strategy=? ORDER BY ts DESC LIMIT ?",
        (int(bot_id), str(strategy), int(window)),
    ).fetchall()
    con.close()
    pnls = [float(r["pnl"]) for r in rows] if rows else []
    pnl_pcts = [float(r["pnl_pct"]) for r in rows if r["pnl_pct"] is not None]
    if not pnls:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "expectancy": 0.0, "max_drawdown": 0.0, "avg_win": 0.0, "avg_loss": 0.0}
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_rate = len(wins) / len(pnls)
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (2.0 if gross_profit > 0 else 0.0)
    expectancy = sum(pnls) / len(pnls)
    avg_win = (sum([p for p in pnl_pcts if p > 0]) / max(1, len([p for p in pnl_pcts if p > 0]))) if pnl_pcts else 0.0
    avg_loss = abs(sum([p for p in pnl_pcts if p < 0]) / max(1, len([p for p in pnl_pcts if p < 0]))) if pnl_pcts else 0.0
    # Simple max drawdown on cumulative pnl
    peak = 0.0
    dd = 0.0
    cum = 0.0
    for p in reversed(pnls):
        cum += p
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    return {
        "trades": len(pnls),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "max_drawdown": dd,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


def get_strategy_leaderboard(window_days: int = 90) -> List[Dict[str, Any]]:
    """
    Aggregate strategy performance across all bots.
    Returns list of {strategy, trades, win_rate, profit_factor, total_pnl, max_drawdown, sharpe_approx, score}.
    Uses strategy_perf_trades if available, else falls back to closed deals.
    """
    import math
    try:
        init_db()
    except Exception:
        pass
    since_ts = now_ts() - (window_days * 86400)
    con = _conn()

    # Try strategy_perf_trades first
    try:
        rows = con.execute(
            """
            SELECT strategy, pnl, pnl_pct, ts FROM strategy_perf_trades
            WHERE ts >= ?
            ORDER BY ts ASC
            """,
            (since_ts,),
        ).fetchall()
        raw_trades = [{"strategy": r["strategy"], "pnl": float(r["pnl"] or 0), "pnl_pct": r["pnl_pct"], "ts": r["ts"]} for r in rows]
    except Exception:
        raw_trades = []

    # Fallback: build from closed deals if no strategy_perf_trades data
    if not raw_trades:
        try:
            deal_rows = con.execute(
                """
                SELECT exit_strategy, entry_strategy, realized_pnl_quote, closed_at, entry_avg, base_amount
                FROM deals
                WHERE state='CLOSED' AND closed_at IS NOT NULL AND closed_at >= ?
                ORDER BY closed_at ASC
                """,
                (since_ts,),
            ).fetchall()
            for r in deal_rows:
                strat = str(r["exit_strategy"] or r["entry_strategy"] or "classic_dca").strip() or "classic_dca"
                pnl = float(r["realized_pnl_quote"] or 0)
                entry = float(r["entry_avg"] or 0)
                base_amt = float(r["base_amount"] or 0)
                notional = entry * base_amt if (entry > 0 and base_amt > 0) else 0.0
                pnl_pct = (pnl / notional) if notional and notional > 0 else None
                raw_trades.append({"strategy": strat, "pnl": pnl, "pnl_pct": pnl_pct, "ts": r["closed_at"]})
        except Exception:
            pass
    con.close()
    if not raw_trades:
        return []

    rows = raw_trades
    by_strategy: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        s = str(r["strategy"] or "unknown").strip() or "unknown"
        if s not in by_strategy:
            by_strategy[s] = []
        by_strategy[s].append({
            "pnl": float(r["pnl"] or 0),
            "pnl_pct": float(r["pnl_pct"]) if r["pnl_pct"] is not None else None,
            "ts": int(r["ts"] or 0),
        })
    result = []
    for strategy, trades in by_strategy.items():
        pnls = [t["pnl"] for t in trades]
        pnl_pcts = [t["pnl_pct"] for t in trades if t["pnl_pct"] is not None]
        n = len(pnls)
        if n == 0:
            continue
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        win_rate = len(wins) / n
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (2.0 if gross_profit > 0 else 0.0)
        total_pnl = sum(pnls)
        peak = 0.0
        cum = 0.0
        max_dd = 0.0
        for p in pnls:
            cum += p
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)
        mean_pct = sum(pnl_pcts) / len(pnl_pcts) if pnl_pcts else 0.0
        std = 0.0
        if len(pnl_pcts) >= 5:
            variance = sum((x - mean_pct) ** 2 for x in pnl_pcts) / len(pnl_pcts)
            std = math.sqrt(variance) if variance > 0 else 0.0
        sharpe_approx = (mean_pct / std) if std > 0.0001 else 0.0
        score = (win_rate * 0.25) + (min(profit_factor, 3.0) / 3.0 * 0.25) + (max(0, 1.0 - max_dd / 50) * 0.25) + (max(0, min(sharpe_approx, 2.0)) / 2.0 * 0.25)
        result.append({
            "strategy": strategy,
            "trades": n,
            "win_rate": round(win_rate * 100, 1),
            "profit_factor": round(profit_factor, 2),
            "total_pnl": round(total_pnl, 2),
            "max_drawdown": round(max_dd, 2),
            "sharpe_approx": round(sharpe_approx, 2),
            "score": round(score * 100, 1),
        })
    result.sort(key=lambda x: x["score"], reverse=True)
    return result


def get_expected_edge(
    symbol: str,
    regime: str,
    strategy: str,
    window: int = 100,
    prior_weight: int = 50,
) -> Dict[str, Any]:
    """
    Bayesian-shrunk expected edge in pct terms using (symbol, regime, strategy).
    expected_edge = P_win * avg_win - P_loss * avg_loss
    """
    con = _conn()
    rows = con.execute(
        """
        SELECT pnl_pct FROM strategy_perf_trades
        WHERE symbol=? AND regime=? AND strategy=?
        ORDER BY ts DESC LIMIT ?
        """,
        (str(symbol), str(regime), str(strategy), int(window)),
    ).fetchall()
    global_rows = con.execute(
        "SELECT pnl_pct FROM strategy_perf_trades ORDER BY ts DESC LIMIT ?",
        (int(window),),
    ).fetchall()
    con.close()

    def _stats(rs):
        vals = [float(r["pnl_pct"]) for r in rs if r["pnl_pct"] is not None]
        wins = [v for v in vals if v > 0]
        losses = [v for v in vals if v < 0]
        p_win = len(wins) / len(vals) if vals else 0.0
        avg_win = (sum(wins) / len(wins)) if wins else 0.0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
        return len(vals), p_win, avg_win, avg_loss

    n, p_win, avg_win, avg_loss = _stats(rows)
    g_n, g_p_win, g_avg_win, g_avg_loss = _stats(global_rows)

    w = max(1, int(prior_weight))
    if g_n <= 0:
        g_p_win, g_avg_win, g_avg_loss = 0.5, 0.01, 0.01

    p_win_b = (n * p_win + w * g_p_win) / (n + w)
    avg_win_b = (n * avg_win + w * g_avg_win) / (n + w)
    avg_loss_b = (n * avg_loss + w * g_avg_loss) / (n + w)
    p_loss_b = 1.0 - p_win_b
    edge = (p_win_b * avg_win_b) - (p_loss_b * avg_loss_b)

    return {
        "trades": n,
        "p_win": p_win_b,
        "p_loss": p_loss_b,
        "avg_win": avg_win_b,
        "avg_loss": avg_loss_b,
        "expected_edge": edge,
    }


def count_orders_today(bot_id: int) -> int:
    """Count order events for bot in last 24h. Used by risk engine."""
    try:
        start = int(time.time()) - (24 * 3600)
        con = _conn()
        row = con.execute(
            "SELECT COUNT(*) as n FROM order_events WHERE bot_id=? AND ts>=?",
            (int(bot_id), start),
        ).fetchone()
        con.close()
        return int(row["n"]) if row else 0
    except Exception:
        return 0


def add_order_event(
    bot_id: int,
    symbol: str,
    side: str,
    ord_type: str,
    price: Optional[float],
    amount: Optional[float],
    order_id: Optional[str],
    tag: Optional[str],
    status: str,
    reason: str,
    is_live: int = 0,
) -> None:
    con = _conn()
    con.execute(
        """
        INSERT INTO order_events(bot_id, ts, symbol, side, ord_type, price, amount, order_id, tag, status, reason, is_live)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            int(bot_id),
            now_ts(),
            str(symbol),
            str(side),
            str(ord_type),
            float(price) if price is not None else None,
            float(amount) if amount is not None else None,
            str(order_id) if order_id else None,
            str(tag) if tag else None,
            str(status),
            str(reason),
            int(1 if is_live else 0),
        ),
    )
    con.commit()
    con.close()


def save_perf_metrics(bot_id: int, strategy: str, payload: str) -> None:
    con = _conn()
    con.execute(
        "INSERT INTO perf_metrics(bot_id, ts, strategy, payload) VALUES (?,?,?,?)",
        (int(bot_id), now_ts(), str(strategy), str(payload)),
    )
    con.commit()
    con.close()


def save_backtest_run(symbol: str, strategy: str, params: str, metrics: str, equity: str) -> None:
    con = _conn()
    con.execute(
        "INSERT INTO backtest_runs(ts, symbol, strategy, params, metrics, equity) VALUES (?,?,?,?,?,?)",
        (now_ts(), str(symbol), str(strategy), str(params), str(metrics), str(equity)),
    )
    con.commit()
    con.close()


def latest_regime(bot_id: int) -> Optional[Dict[str, Any]]:
    con = _conn()
    row = con.execute(
        "SELECT ts, symbol, regime, confidence, why, snapshot FROM regime_snapshots WHERE bot_id=? ORDER BY ts DESC LIMIT 1",
        (int(bot_id),),
    ).fetchone()
    con.close()
    return dict(row) if row else None


def list_strategy_decisions(bot_id: int, limit: int = 100) -> List[Dict[str, Any]]:
    con = _conn()
    rows = con.execute(
        """
        SELECT ts, strategy, action, reason, regime, confidence, payload
        FROM strategy_decisions
        WHERE bot_id=?
        ORDER BY ts DESC
        LIMIT ?
        """,
        (int(bot_id), int(limit)),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def list_backtest_runs(limit: int = 50) -> List[Dict[str, Any]]:
    con = _conn()
    rows = con.execute(
        """
        SELECT ts, symbol, strategy, params, metrics, equity
        FROM backtest_runs
        ORDER BY ts DESC
        LIMIT ?
        """,
        (int(limit),),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


# =========================================================
# Recommendations
# =========================================================
def save_recommendation_snapshot(
    symbol: str,
    horizon: str,
    score: float,
    regime_json: str,
    metrics_json: str,
    reasons_json: str,
    risk_flags_json: str,
) -> int:
    con = _conn()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO recommendations_snapshots(
            symbol, horizon, score, regime_json, metrics_json, reasons_json, risk_flags_json, created_ts
        ) VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            str(symbol),
            str(horizon),
            float(score),
            str(regime_json or ""),
            str(metrics_json or ""),
            str(reasons_json or ""),
            str(risk_flags_json or ""),
            now_ts(),
        ),
    )
    snapshot_id = int(cur.lastrowid)
    cur.execute(
        """
        INSERT INTO recommendations_latest(symbol, horizon, snapshot_id, created_ts)
        VALUES (?,?,?,?)
        ON CONFLICT(symbol, horizon) DO UPDATE SET snapshot_id=excluded.snapshot_id, created_ts=excluded.created_ts
        """,
        (str(symbol), str(horizon), snapshot_id, now_ts()),
    )
    con.commit()
    con.close()
    return snapshot_id


def list_recommendations(
    horizon: str, limit: int = 200, exclude_bases: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Fast query for recommendations. Returns empty list on error.
    exclude_bases: list of crypto base symbols to never return (e.g. ['STABLE'])."""
    try:
        con = _conn()
        con.execute("PRAGMA busy_timeout = 2000")
        rows = con.execute(
            """
            SELECT s.*
            FROM recommendations_snapshots s
            JOIN recommendations_latest l ON l.snapshot_id = s.id
            WHERE l.horizon=?
            ORDER BY s.score DESC, s.created_ts DESC
            LIMIT ?
            """,
            (str(horizon), int(limit)),
        ).fetchall()
        result = [dict(r) for r in rows]
        con.close()
        # Filter blocklisted crypto symbols
        if exclude_bases:
            exclude_set = {str(b).strip().upper() for b in exclude_bases if b}
            filtered = []
            for r in result:
                sym = str(r.get("symbol") or "")
                if "/" in sym:
                    base = (sym.split("/")[0] or "").upper()
                    if base in exclude_set:
                        continue
                filtered.append(r)
            return filtered
        return result
    except Exception as e:
        # Log but don't raise - return empty list
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"list_recommendations error: {e}")
        return []


def delete_recommendations_for_blocklist(bases: List[str]) -> int:
    """Remove recommendations for crypto symbols whose base is in the blocklist.
    Call on startup to purge STABLE and other blocked tokens from Explore.
    Returns count of symbols removed."""
    if not bases:
        return 0
    try:
        con = _conn()
        cur = con.cursor()
        deleted = 0
        for base in bases:
            b = str(base).strip().upper()
            if not b:
                continue
            # Match STABLE/USD, stable/usd, etc. (case-insensitive via UPPER)
            cur.execute(
                "DELETE FROM recommendations_latest WHERE UPPER(symbol) LIKE ? OR UPPER(symbol) = ?",
                (b + "/%", b),
            )
            deleted += cur.rowcount
        con.commit()
        con.close()
        return deleted
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("delete_recommendations_for_blocklist: %s", e)
        return 0


def get_recommendation(symbol: str, horizon: str) -> Optional[Dict[str, Any]]:
    con = _conn()
    row = con.execute(
        """
        SELECT s.*
        FROM recommendations_snapshots s
        JOIN recommendations_latest l ON l.snapshot_id = s.id
        WHERE l.symbol=? AND l.horizon=?
        LIMIT 1
        """,
        (str(symbol), str(horizon)),
    ).fetchone()
    con.close()
    return dict(row) if row else None


# =========================================================
# Recommendation performance tracking
# =========================================================
def link_recommendation_to_bot(
    bot_id: int,
    symbol: str,
    recommendation_date: int,
    score_at_recommendation: float,
    regime_at_recommendation: str,
    metrics_json: str = "",
    reasons_json: str = "",
    snapshot_id: Optional[int] = None,
) -> None:
    """Record that a bot was created from a recommendation. Creates recommendation_performance row with outcome='active'."""
    con = _conn()
    try:
        con.execute(
            """
            INSERT INTO recommendation_performance(
                symbol, recommendation_date, score_at_recommendation, regime_at_recommendation,
                bot_id, outcome, notes, technical_patterns_json, snapshot_id, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                str(symbol),
                int(recommendation_date),
                float(score_at_recommendation),
                str(regime_at_recommendation or ""),
                int(bot_id),
                "active",
                "",
                str(reasons_json or ""),
                int(snapshot_id) if snapshot_id else None,
                now_ts(),
            ),
        )
        con.commit()
    finally:
        con.close()


def _record_recommendation_outcome(
    con: sqlite3.Connection,
    bot_id: int,
    deal_id: int,
    entry_avg: float,
    exit_avg: float,
    realized_pnl: float,
    closed_ts: int,
    opened_ts: int,
) -> None:
    """Update recommendation_performance when a deal closes (first closed deal per bot from recommendation)."""
    row = con.execute(
        """
        SELECT id FROM recommendation_performance
        WHERE bot_id=? AND outcome='active'
        ORDER BY id ASC LIMIT 1
        """,
        (int(bot_id),),
    ).fetchone()
    if not row:
        return
    days_held = (closed_ts - opened_ts) / 86400.0 if closed_ts > opened_ts else 0.0
    outcome = "win" if realized_pnl > 0 else "loss"
    con.execute(
        """
        UPDATE recommendation_performance SET
            deal_id=?, entry_price=?, exit_price=?, pnl_realized=?, days_held=?, outcome=?
        WHERE id=?
        """,
        (int(deal_id), float(entry_avg), float(exit_avg), float(realized_pnl), float(days_held), outcome, int(row["id"])),
    )


def get_recommendation_performance_stats(days: int = 30) -> Dict[str, Any]:
    """Aggregate stats for closed recommendation outcomes. Used by /api/recommendations/performance."""
    since_ts = now_ts() - (int(days) * 86400)
    con = _conn()
    rows = con.execute(
        """
        SELECT score_at_recommendation, regime_at_recommendation, pnl_realized, outcome
        FROM recommendation_performance
        WHERE outcome IN ('win','loss') AND recommendation_date >= ?
        """,
        (since_ts,),
    ).fetchall()
    con.close()
    if not rows:
        return {
            "total_closed": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "avg_profit_per_recommendation": 0.0,
            "by_score_range": [],
            "by_regime": [],
        }
    wins = [r for r in rows if r["outcome"] == "win"]
    losses = [r for r in rows if r["outcome"] == "loss"]
    total = len(rows)
    pnls = [float(r["pnl_realized"] or 0) for r in rows]
    avg_pnl = sum(pnls) / total if total else 0.0

    # By score range
    def _score_bucket(s: float) -> str:
        if s >= 80:
            return "80-100"
        if s >= 60:
            return "60-80"
        if s >= 40:
            return "40-60"
        return "0-40"

    by_range: Dict[str, List[str]] = {}
    for r in rows:
        b = _score_bucket(float(r["score_at_recommendation"] or 0))
        if b not in by_range:
            by_range[b] = []
        by_range[b].append(r["outcome"])
    by_score_range = [
        {"range": k, "total": len(v), "wins": sum(1 for o in v if o == "win"), "win_rate": (sum(1 for o in v if o == "win") / len(v) * 100) if v else 0}
        for k, v in sorted(by_range.items(), key=lambda x: x[0])
    ]

    # By regime
    by_reg: Dict[str, List[str]] = {}
    for r in rows:
        reg = str(r["regime_at_recommendation"] or "").strip() or "unknown"
        if reg not in by_reg:
            by_reg[reg] = []
        by_reg[reg].append(r["outcome"])
    by_regime = [
        {"regime": k, "total": len(v), "wins": sum(1 for o in v if o == "win"), "win_rate": (sum(1 for o in v if o == "win") / len(v) * 100) if v else 0}
        for k, v in sorted(by_reg.items(), key=lambda x: -len(x[1]))
    ]

    return {
        "total_closed": total,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (len(wins) / total * 100) if total else 0.0,
        "avg_profit_per_recommendation": round(avg_pnl, 2),
        "by_score_range": by_score_range,
        "by_regime": by_regime,
    }


def save_scoring_calibration_log(
    scoring_version: str,
    changes_json: str,
    analysis_window_days: int,
    notes: str = "",
) -> None:
    """Log a calibration run for audit trail."""
    con = _conn()
    try:
        con.execute(
            """
            INSERT INTO scoring_calibration_log(ts, scoring_version, changes_json, analysis_window_days, notes)
            VALUES (?,?,?,?,?)
            """,
            (now_ts(), str(scoring_version), str(changes_json), int(analysis_window_days), str(notes)),
        )
        con.commit()
    finally:
        con.close()


def save_dividend_event(
    symbol: str,
    ex_date: int,
    amount: float,
    payment_date: Optional[int] = None,
    dividend_yield_pct: Optional[float] = None,
) -> None:
    """Record a dividend event for tracking."""
    con = _conn()
    try:
        con.execute(
            """
            INSERT INTO dividend_events(symbol, ex_date, payment_date, amount, dividend_yield_pct, recorded_at)
            VALUES (?,?,?,?,?,?)
            """,
            (str(symbol), int(ex_date), int(payment_date) if payment_date else None, float(amount), dividend_yield_pct, now_ts()),
        )
        con.commit()
    finally:
        con.close()


def list_dividend_events(symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """List dividend events, optionally filtered by symbol."""
    con = _conn()
    if symbol:
        rows = con.execute(
            "SELECT * FROM dividend_events WHERE symbol=? ORDER BY ex_date DESC LIMIT ?",
            (str(symbol), int(limit)),
        ).fetchall()
    else:
        rows = con.execute(
            "SELECT * FROM dividend_events ORDER BY ex_date DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def get_trade_journal(deal_id: int) -> Optional[Dict[str, Any]]:
    """Get journal entry for deal."""
    con = _conn()
    row = con.execute("SELECT * FROM trade_journal WHERE deal_id=?", (int(deal_id),)).fetchone()
    con.close()
    return dict(row) if row else None


def upsert_trade_journal(
    deal_id: int,
    entry_reason: Optional[str] = None,
    exit_reason: Optional[str] = None,
    lessons_learned: Optional[str] = None,
    screenshot_data: Optional[str] = None,
) -> None:
    """Create or update trade journal entry. Omit fields to leave unchanged on update."""
    con = _conn()
    now = now_ts()
    cur = con.execute("SELECT * FROM trade_journal WHERE deal_id=?", (int(deal_id),))
    existing = cur.fetchone()
    if existing:
        row = dict(existing)
        er = entry_reason if entry_reason is not None else (row.get("entry_reason") or "")
        xr = exit_reason if exit_reason is not None else (row.get("exit_reason") or "")
        ll = lessons_learned if lessons_learned is not None else (row.get("lessons_learned") or "")
        sc = screenshot_data if screenshot_data is not None else (row.get("screenshot_data") or "")
        con.execute(
            """UPDATE trade_journal SET entry_reason=?, exit_reason=?, lessons_learned=?, screenshot_data=?, updated_at=? WHERE deal_id=?""",
            (er, xr, ll, sc, now, int(deal_id)),
        )
    else:
        con.execute(
            """
            INSERT INTO trade_journal(deal_id, entry_reason, exit_reason, lessons_learned, screenshot_data, updated_at)
            VALUES (?,?,?,?,?,?)
            """,
            (int(deal_id), entry_reason or "", exit_reason or "", lessons_learned or "", screenshot_data or "", now),
        )
    con.commit()
    con.close()


def list_trade_journals_for_deals(deal_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """Return journal entries keyed by deal_id for given deals."""
    if not deal_ids:
        return {}
    con = _conn()
    placeholders = ",".join("?" * len(deal_ids))
    rows = con.execute(
        f"SELECT * FROM trade_journal WHERE deal_id IN ({placeholders})",
        deal_ids,
    ).fetchall()
    con.close()
    return {int(r["deal_id"]): dict(r) for r in rows}


def save_market_event(
    event_date: int,
    event_type: str,
    symbol: Optional[str] = None,
    impact_level: int = 2,
    description: str = "",
) -> None:
    """Record market event (earnings, Fed, etc.). event_date = Unix date midnight."""
    con = _conn()
    try:
        con.execute(
            """
            INSERT INTO market_events(event_date, event_type, symbol, impact_level, description, recorded_at)
            VALUES (?,?,?,?,?,?)
            """,
            (int(event_date), str(event_type), str(symbol or ""), int(impact_level), str(description or ""), now_ts()),
        )
        con.commit()
    finally:
        con.close()


def get_events_for_dates(start_ts: int, end_ts: int, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get market events in date range. start_ts/end_ts = date midnight Unix."""
    con = _conn()
    if symbol:
        rows = con.execute(
            "SELECT * FROM market_events WHERE event_date>=? AND event_date<=? AND (symbol=? OR symbol='' OR symbol IS NULL) ORDER BY event_date",
            (int(start_ts), int(end_ts), str(symbol)),
        ).fetchall()
    else:
        rows = con.execute(
            "SELECT * FROM market_events WHERE event_date>=? AND event_date<=? ORDER BY event_date",
            (int(start_ts), int(end_ts)),
        ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def save_insider_transaction(
    symbol: str,
    transaction_date: int,
    transaction_type: str,
    shares: float,
    value_usd: Optional[float] = None,
    insider_title: Optional[str] = None,
    filing_url: Optional[str] = None,
) -> None:
    """Record SEC Form 4 insider transaction."""
    con = _conn()
    try:
        con.execute(
            """
            INSERT INTO insider_transactions(symbol, transaction_date, transaction_type, shares, value_usd, insider_title, filing_url, recorded_at)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (str(symbol), int(transaction_date), str(transaction_type), float(shares), value_usd, insider_title or "", filing_url or "", now_ts()),
        )
        con.commit()
    finally:
        con.close()


def get_insider_transactions(symbol: str, days_back: int = 90) -> List[Dict[str, Any]]:
    """Get recent insider transactions for symbol."""
    from datetime import datetime, timezone, timedelta
    cutoff = int((datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp())
    con = _conn()
    rows = con.execute(
        "SELECT * FROM insider_transactions WHERE symbol=? AND transaction_date>=? ORDER BY transaction_date DESC LIMIT 100",
        (str(symbol), cutoff),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def save_ml_prediction(
    symbol: str,
    prediction_date: int,
    predicted_direction: str,
    predicted_price: Optional[float] = None,
    confidence: float = 0.0,
    price_at_prediction: Optional[float] = None,
    model_version: Optional[str] = None,
    model_used: Optional[str] = None,
    regime_at_prediction: Optional[str] = None,
) -> int:
    """Log ML prediction. Returns inserted row id."""
    con = _conn()
    try:
        cur = con.execute(
            """
            INSERT INTO ml_predictions(symbol, prediction_date, predicted_direction, predicted_price, confidence,
                price_at_prediction, model_version, model_used, regime_at_prediction, recorded_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (str(symbol), int(prediction_date), str(predicted_direction), predicted_price, float(confidence),
             price_at_prediction, model_version or "", model_used or "", regime_at_prediction or "", now_ts()),
        )
        con.commit()
        return cur.lastrowid or 0
    finally:
        con.close()


def update_ml_prediction_outcome(prediction_id: int, actual_outcome_7d: Optional[float] = None, actual_outcome_30d: Optional[float] = None) -> None:
    """Update prediction with actual outcome after 7/30 days."""
    con = _conn()
    try:
        if actual_outcome_7d is not None and actual_outcome_30d is not None:
            con.execute("UPDATE ml_predictions SET actual_outcome_7d=?, actual_outcome_30d=? WHERE id=?", (actual_outcome_7d, actual_outcome_30d, prediction_id))
        elif actual_outcome_7d is not None:
            con.execute("UPDATE ml_predictions SET actual_outcome_7d=? WHERE id=?", (actual_outcome_7d, prediction_id))
        elif actual_outcome_30d is not None:
            con.execute("UPDATE ml_predictions SET actual_outcome_30d=? WHERE id=?", (actual_outcome_30d, prediction_id))
        con.commit()
    finally:
        con.close()


def get_ml_predictions(symbol: Optional[str] = None, limit: int = 100, days_back: int = 0) -> List[Dict[str, Any]]:
    """Get ML predictions, optionally filtered by symbol."""
    con = _conn()
    cutoff = int(time.time()) - (days_back * 86400) if days_back > 0 else 0
    if symbol:
        rows = con.execute(
            "SELECT * FROM ml_predictions WHERE symbol=? AND recorded_at>=? ORDER BY recorded_at DESC LIMIT ?",
            (str(symbol), cutoff, limit),
        ).fetchall()
    else:
        rows = con.execute(
            "SELECT * FROM ml_predictions WHERE recorded_at>=? ORDER BY recorded_at DESC LIMIT ?",
            (cutoff, limit),
        ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def get_ml_model_accuracy(days_back: int = 30, model_used: Optional[str] = None) -> Dict[str, Any]:
    """Calculate model accuracy over rolling window. Returns accuracy, precision, recall, F1."""
    con = _conn()
    cutoff = int(time.time()) - (days_back * 86400)
    where = "WHERE recorded_at>=? AND actual_outcome_7d IS NOT NULL"
    params = [cutoff]
    if model_used:
        where += " AND (model_used=? OR model_version=?)"
        params.extend([model_used, model_used])
    params = tuple(params)
    rows = con.execute(
        f"SELECT predicted_direction, actual_outcome_7d FROM ml_predictions {where}",
        params,
    ).fetchall()
    con.close()
    if not rows:
        return {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5, "total": 0}
    correct = 0
    tp = fp = fn = tn = 0
    for r in rows:
        pred_up = str(r["predicted_direction"] or "").upper() == "UP"
        actual_up = float(r["actual_outcome_7d"] or 0) > 0
        if pred_up == actual_up:
            correct += 1
        if pred_up and actual_up:
            tp += 1
        elif pred_up and not actual_up:
            fp += 1
        elif not pred_up and actual_up:
            fn += 1
        else:
            tn += 1
    n = len(rows)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.5
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.5
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.5
    return {
        "accuracy": correct / n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total": n,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def save_ml_model_version(model_type: str, version: str, validation_accuracy: float, deployed: bool = False) -> None:
    """Record new model version after training."""
    con = _conn()
    try:
        con.execute(
            "INSERT INTO ml_model_versions(model_type, version, validation_accuracy, trained_at, deployed) VALUES (?,?,?,?,?)",
            (str(model_type), str(version), float(validation_accuracy), now_ts(), 1 if deployed else 0),
        )
        con.commit()
    finally:
        con.close()


def save_intraday_pattern(
    symbol: str,
    pattern_type: str,
    ts: int,
    price: Optional[float] = None,
    vwap: Optional[float] = None,
    or_high: Optional[float] = None,
    or_low: Optional[float] = None,
    volume_spike_ratio: Optional[float] = None,
    bot_id: Optional[int] = None,
    payload_json: str = "",
) -> None:
    """Save intraday pattern (opening range break, VWAP cross, volume spike) for analysis."""
    con = _conn()
    try:
        con.execute(
            """
            INSERT INTO intraday_patterns(symbol, pattern_type, ts, price, vwap, or_high, or_low, volume_spike_ratio, bot_id, payload_json)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (str(symbol), str(pattern_type), int(ts), price, vwap, or_high, or_low, volume_spike_ratio, int(bot_id) if bot_id else None, str(payload_json or "")),
        )
        con.commit()
    finally:
        con.close()


def add_intelligence_decision(
    bot_id: int,
    symbol: str,
    allowed_actions: str,
    final_action: str,
    final_reason: str,
    data_ok: bool,
    data_reasons: str,
    safety_allowed: str,
    safety_reasons: str,
    regime: str,
    regime_confidence: float,
    strategy_mode: str,
    entry_style: str,
    exit_style: str,
    base_size: float,
    order_type: str,
    manage_actions: str,
    proposed_orders: str,
    debug_json: str,
    execution_result: Optional[str] = None,
    realized_slippage: Optional[float] = None,
    fill_quality: Optional[str] = None,
) -> int:
    """Log an intelligence decision to the database."""
    import json
    con = _conn()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO intelligence_decisions(
            bot_id, ts, symbol, allowed_actions, final_action, final_reason,
            data_ok, data_reasons, safety_allowed, safety_reasons,
            regime, regime_confidence, strategy_mode, entry_style, exit_style,
            base_size, order_type, manage_actions, proposed_orders, debug_json,
            execution_result, realized_slippage, fill_quality
        )
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            int(bot_id),
            now_ts(),
            str(symbol),
            str(allowed_actions),
            str(final_action),
            str(final_reason),
            1 if data_ok else 0,
            str(data_reasons),
            str(safety_allowed),
            str(safety_reasons),
            str(regime),
            float(regime_confidence),
            str(strategy_mode),
            str(entry_style),
            str(exit_style),
            float(base_size),
            str(order_type),
            str(manage_actions),
            str(proposed_orders),
            str(debug_json),
            str(execution_result) if execution_result else None,
            float(realized_slippage) if realized_slippage is not None else None,
            str(fill_quality) if fill_quality else None,
        ),
    )
    decision_id = cur.lastrowid
    con.commit()
    con.close()
    return decision_id


def get_intelligence_decisions(bot_id: int, limit: int = 100) -> List[Dict[str, Any]]:
    """Get recent intelligence decisions for a bot."""
    con = _conn()
    rows = con.execute(
        """
        SELECT * FROM intelligence_decisions
        WHERE bot_id=?
        ORDER BY ts DESC
        LIMIT ?
        """,
        (int(bot_id), int(limit)),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def db_vacuum() -> None:
    """Run VACUUM to reclaim space and maintain performance. Safe but can take time on large DBs."""
    con = _conn()
    try:
        con.execute("VACUUM")
        con.commit()
    finally:
        con.close()


def db_analyze() -> None:
    """Run ANALYZE to update query planner statistics. Lightweight, safe to run regularly."""
    con = _conn()
    try:
        con.execute("ANALYZE")
        con.commit()
    finally:
        con.close()


def backup_db(dest_path: Optional[str] = None) -> str:
    """Copy DB to dest_path or defaults to {DB_NAME}.backup_{ts}. Returns path."""
    import shutil
    dest = dest_path or f"{DB_NAME}.backup_{now_ts()}"
    shutil.copy2(DB_NAME, dest)
    return dest
