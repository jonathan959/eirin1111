"""
Phase 2 data fetcher - unified access to candles, order book, trades.
Routes by symbol type to Kraken (crypto) or Alpaca (stocks).
Yahoo Finance fallback for stocks when Alpaca returns no/insufficient data (123.md Fix 2).
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_cached_clients: Dict[str, Any] = {}


def _get_clients() -> tuple:
    """Get (kc, alpaca_client) from worker_api or similar."""
    global _cached_clients
    if _cached_clients:
        return _cached_clients.get("kc"), _cached_clients.get("alpaca")
    try:
        from worker_api import kc, alpaca_paper, alpaca_live
        from alpaca_adapter import AlpacaAdapter
        alpaca = alpaca_live or alpaca_paper
        adapter = AlpacaAdapter(alpaca) if alpaca else None
        _cached_clients = {"kc": kc, "alpaca": adapter}
        return kc, adapter
    except Exception as e:
        logger.debug("phase2_data_fetcher: %s", e)
        return None, None


def _is_stock(symbol: str) -> bool:
    try:
        from symbol_classifier import is_stock_symbol
        return is_stock_symbol(symbol)
    except Exception:
        return len(symbol) < 6 and "/" not in symbol


def _fetch_candles_yahoo(symbol: str, timeframe: str, periods: int) -> List[List[float]]:
    """Yahoo Finance fallback for stocks. Returns [[ts_ms, o, h, l, c, v], ...]."""
    try:
        import yfinance as yf
        from datetime import datetime, timezone, timedelta
    except ImportError:
        logger.debug("yfinance not installed, Yahoo fallback unavailable")
        return []

    tf_lower = (timeframe or "1h").lower()
    interval_map = {
        "1m": "1m", "5m": "5m", "15m": "15m",
        "1h": "1h", "1hour": "1h",
        "4h": "1h", "4hour": "1h",
        "1d": "1d", "1day": "1d", "1w": "1wk",
    }
    interval = interval_map.get(tf_lower, "1d")
    is_intraday = tf_lower in ("1m", "5m", "15m", "1h", "1hour", "4h", "4hour")

    sym_clean = (symbol or "").replace("/USD", "").replace("-USD", "").replace("/", "").strip()
    if not sym_clean:
        return []

    def _history_with_fallback(period=None, start=None, end=None):
        ticker = yf.Ticker(sym_clean)
        if start and end:
            return ticker.history(start=start, end=end, interval=interval, auto_adjust=True)
        return ticker.history(period=period or "1mo", interval=interval, auto_adjust=True)

    def _to_rows(df):
        if df is None or df.empty:
            return []
        if tf_lower in ("4h", "4hour") and interval == "1h":
            df = df.resample("4h").agg({
                "Open": "first", "High": "max", "Low": "min",
                "Close": "last", "Volume": "sum"
            }).dropna()
        out = []
        for _, row in df.tail(min(periods, 500)).iterrows():
            ts = int(row.name.timestamp() * 1000) if hasattr(row.name, "timestamp") else 0
            o = float(row.get("Open", 0) or 0)
            h = float(row.get("High", 0) or o)
            l = float(row.get("Low", 0) or o)
            c = float(row.get("Close", 0) or o)
            v = float(row.get("Volume", 0) or 0)
            out.append([ts, o, h, l, c, v])
        return out

    # Intraday (1h, 4h): use start/end - yfinance limits 1h to ~60d
    if is_intraday:
        end_d = datetime.now(timezone.utc)
        for days in [60, 30, 14, 7]:
            try:
                start_d = end_d - timedelta(days=days)
                df = _history_with_fallback(start=start_d, end=end_d)
                out = _to_rows(df)
                if out:
                    logger.info("Yahoo %s %s (intraday %dd): %d candles", symbol, tf_lower, days, len(out))
                    return out
            except Exception as e:
                logger.debug("Yahoo %s %s intraday %dd: %s", symbol, tf_lower, days, e)
        # Fallback: use daily data for 4h (resample) or 1h (1 bar per day)
        try:
            df = yf.Ticker(sym_clean).history(period="6mo", interval="1d", auto_adjust=True)
            if df is not None and not df.empty and len(df) >= 20:
                if tf_lower in ("4h", "4hour"):
                    df = df.resample("4h").agg({
                        "Open": "first", "High": "max", "Low": "min",
                        "Close": "last", "Volume": "sum"
                    }).dropna()
                out = []
                for _, row in df.tail(min(periods, 500)).iterrows():
                    ts = int(row.name.timestamp() * 1000) if hasattr(row.name, "timestamp") else 0
                    o = float(row.get("Open", 0) or 0)
                    h = float(row.get("High", 0) or o)
                    l = float(row.get("Low", 0) or o)
                    c = float(row.get("Close", 0) or o)
                    v = float(row.get("Volume", 0) or 0)
                    out.append([ts, o, h, l, c, v])
                if out:
                    logger.info("Yahoo %s %s (daily fallback): %d candles", symbol, tf_lower, len(out))
                    return out
        except Exception:
            pass

    # Daily/weekly: use period
    for period in ["2y", "1y", "6mo", "3mo", "1mo"]:
        try:
            df = _history_with_fallback(period=period)
            out = _to_rows(df)
            if out:
                logger.info("Yahoo %s %s period=%s: %d candles", symbol, tf_lower, period, len(out))
                return out
        except Exception as e:
            logger.debug("Yahoo %s %s period=%s: %s", symbol, tf_lower, period, e)
    return []


def fetch_recent_candles(
    symbol: str,
    timeframe: str = "1h",
    periods: int = 100,
) -> List[List[float]]:
    """Fetch OHLCV candles [[ts, o, h, l, c, v], ...]. Yahoo fallback for stocks."""
    kc, alpaca = _get_clients()
    is_stock_sym = _is_stock(symbol)
    min_acceptable = max(20, int(periods * 0.4))
    req_limit = max(periods + 50, min_acceptable + 30)

    ohlcv = []
    try:
        if is_stock_sym and alpaca:
            ohlcv = alpaca.fetch_ohlcv(symbol, timeframe=timeframe, limit=req_limit) or []
            if len(ohlcv) < min_acceptable:
                ohlcv = _fetch_candles_yahoo(symbol, timeframe, req_limit) or ohlcv
        if (not ohlcv or len(ohlcv) < min_acceptable) and kc and hasattr(kc, "fetch_ohlcv"):
            ohlcv = kc.fetch_ohlcv(symbol, timeframe=timeframe, limit=req_limit) or []

        if (not ohlcv or len(ohlcv) < min_acceptable) and is_stock_sym:
            ohlcv = _fetch_candles_yahoo(symbol, timeframe, req_limit) or ohlcv
    except Exception as e:
        logger.debug("fetch_recent_candles %s: %s", symbol, e)
        if is_stock_sym:
            ohlcv = _fetch_candles_yahoo(symbol, timeframe, req_limit) or []

    return ohlcv[-periods:] if ohlcv else []


def fetch_order_book(symbol: str, depth: int = 50) -> Dict[str, Any]:
    """
    Fetch order book. Returns {"bids": [[price, size], ...], "asks": [[price, size], ...]}.
    """
    kc, alpaca = _get_clients()
    is_stock_sym = _is_stock(symbol)
    try:
        if is_stock_sym and alpaca:
            ob = alpaca.fetch_order_book(symbol, limit=depth)
            return ob or {"bids": [], "asks": []}
        if kc and hasattr(kc, "fetch_order_book"):
            ob = kc.fetch_order_book(symbol, limit=depth)
            return ob or {"bids": [], "asks": []}
    except Exception as e:
        logger.debug("fetch_order_book %s: %s", symbol, e)
    return {"bids": [], "asks": []}


def fetch_recent_trades(symbol: str, limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Fetch recent trades. Each: {price, size, side, timestamp}.
    """
    kc, alpaca = _get_clients()
    is_stock_sym = _is_stock(symbol)
    try:
        if kc and hasattr(kc, "ex") and hasattr(kc.ex, "fetch_trades"):
            trades = kc.ex.fetch_trades(symbol, limit=limit)
            out = []
            for t in trades or []:
                d = t if isinstance(t, dict) else {}
                price = d.get("price") or d.get("last")
                amount = d.get("amount") or d.get("cost", 0) / (price or 1)
                side = (d.get("side") or "buy").lower()
                out.append({"price": float(price or 0), "size": float(amount or 0), "side": side})
            return out
        # Alpaca: trades API differs; skip if not available
    except Exception as e:
        logger.debug("fetch_recent_trades %s: %s", symbol, e)
    return []


def get_current_price(symbol: str) -> float:
    """Get current mid/last price."""
    kc, alpaca = _get_clients()
    is_stock_sym = _is_stock(symbol)
    try:
        if is_stock_sym and alpaca and hasattr(alpaca, "client"):
            t = alpaca.client.get_ticker(symbol)
            if t:
                bid = float(t.get("bid") or 0)
                ask = float(t.get("ask") or 0)
                last = float(t.get("last") or 0)
                if bid > 0 and ask > 0:
                    return (bid + ask) / 2
                return last if last > 0 else 0
        if kc and hasattr(kc, "fetch_ticker"):
            ticker = kc.fetch_ticker(symbol)
            if ticker:
                return float(ticker.get("last") or ticker.get("close") or 0)
    except Exception as e:
        logger.debug("get_current_price %s: %s", symbol, e)
    return 0.0


def get_current_spread(symbol: str) -> float:
    """Get current spread as decimal (e.g. 0.001 = 0.1%)."""
    ob = fetch_order_book(symbol)
    bids = ob.get("bids") or []
    asks = ob.get("asks") or []
    if not bids or not asks:
        return 0.0
    best_bid = float(bids[0][0]) if bids else 0
    best_ask = float(asks[0][0]) if asks else 0
    if best_bid <= 0:
        return 0.0
    return (best_ask - best_bid) / best_bid
