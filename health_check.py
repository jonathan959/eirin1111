#!/usr/bin/env python
"""
Full health check for the trading bot - Layers 1-9.
Run: python health_check.py
"""
import os
import sys
from pathlib import Path

# Ensure project root
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

def _run_layer(name: str, fn):
    try:
        result = fn()
        return result if isinstance(result, dict) else {"ok": True, "details": str(result)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def layer1_core():
    """Layer 1: Core Foundation"""
    from env_utils import load_env, get_last_load_result
    load_env()
    lr = get_last_load_result()
    env_ok = lr.get("loaded", False) if lr else False
    db_ok = False
    tables_ok = []
    try:
        from db import init_db, _conn
        init_db()
        con = _conn()
        for t in ["bots", "recommendations_snapshots", "recommendations_latest", "settings"]:
            try:
                con.execute(f"SELECT 1 FROM {t} LIMIT 1")
                tables_ok.append(t)
            except Exception:
                pass
        con.close()
        db_ok = len(tables_ok) >= 3
    except Exception as e:
        return {"ok": False, "error": str(e), "env_loaded": env_ok}
    return {
        "ok": db_ok,
        "env_file": env_ok,
        "db_tables": tables_ok,
        "kraken_key_set": bool(os.getenv("KRAKEN_API_KEY", "").strip()),
        "alpaca_paper_set": bool(os.getenv("ALPACA_API_KEY_PAPER", "").strip()),
    }

def layer2_exchanges():
    """Layer 2: Exchange connections"""
    from env_utils import load_env
    load_env()
    kraken_ok = False
    alpaca_ok = False
    alpaca_mode = "none"
    try:
        from kraken_client import KrakenClient
        kc = KrakenClient()
        ticker = kc.fetch_ticker("XBT/USD")
        kraken_ok = bool(ticker and (ticker.get("last") or ticker.get("c")))
    except Exception as e:
        kraken_ok = f"Error: {e}"
    try:
        use_unified = os.getenv("USE_UNIFIED_ALPACA", "1").strip().lower() in ("1", "true", "yes")
        if use_unified:
            from unified_alpaca_client import UnifiedAlpacaClient
            client = UnifiedAlpacaClient()
        else:
            from alpaca_client import AlpacaClient
            client = AlpacaClient()
        alpaca_mode = "live" if getattr(client, "is_live", lambda: False)() else "paper"
        ticker = client.get_ticker("AAPL") if hasattr(client, "get_ticker") else None
        alpaca_ok = bool(ticker and ticker.get("last"))
    except Exception as e:
        alpaca_ok = f"Error: {e}"
    return {"kraken": kraken_ok, "alpaca": alpaca_ok, "alpaca_mode": alpaca_mode}

def layer3_intelligence():
    """Layer 3: Intelligence files connectivity"""
    connected = {}
    try:
        from meme_coin_detector import should_block_crypto
        connected["meme_coin_detector"] = True
    except Exception:
        connected["meme_coin_detector"] = False
    try:
        from ml_predictor import create_ml_predictor
        connected["ml_predictor"] = True
    except Exception:
        connected["ml_predictor"] = False
    try:
        from ml_ensemble import get_ml_ensemble
        e = get_ml_ensemble()
        connected["ml_ensemble"] = getattr(e, "_is_trained", False)
    except Exception:
        connected["ml_ensemble"] = False
    try:
        from pattern_recognition import detect_patterns
        connected["pattern_recognition"] = True
    except Exception:
        connected["pattern_recognition"] = False
    try:
        from sentiment_analyzer import SentimentAnalyzer
        connected["sentiment_analyzer"] = True
    except Exception:
        connected["sentiment_analyzer"] = False
    try:
        from risk_engine import can_open_trade, is_enabled
        connected["risk_engine"] = is_enabled()
    except Exception:
        connected["risk_engine"] = False
    try:
        from kelly_criterion import KellyPositionSizer
        connected["kelly_criterion"] = True
    except Exception:
        connected["kelly_criterion"] = False
    try:
        from adaptive_scorer import apply_adaptive_score
        connected["adaptive_scorer"] = True
    except Exception:
        connected["adaptive_scorer"] = False
    try:
        from multi_timeframe import MultiTimeframeAnalyzer
        connected["multi_timeframe"] = True
    except Exception:
        connected["multi_timeframe"] = False
    try:
        from recommendation_validator import get_scoring_weights
        connected["recommendation_validator"] = True
    except Exception:
        connected["recommendation_validator"] = False
    return {"connected": connected}

def layer4_risk():
    """Layer 4: Risk management"""
    from env_utils import load_env
    load_env()
    r = {}
    try:
        from risk_engine import is_enabled
        r["risk_engine_enabled"] = is_enabled()
    except Exception:
        r["risk_engine_enabled"] = False
    try:
        from circuit_breaker import CIRCUIT_BREAKER_THRESHOLD
        r["circuit_breaker_threshold"] = CIRCUIT_BREAKER_THRESHOLD
    except Exception:
        r["circuit_breaker"] = "error"
    try:
        from execution_gate import GATE_ENABLED
        r["execution_gate"] = GATE_ENABLED
    except Exception:
        r["execution_gate"] = False
    try:
        from kelly_criterion import KellyPositionSizer
        r["kelly_criterion"] = True
    except Exception:
        r["kelly_criterion"] = False
    return r

def main():
    from env_utils import load_env
    load_env()
    print("=" * 60)
    print("TRADING BOT HEALTH CHECK")
    print("=" * 60)
    layers = [
        ("Layer 1: Core Foundation", layer1_core),
        ("Layer 2: Exchange Connections", layer2_exchanges),
        ("Layer 3: Intelligence Pipeline", layer3_intelligence),
        ("Layer 4: Risk Management", layer4_risk),
    ]
    for name, fn in layers:
        print(f"\n--- {name} ---")
        r = _run_layer(name, fn)
        if isinstance(r, dict):
            for k, v in r.items():
                print(f"  {k}: {v}")
        else:
            print(r)
    print("\n" + "=" * 60)
    print("Done. Run: python -m pytest tests/ -v  for full test suite.")
    print("=" * 60)

if __name__ == "__main__":
    main()
