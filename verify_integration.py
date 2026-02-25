import sys
import os
import json

# Add project root to path
sys.path.append(os.getcwd())

# Mock environment variables if needed
os.environ["WORKER_API_TOKEN"] = "test"

try:
    from worker_api import _scan_symbol, _resolve_symbol
    print("Successfully imported worker_api")
except ImportError as e:
    print(f"Failed to import worker_api: {e}")
    sys.exit(1)

# Mock btc context
btc_ctx = {"risk_off": False}

print("Testing _scan_symbol with 'BTC/USD' (short horizon)...")
try:
    # Use a symbol that likely has data or at least won't crash
    # If fetch fails, it should handle gracefully
    result = _scan_symbol("BTC/USD", "short", btc_ctx)
    print("Result keys:", list(result.keys()))
    print("Score:", result.get("score"))
    print("Regime:", result.get("regime"))
    print("Eligible:", result.get("eligible"))
    
    if "regime" in result and "score" in result:
        print("SUCCESS: Structure is valid")
    else:
        print("FAILURE: Missing keys")
        
except Exception as e:
    print(f"Execution failed: {e}")
    import traceback
    traceback.print_exc()
