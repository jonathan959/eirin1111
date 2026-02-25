from alpaca_client import AlpacaClient
import os
import json

def test_fetch():
    print("Initializing Client...")
    # Load env vars manually if needed, but assuming they are set or loaded by the environment
    # Actually, for this standalone script, I need to load .env
    
    with open(".env", "r") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                os.environ[k] = v

    mode = "live" if os.getenv("ALPACA_API_KEY_LIVE") else "paper"
    print(f"Mode: {mode}")
    
    client = AlpacaClient(mode=mode)
    
    symbol = "HUT"
    print(f"Fetching 1D bars for {symbol}...")
    bars_1d = client.get_ohlcv(symbol, "1d", limit=10)
    print(f"1D Bars: {len(bars_1d)}")
    if bars_1d:
        print(f"Sample: {bars_1d[0]}")
    else:
        print("No 1D bars found.")
        
    print(f"Fetching 1W bars for {symbol}...")
    bars_1w = client.get_ohlcv(symbol, "1w", limit=10)
    print(f"1W Bars: {len(bars_1w)}")
    if bars_1w:
        print(f"Sample: {bars_1w[0]}")
    else:
        print("No 1W bars found.")

    # Check active assets
    print("Fetching active assets...")
    assets = client.get_active_assets()
    print(f"Active Assets: {len(assets)}")
    if assets:
        print(f"First 3: {[a['symbol'] for a in assets[:3]]}")

if __name__ == "__main__":
    try:
        test_fetch()
    except Exception as e:
        print(f"Error: {e}")
