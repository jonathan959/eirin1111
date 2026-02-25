#!/usr/bin/env python3
import os
import sys

# Load env
try:
    with open(".env") as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                parts = line.strip().split("=", 1)
                if len(parts) == 2:
                    k, v = parts
                    os.environ[k] = v.strip('"').strip("'")
except Exception as e:
    print(f"Env load error: {e}")

from alpaca_client import AlpacaClient
print("Creating Alpaca client (paper)...")
client = AlpacaClient(mode="paper")

print("\nTesting market open status...")
try:
    is_open = client.is_market_open()
    print(f"Market open: {is_open}")
except Exception as e:
    print(f"Market open check failed: {type(e).__name__}: {e}")

print("\nTesting INTC ticker...")
try:
    result = client.get_ticker("INTC")
    print(f"Result: {result}")
except Exception as e:
    print(f"Ticker failed: {type(e).__name__}: {e}")

print("\nTesting INTC snapshot...")
try:
    snap = client._request("GET", "/v2/stocks/snapshots", 
                          params={"symbols": "INTC"}, 
                          base_url=client.data_url)
    print(f"Snapshot: {snap}")
except Exception as e:
    print(f"Snapshot failed: {type(e).__name__}: {e}")
