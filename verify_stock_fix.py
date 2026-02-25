import requests
import json
import time

def verify_stock_scan():
    url = "http://localhost:8000/api/recommendations/scan_stocks"
    params = {"horizon": "short", "limit": 20} # Small limit for quick test
    
    print(f"Triggering scan at {url}...")
    try:
        resp = requests.post(url, params=params)
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.json()}")
        
        if resp.status_code == 200:
            print("\nScanning triggered. Waiting a moment for results processing if async (assuming synchronous for this endpoint based on code)...")
            
            # Fetch recommendations results
            reco_url = "http://localhost:8000/api/recommendations"
            reco_params = {"market_type": "stocks", "limit": 20}
            
            reco_resp = requests.get(reco_url, params=reco_params)
            data = reco_resp.json()
            
            items = data.get("items", [])
            print(f"\nFound {len(items)} stock recommendations:")
            for item in items:
                print(f"Symbol: {item['symbol']}, Score: {item['score']}, Rec: {item['label']}, Price: {item['price']}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    verify_stock_scan()
