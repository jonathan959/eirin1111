import requests
import json

try:
    url = "http://127.0.0.1:8000/api/bots/8/start"
    print(f"Calling POST {url}...")
    resp = requests.post(url)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.text}")
    
    if resp.ok:
        print("\nChecking bot status...")
        url_status = "http://127.0.0.1:8000/api/bots/8"
        resp_status = requests.get(url_status)
        print(f"Bot 8 Status: {resp_status.text}")
except Exception as e:
    print(f"Error: {e}")
