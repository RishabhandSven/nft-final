#!/usr/bin/env python3
import subprocess
import time
import requests
import json
import signal
import os

# Start uvicorn on port 9000 in background
proc = subprocess.Popen(
    ['.venv\\Scripts\\python.exe', '-m', 'uvicorn', 'ai_engine.api:app', '--port', '9000'],
    cwd='D:\\nft2\\NFT',
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

try:
    # Wait for server to start
    time.sleep(4)
    
    # Test health
    print("[*] Testing /health...")
    resp = requests.get('http://127.0.0.1:9000/health', timeout=5)
    print("[*] Status:", resp.status_code)
    print("[*] Response:", json.dumps(resp.json(), indent=2))
    
    # Test analyze endpoint
    print("\n[*] Testing /analyze with high-risk scenario...")
    payload = {
        "price_usd": 5000.0,
        "time_since_last_trade": 30,
        "sellerFee_amount": 0.0,
        "buyer_address": "0xA",
        "seller_address": "0xB"
    }
    resp = requests.post('http://127.0.0.1:9000/analyze', json=payload, timeout=5)
    print("[*] Status:", resp.status_code)
    print("[*] Response:", json.dumps(resp.json(), indent=2))
    
finally:
    # Kill the process
    proc.terminate()
    proc.wait(timeout=3)
    print("\n[*] Server stopped")
