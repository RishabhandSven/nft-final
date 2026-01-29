#!/usr/bin/env python3
import requests
import json

url = "http://127.0.0.1:8000/analyze"
payload = {
    "price_usd": 5000.0,
    "time_since_last_trade": 30,
    "sellerFee_amount": 0.0,
    "buyer_address": "0xA",
    "seller_address": "0xB"
}

print("[*] Sending test request to /analyze...")
print("[*] Payload:", json.dumps(payload, indent=2))

try:
    resp = requests.post(url, json=payload, timeout=5)
    print("[*] Status:", resp.status_code)
    print("[*] Response:", json.dumps(resp.json(), indent=2))
except Exception as e:
    print("[!] Error:", e)
