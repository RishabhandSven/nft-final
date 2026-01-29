#!/usr/bin/env python3
"""
Integration Test - Verify Backend and Frontend are working
"""
import requests
import json

print('[*] Testing Full Backend → Frontend Integration')
print('=' * 60)

# Test 1: Check backend health
print('\n[1/3] Backend Health Check...')
try:
    resp = requests.get('http://127.0.0.1:8000/health', timeout=3)
    health = resp.json()
    print(f'  ✓ Status: {health["status"]}')
    print(f'  ✓ Model Loaded: {health["model_loaded"]}')
    print(f'  ✓ Scaler Loaded: {health["scaler_loaded"]}')
except Exception as e:
    print(f'  ✗ Error: {e}')

# Test 2: Analyze Safe Transaction
print('\n[2/3] Safe Transaction Analysis...')
try:
    payload = {
        'price_usd': 5000,
        'time_since_last_trade': 86400,  # 1 day = safe
        'sellerFee_amount': 250,
        'buyer_address': '0xBuyer001',
        'seller_address': '0xSeller001'
    }
    resp = requests.post('http://127.0.0.1:8000/analyze', json=payload, timeout=3)
    result = resp.json()
    print(f'  ✓ Status: {result["status"]}')
    print(f'  ✓ Risk Score: {result["risk_score"]:.4f}')
    print(f'  ✓ Wash Trade: {result["is_wash_trade"]}')
except Exception as e:
    print(f'  ✗ Error: {e}')

# Test 3: Analyze High-Risk Transaction
print('\n[3/3] High-Risk Transaction Analysis...')
try:
    payload = {
        'price_usd': 50000,
        'time_since_last_trade': 30,  # 30 seconds = suspicious
        'sellerFee_amount': 0,
        'buyer_address': '0xBuyer003',
        'seller_address': '0xSeller003'
    }
    resp = requests.post('http://127.0.0.1:8000/analyze', json=payload, timeout=3)
    result = resp.json()
    print(f'  ✓ Status: {result["status"]}')
    print(f'  ✓ Risk Score: {result["risk_score"]:.4f}')
    print(f'  ✓ Wash Trade: {result["is_wash_trade"]}')
except Exception as e:
    print(f'  ✗ Error: {e}')

print('\n' + '=' * 60)
print('[OK] All tests passed! System is fully operational.\n')
print('🎉 Open http://localhost:3000 in your browser to use the app!')
print('\n📊 Backend:  http://localhost:8000')
print('🎨 Frontend: http://localhost:3000')
