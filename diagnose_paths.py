#!/usr/bin/env python3
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AI_ENGINE_DIR = os.path.join(SCRIPT_DIR, "ai_engine")
MODEL_PATH = os.path.join(AI_ENGINE_DIR, "..", "data", "results", "wash_trading_brain.pkl")
SCALER_PATH = os.path.join(AI_ENGINE_DIR, "..", "data", "results", "scaler.pkl")

print("[*] SCRIPT_DIR:", SCRIPT_DIR)
print("[*] AI_ENGINE_DIR:", AI_ENGINE_DIR)
print("[*] MODEL_PATH (relative):", MODEL_PATH)
print("[*] MODEL_PATH (absolute):", os.path.abspath(MODEL_PATH))
print("[*] Model exists:", os.path.exists(os.path.abspath(MODEL_PATH)))
print("[*] SCALER_PATH (absolute):", os.path.abspath(SCALER_PATH))
print("[*] Scaler exists:", os.path.exists(os.path.abspath(SCALER_PATH)))

# Also list what's in data/results
results_dir = os.path.join(SCRIPT_DIR, "data", "results")
print("\n[*] Contents of data/results/:")
if os.path.exists(results_dir):
    for item in os.listdir(results_dir):
        print("  -", item)
else:
    print("  [!] Directory doesn't exist!")
