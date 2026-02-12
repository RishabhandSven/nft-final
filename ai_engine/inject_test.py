#!/usr/bin/env python3
"""
Validation Injection: Inject 200 obvious wash trades, train, and measure Recall.
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "training_chunk.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "data", "results", "wash_trading_brain.pkl")
SCALER_PATH = os.path.join(SCRIPT_DIR, "..", "data", "results", "scaler.pkl")

FEATURES = ['price_usd', 'time_since_last_trade', 'sellerFee_amount', 'is_circular', 'was_holding_previously']
NUM_FAKE_WASH = 200


def generate_fake_wash_trades(n: int) -> pd.DataFrame:
    """Generate obvious wash trades: high price, A->B->A pattern (is_circular=1, was_holding_previously=1)."""
    return pd.DataFrame({
        'price_usd': np.full(n, 50_000),           # High price
        'time_since_last_trade': np.full(n, 60.0), # Suspiciously fast (60 sec)
        'sellerFee_amount': np.full(n, 1_250.0),   # 2.5% of 50k
        'is_circular': np.ones(n, dtype=int),      # A->B->A: buyer==seller
        'was_holding_previously': np.ones(n, dtype=int),  # Buyer sold in past trades
    })


def main():
    print("[*] Loading real data...")
    if not os.path.exists(DATA_PATH):
        print("[!] ERROR: training_chunk.csv not found. Run extract + processor first.")
        return

    real = pd.read_csv(DATA_PATH).fillna(0)

    # Ensure all features exist
    for col in FEATURES:
        if col not in real.columns:
            real[col] = 0

    real = real[FEATURES]
    print(f"    Loaded {len(real):,} real rows")

    print(f"[*] Generating {NUM_FAKE_WASH} obvious wash trades...")
    fake = generate_fake_wash_trades(NUM_FAKE_WASH)

    print("[*] Combining real + fake...")
    combined = pd.concat([real, fake], ignore_index=True)
    print(f"    Total: {len(combined):,} rows")

    print("[*] Training Isolation Forest...")
    scaler = StandardScaler()
    X = combined[FEATURES]
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(contamination=0.01, n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_scaled)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"    Model saved to {MODEL_PATH}")

    print("[*] Evaluating on the 200 injected fake wash trades...")
    X_fake = scaler.transform(fake[FEATURES])
    preds = model.predict(X_fake)

    caught = (preds == -1).sum()
    recall_pct = caught / NUM_FAKE_WASH * 100

    print()
    print("=" * 50)
    print("VALIDATION INJECTION RESULTS")
    print("=" * 50)
    print(f"Fake wash trades injected:  {NUM_FAKE_WASH}")
    print(f"Caught (predicted -1):      {caught}")
    print(f"Recall:                     {recall_pct:.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
