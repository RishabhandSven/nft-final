"""Forensic Report: Pull worst-offender transaction hashes for Etherscan verification."""
import pandas as pd
import joblib

df = pd.read_parquet("data/results/processed_sample.parquet")
model = joblib.load("data/results/wash_trading_iso_forest.pkl")
scaler = joblib.load("data/results/scaler_iso_forest.pkl")

feature_cols = [
    "price", "in_degree", "out_degree", "degree_centrality",
    "cycle_count", "price_vs_avg", "time_delta", "zero_spread_flag",
]
X = df[feature_cols].fillna(0)
X_scaled = scaler.transform(X)
df["is_anomaly"] = model.predict(X_scaled)

wash_trades = df[df["is_anomaly"] == -1]
top = wash_trades[wash_trades["zero_spread_flag"] == 1].sort_values("price", ascending=False).head(5)

print("\n=== WASH TRADING FORENSIC REPORT ===")
print(f"Total Scams Found: {len(wash_trades):,}")
print("\nTop 5 Most Obvious (Full Transaction Hashes - paste into Etherscan.io):")
for _, row in top.iterrows():
    tx = row["transaction_hash"]
    p = row["price"]
    c = row["cycle_count"]
    t = row["token_id"]
    print(f"  {tx}  |  Price: ${p:,.0f}  |  Cycles: {c}  |  Token: {t}")
