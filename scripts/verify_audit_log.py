import pandas as pd
import hashlib
import json

# === CONFIG ===
INPUT_CSV = "data/results/hash_chained_audit_log.csv"

def compute_hash(data_string):
    return hashlib.sha256(data_string.encode("utf-8")).hexdigest()

def verify_hash_chain(df):
    previous_hash = "0" * 64
    tampered_rows = []

    for index, row in df.iterrows():
        record = {
            "timestamp": row["timestamp"],  # must use stored timestamp
            "bitcoins": f"{float(row['bitcoins']):.8f}",
            "price_deviation": f"{float(row['price_deviation']):.8f}",
            "time_since_last_trade": int(row["time_since_last_trade"]),
            "wash": int(row["wash"]),
            "previous_hash": previous_hash
        }

        record_string = json.dumps(record, sort_keys=True)
        recalculated_hash = compute_hash(record_string)

        if recalculated_hash != row["current_hash"]:
            tampered_rows.append(index)

        previous_hash = row["current_hash"]

    if tampered_rows:
        print(f"❌ Tampering detected at rows: {tampered_rows}")
    else:
        print("✅ Audit log verified successfully. No tampering detected.")

if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)
    verify_hash_chain(df)
