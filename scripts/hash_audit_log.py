import pandas as pd
import hashlib
import json
from datetime import datetime

# === CONFIG ===
INPUT_CSV = "data/ml_sample/gox_ml_samples.csv"
OUTPUT_CSV = "data/results/hash_chained_audit_log.csv"

def compute_hash(data_string):
    return hashlib.sha256(data_string.encode("utf-8")).hexdigest()

def create_hash_chained_log(df):
    previous_hash = "0" * 64  # Genesis hash
    hashes = []

    for _, row in df.iterrows():
        record = {
    "timestamp": datetime.utcnow().isoformat(),
    "bitcoins": f"{row['bitcoins']:.8f}",
    "price_deviation": f"{row['price_deviation']:.8f}",
    "time_since_last_trade": int(row["time_since_last_trade"]),
    "wash": int(row["wash"]),
    "previous_hash": previous_hash
    }


        record_string = json.dumps(record, sort_keys=True)
        current_hash = compute_hash(record_string)

        record["current_hash"] = current_hash
        hashes.append(record)

        previous_hash = current_hash

    return pd.DataFrame(hashes)

if __name__ == "__main__":
    print("Generating hash-chained audit log...")

    df = pd.read_csv(INPUT_CSV).head(5000)
    audit_df = create_hash_chained_log(df)

    audit_df.to_csv(OUTPUT_CSV, index=False)

    print("Hash-chained audit log created at:")
    print(OUTPUT_CSV)
