#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import gc

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
HUGE_FILE = os.path.join(DATA_DIR, "nft_transactions_huge.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "training_chunk.csv")

LOOKBACK_TRADES = 10  # Past N trades to check for was_holding_previously

# Memory-efficient dtypes (reduces RAM ~90% for addresses)
DTYPE_SPEC = {
    'buyerAddress': 'category',
    'sellerAddress': 'category',
    'tokenId': 'category',
    'price_usd': np.float32,
    'sellerFee_amount': np.float32,
}

print("[*] Processor starting...")
print("[*] Input file:", HUGE_FILE)
print("[*] Output file:", OUTPUT_FILE)

if not os.path.exists(HUGE_FILE):
    print("[!] ERROR: Input file not found!")
    exit(1)

print("[+] Loading with memory-efficient dtypes (category, float32)...")

try:
    cols_needed = ['buyerAddress', 'sellerAddress', 'price_usd', 'sellerFee_amount', 'timestamp', 'tokenId']

    df = pd.read_csv(
        HUGE_FILE,
        usecols=cols_needed,
        dtype=DTYPE_SPEC,
        parse_dates=['timestamp'],
    )
    print("[*] Loaded {} rows. Memory: ~{:.1f} MB".format(len(df), df.memory_usage(deep=True).sum() / 1024 / 1024))

    # Sort in-place (History/Graph check requires chronological order per token)
    df.sort_values(['tokenId', 'timestamp'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Feature engineering
    df['time_since_last_trade'] = df.groupby('tokenId')['timestamp'].diff().dt.total_seconds().fillna(3600).astype(np.float32)
    df['is_circular'] = (df['buyerAddress'].astype(str).str.strip().str.lower() == df['sellerAddress'].astype(str).str.strip().str.lower()).astype(np.int8)

    # was_holding_previously: Buyer == Previous Seller (History/Graph check)
    def compute_was_holding(group):
        buyers = group['buyerAddress'].astype(str).str.strip().str.lower()
        sellers = group['sellerAddress'].astype(str).str.strip().str.lower()
        n = len(group)
        result = np.zeros(n, dtype=np.int8)
        for i in range(1, n):
            start = max(0, i - LOOKBACK_TRADES)
            past_sellers = set(sellers.iloc[start:i])
            if buyers.iloc[i] in past_sellers:
                result[i] = 1
        return pd.Series(result, index=group.index)

    df['was_holding_previously'] = df.groupby('tokenId', group_keys=False).apply(compute_was_holding, include_groups=False)

    output_cols = ['price_usd', 'time_since_last_trade', 'sellerFee_amount', 'is_circular', 'was_holding_previously']
    output_data = df[output_cols].copy()

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    output_data.to_csv(OUTPUT_FILE, index=False)

    total_saved = len(output_data)
    print("[*] Processed {} rows".format(total_saved))

    # Free RAM before training
    del df
    del output_data
    gc.collect()
    print("[OK] Done! Saved to {}. RAM freed.".format(OUTPUT_FILE))

except Exception as e:
    print("[!] ERROR:", str(e))
    import traceback
    traceback.print_exc()
