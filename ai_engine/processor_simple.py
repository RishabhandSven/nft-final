#!/usr/bin/env python3
import pandas as pd
import os

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
HUGE_FILE = os.path.join(DATA_DIR, "nft_transactions_huge.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "training_chunk.csv")

print("[*] Processor starting...")
print("[*] Input file:", HUGE_FILE)
print("[*] Output file:", OUTPUT_FILE)

if not os.path.exists(HUGE_FILE):
    print("[!] ERROR: Input file not found!")
    exit(1)

print("[+] Reading and processing data...")

try:
    # Read only the columns we need
    cols_needed = ['buyerAddress', 'sellerAddress', 'price_usd', 'sellerFee_amount', 'timestamp', 'tokenId']
    
    chunk_size = 50000
    total_saved = 0
    target_rows = 100000
    
    # Delete old output file
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    
    for i, chunk in enumerate(pd.read_csv(HUGE_FILE, usecols=cols_needed, chunksize=chunk_size)):
        # Basic processing
        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
        chunk = chunk.sort_values(['tokenId', 'timestamp'])
        
        # Feature engineering
        chunk['time_since_last_trade'] = chunk.groupby('tokenId')['timestamp'].diff().dt.total_seconds().fillna(3600)
        chunk['is_circular'] = (chunk['buyerAddress'] == chunk['sellerAddress']).astype(int)
        
        # Keep only what we need
        output_data = chunk[['price_usd', 'time_since_last_trade', 'sellerFee_amount', 'is_circular']]
        
        # Append to CSV
        output_data.to_csv(OUTPUT_FILE, mode='a', index=False, header=(total_saved == 0))
        
        total_saved += len(output_data)
        print("[*] Processed {} rows...".format(total_saved))
        
        if total_saved >= target_rows:
            print("[*] Target rows reached. Stopping.")
            break
    
    print("[OK] Done! Saved {} rows to {}".format(total_saved, OUTPUT_FILE))
    
except Exception as e:
    print("[!] ERROR:", str(e))
    import traceback
    traceback.print_exc()
