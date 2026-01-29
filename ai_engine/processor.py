import pandas as pd
import os
import sys

# Fix encoding for Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- SMART PATH CONFIGURATION ---
# Get the folder where THIS script (processor.py) actually lives
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up one level (..) to 'NFT', then down into 'data'
# This works regardless of whether you run it from 'NFT' or 'ai_engine'
HUGE_FILE = os.path.join(SCRIPT_DIR, "..", "data", "nft_transactions_huge.csv")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "..", "data", "training_chunk.csv")

# Resolve to absolute paths (removes the ..)
HUGE_FILE = os.path.abspath(HUGE_FILE)
OUTPUT_FILE = os.path.abspath(OUTPUT_FILE)

# Columns to keep
REQUIRED_COLS = [
    'transactionHash', 'buyerAddress', 'sellerAddress', 
    'sellerFee_amount', 'timestamp', 'tokenId', 'price_usd' 
]

def process_huge_dataset():
    # Debug print to show exactly where we are looking
    print(f"[*] Looking for dataset at: {HUGE_FILE}")

    if not os.path.exists(HUGE_FILE):
        print(f"[!] Error: File NOT found.")
        return

    print(f"[+] Found it! Streaming 1,000,000 rows...")
    
    # Delete old training file if exists
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    chunk_size = 100000 
    total_saved = 0
    target_rows = 500000 

    # Stream the file
    for chunk in pd.read_csv(HUGE_FILE, usecols=REQUIRED_COLS, chunksize=chunk_size):
        
        # 1. Basic Cleaning
        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
        
        # 2. Sort to find patterns
        chunk = chunk.sort_values(['tokenId', 'timestamp'])
        
        # 3. Feature Engineering
        # Calculate time gap
        chunk['time_since_last_trade'] = chunk.groupby('tokenId')['timestamp'].diff().dt.total_seconds().fillna(3600*24)
        
        # Flag circular trades (Buyer == Seller)
        chunk['is_circular'] = (chunk['buyerAddress'] == chunk['sellerAddress']).astype(int)
        
        # 4. Save small training file
        ai_data = chunk[['price_usd', 'time_since_last_trade', 'sellerFee_amount', 'is_circular']]
        
        ai_data.to_csv(OUTPUT_FILE, mode='a', index=False, header=(total_saved==0))
        
        total_saved += len(ai_data)
        print(f"   Processed {total_saved} rows...")
        
        if total_saved >= target_rows:
            print("[*] Collected enough data. Stopping.")
            break

    print(f"[OK] Success! Training chunk saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_huge_dataset()