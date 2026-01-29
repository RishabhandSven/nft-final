import pandas as pd
import random
import secrets
import os
from datetime import datetime, timedelta

# --- CONFIGURATION ---
OUTPUT_FILE = "data/nft_transactions_huge.csv"
TOTAL_ROWS = 1_000_000  # Change this to 50_000_000 for a massive file
CHUNK_SIZE = 10_000     # Write to disk every 10k rows to save RAM

# --- SETUP MOCK USERS & CONTRACTS ---
# We reuse these to create realistic trading history patterns
print("[*] Generating Mock Users & Collections...")
USERS = [f"0x{secrets.token_hex(20)}" for _ in range(5000)]  # 5,000 unique traders
CONTRACTS = [f"0x{secrets.token_hex(20)}" for _ in range(50)] # 50 NFT collections

def generate_chunk(num_rows, start_id):
    data = []
    base_time = datetime.now() - timedelta(days=365)
    
    for i in range(num_rows):
        # 1. Random Selection
        buyer = random.choice(USERS)
        seller = random.choice(USERS)
        contract = random.choice(CONTRACTS)
        
        # 2. Inject Wash Trading Logic (25% chance)
        # Circular trade = same buyer and seller (wash trading pattern)
        is_wash = random.random() < 0.25
        if is_wash:
            # Make buyer == seller for circular/wash trade
            seller = buyer  # Force circular pattern
            price = random.uniform(5.0, 50.0) # Suspiciously high price
        else:
            price = random.uniform(0.01, 2.0) # Normal price

        # 3. Build Row
        row = {
            "transactionHash": f"0x{secrets.token_hex(32)}",
            "block_number": 12000000 + i,
            "timestamp": (base_time + timedelta(minutes=i)).isoformat(),
            "contractAddress": contract,
            "from": seller,          # The Seller
            "to": buyer,            # The Buyer
            "tokenId": random.randint(1, 1000),
            "tokenName": "BoredMockApe",
            "tokenSymbol": "BAYC",
            "transactionIndex": i % 100,
            "gas": random.randint(21000, 500000),
            "gasPrice": random.randint(10, 100) * 1e9,
            "gasUsed": random.randint(21000, 150000),
            "cumulativeGasUsed": random.randint(100000, 5000000),
            "input": "0x",
            "nonce": random.randint(0, 5000),
            "blockHash": f"0x{secrets.token_hex(32)}",
            
            # --- CRITICAL COLUMNS FOR YOUR AI ---
            "buyerAddress": buyer,
            "sellerAddress": seller,
            "price_usd": round(price * 3000, 2), # Approx ETH -> USD
            "price_crypto": round(price, 4),
            "token_id": random.randint(1, 10000),
            "sellerFee_amount": round(price * 0.025, 5), # 2.5% Fee
            "market_place": random.choice(["OpenSea", "Blur", "LooksRare"]),
            "filter_1234": "pass" # Dummy column you requested
        }
        data.append(row)
    
    return pd.DataFrame(data)

def main():
    print(f"[*] Starting Generation of {TOTAL_ROWS} rows...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Remove old file
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    
    total_generated = 0
    
    while total_generated < TOTAL_ROWS:
        # Calculate how many to make in this batch
        batch_size = min(CHUNK_SIZE, TOTAL_ROWS - total_generated)
        
        # Generate Data
        df = generate_chunk(batch_size, total_generated)
        
        # Save to CSV (Append Mode)
        # We only write the header for the very first chunk
        write_header = (total_generated == 0)
        df.to_csv(OUTPUT_FILE, mode='a', index=False, header=write_header)
        
        total_generated += batch_size
        print(f"   Saved {total_generated:,} / {TOTAL_ROWS:,} rows...")

    print(f"[OK] DONE! Dataset saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()