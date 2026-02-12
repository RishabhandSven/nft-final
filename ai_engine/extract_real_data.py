#!/usr/bin/env python3
"""
Extract real NFT transaction data from zip and format for processor_simple.py.
Streams data from zip (does not unzip entirely).
"""

import zipfile
import json
import csv
import os
from datetime import datetime, timezone

# --- CONFIG ---
ZIP_PATH = r"C:\Users\ask2r\Downloads\nft_data.zip"
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "nft_transactions_huge.csv")
TARGET_ROWS = 1_500_000
ETH_TO_USD_FALLBACK = 2000  # Used when only raw_value (Wei) exists
SELLER_FEE_RATE = 0.025  # 2.5%

# Columns required by processor_simple.py cols_needed
HEADERS = ['buyerAddress', 'sellerAddress', 'price_usd', 'sellerFee_amount', 'timestamp', 'tokenId']


def _get_price_usd(item):
    """
    Get price in USD. Returns float or None if row should be skipped.
    Logic: price.usd > price with value conversion > skip
    """
    price = item.get('price')
    if price is None:
        return None

    if isinstance(price, (int, float)):
        return float(price)

    usd = price.get('usd')
    if usd is not None:
        return float(usd)

    raw_value = price.get('raw_value')
    if raw_value is not None:
        eth = raw_value / 10**18
        return eth * ETH_TO_USD_FALLBACK

    value = price.get('value')
    if value is not None:
        return float(value) * ETH_TO_USD_FALLBACK

    return None


def _row_from_item(item):
    """Convert one JSON item to a dict with HEADERS keys, or None to skip."""
    price_usd = _get_price_usd(item)
    if price_usd is None:
        return None

    # buyerAddress: receiver.address (or to_address/buyer)
    receiver = item.get('receiver') or {}
    buyer = (
        receiver.get('address')
        or item.get('to_address')
        or item.get('buyer')
    )
    if not buyer and isinstance(receiver, str):
        buyer = receiver

    # sellerAddress: sender.address (or from_address/seller)
    sender = item.get('sender') or {}
    seller = (
        sender.get('address')
        or item.get('from_address')
        or item.get('seller')
    )
    if not seller and isinstance(sender, str):
        seller = sender

    if not buyer or not seller:
        return None

    # timestamp: block_timestamp or time -> ISO format
    ts_raw = item.get('block_timestamp') or item.get('time')
    if ts_raw is None:
        return None
    if isinstance(ts_raw, (int, float)):
        ts = datetime.fromtimestamp(int(ts_raw), tz=timezone.utc).isoformat()
    else:
        ts = str(ts_raw)

    # tokenId: token_id or nft.token_id
    nft = item.get('nft') or {}
    token_id = (
        item.get('token_id')
        or item.get('tokenId')
        or nft.get('token_id')
    )
    if token_id is None:
        return None
    token_id = str(token_id)

    # sellerFee_amount: price_usd * 0.025
    seller_fee = round(price_usd * SELLER_FEE_RATE, 5)

    return {
        'buyerAddress': buyer,
        'sellerAddress': seller,
        'price_usd': round(price_usd, 2),
        'sellerFee_amount': seller_fee,
        'timestamp': ts,
        'tokenId': token_id,
    }


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    sample_rows = []
    total = 0
    skipped = 0

    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        json_names = [n for n in z.namelist() if n.endswith('.json') and not n.endswith('/')]

        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as out:
            writer = csv.DictWriter(out, fieldnames=HEADERS)
            writer.writeheader()

            for name in json_names:
                if total >= TARGET_ROWS:
                    break

                with z.open(name) as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue

                items = data if isinstance(data, list) else [data]
                for item in items:
                    row = _row_from_item(item)
                    if row is None:
                        skipped += 1
                        continue

                    writer.writerow(row)
                    total += 1

                    if len(sample_rows) < 5:
                        sample_rows.append(row.copy())

                    if total >= TARGET_ROWS:
                        break

    # --- Verification: print sample rows ---
    print("=" * 60)
    print("VERIFICATION: Sample rows (columns must match processor_simple.py cols_needed)")
    print("cols_needed = ['buyerAddress', 'sellerAddress', 'price_usd', 'sellerFee_amount', 'timestamp', 'tokenId']")
    print("=" * 60)
    for i, row in enumerate(sample_rows, 1):
        print(f"\n--- Row {i} ---")
        for k, v in row.items():
            print(f"  {k}: {v}")
    print("\n" + "=" * 60)
    print(f"[OK] Extracted {total:,} transactions to {OUTPUT_FILE}")
    if total < TARGET_ROWS:
        print(f"     (Target was {TARGET_ROWS:,}; source had {total:,} valid rows)")
    print(f"     Skipped {skipped:,} rows (missing price/address/tokenId)")
    print("=" * 60)


if __name__ == "__main__":
    main()
