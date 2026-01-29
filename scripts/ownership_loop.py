import pandas as pd
import os

# -------------------------------
# Get script directory
# -------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------
# Input & output files
# -------------------------------
INPUT_FILE = os.path.join(
    SCRIPT_DIR, "..", "data", "filtered", "blur_sales_with_holding_time.csv"
)

OUTPUT_FILE = os.path.join(
    SCRIPT_DIR, "..", "data", "filtered", "blur_sales_final.csv"
)

# -------------------------------
# Load data
# -------------------------------
df = pd.read_csv(INPUT_FILE)

# -------------------------------
# Ownership loop (TRUE / FALSE)
# -------------------------------
df['ownership_loop_flag'] = (
    (df['holding_time_hours'] < 1) &
    (df['buyer_24h_trade_count'] > 1) &
    (df['seller_24h_trade_count'] > 1)
)

# -------------------------------
# Save final dataset
# -------------------------------
df.to_csv(OUTPUT_FILE, index=False)

print("✅ Ownership loop flag added successfully!")
print("📁 Final file saved at:", OUTPUT_FILE)
