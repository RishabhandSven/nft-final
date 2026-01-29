import pandas as pd
import os

# -------------------------------
# Get script directory
# -------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------
# File paths
# -------------------------------
INPUT_FILE = os.path.join(
    SCRIPT_DIR, "..", "data", "ml_sample", "Blur_ml_samples.csv"
)

OUTPUT_FILE = os.path.join(
    SCRIPT_DIR, "..", "data", "filtered", "blur_sales_with_holding_time.csv"
)

# -------------------------------
# Load CSV
# -------------------------------
df = pd.read_csv(INPUT_FILE)

# -------------------------------
# Holding time (already present as 'hours')
# -------------------------------
df['holding_time_hours'] = df['hours']
df['holding_time_days'] = df['holding_time_hours'] / 24

# -------------------------------
# 🚨 Wash trade flag (NEW)
# -------------------------------
df['wash_trade_flag'] = df['holding_time_hours'] < 1

# -------------------------------
# Save updated CSV
# -------------------------------
df.to_csv(OUTPUT_FILE, index=False)

print("✅ Holding time + wash trade flag added successfully!")
print("📁 Output file:", OUTPUT_FILE)
