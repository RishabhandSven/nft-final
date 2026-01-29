import pandas as pd
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# --- SMART PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "training_chunk.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "data", "results", "wash_trading_brain.pkl")
SCALER_PATH = os.path.join(SCRIPT_DIR, "..", "data", "results", "scaler.pkl")

# Ensure output directory exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

def train_model():
    print(f"🧠 Loading Training Data from: {DATA_PATH}")
    
    if not os.path.exists(DATA_PATH):
        print("❌ Error: Training data not found. Run processor.py first!")
        return

    # Load data (filling NaNs with 0 just in case)
    df = pd.read_csv(DATA_PATH).fillna(0)
    
    # Define features (MUST match what api.py expects)
    features = ['price_usd', 'time_since_last_trade', 'sellerFee_amount', 'is_circular']
    
    print(f"📊 Training on {len(df)} rows with features: {features}")
    
    X = df[features]

    # Normalize data (Scale prices so $1M doesn't look like an error compared to 1 second)
    print("⚖️ Scaling Data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Isolation Forest
    # contamination=0.05 -> We assume roughly 5% of trades might be wash trades
    print("🤖 Training Isolation Forest (Finding Anomalies)...")
    model = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    model.fit(X_scaled)

    # Save artifacts
    print(f"💾 Saving model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    print("✅ Success! The Brain is trained and saved.")

if __name__ == "__main__":
    train_model()