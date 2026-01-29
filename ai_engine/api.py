from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

app = FastAPI()

# Allow local frontends (dev) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. TRAIN ON STARTUP (Fast & Dirty) ---
# In a real app, you'd load a .pkl file, but for this hackathon approach,
# we retrain on the small chunk every time we restart server.
print("[*] Training Model on Startup...")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "data", "results", "wash_trading_brain.pkl")
SCALER_PATH = os.path.join(SCRIPT_DIR, "..", "data", "results", "scaler.pkl")
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "training_chunk.csv")

# Prefer loading a pre-trained brain (fast). Fall back to training on the CSV if not present.
model = None
scaler = None
MODEL_PATH_ABS = os.path.abspath(MODEL_PATH)
SCALER_PATH_ABS = os.path.abspath(SCALER_PATH)
print("[*] Looking for brain at:", MODEL_PATH_ABS)
print("[*] Model exists?", os.path.exists(MODEL_PATH_ABS))
print("[*] Scaler exists?", os.path.exists(SCALER_PATH_ABS))
if os.path.exists(MODEL_PATH_ABS) and os.path.exists(SCALER_PATH_ABS):
    try:
        model = joblib.load(MODEL_PATH_ABS)
        scaler = joblib.load(SCALER_PATH_ABS)
        print("[OK] Brain Loaded Successfully")
    except Exception as e:
        print("[!] Failed to load brain:", str(e))
        import traceback
        traceback.print_exc()

if model is None:
    if os.path.exists(DATA_PATH):
        df = pd.read_csv("data/training_chunk.csv").fillna(0)
        # Use the columns produced by the processor: price_usd, time_since_last_trade, sellerFee_amount, is_circular
        feature_cols = ['price_usd', 'time_since_last_trade', 'sellerFee_amount', 'is_circular']
        # Fallback if older CSV uses different column names
        for alt in (['price', 'time_since_last_trade', 'sellerFee_amount'], feature_cols):
            if all(c in df.columns for c in alt):
                X = df[alt]
                break
        else:
            print("⚠️ Training CSV missing required columns. Running in Mock Mode.")
            X = None

        if X is not None:
            # If scaler exists load it; otherwise train a fresh model without scaling
            if os.path.exists(SCALER_PATH):
                try:
                    scaler = joblib.load(SCALER_PATH)
                except Exception:
                    scaler = None

            if scaler is not None:
                X_scaled = scaler.transform(X)
                model = IsolationForest(contamination=0.05)
                model.fit(X_scaled)
            else:
                model = IsolationForest(contamination=0.05)
                model.fit(X)
            print("[OK] Model Ready.")
    else:
        print("[!] No Data Found. Running in Mock Mode.")
        model = None

class Transaction(BaseModel):
    price: float
    time_since_last_trade: float
    sellerFee_amount: float


class AnalyzeRequest(BaseModel):
    price_usd: float
    time_since_last_trade: float
    sellerFee_amount: float
    buyer_address: str
    seller_address: str

@app.post("/detect_wash_trade")
def analyze(tx: Transaction):
    if model is None: return {"error": "Model not trained"}

    features = np.array([[tx.price, tx.time_since_last_trade, tx.sellerFee_amount]])
    # If a scaler is available, we need to expand to a 4th feature (is_circular=0) for compatibility
    if scaler is not None:
        # Append is_circular=0 when not provided
        features4 = np.hstack([features, np.array([[0.0]])])
        features_scaled = scaler.transform(features4)
        prediction = model.predict(features_scaled)[0]
        confidence = model.decision_function(features_scaled)[0]
    else:
        prediction = model.predict(features)[0]
        confidence = model.decision_function(features)[0]

    return {
        "is_wash_trade": bool(prediction == -1),
        "risk_score": float(confidence),
        "status": "SUSPICIOUS" if prediction == -1 else "SAFE"
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "scaler_loaded": scaler is not None}


@app.post("/analyze")
def analyze_full(req: AnalyzeRequest):
    """Analyze with full feature set including buyer/seller to compute `is_circular`."""
    print("[DEBUG] In /analyze endpoint. model is None?", model is None, "scaler is None?", scaler is None)
    if model is None:
        return {"error": "Model not loaded"}

    is_circular = 1 if req.buyer_address == req.seller_address else 0
    features = np.array([[req.price_usd, req.time_since_last_trade, req.sellerFee_amount, is_circular]])

    if scaler is not None:
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        confidence = model.decision_function(features_scaled)[0]
    else:
        prediction = model.predict(features)[0]
        confidence = model.decision_function(features)[0]

    return {
        "is_wash_trade": bool(prediction == -1),
        "risk_score": float(confidence),
        "status": "HIGH RISK" if prediction == -1 else "SAFE"
    }