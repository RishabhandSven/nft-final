"""
Evaluate NFT Wash Trading Model.
Works with supervised (is_circular labels) and unsupervised (real data with no labels) datasets.
"""
import pandas as pd
import joblib
import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "training_chunk.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "data", "results", "wash_trading_brain.pkl")
SCALER_PATH = os.path.join(SCRIPT_DIR, "..", "data", "results", "scaler.pkl")
METRICS_PATH = os.path.join(SCRIPT_DIR, "..", "data", "results", "evaluation_metrics.json")

feature_cols = ['price_usd', 'time_since_last_trade', 'sellerFee_amount', 'is_circular', 'was_holding_previously']

df = pd.read_csv(DATA_PATH).fillna(0)
for col in feature_cols:
    if col not in df.columns:
        df[col] = 0
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

X = scaler.transform(df[feature_cols])
scores = model.decision_function(X)
predictions = model.predict(X)

n_total = len(df)
n_flagged = (predictions == -1).sum()
flag_rate = n_flagged / n_total * 100

metrics = {
    "sample_size": n_total,
    "flagged_as_high_risk": int(n_flagged),
    "flag_rate_pct": round(flag_rate, 2),
    "contamination_target": 0.01,
    "decision_score_min": float(scores.min()),
    "decision_score_max": float(scores.max()),
    "decision_score_mean": float(scores.mean()),
}

# Stats on flagged vs safe transactions
flagged_mask = predictions == -1
if n_flagged > 0:
    metrics["flagged_price_usd_mean"] = float(df.loc[flagged_mask, "price_usd"].mean())
    metrics["flagged_price_usd_median"] = float(df.loc[flagged_mask, "price_usd"].median())
    metrics["safe_price_usd_mean"] = float(df.loc[~flagged_mask, "price_usd"].mean())
    metrics["safe_price_usd_median"] = float(df.loc[~flagged_mask, "price_usd"].median())

# Supervised metrics (only if we have circular trades as ground truth)
n_circular = (df["is_circular"] == 1).sum()
if n_circular > 0 and n_circular < n_total:
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
    y_true = df["is_circular"].values
    y_pred = (predictions == -1).astype(int)
    metrics["roc_auc"] = float(roc_auc_score(y_true, -scores))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["supervised_eval"] = True
else:
    metrics["supervised_eval"] = False
    metrics["note"] = "No circular trades in dataset; cannot compute precision/recall/F1"

# Save
os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

# Print
print("=" * 50)
print("NFT WASH TRADING MODEL - EVALUATION METRICS")
print("=" * 50)
print(f"Sample size:          {metrics['sample_size']:,}")
print(f"Flagged as high risk: {metrics['flagged_as_high_risk']:,} ({metrics['flag_rate_pct']}%)")
print(f"Decision score range: [{metrics['decision_score_min']:.4f}, {metrics['decision_score_max']:.4f}]")
if n_flagged > 0:
    print(f"Flagged trades avg price:  ${metrics.get('flagged_price_usd_mean', 0):,.2f}")
    print(f"Safe trades avg price:     ${metrics.get('safe_price_usd_mean', 0):,.2f}")
if metrics.get("supervised_eval"):
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
else:
    print(metrics.get("note", ""))
print("=" * 50)
print(f"Metrics saved to: {METRICS_PATH}")
