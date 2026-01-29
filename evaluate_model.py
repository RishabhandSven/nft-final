"""
Evaluate IsolationForest model on wash trading detection.
Uses is_circular as ground truth label on 100k sample.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score,
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
import json

print("[*] Loading model and scaler...")
model = joblib.load('data/results/wash_trading_brain.pkl')
scaler = joblib.load('data/results/scaler.pkl')

print("[*] Loading training data (500k rows)...")
df = pd.read_csv('data/training_chunk.csv')

# Sample 100k for evaluation
print("[*] Sampling 100k rows for evaluation...")
eval_df = df.sample(n=min(100000, len(df)), random_state=42)

print(f"[*] Dataset shape: {eval_df.shape}")
print(f"[*] Positive rate (is_circular=1): {eval_df['is_circular'].mean():.4%}")

# Prepare features (currently includes is_circular due to training)
# TODO: Retrain without is_circular to eliminate leakage
features = ['price_usd', 'time_since_last_trade', 'sellerFee_amount', 'is_circular']
X = scaler.transform(eval_df[features])
y_true = eval_df['is_circular'].values

# Get model scores and predictions
print("[*] Computing predictions...")
scores = model.decision_function(X)
predictions = model.predict(X)

# Convert predictions: -1 = anomaly (wash trade), 1 = normal
y_pred = (predictions == -1).astype(int)

# Calculate metrics
print("\n" + "="*60)
print("EVALUATION METRICS (is_circular as ground truth)")
print("="*60)

# ROC AUC (use negative scores so higher = more anomalous)
roc_auc = roc_auc_score(y_true, -scores)
print(f"ROC AUC Score:           {roc_auc:.4f}")

# Average Precision (PR AUC)
try:
    avg_precision = average_precision_score(y_true, -scores)
    print(f"Average Precision (PR):  {avg_precision:.6f}")
except Exception as e:
    print(f"Average Precision:       Error - {e}")

# Precision, Recall, F1
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print(f"Precision:               {precision:.6f}")
print(f"Recall:                  {recall:.6f}")
print(f"F1 Score:                {f1:.6f}")

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print(f"\nConfusion Matrix:")
print(f"  True Negatives:        {tn:,}")
print(f"  False Positives:       {fp:,}")
print(f"  False Negatives:       {fn:,}")
print(f"  True Positives:        {tp:,}")

# Specificity and sensitivity
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = recall  # sensitivity = recall for positive class
print(f"\nSensitivity (Recall):    {sensitivity:.6f}")
print(f"Specificity:             {specificity:.6f}")

print("\n" + "="*60)

# Save metrics to JSON
metrics_dict = {
    "roc_auc": float(roc_auc),
    "average_precision": float(avg_precision) if 'avg_precision' in locals() else None,
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "specificity": float(specificity),
    "sensitivity": float(sensitivity),
    "sample_size": len(eval_df),
    "positive_rate": float(eval_df['is_circular'].mean()),
    "true_positives": int(tp),
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn)
}

with open('evaluation_metrics.json', 'w') as f:
    json.dump(metrics_dict, f, indent=2)

print(f"[OK] Metrics saved to evaluation_metrics.json")
print(f"\nInterpretation:")
print(f"  - ROC AUC > 0.7: Good discrimination")
print(f"  - F1 > 0.5: Balanced precision-recall")
print(f"  - Recall > 0.5: Catching most wash trades")
print(f"  - Precision > 0.5: Few false alarms")
