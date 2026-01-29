import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import joblib

# Load full training data
df = pd.read_csv('data/training_chunk.csv')

# Split 70/30
train, test = train_test_split(df, test_size=0.3, random_state=42, stratify=df['is_circular'])

# Load model trained on train set
model = joblib.load('data/results/wash_trading_brain.pkl')
scaler = joblib.load('data/results/scaler.pkl')

# Evaluate on TEST set (unseen data)
X_test = scaler.transform(test[['price_usd', 'time_since_last_trade', 'sellerFee_amount']])
y_test = test['is_circular'].values

scores = model.decision_function(X_test)
predictions = model.predict(X_test)

print(f"ROC AUC: {roc_auc_score(y_test, -scores):.4f}")
print(f"Precision: {precision_score(y_test, predictions == -1):.4f}")
print(f"Recall: {recall_score(y_test, predictions == -1):.4f}")
print(f"F1: {f1_score(y_test, predictions == -1):.4f}")