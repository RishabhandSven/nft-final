'''
    Data Analysis of results
'''

import pandas as pd
from datetime import timedelta

'''
    nft and gox metrics, e.g. ROC AUC, Errors
'''
df = pd.read_csv('data/results/nft_ml_metrics.csv')
print(f"=== Results from NFT Platforms ===")
print(f"Average pred_wash_pct:{round(df['pred_wash_pct'].mean(),2)}%")

# Paths for both datasets
for name, path in [('NFT Platforms', "data/results/nft_ml_metrics.csv"), ('Mt. Gox', "data/results/gox_ml_metrics.csv")]:
    print(f"\n=== Results from {name} ===")
    df = pd.read_csv(path)

    # Grouped performance metrics
    grouped = round(df.groupby(["platform", "model"])[["roc_auc", "abs_error_pct"]].mean(), 3)

    # Compute average runtime in seconds, then convert to h..m..s..
    runtime_sec = df.groupby(["platform", "model"])["time"].mean().round().astype(int)
    runtime_fmt = runtime_sec.apply(lambda x: str(timedelta(seconds=x)).replace("days", "d"))

    # Add formatted runtime
    grouped["avg_runtime"] = runtime_fmt

    # Print results
    print(grouped)
    print(f"Average abs_error_pct: {round(df['abs_error_pct'].mean(), 2)}%")


'''
    gox feature importance
'''
# Load feature importance
df = pd.read_csv("data/results/gox_ml_feature_importance.csv")

# Filter for XGBoost
xgb_df = df[df["model"] == "XGBoost"]

# Drop non-feature columns
feature_cols = [col for col in xgb_df.columns if col not in ["platform", "model", "fold"]]

# Average across folds
avg_importance = xgb_df[feature_cols].mean().sort_values(ascending=False).round(4)

print(f"\n=== Feature Importance (XGB) for Mt. Gox ===")
print(avg_importance)


'''
    all nft feature importance 
'''
# Load the feature importance data
df = pd.read_csv("data/results/nft_ml_feature_importance.csv")

# Identify feature columns (exclude platform, model, fold)
non_feature_cols = ["platform", "model", "fold"]
feature_cols = [col for col in df.columns if col not in non_feature_cols]

# Step 1: Compute overall average importance per feature for ranking
overall_avg = df[feature_cols].mean().sort_values(ascending=False).round(3)
ranked_features = overall_avg.index.tolist()  # Sorted feature names

# Step 2: Group by platform and model, compute mean importance per feature
grouped = df.groupby(["platform", "model"])[feature_cols].mean().round(3)

# Step 3: Reorder rows by ranked_features
grouped = grouped.T.loc[ranked_features]  # Transpose for features as rows

# Step 4: Add rank and feature name as columns
grouped.insert(0, "Feature", grouped.index)
grouped.insert(0, "Rank", range(1, len(grouped) + 1))

# Add final column: average importance across all models and platforms
grouped["Avg"] = grouped.loc[:, grouped.columns[2:]].mean(axis=1).round(3)

# Step 5: Print result
print(f"\n=== Feature Importance for NFT Platforms ===")
print(grouped.to_string(index=False))
# print(grouped.to_latex(index=False, escape=False))


'''
    Check single platform feature importance
'''
# Choose platform
platform = "OpenSea"

# Filter for the platform
df_single = df[df["platform"] == platform]

# Step 1: Compute per-feature average for each model
grouped = df_single.groupby("model")[feature_cols].mean().round(3)

# Step 2: Transpose so features are rows, models are columns
grouped = grouped.T  # Features as rows

# Step 3: Add average across models, rank by this average
grouped["Avg"] = grouped.mean(axis=1).round(3)
grouped = grouped.sort_values("Avg", ascending=False)

# Step 4: Add rank and feature name
grouped.insert(0, "Feature", grouped.index)
grouped.insert(0, "Rank", range(1, len(grouped) + 1))

# Step 5: Print result
print(f"\n=== Feature Importance for {platform} ===")
print(grouped.to_string(index=False))
