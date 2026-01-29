"""
    Step 2 (2/2) of NFT ML task: classification
        runtime: 18:48:51
"""

import pandas as pd
from xgboost import XGBClassifier
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from pprint import pformat
import time
import gc
import os


feature_dtype_dict = {
    'sellerFee_amount': float,
    'round_level': float,
    'cumu_wash_percent_opt': float,
    'buyer_24h_trade_count': float,
    'buyer_7d_trade_count': float,
    'seller_24h_trade_count': float,
    'seller_7d_trade_count': float,
    'buyer_24h_nfttrade_count': float,
    'buyer_7d_nfttrade_count': float,
    'seller_24h_nfttrade_count': float,
    'seller_7d_nfttrade_count': float,
    'buyer_nft_all_trade_count': float,
    'seller_nft_all_trade_count': float,
    'price_deviation': float,
    'time_since_last_trade': float,
    'hours': float,
    'filter_1234': bool  # labels
}

label = 'filter_1234'
features = [k for k in feature_dtype_dict if k != label]
dtype_dict = {k: feature_dtype_dict[k] for k in features + [label]}
usecols_list = features + [label]


marketplaces = [
    ('Blur', 'data/ml_sample/Blur_ml_samples.csv'),
    ('LooksRare', 'data/ml_sample/LooksRare_ml_samples.csv'),
    ('OpenSea', 'data/ml_sample/OpenSea_ml_samples.csv')
]

results_path = "data/results/nft_ml_metrics.csv"
fi_path = "data/results/nft_ml_feature_importance.csv"

# If the file exists, delete it to start fresh
for path in [results_path, fi_path]:
    if os.path.exists(path):
        os.remove(path)


# Helper functions
def optimize_threshold(y_prob, y_true, X_test, num_thresholds=100):
    """
    Finds the optimal decision threshold that minimizes seller-fee-weighted error
    between predicted and true wash trade percentages.

    Parameters:
    - y_prob: np.array, predicted probabilities for the positive class.
    - y_true: np.array, ground-truth binary labels (1 = wash trade).
    - X_test: pd.DataFrame, must include "sellerFee_amount" column.
    - num_thresholds: int, number of threshold values to evaluate.

    Returns:
    - best_threshold: float, threshold that minimizes wash volume percentage error.
    - wash_volume_stats: dict with true and predicted wash % and absolute error.
    """

    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    Y_prob_matrix = np.tile(y_prob[:, np.newaxis], (1, len(thresholds)))
    Y_pred_matrix = (Y_prob_matrix > thresholds).astype(int)

    seller_fee = X_test["sellerFee_amount"]
    total_volume = seller_fee.sum()

    true_wash_pct = seller_fee[y_true == 1].sum() / total_volume * 100
    pred_wash_pcts = (seller_fee.values @ Y_pred_matrix) / total_volume * 100
    abs_errors = np.abs(pred_wash_pcts - true_wash_pct)

    best_idx = np.argmin(abs_errors)
    best_thresh = thresholds[best_idx]

    wash_volume_stats = {
        'true_wash_pct': true_wash_pct,
        'pred_wash_pct': pred_wash_pcts[best_idx],
        'abs_error_pct': abs_errors[best_idx]
    }

    return best_thresh, wash_volume_stats


def run_random_forest(X_train, y_train, X_test, y_test, feature_names):
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize Random Forest
    model = RandomForestClassifier(
        random_state=42,
        n_estimators=100,
        max_depth=5,
        max_features='sqrt',
        n_jobs=-1,
        class_weight='balanced'
    )

    # Train model
    model.fit(X_train_scaled, y_train)

    # Predict probabilities
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Optimize threshold
    best_thresh, wash_volume_stats = optimize_threshold(y_prob, y_test, X_test)

    # Evaluate
    auc = roc_auc_score(y_test, y_prob)

    result = {
        "roc_auc": auc,
        "best_threshold": best_thresh,
        "true_wash_pct": wash_volume_stats['true_wash_pct'],
        "pred_wash_pct": wash_volume_stats['pred_wash_pct'],
        "abs_error_pct": wash_volume_stats['abs_error_pct'],
        "feature_importance": pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        })
    }

    return result


def run_xgboost_rf(X_train, y_train, X_test, y_test, feature_names):
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Compute scale_pos_weight for XGBoost; majority / minority = non_wash / wash
    weight = (y_train == 0).sum() / (y_train == 1).sum()

    # Initialize XGBoost model
    model = XGBClassifier(
        tree_method='hist',
        device='cuda',
        n_jobs=-1,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        alpha=0,
        reg_lambda=1,
        random_state=42,
        scale_pos_weight=weight,
        early_stopping_rounds=5000,
        eval_metric='auc'
    )

    # Train the model
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=True
    )

    # Make predictions
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    # Optimize threshold and compute wash stats
    best_thresh, wash_volume_stats = optimize_threshold(y_prob, y_test, X_test)

    # Evaluate the model
    result = {
        "roc_auc": auc,
        "best_threshold": best_thresh,
        "true_wash_pct": wash_volume_stats['true_wash_pct'],
        "pred_wash_pct": wash_volume_stats['pred_wash_pct'],
        "abs_error_pct": wash_volume_stats['abs_error_pct'],
        "feature_importance": pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        })
    }
    return result


def run_tabnet(X_train, y_train, X_test, y_test, feature_names, n_layer=256, use_gpu=True):
    # Normalize features
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)

    # Check if GPU is available
    device = torch.device(
        "cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    # Initialize TabNet model
    model = TabNetClassifier(
        device_name=device,
        n_d=n_layer,  # Width of decision layer
        n_a=n_layer,  # Width of attention layer
        n_steps=1,  # Number of steps in decision process
        gamma=1.4,  # Scaling factor for sparse regularization
        lambda_sparse=1e-4,  # L1 regularization on attention
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size": 10, "gamma": 0.92},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type="sparsemax",
        seed=42,
        verbose=1,
        momentum=0.2
    )

    # Train TabNet
    model.fit(
        X_train_normalized, y_train,
        eval_set=[(X_test_normalized, y_test)],
        eval_metric=['auc'],
        patience=10,  # changed
        max_epochs=10,  # changed
        batch_size=32768,
        virtual_batch_size=8000,
        num_workers=0  # changed
    )

    # Make predictions
    y_prob = model.predict_proba(X_test_normalized)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    # Optimize decision threshold
    best_thresh, wash_volume_stats = optimize_threshold(y_prob, y_test, X_test)

    # Evaluate the model
    result = {
        "roc_auc": auc,
        "best_threshold": best_thresh,
        "true_wash_pct": wash_volume_stats['true_wash_pct'],
        "pred_wash_pct": wash_volume_stats['pred_wash_pct'],
        "abs_error_pct": wash_volume_stats['abs_error_pct'],
        "feature_importance": pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        })
    }
    return result


def main():
    model_lists = [
        "Random Forest",
        "XGBoost",
        "TabNet"
    ]

    for name, sales_path in tqdm(marketplaces, desc="Going through markets"):
        tqdm.write(f"\nCurrently processing {name}")
        tqdm.write("**************************")

        # Load data with specified dtypes and columns
        df = pd.read_csv(sales_path, dtype=dtype_dict,
                         usecols=usecols_list, engine="pyarrow")
        df = df.sample(frac=0.2, random_state=42)

        # Check and report NaN stats before dropping
        nan_rows_count = df[features].isnull().any(axis=1).sum()
        fraction_nan = nan_rows_count / len(df)
        tqdm.write(
            f"Number of rows with NaN in features: {nan_rows_count} ({fraction_nan:.2%})")
        tqdm.write(
            f"Remaining rows after drop: {len(df) - nan_rows_count} ({1 - fraction_nan:.2%})")

        # Drop rows with NaNs in any feature
        df = df.dropna(subset=features)

        # Cache this clean version so we don't re-load for each model
        df_full = df

        for model_name in model_lists:
            tqdm.write(f"\n>>> Model: {model_name}")

            # Handle OpenSea + TabNet data trimming
            if name == "OpenSea" and model_name == "TabNet":
                trim_len = int(np.ceil(0.05 * len(df_full)))
                df = df_full.iloc[:trim_len].copy()
                tqdm.write(
                    f"Trimmed OpenSea data to first 5% ({trim_len} rows) for TabNet")
            else:
                df = df_full

            # Define X and y
            X = df[features]
            y = df[label]
            feature_names = features

            # Stratified K-Fold
            kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

            for fold_idx, (train_idx, test_idx) in enumerate(tqdm(kfold.split(X, y), desc=f"Folds: {model_name}", leave=False), 1):
                tqdm.write(f"\n--- Fold {fold_idx} ---")

                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

                # Assign model_func based on model_name
                if model_name == "Random Forest":
                    model_func = run_random_forest
                elif model_name == "XGBoost":
                    model_func = run_xgboost_rf
                elif model_name == "TabNet":
                    model_func = lambda *args: run_tabnet(
                        *args, n_layer=128 if name == "OpenSea" else 256)

                # === Pre-cleanup ===
                if model_name in ["TabNet", "XGBoost"]:
                    torch.cuda.empty_cache()
                    gc.collect()

                # Run and time the model
                start_time = time.time()
                result = model_func(X_train, y_train, X_test,
                                    y_test, feature_names)
                duration = time.time() - start_time
                tqdm.write(f"Elapsed time: {duration:.2f} seconds")

                # Save performance summary
                summary = {
                    "platform": name,
                    "model": model_name,
                    "fold": fold_idx,
                    "time": duration,
                    "roc_auc": result["roc_auc"],
                    "best_threshold": result["best_threshold"],
                    "true_wash_pct": result["true_wash_pct"],
                    "pred_wash_pct": result["pred_wash_pct"],
                    "abs_error_pct": result["abs_error_pct"],
                }
                write_header_results = not os.path.exists(results_path)
                pd.DataFrame([summary]).to_csv(results_path, mode='a',
                                               index=False, header=write_header_results)

                tqdm.write(pformat(summary))

                # Save feature importance
                fi_row = result["feature_importance"].set_index(
                    "Feature")["Importance"].to_dict()
                fi_row.update({
                    "platform": name,
                    "model": model_name,
                    "fold": fold_idx
                })
                write_header_fi = not os.path.exists(fi_path)
                pd.DataFrame([fi_row]).to_csv(fi_path, mode='a',
                                              index=False, header=write_header_fi)

                # === Post-cleanup ===
                if model_name in ["TabNet", "XGBoost"]:
                    del result
                    gc.collect()
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
