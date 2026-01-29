"""
step 1 (1/2) of NFT ML task: constructing the samples
"""

import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant
from pandarallel import pandarallel
from tqdm import tqdm
import sys
import os

# Fields required in raw CSVs
required_columns = [
    "transactionHash",
    "contractAddress",
    "buyerAddress",
    "sellerAddress",
    "sellerFee_amount",
    "filter_1234",
    "timestamp",
    "tokenId"
]

# Optional dtype enforcement
dtype_dict = {
    "transactionHash": str,
    "contractAddress": str,
    "buyerAddress": str,
    "sellerAddress": str,
    "sellerFee_amount": float,
    "filter_1234": bool,
    "timestamp": str,
    "tokenId": str
}

marketplaces = [
    ('Blur', 'data/filtered/blur_sales_filtered.csv'),
    ('LooksRare', 'data/filtered/looksrare_sales_filtered.csv'),
    ('OpenSea', 'data/filtered/opensea_sales_filtered.csv')
]

# --------------------------
# Helper Functions
# --------------------------


def round_level(n):
    if pd.isna(n) or n == 0:
        return 0
    ns = str(n)
    if 'e' in ns:
        ns = ns.split('e')[0]
    if '.' in ns:
        ns = ns.rstrip('0').replace('.', '')
    last_nonzero = max([i for i, j in enumerate(ns) if j != '0'])
    last_digits = ns[last_nonzero:]
    return int(last_digits) / int(float(ns))


def log_log_regression_opt(df, name):
    opt_threshold = {
        'Blur': 0.09091818181818183,
        'LooksRare': 0.08183636363636364,
        'OpenSea': 0.002118181818181818
    }

    df['rounded'] = df['round_level'] >= opt_threshold.get(name, 0.05)

    non_wash_df = df[df['filter_1234'] == False]

    weekly_data = non_wash_df.groupby(['week', 'rounded'])['sellerFee_amount'] \
        .sum().unstack(fill_value=0)
    if weekly_data.shape[1] < 2:
        return 0, 0

    weekly_data.columns = ['Unrounded', 'Rounded']
    valid = weekly_data[(weekly_data['Rounded'] > 0) &
                        (weekly_data['Unrounded'] > 0)]
    if valid.empty:
        return 0, 0

    X = np.log(valid['Rounded'].values)
    y = np.log(valid['Unrounded'].values)
    Xc = add_constant(X)
    model = OLS(y, Xc).fit()
    return model.params[0], model.params[1]

# --------------------------
# Core Processing
# --------------------------


def prepare_data(name, sales_path):

    df = pd.read_csv(sales_path, encoding="utf-8-sig", skipinitialspace=True)
    df.columns = df.columns.str.replace('\ufeff', '').str.strip()

    # Ensure required fields exist
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: Missing columns {missing} in {sales_path}")

    df = df[required_columns].copy()

    # Safe dtype conversions
    for col, dtype in dtype_dict.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except:
                pass

    # Timestamp normalize
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)

    df['week'] = df['timestamp'].dt.to_period('W')
    df['day'] = df['timestamp'].dt.to_period('D')
    df['hours'] = df['timestamp'].dt.hour

    week_groups = df.groupby('week')
    complete_weeks = week_groups.filter(
        lambda g: len(g['timestamp'].dt.dayofweek.unique()) == 7
    )['week'].unique()
    df = df[df['week'].isin(complete_weeks)].copy()

    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # Compute round level (no parallel during early testing)
    df['round_level'] = df['sellerFee_amount'].apply(round_level)

    # -------------- features --------------

    # Buyer 24h/7d counts
    df = df.sort_values(by=['buyerAddress', 'timestamp'])
    daily = df.groupby(['buyerAddress', 'day'])[
        'timestamp'].count().reset_index(name='daily_count')
    daily['cumulative_count'] = daily.groupby(
        'buyerAddress')['daily_count'].cumsum()
    cumulative = daily.set_index(['buyerAddress', 'day'])[
        'cumulative_count'].to_dict()

    def past(row, d):
        curr = cumulative.get((row['buyerAddress'], row['day']), 0)
        past = cumulative.get(
            (row['buyerAddress'], row['day'] - pd.Timedelta(days=d)), 0)
        return curr - past

    df['buyer_24h_trade_count'] = df.apply(lambda r: past(r, 1), axis=1)
    df['buyer_7d_trade_count'] = df.apply(lambda r: past(r, 7), axis=1)

    # Seller 24h/7d
    df = df.sort_values(by=['sellerAddress', 'timestamp'])
    daily = df.groupby(['sellerAddress', 'day'])[
        'timestamp'].count().reset_index(name='daily_count')
    daily['cumulative_count'] = daily.groupby(
        'sellerAddress')['daily_count'].cumsum()
    cumulative = daily.set_index(['sellerAddress', 'day'])[
        'cumulative_count'].to_dict()

    def past_s(row, d):
        curr = cumulative.get((row['sellerAddress'], row['day']), 0)
        past = cumulative.get(
            (row['sellerAddress'], row['day'] - pd.Timedelta(days=d)), 0)
        return curr - past

    df['seller_24h_trade_count'] = df.apply(lambda r: past_s(r, 1), axis=1)
    df['seller_7d_trade_count'] = df.apply(lambda r: past_s(r, 7), axis=1)

    # NFT buyer/seller trades
    df['buyer_nft_all_trade_count'] = df.groupby(
        ['contractAddress', 'buyerAddress']).cumcount() + 1
    df['seller_nft_all_trade_count'] = df.groupby(
        ['contractAddress', 'sellerAddress']).cumcount() + 1

    # Price deviation
    df = df.sort_values(by=['contractAddress', 'tokenId', 'timestamp'])
    grp = df.groupby(['contractAddress', 'tokenId'])
    df['last_price'] = grp['sellerFee_amount'].shift(1)
    df['last_time'] = grp['timestamp'].shift(1)
    df['price_deviation'] = (df['sellerFee_amount'] -
                             df['last_price']).fillna(0)
    df['time_since_last_trade'] = (
        df['timestamp'] - df['last_time']).dt.total_seconds().fillna(0)

    # Regression wash values
    constant_p, beta_p = log_log_regression_opt(df, name)

    opt_threshold = {'Blur': 0.0909, 'LooksRare': 0.0818, 'OpenSea': 0.0021}
    df['rounded_opt'] = df['round_level'] >= opt_threshold.get(name, 0.05)
    df['round_vol_opt'] = df['sellerFee_amount'] * df['rounded_opt']
    df['unround_vol_opt'] = df['sellerFee_amount'] * (~df['rounded_opt'])

    grp = df.groupby('contractAddress', group_keys=False)
    df['cumu_nft_sum'] = grp['sellerFee_amount'].cumsum()
    df['cumu_rounded_sum_opt'] = grp['round_vol_opt'].cumsum()
    df['cumu_unrounded_sum_opt'] = grp['unround_vol_opt'].cumsum()

    valid = df['cumu_rounded_sum_opt'] > 0
    logs = np.zeros_like(df['cumu_rounded_sum_opt'])
    logs[valid] = np.log(df.loc[valid, 'cumu_rounded_sum_opt'])
    df['predicted_unrounded_vol_opt'] = np.exp(constant_p + beta_p * logs)
    df['cumu_wash_value_opt'] = df['cumu_unrounded_sum_opt'] - \
        df['predicted_unrounded_vol_opt']
    df['cumu_wash_percent_opt'] = np.where(df['cumu_nft_sum'] > 0,
                                           df['cumu_wash_value_opt'] / df['cumu_nft_sum'], 0)

    features = [
        'sellerFee_amount', 'round_level', 'cumu_wash_percent_opt',
        'buyer_24h_trade_count', 'buyer_7d_trade_count',
        'seller_24h_trade_count', 'seller_7d_trade_count',
        'buyer_nft_all_trade_count', 'seller_nft_all_trade_count',
        'price_deviation', 'time_since_last_trade', 'hours', 'filter_1234'
    ]

    os.makedirs("data/ml_sample", exist_ok=True)
    df[features].to_csv(f"data/ml_sample/{name}_ml_samples.csv", index=False)

    return df


# --------------------------
# Windows-safe entry point
# --------------------------
if __name__ == '__main__':
    pandarallel.initialize(progress_bar=False)
    for name, path in tqdm(marketplaces, desc="Constructing ML Samples"):
        tqdm.write(f"Processing {name}...")
        prepare_data(name, path)
