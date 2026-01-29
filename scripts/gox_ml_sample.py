"""
    step 1 (1/2) of Gox ML task: constructing the samples
"""

import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant
from pandarallel import pandarallel
from tqdm import tqdm

pandarallel.initialize(progress_bar=False)

"""mt gox dataset init"""
path = "data/filtered/gox_sales_labeled.csv"
dtype_dict = {
    "seller": str,
    "buyer": str,
    "Trade_Id": str,
    "bitcoins": float,
    "price_USD": float,
    "time": str,
    "wash": bool
}

usecols_list = list(dtype_dict.keys())
opt_threshold = 0.02  # For log-log_regression; Cumu_Wash feature


"""helper function"""
# Computed round level of trade size a la Cong
def round_level(n):
    if n == 0:
        return 0
    ns = str(n)

    if 'e' in ns:
        ns = ns.split('e')[0]

    if '.' in ns:
        ns = ns.rstrip('0').replace('.', '')

    # find last non-zero index
    nz = [i for i, j in enumerate(ns) if j != '0']
    if not nz:
        return 0

    last_nonzero = max(nz)
    last_digits = ns[last_nonzero:]
    try:
        return int(last_digits) / float(ns)
    except:
        return 0


# Regression using optimal threshold
def log_log_regression_opt(df):
    df['rounded'] = df['round_level'] >= opt_threshold

    non_wash_df = df[df['wash'] == False]
    weekly_data = non_wash_df.groupby(['week', 'rounded'])['bitcoins'].sum().unstack(fill_value=0)

    # enforce consistent names
    weekly_data.columns = ['Unrounded', 'Rounded']

    valid = weekly_data[(weekly_data['Rounded'] > 0) & (weekly_data['Unrounded'] > 0)]
    if valid.empty:
        tqdm.write("No valid data for regression.")
        return 0, 1  # identity fallback => minimal bias

    X = np.log(valid['Rounded'].values)
    y = np.log(valid['Unrounded'].values)

    model = OLS(y, add_constant(X)).fit()
    c, b = model.params[0], model.params[1]
    return c, b


def prepare_data(sales_path):
    df = pd.read_csv(sales_path, dtype=dtype_dict, usecols=usecols_list,
                     engine='python', encoding="utf-8")

    df['time'] = pd.to_datetime(df['time'])
    df['week'] = df['time'].dt.to_period('W')
    df['day'] = df['time'].dt.to_period('D')
    df['hours'] = df['time'].dt.hour

    # keep only complete weeks
    complete_weeks = df.groupby('week').filter(
        lambda g: len(g['time'].dt.dayofweek.unique()) == 7
    )['week'].unique()
    df = df[df['week'].isin(complete_weeks)].copy()

    df = df.sort_values('time').reset_index(drop=True)

    # ---- FIXED ROUND LEVEL ----
    df['round_level'] = df['bitcoins'].apply(round_level)
    df['round_level'] = pd.to_numeric(df['round_level'], errors='coerce').fillna(0)

    tqdm.write('round_level computed.')

    # ---- REGRESSION & cumu wash calc ----
    c, b = log_log_regression_opt(df)
    tqdm.write(f"constant_p, beta_p: {(c, b)}")

    df['rounded_opt'] = df['round_level'] >= opt_threshold
    df = df.sort_values('time').reset_index(drop=True)
    df['cumu_sum'] = df['bitcoins'].cumsum()

    df['round_vol_opt'] = (df['bitcoins'] * df['rounded_opt']).fillna(0)
    df['unround_vol_opt'] = (df['bitcoins'] * ~df['rounded_opt']).fillna(0)

    df['cumu_rounded_sum_opt'] = df['round_vol_opt'].cumsum()
    df['cumu_unrounded_sum_opt'] = df['unround_vol_opt'].cumsum()

    # log only valid rows
    mask = df['cumu_rounded_sum_opt'] > 0
    log_vals = np.zeros(len(df))
    log_vals[mask] = np.log(df.loc[mask, 'cumu_rounded_sum_opt'])

    df['predicted_unrounded_vol_opt'] = np.exp(c + b * log_vals)
    df['cumu_wash_value_opt'] = df['cumu_unrounded_sum_opt'] - df['predicted_unrounded_vol_opt']

    df['cumu_wash_percent_opt'] = np.where(
        df['cumu_sum'] > 0,
        df['cumu_wash_value_opt'] / df['cumu_sum'],
        0
    )

    # ---- TRADE COUNT FEATURES ----
    df = df.sort_values(['buyer', 'time'])
    daily = df.groupby(['buyer', 'day'])['time'].count().reset_index(name='dc')
    daily['cum'] = daily.groupby('buyer')['dc'].cumsum()
    cmap = daily.set_index(['buyer', 'day'])['cum'].to_dict()

    def past(row, d):
        return cmap.get((row['buyer'], row['day']), 0) - cmap.get((row['buyer'], row['day'] - pd.Timedelta(days=d)), 0)

    df['buyer_24h_trade_count'] = df.apply(lambda r: past(r, 1), axis=1)
    df['buyer_7d_trade_count'] = df.apply(lambda r: past(r, 7), axis=1)

    df = df.sort_values(['seller', 'time'])
    daily = df.groupby(['seller', 'day'])['time'].count().reset_index(name='dc')
    daily['cum'] = daily.groupby('seller')['dc'].cumsum()
    cmap = daily.set_index(['seller', 'day'])['cum'].to_dict()

    def past_s(row, d):
        return cmap.get((row['seller'], row['day']), 0) - cmap.get((row['seller'], row['day'] - pd.Timedelta(days=d)), 0)

    df['seller_24h_trade_count'] = df.apply(lambda r: past_s(r, 1), axis=1)
    df['seller_7d_trade_count'] = df.apply(lambda r: past_s(r, 7), axis=1)

    # ---- BTC SUM FEATURES ----
    df = df.sort_values(['buyer', 'time'])
    btc = df.groupby(['buyer', 'day'])['bitcoins'].sum().reset_index(name='btc')
    btc['cum'] = btc.groupby('buyer')['btc'].cumsum()
    cmap = btc.set_index(['buyer', 'day'])['cum'].to_dict()

    def past_b(row, d):
        return cmap.get((row['buyer'], row['day']), 0) - cmap.get((row['buyer'], row['day'] - pd.Timedelta(days=d)), 0)

    df['buyer_24h_btc_sum'] = df.apply(lambda r: past_b(r, 1), axis=1)
    df['buyer_7d_btc_sum'] = df.apply(lambda r: past_b(r, 7), axis=1)

    df = df.sort_values(['seller', 'time'])
    btc = df.groupby(['seller', 'day'])['bitcoins'].sum().reset_index(name='btc')
    btc['cum'] = btc.groupby('seller')['btc'].cumsum()
    cmap = btc.set_index(['seller', 'day'])['cum'].to_dict()

    def past_s_b(row, d):
        return cmap.get((row['seller'], row['day']), 0) - cmap.get((row['seller'], row['day'] - pd.Timedelta(days=d)), 0)

    df['seller_24h_btc_sum'] = df.apply(lambda r: past_s_b(r, 1), axis=1)
    df['seller_7d_btc_sum'] = df.apply(lambda r: past_s_b(r, 7), axis=1)

    df = df.sort_values('time')
    df['buyer_all_btc_sum'] = df.groupby('buyer')['bitcoins'].cumsum()
    df['seller_all_btc_sum'] = df.groupby('seller')['bitcoins'].cumsum()

    df = df.sort_values('time')
    df['price_deviation'] = df['price_USD'].diff().fillna(0)

    df = df.sort_values('time')
    df['time_since_last_trade'] = df['time'].diff().dt.total_seconds().fillna(0)

    features = [
        'bitcoins',
        'round_level',
        'cumu_wash_percent_opt',
        'buyer_24h_trade_count',
        'buyer_7d_trade_count',
        'seller_24h_trade_count',
        'seller_7d_trade_count',
        'buyer_24h_btc_sum',
        'buyer_7d_btc_sum',
        'seller_24h_btc_sum',
        'seller_7d_btc_sum',
        'buyer_all_btc_sum',
        'seller_all_btc_sum',
        'price_deviation',
        'time_since_last_trade',
        'hours',
        'wash'
    ]

    df[features].to_csv("data/ml_sample/gox_ml_samples.csv", index=False)
    tqdm.write("Saved ML dataset → data/ml_sample/gox_ml_samples.csv")

    nan = df[features].isnull().any(axis=1).sum()
    tqdm.write(f"Rows with NaN: {nan} ({nan / len(df):.2%})")


prepare_data(path)
