#!/usr/bin/env python3
"""
Production-grade NFT Wash Trading Detection Training Pipeline.

Handles ~73GB CSV/Parquet with streaming, reservoir sampling,
graph features (NetworkX), and Isolation Forest.
"""

import gc
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import networkx as nx
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm


DATA_PATH = Path(r"C:\Users\ask2r\Downloads\archive")
TARGET_SAMPLE_SIZE = 1_500_000
CHUNK_READ_LIMIT_BYTES = 10 * 1024**3  # 10 GB for diversity sampling
CHUNK_SIZE = 100_000

# Schema mapping: our names -> possible source column names
COLUMN_ALIASES = {
    "from_address": ["from_address", "from", "seller_address", "sellerAddress", "sender"],
    "to_address": ["to_address", "to", "buyer_address", "buyerAddress", "receiver"],
    "token_id": ["token_id", "tokenId", "nft_token_id"],
    "price": ["price", "value", "price_usd", "value_usd", "amount", "transaction_value"],
    "timestamp": ["timestamp", "time", "block_timestamp", "created_at"],
    "transaction_hash": ["transaction_hash", "tx_hash", "hash", "id"],
}


def _resolve_column(df: pd.DataFrame, aliases: list[str]) -> Optional[str]:
    """Find first matching column name (case-insensitive)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in cols_lower:
            return cols_lower[a.lower()]
    return None


def _extract_nested(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    """If column contains dicts with 'address', extract address."""
    s = df[col]
    if s.dtype == object and len(s) > 0:
        sample = s.dropna().iloc[0]
        if isinstance(sample, dict) and "address" in sample:
            return s.apply(lambda x: x.get("address") if isinstance(x, dict) else x)
    return None


def _map_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Map source columns to canonical names. Handles nested receiver/sender."""
    result = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        found = _resolve_column(df, aliases)
        if found:
            if canonical in ("from_address", "to_address"):
                nested = _extract_nested(df, found)
                result[canonical] = nested if nested is not None else df[found]
            else:
                result[canonical] = df[found]
    return pd.DataFrame(result)


class WashTradingTrainer:
    """Production-grade trainer for NFT wash trading detection."""

    def __init__(
        self,
        data_path: Path = DATA_PATH,
        target_sample_size: int = TARGET_SAMPLE_SIZE,
        chunk_size: int = CHUNK_SIZE,
        contamination: float = 0.05,
        n_estimators: int = 100,
        n_jobs: int = -1,
    ):
        self.data_path = Path(data_path)
        self.target_sample_size = target_sample_size
        self.chunk_size = chunk_size
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.raw_sample: Optional[pd.DataFrame] = None
        self.processed_df: Optional[pd.DataFrame] = None
        self.model = None
        self.scaler = None
        self.predictions: Optional[np.ndarray] = None

    SKIP_NAMES = ("processed_sample", "wash_trading_", "scaler_")

    def _discover_data_file(self) -> Optional[tuple[Path, str]]:
        """Find first CSV, Parquet, or SQLite file. Returns (path, type). Skips output artifacts."""
        path = self.data_path
        if path.is_file():
            suf = path.suffix.lower()
            if suf in (".csv", ".parquet", ".sqlite", ".db"):
                return (path, suf)
        if path.is_dir():
            for ext in ("*.csv", "*.sqlite", "*.db", "*.parquet"):
                for f in path.glob(ext):
                    if not any(skip in f.stem for skip in self.SKIP_NAMES):
                        return (f, f.suffix.lower())
        return None

    def _load_from_sqlite(self, db_path: Path) -> pd.DataFrame:
        """Load and sample from SQLite database. For large DBs, consider exporting to CSV first."""
        conn = sqlite3.connect(str(db_path))
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'", conn
        )["name"].tolist()
        # Prefer 'transfers' or 'mints' (have from_address, to_address, price, etc.)
        table = next((t for t in tables if t in ("transfers", "mints")), tables[0] if tables else None)
        if not table:
            conn.close()
            raise ValueError("No tables in SQLite database.")
        count = pd.read_sql_query(f"SELECT COUNT(*) as n FROM {table}", conn)["n"].iloc[0]
        conn.close()
        if count > self.target_sample_size * 2:
            print(f"[!] SQLite has {count:,} rows. Sampling via OFFSET (may be slow)...")
        conn = sqlite3.connect(str(db_path))
        if count <= self.target_sample_size:
            full = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            self.raw_sample = full
        else:
            # Random sample: use ORDER BY RANDOM() LIMIT (slow but correct for <10M rows)
            if count < 10_000_000:
                full = pd.read_sql_query(
                    f"SELECT * FROM {table} ORDER BY RANDOM() LIMIT {self.target_sample_size}",
                    conn,
                )
                self.raw_sample = full
            else:
                # Fallback: first N rows (for very large DBs, export to CSV recommended)
                full = pd.read_sql_query(
                    f"SELECT * FROM {table} LIMIT {self.target_sample_size}",
                    conn,
                )
                self.raw_sample = full
                print("[!] Using first N rows. For random sample, export to CSV and re-run.")
        conn.close()
        print(f"[*] Sampled {len(self.raw_sample):,} rows from SQLite")
        return self.raw_sample

    def load_and_sample_data(self) -> pd.DataFrame:
        """
        Stream data and produce a 1.5M row subset.
        Supports CSV (streaming), Parquet, and SQLite.
        """
        discovered = self._discover_data_file()
        if discovered is None:
            raise FileNotFoundError(
                f"No CSV, Parquet, or SQLite file found in {self.data_path}. "
                f"Expected a massive CSV/Parquet (~73GB) or SQLite in this directory."
            )
        data_file, file_type = discovered

        print(f"[*] Data file: {data_file} (type: {file_type})")
        print(f"[*] Target sample size: {self.target_sample_size:,}")
        print(f"[*] Chunk size: {self.chunk_size:,}")

        if file_type in (".sqlite", ".db"):
            return self._load_from_sqlite(data_file)

        reservoir: list[pd.DataFrame] = []
        total_rows_seen = 0
        bytes_read = 0
        is_parquet = file_type == ".parquet"

        if is_parquet:
            # Parquet: read in row groups or use pyarrow for streaming
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(data_file)
                df_full = table.to_pandas()
                total_rows = len(df_full)
                if total_rows <= self.target_sample_size:
                    indices = np.arange(total_rows)
                else:
                    indices = np.random.default_rng(42).choice(
                        total_rows, size=self.target_sample_size, replace=False
                    )
                self.raw_sample = df_full.iloc[indices].copy()
                print(f"[*] Sampled {len(self.raw_sample):,} rows from Parquet")
                return self.raw_sample
            except Exception as e:
                print(f"[!] Parquet read failed: {e}. Trying chunked read...")
                # Fallback: some parquet libs support predicate pushdown
                self.raw_sample = pd.read_parquet(data_file).sample(
                    n=min(self.target_sample_size, len(pd.read_parquet(data_file))),
                    random_state=42,
                )
                return self.raw_sample

        # CSV: stream with reservoir sampling (first 1.5M or random from first 10GB)
        reader = pd.read_csv(
            data_file,
            chunksize=self.chunk_size,
            low_memory=False,
            on_bad_lines="skip",
        )
        rng = np.random.default_rng(42)
        reservoir_rows: list[pd.DataFrame] = []
        total_rows_seen = 0

        for chunk in tqdm(reader, desc="Loading chunks"):
            total_rows_seen += len(chunk)
            bytes_read += chunk.memory_usage(deep=True).sum()

            if self.raw_sample is None:
                reservoir_rows.append(chunk)
                if sum(len(r) for r in reservoir_rows) >= self.target_sample_size:
                    combined = pd.concat(reservoir_rows, ignore_index=True)
                    if len(combined) > self.target_sample_size:
                        self.raw_sample = combined.sample(
                            n=self.target_sample_size, random_state=42
                        ).copy()
                    else:
                        self.raw_sample = combined.copy()
                    reservoir_rows = []
                    del combined
                    gc.collect()
            else:
                # Reservoir sampling: for each new row, maybe replace one in sample
                for i in range(len(chunk)):
                    total_rows_seen += 1
                    j = rng.integers(0, total_rows_seen)
                    if j < self.target_sample_size:
                        idx = rng.integers(0, len(self.raw_sample))
                        self.raw_sample.iloc[idx] = chunk.iloc[i]

            if bytes_read >= CHUNK_READ_LIMIT_BYTES and self.raw_sample is not None:
                print(f"[*] Read ~10GB. Stopping for diversity.")
                break

        if self.raw_sample is None and reservoir_rows:
            combined = pd.concat(reservoir_rows, ignore_index=True)
            if len(combined) > self.target_sample_size:
                self.raw_sample = combined.sample(
                    n=self.target_sample_size, random_state=42
                ).copy()
            else:
                self.raw_sample = combined.copy()

        if self.raw_sample is None:
            raise ValueError("No data could be loaded.")

        print(f"[*] Sample size: {len(self.raw_sample):,}")
        return self.raw_sample

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize schema."""
        df = _map_schema(df)
        required = ["from_address", "to_address", "token_id", "price", "timestamp"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns after mapping: {missing}")

        df = df.dropna(subset=["price", "token_id"])
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df.dropna(subset=["price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df["from_address"] = df["from_address"].astype(str).str.strip().str.lower()
        df["to_address"] = df["to_address"].astype(str).str.strip().str.lower()
        df["token_id"] = df["token_id"].astype(str)
        return df.reset_index(drop=True)

    def engineer_features(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Build graph (NetworkX) and behavioral features.
        """
        df = df if df is not None else self.raw_sample
        if df is None:
            raise RuntimeError("Run load_and_sample_data() first.")

        df = self._preprocess(df.copy())
        print("[*] Building graph and features...")

        # --- Graph: MultiDiGraph ---
        G = nx.MultiDiGraph()
        edges = list(
            zip(
                df["from_address"].values,
                df["to_address"].values,
                df["token_id"].values,
            )
        )
        G.add_edges_from(
            (u, v, {"token_id": t}) for u, v, t in tqdm(edges, desc="Graph edges")
        )

        # Degree centralities
        in_deg = dict(G.in_degree())
        out_deg = dict(G.out_degree())
        try:
            deg_cent = nx.degree_centrality(G)
        except Exception:
            deg_cent = {n: 0.0 for n in G.nodes()}

        # Simple 2-cycles (A->B->A) per wallet
        cycle_count = {n: 0 for n in G.nodes()}
        for u in tqdm(G.nodes(), desc="Cycle count", total=G.number_of_nodes()):
            for v in G.successors(u):
                if G.has_edge(v, u):
                    cycle_count[u] += 1

        # Map back to rows (each row: from_address= seller, to_address=buyer)
        df["in_degree"] = df["from_address"].map(in_deg).fillna(0).astype(np.int32)
        df["out_degree"] = df["from_address"].map(out_deg).fillna(0).astype(np.int32)
        df["degree_centrality"] = df["from_address"].map(deg_cent).fillna(0).astype(np.float32)
        df["cycle_count"] = df["from_address"].map(cycle_count).fillna(0).astype(np.int32)

        del G, in_deg, out_deg, deg_cent, cycle_count
        gc.collect()

        # --- Behavioral: Avg_Price_Token ---
        df["date"] = df["timestamp"].dt.date
        token_daily_avg = (
            df.groupby(["token_id", "date"])["price"]
            .transform("mean")
            .astype(np.float32)
        )
        df["avg_price_token"] = token_daily_avg
        df["price_vs_avg"] = (df["price"] / token_daily_avg.replace(0, np.nan)).fillna(1).astype(np.float32)

        # --- Time_Delta: time between buy and sell for same asset ---
        df = df.sort_values(["token_id", "timestamp"]).reset_index(drop=True)
        df["time_delta"] = (
            df.groupby("token_id")["timestamp"].diff().dt.total_seconds().fillna(3600).astype(np.float32)
        )

        # --- Zero_Spread_Flag: Buy Price == Sell Price for same token (approx) ---
        token_prices = df.groupby("token_id")["price"].transform(lambda x: x.shift(1))
        df["prev_price"] = token_prices.fillna(df["price"])
        df["zero_spread_flag"] = (np.abs(df["price"] - df["prev_price"]) < 1e-6).astype(np.int8)
        df = df.drop(columns=["prev_price", "date"], errors="ignore")

        self.processed_df = df
        gc.collect()
        print(f"[*] Feature matrix: {df.shape}")
        return self.processed_df

    def train_model(self, df: Optional[pd.DataFrame] = None) -> "WashTradingTrainer":
        """Train Isolation Forest on engineered features."""
        df = df if df is not None else self.processed_df
        if df is None:
            raise RuntimeError("Run engineer_features() first.")

        feature_cols = [
            "price", "in_degree", "out_degree", "degree_centrality", "cycle_count",
            "price_vs_avg", "time_delta", "zero_spread_flag",
        ]
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        X = df[feature_cols].fillna(0).astype(np.float32)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            n_jobs=self.n_jobs,
            random_state=42,
        )
        self.model.fit(X_scaled)
        self.predictions = self.model.predict(X_scaled)

        n_anomalies = (self.predictions == -1).sum()
        print(f"\n[*] Anomalies detected: {n_anomalies:,} ({100 * n_anomalies / len(df):.2f}%)")
        return self

    def save_artifacts(self, output_dir: Optional[Path] = None) -> None:
        """Save model and processed sample."""
        output_dir = Path(output_dir) if output_dir else self.data_path
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / "wash_trading_iso_forest.pkl"
        parquet_path = output_dir / "processed_sample.parquet"
        scaler_path = output_dir / "scaler_iso_forest.pkl"

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"[*] Model saved: {model_path}")
        print(f"[*] Scaler saved: {scaler_path}")

        if self.processed_df is not None:
            self.processed_df.to_parquet(parquet_path, index=False)
            print(f"[*] Processed sample saved: {parquet_path}")

    def plot_anomalies(self, output_path: Optional[Path] = None) -> None:
        """Scatter: Price vs Time, anomalies in red."""
        if self.processed_df is None or self.predictions is None:
            raise RuntimeError("Run train_model() first.")

        df = self.processed_df.copy()
        df["is_anomaly"] = self.predictions == -1

        # Use numeric time for x-axis
        df["time_num"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds()

        fig, ax = plt.subplots(figsize=(12, 8))
        normal = df[~df["is_anomaly"]]
        anomalies = df[df["is_anomaly"]]
        ax.scatter(normal["time_num"], normal["price"], c="steelblue", alpha=0.4, s=5, label="Normal")
        ax.scatter(anomalies["time_num"], anomalies["price"], c="red", alpha=0.7, s=10, label="Anomaly")
        ax.set_xlabel("Time (seconds from start)")
        ax.set_ylabel("Price (USD)")
        ax.set_title("NFT Wash Trading Detection: Price vs Time")
        ax.legend()
        ax.set_yscale("log")
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150)
            print(f"[*] Plot saved: {output_path}")
        try:
            plt.show()
        except Exception:
            plt.close()

    def run(self, output_dir: Optional[Path] = None) -> "WashTradingTrainer":
        """Full pipeline: load, engineer, train, save, plot."""
        output_dir = Path(output_dir) if output_dir else self.data_path
        self.load_and_sample_data()
        self.engineer_features()
        self.train_model()
        self.save_artifacts(output_dir)
        self.plot_anomalies(output_dir / "anomaly_scatter.png")
        return self


def main():
    trainer = WashTradingTrainer(
        data_path=DATA_PATH,
        target_sample_size=TARGET_SAMPLE_SIZE,
        contamination=0.05,
        n_estimators=100,
        n_jobs=-1,
    )
    trainer.run()


if __name__ == "__main__":
    main()

