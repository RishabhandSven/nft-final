import hashlib
import pandas as pd


def sha256_hex(value: str) -> str:
    if pd.isna(value):
        return value
    return hashlib.sha256(value.encode('utf-8')).hexdigest()
