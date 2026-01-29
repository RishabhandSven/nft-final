import pandas as pd
from utils.hash_utils import sha256_hex

# Input → Output file mapping
datasets = [
    {
        "input": "data/filtered/blur_sales_filtered.csv",
        "output": "data/filtered/blur_sales_hashed.csv",
        "tx": "transactionHash",
        "buyer": "buyerAddress",
        "seller": "sellerAddress"
    },
    {
        "input": "data/filtered/looksrare_sales_filtered.csv",
        "output": "data/filtered/looksrare_sales_hashed.csv",
        "tx": "transactionHash",
        "buyer": "buyerAddress",
        "seller": "sellerAddress"
    },
    {
        "input": "data/filtered/opensea_sales_filtered.csv",
        "output": "data/filtered/opensea_sales_hashed.csv",
        "tx": "transactionHash",
        "buyer": "buyerAddress",
        "seller": "sellerAddress"
    },
    {
        "input": "data/filtered/gox_sales_labeled.csv",
        "output": "data/filtered/gox_sales_hashed.csv",
        "tx": None,  # GOX has no transactionHash
        "buyer": "buyer",
        "seller": "seller"
    }
]

for ds in datasets:
    print(f"Processing: {ds['input']}")

    df = pd.read_csv(ds["input"])

    # prepare empty dict for storing hashed output fields
    hashed_output = {}

    # hash transaction if exists
    if ds["tx"] and ds["tx"] in df.columns:
        hashed_output[ds["tx"]] = df[ds["tx"]].apply(sha256_hex)

    # hash buyer
    if ds["buyer"] in df.columns:
        hashed_output[ds["buyer"]] = df[ds["buyer"]].apply(sha256_hex)

    # hash seller
    if ds["seller"] in df.columns:
        hashed_output[ds["seller"]] = df[ds["seller"]].apply(sha256_hex)

    # create minimal dataframe with only hashed fields
    hashed_df = pd.DataFrame(hashed_output)

    hashed_df.to_csv(ds["output"], index=False)

    print(f"✔ Hashed fields saved to: {ds['output']}\n")
