import pandas as pd
import hashlib
import matplotlib.pyplot as plt
import networkx as nx
import os

# Create results folder if not exists
os.makedirs("results_visuals", exist_ok=True)


def hash_address(addr: str) -> str:
    if pd.isna(addr):
        return "NULL"
    return hashlib.sha256(addr.encode()).hexdigest()[:12]


df = pd.read_csv("data/filtered/gox_sales_hashed.csv")

df['buyer_hash'] = df['buyer'].apply(hash_address)
df['seller_hash'] = df['seller'].apply(hash_address)

# ----- 1. Top Sellers -----
top_sellers = df['seller_hash'].value_counts().head(15)
plt.figure(figsize=(10, 4))
top_sellers.plot(kind='bar')
plt.title("Top Sellers (SHA-256 Anonymized)")
plt.xlabel("Seller Hash")
plt.ylabel("Transactions Count")
plt.tight_layout()
plt.savefig("results_visuals/top_sellers.png", dpi=300)
plt.close()
print("Saved: results_visuals/top_sellers.png")

# ----- 2. Wash Ratio (If label exists) -----
label_col = None
for c in ['wash', 'label', 'is_wash']:
    if c in df.columns:
        label_col = c
        break

if label_col:
    wash_ratio = df.groupby('seller_hash')[label_col].mean().head(15)
    plt.figure(figsize=(10, 4))
    wash_ratio.plot(kind='bar')
    plt.title("Wash Ratio per Seller (Anonymized)")
    plt.xlabel("Seller Hash")
    plt.ylabel("Wash Ratio")
    plt.tight_layout()
    plt.savefig("results_visuals/wash_ratio.png", dpi=300)
    plt.close()
    print("Saved: results_visuals/wash_ratio.png")
else:
    print("[INFO] Wash label not found, skipping wash graph")


# ========== 3. Buyer-Seller Weighted Graph (Fixed!) ==========
# Build weighted full graph


G = nx.DiGraph()

# Count weight per (seller,buyer)
edge_weights = (
    df.groupby(['seller_hash', 'buyer_hash'])
    .size()
    .reset_index(name='weight')
)

for _, r in edge_weights.iterrows():
    G.add_edge(r['seller_hash'], r['buyer_hash'], weight=r['weight'])

# --- FIX HERE ---
# Include buyers of the top sellers
buyers_of_top_sellers = set(
    df[df['seller_hash'].isin(top_sellers.index)]['buyer_hash']
)

sub_nodes = set(top_sellers.index) | buyers_of_top_sellers
subgraph = G.subgraph(sub_nodes)

# Draw
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(subgraph, k=0.4, seed=42)

weights = nx.get_edge_attributes(subgraph, 'weight')
nx.draw(subgraph, pos, node_size=900, with_labels=True, font_size=7)
nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=weights, font_size=6)

plt.title("Buyer ↔ Seller Network (Top Sellers + Their Buyers)")
plt.tight_layout()
plt.savefig("results_visuals/buyer_seller_graph.png", dpi=300)
plt.close()

print("Saved fixed graph.")
