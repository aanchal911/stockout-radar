"""
Second ingest source: a live inventory snapshot (as of 'now'), simulating a
feed from the store inventory management system. This gets joined against
the POS sales history to compute days-of-supply.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(11)

def generate_inventory(n_stores=200, n_skus=1000, seed=11):
    rng = np.random.default_rng(seed)
    store_ids = np.arange(1, n_stores + 1)
    sku_ids = np.arange(1, n_skus + 1)

    stores_rep = np.repeat(store_ids, n_skus)
    skus_rep = np.tile(sku_ids, n_stores)
    n = len(stores_rep)

    # Most store-SKUs are reasonably stocked; a meaningful minority are thin
    # (this is what creates real "at risk" rows for the model to surface)
    on_hand = rng.gamma(shape=2.2, scale=9.0, size=n).astype("int32")

    df = pd.DataFrame({
        "store_id": stores_rep,
        "sku_id": skus_rep,
        "on_hand_qty": on_hand,
    })

    # inject a few duplicate rows and a few nulls to simulate a messy real feed
    dup_idx = rng.choice(n, size=int(n * 0.001), replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)
    null_idx = rng.choice(len(df), size=int(len(df) * 0.002), replace=False)
    df.loc[null_idx, "on_hand_qty"] = np.nan

    return df

if __name__ == "__main__":
    import os; os.makedirs("data", exist_ok=True)
    df = generate_inventory()
    df.to_parquet("data/inventory_snapshot.parquet", index=False)
    print(f"Inventory snapshot: {len(df):,} rows, nulls={df['on_hand_qty'].isna().sum()}, "
          f"dupes={df.duplicated(['store_id','sku_id']).sum()}")
