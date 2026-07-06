"""
Synthetic data generator: regional grocery chain POS + inventory feed.

Simulates what would actually land in Cloud Storage each night as Parquet:
one row per (store, SKU, day) with units sold and on-hand inventory.

Scale knobs are deliberately modest here (runs on a laptop-class CPU in this
sandbox); the comments show how this maps to full chain scale.
"""
import numpy as np
import pandas as pd
import time

RNG = np.random.default_rng(7)

N_STORES = 200          # full chain: 500 stores
N_SKUS = 1000            # full chain: 5,000 active SKUs
N_DAYS = 45              # rolling 45-day POS history window

def generate(n_stores=N_STORES, n_skus=N_SKUS, n_days=N_DAYS, seed=7):
    rng = np.random.default_rng(seed)
    store_ids = np.arange(1, n_stores + 1)
    sku_ids = np.arange(1, n_skus + 1)
    days = pd.date_range("2026-05-21", periods=n_days, freq="D")

    # Cartesian product store x sku x day
    stores_rep = np.repeat(store_ids, n_skus * n_days)
    skus_rep = np.tile(np.repeat(sku_ids, n_days), n_stores)
    days_rep = np.tile(days.values, n_stores * n_skus)

    n = len(stores_rep)

    # Per-SKU base demand (some SKUs are fast movers, most are slow)
    sku_base_demand = rng.gamma(shape=2.0, scale=3.0, size=n_skus)  # mean ~6 units/day
    sku_lead_time = rng.integers(1, 6, size=n_skus)  # supplier lead time, days
    sku_perishable = rng.random(n_skus) < 0.25

    base_demand_rep = np.tile(np.repeat(sku_base_demand, n_days), n_stores)
    lead_time_rep = np.tile(np.repeat(sku_lead_time, n_days), n_stores)
    perishable_rep = np.tile(np.repeat(sku_perishable, n_days), n_stores)

    # Store size multiplier (bigger stores sell more of everything)
    store_mult = rng.uniform(0.6, 1.8, size=n_stores)
    store_mult_rep = np.repeat(store_mult, n_skus * n_days)

    # Weekday seasonality (weekend lift)
    dow = pd.DatetimeIndex(days_rep).dayofweek
    weekend_lift = np.where(dow >= 5, 1.35, 1.0)

    # A subset of SKUs are trending up sharply in the last 2 weeks (viral / promo)
    trend_flag_sku = rng.random(n_skus) < 0.05
    trend_flag_rep = np.tile(np.repeat(trend_flag_sku, n_days), n_stores)
    day_idx = np.tile(np.tile(np.arange(n_days), n_skus), n_stores)
    trend_boost = np.where(
        (trend_flag_rep) & (day_idx > n_days - 14),
        1.0 + (day_idx - (n_days - 14)) * 0.12,
        1.0,
    )

    lam = base_demand_rep * store_mult_rep * weekend_lift * trend_boost
    lam = np.clip(lam, 0.05, None)
    units_sold = rng.poisson(lam)

    df = pd.DataFrame({
        "store_id": stores_rep,
        "sku_id": skus_rep,
        "date": days_rep,
        "units_sold": units_sold.astype("int32"),
        "lead_time_days": lead_time_rep.astype("int8"),
        "perishable": perishable_rep,
    })

    # On-hand inventory: simulate as starting stock minus cumulative sales plus periodic restocks
    df = df.sort_values(["store_id", "sku_id", "date"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    t0 = time.time()
    df = generate()
    t1 = time.time()
    print(f"Generated {len(df):,} rows in {t1 - t0:.2f}s")
    import os; os.makedirs("data", exist_ok=True)
    df.to_parquet("data/pos_feed.parquet", index=False)
    print("Saved pos_feed.parquet:", df.memory_usage(deep=True).sum() / 1e6, "MB in memory")
    print(df.head())
