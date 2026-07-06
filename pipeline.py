"""
Stockout Radar — core pipeline.

Stages (each timed, mirroring what would be separate Spark/BigQuery/GPU steps
in production):
  1. Ingest      - read POS history + inventory snapshot (simulating Cloud Storage / BigQuery reads)
  2. Clean       - dedupe, null handling, dtype normalization
  3. Feature eng - rolling sales velocity (7/14/28d) per store-SKU, trend ratio
  4. Join        - attach live on-hand inventory snapshot
  5. Score       - compute stockout risk score per store-SKU
  6. Rank        - surface the top-N urgent rows + transfer recommendations

This same code, unmodified, is the code path that runs under cudf.pandas on
a GPU (see README / run_gpu.sh). We time it here on CPU/pandas as the
baseline.
"""
import time
import numpy as np
import pandas as pd
import json

TIMINGS = {}


def stage(name):
    def deco(fn):
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            TIMINGS[name] = time.perf_counter() - t0
            print(f"[{name}] {TIMINGS[name]:.3f}s")
            return result
        return wrapper
    return deco


@stage("1_ingest")
def ingest():
    pos = pd.read_parquet("data/pos_feed.parquet")
    inv = pd.read_parquet("data/inventory_snapshot.parquet")
    return pos, inv


@stage("2_clean")
def clean(pos, inv):
    pos = pos.drop_duplicates(subset=["store_id", "sku_id", "date"])
    pos["date"] = pd.to_datetime(pos["date"])
    pos["units_sold"] = pos["units_sold"].clip(lower=0)

    inv = inv.drop_duplicates(subset=["store_id", "sku_id"])
    inv["on_hand_qty"] = inv["on_hand_qty"].fillna(inv["on_hand_qty"].median())
    return pos, inv


@stage("3_feature_engineering")
def engineer_features(pos):
    pos = pos.sort_values(["store_id", "sku_id", "date"])
    g = pos.groupby(["store_id", "sku_id"], sort=False)["units_sold"]

    pos["velocity_7"] = g.transform(lambda s: s.rolling(7, min_periods=3).mean())
    pos["velocity_14"] = g.transform(lambda s: s.rolling(14, min_periods=5).mean())
    pos["velocity_28"] = g.transform(lambda s: s.rolling(28, min_periods=10).mean())

    # latest snapshot per store-sku = most recent day's rolling features
    latest = pos.groupby(["store_id", "sku_id"], sort=False).tail(1).copy()
    latest["trend_ratio"] = (latest["velocity_7"] / latest["velocity_28"].replace(0, np.nan)).fillna(1.0)
    return latest


@stage("4_join_inventory")
def join_inventory(latest, inv):
    merged = latest.merge(inv, on=["store_id", "sku_id"], how="left")
    merged["on_hand_qty"] = merged["on_hand_qty"].fillna(0)
    return merged


@stage("5_score_risk")
def score_risk(merged):
    df = merged.copy()
    safe_velocity = df["velocity_7"].clip(lower=0.05)
    df["days_of_supply"] = df["on_hand_qty"] / safe_velocity

    # Coverage relative to how long a reorder actually takes to arrive
    df["coverage_ratio"] = df["days_of_supply"] / df["lead_time_days"].clip(lower=1)

    # Risk score: higher = more urgent.
    # - thin coverage_ratio -> big driver (inverted, capped)
    # - trend_ratio > 1 (accelerating demand) amplifies risk
    # - perishable items get a modest urgency bump (spoilage + can't over-order)
    coverage_component = np.clip(2.5 - df["coverage_ratio"], 0, 2.5) / 2.5   # 0..1
    trend_component = np.clip(df["trend_ratio"] - 1.0, 0, 2.0) / 2.0         # 0..1
    perishable_bump = np.where(df["perishable"], 0.10, 0.0)

    df["risk_score"] = (
        0.65 * coverage_component
        + 0.25 * trend_component
        + perishable_bump
    ).clip(0, 1)

    df["risk_tier"] = pd.cut(
        df["risk_score"], bins=[-0.01, 0.35, 0.65, 1.0],
        labels=["ok", "watch", "urgent"]
    )
    return df


@stage("6_rank_and_recommend")
def rank_and_recommend(df, top_n=25):
    ranked = df.sort_values("risk_score", ascending=False)

    urgent = ranked[ranked["risk_tier"] == "urgent"].copy()

    # For each urgent SKU, find the best transfer-source store: same SKU,
    # healthy coverage, plenty of on-hand relative to its own velocity.
    overstock = df[(df["coverage_ratio"] > 4) & (df["on_hand_qty"] > 15)]
    overstock_by_sku = (
        overstock.sort_values("on_hand_qty", ascending=False)
        .groupby("sku_id")
        .first()[["store_id", "on_hand_qty"]]
        .rename(columns={"store_id": "transfer_from_store", "on_hand_qty": "source_on_hand"})
    )

    urgent = urgent.merge(overstock_by_sku, on="sku_id", how="left")
    top = urgent.head(top_n)[[
        "store_id", "sku_id", "risk_score", "days_of_supply", "coverage_ratio",
        "trend_ratio", "on_hand_qty", "velocity_7", "lead_time_days", "perishable",
        "transfer_from_store", "source_on_hand",
    ]]
    return ranked, top


def run():
    total_t0 = time.perf_counter()
    pos, inv = ingest()
    pos, inv = clean(pos, inv)
    latest = engineer_features(pos)
    merged = join_inventory(latest, inv)
    scored = score_risk(merged)
    ranked, top = rank_and_recommend(scored)
    TIMINGS["total"] = time.perf_counter() - total_t0

    print("\n=== Risk tier breakdown ===")
    print(scored["risk_tier"].value_counts())

    print(f"\nTotal store-SKU combinations scored: {len(scored):,}")
    print(f"Total pipeline wall time: {TIMINGS['total']:.2f}s")

    import os; os.makedirs("data", exist_ok=True)
    top.to_csv("data/top_urgent.csv", index=False)
    scored.to_parquet("data/scored_full.parquet", index=False)
    with open("data/timings_pandas.json", "w") as f:
        json.dump(TIMINGS, f, indent=2)

    return scored, top


if __name__ == "__main__":
    scored, top = run()
    print("\n=== Top 10 urgent restock/transfer candidates ===")
    print(top.head(10).to_string(index=False))
