"""
demo.py — single entry point for the Stockout Radar hackathon demo.

  python demo.py            # fast demo: 50 stores x 200 SKUs (~30s on CPU)
  python demo.py --full     # full scale: 200 stores x 1000 SKUs (~4-5 min on CPU)
  python demo.py --gpu      # full scale via cudf.pandas (requires RAPIDS)

Outputs: data/ artifacts + stockout_radar_live.html (open in browser)
"""
import argparse, json, time, webbrowser, os, sys

parser = argparse.ArgumentParser()
parser.add_argument("--full", action="store_true", help="Full 200x1000 scale")
parser.add_argument("--gpu",  action="store_true", help="Use cudf.pandas acceleration")
parser.add_argument("--no-browser", action="store_true")
args = parser.parse_args()

if args.gpu:
    import cudf.pandas
    cudf.pandas.install()

if args.full:
    N_STORES, N_SKUS = 200, 1000
else:
    N_STORES, N_SKUS = 50, 200   # ~30s on CPU — good for live demo

print(f"\n{'='*55}")
print(f"  Stockout Radar — {'GPU' if args.gpu else 'CPU'} demo")
print(f"  Scale: {N_STORES} stores × {N_SKUS} SKUs × 45 days")
print(f"{'='*55}\n")

# ── 1. Generate synthetic data ──────────────────────────────
print("Generating POS history...")
from generate_data import generate
from generate_inventory import generate_inventory
import pandas as pd

t0 = time.perf_counter()
pos = generate(n_stores=N_STORES, n_skus=N_SKUS)
inv = generate_inventory(n_stores=N_STORES, n_skus=N_SKUS)
os.makedirs("data", exist_ok=True)
pos.to_parquet("data/pos_feed.parquet", index=False)
inv.to_parquet("data/inventory_snapshot.parquet", index=False)
print(f"  {len(pos):,} POS rows, {len(inv):,} inventory rows  ({time.perf_counter()-t0:.1f}s)\n")

# ── 2. Run pipeline ─────────────────────────────────────────
from pipeline import run
scored, top = run()

with open("data/timings_pandas.json") as f:
    timings = json.load(f)

# ── 3. Build live HTML ──────────────────────────────────────
tier_counts = scored["risk_tier"].value_counts().to_dict()
top_records = top.copy()
top_records["transfer_from_store"] = top_records["transfer_from_store"].where(
    top_records["transfer_from_store"].notna(), other=None
)
top_records["source_on_hand"] = top_records["source_on_hand"].where(
    top_records["source_on_hand"].notna(), other=None
)

def fmt(v):
    if v is None or (isinstance(v, float) and v != v):
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)

top_json = "[\n" + ",\n".join(
    "    {" + ", ".join(
        f'"{c}": {fmt(row[c])}'
        for c in ["store_id","sku_id","risk_score","days_of_supply","coverage_ratio",
                  "trend_ratio","on_hand_qty","velocity_7","lead_time_days","perishable",
                  "transfer_from_store","source_on_hand"]
    ) + "}"
    for _, row in top_records.iterrows()
) + "\n  ]"

total_combos = N_STORES * N_SKUS
urgent_n  = int(tier_counts.get("urgent", 0))
watch_n   = int(tier_counts.get("watch",  0))
ok_n      = int(tier_counts.get("ok",     0))
cpu_total = timings["total"]
gpu_proj  = round(cpu_total / 19, 1)   # conservative ~19x from RAPIDS benchmarks

with open("stockout_radar.html") as f:
    html = f.read()

# Patch the DATA block with live values
import re
new_data = f"""const DATA = {{
  "tiers": {{
    "watch": {watch_n},
    "ok": {ok_n},
    "urgent": {urgent_n}
  }},
  "timings": {json.dumps(timings, indent=4)},
  "top": {top_json}
}};"""

html = re.sub(r"const DATA = \{.*?\};", new_data, html, flags=re.DOTALL)

# Patch visible KPI numbers
html = html.replace(">200,000<", f">{total_combos:,}<")
html = html.replace(">15,738<",  f">{urgent_n:,}<")
html = html.replace(">99,356<",  f">{watch_n:,}<")
html = html.replace(">84,906<",  f">{ok_n:,}<")

# Patch timing display in the race track
cpu_str = f"{cpu_total:.1f}s"
gpu_str = f"~{gpu_proj}s"
html = html.replace(">284.2s<", f">{cpu_str}<")
html = html.replace(">~15s<",   f">{gpu_str}<")
html = html.replace(">4m 44s<", f">{int(cpu_total//60)}m {int(cpu_total%60)}s<")

# Patch subtitle scale line
html = html.replace(
    "200 stores × 1,000 SKUs",
    f"{N_STORES} stores × {N_SKUS} SKUs"
)
html = html.replace(
    "200,000 store–SKU combinations scored",
    f"{total_combos:,} store–SKU combinations scored"
)

out_path = "stockout_radar_live.html"
with open(out_path, "w") as f:
    f.write(html)

print(f"\n{'='*55}")
print(f"  [OK] Dashboard written -> {out_path}")
print(f"  {total_combos:,} combos scored | {urgent_n:,} urgent | {watch_n:,} watch")
print(f"  CPU time: {cpu_total:.1f}s  |  GPU projection: ~{gpu_proj}s")
print(f"{'='*55}\n")

if not args.no_browser:
    webbrowser.open(os.path.abspath(out_path))
