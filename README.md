# Stockout Radar

**Real-time stockout risk scoring for a regional grocery chain.**  
Joins overnight POS history with a live inventory snapshot, computes rolling demand velocity per store–SKU pair, and surfaces a ranked restock/transfer queue — refreshable hourly instead of once a day.

---

## Quickstart (30-second live demo)

```bash
pip install -r requirements.txt
python demo.py
```

Opens `stockout_radar_live.html` in your browser with live pipeline results.  
Default scale: **50 stores × 200 SKUs** (~30s on CPU, good for a live demo).

---

## Full scale (200 stores × 1,000 SKUs — matches the dashboard numbers)

```bash
python demo.py --full
```
~4–5 min on CPU. This is the run that produced the 284s baseline in the dashboard.

---

## GPU-accelerated (zero code changes)

```bash
pip install cudf-cu12 --extra-index-url=https://pypi.nvidia.com
python demo.py --gpu --full
```

`cudf.pandas` intercepts `import pandas` and routes groupby/rolling/merge to the GPU.  
Projected speedup: **~10–30×** (RAPIDS published benchmarks) → ~9–28s for the full run.

---

## What the pipeline does

| Stage | What happens |
|---|---|
| Ingest | Read POS history + inventory snapshot (Parquet) |
| Clean | Dedupe, null-fill, dtype normalization |
| Feature eng | Rolling 7/14/28-day sales velocity per store–SKU |
| Join | Attach live on-hand inventory |
| Score | Compute stockout risk score (coverage ratio + demand trend) |
| Rank | Surface top urgent rows + identify transfer-source stores |

---

## Why GPU acceleration matters here

At **200 stores × 1,000 SKUs** the pandas rolling-window stage takes ~282s — almost the entire pipeline.  
At full chain scale (**500 stores × 5,000 SKUs**) that stretches toward **~1 hour** on CPU.  
With `cudf.pandas`, the same code projects to **2–6 minutes** — the difference between a once-a-day batch job and an **hourly refresh** that catches a demand spike before the shelf goes empty.

---

## Files

| File | Purpose |
|---|---|
| `demo.py` | **Start here** — generates data, runs pipeline, writes live HTML |
| `pipeline.py` | Core 6-stage pipeline (also the GPU code path via `cudf.pandas`) |
| `generate_data.py` | Synthetic 9M-row POS history generator |
| `generate_inventory.py` | Synthetic inventory snapshot generator |
| `stockout_radar.html` | Dashboard template |
| `stockout_radar_live.html` | Live output (created by `demo.py`) |
| `run_gpu.sh` | Bare GPU run without demo wrapper |
