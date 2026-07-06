#!/bin/bash
# Run the identical pipeline on GPU via cuDF's pandas accelerator mode.
# Requires an NVIDIA GPU + RAPIDS installed (e.g. an NVIDIA GPU instance on
# Google Cloud, or a GKE node pool with a GPU node pool attached).
#
# No code changes vs pipeline.py — cudf.pandas patches pandas itself.
python3 -m cudf.pandas pipeline.py
