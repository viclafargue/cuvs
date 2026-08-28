#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR"

PYTHON_BIN=${PYTHON_BIN:-python}
NSYS_BIN=${NSYS_BIN:-nsys}
OUTPUT_DIR=${NSYS_OUTPUT_DIR:-"$ROOT_DIR/ooc-kmeans-profile"}

# Defaults target a 96 GiB RTX PRO 6000: two 32 GiB staging buffers plus per-row
# scratch stay under the RMM pool while the 160 GiB pinned dataset is out of core.
DATASET_GIB=${DATASET_GIB:-160}
BATCH_GIB=${BATCH_GIB:-32}
FEATURES=${FEATURES:-256}
CLUSTERS=${CLUSTERS:-1024}
MAX_ITER=${MAX_ITER:-3}
COMPUTE_BATCH_ROWS=${COMPUTE_BATCH_ROWS:-65536}

command -v "$PYTHON_BIN" >/dev/null
command -v "$NSYS_BIN" >/dev/null
test -f "$ROOT_DIR/bench_ooc_kmeans.py"
test -f "$ROOT_DIR/ooc_report.py"
mkdir -p "$OUTPUT_DIR"

output="$OUTPUT_DIR/ooc_pinned"

echo "Profiling pinned out-of-core K-means"
echo "Output: ${output}.nsys-rep"

"$NSYS_BIN" profile \
  --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  -o "$output" \
  "$PYTHON_BIN" "$ROOT_DIR/bench_ooc_kmeans.py" \
    --dataset-gib "$DATASET_GIB" \
    --batch-gib "$BATCH_GIB" \
    --features "$FEATURES" \
    --clusters "$CLUSTERS" \
    --max-iter "$MAX_ITER" \
    --compute-batch-rows "$COMPUTE_BATCH_ROWS" \
    "$@"

if [[ ! -s "${output}.nsys-rep" ]]; then
  echo "ERROR: Nsight Systems did not generate a report" >&2
  exit 1
fi

echo
"$PYTHON_BIN" "$ROOT_DIR/ooc_report.py" "${output}.nsys-rep"
