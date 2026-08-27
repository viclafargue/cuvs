#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR"

PYTHON_BIN=${PYTHON_BIN:-python}
NSYS_BIN=${NSYS_BIN:-nsys}
OUTPUT_DIR=${NSYS_OUTPUT_DIR:-"$ROOT_DIR/nsys_ooc_kmeans"}

# Defaults target a 96 GiB RTX PRO 6000: prefetching holds about two 40 GiB
# transfer buffers while the 160 GiB host dataset remains out of core.
DATASET_GIB=${DATASET_GIB:-160}
BATCH_GIB=${BATCH_GIB:-40}
FEATURES=${FEATURES:-256}
CLUSTERS=${CLUSTERS:-1024}
MAX_ITER=${MAX_ITER:-3}
COMPUTE_BATCH_ROWS=${COMPUTE_BATCH_ROWS:-65536}

command -v "$PYTHON_BIN" >/dev/null
command -v "$NSYS_BIN" >/dev/null
test -f "$ROOT_DIR/bench_ooc_kmeans.py"
mkdir -p "$OUTPUT_DIR"

run_case() {
  local memory=$1
  local prefetch=$2
  local buffering output range prefetch_arg

  if [[ $prefetch == 1 ]]; then
    buffering=prefetch
    prefetch_arg=--prefetch
  else
    buffering=single
    prefetch_arg=--no-prefetch
  fi

  output="$OUTPUT_DIR/ooc_${memory}_${buffering}"
  range="ooc_kmeans/${memory}/prefetch_${prefetch}"

  echo
  echo "Profiling memory=$memory buffering=$buffering"
  echo "Output: ${output}.nsys-rep"

  "$NSYS_BIN" profile \
    --trace=cuda,nvtx,osrt \
    --capture-range=nvtx \
    --nvtx-capture="$range" \
    --capture-range-end=stop \
    --force-overwrite=true \
    -o "$output" \
    "$PYTHON_BIN" "$ROOT_DIR/bench_ooc_kmeans.py" \
      --memory "$memory" \
      "$prefetch_arg" \
      --dataset-gib "$DATASET_GIB" \
      --batch-gib "$BATCH_GIB" \
      --features "$FEATURES" \
      --clusters "$CLUSTERS" \
      --max-iter "$MAX_ITER" \
      --compute-batch-rows "$COMPUTE_BATCH_ROWS"
}

run_case pageable 0
run_case pageable 1
run_case pinned 0
run_case pinned 1

echo
echo "Profiles written to $OUTPUT_DIR"
