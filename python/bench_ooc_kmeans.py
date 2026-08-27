#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile one single-GPU out-of-core KMeans transfer configuration."""

import argparse
import json
import os
import time

import numpy as np


PREFETCH_ENV = "CUVS_KMEANS_OOC_PREFETCH"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memory", choices=("pageable", "pinned"), default="pageable"
    )
    parser.add_argument(
        "--prefetch", action="store_true", help="enable double-buffered H2D"
    )
    parser.add_argument("--rows", type=int, default=8_000_000)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--clusters", type=int, default=2)
    parser.add_argument("--device-buffer-samples", type=int, default=4_000_000)
    parser.add_argument("--batch-samples", type=int, default=32_768)
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_dataset(args):
    shape = (args.rows, args.features)
    if args.memory == "pinned":
        import cupyx

        dataset = cupyx.empty_pinned(shape, dtype=np.float32)
    else:
        dataset = np.empty(shape, dtype=np.float32)

    # Fill in-place so dataset creation does not require another dataset-sized allocation.
    rng = np.random.default_rng(args.seed)
    fill_rows = min(args.device_buffer_samples, args.rows)
    for begin in range(0, args.rows, fill_rows):
        view = dataset[begin : begin + fill_rows]
        rng.random(view.shape, dtype=np.float32, out=view)
    return dataset


def main():
    args = parse_args()
    if args.rows <= 0 or args.features <= 0 or args.clusters <= 0:
        raise ValueError("rows, features, and clusters must be positive")
    if args.clusters > args.rows:
        raise ValueError("clusters cannot exceed rows")
    if args.device_buffer_samples <= 0:
        raise ValueError(
            "device-buffer-samples must be positive to select the OOC path"
        )

    os.environ[PREFETCH_ENV] = "1" if args.prefetch else "0"

    import cupy as cp
    from cuvs.cluster import kmeans

    dataset = make_dataset(args)
    initial_centroids = cp.asarray(dataset[: args.clusters])
    cp.cuda.Device().synchronize()

    params = kmeans.KMeansParams(
        n_clusters=args.clusters,
        init_method="Array",
        max_iter=args.max_iter,
        tol=1e-20,
        n_init=1,
        batch_samples=args.batch_samples,
        device_buffer_samples=args.device_buffer_samples,
    )
    config = {
        "memory": args.memory,
        "prefetch": args.prefetch,
        "rows": args.rows,
        "features": args.features,
        "clusters": args.clusters,
        "device_buffer_samples": args.device_buffer_samples,
        "batch_samples": args.batch_samples,
        "max_iter": args.max_iter,
        "dataset_gib": dataset.nbytes / 2**30,
        "device_input_buffers_gib": (
            min(args.rows, args.device_buffer_samples)
            * args.features
            * np.dtype(np.float32).itemsize
            * (
                2
                if args.prefetch and args.rows > args.device_buffer_samples
                else 1
            )
            / 2**30
        ),
        "env": {PREFETCH_ENV: os.environ[PREFETCH_ENV]},
    }
    print(json.dumps({"configuration": config}, sort_keys=True), flush=True)

    range_name = f"ooc_kmeans/{args.memory}/prefetch_{int(args.prefetch)}"
    cp.cuda.nvtx.RangePush(range_name)
    start = time.perf_counter()
    try:
        _, inertia, n_iter = kmeans.fit(
            params, dataset, centroids=initial_centroids
        )
        cp.cuda.Device().synchronize()
    finally:
        elapsed = time.perf_counter() - start
        cp.cuda.nvtx.RangePop()

    result = {
        "elapsed_seconds": elapsed,
        "inertia": float(inertia),
        "iterations": int(n_iter),
    }
    print(json.dumps({"result": result}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
