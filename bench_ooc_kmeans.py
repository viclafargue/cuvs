#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile host-to-device streaming in single-GPU out-of-core K-means."""

from __future__ import annotations

import argparse
import os
import time

import numpy as np


GIB = 1 << 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memory", choices=("pageable", "pinned"), required=True
    )
    parser.add_argument(
        "--prefetch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable the temporary C++ double-buffered prefetch path.",
    )
    parser.add_argument(
        "--dataset-gib",
        type=float,
        default=160.0,
        help="Host dataset size in GiB (default: 160).",
    )
    parser.add_argument(
        "--batch-gib",
        type=float,
        default=40.0,
        help="Size of each streamed GPU batch in GiB (default: 40).",
    )
    parser.add_argument("--features", type=int, default=256)
    parser.add_argument("--clusters", type=int, default=1024)
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument(
        "--compute-batch-rows",
        type=int,
        default=65_536,
        help="Sample tile used by the inner 1-NN computation.",
    )
    parser.add_argument(
        "--fill-template-mib",
        type=int,
        default=256,
        help="Random template size copied repeatedly to initialize the dataset.",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.dataset_gib <= 0 or args.batch_gib <= 0:
        parser.error("--dataset-gib and --batch-gib must be positive")
    if args.batch_gib > args.dataset_gib:
        parser.error("--batch-gib cannot exceed --dataset-gib")
    for name in ("features", "clusters", "max_iter", "compute_batch_rows"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.fill_template_mib <= 0:
        parser.error("--fill-template-mib must be positive")
    return args


def rows_for_size(size_gib: float, features: int) -> int:
    row_bytes = features * np.dtype(np.float32).itemsize
    return max(1, int(size_gib * GIB) // row_bytes)


def host_available_bytes() -> int | None:
    try:
        with open("/proc/meminfo", encoding="ascii") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def allocate_dataset(cp, memory: str, shape: tuple[int, int]):
    nbytes = int(np.prod(shape)) * np.dtype(np.float32).itemsize
    if memory == "pageable":
        return np.empty(shape, dtype=np.float32), None

    try:
        owner = cp.cuda.alloc_pinned_memory(nbytes)
    except cp.cuda.runtime.CUDARuntimeError as error:
        raise RuntimeError(
            f"Could not allocate {nbytes / GIB:.2f} GiB of pinned host memory"
        ) from error
    return np.ndarray(shape, dtype=np.float32, buffer=owner), owner


def initialize_dataset(
    dataset: np.ndarray, template_mib: int, seed: int
) -> None:
    row_bytes = dataset.shape[1] * dataset.dtype.itemsize
    template_rows = min(
        dataset.shape[0], max(1, (template_mib << 20) // row_bytes)
    )
    template = np.random.default_rng(seed).random(
        (template_rows, dataset.shape[1]), dtype=np.float32
    )
    for begin in range(0, dataset.shape[0], template_rows):
        end = min(begin + template_rows, dataset.shape[0])
        dataset[begin:end] = template[: end - begin]


def main() -> None:
    args = parse_args()

    # Read by the temporary C++ batch iterator implementation.
    os.environ["CUVS_KMEANS_OOC_PREFETCH"] = "1" if args.prefetch else "0"

    import cupy as cp
    from cuvs.cluster import kmeans

    rows = rows_for_size(args.dataset_gib, args.features)
    batch_rows = min(rows, rows_for_size(args.batch_gib, args.features))
    shape = (rows, args.features)
    dataset_bytes = rows * args.features * np.dtype(np.float32).itemsize
    batch_bytes = batch_rows * args.features * np.dtype(np.float32).itemsize

    available = host_available_bytes()
    if available is not None and dataset_bytes > int(available * 0.9):
        raise RuntimeError(
            f"Dataset needs {dataset_bytes / GIB:.2f} GiB but only "
            f"{available / GIB:.2f} GiB of host memory is currently available"
        )

    # Initialize CUDA before the NVTX capture range so context setup is excluded.
    cp.cuda.runtime.getDeviceCount()
    cp.cuda.Device().synchronize()
    free_device, total_device = cp.cuda.Device().mem_info
    resident_batches = 2 if args.prefetch else 1
    transfer_buffers = resident_batches * batch_bytes
    if transfer_buffers > int(total_device * 0.9):
        raise RuntimeError(
            f"The transfer buffers alone need {transfer_buffers / GIB:.2f} GiB "
            f"but the GPU has {total_device / GIB:.2f} GiB"
        )

    print(
        f"Allocating {args.memory} dataset: shape={shape}, "
        f"size={dataset_bytes / GIB:.2f} GiB",
        flush=True,
    )
    dataset, pinned_owner = allocate_dataset(cp, args.memory, shape)
    initialize_dataset(dataset, args.fill_template_mib, args.seed)

    print(
        f"GPU={total_device / GIB:.2f} GiB total, "
        f"{free_device / GIB:.2f} GiB free; "
        f"streaming_batch={batch_rows} rows ({batch_bytes / GIB:.2f} GiB); "
        f"transfer_buffers~={transfer_buffers / GIB:.2f} GiB",
        flush=True,
    )

    params = kmeans.KMeansParams(
        n_clusters=args.clusters,
        init_method="Random",
        max_iter=args.max_iter,
        tol=0.0,
        n_init=1,
        batch_samples=args.compute_batch_rows,
        streaming_batch_size=batch_rows,
    )

    range_name = f"ooc_kmeans/{args.memory}/prefetch_{int(args.prefetch)}"
    print(f"NVTX capture range: {range_name}", flush=True)
    cp.cuda.nvtx.RangePush(range_name)
    start = time.perf_counter()
    try:
        centroids, inertia, n_iter = kmeans.fit(params, dataset)
        cp.cuda.Device().synchronize()
    finally:
        elapsed = time.perf_counter() - start
        cp.cuda.nvtx.RangePop()

    # Keep both allocations alive through the end of the captured fit.
    _ = pinned_owner, centroids
    logical_gib = dataset_bytes * n_iter / GIB
    print(
        f"elapsed={elapsed:.3f}s, iterations={n_iter}, inertia={inertia:.6g}, "
        f"dataset_GiB*iterations={logical_gib:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
