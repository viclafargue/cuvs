#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile host-to-device streaming in single-GPU out-of-core K-means.

The dataset is always allocated in pinned host memory: `cudaMemcpyAsync` on
pageable memory blocks the calling thread for the whole transfer, so the
double-buffered prefetch in the batch iterator cannot overlap anything.

By default the whole run is served from a pre-reserved RMM pool. Without it,
per-batch temporaries (the CUTLASS mutex array in the fused 1-NN kernel, cuB
scratch, ...) hit `cudaMalloc`/`cudaFree`, and `cudaFree` synchronizes the whole
device -- which serializes the copy stream against the compute stream and
destroys the overlap. Use --no-rmm-pool to measure that effect.
"""

from __future__ import annotations

import argparse
import time

import numpy as np


GIB = 1 << 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-gib",
        type=float,
        default=160.0,
        help="Pinned host dataset size in GiB (default: 160).",
    )
    parser.add_argument(
        "--batch-gib",
        type=float,
        default=32.0,
        help="Size of each streamed GPU batch in GiB (default: 32).",
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
        "--rmm-pool",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pre-reserve an RMM pool so the fit issues no driver allocations.",
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


def device_pool_bytes(
    batch_rows: int, batch_bytes: int, tile_rows: int, clusters: int
) -> int:
    """Device memory the streaming fit needs, for pre-reserving the RMM pool.

    Two staging buffers hold the current and the prefetched batch. On top of
    that the fit keeps a few per-row vectors alive (the <label, distance> pairs
    dominate at 16 B/row) and a distance tile for the inner 1-NN computation.
    """
    per_row = (
        16 + 3 * 4 + 1 + 1
    )  # <label,distance>, 3 float vectors, 2 byte buffers
    tile_scratch = tile_rows * clusters * np.dtype(np.float32).itemsize
    return 2 * batch_bytes + batch_rows * per_row + tile_scratch + GIB // 2


def allocate_pinned_dataset(cp, shape: tuple[int, int]):
    nbytes = int(np.prod(shape)) * np.dtype(np.float32).itemsize
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

    import cupy as cp
    from cuvs.cluster import kmeans

    rows = rows_for_size(args.dataset_gib, args.features)
    batch_rows = min(rows, rows_for_size(args.batch_gib, args.features))
    shape = (rows, args.features)
    dataset_bytes = rows * args.features * np.dtype(np.float32).itemsize
    batch_bytes = batch_rows * args.features * np.dtype(np.float32).itemsize
    n_batches = -(-rows // batch_rows)

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

    pool_bytes = device_pool_bytes(
        batch_rows, batch_bytes, args.compute_batch_rows, args.clusters
    )
    if pool_bytes > free_device:
        raise RuntimeError(
            f"Streaming {batch_bytes / GIB:.2f} GiB batches needs about "
            f"{pool_bytes / GIB:.2f} GiB on the device (two staging buffers plus "
            f"per-row scratch) but only {free_device / GIB:.2f} GiB is free. "
            f"Lower --batch-gib."
        )
    if args.rmm_pool:
        import rmm

        # Uncapped so a short estimate degrades into a few extra cudaMallocs
        # rather than a bad_alloc.
        rmm.reinitialize(pool_allocator=True, initial_pool_size=pool_bytes)

    print(
        f"Allocating pinned dataset: shape={shape}, "
        f"size={dataset_bytes / GIB:.2f} GiB",
        flush=True,
    )
    dataset, pinned_owner = allocate_pinned_dataset(cp, shape)
    initialize_dataset(dataset, args.fill_template_mib, args.seed)

    print(
        f"GPU={total_device / GIB:.2f} GiB total, "
        f"{free_device / GIB:.2f} GiB free; "
        f"streaming_batch={batch_rows} rows ({batch_bytes / GIB:.2f} GiB) "
        f"x {n_batches} batches/pass; "
        f"rmm_pool={'%.2f GiB' % (pool_bytes / GIB) if args.rmm_pool else 'off'}",
        flush=True,
    )

    params = kmeans.KMeansParams(
        n_clusters=args.clusters,
        init_method="Random",
        max_iter=args.max_iter,
        tol=1.0e-12,
        n_init=1,
        batch_samples=args.compute_batch_rows,
        device_buffer_samples=batch_rows,
    )

    range_name = "ooc_kmeans/pinned"
    print(f"NVTX capture range: {range_name}", flush=True)
    cp.cuda.runtime.profilerStart()
    cp.cuda.nvtx.RangePush(range_name)
    start = time.perf_counter()
    try:
        centroids, inertia, n_iter = kmeans.fit(params, dataset)
        cp.cuda.Device().synchronize()
    finally:
        elapsed = time.perf_counter() - start
        try:
            cp.cuda.nvtx.RangePop()
        finally:
            cp.cuda.runtime.profilerStop()

    # Keep both allocations alive through the end of the captured fit.
    _ = pinned_owner, centroids

    # One pass per Lloyd iteration, plus the final pass that recomputes inertia
    # against the converged centroids.
    passes = n_iter + 1
    moved_gib = dataset_bytes * passes / GIB
    print(
        f"elapsed={elapsed:.3f}s, iterations={n_iter}, inertia={inertia:.6g}",
        flush=True,
    )
    print(
        f"host->device minimum={moved_gib:.2f} GiB over {passes} passes "
        f"=> {moved_gib / elapsed:.1f} GiB/s effective",
        flush=True,
    )


if __name__ == "__main__":
    main()
