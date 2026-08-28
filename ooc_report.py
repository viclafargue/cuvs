#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Answer one question about an out-of-core K-means Nsight Systems report:
did the host-to-device transfers overlap with compute?

Usage: ooc_report.py <report.nsys-rep | report.sqlite> [...]
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys

GIB = 1 << 30
HTOD = 1  # ENUM_CUDA_MEMCPY_OPER id for host-to-device
BIG_COPY = (
    1 << 28
)  # 256 MiB: the streamed dataset batches, not the small scratch copies
BURST_GAP = (
    5_000_000  # 5 ms; merge closer kernels into one burst for the timeline
)


def as_sqlite(path: str) -> str:
    if path.endswith(".sqlite"):
        return path
    target = path.rsplit(".", 1)[0] + ".sqlite"
    if os.path.exists(target) and os.path.getmtime(target) > os.path.getmtime(
        path
    ):
        return target
    nsys = shutil.which("nsys")
    if nsys is None:
        raise SystemExit(
            "nsys not found on PATH; pass an already exported .sqlite"
        )
    done = subprocess.run(
        [
            nsys,
            "export",
            "--type",
            "sqlite",
            "--force-overwrite",
            "true",
            "-o",
            target,
            path,
        ],
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        raise SystemExit(f"nsys export failed for {path}:\n{done.stderr}")
    return target


def union_busy(intervals: list[tuple[int, int]], lo: int, hi: int) -> int:
    total = 0
    cur_start = cur_end = None
    for start, end in sorted(intervals):
        start, end = max(start, lo), min(end, hi)
        if end <= start:
            continue
        if cur_start is None:
            cur_start, cur_end = start, end
        elif start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    if cur_start is not None:
        total += cur_end - cur_start
    return total


def report(path: str) -> None:
    conn = sqlite3.connect(as_sqlite(path))
    strings = dict(conn.execute("select id, value from StringIds"))

    window = conn.execute(
        "select start, end from NVTX_EVENTS where text like 'ooc_kmeans%'"
    ).fetchone()
    if window is None:
        raise SystemExit(f"{path}: no ooc_kmeans NVTX range found")
    t0, t1 = window
    wall = t1 - t0

    copies = list(
        conn.execute(
            "select start, end, bytes, streamId, copyKind "
            "from CUPTI_ACTIVITY_KIND_MEMCPY "
            "where start >= ? and end <= ? order by start",
            (t0, t1),
        )
    )
    kernels = list(
        conn.execute(
            "select start, end, demangledName from CUPTI_ACTIVITY_KIND_KERNEL "
            "where start >= ? and end <= ? order by start",
            (t0, t1),
        )
    )
    copy_spans = [(s, e) for s, e, _, _, _ in copies]
    kernel_spans = [(s, e) for s, e, _ in kernels]
    big = [c for c in copies if c[2] >= BIG_COPY and c[4] == HTOD]

    copy_busy = union_busy(copy_spans, t0, t1)
    kernel_busy = union_busy(kernel_spans, t0, t1)
    gpu_busy = union_busy(copy_spans + kernel_spans, t0, t1)
    overlap = copy_busy + kernel_busy - gpu_busy
    h2d_bytes = sum(c[2] for c in copies if c[4] == HTOD)
    big_bytes = sum(c[2] for c in big)
    streams = sorted({c[3] for c in big})

    print("=" * 78)
    print(os.path.basename(path))
    print("=" * 78)
    print(f"  wall (NVTX range)        {wall / 1e9:8.2f} s")
    print(f"  host->device             {h2d_bytes / GIB:8.2f} GiB")
    if big:
        big_busy = union_busy([(s, e) for s, e, _, _, _ in big], t0, t1)
        print(
            f"  batch transfers          {len(big):8d} x "
            f"{big_bytes / len(big) / GIB:.2f} GiB at "
            f"{big_bytes / (big_busy / 1e9) / 1e9:.1f} GB/s on stream(s) {streams}"
        )
    print(
        f"  copy busy                {copy_busy / 1e9:8.2f} s ({100 * copy_busy / wall:5.1f} %)"
    )
    print(
        f"  kernel busy              {kernel_busy / 1e9:8.2f} s ({100 * kernel_busy / wall:5.1f} %)"
    )
    overlapped_pct = (
        100 * overlap / min(copy_busy, kernel_busy) if kernel_busy else 0.0
    )
    print(
        f"  copy/kernel overlap      {overlap / 1e9:8.2f} s "
        f"({overlapped_pct:5.1f} % of the shorter of the two)"
    )
    print(f"  gpu idle                 {(wall - gpu_busy) / 1e9:8.2f} s")

    # A fully serialized run costs copy_busy + kernel_busy. A fully overlapped one
    # costs the larger of the two, plus one batch of pipeline fill per pass.
    print(
        f"  serial floor {(copy_busy + kernel_busy) / 1e9:.2f} s, "
        f"overlapped floor {max(copy_busy, kernel_busy) / 1e9:.2f} s, "
        f"actual {wall / 1e9:.2f} s"
    )

    print("\n  timeline (batch transfers and kernel bursts)")
    events = [
        (s, e, f"H2D {b / GIB:5.2f} GiB stream {st}") for s, e, b, st, _ in big
    ]
    bursts: list[list] = []
    for start, end, name in kernels:
        if bursts and start - bursts[-1][1] < BURST_GAP:
            bursts[-1][1] = max(bursts[-1][1], end)
            bursts[-1][2] += 1
        else:
            bursts.append(
                [start, end, 1, {}]
            )  # start, end, count, name -> busy ns
        bursts[-1][3][name] = bursts[-1][3].get(name, 0) + (end - start)
    for start, end, count, by_name in bursts:
        if end - start < BURST_GAP:
            continue
        hottest = max(by_name, key=by_name.get)
        label = strings.get(hottest, "?").split("<")[0].split("(")[0]
        events.append(
            (start, end, f"{count:5d} kernels, mostly {label[-40:]}")
        )

    for start, end, label in sorted(events):
        bar_lo = int(60 * (start - t0) / wall)
        bar_hi = max(bar_lo + 1, int(60 * (end - t0) / wall))
        bar = " " * bar_lo + "#" * (bar_hi - bar_lo)
        print(
            f"  {(start - t0) / 1e9:7.2f} -> {(end - t0) / 1e9:7.2f} "
            f"({(end - start) / 1e9:5.2f}s) |{bar:<60}| {label}"
        )
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for arg in sys.argv[1:]:
        report(arg)
