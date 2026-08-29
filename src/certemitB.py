"""
Emit a replayable certificate for family B (disk + Reuleaux3 + Reuleaux5).

Same tree encoding as family A -- one bit per node, DFS pre-order, boxes never
stored -- but the search threshold is set from the CURVED-core witness deficit,
which is far larger than the polygon case:

    family A (polygon cores):  vertices are exact, only the disk is sampled
                               -> deficit 8.0e-6, 1.3% of a 5.97e-4 margin
    family B (Reuleaux cores): every arc must be inscribed
                               -> deficit 1.6e-5, 4.2% of a 3.81e-4 margin

With K witness points per core allocated in proportion to arc span, each segment
subtends 2*pi/K, so the sagitta is rho(1-cos(pi/K)) <= 1-cos(pi/K) and

    deficit <= eps * perimeter + pi * eps^2

Searching at target + deficit guarantees the verifier's weaker bound still
clears the true target.

Load balancing: imap_unordered with deep seeding.  pool.map pre-assigns chunks,
and one outsized seed subtree then monopolises a single worker while the rest
idle; on-demand assignment avoids that.
"""

import multiprocessing as mp
import os
import struct
import sys
import time

import numpy as np

import certgen
from certgen import box_bound, setup

setup((3, 5))
CIRC5 = certgen.CIRC[1]
TMAX = certgen.TMAX
INFLATE = 1e-11
MAGIC = b"LEBCERTB"


def deficit(K, perim=3.4):
    eps = 1.0 - np.cos(np.pi / K)
    return eps * perim + np.pi * eps * eps


def weights(b):
    return [b[1], b[3], CIRC5 * b[5], b[7], b[9]]


def split_cover(b):
    k = int(np.argmax(weights(b)))
    half = b[2 * k + 1] * 0.5
    out = []
    for s in (-1, 1):
        n = list(b)
        n[2 * k] = b[2 * k] + s * half
        n[2 * k + 1] = half * (1.0 + INFLATE)
        out.append(tuple(n))
    return out


def seeds_at(depth):
    boxes = [(0.0, TMAX, 0.0, TMAX, np.pi / 5, np.pi / 5, 0.0, TMAX, 0.0, TMAX)]
    for _ in range(depth):
        boxes = [c for b in boxes for c in split_cover(b)]
    return boxes


def emit(job):
    box, thresh, hmin, idx = job
    bits = bytearray()
    acc = nbit = n = stuck = 0
    stack = [box]

    def push(bit):
        nonlocal acc, nbit
        acc |= bit << (7 - nbit)
        nbit += 1
        if nbit == 8:
            bits.append(acc)
            acc = 0
            nbit = 0

    while stack:
        b = stack.pop()
        n += 1
        if box_bound(b) >= thresh:
            push(0)
            continue
        if max(weights(b)) < hmin:
            push(0)
            stuck += 1
            continue
        push(1)
        c1, c2 = split_cover(b)
        stack.append(c2)
        stack.append(c1)
    if nbit:
        bits.append(acc)
    return idx, bytes(bits), n, stuck


if __name__ == "__main__":
    target = float(sys.argv[1])
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    # SEEDING DEPTH is set by the LARGEST subtree, not the average.  Work
    # concentrates near the optimum, so a mean of a few hundred nodes per seed can
    # coexist with individual subtrees holding hundreds of millions.  A block
    # cannot be split across workers, since DFS order must be sequential within
    # it, so depth is the only lever: each extra level halves the worst case.
    depth = int(sys.argv[4]) if len(sys.argv) > 4 else 22
    hmin = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-5
    out = sys.argv[6] if len(sys.argv) > 6 else f"cert_B_{target}.bin"

    dfc = deficit(K)
    thresh = target + dfc + 1e-9
    sd = seeds_at(depth)
    print("=" * 100)
    print(f"EMIT family-B CERTIFICATE   target a >= {target}")
    print(f"  witness: {m}-gon disk, {K} arc points per curved core")
    print(f"  arc sagitta {1-np.cos(np.pi/K):.3e}  ->  deficit <= {dfc:.3e}")
    print(f"  search threshold = {thresh:.12f}   (5-D ceiling is 0.834781)")
    print(f"  {len(sd):,} seeds (depth {depth}), hmin {hmin:g}")
    print("=" * 100)

    t0 = time.time()
    blocks = [None] * len(sd)
    total = stuck = done = 0
    with mp.Pool(8) as pool:
        for idx, blk, n, st in pool.imap_unordered(
                emit, [(b, thresh, hmin, i) for i, b in enumerate(sd)], chunksize=1):
            blocks[idx] = blk
            total += n
            stuck += st
            done += 1
            if done % 20000 == 0 or done == len(sd):
                el = time.time() - t0
                print(f"    {done:>7,}/{len(sd):,} seeds   {total:>14,} nodes   "
                      f"[{el/60:6.1f} min, {total/max(el,1e-9):>9,.0f} nodes/s]")
                sys.stdout.flush()
    if any(b is None for b in blocks):
        raise RuntimeError("missing block")

    with open(out, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<dddiiii", target, thresh, TMAX, m, K, depth, len(sd)))
        fh.write(struct.pack("<d", INFLATE))
        fh.write(struct.pack("<Q", len(blocks)))
        for blk in blocks:
            fh.write(struct.pack("<I", len(blk)))
        for blk in blocks:
            fh.write(blk)

    sz = os.path.getsize(out)
    print(f"\n  nodes emitted : {total:,}   ({stuck} stuck)")
    print(f"  certificate   : {out}  {sz/1e6:.1f} MB  ({8*sz/max(total,1):.2f} bits/node)")
    print(f"  wall clock    : {(time.time()-t0)/60:.1f} min")
    if stuck:
        print("  *** WARNING: stuck nodes -- the search did not close")
