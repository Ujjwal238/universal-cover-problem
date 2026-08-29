"""
PHASE 4a: emit a replayable certificate for family A.

The certificate is the branch-and-bound TREE, encoded as one bit per node in
canonical DFS pre-order: 1 = split (two children follow), 0 = leaf (pruned).
Boxes themselves are never stored -- the verifier regenerates every box from the
root using the deterministic split rule, so 926M boxes cost 926M bits.

SEARCH THRESHOLD.  The verifier will re-derive each leaf's bound by elementary
means (witness points + hull + shoelace), which under-estimates the exact bound
because the disk is replaced by an inscribed m-gon.  That deficit is bounded
rigorously: D is inside D_m (+) eps*B with sagitta eps = (1-cos(pi/m))/2, so

    witness >= exact - (eps * perimeter + pi * eps^2)

Searching at  target + deficit  therefore guarantees every emitted leaf will
clear the true target under the verifier's weaker bound, with no adaptivity.

PARALLEL EMISSION.  The tree must be assembled in a canonical order, but the
production search requeues partial stacks across workers, which scrambles it.
So emission runs each seed subtree to completion in one task and concatenates
the bit blocks in seed order.  Seeds are taken deep enough (2^14) that load
imbalance stays acceptable without requeueing.
"""

import multiprocessing as mp
import struct
import sys
import time

import numpy as np

from familyA import (INFLATE, R3, R5, TMAX, box_bound_A, root, split_cover,
                     weights)

MAGIC = b"LEBCERT2"


def witness_deficit(m, perim=3.4):
    """Rigorous bound on how much the inscribed m-gon can lose."""
    eps = 0.5 * (1.0 - np.cos(np.pi / m))
    return eps * perim + np.pi * eps * eps


def seeds_at(depth):
    boxes = [root()]
    for _ in range(depth):
        boxes = [c for b in boxes for c in split_cover(b)]
    return boxes


def emit_subtree(job):
    """Iterative DFS emitting one bit per node, in pre-order."""
    box, thresh, hmin, idx = job
    bits = bytearray()
    acc = 0
    nbit = 0
    stack = [box]
    n = 0
    stuck = 0

    def push(bit):
        nonlocal acc, nbit
        acc |= bit << (7 - nbit)
        nbit += 1
        if nbit == 8:
            bits.append(acc)
            acc = 0
            nbit = 0

    # pre-order DFS: emit this node's bit, then its children
    while stack:
        b = stack.pop()
        n += 1
        if box_bound_A(b) >= thresh:
            push(0)
            continue
        if max(weights(b)) < hmin:
            push(0)
            stuck += 1
            continue
        push(1)
        c1, c2 = split_cover(b)
        stack.append(c2)          # pushed first so c1 is processed first
        stack.append(c1)
    if nbit:
        bits.append(acc)
    return idx, bytes(bits), n, stuck


if __name__ == "__main__":
    target = float(sys.argv[1])
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    depth = int(sys.argv[3]) if len(sys.argv) > 3 else 18
    hmin = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-5
    out = sys.argv[5] if len(sys.argv) > 5 else f"cert_A_{target}.bin"

    dfc = witness_deficit(m)
    thresh = target + dfc + 1e-9
    sd = seeds_at(depth)

    print("=" * 100)
    print(f"EMIT CERTIFICATE  family A,  target a >= {target}")
    print(f"  verifier witness resolution m = {m}  ->  rigorous deficit <= {dfc:.3e}")
    print(f"  search threshold = target + deficit + 1e-9 = {thresh:.12f}")
    print(f"  {len(sd):,} seed subtrees (depth {depth}), hmin {hmin:g}")
    print("=" * 100)

    # DYNAMIC ASSIGNMENT + PROGRESS.  pool.map pre-assigns chunks, so a single
    # outsized seed subtree can monopolise one worker while the rest idle.
    # imap_unordered assigns on demand, and depth-18 seeding makes the largest
    # unit far smaller, so the tail is minutes rather than hours.  Progress is
    # printed so a stalled run is visible instead of silent.
    t0 = time.time()
    blocks = [None] * len(sd)
    total = stuck = done = 0
    with mp.Pool(8) as pool:
        for idx, blk, n, st in pool.imap_unordered(
                emit_subtree,
                [(b, thresh, hmin, i) for i, b in enumerate(sd)],
                chunksize=1):
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
        raise RuntimeError("missing block -- a worker died")

    with open(out, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<dddiii", target, thresh, TMAX, m, depth, len(sd)))
        fh.write(struct.pack("<d", INFLATE))
        fh.write(struct.pack("<Q", len(blocks)))
        for blk in blocks:
            fh.write(struct.pack("<I", len(blk)))
        for blk in blocks:
            fh.write(blk)

    import os
    sz = os.path.getsize(out)
    print(f"\n  nodes emitted : {total:,}   ({stuck} stuck)")
    print(f"  certificate   : {out}  {sz/1e6:.1f} MB  ({8*sz/max(total,1):.2f} bits/node)")
    print(f"  wall clock    : {(time.time()-t0)/60:.1f} min")
    if stuck:
        print("  *** WARNING: stuck nodes present -- the search did not close")
