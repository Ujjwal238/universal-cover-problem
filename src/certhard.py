"""
PHASE 1: hardened certifier -- converts the remaining soundness arguments into
measurements.

Three changes from certgen, each turning a step that could be argued informally rather
than demonstrated:

  1. STRICT PRUNE.  A box is discarded only when its bound exceeds the target by
     EPS = 1e-9.  The bound is a few hundred flops of trig/sqrt/shoelace, so its
     floating-point error is ~1e-13 at worst; requiring 1e-9 of slack leaves four
     orders of headroom.  Without this the honest claim would be
     "a >= target - 1e-13" rather than "a >= target".

  2. INFLATED CHILDREN.  split() previously produced children whose union equalled
     the parent only up to 1 ulp -- the tiling audit measured gaps of 2.2e-16.
     Children are now widened by a relative 1e-11, so their union provably
     CONTAINS the parent, with explicit overlap.  Widening a box can only weaken
     its bound (larger box -> larger delta -> smaller core -> lower area), so this
     is conservative in the safe direction.

  3. COVERAGE ACCOUNTING.  imap_unordered results are counted against jobs
     dispatched every round.  A silently dropped result would mean an unexplored
     subtree and an invalid certificate; now it aborts instead.

It also records the minimum (bound - target) over all pruned boxes, so the
tightness of the closest pruning decision is reported rather than assumed.

box_bound itself is imported unchanged from certgen -- it is the function the
audit and the independent reimplementation both validated.
"""

import multiprocessing as mp
import sys
import time

import numpy as np

import certgen
from certgen import TMAX, box_bound, f_apriori, setup, weights

EPS = 1e-9          # required slack above target before a box may be pruned
INFLATE = 1e-11     # relative widening of children, to guarantee covering


def split_cover(b):
    """Two children whose union provably CONTAINS the parent (with overlap)."""
    k = int(np.argmax(weights(b)))
    half = b[2 * k + 1] * 0.5
    out = []
    for s in (-1, 1):
        n = list(b)
        n[2 * k] = b[2 * k] + s * half
        n[2 * k + 1] = half * (1.0 + INFLATE)
        out.append(tuple(n))
    return out


def task(job):
    box, target, hmin, cap = job
    stack = [box]
    n = 0
    fails = []
    tight = np.inf                     # min (bound - target) over pruned boxes
    while stack:
        if n >= cap:
            return n, fails, stack, tight
        b = stack.pop()
        n += 1
        lb = box_bound(b)
        if lb >= target + EPS:
            tight = min(tight, lb - target)
            continue
        if max(weights(b)) < hmin:
            fails.append((b, lb))
            if len(fails) >= 40:
                return n, fails, stack, tight
            continue
        stack.extend(split_cover(b))
    return n, fails, [], tight


def seeds(depth):
    boxes = [(0.0, TMAX, 0.0, TMAX, np.pi / 5, np.pi / 5, 0.0, TMAX, 0.0, TMAX)]
    for _ in range(depth):
        boxes = [c for b in boxes for c in split_cover(b)]
    return boxes


def _init(orders):
    setup(orders)


if __name__ == "__main__":
    orders = tuple(int(c) for c in sys.argv[1])
    target = float(sys.argv[2])
    hmin = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5
    nproc = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    cap = int(float(sys.argv[5])) if len(sys.argv) > 5 else 120_000
    depth = int(sys.argv[6]) if len(sys.argv) > 6 else 8

    setup(orders)
    print("=" * 110)
    print(f"HARDENED CERTIFY  a >= {target}   family = disk + " +
          " + ".join(f"Reuleaux{n}" for n in orders))
    print(f"  strict prune: bound >= target + {EPS:g}   (floating-point headroom)")
    print(f"  children inflated by {INFLATE:g} -> union provably covers the parent")
    print(f"  coverage accounting on; |t| <= {TMAX} a priori (f = {f_apriori(TMAX):.9f})")
    print("=" * 110)

    queue = seeds(depth)
    total, all_fails, rnd, tight = 0, [], 0, np.inf
    lost = 0
    t0 = time.time()
    with mp.Pool(nproc, initializer=_init, initargs=(orders,)) as pool:
        while queue:
            rnd += 1
            jobs = [(b, target, hmin, cap) for b in queue]
            nxt, got = [], 0
            for n, fails, left, tg in pool.imap_unordered(task, jobs, chunksize=1):
                got += 1
                total += n
                all_fails += fails
                nxt += left
                tight = min(tight, tg)
            if got != len(jobs):
                lost = len(jobs) - got
                print(f"  !! COVERAGE LOSS: {lost} of {len(jobs)} results missing -- ABORT")
                break
            el = time.time() - t0
            print(f"  round {rnd:>3}: {len(jobs):>7,} tasks -> {total:>15,} boxes, "
                  f"{len(nxt):>7,} requeued, {len(all_fails):>4} stuck, "
                  f"tightest prune +{tight:.2e}   [{el/60:6.1f} min, "
                  f"{total/max(el,1e-9):>8,.0f} box/s]")
            sys.stdout.flush()
            if all_fails:
                break
            queue = nxt

    print("\n" + "=" * 110)
    if not all_fails and not lost and not queue:
        print(f"  CERTIFIED:  a >= {target}")
        print(f"  {total:,} boxes, {rnd} rounds, {(time.time()-t0)/60:.1f} min")
        print(f"  tightest pruning decision cleared the target by {tight:.3e} "
              f"(FP error ~1e-13)")
        print(f"  every box pruned with >= {EPS:g} slack; children provably cover parents;")
        print(f"  all {rnd} rounds fully accounted for")
    else:
        print(f"  NOT closed: {len(all_fails)} stuck, {lost} lost, {len(queue)} unfinished "
              f"({total:,} boxes)")
    print("=" * 110)
