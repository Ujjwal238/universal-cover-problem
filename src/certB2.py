"""S2 applied: family-B search with the farthest-corner domain lemma.

Two changes against certgen/certemitB, both from domain.py:

  1. ROOT BOX.  Per-body |t| <= TMAX_n(target) instead of a common 0.70.
  2. PER-BOX BOUND.  f(g_n(d)) in place of f(d), where
         g_n(d) = sqrt(R_n^2 + d^2 + 2 R_n d cos(pi/n))  >=  d,
     so the new estimate dominates the old one on every box, not only at the
     root.  d is the distance from the origin to the nearest point of the box.

Everything else (split rule, inflation, erosion core lemma) is unchanged, so the
node count is directly comparable with the 450,922,384 boxes of cert8344hard.log.
"""

import multiprocessing as mp
import sys
import time

import numpy as np

import certgen
from certgen import f_apriori
from domain import tmax
from geom import disk, erode_reuleaux, hull_area_exact, reuleaux_corners

ORDERS = (3, 5)
CORNERS = [reuleaux_corners(n, 1.0) for n in ORDERS]
CIRC = [float(np.max(np.hypot(V[:, 0], V[:, 1]))) for V in CORNERS]
DISK = disk(0.5)
INFLATE = 1e-11
EPS = 1e-9


def g_n(d, R, n):
    return np.sqrt(R * R + d * d + 2.0 * R * d * np.cos(np.pi / n))


def make(target):
    TM = [tmax(target, CIRC[i], ORDERS[i]) for i in range(len(ORDERS))]
    COSN = [np.cos(np.pi / n) for n in ORDERS]
    return TM, COSN


def root(TM):
    b = [0.0, TM[0], 0.0, TM[0]]
    for i in range(1, len(ORDERS)):
        b += [np.pi / ORDERS[i], np.pi / ORDERS[i], 0.0, TM[i], 0.0, TM[i]]
    return tuple(b)


def weights(b):
    return [b[1], b[3], CIRC[1] * b[5], b[7], b[9]]


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


def box_bound2(b):
    c3x, h3x, c3y, h3y, cr, hr, c5x, h5x, c5y, h5y = b
    bodies = [DISK]
    ap = 0.0

    d3 = np.hypot(max(abs(c3x) - h3x, 0.0), max(abs(c3y) - h3y, 0.0))
    ap = max(ap, f_apriori(g_n(d3, CIRC[0], ORDERS[0])))
    d5 = np.hypot(max(abs(c5x) - h5x, 0.0), max(abs(c5y) - h5y, 0.0))
    ap = max(ap, f_apriori(g_n(d5, CIRC[1], ORDERS[1])))
    if ap >= EPS + 0.0 and ap >= _TARGET:
        return ap

    E = erode_reuleaux(CORNERS[0] + np.array([c3x, c3y]), 1.0, np.hypot(h3x, h3y))
    if E is None:
        return ap
    bodies.append(E)
    ca, sa = np.cos(cr), np.sin(cr)
    V = CORNERS[1] @ np.array([[ca, sa], [-sa, ca]]) + np.array([c5x, c5y])
    E = erode_reuleaux(V, 1.0, np.hypot(h5x, h5y) + 2.0 * CIRC[1] * np.sin(0.5 * hr))
    if E is None:
        return ap
    bodies.append(E)
    return max(ap, hull_area_exact(bodies))


_TARGET = 0.0


def _init(t):
    global _TARGET
    _TARGET = t


def count(job):
    box, target, hmin = job
    n = stuck = 0
    stack = [box]
    while stack:
        b = stack.pop()
        n += 1
        if box_bound2(b) >= target + EPS:
            continue
        if max(weights(b)) < hmin:
            stuck += 1
            continue
        stack.extend(split_cover(b))
    return n, stuck


if __name__ == "__main__":
    target = float(sys.argv[1]) if len(sys.argv) > 1 else 0.8344
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 18
    hmin = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5
    _TARGET = target

    TM, _ = make(target)
    boxes = [root(TM)]
    for _ in range(depth):
        boxes = [c for b in boxes for c in split_cover(b)]

    print("=" * 100)
    print(f"S2 COST MEASUREMENT   family B, target a >= {target}")
    print(f"  root box: R3 |t| <= {TM[0]:.12f}   R5 |t| <= {TM[1]:.12f}   "
          f"(was 0.70 for both)")
    print(f"  per-box estimate f(g_n(d)) dominates the old f(d) on every box")
    print(f"  {len(boxes):,} seeds at depth {depth}, hmin {hmin:g}")
    print(f"  reference: old scheme needed 450,922,384 boxes in 364.9 min")
    print("=" * 100)
    sys.stdout.flush()

    t0 = time.time()
    total = stuck = done = 0
    with mp.Pool(8, initializer=_init, initargs=(target,)) as pool:
        for n, st in pool.imap_unordered(
                count, [(b, target, hmin) for b in boxes], chunksize=1):
            total += n
            stuck += st
            done += 1
            if done % 20000 == 0 or done == len(boxes):
                el = time.time() - t0
                print(f"    {done:>7,}/{len(boxes):,} seeds   {total:>13,} boxes   "
                      f"[{el/60:6.1f} min, {total/max(el,1e-9):>9,.0f} box/s]")
                sys.stdout.flush()

    el = time.time() - t0
    print()
    print(f"  boxes          : {total:,}   ({stuck} stuck)")
    print(f"  wall clock     : {el/60:.1f} min")
    print(f"  vs old scheme  : {450922384/max(total,1):.1f}x fewer boxes, "
          f"{364.9/max(el/60,1e-9):.1f}x faster")
    if stuck:
        print("  *** WARNING: stuck nodes -- search did not close")
    else:
        print(f"  SEARCH CLOSED: a >= {target}")
