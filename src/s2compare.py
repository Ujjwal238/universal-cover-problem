"""S2 measured: does the farthest-corner domain lemma actually reduce cost?

Controlled comparison.  Identical split rule, inflation, hmin, prune epsilon and
seeding depth; the ONLY differences are

    OLD:  root |t| <= 0.70 for both bodies,   a priori estimate f(d)
    NEW:  root |t| <= TMAX_n(target),         a priori estimate f(g_n(d))

run at targets cheap enough to finish both.  Reporting is time-based, because a
seed-count trigger hides progress when a few subtrees dominate.
"""

import multiprocessing as mp
import sys
import time

import numpy as np

from certgen import f_apriori
from domain import tmax
from geom import disk, erode_reuleaux, hull_area_exact, reuleaux_corners

ORDERS = (3, 5)
CORNERS = [reuleaux_corners(n, 1.0) for n in ORDERS]
CIRC = [float(np.max(np.hypot(V[:, 0], V[:, 1]))) for V in CORNERS]
DISK = disk(0.5)
INFLATE = 1e-11
EPS = 1e-9
COSN = [np.cos(np.pi / n) for n in ORDERS]


def g_n(d, i):
    R = CIRC[i]
    return np.sqrt(R * R + d * d + 2.0 * R * d * COSN[i])


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


def box_bound(b, sharp):
    c3x, h3x, c3y, h3y, cr, hr, c5x, h5x, c5y, h5y = b
    d3 = np.hypot(max(abs(c3x) - h3x, 0.0), max(abs(c3y) - h3y, 0.0))
    d5 = np.hypot(max(abs(c5x) - h5x, 0.0), max(abs(c5y) - h5y, 0.0))
    if sharp:
        ap = max(f_apriori(g_n(d3, 0)), f_apriori(g_n(d5, 1)))
    else:
        ap = max(f_apriori(d3), f_apriori(d5))
    bodies = [DISK]
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


def root(sharp, target):
    if sharp:
        TM = [tmax(target, CIRC[i], ORDERS[i]) for i in range(2)]
    else:
        TM = [0.70, 0.70]
    return (0.0, TM[0], 0.0, TM[0], np.pi / 5, np.pi / 5, 0.0, TM[1], 0.0, TM[1]), TM


def count(job):
    box, target, hmin, sharp = job
    n = stuck = 0
    stack = [box]
    thr = target + EPS
    while stack:
        b = stack.pop()
        n += 1
        if box_bound(b, sharp) >= thr:
            continue
        if max(weights(b)) < hmin:
            stuck += 1
            continue
        stack.extend(split_cover(b))
    return n, stuck


def run(target, sharp, depth, hmin):
    r, TM = root(sharp, target)
    boxes = [r]
    for _ in range(depth):
        boxes = [c for b in boxes for c in split_cover(b)]
    tag = "NEW f(g(d))" if sharp else "OLD f(d)   "
    print(f"  {tag}  root |t| <= ({TM[0]:.6f}, {TM[1]:.6f})  {len(boxes):,} seeds")
    sys.stdout.flush()
    t0 = time.time()
    total = stuck = done = 0
    last = t0
    with mp.Pool(8) as pool:
        for n, st in pool.imap_unordered(
                count, [(b, target, hmin, sharp) for b in boxes], chunksize=4):
            total += n
            stuck += st
            done += 1
            if time.time() - last > 60.0:
                last = time.time()
                el = last - t0
                print(f"      {done:>8,}/{len(boxes):,} seeds  {total:>13,} boxes  "
                      f"[{el/60:5.1f} min, {total/el:>9,.0f} box/s]")
                sys.stdout.flush()
    el = time.time() - t0
    print(f"      -> {total:,} boxes, {stuck} stuck, {el/60:.2f} min")
    sys.stdout.flush()
    return total, stuck, el


if __name__ == "__main__":
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 18
    hmin = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-5
    targets = [float(x) for x in sys.argv[3:]] or [0.8330, 0.8340]

    print("=" * 100)
    print("S2 CONTROLLED COMPARISON: old centre-distance vs new farthest-corner domain")
    print(f"  depth {depth}, hmin {hmin:g}, prune eps {EPS:g}, inflate {INFLATE:g}")
    print("=" * 100)
    rows = []
    for tg in targets:
        print(f"\ntarget a >= {tg}")
        o, os_, oe = run(tg, False, depth, hmin)
        n, ns, ne = run(tg, True, depth, hmin)
        rows.append((tg, o, n, oe, ne))
        print(f"      RATIO  boxes {o/max(n,1):.2f}x fewer,  time {oe/max(ne,1e-9):.2f}x faster")

    print("\n" + "=" * 100)
    print(f"{'target':>10} {'old boxes':>15} {'new boxes':>15} {'box ratio':>10} "
          f"{'old min':>9} {'new min':>9}")
    for tg, o, n, oe, ne in rows:
        print(f"{tg:10.4f} {o:15,} {n:15,} {o/max(n,1):10.2f} {oe/60:9.2f} {ne/60:9.2f}")
    print("=" * 100)
