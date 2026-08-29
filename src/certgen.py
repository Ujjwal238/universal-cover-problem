"""
General load-balanced branch-and-bound certificate for  a >= L.

Family: the unit-diameter disk (fixed at the origin) plus width-1 regular
Reuleaux polygons of orders n_1..n_k.  Gauge: the disk pins two translations and
is rotation invariant, and the residual global rotation fixes the FIRST Reuleaux
body's orientation.  So the placement space has dimension 2 + 3(k-1):

    (t1x, t1y)  then  (rho_i, tix, tiy)  for i = 2..k

Each Reuleaux body of order n is invariant under rotation by 2*pi/n, so its rho
need only range over [0, 2*pi/n) -- an independent reduction per body.

BOUND.  Over a box, every placement of body i displaces its points by at most
delta_i from the box-centre placement, so the box-centre body eroded by
delta_i*disk is contained in EVERY placement in the box (because
g0.C subset gC (+) dB and (X (+) dB) (-) dB = X for convex X).  Erosion
distributes over intersection and a width-w Reuleaux polygon is exactly
intersection_j D(V_j, w), hence the core is the same disks with radius
w - delta.  The hull of the disk with all the cores is then a rigorous lower
bound on the area over the whole box.

LOAD BALANCING.  The previous version used pool.map over a fixed set of
subtrees.  Subtree difficulty is wildly uneven -- the few containing the optimum
are orders of magnitude harder -- so with pre-assigned chunks six of eight
workers finished early and idled while two ground on alone at 25% overall
efficiency.  Here each task carries a modest box cap and RETURNS ITS UNFINISHED
DFS STACK, which is re-queued.  Because the search is depth-first the leftover
stack is only O(depth) entries, each standing for a large subtree, so hard
regions get chopped and redistributed every round.  imap_unordered assigns
dynamically, and every round prints a box count -- the progress signal the
previous run lacked.
"""

import multiprocessing as mp
import sys
import time

import numpy as np

from geom import disk, erode_reuleaux, hull_area_exact, reuleaux_corners

TMAX = 0.70
DISK = disk(0.5)


def f_apriori(d):
    """area of conv(disk of radius 1/2, a point at distance d) -- exact."""
    if d <= 0.5:
        return np.pi / 4
    return 0.25 * (np.pi - np.arccos(0.5 / d)) + 0.5 * np.sqrt(d * d - 0.25)


ORDERS = None
CORNERS = None
CIRC = None


def setup(orders):
    global ORDERS, CORNERS, CIRC
    ORDERS = tuple(orders)
    CORNERS = [reuleaux_corners(n, 1.0) for n in ORDERS]
    CIRC = [float(np.max(np.hypot(V[:, 0], V[:, 1]))) for V in CORNERS]


def _init(orders):
    setup(orders)


def root():
    b = [0.0, TMAX, 0.0, TMAX]
    for n in ORDERS[1:]:
        b += [np.pi / n, np.pi / n, 0.0, TMAX, 0.0, TMAX]
    return tuple(b)


def _parts(b):
    """(centre, half-width) per axis, and the per-body (rot, tx, ty) layout."""
    return [(b[2 * i], b[2 * i + 1]) for i in range(len(b) // 2)]


def weights(b):
    """Effective length of each axis, for choosing the split direction."""
    p = _parts(b)
    w = [p[0][1], p[1][1]]                       # body 1 translation
    for i in range(1, len(ORDERS)):
        o = 2 + 3 * (i - 1)
        w += [CIRC[i] * p[o][1], p[o + 1][1], p[o + 2][1]]
    return w


def box_bound(b):
    p = _parts(b)
    bodies = [DISK]
    ap = 0.0

    # body 1: translation only, orientation fixed by the gauge
    cx, hx = p[0]
    cy, hy = p[1]
    ap = max(ap, f_apriori(np.hypot(max(abs(cx) - hx, 0.0), max(abs(cy) - hy, 0.0))))
    E = erode_reuleaux(CORNERS[0] + np.array([cx, cy]), 1.0, np.hypot(hx, hy))
    if E is None:
        return ap
    bodies.append(E)

    for i in range(1, len(ORDERS)):
        o = 2 + 3 * (i - 1)
        cr, hr = p[o]
        cx, hx = p[o + 1]
        cy, hy = p[o + 2]
        ap = max(ap, f_apriori(np.hypot(max(abs(cx) - hx, 0.0), max(abs(cy) - hy, 0.0))))
        d = np.hypot(hx, hy) + 2.0 * CIRC[i] * np.sin(0.5 * hr)
        ca, sa = np.cos(cr), np.sin(cr)
        V = CORNERS[i] @ np.array([[ca, sa], [-sa, ca]]) + np.array([cx, cy])
        E = erode_reuleaux(V, 1.0, d)
        if E is None:
            return ap
        bodies.append(E)

    return max(ap, hull_area_exact(bodies))


def split(b):
    k = int(np.argmax(weights(b)))
    out = []
    for s in (-1, 1):
        n = list(b)
        n[2 * k] = b[2 * k] + s * b[2 * k + 1] * 0.5
        n[2 * k + 1] = b[2 * k + 1] * 0.5
        out.append(tuple(n))
    return out


def task(job):
    """DFS a subtree under a box cap; return the unfinished stack for re-queueing."""
    box, target, hmin, cap = job
    stack = [box]
    n = 0
    fails = []
    while stack:
        if n >= cap:
            return n, fails, stack
        b = stack.pop()
        n += 1
        if box_bound(b) >= target:
            continue
        if max(weights(b)) < hmin:
            fails.append(b)
            if len(fails) >= 40:
                return n, fails, stack
            continue
        stack.extend(split(b))
    return n, fails, []


def seeds(depth):
    boxes = [root()]
    for _ in range(depth):
        boxes = [c for b in boxes for c in split(b)]
    return boxes


def certify(orders, target, hmin=1e-5, cap=120_000, depth=8, nproc=8,
            budget=None, verbose=True):
    setup(orders)
    queue = seeds(depth)
    total, all_fails, rnd = 0, [], 0
    t0 = time.time()
    with mp.Pool(nproc, initializer=_init, initargs=(orders,)) as pool:
        while queue:
            rnd += 1
            nxt = []
            jobs = [(b, target, hmin, cap) for b in queue]
            for n, fails, left in pool.imap_unordered(task, jobs, chunksize=1):
                total += n
                all_fails += fails
                nxt += left
            el = time.time() - t0
            if verbose:
                print(f"  round {rnd:>3}: {len(queue):>7,} tasks -> "
                      f"{total:>14,} boxes cumulative, {len(nxt):>7,} requeued, "
                      f"{len(all_fails):>4} stuck   [{el/60:6.1f} min, "
                      f"{total/max(el,1e-9):>8,.0f} box/s]")
                sys.stdout.flush()
            if all_fails:
                return dict(ok=False, boxes=total, stuck=len(all_fails),
                            secs=el, rounds=rnd, reason="min width")
            if budget and total > budget:
                return dict(ok=False, boxes=total, stuck=0, secs=el, rounds=rnd,
                            reason=f"budget {budget:,} exceeded, {len(nxt):,} tasks left")
            queue = nxt
    return dict(ok=True, boxes=total, stuck=0, secs=time.time() - t0, rounds=rnd,
                reason="closed")


if __name__ == "__main__":
    orders = tuple(int(c) for c in sys.argv[1])          # e.g. "35" or "357"
    target = float(sys.argv[2])
    hmin = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5
    nproc = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    budget = int(float(sys.argv[5])) if len(sys.argv) > 5 else None
    depth = int(sys.argv[6]) if len(sys.argv) > 6 else 8

    setup(orders)
    dim = 2 + 3 * (len(orders) - 1)
    print("=" * 104)
    print(f"CERTIFY  a >= {target}   family = disk + " +
          " + ".join(f"Reuleaux{n}" for n in orders))
    print(f"  dimension {dim} | rho ranges " +
          ", ".join(f"[0,2pi/{n})" for n in orders[1:]) +
          f" | |t| <= {TMAX} (a priori f = {f_apriori(TMAX):.9f})")
    print(f"  hmin {hmin:g} | {nproc} procs | budget {budget if budget else 'none'}")
    print("=" * 104)
    r = certify(orders, target, hmin=hmin, nproc=nproc, budget=budget, depth=depth)
    print("\n" + "=" * 104)
    if r["ok"]:
        print(f"  CERTIFIED (double precision):  a >= {target}")
        print(f"  {r['boxes']:,} boxes, {r['rounds']} rounds, {r['secs']/60:.1f} min")
    else:
        print(f"  NOT closed: {r['reason']}  ({r['boxes']:,} boxes, {r['secs']/60:.1f} min)")
    print("=" * 104)
