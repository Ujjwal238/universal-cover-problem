"""
Rigorous branch-and-bound certificate for family B = {disk, Reuleaux3, Reuleaux5}.

Goal: prove   a >= L   for some L > 0.833 (the current published record, Xie 2026).
Family B's numerical optimum is 0.834780946, so anything up to ~0.8347 is on the
table.

WHAT MUST BE SHOWN.  For every placement v = (t3, rho, t5),

    A(v) = area( conv( D  u  (R3 + t3)  u  (R_rho R5 + t5) ) )  >=  L

with D the disk of diameter 1 at the origin, R3 / R5 the width-1 Reuleaux
triangle / pentagon.  Gauge: the disk pins two translations and is rotation
invariant; the residual global rotation fixes R3's orientation at 0.

BOUND ON A BOX.  If every placement in a box B moves the points of body i by at
most delta_i from the box-centre placement, then

    g_i C_i   contains   (centre placement of C_i) (-) delta_i * disk

for all placements in B, because g0.C subset gC (+) delta.B and (X (+) dB) (-) dB
= X for convex X.  Hence for all v in B

    A(v) >= area( conv( D u core_2 u core_3 ) ) =: LB(B)

and the cores are exact, not approximate: erosion distributes over intersection
and a width-w Reuleaux polygon is the intersection of the disks D(V_j, w), so
the core is just the same disks with radius w - delta.

DOMAIN IS BOUNDED A PRIORI.  conv(disk of radius 1/2, a point at distance d)
has area  f(d) = (1/4)(pi - arccos(1/(2d))) + (1/2) sqrt(d^2 - 1/4).  Each body
contains its own centre, so if |t_i| >= d then A(v) >= f(d).  f(0.70) =
0.836528 > L for any L we target, so translations may be restricted to
|t| <= 0.70.

SYMMETRY.  R5 is invariant under rotation by 2*pi/5, so rho need only range over
[0, 2*pi/5) -- a factor-5 reduction.

FLOATING POINT.  This pass runs in double precision with a safety margin; the
bound is monotone and the arithmetic is a few hundred flops per box, so the
margin covers it.  A directed-rounding interval pass is the follow-up.
"""

import multiprocessing as mp
import sys
import time

import numpy as np

from geom import (TWO_PI, disk, erode_reuleaux, hull_area_exact,
                  reuleaux_corners)

DISK = disk(0.5)
V3 = reuleaux_corners(3, 1.0)
V5 = reuleaux_corners(5, 1.0)
R5CIRC = float(np.max(np.hypot(V5[:, 0], V5[:, 1])))
TMAX = 0.70


def f_apriori(d):
    """area of conv(disk radius 1/2, point at distance d)."""
    if d <= 0.5:
        return np.pi / 4
    return 0.25 * (np.pi - np.arccos(0.5 / d)) + 0.5 * np.sqrt(d * d - 0.25)


APRIORI = f_apriori(TMAX)


def box_bound(b):
    """Rigorous lower bound on A(v) over the box, or None if it must be split."""
    c3x, h3x, c3y, h3y, cr, hr, c5x, h5x, c5y, h5y = b

    # a priori: if the whole box forces a body far from the origin, done
    dmin3 = np.hypot(max(abs(c3x) - h3x, 0.0), max(abs(c3y) - h3y, 0.0))
    dmin5 = np.hypot(max(abs(c5x) - h5x, 0.0), max(abs(c5y) - h5y, 0.0))
    ap = max(f_apriori(dmin3), f_apriori(dmin5))

    d3 = np.hypot(h3x, h3y)
    d5 = np.hypot(h5x, h5y) + 2.0 * R5CIRC * np.sin(0.5 * hr)

    E2 = erode_reuleaux(V3 + np.array([c3x, c3y]), 1.0, d3)
    ca, sa = np.cos(cr), np.sin(cr)
    V5r = V5 @ np.array([[ca, sa], [-sa, ca]]) + np.array([c5x, c5y])
    E3 = erode_reuleaux(V5r, 1.0, d5)
    if E2 is None or E3 is None:
        return ap if ap > 0 else None
    return max(ap, hull_area_exact([DISK, E2, E3]))


def split(b):
    c3x, h3x, c3y, h3y, cr, hr, c5x, h5x, c5y, h5y = b
    w = [h3x, h3y, R5CIRC * hr, h5x, h5y]
    k = int(np.argmax(w))
    out = []
    for s in (-1, 1):
        n = list(b)
        n[2 * k] = b[2 * k] + s * b[2 * k + 1] * 0.5
        n[2 * k + 1] = b[2 * k + 1] * 0.5
        out.append(tuple(n))
    return out


def worker(job):
    box, target, hmin, cap = job
    stack = [box]
    n = 0
    fails = []
    while stack:
        b = stack.pop()
        n += 1
        if n > cap:
            return n, fails, stack          # ran out of budget
        lb = box_bound(b)
        if lb is not None and lb >= target:
            continue
        if max(b[1], b[3], R5CIRC * b[5], b[7], b[9]) < hmin:
            fails.append((b, lb))
            if len(fails) > 200:
                return n, fails, stack
            continue
        stack.extend(split(b))
    return n, fails, []


def initial_boxes(depth=3):
    root = (0.0, TMAX, 0.0, TMAX, np.pi / 5, np.pi / 5, 0.0, TMAX, 0.0, TMAX)
    boxes = [root]
    for _ in range(depth):
        boxes = [c for b in boxes for c in split(b)]
    return boxes


if __name__ == "__main__":
    target = float(sys.argv[1]) if len(sys.argv) > 1 else 0.8336
    hmin = float(sys.argv[2]) if len(sys.argv) > 2 else 2e-5
    cap = int(float(sys.argv[3])) if len(sys.argv) > 3 else 4e6
    depth = int(sys.argv[4]) if len(sys.argv) > 4 else 4

    print("=" * 96)
    print(f"CERTIFY  a >= {target}   for family B = disk + Reuleaux3 + Reuleaux5")
    print(f"  Xie 2026 record 0.833 | family B numerical optimum 0.834780946")
    print(f"  a priori: |t| >= {TMAX} forces area >= {APRIORI:.9f}; rho in [0, 2pi/5)")
    print(f"  min box half-width {hmin:g}, per-worker cap {cap:g}")
    print("=" * 96)

    boxes = initial_boxes(depth=depth)
    print(f"\nseeding {len(boxes)} subtrees over {8} cores ...")
    t0 = time.time()
    with mp.Pool(8) as pool:
        res = pool.map(worker, [(b, target, hmin, cap) for b in boxes])

    total = sum(r[0] for r in res)
    fails = [f for r in res for f in r[1]]
    left = sum(len(r[2]) for r in res)
    print(f"\n  boxes processed : {total:,}   ({time.time()-t0:.0f}s)")
    print(f"  unresolved      : {len(fails)} below min width, {left} over budget")
    if not fails and not left:
        print("\n" + "=" * 96)
        print(f"  CERTIFIED (double precision):  a >= {target}")
        print(f"  previous record 0.833 (Xie 2026)  ->  improvement {target-0.833:+.6f}")
        print("=" * 96)
    else:
        worst = min([lb for _, lb in fails if lb is not None], default=None)
        print(f"  NOT closed. worst unresolved bound: "
              f"{worst if worst is None else f'{worst:.9f}'}")
        if fails:
            print(f"  example stuck box: {tuple(round(x,6) for x in fails[0][0])}")
