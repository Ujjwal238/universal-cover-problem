"""
FAMILY A: independent verification of the published record  a >= 0.833.

Brass-Sharifi (2005) and Xie (arXiv 2606.04458, 2026) both bound Lebesgue's
constant using the three test sets

    disk of diameter 1,  equilateral triangle of side 1,  regular pentagon of
    diameter 1

proving a >= 0.832 and a >= 0.833 respectively.  This file certifies the same
family with completely different machinery, to confirm or refute the record.

Doing so also brackets that framework's ceiling from below.  I measured its
optimum numerically as LB3 = 0.833597388099, which is an UPPER bound on the true
minimum (a search for a minimum bounds it from above).  A certificate at 0.833
would give LB3 in [0.833, 0.8335974].  If the certificate FAILS, then either
LB3 < 0.833 -- so the published record is wrong -- or my machinery is.  Either
outcome is worth more than another decimal on my own bound.

GEOMETRY.  These test sets are POLYGONS, so the erosion differs from the Reuleaux
case.  Erosion distributes over intersection and a convex polygon is an
intersection of half-planes, so P (-) delta*B moves every edge inward by delta.
For a REGULAR polygon all edges are equidistant from the centre, hence

    P (-) delta*B  =  regular n-gon, same centre and rotation, inradius r - delta

i.e. a uniform scaling by (r-delta)/r, and EMPTY exactly when delta >= r.

That emptiness threshold is much tighter than for Reuleaux bodies: the
equilateral triangle of side 1 has inradius 1/(2*sqrt3) = 0.2887 against 0.4227
for the Reuleaux triangle.  Family A's cores therefore vanish at smaller boxes,
which is precisely the regime where a fabricated non-empty core would go
unnoticed -- so the audit below sweeps delta over its whole range.

WHY THE CORE IS VALID.  For a rigid motion differing from the box centre by
rotation alpha and translation t, every point p moves by at most
2|p|sin(alpha/2) + |t| <= 2R sin(alpha/2) + |t| with R the circumradius (the
farthest point of a convex polygon is a vertex).  With delta that bound,
g0.P is inside gP (+) delta*B, and (X (+) delta*B) (-) delta*B = X for convex X,
so g0.P (-) delta*B is inside gP for every placement in the box.
"""

import numpy as np
from scipy.spatial import ConvexHull

from geom import (TWO_PI, disk, hull_area_exact, regular_polygon,
                  regular_polygon_diameter)

DISK = disk(0.5)

# equilateral triangle of side 1: circumradius R = 1/sqrt3, side = R*sqrt3 = 1
R3 = 1.0 / np.sqrt(3.0)
# regular pentagon of DIAMETER 1: longest diagonal = 2R sin(2pi/5) = 1
R5 = 1.0 / (2.0 * np.sin(TWO_PI / 5.0))

SPEC = ((3, R3), (5, R5))
INRAD = tuple(R * np.cos(np.pi / n) for n, R in SPEC)
TMAX = 0.70


def f_apriori(d):
    """area of conv(disk of radius 1/2, a point at distance d) -- exact."""
    if d <= 0.5:
        return np.pi / 4
    return 0.25 * (np.pi - np.arccos(0.5 / d)) + 0.5 * np.sqrt(d * d - 0.25)


APRIORI = f_apriori(TMAX)


def erode_regpoly(n, R, rotation, center, delta):
    """(regular n-gon, circumradius R) eroded by delta*disk.  None iff empty.

    Every edge sits at distance r = R cos(pi/n) from the centre, so eroding moves
    them all inward equally: the result is the same regular polygon with inradius
    r - delta.  Empty exactly when delta >= r.
    """
    r = R * np.cos(np.pi / n)
    if delta >= r - 1e-15:
        return None
    return regular_polygon(n, R * (r - delta) / r, rotation, center)


def erode_vertices(n, R, rotation, center, delta):
    """Vertices of the eroded polygon -- the independent representation."""
    r = R * np.cos(np.pi / n)
    if delta >= r - 1e-15:
        return None
    Rp = R * (r - delta) / r
    a = TWO_PI * np.arange(n) / n + rotation
    return np.stack([center[0] + Rp * np.cos(a), center[1] + Rp * np.sin(a)], 1)


# ---------------------------------------------------------------------------
# the bound, two ways
# ---------------------------------------------------------------------------

def _deltas(b):
    c3x, h3x, c3y, h3y, cr, hr, c5x, h5x, c5y, h5y = b
    d3 = np.hypot(h3x, h3y)
    d5 = np.hypot(h5x, h5y) + 2.0 * R5 * np.sin(0.5 * hr)
    return d3, d5


def _apriori_term(b):
    c3x, h3x, c3y, h3y, cr, hr, c5x, h5x, c5y, h5y = b
    return max(f_apriori(np.hypot(max(abs(c3x) - h3x, 0.0), max(abs(c3y) - h3y, 0.0))),
               f_apriori(np.hypot(max(abs(c5x) - h5x, 0.0), max(abs(c5y) - h5y, 0.0))))


def box_bound_A(b):
    """Lower bound on min A(v) over the box, via the support-function envelope."""
    c3x, h3x, c3y, h3y, cr, hr, c5x, h5x, c5y, h5y = b
    ap = _apriori_term(b)
    d3, d5 = _deltas(b)
    E3 = erode_regpoly(3, R3, 0.0, (c3x, c3y), d3)
    E5 = erode_regpoly(5, R5, cr, (c5x, c5y), d5)
    if E3 is None or E5 is None:
        return ap
    return max(ap, hull_area_exact([DISK, E3, E5]))


def box_bound_A_indep(b, m=4096):
    """Same bound by a wholly different route: vertices + scipy convex hull.

    The cores are POLYGONS, so their hull contribution is exact from the vertices
    -- no sampling error at all.  Only the disk's circular boundary is sampled,
    with deficit O(m^-2) (~3e-7 at m=4096), so this is a very sharp cross-check.
    """
    c3x, h3x, c3y, h3y, cr, hr, c5x, h5x, c5y, h5y = b
    ap = _apriori_term(b)
    d3, d5 = _deltas(b)
    V3 = erode_vertices(3, R3, 0.0, (c3x, c3y), d3)
    V5 = erode_vertices(5, R5, cr, (c5x, c5y), d5)
    if V3 is None or V5 is None:
        return ap
    a = np.linspace(0.0, TWO_PI, m, endpoint=False)
    ring = np.stack([0.5 * np.cos(a), 0.5 * np.sin(a)], 1)
    return max(ap, ConvexHull(np.vstack([ring, V3, V5])).volume)


def true_area_A(v):
    """Exact hull area at a single placement (no erosion involved)."""
    return hull_area_exact([DISK,
                            regular_polygon(3, R3, 0.0, (v[0], v[1])),
                            regular_polygon(5, R5, v[2], (v[3], v[4]))])


def true_area_A_indep(v, m=8192):
    a = np.linspace(0.0, TWO_PI, m, endpoint=False)
    ring = np.stack([0.5 * np.cos(a), 0.5 * np.sin(a)], 1)
    t3 = TWO_PI * np.arange(3) / 3
    t5 = TWO_PI * np.arange(5) / 5 + v[2]
    V3 = np.stack([v[0] + R3 * np.cos(t3), v[1] + R3 * np.sin(t3)], 1)
    V5 = np.stack([v[3] + R5 * np.cos(t5), v[4] + R5 * np.sin(t5)], 1)
    return ConvexHull(np.vstack([ring, V3, V5])).volume


# ---------------------------------------------------------------------------
# branch and bound (hardened: strict prune, covering children, accounting)
# ---------------------------------------------------------------------------

EPS = 1e-9
INFLATE = 1e-11


def weights(b):
    return [b[1], b[3], R5 * b[5], b[7], b[9]]


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


def root():
    return (0.0, TMAX, 0.0, TMAX, np.pi / 5, np.pi / 5, 0.0, TMAX, 0.0, TMAX)


def seeds(depth):
    boxes = [root()]
    for _ in range(depth):
        boxes = [c for b in boxes for c in split_cover(b)]
    return boxes


def task(job):
    box, target, hmin, cap = job
    stack = [box]
    n, fails, tight = 0, [], np.inf
    while stack:
        if n >= cap:
            return n, fails, stack, tight
        b = stack.pop()
        n += 1
        lb = box_bound_A(b)
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


def certify_A(target, hmin=1e-5, cap=120_000, depth=8, nproc=8, budget=None,
              verbose=True):
    import multiprocessing as mp
    import sys
    import time
    queue = seeds(depth)
    total, all_fails, rnd, tight, lost = 0, [], 0, np.inf, 0
    t0 = time.time()
    with mp.Pool(nproc) as pool:
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
                break
            el = time.time() - t0
            if verbose:
                print(f"  round {rnd:>3}: {len(jobs):>7,} tasks -> {total:>14,} boxes, "
                      f"{len(nxt):>7,} requeued, {len(all_fails):>4} stuck, "
                      f"tightest +{tight:.2e}   [{el/60:6.1f} min, "
                      f"{total/max(el,1e-9):>8,.0f} box/s]")
                sys.stdout.flush()
            if all_fails:
                break
            queue = nxt
    ok = (not all_fails) and (not lost) and (not queue)
    return dict(ok=ok, boxes=total, rounds=rnd, secs=time.time() - t0,
                stuck=len(all_fails), lost=lost, tight=tight)
