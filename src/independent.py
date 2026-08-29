"""
PHASE 2: an INDEPENDENT reimplementation of the box bound, to catch code bugs.

The realistic failure mode for the a >= 0.8344 certificate is not floating point
(there are ~11 orders of margin) but a shared assumption running through every
test of a single implementation.  Findings from this route are acted on; each is
silent, plausible, and caught only by a quantity computed a DIFFERENT way.

So this file recomputes the bound along a route that shares no code with
geom.py: no support functions, no upper envelopes, no arc bookkeeping.

THE CORE, ELEMENTARY.  A width-w Reuleaux polygon on corners V is exactly

    R(V) = intersection_j D(V_j, w).

If every corner moves by at most delta, then for any p in
intersection_j D(V_j^c, w - delta),

    |p - V_j^p| <= |p - V_j^c| + |V_j^c - V_j^p| <= (w - delta) + delta = w,

so p lies in R(V^p).  Triangle inequality only -- no Minkowski algebra, no
Hausdorff distance.  Note this needs delta to bound the displacement of the
CORNERS alone, which is exactly what the certifier computes.

THE CORE'S BOUNDARY, EXACTLY.  Along a ray c + t*u, the point stays in D(V, rho)
while t^2 + 2t<d,u> + |d|^2 - rho^2 <= 0 with d = c - V, so

    t_max(V) = -<d,u> + sqrt(<d,u>^2 - |d|^2 + rho^2),

and the core's boundary in direction u is at t = min_j t_max(V_j).  Closed form,
straight from the definition.

THE AREA, INDEPENDENTLY.  Take the convex hull (scipy) of exact boundary points
of the disk and of the two cores.  The hull of a subset of a convex body has area
at most the body's, so this is a LOWER bound on
area(conv(D u core2 u core3)) -- the same direction the certificate needs, and it
converges upward as sampling refines.
"""

import numpy as np
from scipy.spatial import ConvexHull

import certgen
from certgen import TMAX, box_bound, f_apriori, setup, split

setup((3, 5))
CORNERS, CIRC, ORDERS = certgen.CORNERS, certgen.CIRC, certgen.ORDERS
W = 1.0


def core_boundary(V, rho, u):
    """Exact boundary points of intersection_j D(V_j, rho) along rays from the centroid."""
    c = V.mean(axis=0)
    d = c[None, :] - V                                  # (n,2)
    dot = d @ u.T                                       # (n, m)
    disc = dot ** 2 - (d ** 2).sum(1)[:, None] + rho ** 2
    if np.any(disc < 0):
        return None
    t = (-dot + np.sqrt(disc)).min(axis=0)              # (m,)
    if np.any(t <= 0):
        return None
    return c[None, :] + t[:, None] * u


def placed_corners(base, rho_ang, tx, ty):
    ca, sa = np.cos(rho_ang), np.sin(rho_ang)
    return base @ np.array([[ca, sa], [-sa, ca]]) + np.array([tx, ty])


def box_bound_indep(b, m=1400):
    """Independent lower bound on min A(v) over the box. Shares no code with geom.py."""
    ang = np.linspace(0.0, 2 * np.pi, m, endpoint=False)
    u = np.stack([np.cos(ang), np.sin(ang)], 1)

    c3x, h3x, c3y, h3y, cr, hr, c5x, h5x, c5y, h5y = b

    # a priori term, recomputed from the same closed form (checked separately)
    d3 = np.hypot(max(abs(c3x) - h3x, 0.0), max(abs(c3y) - h3y, 0.0))
    d5 = np.hypot(max(abs(c5x) - h5x, 0.0), max(abs(c5y) - h5y, 0.0))
    ap = max(f_apriori(d3), f_apriori(d5))

    delta3 = np.hypot(h3x, h3y)
    delta5 = np.hypot(h5x, h5y) + 2.0 * CIRC[1] * np.sin(0.5 * hr)
    if delta3 >= W or delta5 >= W:
        return ap

    V3 = CORNERS[0] + np.array([c3x, c3y])
    V5 = placed_corners(CORNERS[1], cr, c5x, c5y)
    P3 = core_boundary(V3, W - delta3, u)
    P5 = core_boundary(V5, W - delta5, u)
    if P3 is None or P5 is None:
        return ap

    pts = np.vstack([0.5 * u, P3, P5])
    return max(ap, ConvexHull(pts).volume)


def verify_core_lemma(b, npts=60, nplace=40, rng=None):
    """Check the triangle-inequality lemma numerically, with no support functions.

    Every sampled core point must lie within w of EVERY corner of EVERY placement
    in the box.  Returns the worst violation (<= 0 means sound).
    """
    rng = rng or np.random.default_rng(0)
    c3x, h3x, c3y, h3y, cr, hr, c5x, h5x, c5y, h5y = b
    worst = -np.inf
    for base, (cx, hx, cy, hy), rot, hrot, circ in (
            (CORNERS[0], (c3x, h3x, c3y, h3y), 0.0, 0.0, CIRC[0]),
            (CORNERS[1], (c5x, h5x, c5y, h5y), cr, hr, CIRC[1])):
        delta = np.hypot(hx, hy) + 2.0 * circ * np.sin(0.5 * hrot)
        if delta >= W:
            continue
        Vc = placed_corners(base, rot, cx, cy)
        ang = rng.uniform(0, 2 * np.pi, npts)
        u = np.stack([np.cos(ang), np.sin(ang)], 1)
        P = core_boundary(Vc, W - delta, u)
        if P is None:
            continue
        for _ in range(nplace):
            Vp = placed_corners(base, rot + rng.uniform(-hrot, hrot),
                                cx + rng.uniform(-hx, hx), cy + rng.uniform(-hy, hy))
            dist = np.sqrt(((P[:, None, :] - Vp[None, :, :]) ** 2).sum(-1))
            worst = max(worst, float(dist.max() - W))
    return worst


def sample_boxes(n, rng, opt=(0.006078, -0.010527, 0.83781, -0.0115, 0.019916)):
    """Boxes drawn by random DFS descent from the root, as the real search visits them."""
    root = (0.0, TMAX, 0.0, TMAX, np.pi / 5, np.pi / 5, 0.0, TMAX, 0.0, TMAX)
    out = []
    while len(out) < n:
        b = root
        depth = int(rng.integers(0, 62))
        for _ in range(depth):
            ch = split(b)
            if rng.random() < 0.75:                      # bias toward the optimum
                k = [i for i in range(5) if ch[0][2 * i + 1] != b[2 * i + 1]][0]
                b = min(ch, key=lambda z: abs(z[2 * k] - opt[k]))
            else:
                b = ch[int(rng.integers(0, 2))]
        out.append(b)
    return out


if __name__ == "__main__":
    import sys
    import time

    rng = np.random.default_rng(0)
    t0 = time.time()
    print("=" * 110)
    print("PHASE 2: independent reimplementation vs the certifier's box_bound")
    print("=" * 110)

    # --- 1. does the independent bound agree with box_bound? -----------------
    print("\n[1] box_bound vs box_bound_indep on boxes from a realistic DFS descent")
    print("    (indep converges from BELOW as sampling refines, so it must not exceed)")
    boxes = sample_boxes(3000, rng)
    over, worst_over, rel = 0, 0.0, []
    for b in boxes:
        a = box_bound(b)
        c = box_bound_indep(b)
        if c > a + 5e-7:
            over += 1
            worst_over = max(worst_over, c - a)
        rel.append(a - c)
    rel = np.array(rel)
    print(f"    {len(boxes)} boxes: indep exceeded box_bound {over} times "
          f"(worst {worst_over:.2e})")
    print(f"    box_bound - indep : min {rel.min():.3e}  median {np.median(rel):.3e}  "
          f"max {rel.max():.3e}")
    ok1 = over == 0
    print(f"    -> {'AGREE (no independent evidence of over-estimation)' if ok1 else '*** DISAGREE'}")
    sys.stdout.flush()

    # --- 2. convergence: indep -> box_bound as sampling refines --------------
    print("\n[2] refinement: indep should climb toward box_bound, never past it")
    sub = boxes[:60]
    for m in (200, 700, 2500, 9000):
        d = [box_bound(b) - box_bound_indep(b, m=m) for b in sub]
        print(f"    m={m:>5}: mean gap {np.mean(d):.3e}   max gap {np.max(d):.3e}   "
              f"min gap {np.min(d):.3e}")
        sys.stdout.flush()

    # --- 3. the core lemma, by triangle inequality only ---------------------
    print("\n[3] core lemma: every core point within w of every corner of every placement")
    worst = -np.inf
    for b in sample_boxes(400, rng):
        worst = max(worst, verify_core_lemma(b, rng=rng))
    ok3 = worst <= 1e-12
    print(f"    400 boxes x 40 placements x 60 points: worst (|p-V| - w) = {worst:.3e}")
    print(f"    -> {'SOUND' if ok3 else '*** VIOLATED'}")

    # --- 4. a priori lemma, independent geometry ---------------------------
    print("\n[4] a priori f(d) via the independent hull")
    ang = np.linspace(0, 2 * np.pi, 20000, endpoint=False)
    u = np.stack([np.cos(ang), np.sin(ang)], 1)
    ok4 = True
    for d in (0.55, 0.6, 0.7, 0.8, 0.99):
        pts = np.vstack([0.5 * u, [[d, 0.0]]])
        got = ConvexHull(pts).volume
        err = abs(got - f_apriori(d))
        ok4 &= err < 1e-7
        print(f"    d={d}: indep {got:.12f}  closed form {f_apriori(d):.12f}  err {err:.2e}")

    print("\n" + "=" * 110)
    allok = ok1 and ok3 and ok4
    print(f"  {'ALL INDEPENDENT CHECKS PASS' if allok else '*** INDEPENDENT CHECKS FAILED'}"
          f"   ({time.time()-t0:.0f}s)")
    print("=" * 110)
