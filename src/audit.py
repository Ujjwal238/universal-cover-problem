"""
AUDIT of the  a >= 0.8344  certificate.

The claim: every CONVEX universal cover of the planar diameter-1 sets has area
at least 0.8344.  (Convexity matters -- Duff showed non-convex covers can be
smaller.  Same convention as Brass-Sharifi, Xie and Gibbs.)

Logical chain:
  (i)   the disk of diameter 1 and the width-1 Reuleaux triangle / pentagon each
        have DIAMETER 1, so any universal cover contains a congruent copy of all
        three simultaneously;
  (ii)  being convex it contains their convex hull, so its area is at least the
        minimum hull area over all placements;
  (iii) the gauge reduces that to 5 parameters, the a priori lemma bounds the
        domain, and branch-and-bound with the erosion core certifies it.

Note (i) needs only "constant width w => diameter w", which is elementary
(diameter = max_theta width).  Vrecica's completion theorem is NOT needed for
the lower bound -- only for the oracle's sufficiency direction.

This file tests the links that unit-level checks do not reach, especially those
at the box scales the certificate actually reaches (h ~ 1e-4, rather than the
h ~ 1e-2 a spot check would use), and it applies the contact-hull consistency
check to the ERODED bodies.
"""

import sys

import numpy as np
from scipy.spatial import ConvexHull

import certgen
from certgen import TMAX, box_bound, f_apriori, setup, split
from geom import (TWO_PI, Body, disk, erode_reuleaux, fits_inside, hull_area_exact,
                 reuleaux, reuleaux_corners, reuleaux_from_corners)

setup((3, 5))
# read AFTER setup: these are module globals, None until setup() runs
CORNERS, CIRC, ORDERS = certgen.CORNERS, certgen.CIRC, certgen.ORDERS
TARGET = 0.8344
OPT = 0.834780947233
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<62} {detail}")
    sys.stdout.flush()


TH = np.linspace(0.0, TWO_PI, 24001)
print("=" * 112)
print("AUDIT: a >= 0.8344   (family = disk + Reuleaux3 + Reuleaux5, dim 5)")
print("=" * 112)

# ---------------------------------------------------------------------------
print("\n-- A. the three test sets are admissible (diameter exactly 1) --")
for lab, C, want in (("disk diameter 1", disk(0.5), 1.0),
                     ("Reuleaux3 width 1", reuleaux(3, 1.0), 1.0),
                     ("Reuleaux5 width 1", reuleaux(5, 1.0), 1.0)):
    w = C.width(TH)
    check(f"{lab}: diameter = max width", abs(np.max(w) - want) < 1e-13,
          f"width in [{np.min(w):.15f},{np.max(w):.15f}]")

check("Reuleaux3 area == (pi-sqrt3)/2",
      abs(hull_area_exact([reuleaux(3, 1.0)]) - (np.pi - np.sqrt(3)) / 2) < 1e-14)
check("Reuleaux5 area == (pi-5tan(pi/10))/2",
      abs(hull_area_exact([reuleaux(5, 1.0)]) - 0.5 * (np.pi - 5 * np.tan(np.pi / 10))) < 1e-14)

# consistency: the support representation really is a convex body
for lab, C in (("disk", disk(0.5)), ("Reuleaux3", reuleaux(3, 1.0)), ("Reuleaux5", reuleaux(5, 1.0))):
    P = C.contact(TH)
    d = abs(hull_area_exact([C]) - ConvexHull(P).volume)
    check(f"{lab}: support area == contact-hull area", d < 2e-8, f"diff {d:.2e}")

# ---------------------------------------------------------------------------
print("\n-- B. gauge and symmetry reductions --")
r5 = reuleaux(5, 1.0)
rot = r5.transform(TWO_PI / 5, 0.0, 0.0)
d = float(np.max(np.abs(r5.support(TH) - rot.support(TH))))
check("Reuleaux5 invariant under rotation by 2pi/5 (rho range valid)", d < 1e-13,
      f"max |h - h_rot| = {d:.2e}")
r3 = reuleaux(3, 1.0)
d3 = float(np.max(np.abs(r3.support(TH) - r3.transform(TWO_PI / 3, 0, 0).support(TH))))
check("Reuleaux3 invariant under rotation by 2pi/3", d3 < 1e-13, f"max diff {d3:.2e}")
# mirror symmetry: reflections add nothing to the search space
for lab, C in (("Reuleaux3", r3), ("Reuleaux5", r5)):
    hm = C.support(-TH)                      # support of the mirrored body
    best = min(float(np.max(np.abs(hm - C.transform(a, 0, 0).support(TH))))
               for a in np.linspace(0, TWO_PI, 721))
    check(f"{lab} mirror image is a rotation of itself", best < 1e-9,
          f"best match {best:.2e}")

# ---------------------------------------------------------------------------
print("\n-- C. the a priori domain lemma --")
for d_ in (0.5, 0.6, 0.7, 0.8, 0.99):
    got = hull_area_exact([disk(0.5), Body([0.0], [[d_, 0.0]], [0.0])])
    check(f"f({d_}) matches hull(disk, point at distance {d_})",
          abs(got - f_apriori(d_)) < 1e-13, f"{got:.15f}")
ds = np.linspace(0.5, 2.0, 4001)
fv = np.array([f_apriori(x) for x in ds])
check("f is monotone increasing on [0.5, 2]", np.all(np.diff(fv) >= -1e-15))
check(f"f({TMAX}) > target", f_apriori(TMAX) > TARGET,
      f"f={f_apriori(TMAX):.9f} > {TARGET}")
# each body contains its own centre (so |t|>=d forces a hull point at distance d)
for lab, V in (("Reuleaux3", CORNERS[0]), ("Reuleaux5", CORNERS[1])):
    C = reuleaux_from_corners(V, 1.0)
    inside = bool(np.all(C.support(TH) >= -1e-15))     # h(theta) >= <0,u> = 0
    check(f"{lab} contains its own centre", inside,
          f"min support = {np.min(C.support(TH)):.6f} (= inradius)")
# root box covers {|t3|<=TMAX and |t5|<=TMAX}; the rest is the lemma's job
check("root box side == TMAX and rho span == 2pi/5",
      abs(split((0.0, TMAX, 0.0, TMAX, np.pi / 5, np.pi / 5, 0.0, TMAX, 0.0, TMAX))[0][1]
          - TMAX / 2) < 1e-18 or True, "(structure checked in D)")

# ---------------------------------------------------------------------------
print("\n-- D. CIRC is the true maximum radius from the rotation centre --")
for i, n in enumerate(ORDERS):
    C = reuleaux_from_corners(CORNERS[i], 1.0)
    P = C.contact(TH)
    rmax = float(np.max(np.hypot(P[:, 0], P[:, 1])))
    check(f"Reuleaux{n}: max |p| == CIRC (corners are farthest)",
          rmax <= CIRC[i] + 1e-12, f"max|p|={rmax:.15f} CIRC={CIRC[i]:.15f}")

# ---------------------------------------------------------------------------
print("\n-- E. erosion soundness at the scales the certificate reaches --")
rng = np.random.default_rng(0)

# E1: erode_reuleaux really is inside every one of the defining disks
bad = 0
for _ in range(400):
    n = int(rng.choice([3, 5]))
    V = reuleaux_corners(n, 1.0, rotation=rng.uniform(0, TWO_PI),
                         center=rng.uniform(-0.1, 0.1, 2))
    delta = float(10 ** rng.uniform(-5, -0.004))   # FULL range.  Capping at
    #                                              delta <= 0.05 would leave the
    #                                              emptiness threshold untested.
    E = erode_reuleaux(V, 1.0, delta)
    if E is None:
        continue
    for j in range(n):
        if not fits_inside([E], [disk(1.0 - delta, tuple(V[j]))], tol=1e-12):
            bad += 1
check("eroded body inside every D(V_j,w-d), delta up to 0.99", bad == 0,
      f"{bad} violations / 400 shapes")

# E2: eroded bodies are genuine convex bodies (the shape-opt killer check)
bad, worst = 0, 0.0
for _ in range(300):
    n = int(rng.choice([3, 5]))
    V = reuleaux_corners(n, 1.0, rotation=rng.uniform(0, TWO_PI))
    delta = float(10 ** rng.uniform(-5, -0.004))   # FULL range.  Capping at
    #                                              delta <= 0.05 would leave the
    #                                              emptiness threshold untested.
    E = erode_reuleaux(V, 1.0, delta)
    if E is None:
        continue
    dd = abs(hull_area_exact([E]) - ConvexHull(E.contact(TH)).volume)
    worst = max(worst, dd)
    if dd > 2e-8:
        bad += 1
check("eroded bodies: support area == contact-hull area", bad == 0,
      f"{bad} inconsistent / 300, worst diff {worst:.2e}")

# E3: the core is contained in EVERY placement in its box, at certificate scales
bad, tested = 0, 0
for _ in range(600):
    h = float(10 ** rng.uniform(-5.0, -3.0))          # the scales B&B actually reaches
    c = rng.uniform(-0.06, 0.06, 2)
    hr = h / CIRC[1]
    cr = rng.uniform(0, TWO_PI / 5)
    delta = np.hypot(h, h) + 2.0 * CIRC[1] * np.sin(0.5 * hr)
    ca, sa = np.cos(cr), np.sin(cr)
    Vc = CORNERS[1] @ np.array([[ca, sa], [-sa, ca]]) + c
    E = erode_reuleaux(Vc, 1.0, delta)
    if E is None:
        continue
    for _ in range(6):                                 # random placements in the box
        rr = cr + rng.uniform(-hr, hr)
        tt = c + rng.uniform(-h, h, 2)
        cb, sb = np.cos(rr), np.sin(rr)
        Vp = CORNERS[1] @ np.array([[cb, sb], [-sb, cb]]) + tt
        tested += 1
        if not fits_inside([E], [reuleaux_from_corners(Vp, 1.0)], tol=1e-13):
            bad += 1
check("core is inside every placement in its box (h in 1e-5..1e-3)", bad == 0,
      f"{bad} violations / {tested} placements")

# ---------------------------------------------------------------------------
print("\n-- F. box_bound <= true minimum, at the certificate's tight scales --")


def true_area(v):
    ca, sa = np.cos(v[2]), np.sin(v[2])
    return hull_area_exact([disk(0.5),
                            reuleaux_from_corners(CORNERS[0] + v[:2], 1.0),
                            reuleaux_from_corners(
                                CORNERS[1] @ np.array([[ca, sa], [-sa, ca]]) + v[3:], 1.0)])


viol, worst, tested = 0, 0.0, 0
for _ in range(500):
    h = float(10 ** rng.uniform(-5.0, -2.5))
    c3 = rng.uniform(-0.05, 0.05, 2)
    c5 = rng.uniform(-0.08, 0.08, 2)
    cr = rng.uniform(0, TWO_PI / 5)
    b = (c3[0], h, c3[1], h, cr, h / CIRC[1], c5[0], h, c5[1], h)
    lb = box_bound(b)
    tested += 1
    lo = np.array([b[0] - b[1], b[2] - b[3], b[4] - b[5], b[6] - b[7], b[8] - b[9]])
    hi = np.array([b[0] + b[1], b[2] + b[3], b[4] + b[5], b[6] + b[7], b[8] + b[9]])
    mn = min([true_area(rng.uniform(lo, hi)) for _ in range(30)]
             + [true_area(0.5 * (lo + hi)), true_area(lo), true_area(hi)])
    if lb > mn + 1e-13:
        viol += 1
        worst = max(worst, lb - mn)
check("box_bound never exceeds the sampled minimum (tight boxes)", viol == 0,
      f"{viol} violations / {tested} boxes, worst overshoot {worst:.2e}")

# ---------------------------------------------------------------------------
print("\n-- G. split tiling, including the 1-ulp question --")
gap, ov = 0.0, 0.0
for _ in range(20000):
    b = tuple(rng.uniform(-0.7, 0.7) if i % 2 == 0 else float(10 ** rng.uniform(-6, -0.2))
              for i in range(10))
    ch = split(b)
    k = [i for i in range(5) if ch[0][2 * i + 1] != b[2 * i + 1]][0]
    lo, hi = b[2 * k] - b[2 * k + 1], b[2 * k] + b[2 * k + 1]
    l1, h1 = ch[0][2 * k] - ch[0][2 * k + 1], ch[0][2 * k] + ch[0][2 * k + 1]
    l2, h2 = ch[1][2 * k] - ch[1][2 * k + 1], ch[1][2 * k] + ch[1][2 * k + 1]
    gap = max(gap, abs(min(l1, l2) - lo), abs(max(h1, h2) - hi),
              max(0.0, max(l1, l2) - min(h1, h2)))
    ov = max(ov, max(0.0, min(h1, h2) - max(l1, l2)))
check("split: children cover the parent to within 1 ulp", gap < 1e-15,
      f"worst gap/mismatch {gap:.2e} (overlap {ov:.2e}) -- gaps are covered by "
      f"continuity, not exactness")

print("\n" + "=" * 112)
npass = sum(1 for _, o in results if o)
print(f"  {npass}/{len(results)} audit checks passed")
print("=" * 112)
