"""
Audit of the family-A machinery, before certifying a >= 0.833.

Built around the three ways an audit can agree with itself and prove nothing:

  * sweeping delta over a narrower range than the certifier actually evaluates
    ->  delta is swept over its WHOLE range here, and the emptiness threshold
    (delta = inradius) is straddled explicitly;
  * a one-sided cross-check  ->  disagreement is flagged in BOTH directions;
  * a discrepancy printed and rationalised  ->  any unexplained gap fails the
    audit outright.
"""

import sys

import numpy as np
from scipy.spatial import ConvexHull

from familyA import (APRIORI, DISK, INRAD, R3, R5, TMAX, box_bound_A,
                     box_bound_A_indep, erode_regpoly, erode_vertices,
                     f_apriori, root, split_cover, true_area_A,
                     true_area_A_indep, weights)
from geom import TWO_PI, Body, hull_area_exact, regular_polygon

LB3 = 0.833597388099
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<64} {detail}")
    sys.stdout.flush()


TH = np.linspace(0.0, TWO_PI, 30001)
print("=" * 118)
print("AUDIT of family A = disk + equilateral triangle (side 1) + regular pentagon (diameter 1)")
print(f"  the Brass-Sharifi / Xie test sets.  measured optimum LB3 = {LB3}")
print(f"  circumradii: triangle {R3:.12f}, pentagon {R5:.12f}")
print(f"  inradii:     triangle {INRAD[0]:.12f}, pentagon {INRAD[1]:.12f}  <- erosion empties here")
print("=" * 118)

# ---- A. admissibility: every test set must have diameter exactly 1 ---------
print("\n-- A. admissibility (diameter exactly 1) --")
tri = regular_polygon(3, R3)
pen = regular_polygon(5, R5)
for lab, C in (("disk", DISK), ("equilateral triangle side 1", tri),
               ("regular pentagon diameter 1", pen)):
    w = C.width(TH)
    check(f"{lab}: diameter = max width = 1", abs(np.max(w) - 1.0) < 1e-13,
          f"max width {np.max(w):.15f}")
check("triangle side length == 1",
      abs(np.hypot(*(np.array([R3, 0]) - np.array([R3 * np.cos(TWO_PI / 3),
                                                   R3 * np.sin(TWO_PI / 3)]))) - 1.0) < 1e-14)
check("triangle area == sqrt3/4", abs(hull_area_exact([tri]) - np.sqrt(3) / 4) < 1e-14)
check("pentagon area == (5/2)R^2 sin(72deg)",
      abs(hull_area_exact([pen]) - 2.5 * R5 ** 2 * np.sin(TWO_PI / 5)) < 1e-14)

# ---- B. gauge and symmetry -------------------------------------------------
print("\n-- B. gauge and symmetry reductions --")
d = float(np.max(np.abs(pen.support(TH) - pen.transform(TWO_PI / 5, 0, 0).support(TH))))
check("pentagon invariant under 2pi/5 (rho range valid)", d < 1e-13, f"{d:.2e}")
d3 = float(np.max(np.abs(tri.support(TH) - tri.transform(TWO_PI / 3, 0, 0).support(TH))))
check("triangle invariant under 2pi/3 (orientation gauge valid)", d3 < 1e-13, f"{d3:.2e}")
for lab, C in (("triangle", tri), ("pentagon", pen)):
    best = min(float(np.max(np.abs(C.support(-TH) - C.transform(a, 0, 0).support(TH))))
               for a in np.linspace(0, TWO_PI, 1441))
    check(f"{lab} mirror image is a rotation of itself (reflections redundant)",
          best < 1e-9, f"{best:.2e}")

# ---- C. a priori domain lemma ---------------------------------------------
print("\n-- C. a priori domain lemma --")
for dd in (0.5, 0.6, 0.7, 0.99):
    got = hull_area_exact([DISK, Body([0.0], [[dd, 0.0]], [0.0])])
    check(f"f({dd}) matches hull(disk, point at distance {dd})",
          abs(got - f_apriori(dd)) < 1e-13, f"{got:.15f}")
check("f monotone increasing on [0.5,2]",
      bool(np.all(np.diff([f_apriori(x) for x in np.linspace(0.5, 2, 3001)]) >= -1e-15)))
check(f"f({TMAX}) > 0.833 (domain restriction valid)", APRIORI > 0.833,
      f"f={APRIORI:.9f}")
for lab, C in (("triangle", tri), ("pentagon", pen)):
    check(f"{lab} contains its own centre", bool(np.all(C.support(TH) >= -1e-15)),
          f"min support {np.min(C.support(TH)):.9f} = inradius")
for i, (lab, R, n) in enumerate((("triangle", R3, 3), ("pentagon", R5, 5))):
    P = regular_polygon(n, R).contact(TH)
    rmax = float(np.max(np.hypot(P[:, 0], P[:, 1])))
    check(f"{lab}: max |p| == circumradius (delta formula uses R)",
          abs(rmax - R) < 1e-12, f"max|p|={rmax:.15f} R={R:.15f}")

# ---- D. erosion over the FULL delta range ---------------------------------
print("\n-- D. erosion over the whole delta range, straddling emptiness --")
bad = 0
for n, R, r in ((3, R3, INRAD[0]), (5, R5, INRAD[1])):
    for delta in np.linspace(0.0, 0.999, 600):
        E = erode_regpoly(n, R, 0.3, (0.02, -0.01), delta)
        empty_expected = delta >= r
        if (E is None) != empty_expected:
            bad += 1
            print(f"      n={n} delta={delta:.4f}: returned {'None' if E is None else 'body'}"
                  f" but erosion {'IS' if empty_expected else 'is NOT'} empty")
            continue
        if E is None:
            continue
        # the eroded polygon must sit inside the original
        big = regular_polygon(n, R, 0.3, (0.02, -0.01))
        if float(np.max(E.support(TH) - big.support(TH))) > 1e-12:
            bad += 1
check("erosion None iff delta >= inradius, and always inside the original",
      bad == 0, f"{bad} defects / 1200 (n,delta) pairs over delta in [0,0.999]")

# every vertex of the eroded polygon must be at distance <= r-delta from each edge
bad = 0
for n, R, r in ((3, R3, INRAD[0]), (5, R5, INRAD[1])):
    for delta in np.linspace(0.0, r * 0.999, 200):
        V = erode_vertices(n, R, 0.3, (0.02, -0.01), delta)
        nrm = np.stack([np.cos(TWO_PI * np.arange(n) / n + 0.3 + np.pi / n),
                        np.sin(TWO_PI * np.arange(n) / n + 0.3 + np.pi / n)], 1)
        off = (V - np.array([0.02, -0.01])) @ nrm.T
        if float(np.max(off)) > r - delta + 1e-12:
            bad += 1
check("eroded vertices satisfy every inward-moved edge constraint", bad == 0,
      f"{bad} / 400")

# ---- E. the core lemma: core inside EVERY placement in the box ------------
print("\n-- E. core inside every placement in its box --")
rng = np.random.default_rng(0)
bad, tested = 0, 0
for _ in range(500):
    h = float(10 ** rng.uniform(-5.0, -0.8))          # up to h ~ 0.16
    c = rng.uniform(-0.3, 0.3, 2)
    hr = float(10 ** rng.uniform(-5.0, -0.4))
    cr = rng.uniform(0, TWO_PI / 5)
    for n, R in ((3, R3), (5, R5)):
        rot0 = 0.0 if n == 3 else cr
        hrot = 0.0 if n == 3 else hr
        delta = np.hypot(h, h) + 2.0 * R * np.sin(0.5 * hrot)
        E = erode_regpoly(n, R, rot0, tuple(c), delta)
        if E is None:
            continue
        for _ in range(8):
            pl = regular_polygon(n, R, rot0 + rng.uniform(-hrot, hrot),
                                 tuple(c + rng.uniform(-h, h, 2)))
            tested += 1
            if float(np.max(E.support(TH) - pl.support(TH))) > 1e-12:
                bad += 1
check("core is inside every placement (h up to 0.16, full rotation range)",
      bad == 0, f"{bad} violations / {tested} placements")

# ---- F. two-sided cross-check against the independent implementation ------
print("\n-- F. TWO-SIDED cross-check (independent: vertices + scipy hull) --")


def sample_boxes(k, rng):
    out = []
    while len(out) < k:
        b = root()
        for _ in range(int(rng.integers(0, 60))):
            ch = split_cover(b)
            b = ch[int(rng.integers(0, 2))]
        out.append(b)
    return out


boxes = sample_boxes(4000, rng)
for m in (1024, 4096):
    gap = np.array([box_bound_A(b) - box_bound_A_indep(b, m=m) for b in boxes])
    neg = int((gap < -1e-9).sum())
    big = int((gap > 1e-5).sum())
    check(f"m={m}: no disagreement in either direction", neg == 0 and big == 0,
          f"gap min {gap.min():+.2e} median {np.median(gap):+.2e} max {gap.max():+.2e}; "
          f"neg {neg}, >1e-5 {big}")

# ---- G. bound <= truth, checked with the INDEPENDENT area -----------------
print("\n-- G. box_bound <= true minimum (independent truth) --")
viol, worst, slack = 0, 0.0, []
for _ in range(300):
    h = float(10 ** rng.uniform(-5.0, -2.0))
    c3 = rng.uniform(-0.06, 0.06, 2)
    c5 = rng.uniform(-0.09, 0.09, 2)
    cr = rng.uniform(0, TWO_PI / 5)
    b = (c3[0], h, c3[1], h, cr, h / R5, c5[0], h, c5[1], h)
    lb = box_bound_A(b)
    lo = np.array([b[0] - b[1], b[2] - b[3], b[4] - b[5], b[6] - b[7], b[8] - b[9]])
    hi = np.array([b[0] + b[1], b[2] + b[3], b[4] + b[5], b[6] + b[7], b[8] + b[9]])
    mn = min([true_area_A_indep(rng.uniform(lo, hi)) for _ in range(14)]
             + [true_area_A_indep(0.5 * (lo + hi)), true_area_A_indep(lo),
                true_area_A_indep(hi)])
    slack.append(mn - lb)
    if lb > mn + 1e-12:
        viol += 1
        worst = max(worst, lb - mn)
check("box_bound never exceeds the independently computed minimum", viol == 0,
      f"{viol}/300, worst {worst:.2e}, min slack {min(slack):.3e}")

# the two area routines must agree at single placements
d = max(abs(true_area_A(v) - true_area_A_indep(v))
        for v in [rng.uniform([-0.05, -0.05, 0, -0.05, -0.05],
                              [0.05, 0.05, TWO_PI / 5, 0.05, 0.05]) for _ in range(200)])
check("exact and independent single-placement areas agree", d < 5e-8, f"max diff {d:.2e}")

# ---- H. split tiling ------------------------------------------------------
print("\n-- H. split covering --")
worstgap = 0.0
for _ in range(20000):
    b = tuple(rng.uniform(-0.7, 0.7) if i % 2 == 0 else float(10 ** rng.uniform(-6, -0.2))
              for i in range(10))
    ch = split_cover(b)
    k = [i for i in range(5) if ch[0][2 * i + 1] != b[2 * i + 1]][0]
    lo, hi = b[2 * k] - b[2 * k + 1], b[2 * k] + b[2 * k + 1]
    l1, h1 = ch[0][2 * k] - ch[0][2 * k + 1], ch[0][2 * k] + ch[0][2 * k + 1]
    l2, h2 = ch[1][2 * k] - ch[1][2 * k + 1], ch[1][2 * k] + ch[1][2 * k + 1]
    # a GAP means the children fail to reach the parent's edge, i.e.
    #   min(l1,l2) > lo   or   max(h1,h2) < hi   or   max(l1,l2) > min(h1,h2).
    # The reverse signs are over-coverage from INFLATE, which is safe.
    worstgap = max(worstgap,
                   max(0.0, min(l1, l2) - lo),
                   max(0.0, hi - max(h1, h2)),
                   max(0.0, max(l1, l2) - min(h1, h2)))
check("children provably cover the parent (no gaps)", worstgap == 0.0,
      f"worst uncovered {worstgap:.2e}")

# independent coverage test: sample points inside the parent and require each to
# land in at least one child.  Does not depend on getting any inequality's sign
# right, unlike the interval arithmetic above.
miss = 0
for _ in range(4000):
    b = tuple(rng.uniform(-0.7, 0.7) if i % 2 == 0 else float(10 ** rng.uniform(-6, -0.2))
              for i in range(10))
    ch = split_cover(b)
    lo = np.array([b[0] - b[1], b[2] - b[3], b[4] - b[5], b[6] - b[7], b[8] - b[9]])
    hi = np.array([b[0] + b[1], b[2] + b[3], b[4] + b[5], b[6] + b[7], b[8] + b[9]])
    for _ in range(12):
        p = rng.uniform(lo, hi)
        inside = False
        for c in ch:
            cc = np.array([c[0], c[2], c[4], c[6], c[8]])
            hh = np.array([c[1], c[3], c[5], c[7], c[9]])
            if np.all(np.abs(p - cc) <= hh):
                inside = True
                break
        if not inside:
            miss += 1
check("every sampled point of the parent lies in some child", miss == 0,
      f"{miss} points missed / 48000")

print("\n" + "=" * 118)
n = sum(results)
print(f"  {n}/{len(results)} audit checks passed"
      f"{'  --  CLEAR TO CERTIFY' if n == len(results) else '  --  BLOCKED'}")
print("=" * 118)
raise SystemExit(0 if n == len(results) else 1)
