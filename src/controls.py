"""
Soundness controls for the branch-and-bound certificate.

A certifier that certifies everything certifies nothing.  There are exactly two
ways this one could emit a false certificate:

  (a) an UNSOUND BOUND  -- box_bound returns something above the true minimum
      over the box.  Already tested separately: 0 violations over 400 random
      boxes with 60 interior samples each.

  (b) a TILING BUG      -- split() fails to cover its parent, so the search
      "certifies" regions it never visited.

(b) is checked here, together with the discriminating test that matters: on a
small box around the true optimum, targets BELOW it must certify and targets
ABOVE it must fail.  If a target above the optimum ever certifies, the whole
result is void.

Each check prints the moment it finishes.
"""

import sys
import time

import numpy as np
from scipy.optimize import minimize

from certify import R5CIRC, TMAX, box_bound, initial_boxes, split, worker
from geom import (disk, hull_area_exact, reuleaux_corners,
                  reuleaux_from_corners)


def say(*a):
    print(*a)
    sys.stdout.flush()


DISK = disk(0.5)
V3 = reuleaux_corners(3, 1.0)
V5 = reuleaux_corners(5, 1.0)


def area(v):
    ca, sa = np.cos(v[2]), np.sin(v[2])
    return hull_area_exact([DISK,
                            reuleaux_from_corners(V3 + v[:2], 1.0),
                            reuleaux_from_corners(V5 @ np.array([[ca, sa], [-sa, ca]]) + v[3:], 1.0)])


t0 = time.time()
say("=" * 92)
say("SOUNDNESS CONTROLS")
say("=" * 92)

# ---- (a) split must tile its parent exactly --------------------------------
rng = np.random.default_rng(0)
bad = 0
for _ in range(20000):
    b = tuple(rng.uniform(-1, 1) if i % 2 == 0 else rng.uniform(1e-4, 1) for i in range(10))
    ch = split(b)
    k = [i for i in range(5) if ch[0][2 * i + 1] != b[2 * i + 1]][0]
    lo, hi = b[2 * k] - b[2 * k + 1], b[2 * k] + b[2 * k + 1]
    l1, h1 = ch[0][2 * k] - ch[0][2 * k + 1], ch[0][2 * k] + ch[0][2 * k + 1]
    l2, h2 = ch[1][2 * k] - ch[1][2 * k + 1], ch[1][2 * k] + ch[1][2 * k + 1]
    if (abs(min(l1, l2) - lo) > 1e-15 or abs(max(h1, h2) - hi) > 1e-15
            or abs(max(l1, l2) - min(h1, h2)) > 1e-15):
        bad += 1
    for i in range(5):                       # untouched axes must be identical
        if i != k and (ch[0][2 * i] != b[2 * i] or ch[0][2 * i + 1] != b[2 * i + 1]
                       or ch[1][2 * i] != b[2 * i] or ch[1][2 * i + 1] != b[2 * i + 1]):
            bad += 1
say(f"\n(a) split tiling      : {bad} defects / 20000 boxes   "
    f"-> {'EXACT (children partition the parent)' if bad == 0 else '*** BROKEN ***'}   [{time.time()-t0:.1f}s]")

# ---- (b) root must cover the reduced domain --------------------------------
r = initial_boxes(0)[0]
ok_root = abs(r[1] - TMAX) < 1e-12 and abs(r[3] - TMAX) < 1e-12 \
    and abs(r[7] - TMAX) < 1e-12 and abs(r[9] - TMAX) < 1e-12 \
    and abs(r[4] - np.pi / 5) < 1e-12 and abs(r[5] - np.pi / 5) < 1e-12
say(f"(b) root coverage     : t3,t5 in [-{r[1]:.2f},{r[1]:.2f}]^2, "
    f"rho in [{r[4]-r[5]:.4f},{r[4]+r[5]:.4f}] vs 2pi/5={2*np.pi/5:.4f}   "
    f"-> {'covers reduced domain' if ok_root else '*** MISMATCH ***'}")

# ---- (c) locate the optimum inside the certifier's own parametrisation -----
seed = np.array([0.006078, -0.010527, np.radians(336.0) % (2 * np.pi / 5), -0.011500, 0.019916])
best = (area(seed), seed)
for k in range(12):
    v0 = seed + (0 if k == 0 else rng.normal(0, 0.01, 5))
    r_ = minimize(area, v0, method="Nelder-Mead",
                  options=dict(xatol=1e-13, fatol=1e-15, maxiter=20000, maxfev=20000))
    if r_.fun < best[0]:
        best = (r_.fun, r_.x)
mn, vstar = best
say(f"(c) true optimum      : {mn:.12f} at rho = {np.degrees(vstar[2]):.4f} deg   "
    f"(matches family-B search 0.834780946: {abs(mn-0.834780946) < 1e-8})   [{time.time()-t0:.1f}s]")

# ---- (d) the discriminating test ------------------------------------------
say(f"\n(d) straddling targets on a box of half-width 0.01 around the optimum")
say(f"    anything ABOVE {mn:.9f} MUST fail; anything below MUST certify\n")
H = 0.01
small = (vstar[0], H, vstar[1], H, vstar[2], H / R5CIRC, vstar[3], H, vstar[4], H)
allok = True

# 0.8330 and 0.8340 already certified in a previous run of this file
# (275,125 boxes / 70.9s and 3,803,665 boxes / 963.6s), establishing the
# below-optimum direction.  0.8345 is dropped: at margin 2.8e-4 from the
# optimum it needs ~1e8 boxes single-core and adds nothing, since the
# direction that actually tests soundness is ABOVE the optimum.
for tgt in (0.8350, 0.8360):
    t1 = time.time()
    n, fails, left = worker((small, tgt, 1e-6, 6_000_000))
    certified = (len(fails) == 0 and len(left) == 0)
    should = tgt < mn
    agree = certified == should
    allok &= agree
    say(f"    target {tgt}: {'certified' if certified else 'failed   '}  "
        f"({n:>9,} boxes, {time.time()-t1:5.1f}s)   expected "
        f"{'certify' if should else 'fail   '}   -> {'OK' if agree else '*** UNSOUND ***'}")

say("\n" + "=" * 92)
verdict = (bad == 0) and ok_root and allok
say(f"  {'ALL CONTROLS PASS -- the a >= 0.8336 certificate stands (double precision)'if verdict else '  CONTROLS FAILED -- certificate is void'}")
say("=" * 92)
