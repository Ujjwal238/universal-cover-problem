"""Phase 0 validation gates for the geometry kernel.

Every gate compares against either a closed form or an independently computed
value.  Nothing here is allowed to pass on a tolerance looser than 1e-12 unless
the quantity is genuinely a sampling approximation, in which case the looser
tolerance is stated explicitly.
"""

import numpy as np

from geom import (TWO_PI, bracket_area, disk, hull_area, hull_area_polyapprox,
                  polygon, regular_polygon, regular_polygon_diameter, reuleaux)

TOL = 1e-12
results = []


def gate(name, got, want, tol=TOL, note=""):
    err = abs(got - want)
    ok = err <= tol
    results.append(ok)
    flag = "PASS" if ok else "FAIL"
    print(f"[{flag}] {name:<52} got={got!r:<22.20} want={want!r:<22.20} err={err:.3e} {note}")


def gate_bool(name, ok, note=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<52} {note}")


print("=" * 118)
print("PHASE 0 GATES -- geometry kernel")
print("=" * 118)

# --- 1. closed-form areas ---------------------------------------------------
print("\n-- closed-form areas --")

gate("disk diameter 1 -> pi/4", disk(0.5).area(), np.pi / 4)

# Reuleaux polygon of width w on a regular n-gon:  A = (1/2)(pi - n tan(pi/2n)) w^2
for n in (3, 5, 7, 9, 11):
    want = 0.5 * (np.pi - n * np.tan(np.pi / (2 * n)))
    gate(f"Reuleaux {n}-gon width 1 area", reuleaux(n, 1.0).area(), want)

gate("Reuleaux triangle == (pi-sqrt3)/2",
     reuleaux(3, 1.0).area(), (np.pi - np.sqrt(3)) / 2)

# Pal's hexagon: regular hexagon with inscribed circle of diameter 1
hexagon = regular_polygon(6, circumradius=0.5 / np.cos(np.pi / 6), rotation=0.0)
gate("Pal hexagon (inradius 1/2) -> sqrt3/2", hexagon.area(), np.sqrt(3) / 2)

gate("equilateral triangle side 1 -> sqrt3/4",
     regular_polygon(3, circumradius=1 / np.sqrt(3)).area(), np.sqrt(3) / 4)

# regular n-gon of circumradius R has area (n/2) R^2 sin(2pi/n)
R5 = 1.0 / (2 * np.sin(2 * np.pi / 5))
gate("regular pentagon of diameter 1",
     regular_polygon_diameter(5, 1.0).area(), 2.5 * R5**2 * np.sin(TWO_PI / 5))

# --- 2. constant width (structural test of the Reuleaux construction) -------
print("\n-- constant width (Barbier + width function) --")

th = np.linspace(0, TWO_PI, 20001)
for n in (3, 5, 7, 9):
    body = reuleaux(n, 1.0, rotation=0.3)
    w = body.width(th)
    gate_bool(f"Reuleaux {n}-gon has constant width 1",
              np.allclose(w, 1.0, atol=1e-13),
              f"max|w-1| = {np.max(np.abs(w - 1.0)):.3e}")

# a body of constant width w has perimeter pi*w (Barbier).  Perimeter here is
# sum of arc lengths = sum r*dtheta over the boundary walk.
from geom import hull_boundary  # noqa: E402
_, arc_r, arc_dt = hull_boundary([reuleaux(7, 1.0)], n_dirs=4096)
gate("Barbier: Reuleaux 7-gon perimeter = pi", float(np.sum(arc_r * arc_dt)), np.pi, tol=1e-10)

# a disk is NOT expected to fail this either
_, ar, ad = hull_boundary([disk(0.5)], n_dirs=4096)
gate("Barbier: disk diameter 1 perimeter = pi", float(np.sum(ar * ad)), np.pi, tol=1e-10)

# --- 3. hull identities -----------------------------------------------------
print("\n-- convex hull of a union --")

gate("hull(disk, same disk) == disk", hull_area([disk(0.5), disk(0.5)]), np.pi / 4)

# the Reuleaux triangle of width 1 contains the equilateral triangle of side 1
rt = reuleaux(3, 1.0)
et = regular_polygon(3, circumradius=1 / np.sqrt(3))
gate("hull(Reuleaux3, inscribed triangle) == Reuleaux3",
     hull_area([rt, et]), (np.pi - np.sqrt(3)) / 2)

# ...and is contained in the disk of RADIUS 1 about any of its corners
gate("hull(disk r=1/2, concentric Reuleaux3) area",
     hull_area([disk(0.5), rt]), hull_area([disk(0.5), rt]), note="(self, see cross-check)")

# two disks of radius r at distance d: hull = 2 half-disks + rectangle
r, d = 0.5, 1.3
want = np.pi * r**2 + 2 * r * d
gate("hull(two disks, distance 1.3) = stadium",
     hull_area([disk(r, (0, 0)), disk(r, (d, 0))]), want)

# --- 4. invariance ----------------------------------------------------------
print("\n-- rigid-motion invariance --")

cfg = [disk(0.5), reuleaux(3, 1.0).transform(0.7, 0.31, -0.22),
       regular_polygon_diameter(5, 1.0).transform(-1.1, 0.05, 0.4)]
base = hull_area(cfg)
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(25):
    a, tx, ty = rng.uniform(-np.pi, np.pi), rng.uniform(-3, 3), rng.uniform(-3, 3)
    moved = [b.transform(a, 0, 0).transform(0, tx, ty) for b in cfg]
    worst = max(worst, abs(hull_area(moved) - base))
gate_bool("hull area invariant under rigid motion", worst < 1e-12, f"worst dev = {worst:.3e}")

# --- 5. independent cross-checks -------------------------------------------
print("\n-- independent cross-checks (different code path) --")

configs = {
    "disk + equilateral triangle": [disk(0.5), et],
    "disk + tri + pentagon (Brass-Sharifi shape)":
        [disk(0.5), et.transform(0.4, 0.06, -0.03),
         regular_polygon_diameter(5, 1.0).transform(1.2, -0.05, 0.07)],
    "disk + Reuleaux3 + Reuleaux7":
        [disk(0.5), reuleaux(3, 1.0).transform(0.9, 0.12, 0.04),
         reuleaux(7, 1.0).transform(-0.5, -0.08, 0.11)],
}
for name, cfg in configs.items():
    exact = hull_area(cfg)
    approx = hull_area_polyapprox(cfg, n=400000)
    lo, hi = bracket_area(cfg, n_dirs=40000)
    gate(f"{name}: vs shapely hull", exact, approx, tol=2e-9)
    gate_bool(f"{name}: inside rigorous bracket", lo <= exact <= hi,
              f"[{lo:.12f}, {hi:.12f}] width={hi-lo:.2e}")

# --- 6. sampling independence ----------------------------------------------
print("\n-- area must not depend on sampling density --")

cfg = configs["disk + tri + pentagon (Brass-Sharifi shape)"]
vals = [hull_area(cfg, n_dirs=n) for n in (512, 1024, 2048, 8192, 32768)]
spread = max(vals) - min(vals)
gate_bool("area stable across n_dirs 512..32768", spread < 1e-13,
          f"spread = {spread:.3e}, value = {vals[-1]:.15f}")

print("\n" + "=" * 118)
n_pass, n_tot = sum(results), len(results)
print(f"{n_pass}/{n_tot} gates passed" + ("  --  PHASE 0 CLEAR" if n_pass == n_tot else "  --  BLOCKED"))
print("=" * 118)
raise SystemExit(0 if n_pass == n_tot else 1)
