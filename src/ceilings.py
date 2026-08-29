"""S4: rigorous ceilings for the two test families, from exhibited placements.

For a family S of test bodies let M(S) be the minimum over placements of the area
of the convex hull of their union.  The test-set lemma gives a >= M(S), so M(S)
is what any lower bound from S can possibly reach.

Bounding M(S) from ABOVE therefore needs no search and no certificate: exhibiting
a single placement suffices, because M(S) <= area(that placement).  That is the
whole asymmetry between the two directions.  A lower bound on M(S) requires
exhausting the placement space; an upper bound requires one example.

To keep the example rigorous the area is computed with an OUTER approximation.
The hull of the union has support function h(theta) = max_i h_i(theta), and for
any finite set of directions the intersection of the half planes
{x : <x, u(theta_j)> <= h(theta_j)} CONTAINS the hull.  Its area is therefore an
upper bound on the hull's area, computed by a 2x2 solve per pair of consecutive
constraints and the shoelace formula.  No inner approximation, no sampling
argument, and the bound is one sided by construction.

Consequence for the paper: the classical disk + regular triangle + regular
pentagon family used by Pal (1920), Brass and Sharifi (2005) and Xie (2026)
cannot certify our constant, whatever search is applied to it.
"""

import sys

import numpy as np

import geom


def outer_area(bodies, m):
    """Rigorous upper bound on area(conv(union of bodies)), via m half planes."""
    th = 2.0 * np.pi * np.arange(m) / m
    u = np.stack([np.cos(th), np.sin(th)], axis=1)
    h = np.max(np.array([[b.support(t) for t in th] for b in bodies]), axis=0)

    # vertex j = intersection of constraint j and j+1
    u1, u2 = u, np.roll(u, -1, axis=0)
    h1, h2 = h, np.roll(h, -1)
    det = u1[:, 0] * u2[:, 1] - u1[:, 1] * u2[:, 0]
    if np.min(np.abs(det)) < 1e-14:
        raise RuntimeError("degenerate constraint pair")
    x = (h1 * u2[:, 1] - h2 * u1[:, 1]) / det
    y = (h2 * u1[:, 0] - h1 * u2[:, 0]) / det
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))


def place(kind, v):
    D = geom.disk(0.5)
    if kind == "classical":
        B3 = geom.regular_polygon_diameter(3, 1.0)
        B5 = geom.regular_polygon_diameter(5, 1.0)
    else:
        B3 = geom.reuleaux(3, 1.0)
        B5 = geom.reuleaux(5, 1.0)
    return [D, B3.transform(0.0, v[0], v[1]), B5.transform(v[2], v[3], v[4])]


if __name__ == "__main__":
    TARGET = 0.8344
    from scipy.optimize import minimize

    print("=" * 96)
    print("S4  RIGOROUS CEILINGS FROM EXHIBITED PLACEMENTS")
    print("=" * 96)

    for kind, label in (("classical", "disk + regular triangle + regular pentagon"),
                        ("reuleaux", "disk + Reuleaux triangle + Reuleaux pentagon")):
        f = lambda v: geom.hull_area_exact(place(kind, v))
        rng = np.random.default_rng(17)
        best = None
        for _ in range(250):
            v0 = np.array([rng.uniform(-.3, .3), rng.uniform(-.3, .3),
                           rng.uniform(0, 2 * np.pi / 5),
                           rng.uniform(-.3, .3), rng.uniform(-.3, .3)])
            r = minimize(f, v0, method="Nelder-Mead",
                         options=dict(xatol=1e-13, fatol=1e-15,
                                      maxiter=20000, maxfev=20000))
            if best is None or r.fun < best.fun:
                best = r
        v = best.x
        print(f"\n-- {label} --")
        print(f"  witness placement v = {np.array2string(v, precision=12)}")
        print(f"  exact-kernel area at v         {best.fun:.12f}")
        print(f"{'m':>10}  {'outer bound (rigorous)':>24}")
        prev = None
        for m in (10_000, 100_000, 1_000_000):
            a = outer_area(place(kind, v), m)
            print(f"{m:10,}  {a:24.12f}")
            prev = a
        print(f"  => M(family) <= {prev:.9f}")
        if prev < TARGET:
            print(f"     {prev:.9f} < {TARGET}, so NO lower bound of {TARGET} can be")
            print(f"     obtained from this family by any search whatsoever.")
        else:
            print(f"     {prev:.9f} >= {TARGET}: this family can support the target.")
