"""S2: the sharper a priori domain lemma (farthest corner, not the centre).

The bound currently used in the certifiers applies the area estimate to the
distance of the test body's CENTRE from the origin.  Brass and Sharifi instead
apply it to the farthest VERTEX, which is much stronger because a body of
diameter one has its corners at distance ~0.53-0.58 from its own centre before
any translation at all.

Setup.  Gauge: the cover contains the disk D of radius 1/2 centred at the
origin.  It also contains R(rho)T + t for the test body T, whose corners
V_1..V_n satisfy |V_j| = R (the circumradius; the corners of a regular Reuleaux
n-gon are concyclic about their centroid).  Convexity gives, for every j,

    area  >=  f(|R(rho)V_j + t|),      f(d) = area conv(D, single point at d)
              f(d) = (1/4)(pi - arccos(1/2d)) + (1/2)sqrt(d^2 - 1/4),   d >= 1/2

so  area >= f(max_j |R(rho)V_j + t|)  because f increases.

Key step.  The n vectors R(rho)V_j are equally spaced by 2pi/n, so whatever the
direction of t, some corner lies within angle pi/n of it:

    max_j <R(rho)V_j, t>  >=  R |t| cos(pi/n),

hence, uniformly in rho,

    max_j |R(rho)V_j + t|  >=  sqrt(R^2 + |t|^2 + 2 R |t| cos(pi/n)) =: g(|t|).

g increases, so with d* defined by f(d*) = target, every t with g(|t|) >= d* is
excluded.  Solving g(s) = d* for s gives the closed form

    TMAX_n = -R cos(pi/n) + sqrt(R^2 cos^2(pi/n) - R^2 + d*^2).

The bound is independent of rho, which is what makes it usable as a root box.
Note both circumradii exceed 1/2, so f is always evaluated in its valid range.
"""

import numpy as np

import geom


def f_apriori(d):
    """Area of conv(disk of radius 1/2 at the origin, a point at distance d)."""
    d = np.asarray(d, dtype=float)
    return 0.25 * (np.pi - np.arccos(1.0 / (2.0 * d))) + 0.5 * np.sqrt(d * d - 0.25)


def d_star(target, lo=0.5 + 1e-15, hi=5.0):
    """Smallest d with f(d) >= target, by bisection on the increasing f."""
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f_apriori(mid) < target:
            lo = mid
        else:
            hi = mid
    return hi


def tmax(target, R, n):
    """Largest |t| not excluded by the farthest-corner bound."""
    ds = d_star(target)
    c = np.cos(np.pi / n)
    disc = R * R * c * c - R * R + ds * ds
    if disc < 0.0:
        return 0.0
    return -R * c + np.sqrt(disc)


if __name__ == "__main__":
    TARGET = 0.8344
    OLD = 0.70
    ds = d_star(TARGET)
    print("=" * 96)
    print(f"S2  SHARPER DOMAIN LEMMA      target = {TARGET}")
    print("=" * 96)
    print(f"  d* with f(d*) = target      : {ds:.12f}   (f(d*) = {float(f_apriori(ds)):.12f})")
    print(f"  old lemma (centre distance) : |t| <= {OLD}   [f({OLD}) = {float(f_apriori(OLD)):.9f}]")
    print()

    rng = np.random.default_rng(5)
    rows = []
    for n in (3, 5):
        V = geom.reuleaux_corners(n, 1.0)
        V = V - V.mean(axis=0)
        R = float(np.max(np.hypot(V[:, 0], V[:, 1])))
        # concyclic check: all corners at the same radius
        rad = np.hypot(V[:, 0], V[:, 1])
        tm = tmax(TARGET, R, n)
        print(f"  Reuleaux-{n}:  R = {R:.12f}  (corner radii spread "
              f"{rad.max()-rad.min():.2e})   ->  |t| <= {tm:.12f}")
        rows.append((n, V, R, tm))

        # SOUNDNESS: every t just OUTSIDE the new bound must really be excluded.
        bad = 0
        worst = np.inf
        for _ in range(200000):
            s = tm * (1.0 + 10.0 ** rng.uniform(-9, 0))       # |t| > tm
            ph = rng.uniform(0, 2 * np.pi)
            t = s * np.array([np.cos(ph), np.sin(ph)])
            rho = rng.uniform(0, 2 * np.pi)
            ca, sa = np.cos(rho), np.sin(rho)
            W = V @ np.array([[ca, sa], [-sa, ca]]) + t
            a = float(f_apriori(np.max(np.hypot(W[:, 0], W[:, 1]))))
            worst = min(worst, a)
            bad += (a < TARGET - 1e-13)
        print(f"              exclusion sound on 200,000 samples: {bad} failures, "
              f"min a priori area {worst:.12f} >= {TARGET}")

    print()
    n3, _, _, t3 = rows[0]
    n5, _, _, t5 = rows[1]
    old_vol = (2 * OLD) ** 4
    new_vol = (2 * t3) ** 2 * (2 * t5) ** 2
    print(f"  translation-box 4-volume   old {old_vol:.6f}   new {new_vol:.8f}   "
          f"reduction {old_vol/new_vol:.1f}x")
    print(f"  linear shrink per body     R3 {OLD/t3:.2f}x    R5 {OLD/t5:.2f}x")
    print()
    print("  target sensitivity (domain grows with the target, as it must):")
    for tg in (0.8344, 0.8346, 0.8348, 0.834780952792):
        print(f"    target {tg:.12f}  ->  R3 |t| <= {tmax(tg, rows[0][2], 3):.9f}   "
              f"R5 |t| <= {tmax(tg, rows[1][2], 5):.9f}")
