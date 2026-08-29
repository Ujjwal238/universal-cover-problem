"""S1: rigorous floating-point error bound for the verifier's leaf bound.

The genre precedent at SoCG (Fekete, Gurunathan, Juneja, Keldenich, Kleist,
Scheffer, SoCG 2021) uses interval arithmetic.  This verifier uses double
precision, so the paper owes a rigorous error bound rather than an appeal to
head-room.  Converting 245M leaves to intervals is unnecessary: the leaf bound is
the area of the convex hull of finitely many points, each of which is TESTED for
membership, so only two things can go wrong.

  (E1) An accepted witness point may lie slightly OUTSIDE its body, because the
       membership test uses a computed distance and a tolerance tau.  If every
       point is within eps of the true body, the point set lies in
       (true hull) (+) eps*B, so the computed hull can exceed the truth by at
       most  eps*P + pi*eps^2,  P a perimeter bound.

       A distance sqrt((dx)^2+(dy)^2) carries relative error at most 4u
       (u = 2^-53) for the operation count involved, so acceptance d_hat <= rho+tau
       gives d_true <= (rho+tau)(1+4u), i.e.  eps <= tau + 4u(rho+tau).

  (E2) The shoelace sum is evaluated in floating point.  For N vertices the
       standard bound is  |A_hat - A| <= gamma_N * S,  with
       gamma_N = N u/(1 - N u)  and  S = (1/2) sum |x_i y_{i+1}| + |x_{i+1} y_i|.
       Both N and S are bounded structurally rather than sampled: the witness set
       holds at most m + 2K points, so N <= m + 2K, and every coordinate satisfies
       |x|,|y| <= R = TMAX + circumradius, so each of the N terms is at most R^2
       and S <= N R^2.

  A third effect, the rounding of the corner and arc positions themselves, enters
  through eps: the corners are computed from cos and sin of the box-centre
  rotation, and library trigonometry is not correctly rounded in general.  A few
  ulp of coordinate error is folded into eps as c*u*R below.  It sits four orders
  under the membership tolerance and does not move eta at the precision quoted.

Qhull enters only through WHICH vertices are returned.  Dropping a vertex shrinks
the hull, which errs in the safe direction for a lower bound; a mis-ordering would
show up against the hand-written hull, and the audit already cross-checks leaves
against it.

The certified statement is then  true bound >= computed bound - eta,  with
eta = (E1) + (E2), and the proof stands as long as eta is below the worst leaf
slack the verifier reports.

Part B recomputes the same leaf bound in 60-digit arithmetic on the hardest boxes
(smallest, nearest the optimum) and reports the actual double-precision
discrepancy, as an independent check on the analysis above.
"""

import sys

import numpy as np
from mpmath import mp, mpf, sqrt as msqrt, cos as mcos, sin as msin, atan2 as matan2, pi as mpi

import verifyB as V
import geom

U = 2.0 ** -53
TAU = 1e-12                     # membership tolerance in verifyB.core_points
PERIM = np.pi * 2.0 * (0.70 + 1.0 / np.sqrt(3.0))   # rigorous, old domain


def ring_np(m):
    th = 2 * np.pi * np.arange(m) / m
    return 0.5 * np.stack([np.cos(th), np.sin(th)], 1)


# ---------------------------------------------------------------- part A


RMAX = 0.70 + 1.0 / np.sqrt(3.0)        # |t| bound plus the larger circumradius
NMAX = lambda m, K: m + 2 * K           # structural bound on the hull vertex count


def eps_membership(rho_max=1.0, c_corner=8.0):
    """How far outside its body an accepted witness point can lie.

    tau           the membership tolerance itself
    4u(rho+tau)   rounding of the distance, a sqrt of a sum of two squares
    c*u*R         rounding of the corner and arc coordinates entering that distance
    """
    return TAU + 4.0 * U * (rho_max + TAU) + c_corner * U * RMAX


def shoelace_terms(P):
    """N and S for the shoelace error bound, from the hull vertex list."""
    x, y = P[:, 0], P[:, 1]
    xr, yr = np.roll(x, -1), np.roll(y, -1)
    S = 0.5 * float(np.sum(np.abs(x * yr) + np.abs(xr * y)))
    return len(P), S


def hull_vertices(pts):
    from scipy.spatial import ConvexHull
    h = ConvexHull(pts)
    return pts[h.vertices]


# ---------------------------------------------------------------- part B (mpmath)


def core_points_mp(Vm, rho, K, tol=None):
    """core_points of verifyB, in mpmath.  Mirrors the branch choice exactly."""
    n = len(Vm)
    half = (n - 1) // 2
    # The reconstructed corners sit at distance exactly rho from two of the
    # centres, so a STRICT membership test discards them roughly half the time
    # and the hull collapses.  verifyB accepts d <= rho + 1e-12; mirror that
    # semantics with a tolerance far below the quantity being measured.
    if tol is None:
        tol = mpf(10) ** -45
    if rho <= mpf("1e-40"):
        return None
    cx = sum(p[0] for p in Vm) / n
    cy = sum(p[1] for p in Vm) / n
    sec = max(msqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2) for p in Vm)
    if rho < sec - tol:
        return None
    P = []
    for j in range(n):
        a, b = Vm[j], Vm[(j + 1) % n]
        dx, dy = b[0] - a[0], b[1] - a[1]
        d = msqrt(dx * dx + dy * dy)
        if d == 0 or d > 2 * rho:
            return None
        ux, uy = dx / d, dy / d
        h2 = rho * rho - d * d / 4
        if h2 < 0:
            return None
        mx, my = a[0] + ux * d / 2, a[1] + uy * d / 2
        ox, oy = -uy * msqrt(h2), ux * msqrt(h2)
        old = Vm[(j + half + 1) % n]
        p1 = (mx + ox, my + oy)
        p2 = (mx - ox, my - oy)
        d1 = (p1[0] - old[0]) ** 2 + (p1[1] - old[1]) ** 2
        d2 = (p2[0] - old[0]) ** 2 + (p2[1] - old[1]) ** 2
        P.append(p1 if d1 < d2 else p2)

    pts = list(P)
    spans = []
    for j in range(n):
        prev = P[(j - 1) % n]
        a0 = matan2(prev[1] - Vm[j][1], prev[0] - Vm[j][0])
        a1 = matan2(P[j][1] - Vm[j][1], P[j][0] - Vm[j][0])
        sp = (a1 - a0) % (2 * mpi)
        spans.append((j, a0, sp))
    tot = sum(s for _, _, s in spans)
    if tot == 0:
        tot = 2 * mpi
    for j, a0, sp in spans:
        k = max(1, int(round(float(K * sp / tot))))
        for i in range(1, k):
            t = a0 + sp * i / k
            pts.append((Vm[j][0] + rho * mcos(t), Vm[j][1] + rho * msin(t)))

    out = []
    for p in pts:
        ok = True
        for q in Vm:
            if msqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) > rho + tol:
                ok = False
                break
        if ok:
            out.append(p)
    return out if len(out) >= 3 else None


def hull_area_mp(pts):
    """Monotone chain then shoelace, all in mpmath."""
    pts = sorted(set(pts))
    if len(pts) < 3:
        return mpf(0)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    H = lower[:-1] + upper[:-1]
    A = mpf(0)
    for i in range(len(H)):
        x1, y1 = H[i]
        x2, y2 = H[(i + 1) % len(H)]
        A += x1 * y2 - x2 * y1
    return abs(A) / 2


def leaf_bound_mp(b, m, K):
    c3x, h3x, c3y, h3y, cr, hr, c5x, h5x, c5y, h5y = [mpf(str(v)) for v in b]
    ring = []
    for k in range(m):
        t = 2 * mpi * k / m
        ring.append((mcos(t) / 2, msin(t) / 2))
    C3 = [(mpf(str(p[0])), mpf(str(p[1]))) for p in geom.reuleaux_corners(3, 1.0)]
    C5 = [(mpf(str(p[0])), mpf(str(p[1]))) for p in geom.reuleaux_corners(5, 1.0)]
    circ5 = max(msqrt(p[0] ** 2 + p[1] ** 2) for p in C5)

    d3 = msqrt(h3x ** 2 + h3y ** 2)
    d5 = msqrt(h5x ** 2 + h5y ** 2) + 2 * circ5 * msin(hr / 2)
    V3 = [(p[0] + c3x, p[1] + c3y) for p in C3]
    ca, sa = mcos(cr), msin(cr)
    V5 = [(p[0] * ca - p[1] * sa + c5x, p[0] * sa + p[1] * ca + c5y) for p in C5]

    W3 = core_points_mp(V3, 1 - d3, K)
    W5 = core_points_mp(V5, 1 - d5, K)
    if W3 is None or W5 is None:
        return None
    return hull_area_mp(ring + W3 + W5), len(W3), len(W5)


if __name__ == "__main__":
    mp.dps = 60
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    nbox = int(sys.argv[3]) if len(sys.argv) > 3 else 24

    print("=" * 100)
    print("S1  RIGOROUS FLOATING-POINT ERROR BOUND FOR THE LEAF BOUND")
    print("=" * 100)

    eps = eps_membership()
    e1 = eps * PERIM + np.pi * eps * eps
    nmax = NMAX(1024, 1024)
    smax = nmax * RMAX * RMAX
    gmax = nmax * U / (1.0 - nmax * U)
    e2 = gmax * smax
    print("\n-- Part A: a priori bound --")
    print(f"  u = 2^-53                                  {U:.6e}")
    print(f"  membership tolerance tau                   {TAU:.3e}")
    print(f"  eps = tau + 4u(rho+tau) + 8uR              {eps:.12e}")
    print(f"  rigorous perimeter bound P                 {PERIM:.6f}")
    print(f"  (E1) eps*P + pi*eps^2                      {e1:.6e}")

    print(f"  (E2) structural: N <= m+2K = {nmax}, S <= N R^2 = {smax:.1f}")
    print(f"       gamma_N * S                           {e2:.6e}")

    eta = e1 + e2
    print(f"\n  eta = (E1) + (E2)                          {eta:.6e}")
    print(f"  worst leaf slack reported by the verifier   1.240900e-05")
    print(f"  ratio slack/eta                            {1.2409e-05/eta:,.0f}x")
    print(f"  => certified bound 0.8344 survives subtracting eta: "
          f"{'YES' if 1.2409e-05 > eta else 'NO'}")

    # the certified optimum, used to place the hardest boxes for part B
    o = np.array([0.0060536881, -0.0104850761, 3.3510900886,
                  -0.011498609, 0.0199348702])
    print(f"\n-- Part B: 60-digit recomputation, m={m} K={K}, {nbox} hardest boxes --")
    print(f"{'halfwidth':>11}  {'double':>20}  {'60-digit':>22}  {'diff':>12}  "
          f"pts dbl/mp")
    worst = 0.0
    for i in range(nbox):
        s = 10.0 ** (-3.0 - 3.0 * i / max(nbox - 1, 1))
        b = (o[0], 0.7 * s, o[1], 0.7 * s, o[2], (np.pi / 5) * s,
             o[3], 0.7 * s, o[4], 0.7 * s)
        dbl = V.box_bound(b, ring_np(m), K)
        got = leaf_bound_mp(b, m, K)
        if got is None:
            continue
        ex, n3, n5 = got
        # point-count parity check: the two implementations must keep the same
        # witnesses, otherwise the comparison measures discretization, not roundoff
        c3 = geom.reuleaux_corners(3, 1.0) + np.array([b[0], b[2]])
        ca2, sa2 = np.cos(b[4]), np.sin(b[4])
        c5 = geom.reuleaux_corners(5, 1.0) @ np.array([[ca2, sa2], [-sa2, ca2]]) \
            + np.array([b[6], b[8]])
        circ5d = float(np.max(np.hypot(*geom.reuleaux_corners(5, 1.0).T)))
        d3d = V.core_points(c3, 1.0 - np.hypot(b[1], b[3]), K)
        d5d = V.core_points(c5, 1.0 - (np.hypot(b[7], b[9])
                                       + 2 * circ5d * np.sin(0.5 * b[5])), K)
        par = f"{len(d3d)}/{n3} {len(d5d)}/{n5}"
        d = abs(dbl - float(ex))
        worst = max(worst, d)
        print(f"{s:11.2e}  {dbl:20.15f}  {float(ex):22.15f}  {d:12.3e}  {par}")
    print(f"\n  worst double-vs-60-digit discrepancy       {worst:.3e}")
    print(f"  a priori eta                               {eta:.3e}")
    print(f"  {'consistent: measured error below eta' if worst <= eta else '*** measured error EXCEEDS eta'}")
