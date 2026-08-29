"""
Exact convex geometry kernel for Lebesgue's universal covering problem.

REPRESENTATION
--------------
A convex body is a cyclic list of boundary "pieces".  Piece j carries a centre
c_j and a radius r_j, and is active for outward-normal angles theta in
[b_j, b_{j+1}).  On that range

    support function   h(theta) = <c_j, u(theta)> + r_j
    contact point      p(theta) = c_j + r_j * u(theta)

where u(theta) = (cos theta, sin theta).

    r_j == 0  encodes a CORNER at c_j whose normal cone is [b_j, b_{j+1}).
    r_j >  0  encodes a circular ARC of radius r_j centred at c_j.

Everything we need is closed under this representation:

  * rigid motion   (c, r, [b1,b2]) -> (R_a c + t, r, [b1+a, b2+a])
  * convex hull of a union         h_hull(theta) = max_i h_i(theta)

The second identity is what makes the whole project computationally cheap: the
convex hull of a union of disks, polygons and Reuleaux polygons never has to be
constructed geometrically, only maximised pointwise.

AREA
----
Walking theta from 0 to 2*pi traverses the hull boundary counter-clockwise.
Between two consecutive directions the boundary is either

  * the same arc of the same body  -> add the circular segment
        (r^2 / 2) * (dtheta - sin dtheta)
  * a switch of winning body       -> a straight common-tangent bridge, which
                                      the shoelace term already handles exactly

Piece changes inside one body land exactly on that body's break angles, which
are always included as sample directions.  Changes of the winning body are
located in closed form (see _crossing).  So the computed area is exact up to
floating point, independent of the sampling density, provided the sampling is
fine enough to detect every crossing.
"""

from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


# ----------------------------------------------------------------------------
# bodies
# ----------------------------------------------------------------------------

class Body:
    """Convex body given by support-function pieces (see module docstring)."""

    __slots__ = ("breaks", "centers", "radii", "name")

    def __init__(self, breaks, centers, radii, name=""):
        breaks = np.asarray(breaks, dtype=float) % TWO_PI
        centers = np.asarray(centers, dtype=float).reshape(-1, 2)
        radii = np.asarray(radii, dtype=float).reshape(-1)
        order = np.argsort(breaks, kind="stable")
        self.breaks = breaks[order]
        self.centers = centers[order]
        self.radii = radii[order]
        self.name = name
        if not (len(self.breaks) == len(self.centers) == len(self.radii)):
            raise ValueError("breaks/centers/radii length mismatch")

    # -- basic queries -------------------------------------------------------

    def piece(self, theta):
        """Index of the piece active at normal angle(s) theta.

        searchsorted(...) - 1 returns -1 for theta < breaks[0], which numpy
        resolves to the last piece.  That is exactly the cyclic wrap we want.
        """
        t = np.asarray(theta, dtype=float) % TWO_PI
        return np.searchsorted(self.breaks, t, side="right") - 1

    def support(self, theta):
        t = np.asarray(theta, dtype=float)
        j = self.piece(t)
        return self.centers[j, 0] * np.cos(t) + self.centers[j, 1] * np.sin(t) + self.radii[j]

    def contact(self, theta):
        t = np.asarray(theta, dtype=float)
        j = self.piece(t)
        c, s = np.cos(t), np.sin(t)
        return np.stack([self.centers[j, 0] + self.radii[j] * c,
                         self.centers[j, 1] + self.radii[j] * s], axis=-1)

    # -- transforms ----------------------------------------------------------

    def transform(self, angle=0.0, tx=0.0, ty=0.0):
        ca, sa = np.cos(angle), np.sin(angle)
        R = np.array([[ca, -sa], [sa, ca]])
        return Body(self.breaks + angle,
                    self.centers @ R.T + np.array([tx, ty]),
                    self.radii,
                    self.name)

    def area(self, n_dirs=2048):
        return hull_area([self], n_dirs=n_dirs)

    def width(self, theta):
        """Width in direction theta: h(theta) + h(theta + pi)."""
        return self.support(theta) + self.support(np.asarray(theta) + np.pi)


# ----------------------------------------------------------------------------
# constructors
# ----------------------------------------------------------------------------

def disk(radius=0.5, center=(0.0, 0.0), name="disk"):
    return Body([0.0], [center], [radius], name)


def polygon(vertices, name="polygon"):
    """Convex polygon from counter-clockwise vertices.

    Vertex v_j is the contact point for normals between the outward normal of
    edge (v_{j-1} -> v_j) and that of edge (v_j -> v_{j+1}).
    """
    V = np.asarray(vertices, dtype=float).reshape(-1, 2)
    n = len(V)
    E = np.roll(V, -1, axis=0) - V                       # edge j: v_j -> v_{j+1}
    # outward normal of a ccw edge e = (ex, ey) is (ey, -ex)
    nu = np.arctan2(-E[:, 0], E[:, 1])
    starts = np.roll(nu, 1)                              # vertex j starts at nu_{j-1}
    return Body(starts, V, np.zeros(n), name)


def regular_polygon(n, circumradius=1.0, rotation=0.0, center=(0.0, 0.0), name=None):
    k = np.arange(n)
    ang = TWO_PI * k / n + rotation
    V = np.stack([center[0] + circumradius * np.cos(ang),
                  center[1] + circumradius * np.sin(ang)], axis=1)
    return polygon(V, name or f"reg{n}gon")


def regular_polygon_diameter(n, diameter=1.0, rotation=0.0, center=(0.0, 0.0), name=None):
    """Regular n-gon whose DIAMETER (longest distance between vertices) is given."""
    m = n // 2
    # max chord is between vertices m apart
    circ = diameter / (2.0 * np.sin(np.pi * m / n))
    return regular_polygon(n, circ, rotation, center, name or f"reg{n}gon_d{diameter:g}")


def reuleaux(n, width=1.0, rotation=0.0, center=(0.0, 0.0), name=None):
    """Regular Reuleaux polygon with an odd number n of arcs and given width.

    Built on a regular n-gon with vertices v_0..v_{n-1}.  With m = (n-1)/2 the
    arc centred at v_k has radius `width` and runs from v_{k+m} to v_{k+m+1};
    corner v_j is where the arcs centred at v_{j-m-1} and v_{j-m} meet.
    """
    if n % 2 == 0 or n < 3:
        raise ValueError("Reuleaux polygons need an odd n >= 3")
    m = (n - 1) // 2
    circ = width / (2.0 * np.sin(np.pi * m / n))
    k = np.arange(n)
    ang = TWO_PI * k / n + rotation
    V = np.stack([center[0] + circ * np.cos(ang),
                  center[1] + circ * np.sin(ang)], axis=1)

    starts, centers, radii = [], [], []
    for j in range(n):
        # arc centred at v_j, from v_{j+m} to v_{j+m+1}
        a0 = V[(j + m) % n] - V[j]
        starts.append(np.arctan2(a0[1], a0[0]))
        centers.append(V[j])
        radii.append(width)
        # corner at v_{j+m+1}: starts where that arc ends
        a1 = V[(j + m + 1) % n] - V[j]
        starts.append(np.arctan2(a1[1], a1[0]))
        centers.append(V[(j + m + 1) % n])
        radii.append(0.0)

    return Body(starts, centers, radii, name or f"reuleaux{n}_w{width:g}")


# ----------------------------------------------------------------------------
# convex hull of a union
# ----------------------------------------------------------------------------

def _crossing(bi, ji, bk, jk, lo, hi):
    """Angle in (lo, hi) where h_i == h_k, with pieces ji / jk active.

    h_i - h_k = <c_i - c_k, u(theta)> + (r_i - r_k) = 0
      =>  D cos(theta - phi) = r_k - r_i,  D = |c_i - c_k|,  phi = atan2(dy, dx)
    Solved in closed form; falls back to bisection in degenerate cases.
    """
    d = bi.centers[ji] - bk.centers[jk]
    D = float(np.hypot(d[0], d[1]))
    rhs = float(bk.radii[jk] - bi.radii[ji])
    if D > 1e-14 and abs(rhs) <= D:
        phi = float(np.arctan2(d[1], d[0]))
        w = float(np.arccos(np.clip(rhs / D, -1.0, 1.0)))
        for cand in (phi + w, phi - w):
            for shift in (-TWO_PI, 0.0, TWO_PI):
                t = cand + shift
                if lo < t < hi:
                    return t
    # bisection fallback
    f = lambda t: float(bi.support(t) - bk.support(t))
    a, b = lo, hi
    fa = f(a)
    for _ in range(80):
        mid = 0.5 * (a + b)
        if fa * f(mid) <= 0.0:
            b = mid
        else:
            a, fa = mid, f(mid)
    return 0.5 * (a + b)


def hull_boundary(bodies, n_dirs=2048):
    """Ordered boundary samples of conv(union of bodies).

    Returns (pts, arc_r, arc_dt) where pts[i] -> pts[i+1] is joined by an arc of
    radius arc_r[i] subtending arc_dt[i] (both zero for a straight bridge).
    """
    # sample directions: uniform grid plus every break angle of every body,
    # nudged so each break is bracketed rather than landed on exactly.
    grid = [np.linspace(0.0, TWO_PI, n_dirs, endpoint=False)]
    eps = 1e-11
    for b in bodies:
        grid.append(b.breaks + eps)
        grid.append(b.breaks - eps)
    th = np.unique(np.concatenate(grid) % TWO_PI)

    H = np.stack([b.support(th) for b in bodies])        # (nbodies, ndirs)
    win = np.argmax(H, axis=0)
    pieces = np.stack([b.piece(th) for b in bodies])
    lab_p = pieces[win, np.arange(len(th))]

    pts, arc_r, arc_dt = [], [], []

    def push(p, r, dt):
        pts.append(p)
        arc_r.append(r)
        arc_dt.append(dt)

    N = len(th)
    for i in range(N):
        t0 = th[i]
        t1 = th[(i + 1) % N] + (TWO_PI if i == N - 1 else 0.0)
        bi, ji = bodies[win[i]], lab_p[i]
        p0 = bi.centers[ji] + bi.radii[ji] * np.array([np.cos(t0), np.sin(t0)])

        same_body = win[i] == win[(i + 1) % N]
        same_piece = same_body and lab_p[i] == lab_p[(i + 1) % N]

        if same_piece:
            push(p0, bi.radii[ji], t1 - t0)
        elif same_body:
            # piece change inside one body: breaks are bracketed by +-eps so the
            # two samples straddle it; the connecting step is negligible.
            push(p0, bi.radii[ji], t1 - t0)
        else:
            bk = bodies[win[(i + 1) % N]]
            jk = lab_p[(i + 1) % N]
            tc = _crossing(bi, ji, bk, jk, t0, t1)
            # outgoing body follows its arc up to tc, then a straight bridge
            push(p0, bi.radii[ji], tc - t0)
            pc_out = bi.centers[ji] + bi.radii[ji] * np.array([np.cos(tc), np.sin(tc)])
            pc_in = bk.centers[jk] + bk.radii[jk] * np.array([np.cos(tc), np.sin(tc)])
            push(pc_out, 0.0, 0.0)
            push(pc_in, 0.0, 0.0)
            # remainder of the step belongs to the incoming body; recorded by
            # overwriting the last arc entry
            arc_r[-1] = bk.radii[jk]
            arc_dt[-1] = t1 - tc

    return np.asarray(pts), np.asarray(arc_r), np.asarray(arc_dt)


def hull_area(bodies, n_dirs=2048):
    """Exact area of conv(union of bodies)."""
    pts, arc_r, arc_dt = hull_boundary(bodies, n_dirs=n_dirs)
    x, y = pts[:, 0], pts[:, 1]
    shoelace = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    caps = 0.5 * float(np.sum(arc_r ** 2 * (arc_dt - np.sin(arc_dt))))
    return shoelace + caps


def reuleaux_from_corners(V, width=1.0, name=None):
    """Reuleaux polygon from its corner points, given in boundary order.

    With m = (n-1)/2, the arc centred at V[j] runs from V[j+m] to V[j+m+1], so
    the corners must satisfy the n constraints |V[j] - V[j+m]| = width.  Those n
    constraints on 2n coordinates, modulo the 3-parameter rigid motion group,
    leave an (n-3)-dimensional family:

        n = 3  ->  0 parameters   (every Reuleaux triangle of width w is congruent)
        n = 5  ->  2 parameters
        n = 7  ->  4 parameters

    Note this is n-3, not n-1: an arc's two endpoints each impose a continuity
    condition, giving 2n conditions in total rather than n.
    """
    V = np.asarray(V, dtype=float).reshape(-1, 2)
    n = len(V)
    if n % 2 == 0 or n < 3:
        raise ValueError("need an odd number of corners")
    m = (n - 1) // 2
    res = np.abs(np.linalg.norm(V - np.roll(V, -m, axis=0), axis=1) - width).max()
    if res > 1e-9:
        raise ValueError(f"corners violate |V[j]-V[j+m]|={width} by {res:.2e}")

    starts, centers, radii = [], [], []
    for j in range(n):
        a0 = V[(j + m) % n] - V[j]
        starts.append(np.arctan2(a0[1], a0[0])); centers.append(V[j]); radii.append(width)
        a1 = V[(j + m + 1) % n] - V[j]
        starts.append(np.arctan2(a1[1], a1[0]))
        centers.append(V[(j + m + 1) % n]); radii.append(0.0)
    return Body(starts, centers, radii, name or f"reuleaux{n}_irr")


def reuleaux_corners(n, width=1.0, rotation=0.0, center=(0.0, 0.0)):
    """Corner points of the regular width-`width` Reuleaux n-gon, in order."""
    m = (n - 1) // 2
    circ = width / (2.0 * np.sin(np.pi * m / n))
    ang = TWO_PI * np.arange(n) / n + rotation
    return np.stack([center[0] + circ * np.cos(ang),
                     center[1] + circ * np.sin(ang)], 1)


def erode_reuleaux(V, width, delta, name="core"):
    """The Minkowski erosion  (Reuleaux polygon on corners V)  minus  delta*disk.

    Erosion distributes over intersection, and a Reuleaux polygon of width w is
    exactly the intersection of the disks D(V[j], w).  Hence

        R (-) delta*B  =  intersection_j D(V[j], w - delta)

    i.e. shrink every radius by delta and re-intersect.  The result is again
    arcs meeting at corners, so it is representable in the same Body type.
    Consecutive arcs (centred V[j] and V[j+1]) meet at the circle-circle
    intersection lying near the original corner V[j+m+1].

    Returns None if the erosion is empty / degenerate.
    """
    V = np.asarray(V, dtype=float).reshape(-1, 2)
    n = len(V)
    m = (n - 1) // 2
    rho = width - delta
    if rho <= 1e-9:
        return None

    # EMPTINESS TEST.  intersection_j D(V_j, rho) is non-empty iff rho is at least
    # the radius of the smallest enclosing circle of the corners, because
    # min_p max_j |p - V_j| IS that radius.  Without this test the construction
    # below -- which only makes CONSECUTIVE circles meet -- happily returns a
    # body when the true intersection is empty, inflating the area bound by as
    # much as 0.53 and silently pruning boxes that were never covered.
    ctr = V.mean(axis=0)
    sec = float(np.max(np.hypot(V[:, 0] - ctr[0], V[:, 1] - ctr[1])))
    if rho < sec - 1e-15:
        return None

    P = np.empty((n, 2))
    for j in range(n):
        a, b = V[j], V[(j + 1) % n]
        d = np.hypot(*(b - a))
        if d < 1e-15 or d > 2 * rho:
            return None
        # two circle-circle intersections; keep the one near the old corner
        u = (b - a) / d
        h2 = rho * rho - (d * 0.5) ** 2
        if h2 < 0:
            return None
        mid = a + u * (d * 0.5)
        off = np.array([-u[1], u[0]]) * np.sqrt(h2)
        old = V[(j + m + 1) % n]
        P[j] = mid + off if np.hypot(*(mid + off - old)) < np.hypot(*(mid - off - old)) else mid - off

    # CONSTRUCTION CHECK.  Every constructed corner must lie in EVERY disk, not
    # merely in the two whose circles produced it.  Consecutive-circle geometry
    # alone does not imply membership in the rest, so verify it directly.
    dd = np.sqrt(((P[:, None, :] - V[None, :, :]) ** 2).sum(-1))
    if float(dd.max()) > rho + 1e-12:
        return None

    starts, centers, radii = [], [], []
    for j in range(n):
        a0 = P[(j - 1) % n] - V[j]
        starts.append(np.arctan2(a0[1], a0[0])); centers.append(V[j]); radii.append(rho)
        a1 = P[j] - V[j]
        starts.append(np.arctan2(a1[1], a1[0])); centers.append(P[j]); radii.append(0.0)
    return Body(starts, centers, radii, name)


def reuleaux_manifold(n, params, width=1.0, iters=80):
    """A point on the (n-3)-dimensional family of width-`width` Reuleaux n-gons.

    Perturbs the regular corner set by `params` (length 2n-3, acting on the
    non-gauge coordinates) and projects back onto {|V[j]-V[j+m]| = width} by
    Gauss-Newton.  Returns None if the projection fails or the result is not a
    valid body.  Gauge: V[0] pinned at the origin and V[m] pinned on the +x axis.
    """
    m = (n - 1) // 2
    circ = width / (2.0 * np.sin(np.pi * m / n))
    ang = TWO_PI * np.arange(n) / n
    V = np.stack([circ * np.cos(ang), circ * np.sin(ang)], 1)
    V = V - V[0]
    th0 = np.arctan2(V[m, 1], V[m, 0])
    c0, s0 = np.cos(-th0), np.sin(-th0)
    V = V @ np.array([[c0, s0], [-s0, c0]])

    free = [(i, k) for i in range(n) for k in (0, 1)][2:]     # drop V[0]
    free = [fk for fk in free if fk != (m, 1)]                # drop V[m].y
    p = np.asarray(params, dtype=float).reshape(-1)
    if p.size != len(free):
        raise ValueError(f"expected {len(free)} params for n={n}, got {p.size}")
    for (i, k), dv in zip(free, p):
        V[i, k] += dv

    for _ in range(iters):
        D = V - np.roll(V, -m, axis=0)
        L = np.linalg.norm(D, axis=1)
        r = L - width
        if np.abs(r).max() < 1e-14:
            break
        J = np.zeros((n, len(free)))
        col = {fk: c for c, fk in enumerate(free)}
        for j in range(n):
            u = D[j] / L[j]
            for k in (0, 1):
                if (j, k) in col:
                    J[j, col[(j, k)]] += u[k]
                if ((j + m) % n, k) in col:
                    J[j, col[((j + m) % n, k)]] -= u[k]
        step, *_ = np.linalg.lstsq(J, r, rcond=None)
        for (i, k), d in zip(free, step):
            V[i, k] -= d
    else:
        return None

    # RE-CENTRE.  The gauge above pins V[0] at the origin and V[m] on the +x
    # axis, which leaves the body sitting ~1 circumradius off-centre.  Body
    # transforms rotate about the ORIGIN, so an off-centre body would be swung
    # bodily around rather than spun in place, and a bounded translation range
    # could never bring it back -- producing hull areas above the known upper
    # bound.  Centre the corners so that rotation acts in place.
    V = V - V.mean(axis=0)
    try:
        return reuleaux_from_corners(V, width)
    except ValueError:
        return None


def _flat_pieces(bodies):
    """Flatten all bodies into arrays of pieces (cx, cy, r, a0, length, body)."""
    cx, cy, rr, a0, ln, bd = [], [], [], [], [], []
    for bi, b in enumerate(bodies):
        n = len(b.breaks)
        for j in range(n):
            s = b.breaks[j]
            e = b.breaks[(j + 1) % n]
            L = TWO_PI if n == 1 else (e - s) % TWO_PI
            if L <= 0.0:
                L = TWO_PI
            cx.append(b.centers[j, 0]); cy.append(b.centers[j, 1])
            rr.append(b.radii[j]); a0.append(s); ln.append(L); bd.append(bi)
    return (np.array(cx), np.array(cy), np.array(rr),
            np.array(a0), np.array(ln), np.array(bd))


def envelope(bodies):
    """Upper envelope of the support functions, exactly.

    Returns (t, cx, cy, r): on the angular cell [t[i], t[i+1]) the hull support
    is  h(theta) = cx[i] cos theta + cy[i] sin theta + r[i].  Cells are found by
    enumerating piece boundaries and pairwise sinusoid crossings, so nothing is
    sampled.  This is the object both hull_area_exact and max_excess consume.
    """
    CX, CY, R, A0, LN, _ = _flat_pieces(bodies)
    P = CX.size
    cands = [A0, np.zeros(1)]
    i, j = np.triu_indices(P, k=1)
    dx, dy, dr = CX[i] - CX[j], CY[i] - CY[j], R[i] - R[j]
    M = np.hypot(dx, dy)
    ok = (M > 1e-15) & (np.abs(dr) <= M)
    if ok.any():
        phi = np.arctan2(dy[ok], dx[ok])
        w = np.arccos(np.clip(-dr[ok] / M[ok], -1.0, 1.0))
        cands.append((phi + w) % TWO_PI)
        cands.append((phi - w) % TWO_PI)
    t = np.unique(np.concatenate(cands) % TWO_PI)
    t = t[np.concatenate([[True], np.diff(t) > 1e-14])]
    dt = np.diff(np.append(t, t[0] + TWO_PI))
    mid = t + 0.5 * dt
    act = ((mid[:, None] - A0[None, :]) % TWO_PI) < LN[None, :]
    h = (CX[None, :] * np.cos(mid)[:, None]
         + CY[None, :] * np.sin(mid)[:, None] + R[None, :])
    win = np.argmax(np.where(act, h, -np.inf), axis=1)
    return t, CX[win], CY[win], R[win]


def max_excess(env_a, env_b):
    """max over theta of ( h_a(theta) - h_b(theta) ).

    <= 0 exactly when body A fits inside body B, since for convex bodies
    A subset B iff h_A <= h_B pointwise.  On each merged cell the difference is
    a single sinusoid  D cos(theta - phi) + dr,  whose maximum is D + dr if phi
    lies in the cell and otherwise sits at an endpoint -- all in closed form.
    """
    ta, ax, ay, ar = env_a
    tb, bx, by, br = env_b
    t = np.unique(np.concatenate([ta, tb]))
    nxt = np.append(t[1:], t[0] + TWO_PI)
    ia = np.searchsorted(ta, t, side="right") - 1
    ib = np.searchsorted(tb, t, side="right") - 1
    dx, dy = ax[ia] - bx[ib], ay[ia] - by[ib]
    dr = ar[ia] - br[ib]
    D = np.hypot(dx, dy)
    phi = np.arctan2(dy, dx)
    inside = ((phi - t) % TWO_PI) < (nxt - t)
    ends = np.maximum(D * np.cos(t - phi), D * np.cos(nxt - phi))
    return float(np.max(np.where(inside, D, ends) + dr))


def fits_inside(inner, outer, tol=0.0):
    return max_excess(envelope(inner), envelope(outer)) <= tol


def hull_area_exact(bodies):
    """Area of conv(union of bodies), with NO sampling anywhere.

    Every piece has support  h(t) = cx cos t + cy sin t + r,  i.e. a single
    sinusoid plus a constant.  Two such curves cross at most twice per period
    and the crossings are available in closed form, so the upper envelope
    max_p h_p(t) has finitely many breakpoints that can all be enumerated:

        breakpoints  =  piece boundaries  u  pairwise crossings

    Between consecutive breakpoints the winning piece is constant, so the hull
    boundary there is exactly one arc.  This removes the failure mode of grid
    sampling, where a body that wins on an angular window narrower than the
    grid step is dropped and the area silently under-reported.
    """
    CX, CY, R, A0, LN, BD = _flat_pieces(bodies)
    P = CX.size

    cands = [A0, np.zeros(1)]

    i, j = np.triu_indices(P, k=1)
    dx, dy, dr = CX[i] - CX[j], CY[i] - CY[j], R[i] - R[j]
    M = np.hypot(dx, dy)
    ok = (M > 1e-15) & (np.abs(dr) <= M)
    if ok.any():
        phi = np.arctan2(dy[ok], dx[ok])
        w = np.arccos(np.clip(-dr[ok] / M[ok], -1.0, 1.0))
        cands.append((phi + w) % TWO_PI)
        cands.append((phi - w) % TWO_PI)

    t = np.unique(np.concatenate(cands) % TWO_PI)
    # drop near-duplicates that would create degenerate zero-width intervals
    t = t[np.concatenate([[True], np.diff(t) > 1e-14])]
    nxt = np.roll(t, -1)
    dt = (nxt - t) % TWO_PI
    dt[-1] = TWO_PI - t[-1] + t[0]
    mid = t + 0.5 * dt

    # winning piece on each interval, evaluated strictly inside it
    act = ((mid[:, None] - A0[None, :]) % TWO_PI) < LN[None, :]
    h = (CX[None, :] * np.cos(mid)[:, None]
         + CY[None, :] * np.sin(mid)[:, None] + R[None, :])
    h = np.where(act, h, -np.inf)
    win = np.argmax(h, axis=1)

    cw, sw, rw = CX[win], CY[win], R[win]
    xs, ys = cw + rw * np.cos(t), sw + rw * np.sin(t)          # interval start
    xe, ye = cw + rw * np.cos(nxt), sw + rw * np.sin(nxt)      # interval end

    # boundary walk: start_0, end_0, start_1, end_1, ...  (equal points where
    # the winner is unchanged contribute nothing to the shoelace)
    x = np.empty(2 * t.size); y = np.empty(2 * t.size)
    x[0::2], x[1::2] = xs, xe
    y[0::2], y[1::2] = ys, ye
    shoe = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    caps = 0.5 * float(np.sum(rw ** 2 * (dt - np.sin(dt))))
    return shoe + caps


_GRID = {}


def hull_area_fast(bodies, n_dirs=256):
    """Vectorised equivalent of hull_area -- same exact value, no Python loop.

    Steps where the winning body is unchanged contribute a circular segment in
    bulk.  Steps where it changes get the crossing angle in closed form and a
    three-edge shoelace patch.  A piece change *within* one body always sits
    within 1e-12 rad of a sample (breaks are injected into the grid), so the
    resulting segment error is O(dt^3) ~ 1e-37 and is simply absorbed.
    """
    nb = len(bodies)
    base = _GRID.get(n_dirs)
    if base is None:
        base = _GRID[n_dirs] = np.linspace(0.0, TWO_PI, n_dirs, endpoint=False)

    # Every break must be STRADDLED, not merely hit: inject b-eps and b+eps so
    # that no full-width step can contain a piece change.  Injecting only b+eps
    # leaves the step ending at it spanning the break with the wrong radius.
    brk = [b.breaks for b in bodies if len(b.breaks) > 1]
    if brk:
        bb = np.concatenate(brk)
        th = np.sort(np.concatenate([base, bb - 1e-11, bb + 1e-11]) % TWO_PI)
    else:
        th = base
    N = th.size
    c, s = np.cos(th), np.sin(th)

    H = np.empty((nb, N))
    Cx = np.empty((nb, N))
    Cy = np.empty((nb, N))
    Rr = np.empty((nb, N))
    for i, b in enumerate(bodies):
        j = b.piece(th)
        Cx[i] = b.centers[j, 0]
        Cy[i] = b.centers[j, 1]
        Rr[i] = b.radii[j]
        H[i] = Cx[i] * c + Cy[i] * s + Rr[i]

    idx = np.arange(N)
    win = np.argmax(H, axis=0)
    cw, sw, rw = Cx[win, idx], Cy[win, idx], Rr[win, idx]
    px, py = cw + rw * c, sw + rw * s

    nxt = (idx + 1) % N
    dt = th[nxt] - th
    dt[-1] += TWO_PI

    shoe = 0.5 * float(np.sum(px * py[nxt] - px[nxt] * py))

    same = win == win[nxt]
    d = dt[same]
    caps = 0.5 * float(np.sum(rw[same] ** 2 * (d - np.sin(d))))

    tr = np.nonzero(~same)[0]
    if tr.size:
        i1 = nxt[tr]
        Co = np.stack([cw[tr], sw[tr]], 1)
        Ci = np.stack([cw[i1], sw[i1]], 1)
        Ro, Ri = rw[tr], rw[i1]
        dv = Co - Ci
        D = np.hypot(dv[:, 0], dv[:, 1])
        rhs = Ri - Ro
        ok = (D > 1e-14) & (np.abs(rhs) <= D)
        phi = np.arctan2(dv[:, 1], dv[:, 0])
        wdt = np.arccos(np.clip(rhs / np.where(D > 0, D, 1.0), -1.0, 1.0))

        step = dt[tr]
        best = np.full(tr.size, np.nan)
        for cand in (phi + wdt, phi - wdt):
            delta = (cand - th[tr]) % TWO_PI
            good = ok & (delta > 0.0) & (delta < step) & np.isnan(best)
            best = np.where(good, delta, best)
        best = np.where(np.isnan(best), 0.5 * step, best)

        tc = th[tr] + best
        cc, ss = np.cos(tc), np.sin(tc)
        Pout = Co + Ro[:, None] * np.stack([cc, ss], 1)
        Pin = Ci + Ri[:, None] * np.stack([cc, ss], 1)

        d1, d2 = best, step - best
        caps += 0.5 * float(np.sum(Ro ** 2 * (d1 - np.sin(d1))))
        caps += 0.5 * float(np.sum(Ri ** 2 * (d2 - np.sin(d2))))

        A = np.stack([px[tr], py[tr]], 1)
        B = np.stack([px[i1], py[i1]], 1)
        cr = lambda a, b: a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1]
        shoe += 0.5 * float(np.sum(cr(A, Pout) + cr(Pout, Pin) + cr(Pin, B) - cr(A, B)))

    return shoe + caps


# ----------------------------------------------------------------------------
# independent cross-check (shares no code with hull_area)
# ----------------------------------------------------------------------------

def hull_area_polyapprox(bodies, n=200000):
    """Independent check: convex hull of densely sampled contact points.

    Shares no code path with hull_area (no support-function maximisation, no
    arc bookkeeping) -- it just samples each body's boundary and lets shapely
    take the hull.  Converges to the true area from below like O(n^-2).
    """
    from shapely.geometry import MultiPoint

    th = np.linspace(0.0, TWO_PI, n, endpoint=False)
    pts = np.concatenate([b.contact(th) for b in bodies], axis=0)
    return MultiPoint(pts).convex_hull.area


def bracket_area(bodies, n_dirs=20000):
    """Rigorous sandwich (inner_area, outer_area) on the hull area.

    inner: convex hull of the contact points at the sampled directions -- a
           subset of the body, so its area is a valid lower bound.
    outer: intersection of the supporting half-planes at those directions -- a
           superset, so its area is a valid upper bound.
    Used as a safety net: hull_area must always land inside this interval.
    """
    th = np.linspace(0.0, TWO_PI, n_dirs, endpoint=False)
    H = np.stack([b.support(th) for b in bodies])
    win = np.argmax(H, axis=0)
    h = H[win, np.arange(len(th))]

    pts = np.empty((len(th), 2))
    for i, b in enumerate(bodies):
        sel = win == i
        if sel.any():
            pts[sel] = b.contact(th[sel])
    x, y = pts[:, 0], pts[:, 1]
    inner = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

    # outer: consecutive supporting lines meet at explicit vertices
    t0, t1 = th, np.roll(th, -1)
    h0, h1 = h, np.roll(h, -1)
    det = np.sin(t1 - t0)
    vx = (h0 * np.sin(t1) - h1 * np.sin(t0)) / det
    vy = (h1 * np.cos(t0) - h0 * np.cos(t1)) / det
    outer = 0.5 * float(np.sum(vx * np.roll(vy, -1) - np.roll(vx, -1) * vy))
    return inner, outer
