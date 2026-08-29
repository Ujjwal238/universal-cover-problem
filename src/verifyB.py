"""
STANDALONE VERIFIER for the family-B certificate:  a >= 0.8344.

numpy only.  Imports nothing from the code that produced the certificate -- no
support functions, no upper envelopes, no branch-and-bound, no parallel search.
Everything here is distances, circle-circle intersections, a convex hull, and
the shoelace formula.

WHAT IS PROVED
    Every CONVEX set containing a congruent copy of every planar set of diameter
    1 has area at least TARGET.

TEST SETS, all of diameter exactly 1:
    D  = disk of diameter 1
    R3 = Reuleaux triangle of width 1
    R5 = Reuleaux pentagon of width 1
A body of constant width w has diameter w (diameter = max_theta width), so each
is admissible, a universal cover contains a congruent copy of all three at once,
and being convex it contains their convex hull.  NOTE this needs only the
elementary width/diameter identity -- Vrecica's completion theorem is not used.

    area >= min over placements of area conv(D u g3.R3 u g5.R5)

GAUGE.  D fixed at the origin (two translations gone; D is rotation invariant),
the residual global rotation spent fixing R3's orientation.  Five parameters
remain.  R5 is invariant under rotation by 2pi/5 so rho ranges over [0,2pi/5),
and both Reuleaux bodies are mirror-symmetric so reflections give nothing new.

DOMAIN.  f(d) = (1/4)(pi - arccos(1/2d)) + (1/2)sqrt(d^2-1/4) is the area of
conv(disk of radius 1/2, a point at distance d).  Each body contains its own
centre, so |t| >= d forces area >= f(d); f(0.70) = 0.83655 > TARGET, so
translations outside |t| <= 0.70 need no further argument.

THE CORE.  A width-w Reuleaux polygon on corners V is exactly
intersection_j D(V_j, w).  If every corner moves by at most delta then for any
p in intersection_j D(V_j, w-delta),

    |p - V_j^moved| <= |p - V_j| + |V_j - V_j^moved| <= (w-delta) + delta = w,

so p lies in the moved body.  Triangle inequality only.  Hence the box-centre
body eroded to radius w-delta lies inside EVERY placement in the box, with
delta = |half-diagonal of the translation box| + 2 R sin(hrho/2), R being the
circumradius (a Reuleaux polygon's farthest points are its corners).

The eroded core's own boundary is arcs of radius rho = w-delta centred at the
V_j, meeting at corners obtained as circle-circle intersections.  This verifier
recomputes those corners and CHECKS TWO THINGS the construction can otherwise
get wrong:

    (1) rho >= max_j |V_j - centroid|, else intersection_j D(V_j,rho) is EMPTY
        (min_p max_j |p-V_j| IS that radius);
    (2) every computed corner lies within rho of EVERY V_j, not merely of the
        two circles that produced it.

Omitting either lets a non-empty "core" be fabricated from an empty erosion,
inflating the area bound.  Both are checked below for every leaf.

WITNESS POINTS.  Points are placed along each arc at uniform angular spacing,
plus the corners, plus an inscribed m-gon for the disk.  Their convex hull is
contained in the true hull, so its area is a valid lower bound; the emitter
searched at TARGET + deficit to cover the shortfall, with

    deficit <= eps*perimeter + pi*eps^2,   eps = rho(1-cos(pi/K)) <= 1-cos(pi/K)

recomputed here from the header.

CERTIFICATE.  One bit per node in DFS pre-order: 1 = split, 0 = leaf.  Boxes are
regenerated from the root by the split rule; the verifier checks every leaf
clears TARGET and that the bit stream is consumed exactly.
"""

import multiprocessing as mp
import struct
import sys

import numpy as np

MAGIC = b"LEBCERTB"
TWO_PI = 2.0 * np.pi


def reuleaux_corners(n, w=1.0):
    m = (n - 1) // 2
    circ = w / (2.0 * np.sin(np.pi * m / n))
    a = TWO_PI * np.arange(n) / n
    return np.stack([circ * np.cos(a), circ * np.sin(a)], 1)


V3 = reuleaux_corners(3)
V5 = reuleaux_corners(5)
CIRC3 = float(np.max(np.hypot(V3[:, 0], V3[:, 1])))
CIRC5 = float(np.max(np.hypot(V5[:, 0], V5[:, 1])))


def f_apriori(d):
    if d <= 0.5:
        return np.pi / 4
    return 0.25 * (np.pi - np.arccos(0.5 / d)) + 0.5 * np.sqrt(d * d - 0.25)


def hull_area(P):
    """Monotone chain + shoelace.  No library calls."""
    P = P[np.lexsort((P[:, 1], P[:, 0]))]
    if len(P) < 3:
        return 0.0

    def half(pts):
        out = []
        for p in pts:
            while len(out) >= 2:
                a, b = out[-2], out[-1]
                if (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]) > 0:
                    break
                out.pop()
            out.append(p)
        return out

    H = np.array(half(P)[:-1] + half(P[::-1])[:-1])
    x, y = H[:, 0], H[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))


def core_points(V, rho, K):
    """Witness points on the boundary of intersection_j D(V_j, rho).

    Returns None if the erosion is empty or the reconstructed corners fail the
    all-disk test -- the two checks whose absence permits a fabricated core.
    """
    n = len(V)
    m = (n - 1) // 2
    if rho <= 1e-12:
        return None
    ctr = V.mean(axis=0)
    if rho < float(np.max(np.hypot(V[:, 0]-ctr[0], V[:, 1]-ctr[1]))) - 1e-15:
        return None                                    # (1) erosion is empty

    P = np.empty((n, 2))
    for j in range(n):
        a, b = V[j], V[(j+1) % n]
        d = float(np.hypot(*(b-a)))
        if d < 1e-15 or d > 2.0*rho:
            return None
        u = (b-a)/d
        h2 = rho*rho - 0.25*d*d
        if h2 < 0.0:
            return None
        mid = a + u*(0.5*d)
        off = np.array([-u[1], u[0]])*np.sqrt(h2)
        old = V[(j+m+1) % n]
        P[j] = mid+off if np.hypot(*(mid+off-old)) < np.hypot(*(mid-off-old)) else mid-off


    # arc j is centred at V[j] and runs from P[j-1] to P[j]; sample it at uniform
    # angular spacing, allocating points in proportion to arc span
    pts = [P]
    spans = []
    for j in range(n):
        a0 = np.arctan2(*(P[(j-1) % n] - V[j])[::-1])
        a1 = np.arctan2(*(P[j] - V[j])[::-1])
        sp = (a1 - a0) % TWO_PI
        spans.append((j, a0, sp))
    tot = sum(s for _, _, s in spans) or TWO_PI
    for j, a0, sp in spans:
        k = max(1, int(round(K * sp / tot)))
        t = a0 + sp * np.arange(1, k) / k
        pts.append(V[j] + rho*np.stack([np.cos(t), np.sin(t)], 1))
    W = np.vstack(pts)

    # (2) MEMBERSHIP FILTER.  This is what makes the witness set sound, and it
    # cannot be replaced by an appeal to the arc structure: when an arc span
    # exceeds pi, which happens at rho = circumradius exactly where the core
    # collapses to a point, the construction walks the COMPLEMENTARY arc and
    # emits points far outside the core.  Every point is therefore tested against
    # every disc and anything failing is DROPPED.  What survives is provably
    # inside intersection_j D(V_j, rho), so its hull is a valid inner
    # approximation whatever the arc geometry does.
    d = np.sqrt(((W[:, None, :] - V[None, :, :])**2).sum(-1))
    W = W[d.max(axis=1) <= rho + 1e-12]
    if len(W) < 3:
        return None
    return W


def box_bound(b, ring, K, target=None, hull=hull_area):
    c3x, h3x, c3y, h3y, cr, hr, c5x, h5x, c5y, h5y = b
    ap = max(f_apriori(np.hypot(max(abs(c3x)-h3x, 0.0), max(abs(c3y)-h3y, 0.0))),
             f_apriori(np.hypot(max(abs(c5x)-h5x, 0.0), max(abs(c5y)-h5y, 0.0))))
    if target is not None and ap >= target:
        return ap
    d3 = np.hypot(h3x, h3y)
    d5 = np.hypot(h5x, h5y) + 2.0*CIRC5*np.sin(0.5*hr)
    ca, sa = np.cos(cr), np.sin(cr)
    W3 = core_points(V3 + np.array([c3x, c3y]), 1.0-d3, K)
    W5 = core_points(V5 @ np.array([[ca, sa], [-sa, ca]]) + np.array([c5x, c5y]),
                     1.0-d5, K)
    if W3 is None or W5 is None:
        return ap
    return max(ap, hull(np.vstack([ring, W3, W5])))


def split_cover(b, inflate):
    w = (b[1], b[3], CIRC5*b[5], b[7], b[9])
    k = int(np.argmax(w))
    half = b[2*k+1]*0.5
    out = []
    for s in (-1, 1):
        n = list(b)
        n[2*k] = b[2*k] + s*half
        n[2*k+1] = half*(1.0+inflate)
        out.append(tuple(n))
    return out


def _blk(job):
    si, seed, blk, target, inflate, m, K, fast, sd = job
    a = TWO_PI*np.arange(m)/m
    ring = np.stack([0.5*np.cos(a), 0.5*np.sin(a)], 1)
    if fast:
        from scipy.spatial import ConvexHull
        hull = lambda P: ConvexHull(P).volume
    else:
        hull = hull_area
    rng = np.random.default_rng(sd + si)
    nodes = leaves = xch = 0
    worst, xworst = np.inf, 0.0
    pos = 0
    stack = [seed]
    while stack:
        b = stack.pop()
        bit = (blk[pos >> 3] >> (7 - (pos & 7))) & 1
        pos += 1
        nodes += 1
        if bit == 0:
            leaves += 1
            lb = box_bound(b, ring, K, target, hull)
            if fast and rng.random() < 1e-4:
                xworst = max(xworst, abs(box_bound(b, ring, K, None, hull)
                                         - box_bound(b, ring, K, None, hull_area)))
                xch += 1
            if lb < target:
                return si, False, nodes, leaves, 0.0, xch, xworst, b, lb
            worst = min(worst, lb - target)
        else:
            c1, c2 = split_cover(b, inflate)
            stack.append(c2); stack.append(c1)
    ok = not (pos > 8*len(blk) or (8*len(blk) - pos) >= 8)
    return si, ok, nodes, leaves, worst, xch, xworst, None, None


def verify(path, nproc=8, fast=True, sd=0):
    import time
    with open(path, "rb") as fh:
        assert fh.read(8) == MAGIC, "not a family-B certificate"
        target, thresh, tmax, m, K, depth, nseed = struct.unpack("<dddiiii", fh.read(40))
        (inflate,) = struct.unpack("<d", fh.read(8))
        (nblk,) = struct.unpack("<Q", fh.read(8))
        sizes = [struct.unpack("<I", fh.read(4))[0] for _ in range(nblk)]
        blocks = [fh.read(s) for s in sizes]

    # HEADER SANITY.  The header is data, not a premise.  Two fields decide what
    # the tree actually proves, so both are checked against the theorem rather
    # than trusted:
    #
    #   tmax    the tree tiles |t| <= tmax only.  For the bound to follow, every
    #           placement OUTSIDE that box must already be excluded, which is
    #           exactly f_apriori(tmax) >= target.  A certificate declaring a
    #           smaller tmax covers less of the domain than the theorem needs.
    #   inflate children are widened by (1+inflate) so their union covers the
    #           parent.  A negative value opens gaps between siblings, and the
    #           search would then skip regions it claims to have visited.
    #
    # m and K set only how finely the witness sets approximate the bodies.  Any
    # value is sound: witness points lie in the bodies by construction and are
    # membership-tested, so a coarser grid gives a weaker bound and a finer grid
    # a tighter one, never an invalid one.
    def _f_apriori(d):
        if d <= 0.5:
            return np.pi / 4
        return 0.25 * (np.pi - np.arccos(0.5 / d)) + 0.5 * np.sqrt(d * d - 0.25)

    if not (inflate >= 0.0):
        print(f"  REJECTED: inflate = {inflate!r} is negative; sibling boxes "
              f"need not cover their parent")
        return False
    _ap = _f_apriori(tmax)
    if _ap < target:
        print(f"  REJECTED: domain |t| <= {tmax} is too small for target {target}. "
              f"The a priori bound at the boundary is {_ap:.12f} < {target}, so "
              f"placements outside the box are not excluded and the tree does not "
              f"cover the domain.")
        return False
    print(f"  header sanity: |t| <= {tmax} with a priori bound {_ap:.12f} >= "
          f"{target}; inflate {inflate:g} >= 0")

    # Rigorous perimeter bound: the hull lies within |t| + circumradius of the
    # origin, so diam <= 2(0.70 + 1/sqrt3) and perim <= pi*diam = 8.026.
    eps = 1.0 - np.cos(np.pi / K)
    PERIM = np.pi * 2.0 * (0.70 + 1.0 / np.sqrt(3.0))
    dfc = eps * PERIM + np.pi * eps * eps
    print("=" * 100)
    print(f"VERIFYING  a >= {target}   family B = disk + Reuleaux3 + Reuleaux5")
    print(f"  witness: {m}-gon for the disk, {K} arc points per core")
    print(f"  arc sagitta {eps:.3e}  ->  deficit <= {dfc:.3e}")
    print(f"  emitter searched at {thresh:.12f}; this check uses {target}")
    # ADVISORY ONLY.  The proof is the per-leaf check below: each leaf's bound is
    # computed here from witness points individually verified to lie in the core,
    # over a tree this verifier builds itself.  The deficit bound never enters
    # that argument -- it only predicted whether the emitter's threshold would be
    # generous enough for the leaves to pass.  A shortfall here means the emitter
    # was optimistic, not that the certificate is wrong; the leaves settle it.
    print(f"  advisory: thresh - target = {thresh-target:.3e} vs worst-case deficit "
          f"{dfc:.3e}  ({'ample' if thresh-target >= dfc else 'optimistic -- leaves decide'})")
    print("=" * 100)

    root = (0.0, tmax, 0.0, tmax, np.pi/5, np.pi/5, 0.0, tmax, 0.0, tmax)
    seeds = [root]
    for _ in range(depth):
        seeds = [c for b in seeds for c in split_cover(b, inflate)]
    if len(seeds) != nseed:
        raise ValueError(f"seed count mismatch {len(seeds)} vs {nseed}")

    jobs = [(i, seeds[i], blocks[i], target, inflate, m, K, fast, sd)
            for i in range(nseed)]
    nodes = leaves = xch = 0
    worst, xworst, ok = np.inf, 0.0, True
    t0 = time.time()
    with mp.Pool(nproc) as pool:
        for si, o, nd, lv, w, xc, xw, bad, badlb in pool.imap_unordered(_blk, jobs, chunksize=4):
            nodes += nd; leaves += lv; xch += xc
            xworst = max(xworst, xw); worst = min(worst, w)
            if not o:
                ok = False
                print(f"  *** {'LEAF FAILS bound %.12f < %s' % (badlb, target) if bad is not None else 'BIT-STREAM MISMATCH'} in block {si}")
                break
    el = time.time() - t0
    print(f"\n  nodes replayed {nodes:,}   leaves checked {leaves:,}   "
          f"[{el/60:.1f} min, {leaves/max(el,1e-9):,.0f} leaves/s]")
    print(f"  forest identity 2L - nseed = {2*leaves - nseed:,}  vs nodes {nodes:,}  "
          f"{'OK' if 2*leaves - nseed == nodes else '*** MISMATCH'}")
    if 2*leaves - nseed != nodes:
        ok = False
    if fast:
        print(f"  cross-check: {xch:,} leaves recomputed with the hand-written hull, "
              f"worst disagreement {xworst:.3e}")
        if xworst > 1e-12:
            print("  *** hull implementations disagree -- void"); ok = False
    print(f"  worst leaf slack above target: {worst:.6e}")
    print("\n" + "=" * 100)
    print(f"  {'VERIFIED:  a >= %s' % target if ok else 'VERIFICATION FAILED'}")
    print("=" * 100)
    return ok


if __name__ == "__main__":
    pure = "--pure" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sys.exit(0 if verify(args[0], fast=not pure) else 1)
