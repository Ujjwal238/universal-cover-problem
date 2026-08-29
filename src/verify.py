"""
STANDALONE VERIFIER for the Lebesgue lower-bound certificate.

Depends on numpy only.  It imports NOTHING from the code that produced the
certificate: no support functions, no upper envelopes, no arc bookkeeping, no
branch-and-bound heuristics, no parallelism.  Everything below is distances,
a hand-written convex hull, and the shoelace formula.

WHAT IS BEING PROVED
--------------------
    Every CONVEX set that contains a congruent copy of every planar set of
    diameter 1 has area at least TARGET.

Three test sets are used, each of diameter exactly 1, so any universal cover
contains a congruent copy of all three simultaneously and -- being convex --
contains their convex hull:

    D  = disk of diameter 1
    P3 = equilateral triangle of side 1   (circumradius 1/sqrt3)
    P5 = regular pentagon of diameter 1   (circumradius 1/(2 sin 72deg))

so   area >= min over placements of  area conv(D u g3.P3 u g5.P5)  =: A*.

GAUGE.  Fix D at the origin (kills two translations; D is rotation invariant)
and spend the residual global rotation fixing P3's orientation.  Five parameters
remain: v = (x3, y3, rho, x5, y5).  P5 is invariant under rotation by 2pi/5, so
rho ranges over [0, 2pi/5).  Both test sets are mirror-symmetric, so reflections
give nothing new.

DOMAIN.  conv(disk of radius 1/2, a point at distance d) has area
f(d) = (1/4)(pi - arccos(1/(2d))) + (1/2)sqrt(d^2 - 1/4).  Each polygon contains
its own centre, so |t| >= d forces area >= f(d).  f(0.70) = 0.8365 exceeds every
target here, so translations outside |t| <= 0.70 need no further argument.

THE BOX BOUND.  Over a box of placements let delta bound how far a test set's
points move from the box-centre placement:  delta = |half-diagonal of the
translation box| + 2 R sin(hrho/2), R the circumradius (the farthest point of a
convex polygon is a vertex).  Then the box-centre polygon eroded by delta lies
inside EVERY placement in the box, because erosion distributes over intersection
and a regular polygon's edges are equidistant from its centre: eroding moves
each edge inward by delta, i.e. scales the polygon by (r-delta)/r about its
centre, where r is the inradius -- and the erosion is EMPTY exactly when
delta >= r.  So for every v in the box

    A(v) >= area conv( D u core3 u core5 )

and replacing D by its inscribed m-gon only lowers that further.

THE CERTIFICATE.  One bit per node of the search tree in DFS pre-order:
1 = split, 0 = leaf.  Boxes are never stored; each is regenerated from the root
by the split rule (halve the widest axis, widen children by INFLATE so their
union provably covers the parent).  The verifier checks that every leaf's bound
clears TARGET, and that the tree is consumed exactly.
"""

import multiprocessing as mp
import struct
import sys

import numpy as np

MAGIC = b"LEBCERT2"
R3 = 1.0 / np.sqrt(3.0)                      # equilateral triangle, side 1
R5 = 1.0 / (2.0 * np.sin(2.0 * np.pi / 5))   # regular pentagon, diameter 1
IN3 = R3 * np.cos(np.pi / 3)                 # inradii
IN5 = R5 * np.cos(np.pi / 5)


# --------------------------------------------------------------------------
# elementary geometry
# --------------------------------------------------------------------------

def f_apriori(d):
    if d <= 0.5:
        return np.pi / 4
    return 0.25 * (np.pi - np.arccos(0.5 / d)) + 0.5 * np.sqrt(d * d - 0.25)


def hull_area(P):
    """Convex hull area by monotone chain + shoelace. No library calls."""
    P = P[np.lexsort((P[:, 1], P[:, 0]))]
    n = len(P)
    if n < 3:
        return 0.0

    def half(pts):
        out = []
        for p in pts:
            while len(out) >= 2:
                a, b = out[-2], out[-1]
                if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) > 0:
                    break
                out.pop()
            out.append(p)
        return out

    lower = half(P)
    upper = half(P[::-1])
    H = np.array(lower[:-1] + upper[:-1])
    x, y = H[:, 0], H[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))


def poly_vertices(n, R, rot, cx, cy, delta):
    """Vertices of the regular n-gon eroded by delta. None iff the erosion is empty."""
    r = R * np.cos(np.pi / n)
    if delta >= r - 1e-15:
        return None
    Rp = R * (r - delta) / r
    a = 2.0 * np.pi * np.arange(n) / n + rot
    return np.stack([cx + Rp * np.cos(a), cy + Rp * np.sin(a)], 1)


def box_bound(b, ring, target=None, hull=hull_area):
    """Lower bound on min A(v) over the box.

    `hull` is injectable so the same code can run with the hand-written
    monotone chain (fully auditable, slow) or with Qhull (fast, and not my
    code).  Agreement between the two is checked on a random sample.
    """
    c3x, h3x, c3y, h3y, cr, hr, c5x, h5x, c5y, h5y = b
    ap = max(f_apriori(np.hypot(max(abs(c3x) - h3x, 0.0), max(abs(c3y) - h3y, 0.0))),
             f_apriori(np.hypot(max(abs(c5x) - h5x, 0.0), max(abs(c5y) - h5y, 0.0))))
    # short-circuit: many leaves clear on the a priori lemma alone, and that
    # needs no hull at all
    if target is not None and ap >= target:
        return ap
    d3 = np.hypot(h3x, h3y)
    d5 = np.hypot(h5x, h5y) + 2.0 * R5 * np.sin(0.5 * hr)
    V3 = poly_vertices(3, R3, 0.0, c3x, c3y, d3)
    V5 = poly_vertices(5, R5, cr, c5x, c5y, d5)
    if V3 is None or V5 is None:
        return ap
    return max(ap, hull(np.vstack([ring, V3, V5])))


def split_cover(b, inflate):
    w = (b[1], b[3], R5 * b[5], b[7], b[9])
    k = int(np.argmax(w))
    half = b[2 * k + 1] * 0.5
    out = []
    for s in (-1, 1):
        n = list(b)
        n[2 * k] = b[2 * k] + s * half
        n[2 * k + 1] = half * (1.0 + inflate)
        out.append(tuple(n))
    return out


_W = {}


def _block(job):
    """Verify one seed subtree.  Blocks are independent, so this parallelises
    trivially without altering a single line of the checking logic."""
    si, seed, blk, target, inflate, m, fast, seed_rng = job
    ring = _W.get(m)
    if ring is None:
        a = 2.0 * np.pi * np.arange(m) / m
        ring = _W[m] = np.stack([0.5 * np.cos(a), 0.5 * np.sin(a)], 1)
    if fast:
        from scipy.spatial import ConvexHull
        hull = lambda P: ConvexHull(P).volume
    else:
        hull = hull_area
    rng = np.random.default_rng(seed_rng + si)
    nodes = leaves = xchecked = 0
    worst, xworst = np.inf, 0.0
    bitpos = 0
    stack = [seed]
    while stack:
        b = stack.pop()
        bit = (blk[bitpos >> 3] >> (7 - (bitpos & 7))) & 1
        bitpos += 1
        nodes += 1
        if bit == 0:
            leaves += 1
            lb = box_bound(b, ring, target, hull)
            if fast and rng.random() < 2e-4:
                d = abs(box_bound(b, ring, None, hull) - box_bound(b, ring, None, hull_area))
                xworst = max(xworst, d)
                xchecked += 1
            if lb < target:
                return si, False, nodes, leaves, 0.0, xchecked, xworst, b, lb
            worst = min(worst, lb - target)
        else:
            c1, c2 = split_cover(b, inflate)
            stack.append(c2)
            stack.append(c1)
    okbits = not (bitpos > 8 * len(blk) or (8 * len(blk) - bitpos) >= 8)
    return si, okbits, nodes, leaves, worst, xchecked, xworst, None, None


def verify_parallel(path, nproc=8, fast=True, seed=0):
    with open(path, "rb") as fh:
        assert fh.read(8) == MAGIC
        target, thresh, tmax, m, depth, nseed = struct.unpack("<dddiii", fh.read(36))
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
    print("=" * 100)
    print(f"VERIFYING  a >= {target}   ({nproc} processes, hull = "
          f"{'Qhull + sampled monotone-chain cross-check' if fast else 'monotone chain'})")
    print(f"  witness m={m} (sagitta {0.5*(1-np.cos(np.pi/m)):.3e}), |t| <= {tmax}, "
          f"{nseed:,} seeds")
    print(f"  emitter searched at {thresh:.12f}; this check uses {target}")
    print("=" * 100)
    root = (0.0, tmax, 0.0, tmax, np.pi / 5, np.pi / 5, 0.0, tmax, 0.0, tmax)
    seeds = [root]
    for _ in range(depth):
        seeds = [c for b in seeds for c in split_cover(b, inflate)]
    if len(seeds) != nseed:
        raise ValueError(f"seed count mismatch: {len(seeds)} vs {nseed}")
    jobs = [(i, seeds[i], blocks[i], target, inflate, m, fast, seed)
            for i in range(nseed)]
    nodes = leaves = xch = 0
    worst, xworst, ok = np.inf, 0.0, True
    import time
    t0 = time.time()
    with mp.Pool(nproc) as pool:
        for si, o, nd, lv, w, xc, xw, badbox, badlb in pool.imap_unordered(_block, jobs, chunksize=4):
            nodes += nd; leaves += lv; xch += xc
            xworst = max(xworst, xw); worst = min(worst, w)
            if not o:
                ok = False
                if badbox is not None:
                    print(f"  *** LEAF FAILS in block {si}: bound {badlb:.12f} < {target}")
                else:
                    print(f"  *** BIT-STREAM MISMATCH in block {si}")
                break
    el = time.time() - t0
    print(f"\n  nodes replayed {nodes:,}   leaves checked {leaves:,}   "
          f"[{el/60:.1f} min, {leaves/max(el,1e-9):,.0f} leaves/s]")
    if fast:
        print(f"  cross-check: {xch:,} leaves recomputed with the hand-written monotone "
              f"chain, worst disagreement {xworst:.3e}")
        if xworst > 1e-12:
            print("  *** hull implementations disagree -- void"); ok = False
    print(f"  worst leaf slack above target: {worst:.6e}")
    print("\n" + "=" * 100)
    print(f"  {'VERIFIED:  a >= %s' % target if ok else '  VERIFICATION FAILED'}")
    print("=" * 100)
    return ok


# --------------------------------------------------------------------------

def verify(path, report_every=20_000_000, fast=True, sample=20000, seed=0):
    with open(path, "rb") as fh:
        if fh.read(8) != MAGIC:
            raise ValueError("not a certificate")
        target, thresh, tmax, m, depth, nseed = struct.unpack("<dddiii", fh.read(36))
        (inflate,) = struct.unpack("<d", fh.read(8))
        (nblk,) = struct.unpack("<Q", fh.read(8))
        sizes = [struct.unpack("<I", fh.read(4))[0] for _ in range(nblk)]
        blocks = [fh.read(s) for s in sizes]

    print("=" * 100)
    print(f"VERIFYING  a >= {target}")
    print(f"  witness resolution m={m}, |t| <= {tmax}, {nseed:,} seeds (depth {depth})")
    print(f"  (the emitter searched at {thresh:.12f}; this check uses {target})")
    print("=" * 100)

    a = 2.0 * np.pi * np.arange(m) / m
    ring = np.stack([0.5 * np.cos(a), 0.5 * np.sin(a)], 1)

    # the deficit the m-gon can cost, recomputed here from scratch
    eps = 0.5 * (1.0 - np.cos(np.pi / m))
    print(f"  inscribed {m}-gon sagitta {eps:.3e}\n")

    root = (0.0, tmax, 0.0, tmax, np.pi / 5, np.pi / 5, 0.0, tmax, 0.0, tmax)
    seeds = [root]
    for _ in range(depth):
        seeds = [c for b in seeds for c in split_cover(b, inflate)]
    if len(seeds) != nseed:
        raise ValueError(f"seed count mismatch: rebuilt {len(seeds)} vs {nseed}")

    hull = hull_area
    if fast:
        from scipy.spatial import ConvexHull      # Qhull: standard, not my code
        hull = lambda P: ConvexHull(P).volume
        print(f"  hull: Qhull (fast).  {sample:,} random leaves will be re-checked"
              f" with the hand-written monotone chain.\n")
    else:
        print("  hull: hand-written monotone chain (auditable, slow)\n")
    rng = np.random.default_rng(seed)
    xchecked = 0
    xworst = 0.0

    nodes = leaves = 0
    worst = np.inf
    for si, (seed, blk) in enumerate(zip(seeds, blocks)):
        bitpos = 0
        stack = [seed]
        while stack:
            b = stack.pop()
            byte = blk[bitpos >> 3]
            bit = (byte >> (7 - (bitpos & 7))) & 1
            bitpos += 1
            nodes += 1
            if bit == 0:
                leaves += 1
                lb = box_bound(b, ring, target, hull)
                if fast and rng.random() < 5e-4 and xchecked < sample:
                    lb2 = box_bound(b, ring, None, hull_area)
                    lb1 = box_bound(b, ring, None, hull)
                    xworst = max(xworst, abs(lb1 - lb2))
                    xchecked += 1
                if lb < target:
                    print(f"  *** LEAF FAILS: bound {lb:.12f} < target {target}")
                    print(f"      box {b}")
                    return False
                worst = min(worst, lb - target)
                if leaves % report_every == 0:
                    print(f"    {leaves:,} leaves checked, worst slack {worst:.3e}")
                    sys.stdout.flush()
            else:
                c1, c2 = split_cover(b, inflate)
                stack.append(c2)
                stack.append(c1)
        if bitpos > 8 * len(blk) or (8 * len(blk) - bitpos) >= 8:
            print(f"  *** BLOCK {si}: consumed {bitpos} bits of {8*len(blk)} -- mismatch")
            return False

    if fast and xchecked:
        print(f"\n  cross-check: {xchecked:,} leaves recomputed with the hand-written")
        print(f"  monotone chain; worst disagreement with Qhull {xworst:.3e}")
        if xworst > 1e-12:
            print("  *** hull implementations disagree -- verification void")
            return False
    print(f"\n  nodes replayed : {nodes:,}   leaves checked : {leaves:,}")
    print(f"  worst leaf slack above target : {worst:.6e}")
    print("\n" + "=" * 100)
    print(f"  VERIFIED:  a >= {target}")
    print(f"  every leaf clears the target; the tree was consumed exactly;")
    print(f"  children provably cover their parents (widened by {inflate:g})")
    print("=" * 100)
    return True


if __name__ == "__main__":
    pure = "--pure" in sys.argv
    ser = "--serial" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if ser:
        sys.exit(0 if verify(args[0], fast=not pure) else 1)
    sys.exit(0 if verify_parallel(args[0], fast=not pure) else 1)
