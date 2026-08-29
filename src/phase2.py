"""
PHASE 2 -- does changing the TEST SETS buy anything?

Phase 1 established that the Brass-Sharifi family {disk, equilateral triangle,
regular pentagon} caps out at LB3 = 0.8335974, i.e. 0.0105 short of the Gibbs
upper bound.  So the choice of test sets, frozen since 2005, is the binding
constraint -- not the certification effort spent on them.

Two levers, both untried in the literature:

  (a) FREE UPGRADE.  By Vrecica's theorem every diameter-1 set extends to a
      body of CONSTANT WIDTH 1, so a universal cover must contain a copy of
      every such body.  The equilateral triangle and the regular pentagon are
      not constant width; their completions (Reuleaux triangle / pentagon)
      strictly contain them.  Swapping them in is valid, strictly stronger,
      and costs no extra dimensions.

  (b) MORE TEST SETS.  Each extra body adds 3 dimensions.  Egan's analysis of
      the hardest case in the Gibbs construction points at the Reuleaux 7-gon.

For a family of m bodies the placement space has dimension 3m-4: the disk pins
two translations and is rotation invariant, and the residual global rotation is
spent fixing the second body's orientation.
"""

import multiprocessing as mp
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from geom import (bracket_area, disk, hull_area_exact, hull_area_polyapprox,
                  regular_polygon, regular_polygon_diameter, reuleaux)

TRI = regular_polygon(3, circumradius=1.0 / np.sqrt(3.0))
PENT = regular_polygon_diameter(5, 1.0)
DISK = disk(0.5)

FAMILIES = {
    "A  disk + triangle + pentagon      (Brass-Sharifi 2005, Xie 2026)":
        [DISK, TRI, PENT],
    "B  disk + Reuleaux3 + Reuleaux5    (free constant-width upgrade)":
        [DISK, reuleaux(3, 1.0), reuleaux(5, 1.0)],
    "C  disk + Reuleaux3 + Reuleaux7    (Egan's hardest-case shape)":
        [DISK, reuleaux(3, 1.0), reuleaux(7, 1.0)],
    "D  disk + Reuleaux3 + Reuleaux5 + Reuleaux7":
        [DISK, reuleaux(3, 1.0), reuleaux(5, 1.0), reuleaux(7, 1.0)],
    "E  disk + Reuleaux3 + Reuleaux5 + Reuleaux7 + Reuleaux9":
        [DISK, reuleaux(3, 1.0), reuleaux(5, 1.0), reuleaux(7, 1.0), reuleaux(9, 1.0)],
}

T = 0.70          # translation half-range


def bounds(m):
    lo = [-T, -T]
    hi = [T, T]
    for _ in range(m - 2):
        lo += [0.0, -T, -T]
        hi += [2 * np.pi, T, T]
    return np.array(lo), np.array(hi)


def build(base, v):
    out = [base[0], base[1].transform(0.0, v[0], v[1])]
    for k in range(2, len(base)):
        rho, tx, ty = v[2 + 3 * (k - 2): 5 + 3 * (k - 2)]
        out.append(base[k].transform(rho, tx, ty))
    return out


def _mk(base, lo, hi):
    def f(v):
        return hull_area_exact(build(base, np.minimum(np.maximum(v, lo), hi)))
    return f


def scan(args):
    name, seed, n = args
    base = FAMILIES[name]
    lo, hi = bounds(len(base))
    s = qmc.Sobol(d=len(lo), scramble=True, seed=seed)
    V = qmc.scale(s.random(n), lo, hi)
    f = _mk(base, lo, hi)
    a = np.fromiter((f(v) for v in V), float, n)
    k = min(80, n)
    idx = np.argpartition(a, k)[:k]
    return V[idx], a[idx]


def refine(args):
    name, starts = args
    base = FAMILIES[name]
    lo, hi = bounds(len(base))
    f = _mk(base, lo, hi)
    out = []
    for v0 in starts:
        r = minimize(f, v0, method="Nelder-Mead",
                     options=dict(xatol=1e-11, fatol=1e-14, maxiter=20000, maxfev=20000))
        r2 = minimize(f, r.x, method="Powell",
                      options=dict(xtol=1e-12, ftol=1e-14, maxiter=40000))
        out.append((min(r.fun, r2.fun),
                    np.minimum(np.maximum(r2.x if r2.fun < r.fun else r.x, lo), hi)))
    return out


def run(name, scan_evals, nstart, nproc=8):
    base = FAMILIES[name]
    m = len(base)
    d = 3 * m - 4
    lo, hi = bounds(m)
    t0 = time.time()

    per = 2 ** 16
    chunks = max(nproc, scan_evals // per)
    with mp.Pool(nproc) as pool:
        res = pool.map(scan, [(name, i, per) for i in range(chunks)])
    V = np.concatenate([r[0] for r in res])
    A = np.concatenate([r[1] for r in res])
    V = V[np.argsort(A)]

    starts = list(V[:nstart])
    with mp.Pool(nproc) as pool:
        ref = pool.map(refine, [(name, starts[i::nproc]) for i in range(nproc)])
    cand = sorted([c for b in ref for c in b], key=lambda t: t[0])
    best_f, best_v = cand[0]

    cfg = build(base, best_v)
    exact = hull_area_exact(cfg)
    shp = hull_area_polyapprox(cfg, n=1_500_000)
    blo, bhi = bracket_area(cfg, n_dirs=200_000)
    at_b = bool(np.any(np.isclose(best_v, lo, atol=1e-6) | np.isclose(best_v, hi, atol=1e-6)))
    near = sum(1 for f, _ in cand if f < best_f + 1e-7)
    return dict(name=name, m=m, d=d, LB=exact, shapely=shp, bracket=(blo, bhi),
                v=best_v, secs=time.time() - t0, at_bound=at_b, near=near,
                nstart=nstart, scan=chunks * per)


if __name__ == "__main__":
    which = sys.argv[1:] or list(FAMILIES)
    GIBBS, BS, XIE, LB3 = 0.8440935944, 0.832, 0.833, 0.833597388099

    print("=" * 108)
    print("PHASE 2 : minimum hull area for different test-set families")
    print(f"          reference: Brass-Sharifi 0.832 | Xie 0.833 | LB3 (measured) "
          f"{LB3:.9f} | Gibbs upper {GIBBS}")
    print("=" * 108)

    rows = []
    for key in which:
        name = next(k for k in FAMILIES if k.startswith(key)) if len(key) < 4 else key
        m = len(FAMILIES[name])
        # dimension grows 3 per body; scale the search budget with it
        budget = {3: 4_000_000, 4: 12_000_000, 5: 24_000_000}[m]
        nstart = {3: 1024, 4: 2048, 5: 3072}[m]
        print(f"\n>>> {name}\n    dim={3*m-4}  scan={budget:,}  starts={nstart} ...")
        r = run(name, budget, nstart)
        rows.append(r)
        print(f"    LB = {r['LB']:.12f}   (shapely {r['shapely']:.12f}, "
              f"diff {abs(r['LB']-r['shapely']):.1e})")
        print(f"    bracket [{r['bracket'][0]:.9f}, {r['bracket'][1]:.9f}]  "
              f"boundary={r['at_bound']}  hits={r['near']}/{r['nstart']}  {r['secs']:.0f}s")

    print("\n" + "=" * 108)
    print(f"{'family':<62}{'dim':>4}{'min hull area':>18}{'vs LB3':>12}{'% of gap':>10}")
    print("-" * 108)
    for r in rows:
        pct = (r["LB"] - BS) / (GIBBS - BS) * 100
        print(f"{r['name']:<62}{r['d']:>4}{r['LB']:>18.12f}{r['LB']-LB3:>+12.6f}{pct:>9.1f}%")
    print("=" * 108)
    best = max(rows, key=lambda r: r["LB"])
    print(f"best family: {best['name']}")
    print(f"  numerical ceiling {best['LB']:.12f}   still short of Gibbs by "
          f"{GIBBS-best['LB']:.6f}")
    np.save("phase2_best_v.npy", best["v"])
