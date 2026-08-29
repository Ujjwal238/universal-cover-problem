"""
PHASE 1 -- the true optimum of the Brass-Sharifi three-test-set configuration.

    a  >=  LB3  :=  min_v  area( conv( D  u  T(x3,y3)  u  P(rho,x5,y5) ) )

with  D = disk of diameter 1 centred at the origin
      T = equilateral triangle of side 1, fixed orientation, translated
      P = regular pentagon of diameter 1, rotated by rho and translated

This is exactly the domain of Brass-Sharifi (2005) and of Xie (arXiv 2606.04458),
normalised the same way: the disk pins two translations and is rotation
invariant, and the residual global rotation is spent fixing the triangle's
orientation, leaving the 5 parameters above.

Brass-Sharifi proved  a >= 0.832 ;  Xie proved  a >= 0.833 .
NEITHER reports  LB3  itself.  LB3 is the ceiling of this entire framework:
no amount of extra certification effort with these three test sets can ever
prove more than LB3.  That number is what this script measures.
"""

import multiprocessing as mp
import time

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from geom import (bracket_area, disk, hull_area, hull_area_exact,
                  hull_area_fast, hull_area_polyapprox, regular_polygon,
                  regular_polygon_diameter)

SQ3 = np.sqrt(3.0)
TRI = regular_polygon(3, circumradius=1.0 / SQ3)        # equilateral, side 1
PENT = regular_polygon_diameter(5, 1.0)                 # regular, diameter 1
DISK = disk(0.5)

# rho only needs a fifth of the circle (pentagon has 5-fold symmetry)
LO = np.array([0.0, -0.70, -0.70, -0.70, -0.70])
HI = np.array([2 * np.pi / 5, 0.70, 0.70, 0.70, 0.70])


def config(v):
    rho, x3, y3, x5, y5 = v
    return [DISK, TRI.transform(0.0, x3, y3), PENT.transform(rho, x5, y5)]


def area(v):
    # hull_area_exact, not the sampled variant: a grid can drop a body that
    # wins on a window narrower than the step, under-reporting the area, and a
    # minimiser will happily steer into exactly that error.
    return hull_area_exact(config(v))


def _clip(v):
    return np.minimum(np.maximum(v, LO), HI)


def objective(v):
    return area(_clip(v))


# ---------------------------------------------------------------- worker ----

def scan_chunk(args):
    seed, n = args
    s = qmc.Sobol(d=5, scramble=True, seed=seed)
    V = qmc.scale(s.random(n), LO, HI)
    a = np.empty(n)
    for i in range(n):
        a[i] = area(V[i])
    k = min(60, n)
    idx = np.argpartition(a, k)[:k]
    return V[idx], a[idx]


def refine_chunk(args):
    seed, starts = args
    out = []
    for v0 in starts:
        r = minimize(objective, v0, method="Nelder-Mead",
                     options=dict(xatol=1e-11, fatol=1e-14, maxiter=6000, maxfev=6000))
        r2 = minimize(objective, r.x, method="Powell",
                      options=dict(xtol=1e-12, ftol=1e-14, maxiter=20000))
        v = _clip(r2.x if r2.fun < r.fun else r.x)
        out.append((min(r.fun, r2.fun), v))
    return out


if __name__ == "__main__":
    NPROC = 8
    t0 = time.time()

    print("=" * 100)
    print("PHASE 1 : global minimisation of the Brass-Sharifi 3-test-set area")
    print("=" * 100)

    # ---- stage A: quasi-random global scan --------------------------------
    CHUNKS, PER = 128, 2 ** 16         # ~8.4M evaluations
    print(f"\n[A] Sobol scan: {CHUNKS * PER:,} placements over the 5-D domain "
          f"on {NPROC} cores ...")
    with mp.Pool(NPROC) as pool:
        res = pool.map(scan_chunk, [(i, PER) for i in range(CHUNKS)])
    V = np.concatenate([r[0] for r in res])
    A = np.concatenate([r[1] for r in res])
    order = np.argsort(A)
    V, A = V[order], A[order]
    print(f"    best from raw scan      : {A[0]:.12f}   ({time.time()-t0:.0f}s)")

    # ---- stage B: local refinement from the best basins -------------------
    NSTART = 2048
    starts = list(V[:NSTART])
    print(f"\n[B] Local refinement (Nelder-Mead + Powell) from {NSTART} starts ...")
    batches = [(i, starts[i::NPROC]) for i in range(NPROC)]
    with mp.Pool(NPROC) as pool:
        ref = pool.map(refine_chunk, batches)
    cand = sorted([c for b in ref for c in b], key=lambda t: t[0])
    best_f, best_v = cand[0]
    print(f"    best after refinement   : {best_f:.12f}   ({time.time()-t0:.0f}s)")

    # how many distinct basins land near the optimum?
    near = [c for c in cand if c[0] < best_f + 1e-9]
    print(f"    starts reaching within 1e-9 of the best: {len(near)} / {NSTART}")
    uniq, seen = [], []
    for f, v in cand:
        if f > best_f + 5e-4:
            break
        if not any(np.linalg.norm(v - u) < 1e-4 for u in seen):
            seen.append(v); uniq.append((f, v))
    print(f"    distinct local minima within 5e-4 of the best: {len(uniq)}")
    for f, v in uniq[:8]:
        print(f"       {f:.12f}  rho={np.degrees(v[0]):8.4f}deg  "
              f"tri=({v[1]:+.5f},{v[2]:+.5f})  pent=({v[3]:+.5f},{v[4]:+.5f})")

    # ---- stage C: verify the winner independently -------------------------
    print("\n[C] Verification of the optimal configuration")
    rho, x3, y3, x5, y5 = best_v
    cfg = config(best_v)

    a_ex = hull_area_exact(cfg)
    a_ref = hull_area(cfg, n_dirs=16384)
    a_shp = hull_area_polyapprox(cfg, n=3_000_000)
    lo, hi = bracket_area(cfg, n_dirs=400_000)
    print(f"    hull_area_exact (no sampling): {a_ex:.15f}")
    print(f"    hull_area ref  (16384 dirs)  : {a_ref:.15f}   diff {abs(a_ex-a_ref):.2e}")
    print(f"    shapely, 3e6 samples         : {a_shp:.15f}   diff {abs(a_ex-a_shp):.2e}")
    print(f"    rigorous bracket             : [{lo:.12f}, {hi:.12f}]  contains: {lo<=a_ex<=hi}")

    print(f"\n    optimal placement v* :")
    print(f"      rho (pentagon rotation) = {rho:.12f} rad = {np.degrees(rho):.9f} deg")
    print(f"      triangle centre         = ({x3:.12f}, {y3:.12f})   |t| = {np.hypot(x3,y3):.12f}")
    print(f"      pentagon centre         = ({x5:.12f}, {y5:.12f})   |t| = {np.hypot(x5,y5):.12f}")
    at_bound = np.any(np.isclose(best_v, LO, atol=1e-6) | np.isclose(best_v, HI, atol=1e-6))
    print(f"      touches domain boundary : {at_bound}")

    # ---- verdict ----------------------------------------------------------
    LB3 = a_ex
    BS, XIE, GIBBS = 0.832, 0.833, 0.8440935944
    print("\n" + "=" * 100)
    print(f"  LB3  (true ceiling of the 3-test-set framework)  =  {LB3:.12f}")
    print("=" * 100)
    print(f"  Brass-Sharifi 2005 certified   0.832        -> headroom left unclaimed: {LB3-BS:+.6f}")
    print(f"  Xie 2026 certified             0.833        -> headroom left unclaimed: {LB3-XIE:+.6f}")
    print(f"  Gibbs 2018 upper bound         {GIBBS}  -> framework falls short by:   {GIBBS-LB3:.6f}")
    print(f"  fraction of the 0.832->0.844 gap this framework can ever close: "
          f"{(LB3-BS)/(GIBBS-BS)*100:.1f}%")
    if LB3 < XIE:
        print("\n  *** LB3 < 0.833 : Xie's certified bound EXCEEDS the framework optimum.")
        print("      One of the two is wrong. Needs a careful independent check.")
    print(f"\n  total wall clock: {time.time()-t0:.0f}s")
    np.save("phase1_best_v.npy", best_v)
