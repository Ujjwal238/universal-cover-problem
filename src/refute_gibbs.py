"""Is Gibbs's 0.83699098 a valid lower bound for his own 5-body family?

Gibbs (2014, arXiv:1401.8217) reports 0.83699098 for
    disk + Reuleaux-3 + Reuleaux-5 + Reuleaux-7 + Reuleaux-9
found by simulated annealing, and states it "can only be regarded as an upper
bound on such a lower bound".  If annealing overshot the true placement minimum,
the number is NOT a valid lower bound.  Refuting it needs only ONE placement with
smaller hull area -- far easier than certifying a global minimum -- so this is a
fair and cheap test of his caveat.

Gauge: disk pinned at the origin (rotation-invariant); the residual global
rotation pins R3's orientation.  R3 contributes 2 translations, each of R5/R7/R9
contributes 2 translations + 1 rotation  ->  11 parameters.
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import geom

D  = geom.disk(0.5)
RS = [geom.reuleaux(n, 1.0) for n in (3, 5, 7, 9)]
GIBBS = 0.83699098

def area(v):
    bodies = [D, RS[0].transform(0.0, v[0], v[1])]
    for k in range(3):
        o = 2 + 3*k
        bodies.append(RS[k+1].transform(v[o+2], v[o], v[o+1]))
    return geom.hull_area_exact(bodies)

if __name__ == "__main__":
    rng = np.random.default_rng(11)
    lo = np.array([-0.45,-0.45] + [-0.45,-0.45,0.0]*3)
    hi = np.array([ 0.45, 0.45] + [ 0.45, 0.45,2*np.pi/5]*3)

    best = (np.inf, None)
    print(f"Gibbs 2014 reports {GIBBS} for this family (simulated annealing).")
    print("Searching for ANY placement with smaller hull area...\n", flush=True)

    for trial in range(60):
        v0 = lo + (hi-lo)*rng.random(11)
        r = minimize(area, v0, method="Nelder-Mead",
                     options=dict(xatol=1e-12, fatol=1e-14, maxiter=40000, maxfev=40000))
        r = minimize(area, r.x, method="Nelder-Mead",
                     options=dict(xatol=1e-13, fatol=1e-15, maxiter=40000, maxfev=40000))
        if r.fun < best[0]:
            best = (r.fun, r.x)
            print(f"  trial {trial:3d}: new best {r.fun:.9f}   "
                  f"{'BELOW Gibbs by %.3e' % (GIBBS-r.fun) if r.fun < GIBBS else 'above Gibbs'}",
                  flush=True)

    print("\nDifferential evolution polish...", flush=True)
    de = differential_evolution(area, list(zip(lo, hi)), seed=3, maxiter=400,
                               popsize=28, tol=1e-12, polish=True, init="sobol")
    if de.fun < best[0]:
        best = (de.fun, de.x)
    print(f"  DE best {de.fun:.9f}", flush=True)

    print("\n" + "="*78)
    print(f"  Gibbs 2014 (annealing) : {GIBBS:.9f}")
    print(f"  best found here        : {best[0]:.9f}")
    if best[0] < GIBBS - 1e-9:
        print(f"  => REFUTED as a lower bound: found a placement {GIBBS-best[0]:.3e} SMALLER.")
        print(f"     His caveat was correct; the number overshoots the true minimum.")
    else:
        print(f"  => NOT refuted: no smaller placement found ({best[0]-GIBBS:+.3e}).")
        print(f"     Corroborates his estimate, but is still not a proof.")
    print("="*78)
    np.save("gibbs5_best_v.npy", best[1])
