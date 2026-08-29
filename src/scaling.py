"""
Re-measure certificate cost scaling with the CORRECTED erosion.

The previous exponents (-2.6 in 5-D, -4.9 in 8-D) came from runs whose erosion
fabricated non-empty cores at large delta, so they are void.  The 8-D figure also
rested on only two data points.

PREDICTION TO TEST.  With an erosion deficit ~ c*h at box scale h, a box prunes
once A(centre) >= target + c*h.  Near the optimum A(v) ~ A* + (lambda/2)|v-v*|^2,
so scale-h boxes must tile the ellipsoid where A < target + c*h, of radius
r(h)^2 = 2(c h - m)/lambda with m = A* - target.  The count at scale h is
~(r/h)^d; maximising (c h - m)/h^2 puts the peak at h = 2m/c, giving

    N ~ (c^2 / 2 lambda)^(d/2) * m^(-d/2)

so the exponent should be d/2: 2.5 in 5-D, 4.0 in 8-D, 5.5 in 11-D.  A fit that
lands far from d/2 means the model (or the measurement) is wrong.

Margins are taken against each family's numerically measured optimum, which
involve no erosion:
    family B (disk+R3+R5)      0.834780947
    family D (disk+R3+R5+R7)   0.836494901
Family D's optimum is the weaker of the two -- an 8-D global minimisation with
12M samples -- so its margins carry more uncertainty; flagged in the output.

Cost is reported as boxes AND wall time separately, because 8-D pays twice: more
boxes, and more pieces per hull so fewer boxes per second.
"""

import sys
import time

import numpy as np

from certgen import certify

OPT = {(3, 5): 0.834780947, (3, 5, 7): 0.836494901}

# margins to probe, per family (chosen to span ~1.5 decades at affordable cost)
LADDER = {
    (3, 5): [0.034781, 0.019781, 0.012781, 0.008781, 0.005781, 0.003781, 0.002281],
    (3, 5, 7): [0.030, 0.024, 0.020, 0.016, 0.013, 0.011],
}

# already measured with the corrected code, in the big runs
KNOWN = {((3, 5), 0.000381): (450_922_384, 364.9 * 60)}

BUDGET = 120_000_000


def fit(ms, ns):
    """least-squares exponent of N ~ m^-e"""
    x, y = np.log(np.asarray(ms)), np.log(np.asarray(ns))
    A = np.vstack([np.ones_like(x), -x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef[1]


if __name__ == "__main__":
    print("=" * 112)
    print("COST SCALING with the corrected erosion")
    print("  prediction: exponent = d/2  (2.5 in 5-D, 4.0 in 8-D)")
    print("=" * 112)

    results = {}
    for orders, margins in LADDER.items():
        d = 2 + 3 * (len(orders) - 1)
        opt = OPT[orders]
        print(f"\n>>> family {orders}, dimension {d}, optimum {opt:.9f}")
        print(f"{'margin':>10}{'target':>12}{'boxes':>16}{'min':>9}{'box/s':>10}  status")
        rows = []
        for m in margins:
            t = round(opt - m, 9)
            r = certify(orders, t, hmin=1e-5, cap=120_000, depth=8, nproc=8,
                        budget=BUDGET, verbose=False)
            rate = r["boxes"] / max(r["secs"], 1e-9)
            print(f"{m:>10.6f}{t:>12.6f}{r['boxes']:>16,}{r['secs']/60:>9.2f}{rate:>10,.0f}  "
                  f"{'closed' if r['ok'] else r['reason']}")
            sys.stdout.flush()
            if r["ok"]:
                rows.append((m, r["boxes"], rate))
        for (o, m), (n, s) in KNOWN.items():
            if o == orders:
                rows.append((m, n, n / s))
                print(f"{m:>10.6f}{opt-m:>12.6f}{n:>16,}{s/60:>9.2f}{n/s:>10,.0f}  "
                      f"closed (from the main run)")
        results[orders] = sorted(rows)

    print("\n" + "=" * 112)
    print("EXPONENT FITS")
    for orders, rows in results.items():
        d = 2 + 3 * (len(orders) - 1)
        if len(rows) < 2:
            continue
        ms = [r[0] for r in rows]
        ns = [r[1] for r in rows]
        e = fit(ms, ns)
        print(f"\n  family {orders} (d={d}):  global fit  N ~ m^-{e:.3f}   "
              f"predicted d/2 = {d/2:.1f}   {'CONSISTENT' if abs(e-d/2) < 0.6 else 'DEVIATES'}")
        print(f"    local exponents between consecutive points:")
        for (m1, n1, _), (m2, n2, _) in zip(rows, rows[1:]):
            le = np.log(n1 / n2) / np.log(m2 / m1)
            print(f"      m {m2:.6f} -> {m1:.6f} : {le:.3f}")
        rate = np.median([r[2] for r in rows])
        print(f"    median throughput {rate:,.0f} box/s")

    # what the fits imply for the 8-D prize
    print("\n" + "=" * 112)
    print("IMPLICATION: is 8-D (ceiling 0.836495) reachable above the certified 0.8344?")
    if (3, 5, 7) in results and len(results[(3, 5, 7)]) >= 2:
        rows = results[(3, 5, 7)]
        e = fit([r[0] for r in rows], [r[1] for r in rows])
        m0, n0, _ = rows[-1]
        rate = np.median([r[2] for r in rows])
        for tgt in (0.8340, 0.8344, 0.8346, 0.8350):
            m = OPT[(3, 5, 7)] - tgt
            n = n0 * (m0 / m) ** e
            print(f"    8-D target {tgt}: margin {m:.6f} -> ~{n:.3e} boxes "
                  f"~ {n/rate/86400:,.1f} days at {rate:,.0f} box/s")
    print("=" * 112)
