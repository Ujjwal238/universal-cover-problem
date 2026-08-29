"""How does the certificate cost scale with dimension?

5-D (family B, optimum 0.834781) measured:
    margin 0.008781 ->     345,576 boxes  (0.3 min)
    margin 0.005781 ->   1,252,728 boxes
    margin 0.001181 ->  59,311,656 boxes  (the a >= 0.8336 certificate)
    -> cost ~ margin^-2.6

8-D (family D, optimum 0.836495) is the prize: its ceiling sits +0.0029 above
the banked 0.8336.  But at margin 0.008781 the 8-D run did not close in 9
minutes where 5-D took 18 seconds.  This walks a ladder of generous margins in
8-D to pin down both the dimensional penalty and the 8-D exponent, so the cost
of a useful 8-D target can be extrapolated rather than guessed.
"""

import math
import sys

from certgen import certify

OPT8 = 0.836494901
BUDGET = 60_000_000

if __name__ == "__main__":
    print("=" * 104)
    print("8-D LADDER  (family D = disk + Reuleaux3 + Reuleaux5 + Reuleaux7, dim 8)")
    print(f"  optimum {OPT8};  5-D reference: margin 0.008781 -> 345,576 boxes in 0.3 min")
    print("=" * 104)

    rows = []
    for margin in (0.030, 0.022, 0.016, 0.012, 0.0088):
        target = OPT8 - margin
        print(f"\n>>> margin {margin:.4f}  ->  target {target:.6f}")
        sys.stdout.flush()
        r = certify((3, 5, 7), target, hmin=1e-5, cap=120_000, depth=8,
                    nproc=8, budget=BUDGET)
        rows.append((margin, r))
        print(f"    {'CLOSED' if r['ok'] else 'NOT closed: ' + r['reason']}  "
              f"{r['boxes']:,} boxes, {r['secs']/60:.1f} min")
        sys.stdout.flush()
        if not r["ok"]:
            break

    print("\n" + "=" * 104)
    print(f"{'margin':>10}{'boxes':>18}{'min':>9}   status")
    for m, r in rows:
        print(f"{m:>10.4f}{r['boxes']:>18,}{r['secs']/60:>9.1f}   "
              f"{'closed' if r['ok'] else r['reason']}")

    ok = [(m, r) for m, r in rows if r["ok"]]
    if len(ok) >= 2:
        (m1, r1), (m2, r2) = ok[0], ok[-1]
        e = math.log(r2["boxes"] / r1["boxes"]) / math.log(m1 / m2)
        rate = r2["boxes"] / max(r2["secs"], 1e-9)
        print(f"\n  8-D exponent: cost ~ margin^-{e:.2f}   (5-D was -2.6)")
        print(f"  throughput {rate:,.0f} box/s")
        for tgt in (0.8336, 0.8340, 0.8350, 0.8355):
            mm = OPT8 - tgt
            est = r2["boxes"] * (m2 / mm) ** e
            print(f"    8-D target {tgt}: ~{est:,.0f} boxes ~ {est/rate/3600:,.1f} h")
    print("=" * 104)
