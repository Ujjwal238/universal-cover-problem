import sys, numpy as np
from familyA import certify_A
LB3 = 0.833597388099
if __name__ == "__main__":
    print("FAMILY A cost ladder + negative controls   (optimum LB3 = %.9f)" % LB3)
    print(f"{'target':>10}{'margin':>12}{'boxes':>16}{'min':>9}{'box/s':>10}  status")
    rows=[]
    for t in (0.820, 0.826, 0.830, 0.832):
        r = certify_A(t, hmin=1e-5, depth=8, nproc=8, budget=200_000_000, verbose=False)
        rate = r['boxes']/max(r['secs'],1e-9)
        print(f"{t:>10.4f}{LB3-t:>12.6f}{r['boxes']:>16,}{r['secs']/60:>9.2f}{rate:>10,.0f}  "
              f"{'closed' if r['ok'] else 'NOT closed'}")
        sys.stdout.flush()
        if r['ok']: rows.append((LB3-t, r['boxes'], rate))
    print("\nNEGATIVE CONTROLS -- these are ABOVE LB3 and MUST fail:")
    for t in (0.8340, 0.8345):
        r = certify_A(t, hmin=1e-4, depth=8, nproc=8, budget=40_000_000, verbose=False)
        print(f"{t:>10.4f}{LB3-t:>12.6f}{r['boxes']:>16,}{r['secs']/60:>9.2f}{'':>10}  "
              f"{'*** CERTIFIED - UNSOUND' if r['ok'] else 'correctly refused'}")
        sys.stdout.flush()
    if len(rows)>=2:
        x=np.log([r[0] for r in rows]); y=np.log([r[1] for r in rows])
        e=np.linalg.lstsq(np.vstack([np.ones_like(x),-x]).T,y,rcond=None)[0][1]
        m0,n0,rate=rows[-1]
        m=LB3-0.833
        print(f"\n  fitted exponent {e:.3f}")
        print(f"  local exponents: " + ", ".join(
            f"{np.log(a[1]/b[1])/np.log(b[0]/a[0]):.2f}" for a,b in zip(rows,rows[1:])))
        est=n0*(m0/m)**e
        print(f"  target 0.833 (margin {m:.6f}): ~{est:,.0f} boxes ~ {est/rate/3600:.1f} h")
