"""Second half of the S2 comparison: the sharp scheme alone.

The old scheme was measured separately (s2compare_old0833.log): 25,425,568 boxes
at target 0.833, depth 18, hmin 1e-5.  Everything here is identical except the
root box and the a priori estimate, so the two box counts are comparable.  Run on
its own; two 8-worker pools on 8 cores distort both.
"""
import s2compare as S

if __name__ == "__main__":
    print("=" * 100)
    print("S2, sharp scheme only.  Baseline (old scheme, same settings): 25,425,568 boxes")
    print("=" * 100)
    n, stuck, el = S.run(0.833, True, 18, 1e-5)
    print()
    print(f"  old  25,425,568 boxes")
    print(f"  new  {n:,} boxes   ({stuck} stuck)")
    print(f"  ratio {25425568/max(n,1):.2f}x fewer boxes with the farthest-corner domain")
