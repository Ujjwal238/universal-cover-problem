<div align="center">

# Curves of constant width and a lower bound for Lebesgue's universal covering problem

### Every convex universal cover for the planar sets of diameter one has area at least 0.8344

**[Ujjwal Mishra](https://scholar.google.co.in/citations?user=Ggw7z6sAAAAJ&hl=en)**

Indian Institute of Information Technology Una

[arXiv](LINK)

</div>

---

## Overview

Lebesgue asked, in a 1914 letter to Pál, for the convex set of least area that
contains a congruent copy of every planar set of diameter one. The problem is
open. This repository contains a machine-checkable proof that the answer is at
least **0.8344**, together with the certificate, an independent verifier, and the
audits.

The previous best *proved* lower bounds were 0.832 (Brass and Sharifi, 2005,
peer reviewed) and 0.833 (Xie, 2026, preprint). The best known upper bound is
0.8440935944 (Gibbs, 2018).

|                        | lower bound  | status |
| ---------------------- | ------------ | ------ |
| Pál, 1920              | 0.8257       | proved |
| Elekes, 1994           | 0.8271       | proved |
| Brass and Sharifi, 2005| 0.832        | proved, peer reviewed |
| Xie, 2026              | 0.833        | proved, preprint |
| **this work**          | **0.8344**   | **proved, certificate included** |

### Credit where it is due

The idea of using **curves of constant width**, and Reuleaux polygons in
particular, as test sets for lower bounds is **not ours**. It is due to Philip
Gibbs, *A New Slant on Lebesgue's Universal Covering Problem*, arXiv:1401.8217
(2014), who states it explicitly, runs it by simulated annealing on a five body
family, and reports 0.83699098. He is careful about what that number is:

> "The answer obtained by simulated annealing can only be regarded as an upper
> bound on such a lower bound."

He is right, and the direction matters. A search *exhibits a placement*, so it
returns an upper bound on the family minimum, which is exactly the quantity a
lower bound proof needs bounded from below. Our own search on his family returned
0.837296, and his 2014 placement at 0.836990 proves that number would have been
invalid to publish as a bound (`refute_gibbs.py`).

**What this work adds is the proof, not the idea**: an exhaustive
branch and bound over the entire placement space, recorded as a certificate whose
verification is logically independent of the search that produced it.

## Result

The cover contains congruent copies of a disk of diameter one, a Reuleaux
triangle of width one, and a Reuleaux pentagon of width one. Each has diameter
one, so each is a legal test set; being convex, the cover contains the convex
hull of any placement of the three. Minimising that hull area over all placements
therefore bounds the cover's area from below.

| quantity | value | source |
| --- | --- | --- |
| certified lower bound | 0.8344 | `logs/reverifyB.log` |
| ceiling of the disk + regular triangle + regular pentagon family | at most 0.8336 | `logs/ceilings.log` |
| ceiling of the disk + Reuleaux triangle + Reuleaux pentagon family | at most 0.834781191 | `logs/ceilings.log` |
| search | 450,922,384 boxes, 364.9 min | `logs/cert8344hard.log` |
| certificate | 486,799,600 nodes, 245,496,952 leaves | `logs/reverifyB.log` |
| worst leaf slack above target | 1.240900e-05 | `logs/reverifyB.log` |
| rigorous floating point error bound | 9.287e-12 | `logs/errbound.log` |
| measured double vs 60 digit error | 2.998e-15 | `logs/errbound.log` |

The first ceiling is what makes the result structural rather than a harder
search. Exhibiting one arrangement of the disk, equilateral triangle and regular
pentagon whose hull has area below 0.8336 proves that no computation applied to
those three sets can ever reach 0.8344, whatever search is used. Those are the
test sets of Pál (1920), Brass and Sharifi (2005) and Xie (2026). The bound is
rigorous rather than numerical: the area is measured with an outer polygon that
contains the hull, so it errs upward by construction.

## Method

Three ingredients.

**The erosion core lemma.** A Reuleaux polygon of width `w` is the intersection
of discs of radius `w` about its corners. If every corner moves by at most `δ`,
the intersection of the discs of radius `w − δ` about the *original* corners lies
inside *every* placement, by the triangle inequality. One hull computation
therefore bounds an entire box of placements from below. This is what makes
rigorous branch and bound possible for bodies with curved boundaries.

**A priori domain bound.** The corners of a body of diameter one already sit at
distance 0.53 to 0.58 from its own centre. Applying the area estimate to the
farthest corner rather than the centre confines the translations to
`|t| ≤ 0.1924` and `|t| ≤ 0.1959`, a 169 fold reduction in 4-volume against the
naive `|t| ≤ 0.70` (`src/domain.py`). This reproduces, independently, the
`[−0.19, 0.19]⁴` box stated by Brass and Sharifi.

**Certificate.** The branch and bound tree is stored as one bit per node in DFS
pre-order: 1 for a split, 0 for a pruned leaf. Boxes are never stored. The
verifier regenerates every box from the root using the deterministic split rule,
so 486,799,600 nodes cost 486,799,600 bits. At each leaf it recomputes the bound
by elementary means only, inscribed polygons, the triangle inequality and the
shoelace formula, and checks it against the target.

## Reproducing

```bash
conda env create -f environment.yml && conda activate lebesgue
cd src
```

**Minutes.** Enough to see every part of the argument work.

```bash
python test_geom.py                                  # 28 kernel gates
python domain.py                                     # the domain lemma
python ceilings.py                                   # both family ceilings
python errbound.py 256 256 6                         # floating point error bound
python verifyB.py ../certificates/controls/certB_test.bin   # verifier, small certificate
```

**An hour or two.** The audits and the controls.

```bash
python audit.py                          # 29 checks, family B
python auditA.py                         # 29 checks, family A
python tamper.py /tmp/controls           # 22 controls, both directions
python errbound.py 1024 1024 8           # error bound at full resolution
```

**Fifteen hours on eight cores.** The headline certificate itself.

```bash
gunzip -c ../certificates/cert_B_08344.bin.gz > /tmp/cert_B_08344.bin
shasum -a 256 -c <(grep 'cert_B_08344.bin$' ../certificates/MANIFEST.sha256)
python verifyB.py /tmp/cert_B_08344.bin
```

Expect `VERIFIED:  a >= 0.8344`, the identity `2L - nseed = nodes` holding
exactly, and a worst leaf slack of `1.240900e-05`.

The verifier is independent of the search: it shares no bound code with the
emitter and reads only the header and the bit stream. It validates the header
rather than accepting it, rejecting a declared domain too small for the target or
a non-positive child inflation. See `certificates/REGENERATE.md` to rebuild the
certificates from scratch.

## Validation

The verifier shares no bound code with the search: it reads only the bit stream
and the header, and recomputes every leaf bound from scratch by elementary means.
Independently of that, the following hold.

- **Independent reimplementation.** The leaf bound is implemented a second time
  against a different hull routine (`src/independent.py`), and the two agree.
- **Published constants, reproduced by code never tuned to them.** Pál's
  `π/8 + √3/4` to 1.1e-16, and Brass and Sharifi's Figure 2 hull to 3.2e-6.
- **Controls in both directions.** `src/tamper.py` rebuilds corrupted
  certificates from the valid ones and checks each is refused, and checks that
  valid certificates and legitimate tightenings are still accepted. 22 of 22
  behave correctly (`logs/controls_final.log`). A verifier that accepted
  everything, or refused everything, would fail this.
- **Header validation.** The header is data, not a premise. Both verifiers check
  the declared domain against the a priori bound, `f(tmax) >= target`, and reject
  a non-positive child inflation, rather than accepting either field as given. A
  certificate declaring too small a domain, or an inflation that leaves gaps
  between sibling boxes, is refused.
- **Structural identities.** For a forest of independent seed trees,
  `nodes = 2·leaves − seeds`. This holds exactly on every verification run, so no
  subtree can be silently dropped.
- **Second-routine cross-check.** 24,704 leaves recomputed with a hand written
  hull, worst disagreement 6.046e-13.
- **High precision.** The leaf bound recomputed in 60 digit arithmetic
  (`src/errbound.py`) agrees with double precision to 2.998e-15, against a
  rigorous error bound of 9.287e-12 and a worst leaf slack of 1.240900e-05.
- **Kernel gates.** 28 property tests on the geometry kernel (`src/test_geom.py`)
  and 29 audit checks per family (`src/audit.py`, `src/auditA.py`).

## Limits

The gap `[0.8344, 0.8440936]` is open, and Gibbs's heuristics put the truth near
0.84408. Adding a fourth test body raises the parameter count from 5 to 8, where
the measured cost scaling (`logs/scaling.log`) puts certification out of reach by
several orders of magnitude. The arithmetic is double precision with a rigorous
error bound, not interval arithmetic.

## Layout

```
src/            geometry kernel, emitters, verifiers, audits, experiments
certificates/   gzipped certificates, checksums, controls, regeneration steps
logs/           every run referenced above
docs/           literature review
paper/          LaTeX source, figures, and the compiled paper
```

On the proof path:

| file | role |
| --- | --- |
| `geom.py` | geometry kernel: bodies, support functions, exact hull areas |
| `domain.py` | the a priori domain lemma |
| `ceilings.py` | rigorous ceilings for both families, by outer polygons |
| `certgen.py`, `certhard.py` | the branch and bound search |
| `certemitB.py`, `certemit.py` | certificate emitters, families B and A |
| `verifyB.py`, `verify.py` | independent verifiers |
| `errbound.py` | floating point error bound, and a 60 digit recomputation |
| `tamper.py` | builds the controls and checks both directions |
| `audit.py`, `auditA.py`, `test_geom.py` | audits and kernel gates |

Supporting measurements, not on the proof path: `scaling.py`, `s2compare.py`,
`s2new.py`, `certB2.py`, `ladderA.py`, `ladder8.py` for cost scaling;
`phase1.py`, `phase2.py` for the searches that located the arrangements; `independent.py` for the second implementation of the leaf bound;
`refute_gibbs.py` for the comparison against the five-body family;
`familyA.py`, `certify.py`, `controls.py` for family A.

## Citation

```bibtex
@misc{mishra2026lebesgue,
  title  = {A Certified Lower Bound for Lebesgue's Universal Covering Problem},
  author = {Mishra, Ujjwal},
  year   = {2026},
  eprint = {arXiv:TBD}
}
```

Please also cite the paper that had the idea:

```bibtex
@misc{gibbs2014slant,
  title  = {A New Slant on Lebesgue's Universal Covering Problem},
  author = {Gibbs, Philip},
  year   = {2014},
  eprint = {arXiv:1401.8217}
}
```

## Contact

[ujjwalmishra238@gmail.com](mailto:ujjwalmishra238@gmail.com)
