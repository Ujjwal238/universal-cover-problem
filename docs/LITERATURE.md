# Literature review — Lebesgue's universal covering problem

Purpose: establish whether the improvement used in this project (replacing the
2005 polygonal test sets with their **constant-width completions**) is already
published. Searched 2026-08-14.

---

## 1. The problem

A *universal cover* is a **convex** compact set U ⊂ R² such that every planar set
of diameter 1 is isometric to a subset of U. Let

    a = inf { area(U) : U a universal cover }.

Posed by Lebesgue in a 1914 letter to Pál. Convexity matters: Duff (1980) showed
non-convex covers can be smaller, so every bound below is for the convex problem.

---

## 2. Published UPPER bounds (well documented)

| year | author | area | note |
|---|---|---|---|
| 1920 | Pál | √3/2 = 0.8660254038 | regular hexagon, inradius 1/2 |
| 1920 | Pál | 2 − 2/√3 = 0.8452994616 | two corners removed |
| 1936 | Sprague | 0.844137708436 | third corner region |
| 1992 | Hansen | 0.844137708398 | two slivers; claimed 4e-11 and 6e-18, actual 3.7507e-11 and 8.4541e-21 (corrected by BBG) |
| 2015 | Baez–Bagdasaryan–Gibbs | 0.84411529712841905 | slant angle σ = 1.294389444703601012°, checked by Greg Egan at 2000 digits |
| 2018 | Gibbs | **0.8440935944** | current record, arXiv:1810.10089 |

---

## 3. Published LOWER bounds — complete, from the Brass–Sharifi full text

| year | author | value | test sets | venue |
|---|---|---|---|---|
| 1920 | Pál | π/8 + √3/4 ≈ **0.8257** | circle + equilateral triangle, concentric | Danske Mat.-Fys. Medd. III 2 |
| 1994 | **Elekes** | ≈ **0.8271** | circle + all regular 3^i-gons, concentric and aligned | Discrete Comput. Geom. 12:439–449 |
| 2005 | **Brass & Sharifi** | **0.832** | circle + equilateral triangle + regular pentagon | IJCGA 15(5):537–544 — *peer reviewed* |
| 2026 | Xie | 0.833 | same three | arXiv:2606.04458 — *unrefereed preprint* |
| 2026 | this work | **0.8344** | circle + **Reuleaux** triangle + **Reuleaux** pentagon | — *this is the best PROVED bound* |

**Separately, not a proved bound:** Gibbs (2014), arXiv:1401.8217, reports
**0.83699098** for circle + Reuleaux 3,5,7,9 by simulated annealing, explicitly
"an upper bound on such a lower bound" -- a heuristic estimate, higher than our
certified 0.8344 but never proved. See 6a. Consistent: more test bodies -> larger
true bound; his 11-D family is far beyond certification reach.

### Elekes 1994
G. Elekes, *Generalized breadths, Cantor-type arrangements and the least area
UCC*, Discrete Comput. Geom. **12** (1994) 439-449, gives approximately 0.8271.
The Wikipedia bibliography omits it; Brass and Sharifi cite it as their ref. 13.
It should be cited.

### Consequence for how progress should be described
The *peer-reviewed* lower bound has not moved since **2005**. Xie's 0.833 is an
arXiv preprint (v1 June 2026, v4 July 2026), single author, using nonstandard
terminology and reporting no runtime. Saying "the bound moved +0.001 in 21
years" is imprecise: the refereed bound moved by **zero**.

---

## 3b. External validation of this project's geometry kernel

Both published constants reproduce, on code never tuned to them:

| quantity | published | this kernel | agreement |
|---|---|---|---|
| Pál, π/8 + √3/4 | 0.825711783590943 | 0.825711783590944 | 1.1e-16 |
| Brass–Sharifi Fig. 2 hull | 0.833646 | 0.833649208 | 3.2e-6 (their coords are 6-figure) |

Their Fig. 2 placement lies in the same basin as my measured optimum
(pentagon rotation 61.045° vs 60.008°; translations ~0.006–0.020 vs
~0.005–0.021), and my LB3 = 0.833597388099 is **lower by 4.9e-5** — as it must
be, since they state "we did not make any search for the best placement, but
obtained this only as the best among the centers of boxes covering the space of
placements."

Their framework is identical to the one derived independently here: five
parameters, circle fixed at the origin, triangle orientation gauge-fixed,
pentagon carrying three degrees of freedom.

## 4. The test sets, 2005 → 2026: unchanged

Brass–Sharifi obtained 0.832 "by combining geometric estimates with a computer
search over placements of a **disk, an equilateral triangle, and a regular
pentagon**" of diameter one.

Xie (2026) states explicitly: *"We work in the convex Brass–Sharifi three-test-set
framework, where the test sets are a closed disk, an equilateral triangle, and a
regular pentagon of diameter one."* He improves only the **certification**, not
the shapes.

Checked directly in Xie's introduction/related work:
- lower-bound history cited: **only Brass–Sharifi**;
- alternative or improved test sets: **no discussion**;
- Reuleaux polygons / constant-width bodies as test sets: **no mention**;
- whether the three-test-set choice could be improved: **no comment**.

---

## 5. Where constant width DOES appear in the literature

Constant width is central to this problem — but consistently in the
**sufficiency (upper-bound) direction**, never as lower-bound test sets:

- **Vrećica (1981)**, *A note on sets of constant width*, Publ. Inst. Math. 29:
  every planar set of diameter 1 extends to a curve of constant width 1.
- Baez–Bagdasaryan–Gibbs use exactly this in their **Reduction 2**: "a set will
  be a universal covering if it contains an isometric copy of every curve of
  constant width 1." That is the *converse* direction — used to prove a candidate
  cover is universal, i.e. to build **upper** bounds.
- Egan's analysis (n-Category Café, 2015) notes a Reuleaux 7-gon nearly fills the
  space in the hardest case of the Gibbs construction — again upper-bound work.

**The direction used in this project is the trivial one** (constant width w ⟹
diameter w, so such bodies are admissible test sets) and it is the one that
appears nowhere in the lower-bound literature.

---

## 6. Nearest prior art

### 6a. Gibbs 2014 - the decisive prior art
**Philip Gibbs, "A New Slant on Lebesgue's Universal Covering Problem",
arXiv:1401.8217 (2014).**

The idea of using curves of constant width, and Reuleaux polygons in particular,
as lower-bound test sets is stated explicitly in this paper and carried out in
it. Verbatim:

> "The optimal shapes to use are curves of constant width. Indeed any shape of
> diameter one is contained within a curve of constant width equal to one [14],
> so it is sufficient to consider only curves of constant width in the covering
> problem. **Reuleaux polygons are the most easily constructed curves of constant
> width and empirically they appear to be most effective in maximising the
> minimum area of a cover for a given number of shapes.**"

He states our exact lower-bound lemma (line 133):

> "The minimum area for any given set of shapes of diameter one is a lower bound
> on the minimal convex cover."

He runs it on disk + Reuleaux-3 + Reuleaux-5 + Reuleaux-7 + Reuleaux-9 by
simulated annealing and reports (lines 125-137):

> "**the minimal convex area found was 0.83699098.**"

He also has the correct Reuleaux manifold dimension count -- "It is sufficient to
specify n-3 consecutive angles" (line 105) -- the same n-3 this project initially
got wrong as n-1 and later corrected.

His completion citation is **[14] Grunbaum, "Borsuk's Problem and Related
Questions", AMS Proc. Symp. Pure Math. VII (1963) 271-284** -- earlier than the
Vrecica 1981 reference used here. Cite Grunbaum.

### What Gibbs 2014 does NOT contain -- and this is the whole remaining gap
He is explicit that his number is not a proof (lines 133-137):

> "The answer obtained by simulated annealing **can only be regarded as an upper
> bound on such a lower bound**, but with multiple simulations and a small number
> of shapes the best area can be found with some certainty and accuracy and **can
> be regarded as an empirical lower bound**."

Simulated annealing that *overshoots* the true placement minimum reports a number
that is NOT a valid lower bound. Nothing in the paper bounds that error.

He does use the word "rigorous" once (line 275), and it must not be
over-read: "At any time this gives a rigorous lower bound for the minimum cover
**included in a given hexagon**." That computation is (i) *conditional* on his
unproved "modified Pal hypothesis" that the optimal cover sits in a parallel
hexagon, and (ii) still uses heuristic/greedy inner search over placements. He
concedes the limitation himself (line 428): "it must be stressed that these
points are only lower bounds ... **the ability to reach the true minimum area
could be limited by the choice of possible shapes**."

He never reports a number for the 3-body family disk + R3 + R5, and never
produces an unconditional, exhaustive bound over a placement space.

**Venue.** Not peer reviewed. He searched for a journal and found none accepting
unaffiliated authors, and closes: "I therefore apologise for any faults this
article may contain."

**Attribution -- mandatory and prominent.** The constant-width/Reuleaux test-set
idea is **Gibbs's, published 2014**. This project rediscovered it independently
eleven years later. Any writeup that presents it as new is wrong. The paper must
credit Gibbs 2014 in the abstract, not a footnote.

### 6b. Khandhawit, Pagonakis & Sriswasdi (2011) — different problem
arXiv:1101.5638, later IJCGA 23 (2013) 197–212. "Lower Bound for Convex Hull
Area and Universal Cover Problems." Proves ≥ 0.232239 for unit-length curves and
≥ 0.0879873 for closed unit curves — **Moser's worm family, not diameter-1
sets** — via a convex-hull-of-points-and-a-rectangle estimate. Not this method.

### 6c. Related-problem lower bounds (different problem, listed for completeness)
- convex cover for closed unit curves ≥ 0.0975 (arXiv:1905.00333)
- convex cover for closed unit curves ≥ 0.1 (arXiv:2004.03063)
- asymptotic Lebesgue covering (arXiv:2512.04023)

---

## 6d. What Brass–Sharifi themselves say about improving the test sets

Decisive for novelty, and now read directly rather than inferred:

- They know the family can be extended: *"This lower bound could be improved if
  one could add further sets of diameter 1 to this family... This was already
  observed by Pál, but he found unsurmountable difficulties in extending his
  method from two sets to three sets."*
- They identify the dimension barrier that this project later measured:
  *"adding another set would raise the dimension of the search space to eight and
  make our approach again infeasible."* (Measured here: 8-D costs ~10^11 boxes.)
- They want **better polygons**, never constant-width bodies: *"It would have
  been much more efficient if one could have taken circle, equilateral triangle,
  and regular fivegon, of diameter 1; but the analytic methods do not extend."*
- Their open problem is about translations, not shapes: *"One could use a much
  larger family of test sets if one could determine the minimum for the
  translations directly."* With Rote's observation that the area is convex in the
  translations for k = 2 but not for k ≥ 3.

**They were aware of Reuleaux triangles** — they cite Eggleston's observation
that the union of a Reuleaux triangle and a circle of diameter 1 with antipodal
vertices is a universal cover. That is an **upper**-bound construction. Nowhere
do they consider a Reuleaux body as a **test set**.

## 6e. A sharper domain lemma

Brass and Sharifi restrict translations to **[-0.19, 0.19]^4**. Their argument
applies the area estimate to a body's *farthest vertex* rather than its centre,
which is much stronger, since the corners of a body of diameter one already sit
at distance 0.53 to 0.58 from its own centre before any translation.

Derived independently here (`src/domain.py`) the lemma gives |t| <= 0.192366 and
|t| <= 0.195891 for the Reuleaux triangle and pentagon at target 0.8344, a 169
fold reduction in translation 4-volume against |t| <= 0.70. That it lands on
their stated 0.19 is a useful check that the derivation reproduces theirs.

**It does not reduce certificate cost.** Measured directly at target 0.833 with
everything else held fixed: 25,425,568 boxes with the old domain against
27,350,260 with the sharp one (`logs/s2compare_old0833.log`, `logs/s2new.log`).
The two estimates coincide near the optimum, where the hull term dominates and
the a priori term is slack, and that neighbourhood is what the box count is made
of. The sharp lemma removes far-field boxes, which were already cheap. It is
worth stating because it is the tight description of the search domain, not
because it is faster.

## 6f. The Brass–Sharifi citation graph — enumerated (Google Scholar, 2026-08-14)

All ~33 citing works, classified. **None improves the planar lower bound except
Xie, and none uses constant-width bodies as test sets.**

**Lebesgue planar, LOWER bound (the only category that could be prior art)**
- Xie 2026 — 0.833, same three polygonal test sets. Already assessed.
- *That is the entire category.*

**Lebesgue planar, upper bound**
- Baez–Bagdasaryan–Gibbs 2015; Gibbs 2014, 2018 (+ ResearchGate version); Azimuth blog.

**Different problem — Moser's worm / unit curves / closed arcs**
- Khandhawit–Pagonakis–Sriswasdi 2013; Khandhawit–Sriswasdi 2007;
  Grechuk–Som-Am 2020 (×2, ≥0.0975 and ≥0.1); Wichiramala 2018;
  Som-Am 2010 and 2020 theses.

**Different problem — families of triangles**
- Park–Cheong 2021; Cheong–Devillers–Glisse–Park 2023; Krajči 2023;
  Balitskiy–Mitrofanov–Polyanskii 2026.

**Different setting — higher dimension or non-Euclidean**
- Y. Chen 2026 (arXiv:2607.27227): **R^3**, volume bounds (0.545193, 0.655984)
  via dodecahedra and centrally symmetric polytopes. No constant-width test
  sets; does not reference 0.832 or 0.833.
- Arman–Bondarenko–Prymak et al. 2025: asymptotic, E^n.
- Martini–Spirova 2013: normed planes.

**Algorithmic (convex hull / overlap under translation)**
- Ahn–Brass–Shin 2008; Fukuda–Uno 2006, 2007; **Jung–Kang–Ahn 2025**.
  The last is notable: minimising the convex hull of *two* convex polytopes under
  translation is exactly Brass–Sharifi's stated open problem — but only for k=2,
  which Rote had already observed is the convex (easy) case. Nothing for k ≥ 3.

**Surveys, compendia, unrelated**
- Martini–Montejano–Oliveros 2019 (book, *Bodies of Constant Width*);
  Finch 2020 (*Mathematical Constants* errata); Brass–Moser–Pach 2005;
  Beebe bibliography; Korean survey on Kakeya-type problems 2026;
  Kurz–Mishkin 2012; Chen–Dumitrescu 2015; Aichholzer et al. 2014;
  Gibbs 2016 (Bellman's lost-in-a-forest).

### Two items not fully closed
1. **Martini, Montejano & Oliveros (2019), _Bodies of Constant Width_** — a
   monograph on constant width that cites Brass–Sharifi. The most plausible place
   for the observation to appear as a remark. Full text not consulted. (Caution:
   web results conflate this with the **Blaschke–Lebesgue theorem** — minimum-area
   constant-width body = Reuleaux triangle — a different problem sharing a name.)
2. The 2026 Korean-language survey on Kakeya-type covering problems.

### The decisive indirect argument
Even if the remark exists somewhere in a survey, **nobody executed it**: the
published lower bound stood at 0.832 from 2005 to 2026, and the one improvement
in that span (Xie) stayed inside the polygonal framework and cites no such idea.
Had anyone combined the observation with a computation, the record would have
moved.

---

## 7. Assessment

### What is not novel
**The idea is Gibbs's (2014).** Replacing regular polygons by Reuleaux polygons
as lower-bound test sets is stated explicitly in arXiv:1401.8217, together with
the lower-bound lemma, the n-3 manifold dimension, and a computed five-body value
of 0.83699098 obtained by simulated annealing. The contribution of the present
work is therefore not the idea, and does not claim to be.

### What remains genuinely novel
1. **Rigour.** Gibbs's number is simulated annealing -- "an upper bound on such a
   lower bound", his words. If the anneal overshoots the true placement minimum,
   the "bound" is invalid, and nothing bounds that error. This work delivers an
   *exhaustive* branch-and-bound proof over the entire 5-D placement space, with
   a replayable certificate (486,799,600 nodes / 245,496,952 leaves) checkable by
   a standalone verifier. **First rigorous lower bound in the constant-width
   program.**
2. **It beats the best PROVED bound**: 0.8344 vs 0.832 (Brass-Sharifi 2005,
   refereed) and 0.833 (Xie 2026, preprint). +0.0024, closing 19.85% of the open
   interval to Gibbs's 0.8440936 upper bound.
3. **Framework-exhaustion measurement.** LB3 = 0.833597388099 is the *exact global
   optimum* of the disk + regular-triangle + regular-pentagon configuration --
   the ceiling of the framework used by Pal (1920), Brass-Sharifi (2005) and Xie
   (2026). Nobody had computed it. It proves that framework can never reach
   0.8344, so the improvement is structural rather than a better search.
4. **The 3-body constant-width ceiling**, LB_B = 0.834780952792, never reported.
5. **Certified-cost scaling** for this class of problem (5-D ~ margin^-2.66;
   8-D ~ margin^-5.50, ~10^11 boxes, out of reach), which is what shows why
   Gibbs's larger families cannot currently be certified.

### The honest one-sentence claim
*We convert a decade-old heuristic observation of Gibbs into the first rigorous
theorem, and in doing so raise the best proved lower bound for Lebesgue's problem
from 0.832 to 0.8344.*

### What must NOT be claimed
- Not "a new idea" -- it is Gibbs's.
- Not "the best known lower bound" without the qualifier **proved**: Gibbs's
  heuristic 0.83699098 is numerically higher.
- Not that the problem is closed: [0.8344, 0.8440936] is still open, width
  0.0097, and Gibbs's heuristics suggest the truth is near 0.84408.

---

## 8. Scope of this search

English-language sources; no MathSciNet or Zentralblatt. The Brass and Sharifi
citation graph was enumerated in full (section 6f): among the citing works there
is no competing lower bound and no use of constant-width test sets. Two items
were not read in full, neither of which states a lower bound: the
*Bodies of Constant Width* monograph (Martini, Montejano, Oliveros, 2019) and a
2026 Korean-language survey on Kakeya-type covering problems.

## 9. Sources

- Brass & Sharifi (2005), IJCGA 15(5):537–544, doi:10.1142/S0218195905001828
- Xie (2026), arXiv:2606.04458 — <https://arxiv.org/abs/2606.04458>
- Baez, Bagdasaryan & Gibbs (2015), J. Comput. Geom. 6:288–299, arXiv:1502.01251
- Gibbs (2018), arXiv:1810.10089
- Vrećica (1981), Publ. Inst. Math. 29:289–291
- Hansen (1992), Geom. Dedicata 42(2):205–213
- Pál (1920); Sprague (1936); Duff (1980)
- Khandhawit, Pagonakis & Sriswasdi (2011), arXiv:1101.5638
- Gibbs, GitHub — <https://github.com/PhilipGibbs/Lebesgue>
- n-Category Café (2015) — <https://golem.ph.utexas.edu/category/2015/02/computability_for_lebesgues_un.html>
- Wikipedia, "Lebesgue's universal covering problem"
