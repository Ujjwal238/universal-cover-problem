# Certificates

Three certificates, gzipped. Each is a branch and bound tree stored as one bit
per node in DFS pre-order: `1` = split (two children follow), `0` = pruned leaf.
Boxes are never stored. The verifier regenerates every box from the root using
the deterministic split rule, so the tree costs one bit per node.

| file | target | family | nodes | leaves |
| --- | --- | --- | --- | --- |
| `cert_B_08344.bin.gz` | 0.8344 | disk + Reuleaux 3 + Reuleaux 5 | 486,799,600 | 245,496,952 |
| `cert_A_0833.bin.gz` | 0.833 | disk + regular 3-gon + regular 5-gon | 947,693,104 | 473,977,624 |
| `cert_A_0832.bin.gz` | 0.832 | disk + regular 3-gon + regular 5-gon | 145,137,288 | 72,576,836 |

`cert_B_08344` is the headline result. The two family A certificates reproduce the
Brass and Sharifi (2005) and Xie (2026) bounds on their own test sets, with
machinery those authors did not use; they exist to show the pipeline recovers
known results.

Integrity: `shasum -a 256 -c MANIFEST.sha256`. The manifest also lists the
hashes of the uncompressed files, which is what the verifier reads.

## Verify

```bash
gunzip -c cert_B_08344.bin.gz > /tmp/cert_B_08344.bin
cd ../src && python verifyB.py /tmp/cert_B_08344.bin
```

Expect `VERIFIED:  a >= 0.8344`, the forest identity `2L - nseed = nodes` holding
exactly, and a worst leaf slack of `1.240900e-05`. Wall clock is about 15 hours on
8 cores; the family A certificates take longer.

The verifier shares no bound code with the emitter. At each leaf it recomputes
the bound from scratch using inscribed polygons, the triangle inequality and the
shoelace formula, then checks it against the target read from the file header.

## Rebuild from scratch

```bash
cd ../src
python certemitB.py 0.8344 1024 1024 22 1e-5 cert_B_08344.bin   # family B
python certemit.py  0.833  1024 18 1e-5      cert_A_0833.bin    # family A
python certemit.py  0.832  1024 14 1e-5      cert_A_0832.bin    # family A
```

Emission is deterministic, so a rebuild reproduces the hashes above bit for bit.
Family B took 364.9 minutes on 8 cores for 450,922,384 boxes.

The emitter's search threshold is `target + deficit`, a heuristic for choosing
when to stop splitting. It is **not** what carries the proof. The proof is the
verifier's per-leaf check against `target` itself. On this certificate the
emitter's head-room was in fact smaller than the worst case witness deficit, and
the verifier logged `optimistic -- leaves decide` before passing all 245,496,952
leaves anyway. The heuristic was tested, not assumed.

## Controls

`controls/` holds the small valid certificates and deliberately altered copies of
them. `src/tamper.py` rebuilds the altered copies and runs both directions, since
a verifier that accepted everything and one that refused everything would each
pass a one-sided test. All 22 checks behave as required; see
`logs/controls_final.log`.

Must be **refused**:

| alteration | why it is unsound |
| --- | --- |
| a tree bit flipped | the replayed tree no longer matches the search |
| bit stream truncated | part of the tree is missing |
| header target raised | the tree does not support the larger value |
| declared domain shrunk | the tree covers less than the theorem requires |
| child inflation made negative | sibling boxes need not cover their parent |

Must be **accepted**:

| alteration | why it is still sound |
| --- | --- |
| the valid certificates themselves | nothing is wrong with them |
| witness resolution refined | a finer inscribed polygon is a tighter valid bound, not a corrupted one |

The second table is the half that keeps the first honest. Witness points lie in
the test sets by construction and are membership-tested, so any resolution gives a
valid bound; refusing a refinement would be brittleness rather than soundness.
