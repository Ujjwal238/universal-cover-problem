"""Negative and positive controls for the certificate verifiers.

A verifier that accepts everything proves nothing; a verifier that refuses
everything also proves nothing.  Both directions are tested here.

Controls are regenerated deterministically from the small valid test
certificates, so the corrupted files in the repository can be rebuilt and are not
opaque blobs.

Header layout (both families), byte offsets from 0:
    0   8 bytes   magic, LEBCERT2 (family A) or LEBCERTB (family B)
    8   8 bytes   double  target
   16   8 bytes   double  threshold
   24   8 bytes   double  tmax
   32   4 bytes   int     m           (family A: then depth, nseed)
   36   4 bytes   int     K           (family B only)
   ...
"""

import os
import struct
import subprocess
import sys

MAGIC_A = b"LEBCERT2"
MAGIC_B = b"LEBCERTB"


def inflate_offset(raw):
    if raw[:8] == MAGIC_B:
        return 8 + struct.calcsize("<dddiiii")
    return 8 + struct.calcsize("<dddiii")


def payload_offset(raw):
    """First byte of the concatenated bit blocks."""
    if raw[:8] == MAGIC_B:
        off = 8 + struct.calcsize("<dddiiii")
    else:
        off = 8 + struct.calcsize("<dddiii")
    off += 8                                    # inflate
    (nblk,) = struct.unpack_from("<Q", raw, off)
    off += 8
    return off + 4 * nblk


def flip_bit(raw, byte_from_end, bit):
    off = len(raw) - byte_from_end
    b = bytearray(raw)
    b[off] ^= (1 << bit)
    return bytes(b)


def set_double(raw, off, val):
    b = bytearray(raw)
    struct.pack_into("<d", b, off, val)
    return bytes(b)


def set_int(raw, off, val):
    b = bytearray(raw)
    struct.pack_into("<i", b, off, val)
    return bytes(b)


def build(src, outdir):
    """Returns (path, why, expect) with expect in {"refuse", "accept"}.

    Not every header edit is an attack.  m and K set only how finely the witness
    sets approximate the bodies; witness points lie in the bodies by construction
    and are membership-tested, so any value gives a valid bound and a finer grid
    gives a tighter one.  Inflating m must therefore still be ACCEPTED, and a
    verifier that refused it would be brittle rather than sound.  tmax and
    inflate are different: they decide what the tree actually covers.
    """
    raw = open(src, "rb").read()
    fam = "B" if raw[:8] == MAGIC_B else "A"
    po = payload_offset(raw)
    io_ = inflate_offset(raw)
    made = []

    def emit(name, data, why, expect):
        p = os.path.join(outdir, name)
        with open(p, "wb") as fh:
            fh.write(data)
        made.append((p, why, expect))

    span = len(raw) - po
    for i in range(1, 6):
        off_from_end = max(1, span * i // 6)
        emit(f"ctl_{fam}_bitflip{i}.bin", flip_bit(raw, off_from_end, i % 8),
             f"one tree bit flipped, {off_from_end} bytes from the end", "refuse")
    emit(f"ctl_{fam}_trunc.bin", raw[:-max(1, span // 8)],
         "bit stream truncated by an eighth", "refuse")

    tgt, = struct.unpack_from("<d", raw, 8)
    emit(f"ctl_{fam}_target.bin", set_double(raw, 8, tgt + 0.01),
         f"header target raised {tgt} -> {round(tgt + 0.01, 4)}", "refuse")

    # the tree tiles |t| <= tmax only, so shrinking tmax makes it cover less of
    # the domain than the theorem requires
    tm, = struct.unpack_from("<d", raw, 24)
    emit(f"ctl_{fam}_tmax.bin", set_double(raw, 24, 0.55),
         f"claimed domain shrunk {tm} -> 0.55, no longer excluding outside "
         f"placements", "refuse")

    # negative inflation opens gaps between sibling boxes
    infl, = struct.unpack_from("<d", raw, io_)
    emit(f"ctl_{fam}_inflate.bin", set_double(raw, io_, -1e-9),
         f"child inflation made negative {infl:g} -> -1e-09, opening gaps between "
         f"siblings", "refuse")

    # NOT an attack: a finer witness grid is a tighter valid bound
    m, = struct.unpack_from("<i", raw, 32)
    emit(f"ctl_{fam}_m_finer.bin", set_int(raw, 32, m * 4),
         f"witness resolution refined {m} -> {m * 4} (a tightening, not a "
         f"corruption)", "accept")
    return made


def run(verifier, path):
    r = subprocess.run([sys.executable, verifier, path],
                       capture_output=True, text=True, timeout=3600)
    return r.returncode, r.stdout


if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else "controls_gen"
    os.makedirs(outdir, exist_ok=True)

    jobs = []
    if os.path.exists("certB_test.bin"):
        jobs.append(("certB_test.bin", "verifyB.py"))
    if os.path.exists("cert_test.bin"):
        jobs.append(("cert_test.bin", "verify.py"))

    print("=" * 100)
    print("CERTIFICATE CONTROLS")
    print("=" * 100)

    npass = nfail = 0

    for src, verifier in jobs:
        print(f"\n-- POSITIVE control: {src} must be ACCEPTED by {verifier} --")
        rc, out = run(verifier, src)
        ok = (rc == 0) and ("VERIFIED" in out)
        print(f"   [{'PASS' if ok else 'FAIL'}] exit {rc}   "
              f"{[l for l in out.splitlines() if 'VERIFIED' in l or 'FAILED' in l][:1]}")
        npass += ok
        nfail += (not ok)

        print(f"\n-- controls built from {src} --")
        for path, why, expect in build(src, outdir):
            rc, out = run(verifier, path)
            accepted = (rc == 0) and ("VERIFIED" in out)
            ok = accepted if expect == "accept" else not accepted
            if expect == "accept":
                got = "accepted" if accepted else "*** REFUSED A VALID TIGHTENING"
            else:
                got = ("*** ACCEPTED A CORRUPT CERTIFICATE" if accepted
                       else "refused")
            print(f"   [{'PASS' if ok else 'FAIL'}] {os.path.basename(path):<26} "
                  f"must {expect:<7} {why}\n          -> {got}")
            npass += ok
            nfail += (not ok)

    print("\n" + "=" * 100)
    print(f"  {npass} passed, {nfail} failed")
    print("  " + ("ALL CONTROLS BEHAVED CORRECTLY" if nfail == 0
                  else "*** CONTROL FAILURE -- the verifier is not sound"))
    print("=" * 100)
    sys.exit(1 if nfail else 0)
