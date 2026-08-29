"""Figures for the paper.  Deterministic, vector output, no hand-entered data.

Run from this directory:  python make_figures.py
Every quantity is either derived from the geometry kernel or read from a log.
Line styles carry the distinctions, so the figures survive grayscale printing.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import os
import sys

# Resolve the geometry kernel and the logs without assuming a working directory,
# so the scripts run from either the repository or a working checkout.
_HERE = os.path.dirname(os.path.abspath(__file__))


def _find(*relatives):
    """First existing candidate, searching upward from this file."""
    base = _HERE
    for _ in range(4):
        for r in relatives:
            p = os.path.join(base, r)
            if os.path.exists(p):
                return os.path.abspath(p)
        base = os.path.dirname(base)
    raise FileNotFoundError(relatives[0])


KERNEL = os.path.dirname(_find("src/geom.py", "geom.py"))
SCALING_LOG = _find("logs/scaling.log", "scaling.log")
sys.path.insert(0, KERNEL)

import geom
import verifyB as V

plt.rcParams.update({
    "font.size": 8, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "pdf.fonttype": 42, "ps.fonttype": 42, "text.usetex": False,
})
OUT = dict(format="pdf", bbox_inches="tight", pad_inches=0.02)


def disc_intersection_boundary(V_, rho, K=2000):
    """Boundary of the intersection of discs of radius rho about the rows of V_."""
    W = V.core_points(np.asarray(V_, dtype=float), float(rho), K)
    if W is None:
        return None
    c = W.mean(axis=0)
    order = np.argsort(np.arctan2(W[:, 1] - c[1], W[:, 0] - c[0]))
    W = W[order]
    return np.vstack([W, W[:1]])


def corners(n):
    Vv = geom.reuleaux_corners(n, 1.0)
    return Vv - Vv.mean(axis=0)


# --------------------------------------------------------------- figure 1
def fig_testsets(path):
    fig, axes = plt.subplots(1, 3, figsize=(5.4, 1.95))
    disc = np.array([[0.5 * np.cos(t), 0.5 * np.sin(t)]
                     for t in np.linspace(0, 2 * np.pi, 400)])
    axes[0].plot(disc[:, 0], disc[:, 1], "k-")
    axes[0].set_title(r"$D$", pad=3)
    for ax, n in zip(axes[1:], (3, 5)):
        Vv = corners(n)
        B = disc_intersection_boundary(Vv, 1.0)
        ax.plot(B[:, 0], B[:, 1], "k-", label=r"$B_%d$" % n)
        P = np.vstack([Vv, Vv[:1]])
        ax.plot(P[:, 0], P[:, 1], "k--", lw=0.9, label=r"$P_%d$" % n)
        ax.plot(Vv[:, 0], Vv[:, 1], "ko", ms=2.4)
        ax.set_title(r"$B_%d \supseteq P_%d$" % (n, n), pad=3)
    for ax in axes:
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_xlim(-0.72, 0.72); ax.set_ylim(-0.72, 0.72)
    fig.savefig(path, **OUT); plt.close(fig)


# --------------------------------------------------------------- figure 2
def fig_erosion(path, delta=0.14):
    """The erosion estimate: one set inside every placement the box allows."""
    fig, ax = plt.subplots(figsize=(3.1, 3.1))
    Vv = corners(3)

    # three displaced placements allowed by a box with corner displacement <= delta
    rng = np.random.default_rng(4)
    for k in range(3):
        ang = 2 * np.pi * k / 3 + 0.5
        shift = delta * np.array([np.cos(ang), np.sin(ang)])
        W = Vv + shift
        Bd = disc_intersection_boundary(W, 1.0)
        ax.plot(Bd[:, 0], Bd[:, 1], color="0.55", ls=":", lw=0.9)

    B = disc_intersection_boundary(Vv, 1.0)
    ax.plot(B[:, 0], B[:, 1], "k-", lw=1.2)

    C = disc_intersection_boundary(Vv, 1.0 - delta)
    ax.fill(C[:, 0], C[:, 1], color="0.82", zorder=0)
    ax.plot(C[:, 0], C[:, 1], "k--", lw=1.0)

    ax.plot(Vv[:, 0], Vv[:, 1], "ko", ms=3.2)
    for j, (x, y) in enumerate(Vv):
        ax.annotate(r"$V_%d$" % (j + 1), (x, y), textcoords="offset points",
                    xytext=(5, 4), fontsize=8)
    # the displacement budget around one corner
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(Vv[0, 0] + delta * np.cos(th), Vv[0, 1] + delta * np.sin(th),
            color="0.35", ls="-.", lw=0.8)
    ax.annotate(r"$\delta$", (Vv[0, 0] + delta * 0.75, Vv[0, 1] + delta * 0.75),
                textcoords="offset points", xytext=(3, 1), fontsize=8)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-1.02, 1.12); ax.set_ylim(-1.05, 1.09)
    fig.savefig(path, **OUT); plt.close(fig)


# --------------------------------------------------------------- figure 3
def fig_scaling(path, log=None):
    log = log or SCALING_LOG
    import re
    txt = open(log).read()
    blocks = re.findall(r">>> family \(([\d, ]+)\), dimension (\d+).*?\n(.*?)(?=\n\n|\n>>>|\Z)",
                        txt, re.S)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    styles = [("ko-", "$D,B_3,B_5$   ($d=5$)"), ("ks--", "$D,B_3,B_5,B_7$   ($d=8$)")]
    for (fam, dim, body), (st, lab) in zip(blocks, styles):
        rows = re.findall(r"^\s*([\d.]+)\s+[\d.]+\s+([\d,]+)\s", body, re.M)
        mu = np.array([float(a) for a, _ in rows])
        N = np.array([float(b.replace(",", "")) for _, b in rows])
        o = np.argsort(mu)
        ax.loglog(mu[o], N[o], st, ms=3.4, lw=1.0, label=lab, mfc="w")
    ax.set_xlabel(r"margin $\mu$ to the family ceiling")
    ax.set_ylabel("boxes examined")
    ax.grid(True, which="both", lw=0.3, color="0.85")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.invert_xaxis()
    fig.savefig(path, **OUT); plt.close(fig)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    fig_testsets(os.path.join(here, "testsets.pdf"))
    print("  testsets.pdf")
    fig_erosion(os.path.join(here, "erosion.pdf"))
    print("  erosion.pdf")
    fig_scaling(os.path.join(here, "scaling.pdf"))
    print("  scaling.pdf")
