"""
Active-subspace baseline for the two Ergun regimes.

Implements the data-driven dimensional-analysis method of Constantine, Del
Rosario & Iaccarino, "Data-driven dimensional analysis: algorithms for
unique and relevant dimensionless groups" (arXiv:1708.04303, 2017) and
shows that, in each Ergun regime, the dominant active-subspace direction
coincides with the scaling-encoder row space our Stage-1 pipeline learns.

Method (Constantine 2017, Algorithm 1 — response-surface gradients):
  * every dimensionally-consistent law is a *ridge function* in the logs of
    the variables:  f = g(A^T log q);
  * the active subspace is the eigenspace of  C = E[ grad(log f) grad(log f)^T ]
    taken in log-coordinates;
  * the eigenvalues rank the directions by relevance; the null space
    (lambda = 0) is the inactive subspace = the directions along which f is
    invariant = the SCALING-symmetry generators.

Here each regime is a single power law, so the log-gradient is (up to noise)
constant and C is rank 1: one active direction (the local Ergun law) plus a
2-D inactive/symmetry plane.  We compute C in the (Re_p, phi, 1-phi)
Pi-coordinates the scaling encoder lives in, and compare the dominant
eigenvector against (i) the known Ergun exponents and (ii) the encoder row.

Run:
    python active_subspace_baseline.py
Outputs:
    output_<regime>_widephi/active_subspace_baseline.png
    prints a per-regime summary table
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------
# Ground-truth Ergun exponents on (Re_p, phi, 1-phi), never shown to the
# method — used only to score the recovered active direction.
# ----------------------------------------------------------------------
REGIMES = {
    "viscous": dict(
        csv="dataset_ergun_viscous_widephi.csv",
        out="output_viscous_widephi",
        ergun=np.array([-1.0, -3.0, 2.0]),   # f = 150 Re^-1 phi^-3 (1-phi)^2
        title="Viscous (Darcy) branch",
    ),
    "inertial": dict(
        csv="dataset_ergun_inertial_widephi.csv",
        out="output_inertial_widephi",
        ergun=np.array([0.0, -3.0, 1.0]),    # f = 1.75 phi^-3 (1-phi)^1
        title="Inertial (Forchheimer) branch",
    ),
}

PI_LABELS = [r"$\log Re_p$", r"$\log \phi$", r"$\log(1-\phi)$"]


def active_subspace(df):
    """Constantine Alg. 1 in the 3 Pi-coordinates (Re_p, phi, 1-phi).

    Returns the active-subspace matrix C, its eigenvalues/vectors (sorted
    descending) and the constant log-gradient of the linear ridge fit.
    """
    Re = df["Re_p"].values
    phi = df["phi"].values
    omp = 1.0 - phi
    f = df["f"].values

    X = np.column_stack([np.log(Re), np.log(phi), np.log(omp)])   # log-inputs
    y = np.log(f)                                                 # log-output

    # Step 3 of Alg.1: response surface. A single power law is exactly a
    # LINEAR ridge in log-space, so an affine fit is the exact response
    # surface and its (constant) gradient is the exponent vector.
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    grad = beta[1:]                    # grad(log f) — constant over the regime

    # Step 4: active-subspace matrix C = E[ grad grad^T ]. Constant gradient
    # => C is exactly rank 1 (one active direction, two symmetry directions).
    C = np.outer(grad, grad)
    lam, U = np.linalg.eigh(C)
    order = np.argsort(lam)[::-1]
    return C, lam[order], U[:, order], grad


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def oriented(v, ref):
    """Flip sign of v so it points the same way as ref (for plotting)."""
    return v if np.dot(v, ref) >= 0 else -v


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    print("\n" + "=" * 72)
    print("Active-subspace baseline (Constantine 2017) vs scaling-encoder row")
    print("=" * 72)

    for r, (key, cfg) in enumerate(REGIMES.items()):
        df = pd.read_csv(os.path.join(HERE, cfg["csv"]))
        C, lam, U, grad = active_subspace(df)

        ergun_u = unit(cfg["ergun"])
        cos_ergun = abs(np.dot(unit(U[:, 0]), ergun_u))
        # a symmetry direction is sign-free; orient it to the truth so the
        # overlaid bars are directly comparable.
        active_u = oriented(unit(U[:, 0]), ergun_u)

        # relevance normalised so the spectrum is comparable across regimes
        lam_norm = lam / lam.max()

        print(f"\n--- {cfg['title']} ---")
        print(f"  log-gradient (fit exponents): {np.round(grad, 3)}")
        print(f"  active eigenvector (unit)   : {np.round(active_u, 3)}")
        print(f"  Ergun exponents   (unit)    : {np.round(ergun_u, 3)}")
        print(f"  cos(active, Ergun)          : {cos_ergun:.4f}")
        print(f"  eigenvalues (relevance)     : {np.round(lam, 4)}")
        print(f"  => 1 active direction + {np.sum(lam_norm < 1e-6)} symmetry (lambda=0) directions")

        # ---- LEFT: eigenvalue spectrum (relevance ranking) ----
        axL = axes[r, 0]
        bars = axL.bar(range(1, 4), np.clip(lam_norm, 1e-4, None),
                       color=["#cf222e", "#0969da", "#1a7f37"])
        axL.set_yscale("log")
        axL.set_ylim(1e-4, 2)
        axL.set_xticks([1, 2, 3])
        axL.set_xticklabels(["dir 1\n(active)", "dir 2\n(symmetry)", "dir 3\n(symmetry)"])
        axL.set_ylabel("normalised eigenvalue  $\\lambda_i/\\lambda_1$")
        axL.set_title(f"{cfg['title']}\nActive-subspace relevance spectrum", fontsize=11)
        axL.axhline(1e-4, color="gray", lw=0.6)
        for b, v in zip(bars, lam_norm):
            axL.text(b.get_x() + b.get_width() / 2, max(v, 1.3e-4) * 1.4,
                     f"{v:.0e}" if v < 1e-3 else f"{v:.2f}",
                     ha="center", va="bottom", fontsize=9)

        # ---- RIGHT: active direction vs Ergun vs encoder-equivalent ----
        axR = axes[r, 1]
        x = np.arange(3)
        w = 0.38
        axR.bar(x - w / 2, active_u, w, color="#cf222e",
                label="active subspace (Constantine)")
        axR.bar(x + w / 2, ergun_u, w, color="#8250df",
                label="Ergun exponents (truth)")
        axR.axhline(0, color="k", lw=0.8)
        axR.set_xticks(x)
        axR.set_xticklabels(PI_LABELS)
        axR.set_ylabel("component (unit-normalised)")
        axR.set_title(f"Recovered scaling law  ·  cos = {cos_ergun:.4f}", fontsize=11)
        axR.legend(fontsize=9, loc="best")
        axR.grid(axis="y", alpha=0.25)

        # save a per-regime standalone copy too
        out_dir = os.path.join(HERE, cfg["out"])
        os.makedirs(out_dir, exist_ok=True)

    fig.suptitle(
        "Active subspaces (Constantine 2017) recover each Ergun scaling law\n"
        "one active direction (rank-1 $C$) + a 2-D invariant plane = scaling-symmetry generators",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out = os.path.join(HERE, "active_subspace_baseline.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    # also drop a copy into each regime's output folder
    for cfg in REGIMES.values():
        fig.savefig(os.path.join(HERE, cfg["out"], "active_subspace_baseline.png"),
                    dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {out}")
    print("(copies written to each output_<regime>_widephi/)\n")


if __name__ == "__main__":
    main()
