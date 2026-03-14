from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import QLearning as ql


OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = OUTPUT_DIR / "pep_vs_rho_fixed_state.png"

SNR_VALUES_DB = [5.0, 10.0, 20.0]
# SNR_VALUES_DB = [0, 3.0103, 6.0206]
# SNR_VALUES_DB = [0, 9.0309, 40]

RHO_GRID = np.linspace(0.05, 0.95, 200)


def evaluate_pep_curves(snr_db, delta, sigma_n2, eta):
    state = ql.State(
        snr_db=snr_db,
        delta=np.array(delta, dtype=float),
        sigma_n2=sigma_n2,
        eta=eta,
    )
    pep_curves = np.vstack([ql.compute_pep_pair(state, rho) for rho in RHO_GRID])
    return pep_curves


def plot_pep_vs_rho():
    delta = ql.BASELINE_DELTA.copy()
    sigma_n2 = ql.BASELINE_SIGMA_N2
    eta = ql.BASELINE_ETA

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), sharey=True)

    for axis, snr_db in zip(axes, SNR_VALUES_DB):
        pep_curves = evaluate_pep_curves(snr_db, delta, sigma_n2, eta)

        axis.semilogy(
            RHO_GRID,
            pep_curves[:, 0],
            color="darkorange",
            linewidth=2,
            label="Near user PEP",
        )
        axis.semilogy(
            RHO_GRID,
            pep_curves[:, 1],
            color="forestgreen",
            linewidth=2,
            label="Far user PEP",
        )
        axis.set_title(f"Average PEP vs rho at SNR = {snr_db:.0f} dB")
        axis.set_xlabel("rho (Power Splitting Factor)")
        axis.grid(True, which="both", alpha=0.3)

    axes[0].set_ylabel("Average PEP")
    axes[1].legend(loc="best")

    fig.suptitle(
        "PEP vs rho for Fixed State Variables\n"
        f"Delta = [{delta[0]:.2f}, {delta[1]:.2f}], "
        f"sigma_n2 = {sigma_n2:.1f}, eta = {eta:.1f}",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    plot_pep_vs_rho()
    print(f"Saved PEP vs rho plot to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
