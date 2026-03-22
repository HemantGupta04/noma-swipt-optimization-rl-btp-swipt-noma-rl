from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import QLearning as ql


OUTPUT_DIR = Path(__file__).resolve().parent
PLOT_PATH = OUTPUT_DIR / "throughput_ee_vs_rho_fixed_state.png"
TABLE_PATH = OUTPUT_DIR / "throughput_ee_vs_rho_fixed_state.csv"

SNR_VALUES_DB = [5.0, 10.0, 20.0]
RHO_GRID = np.linspace(0.05, 0.95, 200)

R1 = 0.5
R2 = 0.25
P_C = 0.2  # Representative circuit power in watts.


def throughput_from_pep(pep_pair):
    throughput_near = R1 * (1.0 - pep_pair[0])
    throughput_far = R2 * (1.0 - pep_pair[1])
    throughput_sum = throughput_near + throughput_far
    return throughput_near, throughput_far, throughput_sum


def evaluate_metrics_for_snr(snr_db, delta, sigma_n2, eta):
    state = ql.State(
        snr_db=snr_db,
        delta=np.array(delta, dtype=float),
        sigma_n2=sigma_n2,
        eta=eta,
    )

    p_b = (10.0 ** (snr_db / 10.0)) * sigma_n2
    rows = []
    for rho in RHO_GRID:
        pep_pair = ql.compute_pep_pair(state, rho)
        throughput_near, throughput_far, throughput_sum = throughput_from_pep(pep_pair)
        ee_value = throughput_sum / (p_b + P_C)
        rows.append(
            {
                "snr_db": snr_db,
                "rho": rho,
                "pep_near": pep_pair[0],
                "pep_far": pep_pair[1],
                "throughput_near": throughput_near,
                "throughput_far": throughput_far,
                "throughput_sum": throughput_sum,
                "p_b": p_b,
                "p_c": P_C,
                "ee": ee_value,
            }
        )
    return rows


def save_table(all_rows, output_path=TABLE_PATH):
    header = [
        "SNR_dB",
        "rho",
        "PEP_near",
        "PEP_far",
        "T_near",
        "T_far",
        "T_sum",
        "P_B",
        "P_C",
        "EE",
    ]
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(header) + "\n")
        for row in all_rows:
            handle.write(
                f"{row['snr_db']:.1f},{row['rho']:.10f},{row['pep_near']:.10f},{row['pep_far']:.10f},"
                f"{row['throughput_near']:.10f},{row['throughput_far']:.10f},{row['throughput_sum']:.10f},"
                f"{row['p_b']:.10f},{row['p_c']:.10f},{row['ee']:.10f}\n"
            )


def plot_metrics(curves_by_snr, output_path=PLOT_PATH):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)

    for column_index, snr_db in enumerate(SNR_VALUES_DB):
        rows = curves_by_snr[snr_db]
        rho_values = np.array([row["rho"] for row in rows], dtype=float)
        throughput_near = np.array([row["throughput_near"] for row in rows], dtype=float)
        throughput_far = np.array([row["throughput_far"] for row in rows], dtype=float)
        throughput_sum = np.array([row["throughput_sum"] for row in rows], dtype=float)
        ee_values = np.array([row["ee"] for row in rows], dtype=float)

        axes[0, column_index].plot(
            rho_values,
            throughput_near,
            color="darkorange",
            linewidth=2,
            label="Near throughput",
        )
        axes[0, column_index].plot(
            rho_values,
            throughput_far,
            color="forestgreen",
            linewidth=2,
            label="Far throughput",
        )
        axes[0, column_index].plot(
            rho_values,
            throughput_sum,
            color="midnightblue",
            linewidth=2,
            label="Sum throughput",
        )
        axes[0, column_index].set_title(f"Throughput vs rho at SNR = {snr_db:.0f} dB")
        axes[0, column_index].set_ylabel("Throughput (bits/s/Hz)")
        axes[0, column_index].grid(True, alpha=0.3)

        axes[1, column_index].plot(
            rho_values,
            ee_values,
            color="firebrick",
            linewidth=2,
            label="Energy efficiency",
        )
        axes[1, column_index].set_title(f"EE vs rho at SNR = {snr_db:.0f} dB")
        axes[1, column_index].set_xlabel("rho (Power Splitting Factor)")
        axes[1, column_index].set_ylabel("EE = T_sum / (P_B + P_C)")
        axes[1, column_index].grid(True, alpha=0.3)

    axes[0, 1].legend(loc="best")
    axes[1, 1].legend(loc="best")

    fig.suptitle(
        "Approximate Throughput and EE vs rho for Fixed State Variables\n"
        f"R1 = {R1:.2f}, R2 = {R2:.2f}, P_C = {P_C:.2f} W, "
        f"Delta = [{ql.BASELINE_DELTA[0]:.2f}, {ql.BASELINE_DELTA[1]:.2f}], "
        f"sigma_n2 = {ql.BASELINE_SIGMA_N2:.1f}, eta = {ql.BASELINE_ETA:.1f}",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    delta = ql.BASELINE_DELTA.copy()
    sigma_n2 = ql.BASELINE_SIGMA_N2
    eta = ql.BASELINE_ETA

    curves_by_snr = {}
    all_rows = []
    for snr_db in SNR_VALUES_DB:
        rows = evaluate_metrics_for_snr(snr_db, delta, sigma_n2, eta)
        curves_by_snr[snr_db] = rows
        all_rows.extend(rows)

    save_table(all_rows)
    plot_metrics(curves_by_snr)
    print(f"Saved throughput/EE vs rho plot to: {PLOT_PATH}")
    print(f"Saved throughput/EE vs rho table to: {TABLE_PATH}")


if __name__ == "__main__":
    main()
