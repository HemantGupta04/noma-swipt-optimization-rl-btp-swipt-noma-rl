from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import QLearning as ql


OUTPUT_DIR = Path(__file__).resolve().parent
MODEL_PATH = OUTPUT_DIR / "output" / "q_learning_model.npz"
PLOT_PATH = OUTPUT_DIR / "q_learning_throughput_ee_vs_snr.png"
TABLE_PATH = OUTPUT_DIR / "q_learning_throughput_ee_vs_snr.csv"

SNR_GRID_DB = np.arange(0.0, 41.0, 2.0)
R1 = 0.5
R2 = 0.25
P_C = 0.2  # One representative circuit-power value from the 0.1 to 0.3 W range.


def load_saved_model(model_path=MODEL_PATH):
    model = np.load(model_path)
    return {
        "q_table": model["q_table"],
        "rho_actions": model["rho_actions"],
        "baseline_delta": model["baseline_delta"],
        "baseline_sigma_n2": float(model["baseline_sigma_n2"][0]),
        "baseline_eta": float(model["baseline_eta"][0]),
        "baseline_rho": ql.BASELINE_RHO,
    }


def throughput_from_pep(pep_pair):
    throughput_near = R1 * (1.0 - pep_pair[0])
    throughput_far = R2 * (1.0 - pep_pair[1])
    throughput_sum = throughput_near + throughput_far
    return throughput_near, throughput_far, throughput_sum


def evaluate_metrics_vs_snr(model_data):
    q_table = model_data["q_table"]
    rho_actions = model_data["rho_actions"]
    baseline_delta = model_data["baseline_delta"]
    sigma_n2 = model_data["baseline_sigma_n2"]
    eta = model_data["baseline_eta"]
    baseline_rho = model_data["baseline_rho"]

    rows = []
    for snr_db in SNR_GRID_DB:
        state = ql.State(
            snr_db=snr_db,
            delta=baseline_delta.copy(),
            sigma_n2=sigma_n2,
            eta=eta,
        )

        action_index = ql.greedy_action(q_table[ql.state_to_index(state)])
        learned_rho = float(rho_actions[action_index])

        learned_pep = ql.compute_pep_pair(state, learned_rho)
        baseline_pep = ql.compute_pep_pair(state, baseline_rho)

        learned_t_near, learned_t_far, learned_t_sum = throughput_from_pep(learned_pep)
        baseline_t_near, baseline_t_far, baseline_t_sum = throughput_from_pep(baseline_pep)

        p_b = (10.0 ** (snr_db / 10.0)) * sigma_n2
        learned_ee = learned_t_sum / (p_b + P_C)
        baseline_ee = baseline_t_sum / (p_b + P_C)

        rows.append(
            {
                "snr_db": snr_db,
                "learned_rho": learned_rho,
                "baseline_rho": baseline_rho,
                "learned_pep_near": learned_pep[0],
                "learned_pep_far": learned_pep[1],
                "baseline_pep_near": baseline_pep[0],
                "baseline_pep_far": baseline_pep[1],
                "learned_t_near": learned_t_near,
                "learned_t_far": learned_t_far,
                "learned_t_sum": learned_t_sum,
                "baseline_t_near": baseline_t_near,
                "baseline_t_far": baseline_t_far,
                "baseline_t_sum": baseline_t_sum,
                "p_b": p_b,
                "p_c": P_C,
                "learned_ee": learned_ee,
                "baseline_ee": baseline_ee,
            }
        )

    return rows


def save_table(rows, output_path=TABLE_PATH):
    header = [
        "SNR_dB",
        "learned_rho",
        "baseline_rho",
        "learned_pep_near",
        "learned_pep_far",
        "baseline_pep_near",
        "baseline_pep_far",
        "learned_T_near",
        "learned_T_far",
        "learned_T_sum",
        "baseline_T_near",
        "baseline_T_far",
        "baseline_T_sum",
        "P_B",
        "P_C",
        "learned_EE",
        "baseline_EE",
    ]
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(
                f"{row['snr_db']:.1f},{row['learned_rho']:.4f},{row['baseline_rho']:.4f},"
                f"{row['learned_pep_near']:.10f},{row['learned_pep_far']:.10f},"
                f"{row['baseline_pep_near']:.10f},{row['baseline_pep_far']:.10f},"
                f"{row['learned_t_near']:.10f},{row['learned_t_far']:.10f},{row['learned_t_sum']:.10f},"
                f"{row['baseline_t_near']:.10f},{row['baseline_t_far']:.10f},{row['baseline_t_sum']:.10f},"
                f"{row['p_b']:.10f},{row['p_c']:.10f},{row['learned_ee']:.10f},{row['baseline_ee']:.10f}\n"
            )


def plot_metrics(rows, output_path=PLOT_PATH):
    snr_values = np.array([row["snr_db"] for row in rows], dtype=float)

    learned_t_near = np.array([row["learned_t_near"] for row in rows], dtype=float)
    learned_t_far = np.array([row["learned_t_far"] for row in rows], dtype=float)
    learned_t_sum = np.array([row["learned_t_sum"] for row in rows], dtype=float)
    baseline_t_near = np.array([row["baseline_t_near"] for row in rows], dtype=float)
    baseline_t_far = np.array([row["baseline_t_far"] for row in rows], dtype=float)
    baseline_t_sum = np.array([row["baseline_t_sum"] for row in rows], dtype=float)
    learned_ee = np.array([row["learned_ee"] for row in rows], dtype=float)
    baseline_ee = np.array([row["baseline_ee"] for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))

    axes[0].plot(snr_values, learned_t_near, "-o", linewidth=2, color="darkorange", label="Near throughput, learned")
    axes[0].plot(snr_values, learned_t_far, "-s", linewidth=2, color="forestgreen", label="Far throughput, learned")
    axes[0].plot(snr_values, learned_t_sum, "-^", linewidth=2, color="midnightblue", label="Sum throughput, learned")
    axes[0].plot(snr_values, baseline_t_near, "--o", linewidth=1.5, color="peru", label="Near throughput, baseline")
    axes[0].plot(snr_values, baseline_t_far, "--s", linewidth=1.5, color="limegreen", label="Far throughput, baseline")
    axes[0].plot(snr_values, baseline_t_sum, "--^", linewidth=1.5, color="slateblue", label="Sum throughput, baseline")
    axes[0].set_title("Approximate Throughput vs SNR")
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("Throughput (bits/s/Hz)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(snr_values, learned_ee, "-o", linewidth=2, color="firebrick", label="Energy efficiency, learned")
    axes[1].plot(snr_values, baseline_ee, "--s", linewidth=1.8, color="teal", label="Energy efficiency, baseline")
    axes[1].set_title("Approximate Energy Efficiency vs SNR")
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("EE = T_sum / (P_B + P_C)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(
        "Q-learning Throughput and EE using PEP as an outage proxy\n"
        f"R1 = {R1:.2f}, R2 = {R2:.2f}, P_C = {P_C:.2f} W, "
        f"Delta = [{ql.BASELINE_DELTA[0]:.2f}, {ql.BASELINE_DELTA[1]:.2f}], "
        f"sigma_n2 = {ql.BASELINE_SIGMA_N2:.1f}, eta = {ql.BASELINE_ETA:.1f}",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    model_data = load_saved_model()
    rows = evaluate_metrics_vs_snr(model_data)
    save_table(rows)
    plot_metrics(rows)
    print(f"Loaded Q-learning model from: {MODEL_PATH}")
    print(f"Saved throughput/EE plot to: {PLOT_PATH}")
    print(f"Saved throughput/EE table to: {TABLE_PATH}")


if __name__ == "__main__":
    main()
