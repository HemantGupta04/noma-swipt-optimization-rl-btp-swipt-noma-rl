from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import DDPG as ddpg
import QLearning as ql


OUTPUT_DIR = Path(__file__).resolve().parent

SNR_GRID = np.arange(0.0, 41.0, 2.0)
ETA_GRID = ql.ETA_EVAL_GRID.copy()
DELTA_GRID = ql.DELTA_EVAL_GRID.copy()


def style_map():
    return {
        "near_q": {"color": "darkorange", "linestyle": "-", "marker": "o", "linewidth": 2, "markersize": 5},
        "near_ddpg": {"color": "firebrick", "linestyle": "-.", "marker": "^", "linewidth": 2, "markersize": 5},
        "near_baseline": {"color": "peru", "linestyle": "--", "marker": "o", "linewidth": 1.5, "markersize": 4},
        "far_q": {"color": "forestgreen", "linestyle": "-", "marker": "s", "linewidth": 2, "markersize": 5},
        "far_ddpg": {"color": "teal", "linestyle": "-.", "marker": "^", "linewidth": 2, "markersize": 5},
        "far_baseline": {"color": "limegreen", "linestyle": "--", "marker": "s", "linewidth": 1.5, "markersize": 4},
        "rho_q": {"color": "midnightblue", "linestyle": "-", "marker": "o", "linewidth": 2, "markersize": 5},
        "rho_ddpg": {"color": "purple", "linestyle": "-.", "marker": "^", "linewidth": 2, "markersize": 5},
    }


def evaluate_q_learning(q_table):
    return {
        "snr": ql.evaluate_policy(q_table),
        "eta": ql.evaluate_eta_sensitivity(q_table),
        "delta": ql.evaluate_delta_sensitivity(q_table),
    }


def ddpg_action(agent, state):
    return agent.select_action(ddpg.state_to_vector(state), add_noise=False)


def evaluate_ddpg_snr(agent):
    learned_rho = []
    learned_pep = []
    baseline_pep = []

    for snr_db in SNR_GRID:
        state = ddpg.State(
            snr_db=snr_db,
            delta=ddpg.BASELINE_DELTA.copy(),
            sigma_n2=ddpg.BASELINE_SIGMA_N2,
            eta=ddpg.BASELINE_ETA,
        )
        rho_star = ddpg_action(agent, state)
        learned_rho.append(rho_star)
        learned_pep.append(ddpg.compute_pep_pair(state, rho_star))
        baseline_pep.append(ddpg.compute_pep_pair(state, ddpg.BASELINE_RHO))

    return {
        "snr_grid": SNR_GRID.copy(),
        "learned_rho": np.asarray(learned_rho, dtype=float),
        "learned_pep": np.vstack(learned_pep),
        "baseline_pep": np.vstack(baseline_pep),
    }


def evaluate_ddpg_eta(agent):
    learned_rho = []
    learned_pep = []
    baseline_pep = []

    for eta_value in ETA_GRID:
        state = ddpg.State(
            snr_db=ddpg.BASELINE_SNR_DB if hasattr(ddpg, "BASELINE_SNR_DB") else 20.0,
            delta=ddpg.BASELINE_DELTA.copy(),
            sigma_n2=ddpg.BASELINE_SIGMA_N2,
            eta=eta_value,
        )
        rho_star = ddpg_action(agent, state)
        learned_rho.append(rho_star)
        learned_pep.append(ddpg.compute_pep_pair(state, rho_star))
        baseline_pep.append(ddpg.compute_pep_pair(state, ddpg.BASELINE_RHO))

    return {
        "eta_grid": ETA_GRID.copy(),
        "learned_rho": np.asarray(learned_rho, dtype=float),
        "learned_pep": np.vstack(learned_pep),
        "baseline_pep": np.vstack(baseline_pep),
    }


def evaluate_ddpg_delta(agent):
    scenarios = {
        "delta1_only": {
            "title": "Average PEP vs Delta_1 (Delta_2 fixed at 0.10)",
            "x_label": "Delta_1",
            "states": [
                ddpg.State(
                    snr_db=ql.BASELINE_SNR_DB,
                    delta=np.array([delta_value, ql.BASELINE_DELTA[1]], dtype=float),
                    sigma_n2=ql.BASELINE_SIGMA_N2,
                    eta=ql.BASELINE_ETA,
                )
                for delta_value in DELTA_GRID
            ],
        },
        "delta2_only": {
            "title": "Average PEP vs Delta_2 (Delta_1 fixed at 0.05)",
            "x_label": "Delta_2",
            "states": [
                ddpg.State(
                    snr_db=ql.BASELINE_SNR_DB,
                    delta=np.array([ql.BASELINE_DELTA[0], delta_value], dtype=float),
                    sigma_n2=ql.BASELINE_SIGMA_N2,
                    eta=ql.BASELINE_ETA,
                )
                for delta_value in DELTA_GRID
            ],
        },
        "delta_both": {
            "title": "Average PEP vs Combined Delta Change (Delta_1 = Delta_2)",
            "x_label": "Common Delta Value",
            "states": [
                ddpg.State(
                    snr_db=ql.BASELINE_SNR_DB,
                    delta=np.array([delta_value, delta_value], dtype=float),
                    sigma_n2=ql.BASELINE_SIGMA_N2,
                    eta=ql.BASELINE_ETA,
                )
                for delta_value in DELTA_GRID
            ],
        },
    }

    results = {}
    for scenario_name, config in scenarios.items():
        learned_rho = []
        learned_pep = []
        baseline_pep = []

        for state in config["states"]:
            rho_star = ddpg_action(agent, state)
            learned_rho.append(rho_star)
            learned_pep.append(ddpg.compute_pep_pair(state, rho_star))
            baseline_pep.append(ddpg.compute_pep_pair(state, ddpg.BASELINE_RHO))

        results[scenario_name] = {
            "title": config["title"],
            "x_label": config["x_label"],
            "x_values": DELTA_GRID.copy(),
            "learned_rho": np.asarray(learned_rho, dtype=float),
            "learned_pep": np.vstack(learned_pep),
            "baseline_pep": np.vstack(baseline_pep),
        }

    return results


def evaluate_ddpg(agent):
    return {
        "snr": evaluate_ddpg_snr(agent),
        "eta": evaluate_ddpg_eta(agent),
        "delta": evaluate_ddpg_delta(agent),
    }


def plot_method_comparison(axis, x_values, q_pep, ddpg_pep, baseline_pep, x_label, title):
    styles = style_map()

    axis.semilogy(x_values, q_pep[:, 0], label="Near user, Q-learning", **styles["near_q"])
    axis.semilogy(x_values, ddpg_pep[:, 0], label="Near user, DDPG", **styles["near_ddpg"])
    axis.semilogy(x_values, baseline_pep[:, 0], label="Near user, fixed rho = 0.5", **styles["near_baseline"])
    axis.semilogy(x_values, q_pep[:, 1], label="Far user, Q-learning", **styles["far_q"])
    axis.semilogy(x_values, ddpg_pep[:, 1], label="Far user, DDPG", **styles["far_ddpg"])
    axis.semilogy(x_values, baseline_pep[:, 1], label="Far user, fixed rho = 0.5", **styles["far_baseline"])

    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Average PEP")
    axis.set_ylim(1e-5, 1.0)
    axis.grid(True, which="both", alpha=0.3)


def plot_snr_comparison(q_results, ddpg_results, output_path):
    styles = style_map()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))

    axes[0].plot(
        q_results["snr"]["snr_grid"],
        q_results["snr"]["learned_rho"],
        label="Q-learning rho",
        **styles["rho_q"],
    )
    axes[0].plot(
        ddpg_results["snr"]["snr_grid"],
        ddpg_results["snr"]["learned_rho"],
        label="DDPG rho",
        **styles["rho_ddpg"],
    )
    axes[0].set_title("Greedy rho vs SNR")
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("rho")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    plot_method_comparison(
        axes[1],
        q_results["snr"]["snr_grid"],
        q_results["snr"]["learned_pep"],
        ddpg_results["snr"]["learned_pep"],
        q_results["snr"]["baseline_pep"],
        "SNR (dB)",
        "PEP vs SNR: Q-learning vs DDPG vs Baseline",
    )
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_eta_comparison(q_results, ddpg_results, output_path):
    styles = style_map()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))

    axes[0].plot(
        q_results["eta"]["eta_grid"],
        q_results["eta"]["learned_rho"],
        label="Q-learning rho",
        **styles["rho_q"],
    )
    axes[0].plot(
        ddpg_results["eta"]["eta_grid"],
        ddpg_results["eta"]["learned_rho"],
        label="DDPG rho",
        **styles["rho_ddpg"],
    )
    axes[0].set_title("Greedy rho vs eta")
    axes[0].set_xlabel("eta (Energy Conversion Efficiency)")
    axes[0].set_ylabel("rho")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    plot_method_comparison(
        axes[1],
        q_results["eta"]["eta_grid"],
        q_results["eta"]["learned_pep"],
        ddpg_results["eta"]["learned_pep"],
        q_results["eta"]["baseline_pep"],
        "eta (Energy Conversion Efficiency)",
        "PEP vs eta: Q-learning vs DDPG vs Baseline",
    )
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_delta_comparison(q_results, ddpg_results, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), sharey=True)

    for axis, scenario_key in zip(axes, ("delta1_only", "delta2_only", "delta_both")):
        scenario_q = q_results["delta"][scenario_key]
        scenario_ddpg = ddpg_results["delta"][scenario_key]
        plot_method_comparison(
            axis,
            scenario_q["x_values"],
            scenario_q["learned_pep"],
            scenario_ddpg["learned_pep"],
            scenario_q["baseline_pep"],
            scenario_q["x_label"],
            scenario_q["title"],
        )
        if axis is not axes[0]:
            axis.set_ylabel("")

    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        f"PEP vs Delta: Q-learning vs DDPG vs Baseline at SNR = {ql.BASELINE_SNR_DB:.0f} dB, "
        f"sigma_n2 = {ql.BASELINE_SIGMA_N2:.1f}, eta = {ql.BASELINE_ETA:.1f}",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Training Q-learning model...")
    q_table, q_history = ql.train_agent(seed=7)

    print("Training DDPG model...")
    ddpg_agent, ddpg_history = ddpg.train_ddpg(seed=7)

    print("Evaluating both policies on shared grids...")
    q_results = evaluate_q_learning(q_table)
    ddpg_results = evaluate_ddpg(ddpg_agent)

    snr_plot_path = OUTPUT_DIR / "q_vs_ddpg_pep_vs_snr.png"
    eta_plot_path = OUTPUT_DIR / "q_vs_ddpg_pep_vs_eta.png"
    delta_plot_path = OUTPUT_DIR / "q_vs_ddpg_delta_sensitivity.png"

    plot_snr_comparison(q_results, ddpg_results, snr_plot_path)
    plot_eta_comparison(q_results, ddpg_results, eta_plot_path)
    plot_delta_comparison(q_results, ddpg_results, delta_plot_path)

    print("Comparison plots saved:")
    print(f"  {snr_plot_path}")
    print(f"  {eta_plot_path}")
    print(f"  {delta_plot_path}")
    print(
        "Reward mean over last 250 episodes: "
        f"Q-learning = {np.mean(q_history['reward'][-250:]):.4f}, "
        f"DDPG = {np.mean(ddpg_history['reward'][-250:]):.4f}"
    )


if __name__ == "__main__":
    main()
