from pathlib import Path

import numpy as np

import DDPG as ddpg
import QLearning as ql
from compare_q_learning_vs_ddpg import evaluate_ddpg_snr


OUTPUT_PATH = Path(__file__).resolve().parent / "pep_vs_snr_table.csv"


def build_rows(q_results, ddpg_results):
    rows = []
    snr_grid = q_results["snr_grid"]
    baseline_rho = ql.BASELINE_RHO

    for index, snr_db in enumerate(snr_grid):
        rows.append(
            {
                "snr_db": float(snr_db),
                "q_rho": float(q_results["learned_rho"][index]),
                "q_pep_near": float(q_results["learned_pep"][index, 0]),
                "q_pep_far": float(q_results["learned_pep"][index, 1]),
                "ddpg_rho": float(ddpg_results["learned_rho"][index]),
                "ddpg_pep_near": float(ddpg_results["learned_pep"][index, 0]),
                "ddpg_pep_far": float(ddpg_results["learned_pep"][index, 1]),
                "baseline_rho": float(baseline_rho),
                "baseline_pep_near": float(q_results["baseline_pep"][index, 0]),
                "baseline_pep_far": float(q_results["baseline_pep"][index, 1]),
            }
        )

    return rows


def save_csv(rows, output_path=OUTPUT_PATH):
    header = [
        "SNR_dB",
        "QLearning_rho",
        "QLearning_near_user_PEP",
        "QLearning_far_user_PEP",
        "DDPG_rho",
        "DDPG_near_user_PEP",
        "DDPG_far_user_PEP",
        "Baseline_rho",
        "Baseline_near_user_PEP",
        "Baseline_far_user_PEP",
    ]

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(
                f"{row['snr_db']:.1f},"
                f"{row['q_rho']:.10f},{row['q_pep_near']:.10f},{row['q_pep_far']:.10f},"
                f"{row['ddpg_rho']:.10f},{row['ddpg_pep_near']:.10f},{row['ddpg_pep_far']:.10f},"
                f"{row['baseline_rho']:.10f},{row['baseline_pep_near']:.10f},{row['baseline_pep_far']:.10f}\n"
            )


def main():
    print("Training Q-learning model for PEP vs SNR table...")
    q_table, _ = ql.train_agent(seed=7)
    q_results = ql.evaluate_policy(q_table)

    print("Training DDPG model for PEP vs SNR table...")
    ddpg_agent, _ = ddpg.train_ddpg(seed=7)
    ddpg_results = evaluate_ddpg_snr(ddpg_agent)

    rows = build_rows(q_results, ddpg_results)
    save_csv(rows)
    print(f"Saved PEP vs SNR comparison table to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
