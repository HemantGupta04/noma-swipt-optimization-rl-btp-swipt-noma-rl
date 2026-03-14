# SWIPT-NOMA Optimization using Reinforcement Learning

This project implements Reinforcement Learning (RL) agents to optimize the **Pairwise Error Probability (PEP)** in a Cooperative Non-Orthogonal Multiple Access (NOMA) system with Simultaneous Wireless Information and Power Transfer (SWIPT). 

By intelligently controlling the Continuous Power Splitting ratio ($\rho$), the RL agents minimize the PEP for both near and far users simultaneously, significantly outperforming fixed-ratio baselines.

---

## 📖 Algorithm Details

This project compares two different RL approaches to solve the continuous state optimization problem:

1. **Q-Learning (Tabular Contextual Bandit)** (`QLearning.py`)
   - **Formulation:** Single-step MDP ($\gamma = 0$) where the agent learns the expected immediate Pareto-reward for choosing a discretized $\rho$.
   - **State Space:** Binned representations of SNR, SNR distances ($\delta$), noise variance ($\sigma_n^2$), and energy efficiency ($\eta$).
   - **Action Space:** 19 discrete actions mapping to $\rho \in [0.05, 0.95]$.

2. **DDPG - Deep Deterministic Policy Gradient** (`DDPG.py`)
   - **Formulation:** Actor-Critic continuous control method acting as a contextual bandit ($\gamma = 0$).
   - **State Space:** Continuous 5-dimensional vector: `[SNR_dB / 40.0, delta_1, delta_2, sigma_n2, eta]`.
   - **Action Space:** Continuous $\rho \in (0, 1)$ output via a Sigmoid activation (scaled to $[0.05, 0.95]$).
   - **Networks:** 2-layer MLPs (128 -> 64 hidden units).
   - **Exploration:** Ornstein-Uhlenbeck (OU) Noise injected into the continuous action space.

### Reward Function (Pareto Hypervolume)
Both agents use a custom, unified reward function designed for Multi-Objective Optimization to balance the near-user and far-user PEPs:
- Computes a state-specific **Pareto Front** of PEPs.
- Grants higher rewards for maximizing the **Hypervolume** (pushing PEPs closer to zero).
- Penalizes actions that are strictly **dominated** by other $\rho$ choices.
- Grants a boolean +0.5 bonus if the chosen $\rho$ lands exactly on the Pareto Frontier.

---

## ⚙️ System Variables & Context Values

Here are the environment variables defining the NOMA-SWIPT channel and the ranges they traverse during training and evaluation:

| Parameter | Type | Base Value | Grid / Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| `K` | Int | `5` | - | Total number of relays available. |
| `M` | Int | `2` | - | Total number of users receiving the superposition coded signal. |
| `K_ORDER` | Int | `3` | - | Target index for the partially selected relay. |
| `USER_ORDERS` | Tuple | `(1, 2)` | - | Indices representing the near (1) and far (2) users. |
| `LAMBDA` | Float | `1.0` | - | Path loss / structural environmental scale factor. |
| `BASELINE_RHO` | Float | `0.5` | - | The constant, unoptimized power-splitting ratio baseline. |
| `SNR_DB` | Float | `20.0 dB` | `[0.0, 40.0]` | Signal-to-Noise Ratio at the source transmitter. |
| `DELTA` ($\delta$) | Array | `[0.05, 0.10]` | `[0.0, 1.0]` | The distances/channel variances for the near and far users. |
| `SIGMA_N2` ($\sigma_n^2$) | Float | `1.0` | `[0.5, 1.0]` | Additive White Gaussian Noise (AWGN) variance. |
| `ETA` ($\eta$) | Float | `0.8` | `[0.6, 1.0]` | Energy conversion efficiency of the SWIPT relay harvesters. |

---

## 📊 Results and Evaluation

The agents are evaluated on completely unseen continuous states. We track the rolling **Average Reward**, the raw **PEP values**, and the **Hit Rate** measuring how often the generalized network correctly predicts the optimal continuous Pareto $\rho$.

You can view the specific learning curves and policy behavior against baselines in the `plots/` directory:

- `plots/q_learning_training_progress.png`: Q-Table convergence over 10,000 episodes
- `plots/ddpg_training_progress.png`: Actor/Critic convergence and OU-Noise exploration curves.
- `plots/q_learning_policy_evaluation.png` / `plots/ddpg_pep_vs_snr.png`: Sweeps demonstrating the massive, orders-of-magnitude PEP reduction RL achieves over a fixed $0.5$ rho at high SNRs.
- `plots/pep_vs_rho_fixed_state.png`: Visualizes why optimization is hard—showing the conflicting "bathtub" curves of the near and far user PEPs across the $\rho$ axis.
