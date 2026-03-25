import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.special import kve


# ===================== System Parameters =====================
K = 3
M = 3
LAMBDA = 1.0
SELECTED_RELAY_ORDER = 2
SELECTED_USER_ORDER = 2
BASELINE_RHO = 0.5
BASELINE_DELTA = np.array([0.01, 0.10], dtype=float)
BASELINE_SIGMA_N2 = 1.0
BASELINE_ETA = 0.8

EPS = np.finfo(float).eps
OUTPUT_DIR = Path(__file__).resolve().parent

AK = math.factorial(K) / (
    math.factorial(SELECTED_RELAY_ORDER - 1) * math.factorial(K - SELECTED_RELAY_ORDER)
)
AM = math.factorial(M) / (
    math.factorial(SELECTED_USER_ORDER - 1) * math.factorial(M - SELECTED_USER_ORDER)
)


# ===================== DDPG Hyper-parameters =====================
TRAINING_EPISODES = 10_000
BATCH_SIZE = 64
REPLAY_CAPACITY = 50_000
GAMMA = 0.0  # single-step (contextual bandit)
TAU = 0.005  # soft target update
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
OU_MU = 0.0
OU_THETA = 0.15
OU_SIGMA = 0.2
RHO_LOW = 0.05
RHO_HIGH = 0.95

# Reference rho grid used to evaluate Pareto reward per state
RHO_EVAL_GRID = np.linspace(0.05, 0.95, 19)

# Sampling edges (unchanged from original)
SNR_EDGES = np.linspace(0.0, 40.0, 7)
DELTA_EDGES = np.array([0.0, 0.03, 0.08, 0.20, 1.0])
SIGMA_EDGES = np.linspace(0.5, 1.0, 4)
ETA_EDGES = np.linspace(0.6, 1.0, 4)

DELTA_BIN_WEIGHTS = np.array([0.45, 0.30, 0.15, 0.10], dtype=float)
DELTA_BIN_WEIGHTS /= DELTA_BIN_WEIGHTS.sum()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class State:
    snr_db: float
    delta: np.ndarray
    sigma_n2: float
    eta: float


def clip_open_interval(value, low, high):
    low_open = np.nextafter(low, high)
    high_open = np.nextafter(high, low)
    if not low_open < high_open:
        return float((low + high) / 2.0)
    return float(np.clip(value, low_open, high_open))


def sample_from_edges(rng, edges, weights=None):
    bin_index = int(rng.choice(len(edges) - 1, p=weights))
    low = edges[bin_index]
    high = edges[bin_index + 1]
    return clip_open_interval(rng.uniform(low, high), low, high)


def sample_delta_above(rng, lower_bound):
    conditional_weights = np.zeros(len(DELTA_EDGES) - 1, dtype=float)

    for bin_index, base_weight in enumerate(DELTA_BIN_WEIGHTS):
        bin_low = DELTA_EDGES[bin_index]
        bin_high = DELTA_EDGES[bin_index + 1]
        valid_low = max(bin_low, lower_bound)
        valid_length = max(bin_high - valid_low, 0.0)
        bin_width = bin_high - bin_low
        conditional_weights[bin_index] = base_weight * (valid_length / bin_width)

    weight_sum = conditional_weights.sum()
    if weight_sum <= 0.0:
        raise ValueError(f"No feasible Delta_far value exists above Delta_near={lower_bound}.")

    conditional_weights /= weight_sum
    bin_index = int(rng.choice(len(DELTA_EDGES) - 1, p=conditional_weights))
    low = max(DELTA_EDGES[bin_index], lower_bound)
    high = DELTA_EDGES[bin_index + 1]
    return clip_open_interval(rng.uniform(low, high), low, high)


def sample_state(rng):
    delta_near = sample_from_edges(rng, DELTA_EDGES, DELTA_BIN_WEIGHTS)
    # Enforce the system assumption that the far user experiences larger
    # residual SIC error than the near user.
    delta_far = sample_delta_above(rng, delta_near)
    return State(
        snr_db=sample_from_edges(rng, SNR_EDGES),
        delta=np.array(
            [delta_near, delta_far],
            dtype=float,
        ),
        sigma_n2=sample_from_edges(rng, SIGMA_EDGES),
        eta=sample_from_edges(rng, ETA_EDGES),
    )


def state_to_vector(state):
    """Convert a State to a numpy vector for neural network input."""
    return np.array(
        [state.snr_db / 40.0, state.delta[0], state.delta[1],
         state.sigma_n2, state.eta],
        dtype=np.float32,
    )


STATE_DIM = 5  # snr_db (normalised), delta[0], delta[1], sigma_n2, eta
ACTION_DIM = 1  # rho


def compute_user_pep(snr_db, delta_u, sigma_n2, eta, rho, user_index):
    snr_linear = 10.0 ** (snr_db / 10.0)
    pb = snr_linear * sigma_n2
    pr = eta * rho * LAMBDA * pb

    # The updated closed form uses the same combinatorial orders for both users;
    # the user-specific dependence enters through delta_u only.
    _ = user_index

    g_fixed = np.sqrt(pr / ((1.0 - rho) * LAMBDA * pb + (2.0 - rho) * sigma_n2 + EPS))
    g_fixed_prime = g_fixed * np.sqrt(1.0 - rho)
    g_relay = np.sqrt(g_fixed**2 + g_fixed_prime**2)
    signal_term = g_fixed**2 * (1.0 - rho) * pb
    interference_term = g_fixed**2 * sigma_n2 + g_relay**2 * pb * (delta_u**2)

    sum_total = 0.0

    for ell in range(SELECTED_RELAY_ORDER):
        for ii in range(SELECTED_USER_ORDER):
            k_prime = K - SELECTED_RELAY_ORDER + ell + 1
            m_prime = M - SELECTED_USER_ORDER + ii + 1

            uk = np.sqrt(
                1.0
                + (4.0 * m_prime * interference_term) / (LAMBDA * signal_term + EPS)
            )
            ak = k_prime * (1.0 - 1.0 / (uk**2)) / (2.0 * LAMBDA * signal_term + EPS)

            if ak <= 0.0:
                continue

            term = (
                math.comb(SELECTED_RELAY_ORDER - 1, ell)
                * ((-1) ** ell)
                * math.comb(SELECTED_USER_ORDER - 1, ii)
                * ((-1) ** ii)
                * (ak / (k_prime * m_prime * uk + EPS))
                * (kve(1, ak) - kve(0, ak))
            )

            if np.isfinite(term):
                sum_total += term

    pep_value = 0.5 - (AK * AM / 2.0) * sum_total
    return float(np.clip(np.real_if_close(pep_value), 1e-8, 1.0))


def compute_pep_pair(state, rho):
    return np.array(
        [
            compute_user_pep(state.snr_db, state.delta[0], state.sigma_n2, state.eta, rho, 0),
            compute_user_pep(state.snr_db, state.delta[1], state.sigma_n2, state.eta, rho, 1),
        ],
        dtype=float,
    )


def pareto_front_metrics(costs):
    num_actions = costs.shape[0]
    domination_counts = np.zeros(num_actions, dtype=int)
    frontier_mask = np.ones(num_actions, dtype=bool)

    for i in range(num_actions):
        for j in range(num_actions):
            if i == j:
                continue
            dominates = np.all(costs[j] <= costs[i] + 1e-12) and np.any(costs[j] < costs[i] - 1e-12)
            if dominates:
                domination_counts[i] += 1
                frontier_mask[i] = False

    return frontier_mask, domination_counts


def pareto_reward(costs, action_index):
    normalized_costs = (costs - costs.min(axis=0)) / np.maximum(np.ptp(costs, axis=0), EPS)
    hypervolume_scores = np.prod(1.1 - normalized_costs, axis=1)
    frontier_mask, domination_counts = pareto_front_metrics(costs)

    reward = hypervolume_scores[action_index] - domination_counts[action_index] / max(len(costs) - 1, 1)
    if frontier_mask[action_index]:
        reward += 0.5

    return float(reward), bool(frontier_mask[action_index]), float(hypervolume_scores[action_index])


def rolling_mean(values, window):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values

    window = min(window, values.size)
    kernel = np.ones(window, dtype=float) / window
    averaged = np.convolve(values, kernel, mode="valid")
    padded = np.full(values.size, np.nan, dtype=float)
    padded[window - 1 :] = averaged
    return padded


# ===================== DDPG Neural Networks =====================

class ActorNetwork(nn.Module):
    """Maps state -> rho in (0, 1) via sigmoid."""

    def __init__(self, state_dim, hidden1=128, hidden2=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
            nn.Sigmoid(),
        )

    def forward(self, state):
        return self.net(state)


class CriticNetwork(nn.Module):
    """Maps (state, action) -> Q-value."""

    def __init__(self, state_dim, action_dim, hidden1=128, hidden2=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


# ===================== Replay Buffer =====================

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_vec, action, reward):
        self.buffer.append((state_vec, action, reward))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states = np.array([b[0] for b in batch], dtype=np.float32)
        actions = np.array([b[1] for b in batch], dtype=np.float32).reshape(-1, 1)
        rewards = np.array([b[2] for b in batch], dtype=np.float32).reshape(-1, 1)
        return (
            torch.FloatTensor(states).to(DEVICE),
            torch.FloatTensor(actions).to(DEVICE),
            torch.FloatTensor(rewards).to(DEVICE),
        )

    def __len__(self):
        return len(self.buffer)


# ===================== Ornstein-Uhlenbeck Noise =====================

class OUNoise:
    def __init__(self, size, mu=OU_MU, theta=OU_THETA, sigma=OU_SIGMA):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.copy(self.mu)

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.mu))
        self.state += dx
        return self.state


# ===================== DDPG Agent =====================

class DDPGAgent:
    def __init__(self):
        # Actor
        self.actor = ActorNetwork(STATE_DIM).to(DEVICE)
        self.actor_target = ActorNetwork(STATE_DIM).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)

        # Critic
        self.critic = CriticNetwork(STATE_DIM, ACTION_DIM).to(DEVICE)
        self.critic_target = CriticNetwork(STATE_DIM, ACTION_DIM).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        # Exploration noise
        self.noise = OUNoise(ACTION_DIM)

        # Replay buffer
        self.memory = ReplayBuffer(REPLAY_CAPACITY)

    def select_action(self, state_vec, add_noise=True):
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(DEVICE)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().flatten()
        self.actor.train()

        if add_noise:
            action = action + self.noise.sample()

        # Scale from sigmoid [0,1] to [RHO_LOW, RHO_HIGH] and clip
        rho = float(np.clip(action[0] * (RHO_HIGH - RHO_LOW) + RHO_LOW, RHO_LOW, RHO_HIGH))
        return rho

    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return

        states, actions, rewards = self.memory.sample(BATCH_SIZE)

        # ---- Update Critic ----
        # With GAMMA=0 (single-step), target Q = reward (no next-state bootstrap)
        target_q = rewards
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---- Update Actor ----
        # Actor outputs raw sigmoid [0,1]; scale for critic input
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---- Soft update target networks ----
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)


# ===================== Training Loop =====================

def train_ddpg(seed=7):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = DDPGAgent()

    history = {
        "reward": [],
        "pep_near": [],
        "pep_far": [],
        "rho": [],
        "pareto_hit": [],
        "hypervolume": [],
    }

    for episode in range(TRAINING_EPISODES):
        state = sample_state(rng)
        state_vec = state_to_vector(state)

        # Actor selects continuous rho with exploration noise
        rho = agent.select_action(state_vec, add_noise=True)

        # Evaluate PEP for the chosen rho and a reference grid for Pareto reward
        all_rhos = np.append(RHO_EVAL_GRID, rho)
        all_peps = np.vstack([compute_pep_pair(state, r) for r in all_rhos])
        action_index = len(all_rhos) - 1  # last entry is the chosen rho

        chosen_peps = all_peps[action_index]
        reward, pareto_hit, hypervolume = pareto_reward(all_peps, action_index)

        # Normalise action to [0,1] for replay buffer (match sigmoid output range)
        action_normalised = (rho - RHO_LOW) / (RHO_HIGH - RHO_LOW)
        agent.memory.push(state_vec, action_normalised, reward)

        # Update actor and critic from replay buffer
        agent.update()

        history["reward"].append(reward)
        history["pep_near"].append(chosen_peps[0])
        history["pep_far"].append(chosen_peps[1])
        history["rho"].append(rho)
        history["pareto_hit"].append(float(pareto_hit))
        history["hypervolume"].append(hypervolume)

        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(history["reward"][-250:])
            avg_rho = np.mean(history["rho"][-250:])
            print(f"Episode {episode + 1}/{TRAINING_EPISODES}  "
                  f"Avg Reward (last 250): {avg_reward:.4f}  "
                  f"Avg rho: {avg_rho:.4f}")

    return agent, history


# ===================== Policy Evaluation =====================

def evaluate_ddpg_policy(agent):
    """Sweep SNR and eta independently while keeping other params at baseline."""

    # --- PEP vs SNR ---
    snr_grid = np.arange(0.0, 41.0, 2.0)
    learned_rho_snr = []
    learned_pep_snr = []
    baseline_pep_snr = []

    for snr_db in snr_grid:
        state = State(
            snr_db=snr_db,
            delta=BASELINE_DELTA.copy(),
            sigma_n2=BASELINE_SIGMA_N2,
            eta=BASELINE_ETA,
        )
        rho_star = agent.select_action(state_to_vector(state), add_noise=False)
        learned_rho_snr.append(rho_star)
        learned_pep_snr.append(compute_pep_pair(state, rho_star))
        baseline_pep_snr.append(compute_pep_pair(state, BASELINE_RHO))

    # --- PEP vs eta ---
    eta_grid = np.linspace(0.1, 1.0, 19)
    learned_rho_eta = []
    learned_pep_eta = []
    baseline_pep_eta = []

    for eta_val in eta_grid:
        state = State(
            snr_db=20.0,  # mid-range SNR for the eta sweep
            delta=BASELINE_DELTA.copy(),
            sigma_n2=BASELINE_SIGMA_N2,
            eta=eta_val,
        )
        rho_star = agent.select_action(state_to_vector(state), add_noise=False)
        learned_rho_eta.append(rho_star)
        learned_pep_eta.append(compute_pep_pair(state, rho_star))
        baseline_pep_eta.append(compute_pep_pair(state, BASELINE_RHO))

    return {
        "snr_grid": snr_grid,
        "learned_rho_snr": np.asarray(learned_rho_snr),
        "learned_pep_snr": np.vstack(learned_pep_snr),
        "baseline_pep_snr": np.vstack(baseline_pep_snr),
        "eta_grid": eta_grid,
        "learned_rho_eta": np.asarray(learned_rho_eta),
        "learned_pep_eta": np.vstack(learned_pep_eta),
        "baseline_pep_eta": np.vstack(baseline_pep_eta),
    }


# ===================== Plotting Functions =====================

def plot_training_history(history, output_path):
    episodes = np.arange(1, len(history["reward"]) + 1)
    window = 250

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(episodes, history["reward"], color="steelblue", alpha=0.25, linewidth=1)
    axes[0, 0].plot(
        episodes,
        rolling_mean(history["reward"], window),
        color="midnightblue",
        linewidth=2,
        label=f"{window}-episode mean",
    )
    axes[0, 0].set_title("Pareto Reward During Training")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(
        episodes,
        rolling_mean(history["pep_near"], window),
        linewidth=2,
        color="darkorange",
        label="Near-user PEP",
    )
    axes[0, 1].plot(
        episodes,
        rolling_mean(history["pep_far"], window),
        linewidth=2,
        color="forestgreen",
        label="Far-user PEP",
    )
    axes[0, 1].set_title("Rolling Average PEP of Selected rho")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Average PEP")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(
        episodes,
        100.0 * rolling_mean(history["pareto_hit"], window),
        linewidth=2,
        color="firebrick",
        label="Pareto-front hit rate",
    )
    axes[1, 0].set_title("How Often the Agent Picks a Pareto-Optimal rho")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Hit Rate (%)")
    axes[1, 0].set_ylim(0.0, 105.0)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(episodes, history["rho"], color="purple", alpha=0.18, linewidth=1)
    axes[1, 1].plot(
        episodes,
        rolling_mean(history["rho"], window),
        color="black",
        linewidth=2,
        label=f"{window}-episode mean rho",
    )
    axes[1, 1].set_title("rho Selected by the Agent")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("rho")
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.suptitle("DDPG Training Progress (Actor-Critic, Continuous rho)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pep_vs_snr(results, output_path):
    """PEP vs SNR for learned DDPG policy vs fixed-rho baseline."""
    snr_grid = results["snr_grid"]
    learned_pep = results["learned_pep_snr"]
    baseline_pep = results["baseline_pep_snr"]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogy(
        snr_grid, learned_pep[:, 0], "-o", linewidth=2, color="darkorange",
        markersize=5, label="Near user (DDPG learned rho)",
    )
    ax.semilogy(
        snr_grid, learned_pep[:, 1], "-s", linewidth=2, color="forestgreen",
        markersize=5, label="Far user (DDPG learned rho)",
    )
    ax.semilogy(
        snr_grid, baseline_pep[:, 0], "--o", linewidth=1.5, color="peru",
        markersize=4, label="Near user (fixed rho = 0.5)",
    )
    ax.semilogy(
        snr_grid, baseline_pep[:, 1], "--s", linewidth=1.5, color="limegreen",
        markersize=4, label="Far user (fixed rho = 0.5)",
    )

    ax.set_title("PEP vs SNR: DDPG Policy vs Fixed rho Baseline", fontsize=14)
    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("Pairwise Error Probability (PEP)", fontsize=12)
    ax.set_ylim(1e-5, 1.0)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pep_vs_eta(results, output_path):
    """PEP vs eta for learned DDPG policy vs fixed-rho baseline."""
    eta_grid = results["eta_grid"]
    learned_pep = results["learned_pep_eta"]
    baseline_pep = results["baseline_pep_eta"]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogy(
        eta_grid, learned_pep[:, 0], "-o", linewidth=2, color="darkorange",
        markersize=5, label="Near user (DDPG learned rho)",
    )
    ax.semilogy(
        eta_grid, learned_pep[:, 1], "-s", linewidth=2, color="forestgreen",
        markersize=5, label="Far user (DDPG learned rho)",
    )
    ax.semilogy(
        eta_grid, baseline_pep[:, 0], "--o", linewidth=1.5, color="peru",
        markersize=4, label="Near user (fixed rho = 0.5)",
    )
    ax.semilogy(
        eta_grid, baseline_pep[:, 1], "--s", linewidth=1.5, color="limegreen",
        markersize=4, label="Far user (fixed rho = 0.5)",
    )

    ax.set_title("PEP vs eta: DDPG Policy vs Fixed rho Baseline", fontsize=14)
    ax.set_xlabel("Energy Harvesting Efficiency (eta)", fontsize=12)
    ax.set_ylabel("Pairwise Error Probability (PEP)", fontsize=12)
    ax.set_ylim(1e-5, 1.0)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ===================== Main =====================

def main():
    print("=" * 60)
    print("  DDPG for rho Optimisation in Cooperative NOMA Relay")
    print("=" * 60)

    agent, history = train_ddpg()
    evaluation_results = evaluate_ddpg_policy(agent)

    # Save plots
    training_plot_path = OUTPUT_DIR / "ddpg_training_progress.png"
    pep_snr_plot_path = OUTPUT_DIR / "ddpg_pep_vs_snr.png"
    pep_eta_plot_path = OUTPUT_DIR / "ddpg_pep_vs_eta.png"

    plot_training_history(history, training_plot_path)
    plot_pep_vs_snr(evaluation_results, pep_snr_plot_path)
    plot_pep_vs_eta(evaluation_results, pep_eta_plot_path)

    # Print summary
    print("\n" + "=" * 60)
    print("  Training Complete")
    print("=" * 60)
    print(f"Algorithm: DDPG (Deep Deterministic Policy Gradient)")
    print(f"Episodes: {TRAINING_EPISODES}")
    print(f"Formulation: single-step with gamma = {GAMMA} (contextual bandit)")
    print(f"Reward mean over last 250 episodes: {np.mean(history['reward'][-250:]):.4f}")
    print(f"Pareto-front hit rate over last 250 episodes: "
          f"{100.0 * np.mean(history['pareto_hit'][-250:]):.2f}%")
    print(f"Average learned rho over last 250 episodes: "
          f"{np.mean(history['rho'][-250:]):.4f}")

    print(f"\nSaved training plot: {training_plot_path}")
    print(f"Saved PEP vs SNR plot: {pep_snr_plot_path}")
    print(f"Saved PEP vs eta plot: {pep_eta_plot_path}")


if __name__ == "__main__":
    main()
