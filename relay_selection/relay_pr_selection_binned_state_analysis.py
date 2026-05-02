import numpy as np


# Edit these values to run your own checks.
SEED = 8
NUM_STATES = 2000
K = 8
DISTANCE_MIN_M = 10.0
DISTANCE_MAX_M = 90.0
REFERENCE_DISTANCE_M = 100.0
ALPHAS = (2, 3, 4)

RHO_ACTIONS = np.linspace(0.05, 0.95, 19)
LAMBDA = 1.0

# Same state-variable bins used by the Q-learning scripts.
SNR_EDGES = np.linspace(0.0, 40.0, 21)
DELTA_EDGES = np.array([0.0, 0.03, 0.08, 0.20, 1.0])
SIGMA_EDGES = np.linspace(0.5, 1.0, 4)
ETA_EDGES = np.linspace(0.6, 1.0, 4)

DELTA_BIN_WEIGHTS = np.array([0.45, 0.30, 0.15, 0.10], dtype=float)
DELTA_BIN_WEIGHTS /= DELTA_BIN_WEIGHTS.sum()


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

    conditional_weights /= conditional_weights.sum()
    bin_index = int(rng.choice(len(DELTA_EDGES) - 1, p=conditional_weights))
    low = max(DELTA_EDGES[bin_index], lower_bound)
    high = DELTA_EDGES[bin_index + 1]
    return clip_open_interval(rng.uniform(low, high), low, high)


def sample_state_variables(rng):
    delta_near = sample_from_edges(rng, DELTA_EDGES, DELTA_BIN_WEIGHTS)
    return {
        "snr_db": sample_from_edges(rng, SNR_EDGES),
        "delta_near": delta_near,
        "delta_far": sample_delta_above(rng, delta_near),
        "sigma_n2": sample_from_edges(rng, SIGMA_EDGES),
        "eta": sample_from_edges(rng, ETA_EDGES),
        "rho": float(rng.choice(RHO_ACTIONS)),
        "alpha": float(rng.choice(ALPHAS)),
    }


def omega(distance, alpha):
    return distance ** (-alpha)


def relay_gain(distance, x_value, y_value, alpha):
    omega_value = omega(distance, alpha)
    hsm_squared = (omega_value / 2.0) * (x_value**2 + y_value**2)
    return omega_value, hsm_squared


def calculate_pb(snr_db, sigma_n2):
    return (10.0 ** (snr_db / 10.0)) * sigma_n2


def calculate_pr_current_formula(pb, eta, rho, lambda_value=LAMBDA):
    return eta * rho * lambda_value * pb


def calculate_pr_with_hsm(hsm_squared, pb, eta, rho):
    return eta * rho * pb * hsm_squared


def calculate_pr_with_distance_pathloss(distance, hsm_squared, alpha, pb, eta, rho):
    return eta * rho * pb * hsm_squared * (distance ** (-alpha))


def sample_relay_block(rng):
    distances = rng.uniform(DISTANCE_MIN_M, DISTANCE_MAX_M, size=K)
    distances = distances / REFERENCE_DISTANCE_M
    x_values = rng.normal(0.0, 1.0, size=K)
    y_values = rng.normal(0.0, 1.0, size=K)
    return distances, x_values, y_values


def analyze_once(rng):
    state = sample_state_variables(rng)
    distances, x_values, y_values = sample_relay_block(rng)
    pb = calculate_pb(state["snr_db"], state["sigma_n2"])

    omega_values = np.empty(K)
    hsm_squared = np.empty(K)
    pr_hsm = np.empty(K)
    pr_distance = np.empty(K)

    for relay_index in range(K):
        omega_values[relay_index], hsm_squared[relay_index] = relay_gain(
            distances[relay_index],
            x_values[relay_index],
            y_values[relay_index],
            state["alpha"],
        )
        pr_hsm[relay_index] = calculate_pr_with_hsm(
            hsm_squared[relay_index],
            pb,
            state["eta"],
            state["rho"],
        )
        pr_distance[relay_index] = calculate_pr_with_distance_pathloss(
            distances[relay_index],
            hsm_squared[relay_index],
            state["alpha"],
            pb,
            state["eta"],
            state["rho"],
        )

    return {
        "state": state,
        "distances": distances,
        "omega": omega_values,
        "hsm_squared": hsm_squared,
        "pr_current": np.full(
            K,
            calculate_pr_current_formula(pb, state["eta"], state["rho"]),
            dtype=float,
        ),
        "pr_hsm": pr_hsm,
        "pr_distance": pr_distance,
        "winner_pr_current": 0,
        "winner_pr_distance": int(np.argmax(pr_distance)),
        "winner_pr_hsm": int(np.argmax(pr_hsm)),
        "winner_hsm": int(np.argmax(hsm_squared)),
        "winner_omega": int(np.argmax(omega_values)),
        "winner_closest": int(np.argmin(distances)),
    }


def summarize(results):
    total = len(results)
    closest_winners = np.array([item["winner_closest"] for item in results])
    hsm_winners = np.array([item["winner_hsm"] for item in results])

    print("Formulas used")
    print("distance = Uniform(10, 90) / 100")
    print("Omega = distance^(-alpha)")
    print("hsm^2 = (Omega / 2) * (X^2 + Y^2), X,Y ~ N(0,1)")
    print("Pr current = eta * rho * Pb * lambda")
    print("Pr hsm = eta * rho * Pb * hsm^2")
    print("Pr distance pathloss = eta * rho * Pb * hsm^2 * distance^(-alpha)")
    print()

    print(f"States sampled: {total}")
    print(f"Seed: {SEED}")
    print()

    formula_specs = [
        ("Pr hsm", "pr_hsm", "winner_pr_hsm"),
        ("Pr distance pathloss", "pr_distance", "winner_pr_distance"),
    ]

    print("Closest relay vs max-Pr relay")
    print(
        "| Formula | max-Pr same as closest | max-Pr same as max hsm^2 | avg closest Pr | avg max Pr | "
        "median closest Pr | median max Pr | avg max/closest |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")

    for label, pr_key, winner_key in formula_specs:
        max_winners = np.array([item[winner_key] for item in results])
        closest_pr = np.array(
            [item[pr_key][item["winner_closest"]] for item in results]
        )
        max_pr = np.array([item[pr_key][item[winner_key]] for item in results])
        ratio = max_pr / np.maximum(closest_pr, np.finfo(float).eps)
        print(
            f"| {label} | "
            f"{np.mean(max_winners == closest_winners) * 100:.2f}% | "
            f"{np.mean(max_winners == hsm_winners) * 100:.2f}% | "
            f"{np.mean(closest_pr):.6e} | "
            f"{np.mean(max_pr):.6e} | "
            f"{np.median(closest_pr):.6e} | "
            f"{np.median(max_pr):.6e} | "
            f"{np.mean(ratio):.4f} |"
        )

    print()
    print("Closest-match percentage by alpha")
    print("| Formula | alpha=2 | alpha=3 | alpha=4 |")
    print("|---|---:|---:|---:|")
    for label, _, winner_key in formula_specs:
        row_values = []
        for alpha in ALPHAS:
            alpha_results = [
                item for item in results if item["state"]["alpha"] == float(alpha)
            ]
            alpha_winners = np.array([item[winner_key] for item in alpha_results])
            alpha_closest = np.array(
                [item["winner_closest"] for item in alpha_results]
            )
            row_values.append(f"{np.mean(alpha_winners == alpha_closest) * 100:.2f}%")
        print(f"| {label} | {row_values[0]} | {row_values[1]} | {row_values[2]} |")

    print()
    print("Max-hsm^2-match percentage by alpha")
    print("| Formula | alpha=2 | alpha=3 | alpha=4 |")
    print("|---|---:|---:|---:|")
    for label, _, winner_key in formula_specs:
        row_values = []
        for alpha in ALPHAS:
            alpha_results = [
                item for item in results if item["state"]["alpha"] == float(alpha)
            ]
            alpha_winners = np.array([item[winner_key] for item in alpha_results])
            alpha_hsm = np.array([item["winner_hsm"] for item in alpha_results])
            row_values.append(f"{np.mean(alpha_winners == alpha_hsm) * 100:.2f}%")
        print(f"| {label} | {row_values[0]} | {row_values[1]} | {row_values[2]} |")

    current_pr_average = np.mean([item["pr_current"][0] for item in results])
    max_pr_hsm_average = np.mean([np.max(item["pr_hsm"]) for item in results])
    max_pr_distance_average = np.mean(
        [np.max(item["pr_distance"]) for item in results]
    )

    print()
    print("Average Pr using best relay per formula")
    print("| Formula | Average Pr |")
    print("|---|---:|")
    print(f"| Current Pr | {current_pr_average:.6e} |")
    print(f"| Pr with hsm | {max_pr_hsm_average:.6e} |")
    print(f"| Pr with distance pathloss | {max_pr_distance_average:.6e} |")

def main():
    rng = np.random.default_rng(SEED)
    results = [analyze_once(rng) for _ in range(NUM_STATES)]
    summarize(results)


if __name__ == "__main__":
    main()
