import numpy as np



SEED = 7
K = 8
DISTANCE_MIN_M = 10.0
DISTANCE_MAX_M = 90.0
REFERENCE_DISTANCE_M = 100.0
ALPHAS = (2, 3, 4)

SNR_DB = 20.0
SIGMA_N2 = 1.0
ETA = 0.8
RHO = 0.25
LAMBDA = 1.0


def sample_block(seed=SEED):
    rng = np.random.default_rng(seed)
    distances = rng.uniform(DISTANCE_MIN_M, DISTANCE_MAX_M, size=K)

    distances /= REFERENCE_DISTANCE_M  # normalize distances by reference distance for easier interpretation
    x_values = rng.normal(0.0, 1.0, size=K)
    y_values = rng.normal(0.0, 1.0, size=K)

    return distances, x_values, y_values


def omega(distance, alpha):
    # return 1.0 / (1.0 + (distance / REFERENCE_DISTANCE_M) ** alpha)
    return distance ** (-alpha)  #


def relay_gain(distance, x_value, y_value, alpha):
    omega_value = omega(distance, alpha)
    hsm_squared = (omega_value / 2.0) * (x_value**2 + y_value**2)

    # hsm_bounded = hsm_squared / (1 + hsm_squared)
    # hsm_squared = hsm_bounded 
    return omega_value, hsm_squared


def calculate_pb(snr_db=SNR_DB, sigma_n2=SIGMA_N2):
    return (10.0 ** (snr_db / 10.0)) * sigma_n2


def calculate_pr_current_formula(pb, eta=ETA, rho=RHO, lambda_value=LAMBDA):
    return eta * rho * lambda_value * pb


def calculate_pr_with_hsm(distance, hsm_squared, alpha, pb, eta=ETA, rho=RHO):
    return eta * rho * pb * hsm_squared


def calculate_pr_with_distance_pathloss(distance, hsm_squared, alpha, pb, eta=ETA, rho=RHO):
    return eta * rho * pb * hsm_squared * (distance ** (-alpha))


def print_table(alpha, distances, x_values, y_values):
    pb = calculate_pb()
    pr_current = calculate_pr_current_formula(pb)
    order = np.argsort(distances)

    rows = []
    for relay_index in range(K):
        omega_value, hsm_squared = relay_gain(
            distances[relay_index],
            x_values[relay_index],
            y_values[relay_index],
            alpha,
        )
        pr_hsm = calculate_pr_with_hsm(
            distances[relay_index],
            hsm_squared,
            alpha,
            pb,
        )
        pr_distance = calculate_pr_with_distance_pathloss(
            distances[relay_index],
            hsm_squared,
            alpha,
            pb,
        )
        rows.append((relay_index, omega_value, hsm_squared, pr_hsm, pr_distance))

    # Select the relay by the same final Pr metric shown in the last column.
    selected_relay_index = max(rows, key=lambda row: row[4])[0]

    print(f"Alpha = {alpha}")
    print("Selected relay = argmax(Pr distance pathloss)")
    print("| Relay | distance d | Omega | X | Y | hsm^2=|hsm|^2 | Pr current | Pr hsm | Pr distance pathloss |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for relay_index in order:
        _, omega_value, hsm_squared, pr_hsm, pr_distance = rows[relay_index]
        selected = " **selected**" if relay_index == selected_relay_index else ""
        print(
            f"| {relay_index + 1}{selected} | "
            f"{distances[relay_index]:.3f} | "
            f"{omega_value:.6f} | "
            f"{x_values[relay_index]:.6f} | "
            f"{y_values[relay_index]:.6f} | "
            f"{hsm_squared:.6f} | "
            f"{pr_current:.6f} | "
            f"{pr_hsm:.6f} | "
            f"{pr_distance:.6e} |"
        )

    selected_row = rows[selected_relay_index]
    print(
        f"selected_relay={selected_relay_index + 1}, "
        f"selected_hsm2={selected_row[2]:.6f}, "
        f"selected_Pr_current={pr_current:.6f}, "
        f"selected_Pr_hsm={selected_row[3]:.6f}, "
        f"selected_Pr_distance_pathloss={selected_row[4]:.6e}"
    )
    print()


def main():
    distances, x_values, y_values = sample_block()
    pb = calculate_pb()

    # print("Formulas used:")
    # print("Omega(d, alpha) = 1 / (1 + (d / 100)^alpha)")
    # print("X, Y ~ N(0, 1)")
    # print("hsm = sqrt(Omega / 2) * (X + jY)")
    # print("hsm^2 = |hsm|^2 = (Omega / 2) * (X^2 + Y^2)")
    # print("Current-code Pr = eta * rho * LAMBDA * Pb")
    # print("HSM Pr = eta * rho * Pb * hsm^2")
    # print("Distance-pathloss Pr = eta * rho * Pb * hsm^2 * distance^(-alpha)")
    # print()
    print(
        f"Parameters: seed={SEED}, K={K}, distance=[{DISTANCE_MIN_M}, {DISTANCE_MAX_M}] m, "
        f"reference_distance={REFERENCE_DISTANCE_M} m"
    )
    print(f"SNR={SNR_DB} dB, sigma_n2={SIGMA_N2}, Pb={pb:.6f}, eta={ETA}, rho={RHO}")
    print()

    for alpha in ALPHAS:
        print_table(alpha, distances, x_values, y_values)


if __name__ == "__main__":
    main()
