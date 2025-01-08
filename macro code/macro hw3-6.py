import math

# Define the alpha coefficients based on your steady-state computations
alpha1 = 1.0204  # Example values, replace these with your calculated values
alpha2 = 0.1011
alpha3 = 0.0  # Since \bar{a} = 0
alpha4 = -0.0288
alpha5 = 1.0
alpha6 = 0.0  # Since \bar{a} = 0
rho = 0.9  # Example persistence of a_t, replace if needed

# Compute \eta_{kk}
discriminant = (alpha1 - (alpha2 * alpha4 / alpha5) + (1 / alpha5))**2 - 4 * (alpha1 / alpha5)

if discriminant < 0:
    print(f"Error: Negative discriminant ({discriminant}). No real solutions for \eta_kk.")
else:
    eta_kk_positive = 0.5 * (alpha1 - (alpha2 * alpha4 / alpha5) + (1 / alpha5) + math.sqrt(discriminant))
    eta_kk_negative = 0.5 * (alpha1 - (alpha2 * alpha4 / alpha5) + (1 / alpha5) - math.sqrt(discriminant))

    # Choose the root that satisfies the constraint (e.g., \eta_{kk} < 1)
    eta_kk = eta_kk_positive if eta_kk_positive < 1 else eta_kk_negative

    print(f"\nDiscriminant: {discriminant:.4f}")
    print(f"\eta_kk (positive root): {eta_kk_positive:.4f}")
    print(f"\eta_kk (negative root): {eta_kk_negative:.4f}")
    print(f"\nChosen \eta_kk: {eta_kk:.4f}")

# Compute \eta_{\lambda k}
eta_lambda_k = (eta_kk - alpha1) / alpha2

# Compute \eta_{\lambda a}
eta_lambda_a = (alpha3 * alpha4 + alpha3 * alpha5 * eta_lambda_k + alpha6 * rho) / (1 - alpha2 * alpha4 - alpha2 * alpha5 * eta_lambda_k - alpha5 * rho)

# Compute \eta_{ka}
eta_ka = alpha2 * eta_lambda_a + alpha3

# Display results
print("\nOptimal Coefficients:")
print(f"\eta_kk = {eta_kk:.4f}")
print(f"\eta_lambda_k = {eta_lambda_k:.4f}")
print(f"\eta_lambda_a = {eta_lambda_a:.4f}")
print(f"\eta_ka = {eta_ka:.4f}")
