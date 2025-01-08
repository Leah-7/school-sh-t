import numpy as np
from scipy.optimize import fsolve

# Model Parameters
alpha = 0.36      # Capital share
beta = 0.98       # Discount factor
delta = 0.025     # Depreciation rate
rho = 0.95        # Persistence of technology shock

# Steady-State Equation for Capital
def steady_state_k(K):
    return 1 - beta * (alpha * K**(alpha - 1) + 1 - delta)

# Solve for Steady-State Capital (K)
K_ss = fsolve(steady_state_k, x0=10)[0]  # Initial guess: 10

# Compute Steady-State Consumption (C)
C_ss = K_ss**alpha - delta * K_ss

# Log-linearize around the steady state
# Define matrices for the linearized system
A = np.array([
    [1, 0, -1],  # Resource constraint: C + K' = alpha*K + (1-delta)*K
    [0, beta, -1],  # Euler equation linearized
    [0, 0, rho]    # Technology shock process: a' = rho*a
])

B = np.array([
    [alpha, (1 - delta)],  # Resource constraint coefficients
    [(1 - alpha), -1],     # Euler equation coefficients
    [0, 1]                 # Technology shock coefficients
])

# Solve the linearized system for decision rules
def solve_linear_system():
    # Coefficients for policy functions
    phi_k = alpha / (1 - beta * (1 - delta))
    phi_a = (1 - alpha) / (1 - beta * rho)
    
    print("Log-Linearized Results:")
    print(f"Capital policy coefficient (phi_k): {phi_k:.4f}")
    print(f"Technology shock coefficient (phi_a): {phi_a:.4f}")
    return phi_k, phi_a

# Compute the policy coefficients
phi_k, phi_a = solve_linear_system()

# Print Steady-State Results
print(f"Steady-State Capital (K): {K_ss:.4f}")
print(f"Steady-State Consumption (C): {C_ss:.4f}")
