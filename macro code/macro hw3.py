import numpy as np
from scipy.optimize import fsolve

# Model Parameters
alpha = 0.36      # Capital share
beta = 0.98       # Discount factor
delta = 0.025     # Depreciation rate

# Steady-State Equation for Capital
# Derived from: 1 = beta * [alpha * K^(alpha-1) + (1 - delta)]
def steady_state_k(K):
    return 1 - beta * (alpha * K**(alpha - 1) + 1 - delta)

# Solve for Steady-State Capital (K)
K_ss = fsolve(steady_state_k, x0=10)[0]  # Initial guess: 10

# Compute Steady-State Consumption (C)
C_ss = K_ss**alpha - delta * K_ss

# Print Results
print(f"Steady-State Capital (K): {K_ss:.4f}")
print(f"Steady-State Consumption (C): {C_ss:.4f}")
