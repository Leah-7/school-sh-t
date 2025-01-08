import numpy as np
from scipy.optimize import fsolve
import math

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

# Compute Steady-State Output and Consumption
Y_ss = K_ss**alpha
C_ss = Y_ss - delta * K_ss
R_ss = alpha * Y_ss/K_ss +1-delta
# Compute Ratios for Log-Linearization
y_k_ratio = Y_ss / K_ss
c_k_ratio = C_ss / K_ss



# Define Log-Linearized System Coefficients
alpha1 = alpha * y_k_ratio + (1 - delta)
alpha2 = c_k_ratio 
alpha3 =0
alpha4 = alpha * (alpha - 1) * y_k_ratio / R_ss
alpha5 = 1 
alpha6 =0



    

# Display Steady-State Values
print("Steady-State Results:")
print(f"Steady-State Capital (K): {K_ss:.4f}")
print(f"Steady-State Output (Y): {Y_ss:.4f}")
print(f"Steady-State Consumption (C): {C_ss:.4f}")
print(f"Steady-State R (R): {R_ss:.4f}")
print(f"Y/K Ratio: {y_k_ratio:.4f}, C/K Ratio: {c_k_ratio:.4f}")


# Display Coefficients
print("\nLog-Linearized System Coefficients:")
print(f"alpha1 = {alpha1:.4f}")
print(f"alpha2 = {alpha2:.4f}")
print(f"alpha3 = {alpha3:.4f}")
print(f"alpha4 = {alpha4:.4f}")
print(f"alpha5 = {alpha5:.4f}")
print(f"alpha6 = {alpha6:.4f}")


