import numpy as np

# Parameters given in the problem
beta = 0.99  # Discount factor
investment_output_ratio = 0.225  # Investment-to-output ratio
labor_supply = 1 / 3  # Steady-state labor supply
income_share_capital = 1 / 3  # Capital share in production

# Step 1: Calibration (Q5)
alpha = income_share_capital  # Capital share in output
# Calculate the depreciation rate (delta)
delta = investment_output_ratio / (1 / beta - 1)
# Capital-output ratio
capital_output_ratio = investment_output_ratio / delta
# Steady-state output using Cobb-Douglas production (assuming A = 1)
A = 1  # Normalized technology level
steady_state_Y_calibration = ((labor_supply ** (1 - alpha)) / capital_output_ratio) ** (1 / (1 - alpha))
# Steady-state capital
steady_state_K_calibration = capital_output_ratio * steady_state_Y_calibration
# Steady-state wage
steady_state_w_calibration = (1 - alpha) * steady_state_Y_calibration / labor_supply
# Calibrate theta (using FOC for utility)
theta = steady_state_w_calibration * (1 - labor_supply)

# Print calibration results
print("Q5: Calibration Results")
print(f"Calibrated capital share (alpha): {alpha:.4f}")
print(f"Calibrated depreciation rate (delta): {delta:.4f}")
print(f"Calibrated theta (θ): {theta:.4f}")
print(f"Steady-state output (Y): {steady_state_Y_calibration:.4f}")
print(f"Steady-state capital (K): {steady_state_K_calibration:.4f}")
print(f"Steady-state wage (w): {steady_state_w_calibration:.4f}")
print("")

# Step 2: Steady-State Calculation (Q6)
# Recalculate steady-state output, capital, consumption, and other variables using calibrated values
steady_state_K = steady_state_K_calibration
steady_state_Y = steady_state_Y_calibration
steady_state_C = steady_state_Y - delta * steady_state_K
steady_state_w = steady_state_w_calibration
steady_state_r = alpha * steady_state_Y / steady_state_K - delta

# Print steady-state results
print("Q6: Steady-State Results")
print(f"Steady-state output (Y): {steady_state_Y:.4f}")
print(f"Steady-state capital (K): {steady_state_K:.4f}")
print(f"Steady-state consumption (C): {steady_state_C:.4f}")
print(f"Steady-state wage (w): {steady_state_w:.4f}")
print(f"Steady-state interest rate (r): {steady_state_r:.4f}")
print(f"Depreciation rate (delta): {delta:.4f}")