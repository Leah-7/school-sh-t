import sympy as sp

# Define symbols and functions
k_t, k_t1, a_t = sp.symbols('k_t k_t1 a_t')
eta_kk, eta_ka, eta_lambda_k, eta_lambda_a = sp.symbols('eta_kk eta_ka eta_lambda_k eta_lambda_a')

# Given parameters
alpha = 0.36
beta = 0.98
delta = 0.025
rho = 0.95

# Define symbols
K, C, Y, R, delta, alpha = sp.symbols('K C Y R delta alpha')

# Given parameters
alpha_value = 0.36
delta_value = 0.025

# Production function
Y_expr = K**alpha_value

# Resource constraint
resource_constraint = sp.Eq(Y_expr, C + delta_value * K)

# Solve for steady-state capital K
K_value = sp.solve(resource_constraint, K)[0]

# Calculate steady-state output Y
Y_value = Y_expr.subs(K, K_value)

# Calculate steady-state consumption C
C_value = Y_value - delta_value * K_value

# Calculate steady-state return on capital R
R_value = alpha_value * K_value**(alpha_value - 1)

# Display results
print("Steady-state capital (K):", K_value)
print("Steady-state output (Y):", Y_value)
print("Steady-state consumption (C):", C_value)
print("Steady-state return on capital (R):", R_value)





alpha_1 = (1 - delta) + alpha * (bar_Y / bar_K)
alpha_2 = -alpha * (bar_C / bar_K)
alpha_3 = bar_Y / bar_K
alpha_4 = alpha * (alpha - 1) * (bar_Y / (bar_K * bar_R))
alpha_5 = 1
alpha_6 = alpha * (bar_Y / (bar_K * bar_R))

# Decision rules
k_t1 = eta_kk * k_t + eta_ka * a_t
lambda_t = eta_lambda_k * k_t + eta_lambda_a * a_t

# Equations
eq1 = -eta_kk + alpha_1 + alpha_2 * eta_lambda_k
eq2 = -eta_lambda_k + alpha_4 * eta_kk + alpha_5 * eta_lambda_k * eta_kk
eq3 = -eta_ka + alpha_2 * eta_lambda_a + alpha_3
eq4 = -eta_lambda_a + alpha_4 * eta_ka + alpha_5 * eta_lambda_k * eta_ka + (alpha_5 * eta_lambda_a + alpha_6) * rho

# Solving the system of equations
solution = sp.solve([eq1, eq2, eq3, eq4], [eta_kk, eta_lambda_k, eta_ka, eta_lambda_a])

# Display results
print("Solution for eta_kk:", solution[eta_kk])
print("Solution for eta_lambda_k:", solution[eta_lambda_k])
print("Solution for eta_ka:", solution[eta_ka])
print("Solution for eta_lambda_a:", solution[eta_lambda_a])
