import sympy as sp
import matplotlib.pyplot as plt

# Define symbols for variables and parameters
Y, K, N, C, I, w, r, A = sp.symbols('Y K N C I w r A')  # Variables
Y_ss, K_ss, N_ss, C_ss, I_ss = sp.symbols('Y_ss K_ss N_ss C_ss I_ss')  # Steady states
alpha, delta, rho, epsilon = sp.symbols('alpha delta rho epsilon')  # Parameters

# Define log deviations from steady state
hat_Y = sp.Symbol('hat_Y')  # Log deviation for Y
hat_K = sp.Symbol('hat_K')  # Log deviation for K
hat_N = sp.Symbol('hat_N')  # Log deviation for N
hat_C = sp.Symbol('hat_C')  # Log deviation for C
hat_I = sp.Symbol('hat_I')  # Log deviation for I
hat_w = sp.Symbol('hat_w')  # Log deviation for w
hat_A = sp.Symbol('hat_A')  # Log deviation for A

# Define log-linearized equations
production_func = sp.Eq(hat_Y, hat_A + alpha * hat_K + (1 - alpha) * hat_N)
resource_constraint = sp.Eq(hat_Y, (C_ss / Y_ss) * hat_C + (I_ss / Y_ss) * hat_I)
wage_equation = sp.Eq(hat_w, hat_A + alpha * hat_K - alpha * hat_N)

# Solve for steady-state deviations
print("\nSolving steady-state equations...")
steady_state_solution = sp.solve(
    [production_func, resource_constraint, wage_equation],
    [hat_Y, hat_C, hat_w]
)

# Simplify the steady-state solutions
simplified_solution = {var: sp.simplify(expr) for var, expr in steady_state_solution.items()}
print("\nSimplified Steady-state solutions:")
print(simplified_solution)

# Example parameter values
params = {alpha: 0.33, C_ss: 0.7, Y_ss: 1, I_ss: 0.3}
additional_params = {
    hat_A: 0.1,  # Example value for hat_A
    hat_K: 0.2,  # Example value for hat_K
    hat_N: 0.3,  # Example value for hat_N
    hat_I: 0.05  # Example value for hat_I
}

# Substitute parameter values into the solutions
numerical_solution = {var: expr.subs({**params, **additional_params}) for var, expr in steady_state_solution.items()}
print("\nFully substituted numerical solutions:")
print(numerical_solution)

# Evaluate numerical solutions
try:
    numerical_values = {var: float(expr.evalf()) for var, expr in numerical_solution.items()}
    print("\nFinal numerical values:")
    print(numerical_values)
except Exception as e:
    print("\nError during numerical evaluation:", e)
    print("Unresolved variables:", numerical_solution)

# Bar Plot for Percentage Deviations
variables = ["Consumption (C)", "Output (Y)", "Wage (w)"]
values = [numerical_values[hat_C], numerical_values[hat_Y], numerical_values[hat_w]]

plt.bar(variables, values)
plt.title("Steady-State Deviations")
plt.ylabel("Percentage Deviation")
plt.show()

# Sensitivity Analysis for alpha
alpha_values = [0.3, 0.33, 0.36]
sensitivity_results = []

for a in alpha_values:
    params[alpha] = a
    numerical_solution = {var: expr.subs({**params, **additional_params}) for var, expr in steady_state_solution.items()}
    numerical_values = {var: float(expr.evalf()) for var, expr in numerical_solution.items()}
    sensitivity_results.append(numerical_values)

# Display Sensitivity Results
print("\nSensitivity Analysis Results:")
for i, res in enumerate(sensitivity_results):
    print(f"\nResults for alpha = {alpha_values[i]}:")
    print(res)

# Optional: Plot Sensitivity Results
plt.figure()
for i, res in enumerate(sensitivity_results):
    values = [res[hat_C], res[hat_Y], res[hat_w]]
    plt.plot(variables, values, label=f"alpha = {alpha_values[i]}")
plt.title("Sensitivity Analysis of Deviations by Alpha")
plt.ylabel("Percentage Deviation")
plt.legend()
plt.show()