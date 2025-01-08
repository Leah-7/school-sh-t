import sympy as sp

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
hat_r = sp.Symbol('hat_r')  # Log deviation for r
hat_A = sp.Symbol('hat_A')  # Log deviation for A

# Define log-linearized equations
production_func = sp.Eq(hat_Y, hat_A + alpha * hat_K + (1 - alpha) * hat_N)
resource_constraint = sp.Eq(hat_Y, (C_ss / Y_ss) * hat_C + (I_ss / Y_ss) * hat_I)

# Solve for steady-state deviations
print("\nSolving steady-state equations...")
steady_state_solution = sp.solve([production_func, resource_constraint], [hat_Y, hat_C])

# Simplify the steady-state solutions
simplified_solution = {var: sp.simplify(expr) for var, expr in steady_state_solution.items()}
print("\nSimplified Steady-state solutions:")
print(simplified_solution)

# Example parameter values
params = {alpha: 0.33, C_ss: 0.7, Y_ss: 1, I_ss: 0.3}

# Add additional parameter values for unresolved variables
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