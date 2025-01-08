import numpy as np

# Given parameters
alpha = 0.36
beta = 0.98
delta = 0.025
rho = 0.95
omega = 0.01  # not directly needed for the deterministic steady state or linearization

# 1. Compute steady state
# From the steady state Euler equation:
# 1 = beta * [alpha*K^(alpha-1) + 1 - delta]
# Solve for K
# alpha * K^(alpha-1) + 1 - delta = 1/beta

lhs_target = 1.0 / beta
# alpha * K^(alpha-1) + (1 - delta) = lhs_target
# => alpha * K^(alpha-1) = lhs_target - (1 - delta)
val = lhs_target - (1 - delta)

# K^(alpha-1) = val / alpha
K_steady = (val / alpha)**(1/(alpha-1))

# Compute other steady states
Y_steady = K_steady**alpha
I_steady = delta * K_steady
C_steady = Y_steady - I_steady

# Ratios
C_Y = C_steady / Y_steady
I_Y = I_steady / Y_steady

# 2. We define the linearized system.
# Variables: We want decision rules:
# hat(c_t) = c_k * hat(k_t) + c_a * hat(a_t)
# hat(k_{t+1}) = k_k * hat(k_t) + k_a * hat(a_t)

# From the model’s linearization, we have two key equations (Euler and cap. acc):
# Euler (log-linearized) typically:
# hat(c_t) = E_t[hat(c_{t+1})] + (1 - alpha)*hat(k_t) - rho * hat(a_t)

# Capital accumulation with resource constraint:
# hat(k_{t+1}) = (1-delta)*hat(k_t) + delta*hat(i_t)
# and from resource constraint:
# hat(a_t) + alpha*hat(k_t) = C_Y*hat(c_t) + I_Y*hat(i_t)
# => hat(i_t) = (hat(a_t) + alpha*hat(k_t) - C_Y*hat(c_t))/I_Y

# Substitute hat(i_t) into cap. accumulation:
# hat(k_{t+1}) = (1-delta)*hat(k_t) + delta*(hat(a_t) + alpha*hat(k_t) - C_Y*hat(c_t))/I_Y

# Also, we must handle expectations:
# E_t[hat(c_{t+1})] = c_k * E_t[hat(k_{t+1})] + c_a * E_t[hat(a_{t+1})]
# E_t[hat(k_{t+1})] = k_k * hat(k_t) + k_a * hat(a_t)
# E_t[hat(a_{t+1})] = rho*hat(a_t)

# So:
# E_t[hat(c_{t+1})] = c_k*(k_k*hat(k_t) + k_a*hat(a_t)) + c_a*(rho*hat(a_t))

# We want a system in terms of coefficients c_k, c_a, k_k, k_a. 
# Consider that these must hold for any (hat(k_t), hat(a_t)), so we equate coefficients by state.

# Let's set up the coefficient matrices. We'll get 4 equations (matching coefficients on hat(k_t) and hat(a_t) from both main equations).

# Equation 1: Euler eq:
# hat(c_t) = c_k*hat(k_t) + c_a*hat(a_t)
# E_t[hat(c_{t+1})] = c_k*(k_k*hat(k_t) + k_a*hat(a_t)) + c_a*(rho*hat(a_t))
# => = c_k*k_k*hat(k_t) + (c_k*k_a + c_a*rho)*hat(a_t)

# Euler:
# c_k*hat(k_t) + c_a*hat(a_t) = [c_k*k_k*hat(k_t) + (c_k*k_a + c_a*rho)*hat(a_t)] + (1-alpha)*hat(k_t) - rho*hat(a_t)

# Collect terms by hat(k_t) and hat(a_t):

# For hat(k_t):
# c_k = c_k*k_k + (1 - alpha)

# For hat(a_t):
# c_a = (c_k*k_a + c_a*rho) - rho

# That gives us two equations.

# Equation 2: Capital accumulation and resource constraint combined:
# hat(k_{t+1}) = k_k*hat(k_t) + k_a*hat(a_t)
#
# Also:
# hat(k_{t+1}) = (1-delta)*hat(k_t) + (delta/I_Y)*[hat(a_t) + alpha*hat(k_t) - C_Y*(c_k*hat(k_t)+c_a*hat(a_t))]
#
# Collect hat(k_t), hat(a_t):

# hat(k_{t+1}) on LHS: k_k*hat(k_t) + k_a*hat(a_t)
# On RHS:
# = (1-delta)*hat(k_t) + (delta/I_Y)[hat(a_t) + alpha*hat(k_t) - C_Y*(c_k*hat(k_t)+c_a*hat(a_t))]
# = (1-delta)*hat(k_t) + (delta/I_Y)[alpha*hat(k_t) + hat(a_t) - C_Y*c_k*hat(k_t) - C_Y*c_a*hat(a_t)]

# Combine hat(k_t):
# = [(1-delta) + (delta/I_Y)*alpha - (delta/I_Y)*C_Y*c_k]*hat(k_t)
# Combine hat(a_t):
# = [(delta/I_Y) - (delta/I_Y)*C_Y*c_a]*hat(a_t)

# Equate to k_k*hat(k_t) + k_a*hat(a_t):

# For hat(k_t):
# k_k = (1-delta) + (delta/I_Y)*alpha - (delta/I_Y)*C_Y*c_k

# For hat(a_t):
# k_a = (delta/I_Y) - (delta/I_Y)*C_Y*c_a

# Now we have 4 equations in total:

# From Euler eq by matching coefficients:
# (i)   c_k = c_k*k_k + (1 - alpha)
# (ii)  c_a = c_k*k_a + c_a*rho - rho

# From Cap. acc. eq by matching coefficients:
# (iii) k_k = (1-delta) + (delta/I_Y)*alpha - (delta/I_Y)*C_Y*c_k
# (iv)  k_a = (delta/I_Y) - (delta/I_Y)*C_Y*c_a

# Notice equations (i) and (ii) involve c_k,c_a,k_k,k_a and (iii) and (iv) do as well.

# Let's rewrite them in standard linear form A*x = b, with unknowns x = [c_k, c_a, k_k, k_a].

# (i) c_k - c_k*k_k = 1 - alpha
#     c_k - k_k*c_k = 1 - alpha  -> c_k*(1 - k_k) + 0*c_a + 0*k_a - 0*k_a = 1 - alpha
# Actually we have k_k on the left as well. Let's rearrange to isolate terms:

# Equations:
# (i)   c_k = c_k*k_k + (1 - alpha)
#       c_k - c_k*k_k = 1 - alpha
#       c_k(1 - k_k) = 1 - alpha
# This links c_k and k_k directly. But we have 4 unknowns. Keep as is for now.

# (ii)  c_a = c_k*k_a + c_a*rho - rho
#       c_a - c_a*rho = c_k*k_a - rho
#       c_a(1 - rho) - c_k*k_a = -rho

# (iii) k_k = (1-delta) + (delta/I_Y)*alpha - (delta/I_Y)*C_Y*c_k
#       k_k + (delta/I_Y)*C_Y*c_k = (1-delta) + (delta/I_Y)*alpha
#       k_k + [(delta*C_Y)/I_Y]*c_k = (1-delta) + (delta*alpha)/I_Y

# (iv)  k_a = (delta/I_Y) - (delta/I_Y)*C_Y*c_a
#       k_a + ((delta*C_Y)/I_Y)*c_a = (delta/I_Y)
#
# We have 4 equations, but let's organize them as linear in [c_k, c_a, k_k, k_a].

# Let's choose an order for variables: x = [c_k, c_a, k_k, k_a]

# (i)   c_k(1 - k_k) = 1 - alpha
#      Not linear yet because it has c_k*k_k. We need to use all equations at once.
# Let's use a trick: we know (i) and (ii) involve c_k,c_a,k_k,k_a in nonlinear form. 
# Actually, we must treat them simultaneously. Let's rearrange all four to a linear form.

# We'll replace k_k and k_a from (iii) and (iv) into (i) and (ii). 
# Actually, it's easier to form a 4x4 linear system if we move all terms to one side:

# From (i):
# c_k = c_k*k_k + (1-alpha)
# c_k - c_k*k_k = 1 - alpha
# c_k(1 - k_k) = 1 - alpha
# c_k - k_k*c_k = 1 - alpha is not linear due to product c_k*k_k.
#
# We need another approach: We have 4 equations and 4 unknowns. Let's write them carefully:

# We'll use (iii) and (iv) to express k_k and k_a in terms of c_k and c_a, then substitute back into (i) and (ii).

# From (iii):
# k_k = (1-delta) + (delta*alpha/I_Y) - (delta*C_Y/I_Y)*c_k

# From (iv):
# k_a = (delta/I_Y) - (delta*C_Y/I_Y)*c_a

# Now substitute these k_k and k_a into (i) and (ii):

# (i): c_k = c_k*k_k + (1 - alpha)
# Substitute k_k:
# c_k = c_k[(1-delta) + (delta*alpha/I_Y) - (delta*C_Y/I_Y)*c_k] + (1 - alpha)

# Expand:
# c_k = c_k(1-delta) + c_k(delta*alpha/I_Y) - c_k((delta*C_Y/I_Y)*c_k) + (1 - alpha)

# This has a c_k^2 term, which suggests we made it too complicated. Actually, the linearized system should be linear. Let's re-derive the Euler equation condition more carefully in a linear format suitable for matrix solution:

# Re-derivation in linear form:

# Euler equation:
# hat(c_t) - E_t[hat(c_{t+1})] = (1 - alpha)*hat(k_t) - rho*hat(a_t)

# Using the assumed rules:
# hat(c_t) = c_k*hat(k_t) + c_a*hat(a_t)
# E_t[hat(c_{t+1})] = c_k*(k_k*hat(k_t)+k_a*hat(a_t)) + c_a*(rho*hat(a_t))

# LHS:
# c_k*hat(k_t) + c_a*hat(a_t) - [c_k*k_k*hat(k_t) + c_k*k_a*hat(a_t) + c_a*rho*hat(a_t)]
# = c_k*hat(k_t) - c_k*k_k*hat(k_t) + c_a*hat(a_t) - c_k*k_a*hat(a_t) - c_a*rho*hat(a_t)
# = (c_k - c_k*k_k)*hat(k_t) + (c_a - c_k*k_a - c_a*rho)*hat(a_t)

# RHS:
# (1 - alpha)*hat(k_t) - rho*hat(a_t)

# Match coefficients of hat(k_t) and hat(a_t):
# hat(k_t): c_k - c_k*k_k = 1 - alpha
# hat(a_t): c_a - c_k*k_a - c_a*rho = -rho

# Simplify hat(k_t) eq:
# c_k(1 - k_k) = 1 - alpha
# hat(a_t) eq:
# c_a(1 - rho) - c_k*k_a = -rho

# From capital accumulation:
# hat(k_{t+1}) = k_k*hat(k_t) + k_a*hat(a_t)
# Also:
# hat(k_{t+1}) = (1-delta)*hat(k_t) + (delta/I_Y)[hat(a_t)+alpha*hat(k_t)-C_Y(c_k*hat(k_t)+c_a*hat(a_t))]

# Group hat(k_t):
# = [(1-delta) + (delta*alpha/I_Y) - (delta*C_Y/I_Y)*c_k]*hat(k_t) + [(delta/I_Y) - (delta*C_Y/I_Y)*c_a]*hat(a_t)

# Match coefficients:
# k_k = (1-delta) + (delta*alpha/I_Y) - (delta*C_Y/I_Y)*c_k
# k_a = (delta/I_Y) - (delta*C_Y/I_Y)*c_a

# Now we have 4 equations linear in the four unknowns if we arrange them correctly:

# From the Euler (k_t):
# c_k - c_k*k_k = 1 - alpha
# c_k(1 - k_k) = 1 - alpha

# Substitute k_k from cap. accum. eq:
# k_k = A - B*c_k, where A = (1-delta) + (delta*alpha/I_Y), B = (delta*C_Y/I_Y)
# Then c_k(1 - (A - B*c_k)) = 1 - alpha
# c_k(1 - A + B*c_k) = 1 - alpha
# c_k - c_k*A + c_k*B*c_k = 1 - alpha
# c_k - A*c_k + B*c_k^2 = 1 - alpha
#
# We see a quadratic term again. This suggests we need a different approach:
#
# **A More Direct Linear System Approach**:
# Actually, the system is linear in terms of expectations. To solve directly for the policy functions, it's common practice to arrange the system in a "state-space" form and use a matrix approach. Let's do that:

# The standard RBC model linearization can be represented in a form:
# We have a 2x2 system that relates state variables (k_t, a_t) to controls (c_t) and next period states (k_{t+1}).

# A simpler approach: Let's set up the matrix system numerically and solve. We'll use the final linear approximations known from standard RBC expansions:

# Final linear system known from RBC linearization (see any RBC solve-by-hand example):
# We have two equations in four unknowns, but actually we have four equations total (two from Euler eq. matching k_t and a_t terms, and two from capital eq. matching k_t and a_t terms):

# Equations (from matching coefficients) are:
# 1) For hat(k_t) in Euler:    c_k - c_k*k_k = 1 - alpha
# 2) For hat(a_t) in Euler:    c_a - c_k*k_a - c_a*rho = -rho
# 3) For hat(k_t) in capital:  k_k = (1-delta)+(delta*alpha/I_Y)-(delta*C_Y/I_Y)*c_k
# 4) For hat(a_t) in capital:  k_a = (delta/I_Y)-(delta*C_Y/I_Y)*c_a

# We have 4 equations, but equations (3) and (4) give k_k and k_a directly in terms of c_k and c_a. 
# We can substitute k_k and k_a into (1) and (2) to get two equations in (c_k, c_a) only, then solve for them, then back out k_k, k_a.

A = (1 - delta) + (delta*alpha/I_Y)
B = (delta*C_Y/I_Y)
C = (delta/I_Y)

# From (3) and (4):
# k_k = A - B*c_k
# k_a = C - B*c_a

# Substitute into Euler eqs:

# (1) c_k - c_k*k_k = 1 - alpha
# c_k - c_k(A - B*c_k) = 1 - alpha
# c_k - c_k*A + c_k*B*c_k = 1 - alpha
# c_k(1 - A) + B*c_k^2 = 1 - alpha

# This is nonlinear. Let's also do (2):
# (2) c_a - c_k*k_a - c_a*rho = -rho
# c_a(1 - rho) - c_k(C - B*c_a) = -rho
# c_a(1 - rho) - c_k*C + c_k*B*c_a = -rho
# (1 - rho)*c_a + c_k*B*c_a - c_k*C = -rho

# We have a quadratic form due to c_k^2 and c_k*c_a terms. This suggests that typically one uses a matrix method known from the linear rational expectations literature. 

# **For simplicity**, let's use a numerical root finder to solve these equations. This is a valid approach if doing by hand is too tedious.

from sympy import symbols, Eq, solve

c_k_sym, c_a_sym = symbols('c_k c_a', real=True)

# Eq (1): c_k(1 - A) + B*c_k^2 = 1 - alpha
eq1 = Eq(c_k_sym*(1 - A) + B*c_k_sym**2, 1 - alpha)

# Eq (2): (1 - rho)*c_a + c_k*B*c_a - c_k*C = -rho
# Rearrange terms:
# B*c_k*c_a + (1 - rho)*c_a - c_k*C + rho = 0
# c_a(B*c_k + (1 - rho)) - c_k*C + rho = 0
eq2 = Eq((B*c_k_sym + (1 - rho))*c_a_sym - c_k_sym*C + rho, 0)

sol = solve([eq1, eq2],[c_k_sym, c_a_sym], dict=True)

print("Solutions for c_k and c_a:")
print(sol)

# Once we have c_k and c_a, we can find k_k and k_a:
# k_k = A - B*c_k
# k_a = C - B*c_a

if len(sol) > 0:
    c_k_val = sol[0][c_k_sym]
    c_a_val = sol[0][c_a_sym]

    k_k_val = A - B*c_k_val
    k_a_val = C - B*c_a_val

    print("c_k =", c_k_val)
    print("c_a =", c_a_val)
    print("k_k =", k_k_val)
    print("k_a =", k_a_val)
else:
    print("No solution found.")
