import math

# Given alpha values
alpha1 = 1.0204
alpha2 = 0.1011
alpha3 = 0
alpha4 = -0.0285
alpha5 = 1
alpha6 = 0

# Steady-state parameter
rho = 0.95  # Assuming rho is given or needs to be defined

discriminant = (alpha1 - (alpha2 * alpha4 / alpha5) + (1 / alpha5))**2 - 4 * (alpha1 / alpha5)
if discriminant < 0:
    print("Error: Negative discriminant. No real solutions for η_kk.")
else:
    root1 = 0.5 * ((alpha1 - (alpha2 * alpha4 / alpha5) + (1 / alpha5)) + math.sqrt(discriminant))
    root2 = 0.5 * ((alpha1 - (alpha2 * alpha4 / alpha5) + (1 / alpha5)) - math.sqrt(discriminant))
    
    # Choose the positive root less than 1
    eta_kk = root1 if 0 < root1 < 1 else root2 if 0 < root2 < 1 else None
    

# Calculating η_λk
eta_lambda_k = (eta_kk - alpha1) / alpha2

# Calculating η_λa
eta_lambda_a = (
    (alpha3 * alpha4 + alpha3 * alpha5 * eta_lambda_k + alpha6 * rho)
    / (1 - alpha2 * alpha4 - alpha2 * alpha5 * eta_lambda_k - alpha5 * rho)
)

# Calculating η_ka
eta_ka = alpha2 * eta_lambda_a + alpha3

# Display results
print(f"η_kk = {eta_kk:.4f}")
print(f"η_λk = {eta_lambda_k:.4f}")
print(f"η_λa = {eta_lambda_a:.4f}")
print(f"η_ka = {eta_ka:.4f}")

  


   



