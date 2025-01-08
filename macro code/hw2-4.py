import numpy as np

# Transition matrix
P = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.1, 0.1, 0.8]
])

# Solve for the invariant distribution
eigvals, eigvecs = np.linalg.eig(P.T)
stationary = eigvecs[:, np.isclose(eigvals, 1)]  # Extract eigenvector for eigenvalue 1
stationary = stationary / np.sum(stationary)    # Normalize to sum to 1

print("Invariant Distribution:", stationary.real.flatten())