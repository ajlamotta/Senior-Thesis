import numpy as np

# Set the matrix dimension
n = 1000

# Generate an n x n matrix with iid normal entries
A = np.random.randn(n, n)

# Form a symmetric matrix and scale it so that the eigenvalue density converges to the semicircle law
H = (A + A.T) / np.sqrt(2*n)

# Compute the eigenvalues (which are real)
eigenvalues = np.linalg.eigvalsh(H)

# Save the eigenvalues to a CSV file (one column)
np.savetxt("wigner.csv", eigenvalues, delimiter=",")
