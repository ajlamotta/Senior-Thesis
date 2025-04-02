import numpy as np
import pandas as pd
from scipy.linalg import eigh

# Simulate largest eigenval of n x n GOE matrix
def simulate_goe_largest(n):
    X = np.random.randn(n, n)
    A = (X + X.T) / np.sqrt(2 * n)
    eigvals = eigh(A, eigvals_only=True)
    return np.max(eigvals)

# Compute empirical cdf
def empirical_cdf(data, grid):
    data_sorted = np.sort(data)
    cdf = np.searchsorted(data_sorted, grid, side='right') / float(len(data))
    return cdf

ns = [10, 100, 1000]
num_samples = 5000
results = {}
for n in ns:
    samples = []
    for _ in range(num_samples):
        lam_max = simulate_goe_largest(n)
        scaled = n**(2/3) * (lam_max - 2)
        samples.append(scaled)
    results[n] = np.array(samples)

# Determine a common grid covering the range of simulated s-values
s_min_empirical = min(np.min(results[n]) for n in ns)
s_max_empirical = max(np.max(results[n]) for n in ns)
x_grid = np.linspace(s_min_empirical, s_max_empirical, 1000)

# Compute the empirical CDFs on the common grid
ecdf_10   = empirical_cdf(results[10], x_grid)
ecdf_100  = empirical_cdf(results[100], x_grid)
ecdf_1000 = empirical_cdf(results[1000], x_grid)

# Create a DataFrame and write to CSV
df = pd.DataFrame({
    'x': x_grid,
    'ecdf_n10': ecdf_10,
    'ecdf_n100': ecdf_100,
    'ecdf_n1000': ecdf_1000
})
df.to_csv("empirical_distribution.csv", index=False)

# Print when done
print("CSV file 'empirical_distribution.csv' generated successfully.")