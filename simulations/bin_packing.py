import numpy as np

def binpack_minimum(weights, bin_capacity=1.0):
    weights_sorted = sorted(weights, reverse=True)
    n = len(weights_sorted)
    best_solution = [n] # Worst case: 1 item per bin

    def backtrack(i, bin_sums, used_bins):
        if i == n:  # All items placed
            best_solution[0] = min(best_solution[0], used_bins)
            return
        if used_bins >= best_solution[0]:
            # Already worse than known solution, prune
            return

        # Try putting weights_sorted[i] in existing bins
        for b in range(used_bins):
            if bin_sums[b] + weights_sorted[i] <= bin_capacity:
                bin_sums[b] += weights_sorted[i]
                backtrack(i+1, bin_sums, used_bins)
                bin_sums[b] -= weights_sorted[i]

        # Or open a new bin
        if used_bins < best_solution[0]:
            bin_sums.append(weights_sorted[i])
            backtrack(i+1, bin_sums, used_bins+1)
            bin_sums.pop()

    backtrack(0, [], 0)
    return best_solution[0]

def simulate_binpacking_variance(n_max=20, num_trials=100, csv_filename="binpacking_var.csv"):
    rng = np.random.default_rng()
    results = []

    for n in range(1, n_max + 1):
        B_values = []
        for _ in range(num_trials):
            # Sample weights: W_i ~ Beta(1, i)
            weights = [rng.beta(1, i) for i in range(1, n + 1)]
            B = binpack_minimum(weights)
            B_values.append(B)

        var_B = np.var(B_values, ddof=1) # Sample variance
        results.append((n, var_B))
        print(f"n={n}, Estimated Var(B)={var_B:.4f}")

    # Save results to CSV
    with open(csv_filename, 'w') as f:
        f.write("n,var_b\n") # Header
        for (n, var_B) in results:
            f.write(f"{n},{var_B}\n")

    return results

# Simulate
N_MAX = 20
NUM_TRIALS = 100
CSV_FILENAME = "binpacking_var.csv"
simulate_binpacking_variance(n_max=N_MAX, num_trials=NUM_TRIALS, csv_filename=CSV_FILENAME)
