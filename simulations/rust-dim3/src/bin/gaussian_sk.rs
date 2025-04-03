use rand::prelude::*;
use rand_distr::StandardNormal;
use rayon::prelude::*;
use std::f64;
use std::fs::File;
use std::io::{BufWriter, Write};

/// Simulate one sample of F_{N, beta}.
///
/// For a given system size \(N\) and inverse temperature \(\beta\),
/// we define
/// \[
/// F_{N,\beta} = \frac{1}{\beta} \ln\left( \sum_{\sigma\in\{\pm1\}^N} \exp\left(\beta H_N(\sigma)\right) \right),
/// \]
/// where
/// \[
/// H_N(\sigma) = \frac{1}{\sqrt{N}} \sum_{i<j} g_{ij} \sigma_i\sigma_j,
/// \]
/// and \(g_{ij}\sim\mathcal{N}(0,1)\).
fn simulate_sample(n: usize, beta: f64, pairs: &[(usize, usize)]) -> f64 {
    let num_configs = 1 << n; // Total number of configurations
    let sqrt_n = (n as f64).sqrt();
    let mut rng = rand::thread_rng();

    // Generate Gaussian couplings for each pair (i, j)
    let g: Vec<f64> = pairs.iter().map(|_| rng.sample(StandardNormal)).collect();

    // First pass: determine the maximum value for numerical stability
    let mut max_val = f64::NEG_INFINITY;
    for sigma in 0..num_configs {
        let mut energy = 0.0;
        for (k, &(i, j)) in pairs.iter().enumerate() {
            let spin_i = if (sigma >> i) & 1 == 1 { 1.0 } else { -1.0 };
            let spin_j = if (sigma >> j) & 1 == 1 { 1.0 } else { -1.0 };
            energy += spin_i * spin_j * g[k];
        }
        let H = energy / sqrt_n;
        let val = beta * H;
        if val > max_val {
            max_val = val;
        }
    }

    // Second pass: compute the shifted sum for the log-sum-exp trick.
    let mut sum_config = 0.0;
    for sigma in 0..num_configs {
        let mut energy = 0.0;
        for (k, &(i, j)) in pairs.iter().enumerate() {
            let spin_i = if (sigma >> i) & 1 == 1 { 1.0 } else { -1.0 };
            let spin_j = if (sigma >> j) & 1 == 1 { 1.0 } else { -1.0 };
            energy += spin_i * spin_j * g[k];
        }
        let H = energy / sqrt_n;
        let val = beta * H;
        sum_config += (val - max_val).exp();
    }
    let logz = max_val + sum_config.ln();
    logz / beta
}

/// For a given system size \(N\) and \(\beta\), run many independent simulations
/// to compute the sample variance of \(F_{N,\beta}\) and return the value
/// \(\Var(F_{N,\beta})/N\).
fn simulate_variance(n: usize, beta: f64, num_samples: usize) -> f64 {
    // Precompute all index pairs (i, j) with i < j.
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .collect();

    // Run num_samples independent simulations in parallel.
    let samples: Vec<f64> = (0..num_samples)
        .into_par_iter()
        .map(|_| simulate_sample(n, beta, &pairs))
        .collect();

    // Compute the sample mean and variance.
    let mean = samples.iter().sum::<f64>() / (num_samples as f64);
    let var = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (num_samples as f64);
    var / (n as f64)
}

fn main() -> std::io::Result<()> {
    // Set the maximum system size.
    let max_n: usize = 20;
    let ns: Vec<usize> = (1..=max_n).collect();
    let betas = vec![0.1, 1.0, 10.0];
    let num_samples = 100;

    // Open (or create) the CSV file for writing.
    let file = File::create("simulation_data.csv")?;
    let mut writer = BufWriter::new(file);

    // Write the header: the columns are beta, N, and variance_over_N.
    writeln!(writer, "beta,N,variance_over_N")?;

    // For each beta and each N, compute the variance and write the result.
    for beta in betas {
        for &n in &ns {
            println!("Simulating for beta = {}, N = {}", beta, n);
            let var_over_n = simulate_variance(n, beta, num_samples);
            writeln!(writer, "{},{},{}", beta, n, var_over_n)?;
        }
    }

    println!("Simulation data saved to simulation_data.csv");
    Ok(())
}