use nalgebra::{DMatrix, DVector};
use rand::prelude::*;
use rand_distr::Normal;
use std::error::Error;
use std::fs::File;
use std::io::Write;

/// Returns a Haar-distributed orthogonal matrix of size n×n.
fn sample_haar_orthogonal(n: usize) -> DMatrix<f64> {
    let mut rng = rand::thread_rng();
    if n == 1 {
        let sign = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
        DMatrix::from_element(1, 1, sign)
    } else {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let data: Vec<f64> = (0..n * n).map(|_| normal.sample(&mut rng)).collect();
        let x = DMatrix::from_vec(n, n, data);
        let qr = x.qr();
        let mut q = qr.q();
        let r = qr.r();
        // Ensure positive diagonal entries of R.
        for i in 0..n {
            if r[(i, i)] < 0.0 {
                for j in 0..n {
                    q[(j, i)] = -q[(j, i)];
                }
            }
        }
        q
    }
}

/// Runs a Metropolis algorithm to sample σ ∈ {–1,1}ⁿ according to 
/// p(σ) ∝ exp(β σᵀ A σ). Returns a vector of σ samples.
fn metropolis_sample(
    A: &DMatrix<f64>,
    beta: f64,
    n: usize,
    n_steps: usize,
    burn_in: usize,
) -> Vec<DVector<i32>> {
    let mut rng = rand::thread_rng();
    let sigma: Vec<i32> = (0..n)
        .map(|_| if rng.gen_bool(0.5) { 1 } else { -1 })
        .collect();
    let mut sigma = DVector::from_vec(sigma);
    let mut samples = Vec::new();
    for step in 0..n_steps {
        let i = rng.gen_range(0..n);
        let mut sigma_new = sigma.clone();
        sigma_new[i] = -sigma_new[i];
        let sigma_f = sigma.map(|x| x as f64);
        let sigma_new_f = sigma_new.map(|x| x as f64);
        let energy_old = (sigma_f.transpose() * A * &sigma_f)[(0, 0)];
        let energy_new = (sigma_new_f.transpose() * A * &sigma_new_f)[(0, 0)];
        let delta = beta * (energy_new - energy_old);
        if rng.gen::<f64>() < delta.exp() {
            sigma = sigma_new;
        }
        if step >= burn_in && (step - burn_in) % 10 == 0 {
            samples.push(sigma.clone());
        }
    }
    samples
}

/// Computes the squared gradient:
/// ||∇F||² = 4 * Σ_{i<j} (σ_j (Aσ)_i − σ_i (Aσ)_j)²,
/// for one σ sample.
fn compute_gradient_squared(A: &DMatrix<f64>, sigma: &DVector<i32>) -> f64 {
    let n = sigma.len();
    let sigma_f = sigma.map(|x| x as f64);
    let Asigma = A * sigma_f;
    let mut grad_sq = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let diff = (sigma[j] as f64) * Asigma[i] - (sigma[i] as f64) * Asigma[j];
            grad_sq += diff * diff;
        }
    }
    4.0 * grad_sq
}

/// For a given n and β, tests several different diagonal matrices (with entries sampled i.i.d. from Unif(-1,1)).
 /// For each diagonal matrix, it runs multiple simulations and computes the average squared gradient.
 /// It then returns the maximum average squared gradient among the different diagonal matrices.
fn simulate_max_grad_sq(
    n: usize,
    beta: f64,
    n_diag: usize,
    n_samples: usize,
    n_steps: usize,
    burn_in: usize,
) -> f64 {
    let mut max_avg = std::f64::MIN;
    let mut rng = rand::thread_rng();
    for _ in 0..n_diag {
        // Sample a diagonal matrix with i.i.d. entries from Uniform(-1,1).
        let lambda_vals: Vec<f64> = (0..n)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        let Lambda = DMatrix::from_diagonal(&DVector::from_vec(lambda_vals));

        let mut grad_sq_vals = Vec::new();
        for _ in 0..n_samples {
            let O = sample_haar_orthogonal(n);
            let A = O.transpose() * &Lambda * O;
            let sigma_samples = metropolis_sample(&A, beta, n, n_steps, burn_in);
            let total: f64 = sigma_samples
                .iter()
                .map(|sigma| compute_gradient_squared(&A, sigma))
                .sum();
            grad_sq_vals.push(total / (sigma_samples.len() as f64));
        }
        let avg = grad_sq_vals.iter().sum::<f64>() / (grad_sq_vals.len() as f64);
        if avg > max_avg {
            max_avg = avg;
        }
    }
    max_avg
}

fn main() -> Result<(), Box<dyn Error>> {
    let beta = 1.0;
    let max_n = 40;
    let n_diag = 10;      // Number of different diagonal matrices to test per N.
    let n_samples = 30;   // Number of Metropolis simulations per diagonal matrix.
    let n_steps = 10000;
    let burn_in = 1000;
    let mut simulation_data = Vec::new();
    
    // Run simulations for n = 1,2,...,max_n.
    for n in 1..=max_n {
        println!("Simulating for n = {}", n);
        let max_avg = simulate_max_grad_sq(n, beta, n_diag, n_samples, n_steps, burn_in);
        simulation_data.push((n, max_avg));
    }
    
    // Write CSV file with header "N,value".
    let filename = "grad_simulation.csv";
    let mut file = File::create(filename)?;
    writeln!(file, "N,value")?;
    for (n, avg) in simulation_data {
        writeln!(file, "{},{}", n, avg)?;
    }
    
    println!("CSV data written to {}", filename);
    Ok(())
}