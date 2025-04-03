use nalgebra::{DMatrix, DVector};
use nalgebra::linalg::QR;
use rand::prelude::*;
use rand_distr::{Uniform, StandardNormal};
use std::error::Error;
use std::f64::consts::PI;
use csv;

// ----- Tweakable parameters -----
const MAX_N: usize = 20;           // Maximum N to simulate.
const R_ORDER: usize = 1;          // Order r (e.g., 1 for Δ¹F, 2 for Δ²F, etc.).
const NUM_EDGE_SAMPLES: usize = 20; // Number of random edge combinations to sample.
const EPSILON: f64 = 1e-2;         // Finite difference step size (increased from 1e-4).
// We'll also reduce the number of samples:
const NUM_HAAR_SAMPLES: usize = 10;    // Haar samples per Λ trial.
const NUM_LAMBDA_SAMPLES: usize = 50;    // Number of independent Λ samples per N.
const N_STEPS: usize = 5000;             // Metropolis steps for free energy estimation.
const BURN_IN: usize = 500;              // Burn-in steps.
// ---------------------------------

/// Generate a Haar-distributed orthogonal matrix of size n x n.
fn generate_haar(n: usize) -> DMatrix<f64> {
    let mut rng = rand::thread_rng();
    let data: Vec<f64> = (0..n * n).map(|_| rng.sample(StandardNormal)).collect();
    let m = DMatrix::from_vec(n, n, data);
    let qr = QR::new(m);
    let mut q = qr.q();
    let r = qr.r();
    // Adjust sign ambiguity.
    for i in 0..n {
        if r[(i, i)] < 0.0 {
            for j in 0..n {
                q[(j, i)] = -q[(j, i)];
            }
        }
    }
    q
}

/// Returns an n x n rotation matrix in the (i, j) plane by angle epsilon.
fn rotation_matrix(n: usize, i: usize, j: usize, epsilon: f64) -> DMatrix<f64> {
    let mut r = DMatrix::<f64>::identity(n, n);
    let c = epsilon.cos();
    let s = epsilon.sin();
    r[(i, i)] = c;
    r[(j, j)] = c;
    r[(i, j)] = s;
    r[(j, i)] = -s;
    r
}

/// Metropolis–Hastings sampling for σ ∈ {±1}ⁿ with target density
/// p(σ) ∝ exp(σᵀ A σ), where A = Oᵀ Λ O.
fn metropolis_sample(A: &DMatrix<f64>, n: usize, n_steps: usize, burn_in: usize) -> Vec<DVector<i32>> {
    let mut rng = rand::thread_rng();
    let sigma_init: Vec<i32> = (0..n)
        .map(|_| if rng.gen_bool(0.5) { 1 } else { -1 })
        .collect();
    let mut sigma = DVector::from_vec(sigma_init);
    let mut samples = Vec::new();
    for step in 0..n_steps {
        let i = rng.gen_range(0..n);
        let mut sigma_new = sigma.clone();
        sigma_new[i] = -sigma_new[i];
        let sigma_f = sigma.map(|x| x as f64);
        let sigma_new_f = sigma_new.map(|x| x as f64);
        let energy_old = (sigma_f.transpose() * A * &sigma_f)[(0, 0)];
        let energy_new = (sigma_new_f.transpose() * A * &sigma_new_f)[(0, 0)];
        if rng.gen::<f64>() < (energy_new - energy_old).exp() {
            sigma = sigma_new;
        }
        if step >= burn_in {
            samples.push(sigma.clone());
        }
    }
    samples
}

/// Approximate the free energy F(O) = ln(∑ₛ exp(σᵀ A σ))
/// where A = Oᵀ Λ O. By sampling σ from p(σ) ∝ exp(σᵀ A σ),
/// one has E[exp(-σᵀ A σ)] = 2ⁿ / Z, so that
/// Z = 2ⁿ / (average(exp(-energy))) and F = n ln 2 - ln(average(exp(-energy))).
fn free_energy_mh(
    o: &DMatrix<f64>,
    lambda: &DMatrix<f64>,
    n: usize,
    n_steps: usize,
    burn_in: usize,
) -> f64 {
    let A = o.transpose() * lambda * o;
    let samples = metropolis_sample(&A, n, n_steps, burn_in);
    let avg_exp_neg: f64 = samples.iter().map(|sigma| {
        let sigma_f = sigma.map(|x| x as f64);
        let energy = (sigma_f.transpose() * &A * &sigma_f)[(0, 0)];
        (-energy).exp()
    }).sum::<f64>() / (samples.len() as f64);
    (n as f64) * (2.0_f64.ln()) - avg_exp_neg.ln()
}

/// Compute F(O) for a perturbation along a given list of edges with perturbation factors `deltas`.
/// The perturbed matrix is O(δ) = O₀ ∏ₖ R_{iₖ,jₖ}(δₖ * epsilon).
fn free_energy_multi(
    o0: &DMatrix<f64>,
    edges: &[(usize, usize)],
    deltas: &[f64],
    epsilon: f64,
    lambda: &DMatrix<f64>,
    n: usize,
    n_steps: usize,
    burn_in: usize,
) -> f64 {
    let n_dim = o0.nrows();
    let mut o_perturbed = o0.clone();
    for (&(i, j), &delta) in edges.iter().zip(deltas.iter()) {
        let r_mat = rotation_matrix(n_dim, i, j, delta * epsilon);
        o_perturbed = o_perturbed * r_mat;
    }
    free_energy_mh(&o_perturbed, lambda, n, n_steps, burn_in)
}

/// Compute the finite-difference approximation to the r-fold second derivative
/// (via central differences) for the edges provided.
fn multi_second_derivative(
    o0: &DMatrix<f64>,
    edges: &[(usize, usize)],
    lambda: &DMatrix<f64>,
    epsilon: f64,
    n: usize,
    n_steps: usize,
    burn_in: usize,
) -> f64 {
    let r = edges.len();
    let total_grid = 3_usize.pow(r as u32);
    let mut sum = 0.0;
    for idx in 0..total_grid {
        let mut current = idx;
        let mut coeff = 1.0;
        let mut deltas = Vec::with_capacity(r);
        for _ in 0..r {
            let rem = current % 3;
            current /= 3;
            let delta = match rem {
                0 => -1.0,
                1 => 0.0,
                2 => 1.0,
                _ => unreachable!(),
            };
            let c = match rem {
                0 => 1.0,
                1 => -2.0,
                2 => 1.0,
                _ => unreachable!(),
            };
            coeff *= c;
            deltas.push(delta);
        }
        let f_val = free_energy_multi(o0, edges, &deltas, epsilon, lambda, n, n_steps, burn_in);
        sum += coeff * f_val;
    }
    sum / epsilon.powi((2 * r) as i32)
}

/// Returns all combinations of r elements from the slice `items`.
fn combinations(items: &[(usize, usize)], r: usize) -> Vec<Vec<(usize, usize)>> {
    let mut result = Vec::new();
    let mut combo = Vec::new();
    fn combine(
        start: usize,
        items: &[(usize, usize)],
        r: usize,
        combo: &mut Vec<(usize, usize)>,
        result: &mut Vec<Vec<(usize, usize)>>,
    ) {
        if r == 0 {
            result.push(combo.clone());
            return;
        }
        for i in start..items.len() {
            combo.push(items[i]);
            combine(i + 1, items, r - 1, combo, result);
            combo.pop();
        }
    }
    combine(0, items, r, &mut combo, &mut result);
    result
}

/// For a given n and Λ, compute the sample average (over Haar samples) of (Δ^r F)²,
/// where Δ^r F is the sum over a collection of r-fold second derivative estimates.
/// Instead of summing over all edge combinations (which is expensive),
/// we randomly sample a subset of combinations and scale the result appropriately.
fn simulate_for_lambda(
    n: usize,
    lambda: &DMatrix<f64>,
    num_haar_samples: usize,
    epsilon: f64,
    r_order: usize,
    n_steps: usize,
    burn_in: usize,
) -> f64 {
    // Build list of all edges (i,j) with i < j.
    let mut edges = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            edges.push((i, j));
        }
    }
    if edges.is_empty() {
        return 0.0;
    }
    let full_combs = combinations(&edges, r_order);
    let total_combs = full_combs.len();
    
    // Randomly sample a subset if needed.
    let (sampled_combs, scale) = if total_combs > NUM_EDGE_SAMPLES {
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..total_combs).collect();
        indices.shuffle(&mut rng);
        let selected: Vec<Vec<(usize, usize)>> = indices.iter()
            .take(NUM_EDGE_SAMPLES)
            .map(|&i| full_combs[i].clone())
            .collect();
        (selected, total_combs as f64 / NUM_EDGE_SAMPLES as f64)
    } else {
        (full_combs, 1.0)
    };

    let mut samples = Vec::with_capacity(num_haar_samples);
    for _ in 0..num_haar_samples {
        let o0 = generate_haar(n);
        let mut derivative_sum = 0.0;
        for comb in &sampled_combs {
            let deriv = multi_second_derivative(&o0, comb, lambda, epsilon, n, n_steps, burn_in);
            derivative_sum += deriv;
        }
        derivative_sum *= scale;
        samples.push(derivative_sum);
    }
    samples.iter().map(|&x| x * x).sum::<f64>() / (samples.len() as f64)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut results = Vec::new();

    // Loop over n from 1 to MAX_N.
    for n in 1..=MAX_N {
        println!("Processing N = {}", n);
        let mut max_avg = std::f64::MIN;
        for _ in 0..NUM_LAMBDA_SAMPLES {
            // Create a diagonal Λ with i.i.d. entries from Uniform[-1,1].
            let mut rng = rand::thread_rng();
            let uniform = Uniform::new_inclusive(-1.0, 1.0);
            let diag: Vec<f64> = (0..n).map(|_| rng.sample(uniform)).collect();
            let lambda = DMatrix::from_diagonal(&DVector::from_vec(diag));
            let avg_val = simulate_for_lambda(n, &lambda, NUM_HAAR_SAMPLES, EPSILON, R_ORDER, N_STEPS, BURN_IN);
            if avg_val > max_avg {
                max_avg = avg_val;
            }
        }
        println!("  N = {}: Maximum sample average E[(Δ^{}F)²] = {:.4e}", n, R_ORDER, max_avg);
        results.push((n as f64, max_avg));
    }

    // Write CSV file with header "N,value".
    let filename = format!("{}-laplacian.csv", R_ORDER);
    let mut wtr = csv::Writer::from_path(&filename)?;
    wtr.write_record(&["N", "value"])?;
    for (n, val) in &results {
        wtr.write_record(&[n.to_string(), val.to_string()])?;
    }
    wtr.flush()?;
    println!("CSV data saved to {}", filename);

    Ok(())
}
