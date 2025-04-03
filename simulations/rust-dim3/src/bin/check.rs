use nalgebra::{DMatrix, DVector};
use nalgebra::linalg::QR;
use rand::prelude::*;
use rand_distr::{Uniform, StandardNormal};
use std::error::Error;

/// Generate a Haar-distributed orthogonal matrix of size n x n.
fn generate_haar(n: usize) -> DMatrix<f64> {
    let mut rng = rand::thread_rng();
    let data: Vec<f64> = (0..n*n).map(|_| rng.sample(StandardNormal)).collect();
    let m = DMatrix::from_vec(n, n, data);
    let qr = QR::new(m);
    let mut q = qr.q();
    let r = qr.r();
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

/// Enumerate all spin states in {±1}^n.
fn generate_sigma_states(n: usize) -> Vec<DVector<f64>> {
    let num_states = 1 << n;
    let mut states = Vec::with_capacity(num_states);
    for state in 0..num_states {
        let mut vec = Vec::with_capacity(n);
        for i in 0..n {
            let bit = (state >> i) & 1;
            vec.push(if bit == 1 { 1.0 } else { -1.0 });
        }
        states.push(DVector::from_vec(vec));
    }
    states
}

/// Compute the free energy:
/// F(O) = ln(∑_{σ∈{±1}^n} exp((OᵀΛO σ) ⋅ σ)).
fn free_energy(o: &DMatrix<f64>, lambda: &DMatrix<f64>, sigma_states: &Vec<DVector<f64>>) -> f64 {
    let a = o.transpose() * lambda * o;
    let mut z = 0.0;
    for sigma in sigma_states {
        let h_val = (sigma.transpose() * &a * sigma)[(0, 0)];
        z += h_val.exp();
    }
    z.ln()
}

/// Compute the free energy when O is perturbed in one edge direction.
fn free_energy_edge(o0: &DMatrix<f64>, edge: (usize, usize), epsilon: f64, lambda: &DMatrix<f64>, sigma_states: &Vec<DVector<f64>>) -> f64 {
    let n = o0.nrows();
    let r_mat = rotation_matrix(n, edge.0, edge.1, epsilon);
    let o_eps = o0 * r_mat;
    free_energy(&o_eps, lambda, sigma_states)
}

/// Compute the second derivative in a given edge direction using central finite differences:
/// ∂²F/∂x_{ij}² ≈ [F(ε) - 2F(0) + F(−ε)] / ε².
fn second_derivative(o0: &DMatrix<f64>, edge: (usize, usize), sigma_states: &Vec<DVector<f64>>, lambda: &DMatrix<f64>, epsilon: f64) -> f64 {
    let f0 = free_energy_edge(o0, edge, 0.0, lambda, sigma_states);
    let f_plus = free_energy_edge(o0, edge, epsilon, lambda, sigma_states);
    let f_minus = free_energy_edge(o0, edge, -epsilon, lambda, sigma_states);
    (f_plus - 2.0 * f0 + f_minus) / (epsilon * epsilon)
}

fn main() -> Result<(), Box<dyn Error>> {
    // Parameters:
    let n: usize = 6;           // Dimension (n=6 so disjoint edges exist)
    let num_samples = 200;      // Number of Haar samples
    let epsilon = 1e-4;         // Finite-difference step size

    // Choose two disjoint edges: α = (0,1) and β = (2,3)
    let alpha = (0, 1);
    let beta = (2, 3);

    // Precompute all spin states in {±1}^n.
    let sigma_states = generate_sigma_states(n);

    // Use a fixed diagonal Λ for reproducibility.
    let lambda_diag: Vec<f64> = (1..=n).map(|x| x as f64).collect();
    let lambda = DMatrix::from_diagonal(&DVector::from_vec(lambda_diag));

    // Variables to accumulate sums.
    let mut sum_product = 0.0;
    let mut sum_alpha = 0.0;
    let mut sum_beta = 0.0;
    let mut samples = 0;
    
    for _ in 0..num_samples {
        let o0 = generate_haar(n);
        let d_alpha = second_derivative(&o0, alpha, &sigma_states, &lambda, epsilon);
        let d_beta  = second_derivative(&o0, beta, &sigma_states, &lambda, epsilon);
        sum_product += d_alpha * d_beta;
        sum_alpha += d_alpha;
        sum_beta  += d_beta;
        samples += 1;
    }

    let avg_product = sum_product / (samples as f64);
    let avg_alpha = sum_alpha / (samples as f64);
    let avg_beta  = sum_beta / (samples as f64);
    let covariance = avg_product - avg_alpha * avg_beta;

    println!("For disjoint edges α = {:?} and β = {:?}", alpha, beta);
    println!("E[∂²F/∂x_{:?}² · ∂²F/∂x_{:?}²] = {:.4e}", alpha, beta, avg_product);
    println!("E[∂²F/∂x_{:?}²] = {:.4e}", alpha, avg_alpha);
    println!("E[∂²F/∂x_{:?}²] = {:.4e}", beta, avg_beta);
    println!("Covariance = E[product] - (E[α]*E[β]) = {:.4e}", covariance);

    Ok(())
}