use nalgebra::{DMatrix, DVector};
use nalgebra::linalg::QR;
use plotters::prelude::*;
use rand::prelude::*;
use rand_distr::{Uniform, StandardNormal};
use std::error::Error;
use std::f64::consts::PI;

// ----- Tweakable parameters -----
const MAX_N: usize = 10;      // Maximum N to simulate.
const R_ORDER: usize = 2;     // Order r (e.g., 1 for Δ¹F, 2 for Δ²F, etc.)
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

/// Generate all states in {±1}^n.
fn generate_sigma_states(n: usize) -> Vec<DVector<f64>> {
    let num_states = 1 << n; // 2^n states.
    let mut states = Vec::with_capacity(num_states);
    for state in 0..num_states {
        let mut vec = Vec::with_capacity(n);
        for i in 0..n {
            vec.push(if (state >> i) & 1 == 1 { 1.0 } else { -1.0 });
        }
        states.push(DVector::from_vec(vec));
    }
    states
}

/// Compute the free energy:
/// F(O) = ln(sum_{σ in {±1}^n} exp(σ^T A σ))
/// where A = O^T * Λ * O. Here we take β = 1.
fn free_energy(o: &DMatrix<f64>, lambda: &DMatrix<f64>, sigma_states: &Vec<DVector<f64>>) -> f64 {
    let a = o.transpose() * lambda * o;
    let mut z = 0.0;
    for sigma in sigma_states {
        let h_val = (sigma.transpose() * &a * sigma)[(0, 0)];
        z += h_val.exp();
    }
    z.ln()
}

/// Compute F(O) for a perturbation along a given list of edges with perturbation factors provided in `deltas`.
/// The perturbation is applied in sequence: 
/// O(δ) = O₀ * ∏_{k=1}^r R_{i_k,j_k}(δ_k * epsilon).
fn free_energy_multi(
    o0: &DMatrix<f64>,
    edges: &[(usize, usize)],
    deltas: &[f64],
    epsilon: f64,
    lambda: &DMatrix<f64>,
    sigma_states: &Vec<DVector<f64>>,
) -> f64 {
    let n = o0.nrows();
    let mut o_perturbed = o0.clone();
    for (&(i, j), &delta) in edges.iter().zip(deltas.iter()) {
        let r_mat = rotation_matrix(n, i, j, delta * epsilon);
        o_perturbed = o_perturbed * r_mat;
    }
    free_energy(&o_perturbed, lambda, sigma_states)
}

/// For a given set of edges (of length r), compute the finite difference approximation to
/// ∂²_{α₁} ... ∂²_{α_r} F(0) using a multidimensional central difference.
fn multi_second_derivative(
    o0: &DMatrix<f64>,
    edges: &[(usize, usize)],
    sigma_states: &Vec<DVector<f64>>,
    lambda: &DMatrix<f64>,
    epsilon: f64,
) -> f64 {
    let r = edges.len();
    // Iterate over all combinations of deltas in {-1,0,1}^r.
    let total = 3_usize.pow(r as u32);
    let mut sum = 0.0;
    for idx in 0..total {
        let mut current = idx;
        let mut coeff = 1.0;
        let mut deltas = Vec::with_capacity(r);
        // For each coordinate, decode a value in {0,1,2} mapping to {-1,0,1}.
        for _ in 0..r {
            let rem = current % 3;
            current /= 3;
            let delta = match rem {
                0 => -1.0,
                1 => 0.0,
                2 => 1.0,
                _ => unreachable!(),
            };
            // Coefficient for second derivative: c(-1)=1, c(0)=-2, c(1)=1.
            let c = match rem {
                0 => 1.0,
                1 => -2.0,
                2 => 1.0,
                _ => unreachable!(),
            };
            coeff *= c;
            deltas.push(delta);
        }
        let f_val = free_energy_multi(o0, edges, &deltas, epsilon, lambda, sigma_states);
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

/// For a given n and Λ, compute the sample average (over Haar samples) of (Δ^rF)^2,
/// where Δ^rF is the sum over all r-fold second derivatives (over all combinations of r distinct edges).
fn simulate_for_lambda(
    n: usize,
    lambda: &DMatrix<f64>,
    num_haar_samples: usize,
    epsilon: f64,
    r_order: usize,
) -> f64 {
    let sigma_states = generate_sigma_states(n);
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
    // Generate all combinations of edges of length r_order.
    let combs = combinations(&edges, r_order);
    let mut samples = Vec::with_capacity(num_haar_samples);
    for _ in 0..num_haar_samples {
        let o0 = generate_haar(n);
        let mut derivative_sum = 0.0;
        for comb in &combs {
            let deriv = multi_second_derivative(&o0, comb, &sigma_states, lambda, epsilon);
            derivative_sum += deriv;
        }
        samples.push(derivative_sum);
    }
    samples.iter().map(|&x| x * x).sum::<f64>() / (samples.len() as f64)
}

fn main() -> Result<(), Box<dyn Error>> {
    let num_haar_samples = 20;     // Haar samples per Λ trial.
    let num_lambda_samples = 100;  // Number of independent Λ samples per N.
    let epsilon = 1e-4;            // Finite difference step size.
    
    let mut results = Vec::new();

    // Loop over n from 1 to MAX_N.
    for n in 1..=MAX_N {
        println!("Processing N = {}", n);
        let mut max_avg = std::f64::MIN;
        for _ in 0..num_lambda_samples {
            // Create a diagonal Λ with i.i.d. entries from Uniform[-1,1].
            let mut rng = rand::thread_rng();
            let uniform = Uniform::new_inclusive(-1.0, 1.0);
            let diag: Vec<f64> = (0..n).map(|_| rng.sample(uniform)).collect();
            let lambda = DMatrix::from_diagonal(&DVector::from_vec(diag));
            let avg_val = simulate_for_lambda(n, &lambda, num_haar_samples, epsilon, R_ORDER);
            if avg_val > max_avg {
                max_avg = avg_val;
            }
        }
        println!("  N = {}: Maximum sample average E[(Δ^{}F)^2] = {:.4e}", n, R_ORDER, max_avg);
        results.push((n as f64, max_avg));
    }

    // Generate a file name based on R_ORDER, e.g., "1-laplacian.csv" if R_ORDER=1.
    let filename = format!("{}-laplacian.csv", R_ORDER);
    let mut wtr = csv::Writer::from_path(&filename)?;
    wtr.write_record(&["N", "value"])?;
    for (n, val) in &results {
        wtr.write_record(&[n.to_string(), val.to_string()])?;
    }
    wtr.flush()?;
    println!("CSV data saved to {}", filename);

    // Plot the data.
    let root = BitMapBackend::new("max_plot.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let max_y = results.iter().map(|&(_, y)| y).fold(0.0, f64::max);
    let x_max = (MAX_N + 1) as f64;
    let mut chart = ChartBuilder::on(&root)
        .caption(format!("Max E[(Δ^{}F)^2] vs N", R_ORDER), ("sans-serif", 30).into_font())
        .margin(40)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0f64..x_max, 0f64..(max_y * 1.1))?;
    
    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        results.clone(),
        &BLUE,
    ))?
    .label(format!("Max E[(Δ^{}F)^2]", R_ORDER))
    .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &BLUE));

    chart.configure_series_labels()
         .border_style(&BLACK)
         .draw()?;

    println!("Plot saved to max_plot.png");
    Ok(())
}













// use nalgebra::{DMatrix, DVector};
// use nalgebra::linalg::QR;
// use plotters::prelude::*;
// use rand::prelude::*;
// use rand_distr::{Uniform, StandardNormal};
// use std::error::Error;
// use std::f64::consts::PI;

// /// Set the maximum N here. Change MAX_N to adjust the range.
// const MAX_N: usize = 15;

// /// Generate a Haar-distributed orthogonal matrix of size n x n.
// fn generate_haar(n: usize) -> DMatrix<f64> {
//     let mut rng = rand::thread_rng();
//     let data: Vec<f64> = (0..n*n).map(|_| rng.sample(StandardNormal)).collect();
//     let m = DMatrix::from_vec(n, n, data);
//     let qr = QR::new(m);
//     let mut q = qr.q();
//     let r = qr.r();
//     // Adjust sign ambiguity.
//     for i in 0..n {
//         if r[(i, i)] < 0.0 {
//             for j in 0..n {
//                 q[(j, i)] = -q[(j, i)];
//             }
//         }
//     }
//     q
// }

// /// Returns an n x n rotation matrix in the (i, j) plane by angle epsilon.
// fn rotation_matrix(n: usize, i: usize, j: usize, epsilon: f64) -> DMatrix<f64> {
//     let mut r = DMatrix::<f64>::identity(n, n);
//     let c = epsilon.cos();
//     let s = epsilon.sin();
//     r[(i, i)] = c;
//     r[(j, j)] = c;
//     r[(i, j)] = s;
//     r[(j, i)] = -s;
//     r
// }

// /// Generate all states in {±1}^n.
// fn generate_sigma_states(n: usize) -> Vec<DVector<f64>> {
//     let num_states = 1 << n; // 2^n states.
//     let mut states = Vec::with_capacity(num_states);
//     for state in 0..num_states {
//         let mut vec = Vec::with_capacity(n);
//         for i in 0..n {
//             vec.push(if (state >> i) & 1 == 1 { 1.0 } else { -1.0 });
//         }
//         states.push(DVector::from_vec(vec));
//     }
//     states
// }

// /// Compute the free energy:
// /// F(O) = ln(sum_{σ in {±1}^n} exp(σ^T A σ))
// /// where A = O^T * Λ * O. Here we take β = 1.
// fn free_energy(o: &DMatrix<f64>, lambda: &DMatrix<f64>, sigma_states: &Vec<DVector<f64>>) -> f64 {
//     let a = o.transpose() * lambda * o;
//     let mut z = 0.0;
//     for sigma in sigma_states {
//         let h_val = (sigma.transpose() * &a * sigma)[(0, 0)];
//         z += h_val.exp();
//     }
//     z.ln()
// }

// /// Compute the free energy for a perturbation in one edge direction.
// /// The perturbation is applied by multiplying O₀ on the right by a rotation in the (i,j)-plane.
// fn free_energy_edge(o0: &DMatrix<f64>, edge: (usize, usize), epsilon: f64, lambda: &DMatrix<f64>, sigma_states: &Vec<DVector<f64>>) -> f64 {
//     let n = o0.nrows();
//     let r_mat = rotation_matrix(n, edge.0, edge.1, epsilon);
//     let o_eps = o0 * r_mat;
//     free_energy(&o_eps, lambda, sigma_states)
// }

// /// Compute the second derivative in the direction corresponding to a given edge using a central finite difference.
// fn second_derivative(o0: &DMatrix<f64>, edge: (usize, usize), sigma_states: &Vec<DVector<f64>>, lambda: &DMatrix<f64>, epsilon: f64) -> f64 {
//     let f0 = free_energy_edge(o0, edge, 0.0, lambda, sigma_states);
//     let f_plus = free_energy_edge(o0, edge, epsilon, lambda, sigma_states);
//     let f_minus = free_energy_edge(o0, edge, -epsilon, lambda, sigma_states);
//     (f_plus - 2.0 * f0 + f_minus) / (epsilon * epsilon)
// }

// /// For a given n and Λ, compute the sample average (over Haar samples) of (Δ^1F)^2,
// /// where Δ^1F is the sum over all second derivatives along each edge.
// fn simulate_for_lambda(n: usize, lambda: &DMatrix<f64>, num_haar_samples: usize, epsilon: f64) -> f64 {
//     let sigma_states = generate_sigma_states(n);
    
//     // List all edges (i, j) with i < j.
//     let mut edges = Vec::new();
//     for i in 0..n {
//         for j in (i+1)..n {
//             edges.push((i, j));
//         }
//     }
    
//     // For n=1, no edges exist.
//     if edges.is_empty() {
//         return 0.0;
//     }
    
//     let mut laplacian_samples = Vec::with_capacity(num_haar_samples);
//     for _ in 0..num_haar_samples {
//         let o0 = generate_haar(n);
//         let mut laplacian = 0.0;
//         for &edge in &edges {
//             let d2f = second_derivative(&o0, edge, &sigma_states, lambda, epsilon);
//             laplacian += d2f;
//         }
//         laplacian_samples.push(laplacian);
//     }
    
//     // Return the average of the squared Laplacian values.
//     laplacian_samples.iter().map(|&x| x * x).sum::<f64>() / (laplacian_samples.len() as f64)
// }

// fn main() -> Result<(), Box<dyn Error>> {
//     let num_haar_samples = 20;     // Haar samples per Λ trial.
//     let num_lambda_samples = 100;  // Number of independent Λ samples per N.
//     let epsilon = 1e-4;            // Finite difference step size.
    
//     let mut results = Vec::new();

//     // Loop over n from 1 to MAX_N.
//     for n in 1..=MAX_N {
//         println!("Processing N = {}", n);
        
//         let mut max_avg = std::f64::MIN;
//         for _ in 0..num_lambda_samples {
//             // Create a diagonal Λ with i.i.d. entries from Uniform[-1,1].
//             let mut rng = rand::thread_rng();
//             let uniform = Uniform::new_inclusive(-1.0, 1.0);
//             let diag: Vec<f64> = (0..n).map(|_| rng.sample(uniform)).collect();
//             let lambda = DMatrix::from_diagonal(&DVector::from_vec(diag));
            
//             let avg_val = simulate_for_lambda(n, &lambda, num_haar_samples, epsilon);
//             if avg_val > max_avg {
//                 max_avg = avg_val;
//             }
//         }
//         println!("  N = {}: Maximum sample average E[(Δ^1F)^2] = {:.4e}", n, max_avg);
//         results.push((n as f64, max_avg));
//     }

//     // Write the results to CSV.
//     let mut wtr = csv::Writer::from_path("data.csv")?;
//     wtr.write_record(&["N", "value"])?;
//     for (n, val) in &results {
//         wtr.write_record(&[n.to_string(), val.to_string()])?;
//     }
//     wtr.flush()?;
//     println!("CSV data saved to data.csv");

//     // Plot the data.
//     let root = BitMapBackend::new("max_plot.png", (640, 480)).into_drawing_area();
//     root.fill(&WHITE)?;
//     let max_y = results.iter().map(|&(_, y)| y).fold(0.0, f64::max);
//     let x_max = (MAX_N + 1) as f64;
//     let mut chart = ChartBuilder::on(&root)
//         .caption("Max E[(Δ^1F)^2] vs N", ("sans-serif", 30).into_font())
//         .margin(40)
//         .x_label_area_size(40)
//         .y_label_area_size(50)
//         .build_cartesian_2d(0f64..x_max, 0f64..(max_y * 1.1))?;
    
//     chart.configure_mesh().draw()?;

//     chart.draw_series(LineSeries::new(
//         results.clone(),
//         &BLUE,
//     ))?
//     .label("Max E[(Δ^1F)^2]")
//     .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &BLUE));

//     chart.configure_series_labels()
//          .border_style(&BLACK)
//          .draw()?;

//     println!("Plot saved to max_plot.png");
//     Ok(())
// }
















// use std::cmp::min;

// /// Compute the quadratic Casimir eigenvalue for a highest weight vector.
// /// Here, lambda_vec is assumed to be nonnegative and in nonincreasing order,
// /// and N is the dimension parameter for SO_N.
// fn casimir_eigenvalue(lambda_vec: &Vec<i32>, N: usize) -> i32 {
//     let mut s = 0;
//     for (i, &lam) in lambda_vec.iter().enumerate() {
//         // Use 1-indexing in the formula.
//         s += lam * (lam + (N as i32) - 2 * ((i + 1) as i32));
//     }
//     s
// }

// /// Compute the dimension for SO_N with N odd.
// /// Let n = floor(N/2). Extend lambda_vec to length n by appending zeros if needed.
// fn dimension_odd(lambda_vec: &Vec<i32>, N: usize) -> f64 {
//     let n = N / 2;
//     let mut lam = lambda_vec.clone();
//     while lam.len() < n {
//         lam.push(0);
//     }
//     let mut prod1 = 1.0;
//     for i in 1..=n {
//         let num = lam[i - 1] as f64 + (N as f64) / 2.0 - (i as f64);
//         let den = (N as f64) / 2.0 - (i as f64);
//         prod1 *= num / den;
//     }
//     let mut prod2 = 1.0;
//     for i in 1..n {
//         for j in (i + 1)..=n {
//             let num = (lam[i - 1] + lam[j - 1] + (N as i32) - (i as i32) - (j as i32)) as f64 *
//                       (lam[i - 1] - lam[j - 1] + (j as i32) - (i as i32)) as f64;
//             let den = ((N as i32) - (i as i32) - (j as i32)) as f64 * ((j - i) as f64);
//             prod2 *= num / den;
//         }
//     }
//     prod1 * prod2
// }

// /// Compute the dimension for SO_N with N even.
// /// Here, n = N/2 and the formula is given by the product over 1 ≤ i < j ≤ n.
// fn dimension_even(lambda_vec: &Vec<i32>, N: usize) -> f64 {
//     let n = N / 2;
//     let mut lam = lambda_vec.clone();
//     while lam.len() < n {
//         lam.push(0);
//     }
//     let mut prod = 1.0;
//     for i in 1..n {
//         for j in (i + 1)..=n {
//             let num = (lam[i - 1] + lam[j - 1] + (N as i32) - (i as i32) - (j as i32)) as f64 *
//                       (lam[i - 1] - lam[j - 1] + (j as i32) - (i as i32)) as f64;
//             let den = ((N as i32) - (i as i32) - (j as i32)) as f64 * ((j - i) as f64);
//             prod *= num / den;
//         }
//     }
//     prod
// }

// /// Compute the dimension of the irreducible representation of SO_N with highest weight lambda_vec.
// fn dimension_so(lambda_vec: &Vec<i32>, N: usize) -> f64 {
//     if N % 2 == 1 {
//         dimension_odd(lambda_vec, N)
//     } else {
//         dimension_even(lambda_vec, N)
//     }
// }

// /// Search over highest weights (nonincreasing sequences) of length k = min(M, floor(N/2))
// /// with entries in {0,1,...,M-1} subject to the constraint that the Casimir eigenvalue is less than N*M.
// /// Returns a tuple (max_dimension, best_highest_weight).
// fn search_max_dimension(N: u32, M: u32) -> (f64, Option<Vec<i32>>) {
//     // Define the threshold as N * M.
//     let threshold = (N * M) as i32;
//     let n = N / 2;
//     let k = min(M, n);
//     let mut max_dim = 0.0;
//     let mut best_lambda: Option<Vec<i32>> = None;

//     // Recursive helper function.
//     fn rec(
//         seq: Vec<i32>,
//         last: i32,
//         k: u32,
//         N: u32,
//         threshold: i32,
//         max_dim: &mut f64,
//         best_lambda: &mut Option<Vec<i32>>,
//     ) {
//         if seq.len() == k as usize {
//             let cp = casimir_eigenvalue(&seq, N as usize);
//             if cp < threshold {
//                 let d = dimension_so(&seq, N as usize);
//                 if d > *max_dim {
//                     *max_dim = d;
//                     *best_lambda = Some(seq);
//                 }
//             }
//             return;
//         }
//         // Iterate in descending order from 'last' down to 0.
//         for x in (0..=last).rev() {
//             let mut new_seq = seq.clone();
//             new_seq.push(x);
//             rec(new_seq, x, k, N, threshold, max_dim, best_lambda);
//         }
//     }

//     rec(vec![], threshold - 1, k, N, threshold, &mut max_dim, &mut best_lambda);
//     (max_dim, best_lambda)
// }

// use plotters::prelude::*;
// use std::error::Error;

// fn main() -> Result<(), Box<dyn Error>> {
//     let M_fixed: u32 = 10; // The constraint is λ_π < N*M_fixed.
//     println!("Searching for maximum dimension subject to λ_π < N*M, with M = {}", M_fixed);

//     let mut simulation_data = Vec::new();
//     let mut n_pow_m_data = Vec::new();

//     // Loop over N from 1 to 10.
//     for N in 1..=10 {
//         let (max_dim, best_lambda) = search_max_dimension(N, M_fixed);
//         println!(
//             "N = {}, max d_π = {:.2}, N^M = {} best highest weight = {:?}",
//             N,
//             max_dim,
//             N.pow(M_fixed),
//             best_lambda
//         );
//         simulation_data.push((N as f64, max_dim));
//         // Compute N^M (convert to f64)
//         let n_pow_m = (N.pow(M_fixed)) as f64;
//         n_pow_m_data.push((N as f64, n_pow_m));
//     }

//     // Determine y-axis range.
//     // Use a lower bound that's positive (log scale cannot include 0).
//     let sim_y_min = simulation_data.iter().map(|&(_, y)| y).fold(f64::INFINITY, f64::min);
//     let n_pow_m_y_min = n_pow_m_data.iter().map(|&(_, y)| y).fold(f64::INFINITY, f64::min);
//     let y_min = sim_y_min.min(n_pow_m_y_min).max(0.1);
//     let sim_y_max = simulation_data.iter().map(|&(_, y)| y).fold(0.0, f64::max);
//     let n_pow_m_y_max = n_pow_m_data.iter().map(|&(_, y)| y).fold(0.0, f64::max);
//     let y_max = sim_y_max.max(n_pow_m_y_max) * 1.1;

//     // Set up the drawing area using a logarithmic y-axis.
//     let filename = "max_dimension_plot.png";
//     let root = BitMapBackend::new(&filename, (800, 600)).into_drawing_area();
//     root.fill(&WHITE)?;

//     // Build a chart with x from 0 to 11 and y as log-scale.
//     let mut chart = ChartBuilder::on(&root)
//         .caption("Max Dimension vs. N (Log Scale)", ("sans-serif", 20))
//         .margin(20)
//         .x_label_area_size(40)
//         .y_label_area_size(60)
//         .build_cartesian_2d(0.0..11.0, (y_min..y_max).log_scale())?;

//     chart.configure_mesh()
//          .x_desc("N")
//          .y_desc("Value (log scale)")
//          .draw()?;

//     // Draw simulation data (max d_π) as red circles connected by a red line.
//     chart.draw_series(LineSeries::new(
//         simulation_data.iter().cloned(),
//         &RED,
//     ))?;
//     chart.draw_series(simulation_data.iter().map(|&(x, y)| Circle::new((x, y), 5, RED.filled())))?;

//     // Draw N^M data as blue circles connected by a blue line.
//     chart.draw_series(LineSeries::new(
//         n_pow_m_data.iter().cloned(),
//         &BLUE,
//     ))?;
//     chart.draw_series(n_pow_m_data.iter().map(|&(x, y)| Circle::new((x, y), 5, BLUE.filled())))?;

//     chart.configure_series_labels()
//          .background_style(&WHITE.mix(0.8))
//          .border_style(&BLACK)
//          .draw()?;

//     println!("Plot saved to {}", filename);
//     Ok(())
// }