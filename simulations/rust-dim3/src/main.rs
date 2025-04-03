use nalgebra::{DMatrix, DVector};
use rand::prelude::*;
use rand_distr::Normal;
use plotters::prelude::*;
use std::error::Error;

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

/// For a given n and β, runs multiple simulations (with a fixed Λ) and returns
/// the average squared gradient.
fn simulate_grad_sq(n: usize, beta: f64, n_samples: usize, n_steps: usize, burn_in: usize) -> f64 {
    let mut grad_sq_vals = Vec::new();
    // Sample lambda_j's once for this n from the semicircular distribution on [-1,1].
    let lambda_vals: Vec<f64> = (0..n)
        .map(|_| {
            let mut rng = rand::thread_rng();
            let x: f64 = rng.gen_range(-1.0..1.0);
            let y: f64 = rng.gen_range(0.0..(2.0 / std::f64::consts::PI));
            if y < (2.0 / std::f64::consts::PI) * (1.0 - x * x).sqrt() {
                x
            } else {
                // If not, try again.
                loop {
                    let x: f64 = rng.gen_range(-1.0..1.0);
                    let y: f64 = rng.gen_range(0.0..(2.0 / std::f64::consts::PI));
                    if y < (2.0 / std::f64::consts::PI) * (1.0 - x * x).sqrt() {
                        break x;
                    }
                }
            }
        })
        .collect();
    let Lambda = DMatrix::from_diagonal(&DVector::from_vec(lambda_vals));
    
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
    grad_sq_vals.iter().sum::<f64>() / (grad_sq_vals.len() as f64)
}

fn main() -> Result<(), Box<dyn Error>> {
    let beta = 1.0;
    let max_n = 40;
    let mut simulation_data = Vec::new();
    
    // Run simulations for n = 1,2,...,40.
    for n in 1..=max_n {
        println!("Simulating for n = {}", n);
        let avg = simulate_grad_sq(n, beta, 30, 10000, 1000);
        simulation_data.push((n as f64, avg));
    }
    
    // Prepare smooth data for f(n) = 16 * (n choose 2) = 8*n*(n-1).
    let n_fine_points = 200;
    let n_min = 1.0;
    let n_max = max_n as f64;
    let mut smooth_data = Vec::with_capacity(n_fine_points);
    for i in 0..n_fine_points {
        let n_val = n_min + (n_max - n_min) * (i as f64) / ((n_fine_points - 1) as f64);
        let f_val = 8.0 * n_val * (n_val - 1.0);
        smooth_data.push((n_val, f_val));
    }
    
    // Set up the drawing area.
    let filename = "grad_simulation.png";
    let root = BitMapBackend::new(&filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let sim_y_max = simulation_data.iter().map(|&(_, y)| y).fold(0.0, f64::max);
    let smooth_y_max = smooth_data.iter().map(|&(_, y)| y).fold(0.0, f64::max);
    let y_max_val = sim_y_max.max(smooth_y_max);
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Average Squared Gradient vs. N", ("sans-serif", 20))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(1.0..(max_n as f64 + 0.5), 0.0..(y_max_val * 1.1))?;
    
    chart.configure_mesh().x_desc("N").y_desc("Avg Squared Gradient").draw()?;
    
    // Draw simulation data as red circles and a connecting red line.
    chart.draw_series(
        simulation_data.iter().map(|&(x, y)| Circle::new((x, y), 3, RED.filled())),
    )?
    .label("Simulated Data")
    .legend(|(x, y)| Circle::new((x, y), 3, RED.filled()));
    
    chart.draw_series(LineSeries::new(
        simulation_data.iter().map(|&(x, y)| (x, y)),
        &RED,
    ))?;
    
    // Draw the smooth function as a blue line.
    chart.draw_series(LineSeries::new(
        smooth_data.into_iter(),
        &BLUE,
    ))?
    .label("16 (N choose 2)")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));
    
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;
    
    println!("Plot saved to {}", filename);
    Ok(())
}