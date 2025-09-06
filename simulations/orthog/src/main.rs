use nalgebra::DMatrix;
use rand::distributions::Uniform;
use rand_distr::Normal;
use rand::prelude::*;
use csv::Writer;
use std::fs::File;

fn main() {
    // Create CSV writers to store the results
    let file_var = File::create("orthog_var.csv").expect("Cannot create orthog_var.csv");
    let mut wtr_var = Writer::from_writer(file_var);
    wtr_var.write_record(&["N", "beta", "max_variance"]).unwrap();

    let file_mean = File::create("orthog_mean.csv").expect("Cannot create orthog_mean.csv");
    let mut wtr_mean = Writer::from_writer(file_mean);
    wtr_mean.write_record(&["N", "beta", "max_mean"]).unwrap();

    // The three beta values
    let beta_values = vec![0.1, 1.0, 10.0];

    // Random number generators
    let mut rng = rand::thread_rng();
    let uniform_dist = Uniform::new(-1.0, 1.0);
    let normal_dist = Normal::new(0.0, 1.0).unwrap();

    for n in 1..=15 {
        println!("Processing N = {}", n);
        for &beta in &beta_values {
            let mut max_variance = std::f64::MIN;
            let mut max_mean = std::f64::MIN;
            for _ in 0..100 {
                // Generate Lambda_N with diagonal entries from Uniform(-1,1)
                let lambda: Vec<f64> = (0..n).map(|_| uniform_dist.sample(&mut rng)).collect();

                // Collect F_{N,beta} values from 1000 independent Haar samples
                let mut f_values = Vec::with_capacity(1000);
                for _ in 0..1000 {
                    // Generate an n x n matrix G with i.i.d. N(0,1) entries
                    let g_data: Vec<f64> = (0..n*n).map(|_| normal_dist.sample(&mut rng)).collect();
                    let g = DMatrix::from_vec(n, n, g_data);

                    // Compute the QR decomposition of G
                    let qr = g.qr();
                    let mut q = qr.q();

                    // Adjust Q so that the diagonal of R is positive
                    let r = qr.r();
                    for i in 0..n {
                        if r[(i, i)] < 0.0 {
                            for j in 0..n {
                                q[(j, i)] = -q[(j, i)];
                            }
                        }
                    }

                    // Compute J = Q^T * Lambda * Q
                    let mut d_q = DMatrix::zeros(n, n);
                    for i in 0..n {
                        for j in 0..n {
                            d_q[(i, j)] = lambda[i] * q[(i, j)];
                        }
                    }
                    let j_matrix = q.transpose() * d_q;

                    // Compute F_{N,beta}
                    let mut sum_exp = 0.0;
                    for s in 0..(1 << n) {
                        let mut quad_form = 0.0;
                        for i in 0..n {
                            let sigma_i = if (s >> i) & 1 == 1 { 1.0 } else { -1.0 };
                            for j in 0..n {
                                let sigma_j = if (s >> j) & 1 == 1 { 1.0 } else { -1.0 };
                                quad_form += sigma_i * j_matrix[(i, j)] * sigma_j;
                            }
                        }
                        sum_exp += (beta * quad_form).exp();
                    }
                    let f_val = (1.0 / beta) * sum_exp.ln();
                    f_values.push(f_val);
                }

                // Compute the sample mean and variance and update maxima
                let mean: f64 = f_values.iter().sum::<f64>() / (f_values.len() as f64);
                let variance: f64 = f_values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / ((f_values.len() - 1) as f64);

                if variance > max_variance {
                    max_variance = variance;
                }

                if mean > max_mean {
                    max_mean = mean;
                }
            }
            // Write CSV records for this combination of n and beta
            wtr_var.write_record(&[n.to_string(), beta.to_string(), max_variance.to_string()]).unwrap();
            println!("   beta = {}: max variance = {}", beta, max_variance);

            wtr_mean.write_record(&[n.to_string(), beta.to_string(), max_mean.to_string()]).unwrap();
            println!("   beta = {}: max mean = {}", beta, max_mean);
        }
    }
    // Flush the CSV writers
    wtr_var.flush().unwrap();
    println!("Simulation complete. Results saved to orthog_var.csv.");

    wtr_mean.flush().unwrap();
    println!("Simulation complete. Results saved to orthog_mean.csv.");
}
