use csv::Reader;
use serde::Deserialize;
use std::error::Error;

#[derive(Debug, Deserialize)]
struct Record {
    #[serde(rename = "N")]
    n: f64,
    value: f64,
}

fn estimate_exponent(data: &[(f64, f64)]) -> f64 {
    let n_points = data.len() as f64;
    let sum_log_n: f64 = data.iter().map(|&(n, _)| n.ln()).sum();
    let sum_log_y: f64 = data.iter().map(|&(_, y)| y.ln()).sum();
    let sum_log_n_log_y: f64 = data.iter().map(|&(n, y)| n.ln() * y.ln()).sum();
    let sum_log_n_sq: f64 = data.iter().map(|&(n, _)| n.ln().powi(2)).sum();
    
    // Compute the slope alpha = (n * sum(log n * log y) - sum(log n)*sum(log y)) / (n * sum(log n^2) - (sum(log n))^2)
    (n_points * sum_log_n_log_y - sum_log_n * sum_log_y)
        / (n_points * sum_log_n_sq - sum_log_n.powi(2))
}

fn main() -> Result<(), Box<dyn Error>> {
    // Open the CSV file.
    let mut rdr = Reader::from_path("1-laplacian.csv")?;
    let mut data = Vec::new();
    
    // Deserialize records. We filter out any with nonpositive values (since ln(0) is undefined).
    for result in rdr.deserialize() {
        let record: Record = result?;
        if record.value > 0.0 {
            data.push((record.n, record.value));
        }
    }
    
    if data.len() < 2 {
        println!("Not enough data points with positive values.");
        return Ok(());
    }
    
    let alpha = estimate_exponent(&data);
    println!("Estimated exponent α ≈ {:.4}", alpha);
    
    Ok(())
}