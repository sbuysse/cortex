//! Grid Cell Encoder — hexagonal spatial encoding of concept space.
//!
//! Inspired by Doeller et al. 2010 (PMC3173857):
//! projects 512-dim embeddings to 2D coordinates with multi-scale
//! hexagonal periodic activations (60° rotational symmetry).

use ndarray::{Array1, Array2};
use serde::Serialize;

const GRID_ORIENTATIONS: [f32; 3] = [0.0, std::f32::consts::PI / 9.0, 2.0 * std::f32::consts::PI / 9.0];

/// Multi-scale hexagonal grid encoder.
pub struct GridCellEncoder {
    projection: Option<Array2<f32>>,  // (512, 2) PCA projection
    mean: Option<Array1<f32>>,         // (512,) center
    scales: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GridPoint {
    pub label: String,
    pub x: f32,
    pub y: f32,
    pub grid_activation: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct GridNearby {
    pub label: String,
    pub distance: f32,
    pub grid_x: f32,
    pub grid_y: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct GridPathPoint {
    pub x: f32,
    pub y: f32,
}

impl GridCellEncoder {
    pub fn new(scales: Vec<f32>) -> Self {
        Self { projection: None, mean: None, scales }
    }

    /// Fit the 2D projection from a set of embeddings (concept codebook).
    pub fn fit(&mut self, embeddings: &Array2<f32>) {
        let n = embeddings.nrows();
        let d = embeddings.ncols();

        // Compute mean
        let mean = embeddings.mean_axis(ndarray::Axis(0)).unwrap();

        // Center the data
        let mut centered = embeddings.clone();
        for mut row in centered.rows_mut() {
            row -= &mean;
        }

        // Covariance matrix (simplified: C = X^T X / (n-1))
        let cov = centered.t().dot(&centered) / (n.max(2) - 1) as f32;

        // Power iteration for top 2 eigenvectors (faster than full eigendecomp)
        let mut v1 = Array1::from_elem(d, 1.0 / (d as f32).sqrt());
        let mut v2 = Array1::zeros(d);
        v2[0] = 1.0;

        for _ in 0..100 {
            v1 = cov.dot(&v1);
            let norm = v1.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            v1.mapv_inplace(|x| x / norm);
        }

        // Second eigenvector: orthogonalize then iterate
        let proj = v1.dot(&v2);
        v2 = &v2 - &(&v1 * proj);
        let norm = v2.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        v2.mapv_inplace(|x| x / norm);
        for _ in 0..100 {
            v2 = cov.dot(&v2);
            let proj = v1.dot(&v2);
            v2 = &v2 - &(&v1 * proj);
            let norm = v2.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            v2.mapv_inplace(|x| x / norm);
        }

        // Build projection matrix (d, 2)
        let mut projection = Array2::zeros((d, 2));
        projection.column_mut(0).assign(&v1);
        projection.column_mut(1).assign(&v2);

        self.mean = Some(mean);
        self.projection = Some(projection);
    }

    /// Project a 512-dim embedding to 2D.
    pub fn to_2d(&self, embedding: &[f32]) -> [f32; 2] {
        let (proj, mean) = match (&self.projection, &self.mean) {
            (Some(p), Some(m)) => (p, m),
            _ => return [0.0, 0.0],
        };
        let d = embedding.len();
        let mut result = [0.0f32; 2];
        for j in 0..2 {
            for i in 0..d {
                result[j] += (embedding[i] - mean[i]) * proj[[i, j]];
            }
        }
        result
    }

    /// Compute hexagonal grid cell activations.
    pub fn grid_activation(&self, pos: [f32; 2]) -> Vec<f32> {
        let mut activations = Vec::new();
        for &scale in &self.scales {
            for &theta in &GRID_ORIENTATIONS {
                let cos_t = theta.cos();
                let sin_t = theta.sin();
                let rotated = [
                    pos[0] * cos_t - pos[1] * sin_t,
                    pos[0] * sin_t + pos[1] * cos_t,
                ];
                // Hexagonal: sum of 3 cosines at 60° intervals
                let mut hex_act = 0.0f32;
                for k in 0..3 {
                    let angle = k as f32 * std::f32::consts::PI / 3.0;
                    let dir = [angle.cos(), angle.sin()];
                    let dot = rotated[0] * dir[0] + rotated[1] * dir[1];
                    hex_act += (2.0 * std::f32::consts::PI * dot / scale).cos();
                }
                activations.push(hex_act / 3.0);
                // Phase
                let phase = (2.0 * std::f32::consts::PI * rotated[0] / scale).sin()
                    .atan2((2.0 * std::f32::consts::PI * rotated[0] / scale).cos());
                activations.push(phase / std::f32::consts::PI);
            }
        }
        activations
    }

    /// Grid-metric distance between two embeddings.
    pub fn grid_distance(&self, emb_a: &[f32], emb_b: &[f32]) -> f32 {
        let pos_a = self.to_2d(emb_a);
        let pos_b = self.to_2d(emb_b);
        let dx = pos_a[0] - pos_b[0];
        let dy = pos_a[1] - pos_b[1];
        let euclidean = (dx * dx + dy * dy).sqrt();

        let act_a = self.grid_activation(pos_a);
        let act_b = self.grid_activation(pos_b);
        let dot: f32 = act_a.iter().zip(&act_b).map(|(a, b)| a * b).sum();
        let norm_a: f32 = act_a.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        let norm_b: f32 = act_b.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        let cos_sim = dot / (norm_a * norm_b);

        euclidean * (1.0 - cos_sim * 0.5)
    }

    /// Find nearby concepts on the grid.
    pub fn find_nearby(&self, embedding: &[f32], candidates: &Array2<f32>,
                       labels: &[String], top_k: usize) -> Vec<GridNearby> {
        let pos = self.to_2d(embedding);
        let n = candidates.nrows();
        let mut scored: Vec<(usize, f32)> = (0..n).map(|i| {
            let row = candidates.row(i);
            let cpos = self.to_2d(row.as_slice().unwrap());
            let dx = pos[0] - cpos[0];
            let dy = pos[1] - cpos[1];
            (i, (dx * dx + dy * dy).sqrt())
        }).collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        scored.into_iter().take(top_k).map(|(i, dist)| {
            let row = candidates.row(i);
            let cpos = self.to_2d(row.as_slice().unwrap());
            GridNearby {
                label: labels.get(i).cloned().unwrap_or_default(),
                distance: (dist * 10000.0).round() / 10000.0,
                grid_x: (cpos[0] * 10000.0).round() / 10000.0,
                grid_y: (cpos[1] * 10000.0).round() / 10000.0,
            }
        }).collect()
    }

    pub fn is_fitted(&self) -> bool { self.projection.is_some() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn mock_codebook() -> (Array2<f32>, Vec<String>) {
        let mut cb = Array2::zeros((10, 512));
        for i in 0..10 {
            for j in 0..512 {
                cb[[i, j]] = ((i * 512 + j) as f32 * 0.001).sin();
            }
        }
        // L2 normalize rows
        for mut row in cb.rows_mut() {
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            row.mapv_inplace(|x| x / norm);
        }
        let labels: Vec<String> = (0..10).map(|i| format!("concept_{i}")).collect();
        (cb, labels)
    }

    #[test]
    fn test_fit_and_project() {
        let (cb, _) = mock_codebook();
        let mut enc = GridCellEncoder::new(vec![0.05, 0.15, 0.5]);
        enc.fit(&cb);
        assert!(enc.is_fitted());
        let pos = enc.to_2d(cb.row(0).as_slice().unwrap());
        assert!(pos[0].is_finite() && pos[1].is_finite());
    }

    #[test]
    fn test_grid_activation_shape() {
        let (cb, _) = mock_codebook();
        let mut enc = GridCellEncoder::new(vec![0.05, 0.15, 0.5]);
        enc.fit(&cb);
        let pos = enc.to_2d(cb.row(0).as_slice().unwrap());
        let act = enc.grid_activation(pos);
        // 3 scales * 3 orientations * 2 = 18
        assert_eq!(act.len(), 18);
    }

    #[test]
    fn test_self_distance_near_zero() {
        let (cb, _) = mock_codebook();
        let mut enc = GridCellEncoder::new(vec![0.05, 0.15, 0.5]);
        enc.fit(&cb);
        let emb = cb.row(0).to_vec();
        let d = enc.grid_distance(&emb, &emb);
        assert!(d < 0.01, "Self-distance should be ~0, got {d}");
    }

    #[test]
    fn test_find_nearby() {
        let (cb, labels) = mock_codebook();
        let mut enc = GridCellEncoder::new(vec![0.05, 0.15, 0.5]);
        enc.fit(&cb);
        let results = enc.find_nearby(cb.row(0).as_slice().unwrap(), &cb, &labels, 5);
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].label, "concept_0"); // closest to itself
    }
}
