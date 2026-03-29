//! Concept codebook — category centroids in MLP-projected space.
//! Also handles semantic search: text → nearest category labels.

use ndarray::Array2;
use std::path::Path;

/// Concept codebook: one 512-dim centroid per category.
pub struct ConceptCodebook {
    pub centroids: Array2<f32>,  // (N, 512)
    pub labels: Vec<String>,
}

impl ConceptCodebook {
    /// Build codebook from pre-encoded label embeddings + MLP projection.
    /// label_embeddings: (N, 384) from text encoder
    /// w_v: (384, 512) MLP visual projection
    pub fn build(label_embeddings: &[(String, Vec<f32>)], w_v: &Array2<f32>) -> Self {
        let n = label_embeddings.len();
        let mut centroids = Array2::zeros((n, 512));
        let mut labels = Vec::with_capacity(n);

        for (i, (label, emb_384)) in label_embeddings.iter().enumerate() {
            // Project through MLP: ReLU(emb @ W_v), L2-norm
            let proj = brain_inference::mlp::project(emb_384, w_v, 0);
            for (j, &v) in proj.iter().enumerate() {
                if j < 512 { centroids[[i, j]] = v; }
            }
            labels.push(label.clone());
        }

        // Mean-center and re-normalize
        let mean = centroids.mean_axis(ndarray::Axis(0)).unwrap();
        for mut row in centroids.rows_mut() {
            row -= &mean;
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            row.mapv_inplace(|x| x / norm);
        }

        tracing::info!("Concept codebook built: {} categories", n);
        Self { centroids, labels }
    }

    /// Find nearest concepts to a 512-dim embedding.
    pub fn nearest(&self, emb: &[f32], top_k: usize) -> Vec<(String, f32)> {
        let n = self.centroids.nrows();
        let emb_norm = l2_norm(emb);

        let mut scored: Vec<(usize, f32)> = (0..n).map(|i| {
            let row = self.centroids.row(i);
            let sim: f32 = row.iter().zip(&emb_norm).map(|(a, b)| a * b).sum();
            (i, sim)
        }).collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        scored.into_iter().take(top_k).map(|(i, sim)| {
            (self.labels[i].clone(), (sim * 10000.0).round() / 10000.0)
        }).collect()
    }

    /// Decompose an embedding into concept components.
    pub fn decompose(&self, emb: &[f32], k: usize) -> Vec<(String, f32)> {
        self.nearest(emb, k)
    }

    /// Return the centroid embedding (512-dim) of the concept nearest to `emb`.
    /// Returns None if the codebook is empty.
    pub fn top1_centroid(&self, emb: &[f32]) -> Option<Vec<f32>> {
        let n = self.centroids.nrows();
        if n == 0 { return None; }
        let emb_norm = l2_norm(emb);
        let best = (0..n).max_by(|&i, &j| {
            let si: f32 = self.centroids.row(i).iter().zip(&emb_norm).map(|(a, b)| a * b).sum();
            let sj: f32 = self.centroids.row(j).iter().zip(&emb_norm).map(|(a, b)| a * b).sum();
            si.partial_cmp(&sj).unwrap_or(std::cmp::Ordering::Equal)
        })?;
        Some(self.centroids.row(best).to_vec())
    }

    /// Compose: add/subtract concept vectors.
    pub fn compose(&self, add: &[&str], subtract: &[&str], top_k: usize) -> Vec<(String, f32)> {
        let mut result = vec![0.0f32; 512];

        for label in add {
            if let Some(idx) = self.labels.iter().position(|l| l.to_lowercase().contains(&label.to_lowercase())) {
                let row = self.centroids.row(idx);
                for (r, v) in result.iter_mut().zip(row.iter()) { *r += v; }
            }
        }
        for label in subtract {
            if let Some(idx) = self.labels.iter().position(|l| l.to_lowercase().contains(&label.to_lowercase())) {
                let row = self.centroids.row(idx);
                for (r, v) in result.iter_mut().zip(row.iter()) { *r -= v; }
            }
        }

        // L2 normalize
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for v in &mut result { *v /= norm; }

        self.nearest(&result, top_k)
    }

    pub fn len(&self) -> usize { self.labels.len() }
    pub fn is_empty(&self) -> bool { self.labels.is_empty() }

    /// Get the centroid vector for a concept by index.
    pub fn centroid(&self, idx: usize) -> Vec<f32> {
        self.centroids.row(idx).to_vec()
    }

    /// Get all labels.
    pub fn all_labels(&self) -> &[String] { &self.labels }

    /// Add a new concept to the codebook at runtime.
    /// Returns the index of the new concept.
    pub fn add_concept(&mut self, label: String, embedding_512: &[f32]) -> usize {
        if embedding_512.len() != 512 { return self.labels.len(); }
        // Check if already exists
        if let Some(idx) = self.labels.iter().position(|l| l == &label) {
            return idx;
        }
        let emb_norm = l2_norm(embedding_512);
        let idx = self.labels.len();
        // Grow centroids matrix
        let mut new_centroids = Array2::zeros((idx + 1, 512));
        for i in 0..idx {
            for j in 0..512 { new_centroids[[i, j]] = self.centroids[[i, j]]; }
        }
        for (j, &v) in emb_norm.iter().enumerate() { new_centroids[[idx, j]] = v; }
        self.centroids = new_centroids;
        self.labels.push(label);
        idx
    }

    /// Bulk-add concepts from prototypes (MemoryDb blobs).
    /// Filters out noisy names (URLs, test data, overly long names).
    pub fn grow_from_prototypes(&mut self, prototypes: &[super::memory_db::PrototypeRow]) -> usize {
        let mut added = 0;
        for proto in prototypes {
            if self.labels.contains(&proto.name) { continue; }
            // Filter junk prototype names
            let name = &proto.name;
            if name.starts_with("novel_") || name.starts_with("test_")
                || name.contains("http") || name.contains("download")
                || name.contains("mp3") || name.contains("free")
                || name.len() > 60 || name.len() < 2
                || proto.count < 2 { continue; }
            if proto.centroid_blob.len() >= 512 * 4 {
                let emb: Vec<f32> = proto.centroid_blob.chunks_exact(4).take(512)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                self.add_concept(proto.name.clone(), &emb);
                added += 1;
            }
        }
        if added > 0 {
            tracing::info!("Codebook grew by {added} prototypes → {} total", self.labels.len());
        }
        added
    }

    /// Open-vocabulary search: combine codebook + on-demand text encoding.
    /// Falls back to text_encode → MLP project for unknown concepts.
    pub fn open_nearest(
        &self,
        emb: &[f32],
        top_k: usize,
        extra_labels: &[(String, Vec<f32>)],
    ) -> Vec<(String, f32)> {
        let emb_norm = l2_norm(emb);

        // Score against codebook
        let mut scored: Vec<(String, f32)> = (0..self.centroids.nrows()).map(|i| {
            let row = self.centroids.row(i);
            let sim: f32 = row.iter().zip(&emb_norm).map(|(a, b)| a * b).sum();
            (self.labels[i].clone(), sim)
        }).collect();

        // Score against extra concepts (prototypes, fast memory, etc.)
        for (label, extra_emb) in extra_labels {
            let extra_norm = l2_norm(extra_emb);
            let sim: f32 = extra_norm.iter().zip(&emb_norm).map(|(a, b)| a * b).sum();
            scored.push((label.clone(), sim));
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.dedup_by(|a, b| a.0 == b.0);
        scored.into_iter().take(top_k).map(|(l, s)| (l, (s * 10000.0).round() / 10000.0)).collect()
    }
}

fn l2_norm(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter().map(|x| x / norm).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_codebook_build_and_search() {
        let w_v = Array2::from_elem((384, 512), 0.001f32);
        let label_embs: Vec<(String, Vec<f32>)> = (0..5).map(|i| {
            let mut emb = vec![0.0f32; 384];
            emb[i] = 1.0;
            (format!("concept_{i}"), emb)
        }).collect();

        let cb = ConceptCodebook::build(&label_embs, &w_v);
        assert_eq!(cb.len(), 5);

        let results = cb.nearest(&vec![0.1f32; 512], 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_top1_centroid_returns_512dim() {
        // Build a minimal codebook with 2 centroids directly
        let mut centroids = Array2::<f32>::zeros((2, 512));
        centroids[[0, 0]] = 1.0;
        centroids[[1, 1]] = 1.0;
        let cb = ConceptCodebook { centroids, labels: vec!["a".into(), "b".into()] };

        // Query near centroid 0
        let mut q = vec![0.0f32; 512];
        q[0] = 1.0;
        let result = cb.top1_centroid(&q);
        assert!(result.is_some());
        assert_eq!(result.unwrap().len(), 512);
    }

    #[test]
    fn test_top1_centroid_empty_returns_none() {
        let cb = ConceptCodebook {
            centroids: Array2::zeros((0, 512)),
            labels: vec![],
        };
        let q = vec![0.0f32; 512];
        assert!(cb.top1_centroid(&q).is_none());
    }
}
