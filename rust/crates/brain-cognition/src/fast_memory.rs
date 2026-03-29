//! Hopfield Fast Memory — one-shot pattern store for rapid associative recall.
//!
//! Inspired by hippocampal fast learning (PMC11591613):
//! stores patterns instantly, retrieves via cosine similarity.

use serde::Serialize;

/// Fast associative memory (circular buffer of L2-normalized patterns).
pub struct HopfieldMemory {
    patterns: Vec<Vec<f32>>,  // (capacity, dim)
    labels: Vec<String>,
    count: usize,
    capacity: usize,
    dim: usize,
}

#[derive(Debug, Serialize)]
pub struct FastMemoryMatch {
    pub label: String,
    pub similarity: f32,
    pub idx: usize,
}

impl HopfieldMemory {
    pub fn new(dim: usize, capacity: usize) -> Self {
        Self {
            patterns: Vec::with_capacity(capacity),
            labels: Vec::with_capacity(capacity),
            count: 0,
            capacity,
            dim,
        }
    }

    /// Store a pattern (one-shot, instant).
    pub fn store(&mut self, pattern: &[f32], label: &str) {
        if pattern.len() != self.dim { return; }
        let norm: f32 = pattern.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        let normalized: Vec<f32> = pattern.iter().map(|x| x / norm).collect();

        let idx = self.count % self.capacity;
        if idx >= self.patterns.len() {
            self.patterns.push(normalized);
            self.labels.push(label.to_string());
        } else {
            self.patterns[idx] = normalized;
            self.labels[idx] = label.to_string();
        }
        self.count += 1;
    }

    /// Pattern completion via cosine similarity.
    pub fn retrieve(&self, query: &[f32], top_k: usize) -> Vec<FastMemoryMatch> {
        if self.patterns.is_empty() { return Vec::new(); }
        let norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        let q: Vec<f32> = query.iter().map(|x| x / norm).collect();

        let n = self.patterns.len();
        let mut scored: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let sim: f32 = q.iter().zip(&self.patterns[i]).map(|(a, b)| a * b).sum();
                (i, sim)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        scored.into_iter().take(top_k).map(|(i, sim)| {
            FastMemoryMatch {
                label: self.labels[i].clone(),
                similarity: (sim * 10000.0).round() / 10000.0,
                idx: i,
            }
        }).collect()
    }

    /// Get recent patterns for consolidation transfer.
    pub fn get_recent(&self, n: usize) -> Vec<&[f32]> {
        let total = self.patterns.len();
        let start = if total > n { total - n } else { 0 };
        self.patterns[start..].iter().map(|p| p.as_slice()).collect()
    }

    /// Get the stored (L2-normalized) pattern at a given index.
    /// Returns None if idx is out of range.
    pub fn pattern_at(&self, idx: usize) -> Option<&[f32]> {
        self.patterns.get(idx).map(|p| p.as_slice())
    }

    pub fn count(&self) -> usize { self.count }
    pub fn capacity(&self) -> usize { self.capacity }
    pub fn recent_labels(&self, n: usize) -> Vec<&str> {
        let total = self.labels.len();
        let start = if total > n { total - n } else { 0 };
        self.labels[start..].iter().map(|s| s.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_and_retrieve() {
        let mut mem = HopfieldMemory::new(512, 100);
        let pattern = vec![0.1f32; 512];
        mem.store(&pattern, "test");
        let results = mem.retrieve(&pattern, 1);
        assert_eq!(results.len(), 1);
        assert!(results[0].similarity > 0.99);
        assert_eq!(results[0].label, "test");
    }

    #[test]
    fn test_capacity_wrap() {
        let mut mem = HopfieldMemory::new(4, 3);
        mem.store(&[1.0, 0.0, 0.0, 0.0], "a");
        mem.store(&[0.0, 1.0, 0.0, 0.0], "b");
        mem.store(&[0.0, 0.0, 1.0, 0.0], "c");
        mem.store(&[0.0, 0.0, 0.0, 1.0], "d"); // wraps, replaces "a"
        assert_eq!(mem.count(), 4);
        assert_eq!(mem.patterns.len(), 3); // capacity=3
        let results = mem.retrieve(&[1.0, 0.0, 0.0, 0.0], 3);
        // "a" was overwritten, so it shouldn't be the top match
        assert_ne!(results[0].label, "a");
    }

    #[test]
    fn test_empty_retrieve() {
        let mem = HopfieldMemory::new(512, 100);
        let results = mem.retrieve(&vec![0.1; 512], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_pattern_at() {
        let mut fm = HopfieldMemory::new(512, 10);
        let pattern: Vec<f32> = (0..512).map(|i| i as f32 / 512.0).collect();
        fm.store(&pattern, "test");
        // retrieve gives idx=0 for the only stored pattern
        let matches = fm.retrieve(&pattern, 1);
        assert_eq!(matches.len(), 1);
        let idx = matches[0].idx;
        let retrieved = fm.pattern_at(idx);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().len(), 512);
    }
}
