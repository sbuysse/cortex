//! 4-bit scalar-quantized embedding index.
//!
//! Reduces f32 embeddings (4 bytes/dim) to 4-bit packed codes (0.5 bytes/dim)
//! for 8× memory reduction and integer dot-product search.
//!
//! Pre-processing: random sign flip (±1 per dim) to randomise coordinate
//! distribution before quantization. Cost: O(d) per vector.
//!
//! Quantization: per-dim linear map using 1st–99th percentile as range.
//! Two 4-bit values packed per byte: low nibble = even index, high nibble = odd.

/// 4-bit quantized embedding index with random-sign pre-processing.
pub struct QuantizedIndex {
    /// Per-dimension sign flip: ±1. Applied before quantization and search.
    signs: Vec<i8>,
    /// Per-dim scale: (p99 - p1) / 15.0.
    scale: Vec<f32>,
    /// Per-dim zero point (p1 of rotated values).
    zero: Vec<f32>,
    /// Packed 4-bit codes. Entry i is `ceil(dim / 2)` bytes.
    /// low nibble = even index coordinate, high nibble = odd index coordinate.
    codes: Vec<Vec<u8>>,
    /// Labels parallel to codes.
    labels: Vec<String>,
    /// Embedding dimension.
    dim: usize,
}

impl QuantizedIndex {
    /// Build a quantized index from labeled f32 vectors.
    pub fn build(labels: &[String], vecs: &[&[f32]]) -> Self {
        assert_eq!(labels.len(), vecs.len());
        if vecs.is_empty() {
            return Self {
                signs: vec![],
                scale: vec![],
                zero: vec![],
                codes: vec![],
                labels: vec![],
                dim: 0,
            };
        }
        let dim = vecs[0].len();
        let signs = gen_signs(dim, 0x1234_5678_9ABC_DEF0u64);
        let rotated: Vec<Vec<f32>> = vecs.iter().map(|v| apply_signs(v, &signs)).collect();
        let (scale, zero) = calibrate(&rotated, dim);
        let codes: Vec<Vec<u8>> = rotated.iter().map(|rv| pack_4bit(rv, &scale, &zero)).collect();
        Self {
            signs,
            scale,
            zero,
            codes,
            labels: labels.to_vec(),
            dim,
        }
    }

    /// Append one vector to a live index (for dynamic indices).
    /// Silently ignored if vec.len() != self.dim.
    pub fn insert(&mut self, label: &str, vec: &[f32]) {
        if self.dim == 0 || vec.len() != self.dim {
            return;
        }
        let rotated = apply_signs(vec, &self.signs);
        let code = pack_4bit(&rotated, &self.scale, &self.zero);
        self.codes.push(code);
        self.labels.push(label.to_string());
    }

    /// Find top-k nearest vectors by approximate inner product.
    /// Returns (label, score) pairs sorted descending.
    pub fn nearest(&self, query: &[f32], k: usize) -> Vec<(String, f32)> {
        if self.codes.is_empty() || query.len() != self.dim {
            return vec![];
        }
        let rq = apply_signs(query, &self.signs);
        let mut scores: Vec<(usize, f32)> = self
            .codes
            .iter()
            .enumerate()
            .map(|(i, code)| {
                (
                    i,
                    dot_dequant(&rq, code, &self.scale, &self.zero, self.dim),
                )
            })
            .collect();
        let k = k.min(scores.len());
        scores.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scores[..k]
            .iter()
            .map(|(i, s)| (self.labels[*i].clone(), *s))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.codes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }
}

fn gen_signs(dim: usize, seed: u64) -> Vec<i8> {
    let mut state = seed;
    (0..dim)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            if (state >> 63) == 0 { 1i8 } else { -1i8 }
        })
        .collect()
}

fn apply_signs(v: &[f32], signs: &[i8]) -> Vec<f32> {
    v.iter()
        .zip(signs.iter())
        .map(|(&x, &s)| x * s as f32)
        .collect()
}

fn calibrate(vecs: &[Vec<f32>], dim: usize) -> (Vec<f32>, Vec<f32>) {
    let n = vecs.len();
    // With fewer than ~100 vectors the p1/p99 percentile collapses to the same
    // element; fall back to a symmetric fixed range that covers unit vectors.
    if n < 10 {
        let scale = vec![2.0f32 / 15.0; dim]; // maps [-1, 1] to [0, 15]
        let zero = vec![-1.0f32; dim];
        return (scale, zero);
    }
    let mut scale = vec![1.0f32; dim];
    let mut zero = vec![0.0f32; dim];
    let mut col: Vec<f32> = Vec::with_capacity(n);
    for d in 0..dim {
        col.clear();
        for v in vecs {
            col.push(v[d]);
        }
        col.sort_unstable_by(|a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let lo_idx = ((n as f32 * 0.01) as usize).min(n - 1);
        let hi_idx = ((n as f32 * 0.99) as usize).min(n - 1);
        let lo = col[lo_idx];
        let hi = col[hi_idx];
        zero[d] = lo;
        scale[d] = if hi > lo { (hi - lo) / 15.0 } else { 1.0 };
    }
    (scale, zero)
}

fn pack_4bit(rotated: &[f32], scale: &[f32], zero: &[f32]) -> Vec<u8> {
    let dim = rotated.len();
    let bytes = (dim + 1) / 2;
    let mut code = vec![0u8; bytes];
    for i in 0..dim {
        let q = ((rotated[i] - zero[i]) / scale[i])
            .round()
            .clamp(0.0, 15.0) as u8;
        if i % 2 == 0 {
            code[i / 2] = q & 0xF;
        } else {
            code[i / 2] |= (q & 0xF) << 4;
        }
    }
    code
}

fn dot_dequant(rq: &[f32], code: &[u8], scale: &[f32], zero: &[f32], dim: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..dim {
        let q = if i % 2 == 0 {
            code[i / 2] & 0xF
        } else {
            (code[i / 2] >> 4) & 0xF
        };
        let v = q as f32 * scale[i] + zero[i];
        sum += rq[i] * v;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_recall() {
        let dim = 512usize;
        let n = 100usize;
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        let vecs: Vec<Vec<f32>> = (0..n).map(|_| rand_unit_vec(dim, &mut rng)).collect();
        let labels: Vec<String> = (0..n).map(|i| format!("vec_{i}")).collect();
        let slices: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let qi = QuantizedIndex::build(&labels, &slices);
        assert_eq!(qi.len(), n);
        let top1 = qi.nearest(&vecs[0], 1);
        assert_eq!(top1.len(), 1);
        assert_eq!(top1[0].0, "vec_0");
    }

    #[test]
    fn test_recall_at_1_rate() {
        let dim = 512usize;
        let n = 500usize;
        let mut rng = 0xDEAD_BEEF_1234_5678u64;
        let vecs: Vec<Vec<f32>> = (0..n).map(|_| rand_unit_vec(dim, &mut rng)).collect();
        let labels: Vec<String> = (0..n).map(|i| format!("v{i}")).collect();
        let slices: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let qi = QuantizedIndex::build(&labels, &slices);
        let mut hits = 0usize;
        for (i, v) in vecs.iter().enumerate() {
            let top1 = qi.nearest(v, 1);
            if top1.first().map(|(l, _)| l.as_str()) == Some(&labels[i]) {
                hits += 1;
            }
        }
        let rate = hits as f32 / n as f32;
        assert!(rate > 0.95, "recall@1 = {rate:.3} (expected >0.95)");
    }

    #[test]
    fn test_insert_dynamic() {
        let dim = 64usize;
        let mut rng = 0xABCD_EF01u64;
        let v0 = rand_unit_vec(dim, &mut rng);
        let v1 = rand_unit_vec(dim, &mut rng);
        let mut qi = QuantizedIndex::build(&[String::from("first")], &[v0.as_slice()]);
        assert_eq!(qi.len(), 1);
        qi.insert("second", &v1);
        assert_eq!(qi.len(), 2);
        let top1 = qi.nearest(&v1, 1);
        assert_eq!(top1[0].0, "second");
    }

    #[test]
    fn test_empty_index() {
        let qi = QuantizedIndex::build(&[], &[]);
        assert!(qi.is_empty());
        let result = qi.nearest(&vec![0.0f32; 512], 5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_wrong_dim_insert() {
        let dim = 8usize;
        let mut qi = QuantizedIndex::build(&[String::from("a")], &[&vec![0.1f32; dim]]);
        qi.insert("bad", &vec![0.1f32; dim + 1]);
        assert_eq!(qi.len(), 1);
    }

    fn rand_unit_vec(dim: usize, state: &mut u64) -> Vec<f32> {
        let v: Vec<f32> = (0..dim)
            .map(|_| {
                *state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bits = (*state >> 33) as u32;
                (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        v.iter().map(|x| x / norm).collect()
    }
}
