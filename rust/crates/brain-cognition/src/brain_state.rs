//! Brain state composition — assembles a 512-dim cognitive vector from live signals.
//!
//! Called before each `generate_grounded()` call to give HOPE a live snapshot
//! of what the Brain is currently experiencing.

use crate::working_memory::WorkingMemory;
use crate::fast_memory::HopfieldMemory;
use crate::concepts::ConceptCodebook;
use std::path::Path;

/// Emotion name → index mapping (must match `generate_emotion_table.py`).
pub fn emotion_to_idx(emotion: &str) -> usize {
    match emotion {
        "neutral"  => 0,
        "sad"      => 1,
        "pain"     => 2,
        "happy"    => 3,
        "fearful"  => 4,
        "angry"    => 5,
        "confused" => 6,
        "tired"    => 7,
        _          => 0, // unknown → neutral
    }
}

/// Compose a 512-dim Brain state vector from live cognitive signals.
///
/// Components and weights:
///   - Working memory centroid (0.35): average of active WM item embeddings
///   - Fast memory retrieval (0.25):   top-1 Hopfield pattern for WM centroid
///   - Concept centroid (0.25):        nearest concept centroid to WM centroid
///   - Emotion embedding (0.15):       row from emotion_table
///
/// Missing components have their weight redistributed proportionally.
/// Output is L2-normalized to unit length.
/// Returns a zero vector if all signals are unavailable.
pub fn compose_brain_state(
    wm: &WorkingMemory,
    fm: &HopfieldMemory,
    codebook: Option<&ConceptCodebook>,
    emotion: &str,
    emotion_table: &[[f32; 512]],
) -> Vec<f32> {
    const DIM: usize = 512;

    // 1. Working memory centroid: average of 512-dim WM embeddings
    let wm_embs: Vec<&[f32]> = wm.get_embeddings()
        .into_iter()
        .filter(|e| e.len() == DIM)
        .collect();

    let wm_centroid: Option<Vec<f32>> = if wm_embs.is_empty() {
        None
    } else {
        let mut sum = vec![0.0f32; DIM];
        for e in &wm_embs {
            for (s, v) in sum.iter_mut().zip(*e) { *s += v; }
        }
        Some(l2_normalize(&sum))
    };

    // 2. Fast memory retrieval: top-1 Hopfield pattern for WM centroid
    let fm_vec: Option<Vec<f32>> = wm_centroid.as_ref().and_then(|wm_c| {
        fm.retrieve(wm_c, 1)
            .first()
            .and_then(|m| fm.pattern_at(m.idx))
            .map(|p| l2_normalize(p))
    });

    // 3. Concept centroid: nearest concept to WM centroid
    let concept_vec: Option<Vec<f32>> = wm_centroid.as_ref().and_then(|wm_c| {
        codebook.and_then(|cb| cb.top1_centroid(wm_c))
    });

    // 4. Emotion embedding: fixed lookup table row
    let emotion_idx = emotion_to_idx(emotion);
    let emotion_vec: Option<&[f32; 512]> = emotion_table.get(emotion_idx);

    // Weighted sum with proportional redistribution for missing components
    let mut result = vec![0.0f32; DIM];
    let mut total_weight = 0.0f32;

    if let Some(ref v) = wm_centroid {
        let w = 0.35;
        for (r, x) in result.iter_mut().zip(v) { *r += w * x; }
        total_weight += w;
    }
    if let Some(ref v) = fm_vec {
        let w = 0.25;
        for (r, x) in result.iter_mut().zip(v) { *r += w * x; }
        total_weight += w;
    }
    if let Some(ref v) = concept_vec {
        let w = 0.25;
        for (r, x) in result.iter_mut().zip(v) { *r += w * x; }
        total_weight += w;
    }
    if let Some(ev) = emotion_vec {
        let w = 0.15;
        for (r, x) in result.iter_mut().zip(ev.iter()) { *r += w * x; }
        total_weight += w;
    }

    if total_weight < 1e-12 {
        return vec![0.0f32; DIM]; // all signals unavailable
    }

    // Normalize by total weight (equivalent to proportional redistribution)
    for r in &mut result { *r /= total_weight; }
    l2_normalize(&result)
}

/// Load emotion table from `emotion_table.bin` (raw f32 LE, 8×512 row-major).
/// Returns 8 zero rows if the file is missing or malformed — HOPE falls back to text-only.
pub fn load_emotion_table(path: &Path) -> Vec<[f32; 512]> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(_) => return vec![[0.0f32; 512]; 8],
    };
    let expected = 8 * 512 * 4;
    if bytes.len() < expected {
        tracing::warn!("emotion_table.bin too small ({} bytes, need {})", bytes.len(), expected);
        return vec![[0.0f32; 512]; 8];
    }
    let mut table = Vec::with_capacity(8);
    for i in 0..8 {
        let mut row = [0.0f32; 512];
        for j in 0..512 {
            let off = (i * 512 + j) * 4;
            row[j] = f32::from_le_bytes([bytes[off], bytes[off+1], bytes[off+2], bytes[off+3]]);
        }
        table.push(row);
    }
    table
}

fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter().map(|x| x / norm).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::working_memory::WorkingMemory;
    use crate::fast_memory::HopfieldMemory;

    fn zero_table() -> Vec<[f32; 512]> {
        vec![[0.0f32; 512]; 8]
    }

    fn unit_table() -> Vec<[f32; 512]> {
        let mut table = vec![[0.0f32; 512]; 8];
        for (i, row) in table.iter_mut().enumerate() {
            row[i] = 1.0; // each emotion: unit vector in its own dimension
        }
        table
    }

    #[test]
    fn test_empty_memory_returns_zeros() {
        let wm = WorkingMemory::new(7, 0.85, 0.15);
        let fm = HopfieldMemory::new(512, 100);
        let result = compose_brain_state(&wm, &fm, None, "neutral", &zero_table());
        assert_eq!(result.len(), 512);
        assert!(result.iter().all(|&x| x == 0.0), "Expected zero vector");
    }

    #[test]
    fn test_emotion_only_produces_unit_vector() {
        let wm = WorkingMemory::new(7, 0.85, 0.15);
        let fm = HopfieldMemory::new(512, 100);
        let table = unit_table();
        // "sad" = index 1 → table[1] = unit vector in dim 1
        let result = compose_brain_state(&wm, &fm, None, "sad", &table);
        assert_eq!(result.len(), 512);
        // With zero wm/fm/concept: only emotion contributes (weight 0.15/0.15 = 1.0 after normalize)
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm={norm}");
    }

    #[test]
    fn test_wm_item_contributes() {
        let mut wm = WorkingMemory::new(7, 0.85, 0.15);
        let emb: Vec<f32> = (0..512).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
        wm.update(emb, "test".into(), "audio".into());
        let fm = HopfieldMemory::new(512, 100);
        let result = compose_brain_state(&wm, &fm, None, "neutral", &zero_table());
        assert_eq!(result.len(), 512);
        assert!(result[0] > 0.0, "Expected WM contribution in dim 0");
    }

    #[test]
    fn test_output_is_unit_norm_when_nonempty() {
        let mut wm = WorkingMemory::new(7, 0.85, 0.15);
        let emb: Vec<f32> = (0..512).map(|i| i as f32 / 512.0).collect();
        wm.update(emb, "test".into(), "audio".into());
        let fm = HopfieldMemory::new(512, 100);
        let result = compose_brain_state(&wm, &fm, None, "neutral", &zero_table());
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm={norm}");
    }

    #[test]
    fn test_emotion_to_idx_unknown_maps_to_neutral() {
        assert_eq!(emotion_to_idx("blorp"), 0);
        assert_eq!(emotion_to_idx("sad"), 1);
        assert_eq!(emotion_to_idx("tired"), 7);
    }
}
