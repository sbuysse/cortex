//! Association Network — the top-level Associative Core module.
//!
//! Composes Hebbian associations + temporal traces connecting up to 5 modalities:
//! vision (v), audio (a), emotion (e), speech (s), properties (p).
//!
//! With all 5 modalities active: C(5,2) = 10 association matrices.
//! Gracefully degrades when optional modalities (s, p) are absent.

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::hebbian::HebbianAssociation;
use crate::temporal_trace::TemporalTrace;

/// Modality names.
pub const MODALITIES: [&str; 5] = ["vision", "audio", "emotion", "speech", "properties"];

/// All possible modality pairs (10 total).
const PAIR_KEYS: [(&str, &str, &str); 10] = [
    ("M_va", "vision", "audio"),
    ("M_ve", "vision", "emotion"),
    ("M_ae", "audio", "emotion"),
    ("M_vs", "vision", "speech"),
    ("M_as", "audio", "speech"),
    ("M_es", "emotion", "speech"),
    ("M_vp", "vision", "properties"),
    ("M_ap", "audio", "properties"),
    ("M_ep", "emotion", "properties"),
    ("M_sp", "speech", "properties"),
];

/// Configuration for the association network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssociationConfig {
    pub d: usize,
    pub lr: f32,
    pub decay_rate: f32,
    pub temporal_decay: f32,
    pub trace_weight: f32,
    pub max_norm: f32,
}

impl Default for AssociationConfig {
    fn default() -> Self {
        Self {
            d: 512,
            lr: 0.01,
            decay_rate: 0.999,
            temporal_decay: 0.95,
            trace_weight: 0.2,
            max_norm: 100.0,
        }
    }
}

/// Hebbian association network connecting up to 5 modalities.
pub struct AssociationNetwork {
    pub d: usize,
    pub trace_weight: f32,

    // Core 3 Hebbian association matrices (always active)
    pub m_va: HebbianAssociation,
    pub m_ve: HebbianAssociation,
    pub m_ae: HebbianAssociation,

    // Extended association matrices (active when speech/properties available)
    pub m_vs: HebbianAssociation,
    pub m_as: HebbianAssociation,
    pub m_es: HebbianAssociation,
    pub m_vp: HebbianAssociation,
    pub m_ap: HebbianAssociation,
    pub m_ep: HebbianAssociation,
    pub m_sp: HebbianAssociation,

    // Temporal traces (one per modality)
    pub visual_trace: TemporalTrace,
    pub audio_trace: TemporalTrace,
    pub emotion_trace: TemporalTrace,
    pub speech_trace: TemporalTrace,
    pub properties_trace: TemporalTrace,

    // Pre-allocated blend buffers
    blend_v: Array2<f32>,
    blend_a: Array2<f32>,
    blend_e: Array2<f32>,
    blend_s: Array2<f32>,
    blend_p: Array2<f32>,
    blend_batch_size: usize,
}

impl AssociationNetwork {
    pub fn new(config: &AssociationConfig) -> Self {
        let d = config.d;
        let lr = config.lr;
        let dr = config.decay_rate;
        let mn = config.max_norm;
        let td = config.temporal_decay;

        Self {
            d,
            trace_weight: config.trace_weight,
            m_va: HebbianAssociation::new(d, lr, dr, mn),
            m_ve: HebbianAssociation::new(d, lr, dr, mn),
            m_ae: HebbianAssociation::new(d, lr, dr, mn),
            m_vs: HebbianAssociation::new(d, lr, dr, mn),
            m_as: HebbianAssociation::new(d, lr, dr, mn),
            m_es: HebbianAssociation::new(d, lr, dr, mn),
            m_vp: HebbianAssociation::new(d, lr, dr, mn),
            m_ap: HebbianAssociation::new(d, lr, dr, mn),
            m_ep: HebbianAssociation::new(d, lr, dr, mn),
            m_sp: HebbianAssociation::new(d, lr, dr, mn),
            visual_trace: TemporalTrace::new(d, td),
            audio_trace: TemporalTrace::new(d, td),
            emotion_trace: TemporalTrace::new(d, td),
            speech_trace: TemporalTrace::new(d, td),
            properties_trace: TemporalTrace::new(d, td),
            blend_v: Array2::zeros((0, 0)),
            blend_a: Array2::zeros((0, 0)),
            blend_e: Array2::zeros((0, 0)),
            blend_s: Array2::zeros((0, 0)),
            blend_p: Array2::zeros((0, 0)),
            blend_batch_size: 0,
        }
    }

    /// Run one Hebbian learning step with all available modalities.
    #[inline]
    pub fn forward(
        &mut self,
        v_proj: &Array2<f32>,
        a_proj: &Array2<f32>,
        e_proj: Option<&Array2<f32>>,
    ) {
        self.forward_all(v_proj, a_proj, e_proj, None, None);
    }

    /// Run one Hebbian learning step with up to 5 modalities.
    ///
    /// Updates temporal traces, blends with trace_weight, then
    /// performs Hebbian updates on all applicable pairs.
    pub fn forward_all(
        &mut self,
        v_proj: &Array2<f32>,
        a_proj: &Array2<f32>,
        e_proj: Option<&Array2<f32>>,
        s_proj: Option<&Array2<f32>>,
        p_proj: Option<&Array2<f32>>,
    ) {
        let batch_size = v_proj.nrows();
        let d = self.d;
        let tw = self.trace_weight;

        // Resize blend buffers if needed
        if self.blend_batch_size != batch_size {
            self.blend_v = Array2::zeros((batch_size, d));
            self.blend_a = Array2::zeros((batch_size, d));
            self.blend_e = Array2::zeros((batch_size, d));
            self.blend_s = Array2::zeros((batch_size, d));
            self.blend_p = Array2::zeros((batch_size, d));
            self.blend_batch_size = batch_size;
        }

        // Core modalities (always active)
        let v_traced = self.visual_trace.update(v_proj);
        let a_traced = self.audio_trace.update(a_proj);
        blend_into(v_proj, v_traced, tw, &mut self.blend_v);
        blend_into(a_proj, a_traced, tw, &mut self.blend_a);
        self.m_va.update(&self.blend_v, &self.blend_a);

        // Emotion (optional)
        if let Some(e) = e_proj {
            let e_traced = self.emotion_trace.update(e);
            blend_into(e, e_traced, tw, &mut self.blend_e);
            self.m_ve.update(&self.blend_v, &self.blend_e);
            self.m_ae.update(&self.blend_a, &self.blend_e);
        }

        // Speech (optional)
        if let Some(s) = s_proj {
            let s_traced = self.speech_trace.update(s);
            blend_into(s, s_traced, tw, &mut self.blend_s);
            self.m_vs.update(&self.blend_v, &self.blend_s);
            self.m_as.update(&self.blend_a, &self.blend_s);
            if e_proj.is_some() {
                self.m_es.update(&self.blend_e, &self.blend_s);
            }
        }

        // Properties (optional)
        if let Some(p) = p_proj {
            let p_traced = self.properties_trace.update(p);
            blend_into(p, p_traced, tw, &mut self.blend_p);
            self.m_vp.update(&self.blend_v, &self.blend_p);
            self.m_ap.update(&self.blend_a, &self.blend_p);
            if e_proj.is_some() {
                self.m_ep.update(&self.blend_e, &self.blend_p);
            }
            if s_proj.is_some() {
                self.m_sp.update(&self.blend_s, &self.blend_p);
            }
        }
    }

    /// Pattern completion: recall target modality from source.
    pub fn recall(
        &self,
        query: &Array2<f32>,
        source: &str,
        target: &str,
    ) -> Option<Array2<f32>> {
        if source == target {
            return Some(query.clone());
        }

        // Direct path lookup
        if let Some((assoc, forward)) = self.find_direct_path(source, target) {
            return Some(assoc.recall(query, forward));
        }

        // Multi-hop via intermediate modality
        for &mid in &MODALITIES {
            if mid == source || mid == target {
                continue;
            }
            if self.find_direct_path(source, mid).is_some()
                && self.find_direct_path(mid, target).is_some()
            {
                let mid_result = self.recall(query, source, mid)?;
                return self.recall(&mid_result, mid, target);
            }
        }

        None
    }

    /// Set learning rate on all association matrices.
    #[inline]
    pub fn set_lr(&mut self, lr: f32) {
        self.m_va.lr = lr;
        self.m_ve.lr = lr;
        self.m_ae.lr = lr;
        self.m_vs.lr = lr;
        self.m_as.lr = lr;
        self.m_es.lr = lr;
        self.m_vp.lr = lr;
        self.m_ap.lr = lr;
        self.m_ep.lr = lr;
        self.m_sp.lr = lr;
    }

    /// Prune all matrices.
    pub fn prune(&mut self, threshold: f32) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("M_va".to_string(), self.m_va.prune(threshold));
        stats.insert("M_ve".to_string(), self.m_ve.prune(threshold));
        stats.insert("M_ae".to_string(), self.m_ae.prune(threshold));
        stats.insert("M_vs".to_string(), self.m_vs.prune(threshold));
        stats.insert("M_as".to_string(), self.m_as.prune(threshold));
        stats.insert("M_es".to_string(), self.m_es.prune(threshold));
        stats.insert("M_vp".to_string(), self.m_vp.prune(threshold));
        stats.insert("M_ap".to_string(), self.m_ap.prune(threshold));
        stats.insert("M_ep".to_string(), self.m_ep.prune(threshold));
        stats.insert("M_sp".to_string(), self.m_sp.prune(threshold));
        stats
    }

    /// Reset all temporal traces.
    pub fn reset_traces(&mut self) {
        self.visual_trace.reset();
        self.audio_trace.reset();
        self.emotion_trace.reset();
        self.speech_trace.reset();
        self.properties_trace.reset();
    }

    /// Aggregate statistics from all components.
    pub fn get_all_stats(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();
        for (name, assoc) in self.all_matrices() {
            stats.insert(name.to_string(), serde_json::to_value(assoc.get_stats()).unwrap());
        }
        let traces = serde_json::json!({
            "vision": {"norm": self.visual_trace.norm(), "mean": self.visual_trace.mean()},
            "audio": {"norm": self.audio_trace.norm(), "mean": self.audio_trace.mean()},
            "emotion": {"norm": self.emotion_trace.norm(), "mean": self.emotion_trace.mean()},
            "speech": {"norm": self.speech_trace.norm(), "mean": self.speech_trace.mean()},
            "properties": {"norm": self.properties_trace.norm(), "mean": self.properties_trace.mean()},
        });
        stats.insert("traces".to_string(), traces);
        stats
    }

    /// Iterator over all (name, ref) pairs.
    fn all_matrices(&self) -> Vec<(&str, &HebbianAssociation)> {
        vec![
            ("M_va", &self.m_va), ("M_ve", &self.m_ve), ("M_ae", &self.m_ae),
            ("M_vs", &self.m_vs), ("M_as", &self.m_as), ("M_es", &self.m_es),
            ("M_vp", &self.m_vp), ("M_ap", &self.m_ap), ("M_ep", &self.m_ep),
            ("M_sp", &self.m_sp),
        ]
    }

    /// Mutable access to all matrices for initialization.
    pub fn all_matrices_mut(&mut self) -> Vec<(&str, &mut Array2<f32>)> {
        vec![
            ("M_va", &mut self.m_va.m), ("M_ve", &mut self.m_ve.m), ("M_ae", &mut self.m_ae.m),
            ("M_vs", &mut self.m_vs.m), ("M_as", &mut self.m_as.m), ("M_es", &mut self.m_es.m),
            ("M_vp", &mut self.m_vp.m), ("M_ap", &mut self.m_ap.m), ("M_ep", &mut self.m_ep.m),
            ("M_sp", &mut self.m_sp.m),
        ]
    }

    /// Find a direct association between two modalities.
    fn find_direct_path(&self, source: &str, target: &str) -> Option<(&HebbianAssociation, bool)> {
        for &(name, mod_a, mod_b) in &PAIR_KEYS {
            let assoc = match name {
                "M_va" => &self.m_va, "M_ve" => &self.m_ve, "M_ae" => &self.m_ae,
                "M_vs" => &self.m_vs, "M_as" => &self.m_as, "M_es" => &self.m_es,
                "M_vp" => &self.m_vp, "M_ap" => &self.m_ap, "M_ep" => &self.m_ep,
                "M_sp" => &self.m_sp,
                _ => unreachable!(),
            };
            if source == mod_a && target == mod_b {
                return Some((assoc, true));
            }
            if source == mod_b && target == mod_a {
                return Some((assoc, false));
            }
        }
        None
    }
}

/// Blend activations with trace into pre-allocated output buffer.
#[inline]
fn blend_into(proj: &Array2<f32>, traced: &Array2<f32>, tw: f32, output: &mut Array2<f32>) {
    let otw = 1.0 - tw;
    let traced_slice = traced.as_slice().unwrap();
    let proj_slice = proj.as_slice().unwrap();
    let out_slice = output.as_slice_mut().unwrap();
    let cols = proj.ncols();
    let rows = proj.nrows();

    for r in 0..rows {
        let row_off = r * cols;
        for j in 0..cols {
            out_slice[row_off + j] = otw * proj_slice[row_off + j] + tw * traced_slice[j];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_basic() {
        let config = AssociationConfig::default();
        let mut net = AssociationNetwork::new(&config);
        let v = Array2::from_shape_fn((4, 512), |_| 0.1);
        let a = Array2::from_shape_fn((4, 512), |_| 0.1);
        net.forward(&v, &a, None);
        assert!(net.m_va.m.iter().any(|&x| x != 0.0));
        assert_eq!(net.m_va.update_count, 1);
    }

    #[test]
    fn test_forward_with_emotion() {
        let config = AssociationConfig::default();
        let mut net = AssociationNetwork::new(&config);
        let v = Array2::from_shape_fn((4, 512), |_| 0.1);
        let a = Array2::from_shape_fn((4, 512), |_| 0.1);
        let e = Array2::from_shape_fn((4, 512), |_| 0.1);
        net.forward(&v, &a, Some(&e));
        assert!(net.m_ve.m.iter().any(|&x| x != 0.0));
        assert!(net.m_ae.m.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_forward_all_modalities() {
        let config = AssociationConfig::default();
        let mut net = AssociationNetwork::new(&config);
        let data = Array2::from_shape_fn((4, 512), |_| 0.1);
        net.forward_all(&data, &data, Some(&data), Some(&data), Some(&data));
        // All 10 matrices should be updated
        assert_eq!(net.m_va.update_count, 1);
        assert_eq!(net.m_sp.update_count, 1);
    }

    #[test]
    fn test_recall() {
        let config = AssociationConfig::default();
        let mut net = AssociationNetwork::new(&config);
        net.m_va.m = Array2::eye(512);
        let query = Array2::from_shape_fn((2, 512), |(_, j)| if j == 0 { 1.0 } else { 0.0 });
        let result = net.recall(&query, "vision", "audio").unwrap();
        assert_eq!(result.dim(), (2, 512));
    }
}
