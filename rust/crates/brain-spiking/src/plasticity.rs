use crate::synapse::{weight_to_i16, weight_from_i16};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlasticityParams {
    pub a_plus: f32,
    pub a_minus: f32,
    pub tau_plus: f32,
    pub tau_minus: f32,
    pub eligibility_decay: f32,
    pub tacos_lambda: f32,
    pub tacos_alpha: f32,
}

impl Default for PlasticityParams {
    fn default() -> Self {
        Self {
            a_plus: 0.01,
            a_minus: -0.012,
            tau_plus: 20.0,
            tau_minus: 20.0,
            eligibility_decay: 0.99,
            tacos_lambda: 0.1,
            tacos_alpha: 0.01,
        }
    }
}

/// Classic STDP window.
/// `dt` = t_post - t_pre. Negative = pre-before-post = LTP.
pub fn update_stdp(dt: f32, params: &PlasticityParams) -> f32 {
    if dt < 0.0 {
        params.a_plus * (dt / params.tau_plus).exp()
    } else if dt > 0.0 {
        params.a_minus * (-dt / params.tau_minus).exp()
    } else {
        0.0
    }
}

/// TACOS dual-weight update.
/// w moves toward w_ref (heterosynaptic decay).
/// w_ref slowly tracks w (consolidation).
pub fn update_tacos(w: &mut i16, w_ref: &mut i16, lambda: f32, alpha: f32) {
    let wf = weight_from_i16(*w);
    let rf = weight_from_i16(*w_ref);
    let new_w = wf + lambda * (rf - wf);
    *w = weight_to_i16(new_w);
    let new_ref = rf + alpha * (wf - rf);
    *w_ref = weight_to_i16(new_ref);
}

/// Update eligibility trace for a synapse.
#[inline]
pub fn update_eligibility(eligibility: &mut i16, stdp_delta: f32) {
    let e = weight_from_i16(*eligibility) + stdp_delta;
    *eligibility = weight_to_i16(e.clamp(-1.0, 1.0));
}

/// Decay all eligibility traces in a CSR by the decay factor.
pub fn decay_eligibilities(eligibilities: &mut [i16], decay: f32) {
    for e in eligibilities.iter_mut() {
        let ef = weight_from_i16(*e) * decay;
        *e = weight_to_i16(ef);
    }
}

/// Apply three-factor learning: dw = lr * eligibility * modulator.
pub fn apply_three_factor(
    weights: &mut [i16],
    eligibilities: &[i16],
    learning_rate: f32,
    modulator: f32,
) {
    for (w, e) in weights.iter_mut().zip(eligibilities.iter()) {
        let ef = weight_from_i16(*e);
        if ef.abs() > 0.001 {
            let wf = weight_from_i16(*w);
            let dw = learning_rate * ef * modulator;
            *w = weight_to_i16((wf + dw).clamp(-1.0, 1.0));
        }
    }
}
