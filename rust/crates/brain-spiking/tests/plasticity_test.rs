use brain_spiking::plasticity::{PlasticityParams, update_stdp, update_tacos};
use brain_spiking::synapse::{weight_to_i16, weight_from_i16};

#[test]
fn test_stdp_potentiation() {
    let params = PlasticityParams::default();
    let dt_spikes = -5.0_f32;
    let dw = update_stdp(dt_spikes, &params);
    assert!(dw > 0.0, "pre-before-post should potentiate: dw={dw}");
}

#[test]
fn test_stdp_depression() {
    let params = PlasticityParams::default();
    let dt_spikes = 5.0_f32;
    let dw = update_stdp(dt_spikes, &params);
    assert!(dw < 0.0, "post-before-pre should depress: dw={dw}");
}

#[test]
fn test_stdp_larger_dt_weaker() {
    let params = PlasticityParams::default();
    let dw_close = update_stdp(-2.0, &params);
    let dw_far = update_stdp(-20.0, &params);
    assert!(dw_close > dw_far, "closer spikes should have stronger effect");
}

#[test]
fn test_three_factor_modulation() {
    let params = PlasticityParams::default();
    let stdp = update_stdp(-5.0, &params);
    let dw_high_da = stdp * 2.0;
    let dw_low_da = stdp * 0.1;
    assert!(dw_high_da > dw_low_da);
}

#[test]
fn test_tacos_consolidation() {
    let mut w = weight_to_i16(0.5);
    let mut w_ref = weight_to_i16(0.3);
    update_tacos(&mut w, &mut w_ref, 0.1, 0.01);
    let new_w = weight_from_i16(w);
    assert!(new_w < 0.5, "TACOS should pull w toward w_ref: new_w={new_w}");
    let new_ref = weight_from_i16(w_ref);
    assert!(new_ref > 0.3, "w_ref should slowly track w: new_ref={new_ref}");
}
