use brain_spiking::spike_encoder::LatencyEncoder;
use brain_spiking::regions::association::build_association_cortex;

#[test]
fn test_association_cortex_cross_modal_binding() {
    let mut net = build_association_cortex(100, 0.1);
    let visual_encoder = LatencyEncoder::new(50, 20);
    let audio_encoder = LatencyEncoder::new(50, 20);

    let visual_emb: Vec<f32> = (0..50).map(|i| ((i as f32 * 0.1).sin() + 1.0) / 2.0).collect();
    let audio_emb: Vec<f32> = (0..50).map(|i| ((i as f32 * 0.1).sin() + 1.0) / 2.0).collect();

    let v_spikes = visual_encoder.encode(&visual_emb);
    let a_spikes = audio_encoder.encode(&audio_emb);

    for step in 0..30_u16 {
        for (i, &t) in v_spikes.iter().enumerate() {
            if t == step { net.inject_current(0, i, 3.0); }
        }
        for (i, &t) in a_spikes.iter().enumerate() {
            if t == step { net.inject_current(0, 50 + i, 3.0); }
        }
        net.step();
    }

    let stats = net.stats();
    assert!(stats.total_neurons == 100);
    assert!(stats.total_synapses > 0);
}

#[test]
fn test_association_cortex_scales_to_500() {
    let net = build_association_cortex(500, 0.05);
    let stats = net.stats();
    assert_eq!(stats.total_neurons, 500);
    assert!(stats.total_synapses > 0);
}
