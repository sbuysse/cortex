use brain_spiking::spike_encoder::LatencyEncoder;
use brain_spiking::spike_decoder::RateDecoder;

#[test]
fn test_latency_encode_high_value_spikes_first() {
    let encoder = LatencyEncoder::new(4, 20);
    let embedding = vec![0.9, 0.1, 0.5, 0.0];
    let spike_times = encoder.encode(&embedding);
    assert!(spike_times[0] < spike_times[1], "0.9 should spike before 0.1");
    assert!(spike_times[2] < spike_times[1], "0.5 should spike before 0.1");
    assert_eq!(spike_times[3], 20, "0.0 spikes at T_MAX");
}

#[test]
fn test_latency_encode_dim_512() {
    let encoder = LatencyEncoder::new(512, 20);
    let embedding: Vec<f32> = (0..512).map(|i| i as f32 / 512.0).collect();
    let spike_times = encoder.encode(&embedding);
    assert_eq!(spike_times.len(), 512);
    assert!(spike_times[500] < spike_times[10]);
}

#[test]
fn test_rate_decode() {
    let mut decoder = RateDecoder::new(4, 50);
    decoder.record_spike(0, 5);
    decoder.record_spike(0, 15);
    decoder.record_spike(0, 25);
    decoder.record_spike(1, 10);
    let embedding = decoder.decode();
    assert_eq!(embedding.len(), 4);
    assert!(embedding[0] > embedding[1], "neuron 0 fired more -> higher value");
    assert!(embedding[1] > embedding[2], "neuron 1 fired -> higher than silent");
    assert_eq!(embedding[2], 0.0, "neuron 2 never fired");
}
