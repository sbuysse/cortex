use brain_spiking::neuron::NeuronArray;
use brain_spiking::config::NeuronParams;

#[test]
fn test_alif_subthreshold_no_spike() {
    let params = NeuronParams::default();
    let mut neurons = NeuronArray::new(4, &params);
    neurons.set_current(0, 0.3);
    let spikes = neurons.step();
    assert_eq!(spikes.count(), 0, "subthreshold current should not cause spike");
}

#[test]
fn test_alif_suprathreshold_spike() {
    let params = NeuronParams::default();
    let mut neurons = NeuronArray::new(4, &params);
    for _ in 0..50 {
        neurons.set_current(0, 2.0);
        let spikes = neurons.step();
        if spikes.count() > 0 {
            assert!(spikes.fired(0), "neuron 0 should fire");
            assert!(neurons.voltage(0) < params.v_threshold);
            return;
        }
    }
    panic!("neuron should have spiked within 50 steps at 2.0 current");
}

#[test]
fn test_alif_adaptation_reduces_firing() {
    let params = NeuronParams::default();
    let mut neurons = NeuronArray::new(1, &params);
    let mut spike_count_first = 0;
    for _ in 0..100 {
        neurons.set_current(0, 2.0);
        spike_count_first += neurons.step().count();
    }
    let mut spike_count_second = 0;
    for _ in 0..100 {
        neurons.set_current(0, 2.0);
        spike_count_second += neurons.step().count();
    }
    assert!(
        spike_count_second <= spike_count_first,
        "adaptation should reduce firing: first={spike_count_first}, second={spike_count_second}"
    );
}

#[test]
fn test_neuron_array_batch_1000() {
    let params = NeuronParams::default();
    let mut neurons = NeuronArray::new(1000, &params);
    let mut total_spikes = 0;
    for _ in 0..50 {
        for i in (0..1000).step_by(3) {
            neurons.set_current(i, 2.0);
        }
        total_spikes += neurons.step().count();
    }
    assert!(total_spikes > 0, "some neurons should have fired");
    assert!(total_spikes < 334 * 50, "not every stimulated neuron fires every step");
}
