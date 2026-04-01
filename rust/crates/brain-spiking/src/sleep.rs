use crate::network::SpikingNetwork;
use crate::synapse::{weight_from_i16, weight_to_i16};

pub fn nrem_consolidation(net: &mut SpikingNetwork, steps: usize, weight_decay: f32) {
    for region_id in 0..net.num_regions() {
        if let Some(synapses) = net.region_mut(region_id).synapses_mut() {
            for i in 0..synapses.weights.len() {
                let eligibility = weight_from_i16(synapses.eligibilities[i]).abs();
                let effective_decay = if eligibility > 0.3 {
                    1.0 - (1.0 - weight_decay) * 0.1
                } else {
                    weight_decay
                };
                let w = weight_from_i16(synapses.weights[i]) * effective_decay;
                synapses.weights[i] = weight_to_i16(w);
            }
        }
    }
    for _ in 0..steps { net.step(); }
}

pub fn rem_consolidation(net: &mut SpikingNetwork, steps: usize, weight_decay: f32) {
    use rand::Rng;
    let mut rng = rand::rng();

    for region_id in 0..net.num_regions() {
        if let Some(synapses) = net.region_mut(region_id).synapses_mut() {
            for w in synapses.weights.iter_mut() {
                let wf = weight_from_i16(*w) * weight_decay;
                *w = weight_to_i16(wf);
            }
        }
    }

    for _ in 0..steps {
        for region_id in 0..net.num_regions() {
            let n = net.region(region_id).num_neurons();
            for _ in 0..n.min(10) {
                let idx = rng.random_range(0..n);
                let noise = rng.random_range(-0.5..1.0_f32);
                net.inject_current(region_id, idx, noise);
            }
        }
        net.step();
    }
}

pub fn sleep_cycle(net: &mut SpikingNetwork, total_steps: usize) {
    let nrem_steps = (total_steps * 6) / 10;
    let rem_steps = total_steps - nrem_steps;
    tracing::info!("Sleep cycle: NREM {nrem_steps} steps, REM {rem_steps} steps");
    nrem_consolidation(net, nrem_steps, 0.98);
    rem_consolidation(net, rem_steps, 0.95);
    tracing::info!("Sleep cycle complete");
}
