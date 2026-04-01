use crate::config::RegionConfig;
use crate::neuron::NeuronArray;
use crate::synapse::{SynapseBuilder, SynapseCSR};

/// A brain region: a population of neurons with internal connectivity.
pub struct BrainRegion {
    config: RegionConfig,
    neurons: NeuronArray,
    synapses: Option<SynapseCSR>,
    builder: Option<SynapseBuilder>,
    last_spikes: Vec<usize>,
}

impl BrainRegion {
    pub fn new(config: RegionConfig) -> Self {
        let n = config.total_neurons();
        let neurons = NeuronArray::new(n, &config.neuron_params);
        Self {
            builder: Some(SynapseBuilder::new(n)),
            config,
            neurons,
            synapses: None,
            last_spikes: Vec::new(),
        }
    }

    pub fn name(&self) -> &str { &self.config.name }
    pub fn num_neurons(&self) -> usize { self.config.total_neurons() }
    pub fn num_excitatory(&self) -> usize { self.config.num_excitatory }

    /// Add a synapse during construction. Panics if already finalized.
    pub fn connect(&mut self, src: usize, tgt: usize, weight: f32, delay: u8) {
        if let Some(ref mut builder) = self.builder {
            builder.add(src, tgt, weight, delay);
        } else {
            panic!("cannot add connections after finalize()");
        }
    }

    /// Freeze connectivity from COO to CSR. Must be called before stepping.
    pub fn finalize(&mut self) {
        if let Some(builder) = self.builder.take() {
            self.synapses = Some(builder.freeze());
        }
    }

    /// Inject external current into a specific neuron.
    #[inline]
    pub fn inject_current(&mut self, idx: usize, current: f32) {
        self.neurons.add_current(idx, current);
    }

    /// Advance one timestep. Returns indices of neurons that fired.
    pub fn step(&mut self) -> &[usize] {
        // Auto-finalize on first step
        if self.synapses.is_none() && self.builder.is_some() {
            self.finalize();
        }

        // 1. Update neurons
        let spike_output = self.neurons.step();

        // 2. Collect fired neuron indices
        self.last_spikes.clear();
        for idx in spike_output.iter_fired() {
            self.last_spikes.push(idx);
        }

        // 3. Deliver spikes through internal synapses
        if let Some(ref synapses) = self.synapses {
            synapses.deliver_spikes(&self.last_spikes, &mut self.neurons.i_ext);
        }

        &self.last_spikes
    }

    pub fn voltage(&self, idx: usize) -> f32 { self.neurons.voltage(idx) }
    pub fn last_spikes(&self) -> &[usize] { &self.last_spikes }
    pub fn synapses_mut(&mut self) -> Option<&mut SynapseCSR> { self.synapses.as_mut() }
    pub fn synapses(&self) -> Option<&SynapseCSR> { self.synapses.as_ref() }
    pub fn neurons_mut(&mut self) -> &mut NeuronArray { &mut self.neurons }
}
