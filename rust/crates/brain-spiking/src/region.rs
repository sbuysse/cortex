use crate::config::RegionConfig;
#[cfg(feature = "gpu")]
use crate::gpu::GpuSynapses;
use crate::neuron::NeuronArray;
use crate::synapse::{SynapseBuilder, SynapseCSR, weight_from_i16, weight_to_i16};
use crate::plasticity::{PlasticityParams, update_stdp, update_eligibility};

/// A brain region: a population of neurons with internal connectivity.
pub struct BrainRegion {
    config: RegionConfig,
    neurons: NeuronArray,
    synapses: Option<SynapseCSR>,
    /// GPU-accelerated synapse matrix (if CUDA available).
    #[cfg(feature = "gpu")]
    gpu_synapses: Option<GpuSynapses>,
    builder: Option<SynapseBuilder>,
    last_spikes: Vec<usize>,
    /// Last spike time per neuron (u32::MAX = never spiked).
    last_spike_time: Vec<u32>,
    /// Current simulation step within this region.
    step_count: u32,
    /// Plasticity parameters.
    plasticity: PlasticityParams,
    /// Enable/disable online learning.
    pub learning_enabled: bool,
}

impl BrainRegion {
    pub fn new(config: RegionConfig) -> Self {
        let n = config.total_neurons();
        let neurons = NeuronArray::new(n, &config.neuron_params);
        Self {
            builder: Some(SynapseBuilder::new(n)),
            #[cfg(feature = "gpu")]
            gpu_synapses: None,
            last_spike_time: vec![u32::MAX; n],
            step_count: 0,
            plasticity: PlasticityParams::default(),
            learning_enabled: true,
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

    /// Freeze connectivity from COO to CSR. Creates GPU version if CUDA available.
    pub fn finalize(&mut self) {
        if let Some(builder) = self.builder.take() {
            let csr = builder.freeze();
            #[cfg(feature = "gpu")]
            {
                tracing::info!("Region '{}': {} synapses, checking GPU...", self.config.name, csr.num_synapses());
                if csr.num_synapses() > 10000 {
                    let gpu = GpuSynapses::from_csr(&csr);
                    if gpu.is_cuda() {
                        tracing::info!("Region '{}': GPU spike delivery ENABLED", self.config.name);
                        self.gpu_synapses = Some(gpu);
                    } else {
                        tracing::info!("Region '{}': no CUDA, using CPU", self.config.name);
                    }
                }
            }
            self.synapses = Some(csr);
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

        // 3. Deliver spikes through internal synapses (GPU if available, CPU fallback)
        #[cfg(feature = "gpu")]
        if let Some(ref gpu) = self.gpu_synapses {
            gpu.deliver_spikes(&self.last_spikes, &mut self.neurons.i_ext);
        } else if let Some(ref synapses) = self.synapses {
            synapses.deliver_spikes(&self.last_spikes, &mut self.neurons.i_ext);
        }
        #[cfg(not(feature = "gpu"))]
        if let Some(ref synapses) = self.synapses {
            synapses.deliver_spikes(&self.last_spikes, &mut self.neurons.i_ext);
        }

        // 4. STDP: update eligibility traces for synapses of firing neurons
        if self.learning_enabled {
            if let Some(ref mut synapses) = self.synapses {
                let t = self.step_count;
                for &post_idx in &self.last_spikes {
                    self.last_spike_time[post_idx] = t;
                }
                // For each post-synaptic neuron that fired, update eligibility
                // of all incoming synapses (pre→post).
                // CSR is stored by pre-synaptic neuron, so we iterate outgoing synapses
                // of recently-fired PRE neurons and check if their targets fired.
                for &pre_idx in &self.last_spikes {
                    let start = synapses.row_ptr[pre_idx] as usize;
                    let end = synapses.row_ptr[pre_idx + 1] as usize;
                    for syn_i in start..end {
                        let tgt = synapses.col_idx[syn_i] as usize;
                        let tgt_spike = self.last_spike_time[tgt];
                        if tgt_spike != u32::MAX && tgt_spike != t {
                            // Pre fired now, post fired at tgt_spike
                            let dt = t as f32 - tgt_spike as f32; // positive = post-before-pre
                            let dw = update_stdp(dt, &self.plasticity);
                            update_eligibility(&mut synapses.eligibilities[syn_i], dw);
                        }
                    }
                }

                // Decay eligibilities every 10 steps (avoid per-step overhead on all synapses)
                if t % 10 == 0 {
                    let decay = self.plasticity.eligibility_decay.powi(10);
                    for e in synapses.eligibilities.iter_mut() {
                        if *e != 0 {
                            let ef = weight_from_i16(*e) * decay;
                            *e = weight_to_i16(ef);
                        }
                    }
                }
            }
        }

        self.step_count += 1;
        &self.last_spikes
    }

    /// Advance one timestep with a custom synaptic drive clamp.
    /// Used during spiking recall where imprinted weights need higher clamp.
    pub fn step_with_clamp(&mut self, max_drive: f32) -> &[usize] {
        if self.synapses.is_none() && self.builder.is_some() {
            self.finalize();
        }
        let spike_output = self.neurons.step();
        self.last_spikes.clear();
        for idx in spike_output.iter_fired() {
            self.last_spikes.push(idx);
        }
        #[cfg(feature = "gpu")]
        if let Some(ref gpu) = self.gpu_synapses {
            gpu.deliver_spikes(&self.last_spikes, &mut self.neurons.i_ext);
        } else if let Some(ref synapses) = self.synapses {
            synapses.deliver_spikes_with_clamp(&self.last_spikes, &mut self.neurons.i_ext, max_drive);
        }
        #[cfg(not(feature = "gpu"))]
        if let Some(ref synapses) = self.synapses {
            synapses.deliver_spikes_with_clamp(&self.last_spikes, &mut self.neurons.i_ext, max_drive);
        }
        self.step_count += 1;
        &self.last_spikes
    }

    pub fn voltage(&self, idx: usize) -> f32 { self.neurons.voltage(idx) }
    pub fn last_spikes(&self) -> &[usize] { &self.last_spikes }
    pub fn synapses_mut(&mut self) -> Option<&mut SynapseCSR> { self.synapses.as_mut() }
    pub fn synapses(&self) -> Option<&SynapseCSR> { self.synapses.as_ref() }
    pub fn neurons_mut(&mut self) -> &mut NeuronArray { &mut self.neurons }
}
