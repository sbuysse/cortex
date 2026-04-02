use crate::config::RegionConfig;
use crate::neuromodulation::Neuromodulators;
use crate::plasticity;
use crate::region::BrainRegion;
use serde::{Deserialize, Serialize};

pub type BrainRegionId = usize;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub regions: Vec<RegionConfig>,
}

struct InterRegionLink {
    src_region: BrainRegionId,
    src_neuron: u32,
    tgt_region: BrainRegionId,
    tgt_neuron: u32,
    weight: f32,
    #[allow(dead_code)]
    delay: u8,
}

pub struct NetworkStats {
    pub total_neurons: usize,
    pub total_synapses: usize,
    pub num_regions: usize,
    pub total_spikes_last_step: usize,
}

pub struct SpikingNetwork {
    regions: Vec<BrainRegion>,
    inter_links: Vec<InterRegionLink>,
    pub modulators: Neuromodulators,
    step_count: u64,
    finalized: bool,
}

impl SpikingNetwork {
    pub fn new(config: NetworkConfig) -> Self {
        let regions = config.regions.into_iter().map(BrainRegion::new).collect();
        Self {
            regions,
            inter_links: Vec::new(),
            modulators: Neuromodulators::default(),
            step_count: 0,
            finalized: false,
        }
    }

    pub fn connect_intra(
        &mut self,
        region: BrainRegionId,
        src: usize,
        tgt: usize,
        weight: f32,
        delay: u8,
    ) {
        self.regions[region].connect(src, tgt, weight, delay);
    }

    pub fn connect_inter(
        &mut self,
        src_region: BrainRegionId,
        src_neuron: usize,
        tgt_region: BrainRegionId,
        tgt_neuron: usize,
        weight: f32,
        delay: u8,
    ) {
        self.inter_links.push(InterRegionLink {
            src_region,
            src_neuron: src_neuron as u32,
            tgt_region,
            tgt_neuron: tgt_neuron as u32,
            weight,
            delay,
        });
    }

    pub fn finalize(&mut self) {
        for region in &mut self.regions {
            region.finalize();
        }
        self.finalized = true;
    }

    pub fn inject_current(&mut self, region: BrainRegionId, neuron: usize, current: f32) {
        self.regions[region].inject_current(neuron, current);
    }

    pub fn step(&mut self) {
        if !self.finalized {
            self.finalize();
        }

        // Step all regions
        for region in &mut self.regions {
            region.step();
        }

        // Collect fired neurons per region (to avoid borrow issues)
        let fired: Vec<Vec<usize>> = self
            .regions
            .iter()
            .map(|r| r.last_spikes().to_vec())
            .collect();

        // Deliver inter-region spikes
        for link in &self.inter_links {
            if fired[link.src_region].contains(&(link.src_neuron as usize)) {
                self.regions[link.tgt_region]
                    .inject_current(link.tgt_neuron as usize, link.weight);
            }
        }

        // Apply three-factor learning every 100 steps
        // dw = lr * eligibility * neuromodulator
        if self.step_count % 100 == 0 {
            let modulator = self.modulators.learning_modulator();
            for region in &mut self.regions {
                if let Some(synapses) = region.synapses_mut() {
                    plasticity::apply_three_factor(
                        &mut synapses.weights,
                        &synapses.eligibilities,
                        0.001, // learning rate
                        modulator,
                    );
                }
            }
        }

        self.modulators.step();
        self.step_count += 1;
    }

    /// Step only specific regions (selective attention — saves CPU).
    /// Inter-region spikes are only delivered between active regions.
    pub fn step_selective(&mut self, active_regions: &[BrainRegionId]) {
        if !self.finalized {
            self.finalize();
        }

        // Step only active regions
        for &r in active_regions {
            self.regions[r].step();
        }

        // Collect fired neurons from active regions
        let fired: Vec<Vec<usize>> = self.regions.iter()
            .map(|r| r.last_spikes().to_vec())
            .collect();

        // Deliver inter-region spikes (only between active regions)
        for link in &self.inter_links {
            if active_regions.contains(&link.src_region) && active_regions.contains(&link.tgt_region) {
                if fired[link.src_region].contains(&(link.src_neuron as usize)) {
                    self.regions[link.tgt_region]
                        .inject_current(link.tgt_neuron as usize, link.weight);
                }
            }
        }

        self.step_count += 1;
    }

    pub fn region(&self, id: BrainRegionId) -> &BrainRegion {
        &self.regions[id]
    }
    pub fn region_mut(&mut self, id: BrainRegionId) -> &mut BrainRegion {
        &mut self.regions[id]
    }
    pub fn num_regions(&self) -> usize {
        self.regions.len()
    }
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    pub fn stats(&self) -> NetworkStats {
        let total_neurons: usize = self.regions.iter().map(|r| r.num_neurons()).sum();
        let total_synapses: usize = self
            .regions
            .iter()
            .filter_map(|r| r.synapses().map(|s| s.num_synapses()))
            .sum::<usize>()
            + self.inter_links.len();
        let total_spikes: usize = self.regions.iter().map(|r| r.last_spikes().len()).sum();
        NetworkStats {
            total_neurons,
            total_synapses,
            num_regions: self.regions.len(),
            total_spikes_last_step: total_spikes,
        }
    }
}
