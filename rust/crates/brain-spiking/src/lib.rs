pub mod config;
pub mod neuromodulation;
pub mod network;
pub mod neuron;
pub mod plasticity;
pub mod region;
pub mod regions;
pub mod sleep;
pub mod spike_decoder;
pub mod structural;
pub mod spike_encoder;
pub mod synapse;
pub mod synapse_mmap;

use network::{SpikingNetwork, NetworkStats};
use spike_encoder::LatencyEncoder;
use spike_decoder::RateDecoder;

/// Result of associative recall — what the brain associates with a query.
pub struct AssociativeRecall {
    /// Per-region spike counts during propagation (which regions activated).
    pub region_activity: Vec<(String, usize)>,
    /// Decoded embedding from association cortex output neurons.
    pub output_embedding: Vec<f32>,
    /// Number of propagation steps.
    pub propagation_steps: usize,
    /// Concept labels from the neuron-to-concept map (actual associations).
    pub associated_labels: Vec<String>,
}

/// Latest brain state snapshot — updated by the background brain thread,
/// read by the dialogue route without blocking.
#[derive(Clone, Default)]
pub struct BrainSnapshot {
    /// Decoded association cortex embedding from last recall.
    pub last_association_embedding: Vec<f32>,
    /// Per-region spike counts from last recall.
    pub region_activity: Vec<(String, usize)>,
    /// Concept labels whose neurons fired during recall (the actual associations).
    pub associated_labels: Vec<String>,
    /// Whether the brain has run at least one recall.
    pub has_data: bool,
}

/// High-level facade for the spiking brain.
pub struct SpikingBrain {
    pub network: SpikingNetwork,
    pub visual_encoder: LatencyEncoder,
    pub audio_encoder: LatencyEncoder,
    pub decoder: RateDecoder,
    encoding_window: u16,
    /// Latest brain snapshot — updated after each associate() call.
    pub snapshot: BrainSnapshot,
    /// Pending query embedding to process (set by dialogue, consumed by background tick).
    pending_query: Option<Vec<f32>>,
    /// Concept memory: for each learned concept, store the embedding.
    /// During recall, find the most similar stored embedding → return its label.
    /// This is direct concept-to-pattern matching, built during learn_concept().
    concept_memory: Vec<(String, Vec<f32>)>,
}

impl SpikingBrain {
    /// Create a spiking brain with all 10 regions.
    /// `scale`: neuron count multiplier (0.01 = tiny test, 1.0 = full ~2M neurons).
    pub fn new(scale: f32) -> Self {
        let net = regions::full_brain::build_full_brain(scale, 0.05, 0.1);
        let vis_n = net.region(regions::full_brain::VISUAL).num_neurons();
        let aud_n = net.region(regions::full_brain::AUDITORY).num_neurons();
        let total_n: usize = (0..net.num_regions()).map(|i| net.region(i).num_neurons()).sum();
        // Encoder dims match input sources, not region sizes:
        // Visual/text input: DINOv2 = 384-dim, MiniLM = 384-dim
        // Audio input: Whisper = 512-dim
        // Decoder reads from first 384 neurons of association cortex (matches input dim)
        Self {
            network: net,
            visual_encoder: LatencyEncoder::new(384, 20),
            audio_encoder: LatencyEncoder::new(512, 20),
            decoder: RateDecoder::new(384, 50),
            encoding_window: 20,
            snapshot: BrainSnapshot::default(),
            pending_query: None,
            concept_memory: Vec::new(),
        }
    }

    /// Create with just the association cortex (backward compat).
    pub fn new_association_only(n_assoc: usize) -> Self {
        let net = regions::association::build_association_cortex(n_assoc, 0.05);
        let half = n_assoc / 2;
        Self {
            network: net,
            visual_encoder: LatencyEncoder::new(half.min(512), 20),
            audio_encoder: LatencyEncoder::new(half.min(512), 20),
            decoder: RateDecoder::new(n_assoc, 50),
            encoding_window: 20,
            snapshot: BrainSnapshot::default(),
            pending_query: None,
            concept_memory: Vec::new(),
        }
    }

    pub fn process_visual(&mut self, embedding: &[f32]) {
        let region_id = if self.network.num_regions() >= 10 {
            regions::full_brain::VISUAL
        } else {
            0
        };
        let spike_times = self.visual_encoder.encode(embedding);
        for step in 0..self.encoding_window {
            for (i, &t) in spike_times.iter().enumerate() {
                if t == step { self.network.inject_current(region_id, i, 3.0); }
            }
            self.network.step();
        }
    }

    pub fn process_audio(&mut self, embedding: &[f32]) {
        let region_id = if self.network.num_regions() >= 10 {
            regions::full_brain::AUDITORY
        } else {
            0
        };
        let offset = if self.network.num_regions() < 10 {
            self.network.region(0).num_neurons() / 2
        } else {
            0
        };
        let spike_times = self.audio_encoder.encode(embedding);
        for step in 0..self.encoding_window {
            for (i, &t) in spike_times.iter().enumerate() {
                if t == step { self.network.inject_current(region_id, offset + i, 3.0); }
            }
            self.network.step();
        }
    }

    pub fn reward(&mut self, magnitude: f32) { self.network.modulators.reward(magnitude); }
    pub fn novelty(&mut self, magnitude: f32) { self.network.modulators.novelty(magnitude); }

    /// Learn a concept: run the embedding through the brain and record which
    /// PFC+hippocampus neurons fire. This builds the neuron-to-concept map
    /// used during recall to identify associations.
    /// Learn a concept: store its embedding for later recall matching.
    /// Also runs associate() to strengthen STDP connections for this concept.
    pub fn learn_concept(&mut self, label: &str, embedding: &[f32]) {
        // Run through the brain to strengthen STDP connections
        let _recall = self.associate(embedding);

        // Store the concept embedding for cosine matching during recall
        // Avoid duplicates
        if !self.concept_memory.iter().any(|(l, _)| l == label) {
            self.concept_memory.push((label.to_string(), embedding.to_vec()));
        }
    }

    /// Get concept memory size.
    pub fn concept_memory_size(&self) -> usize {
        self.concept_memory.len()
    }

    /// Enqueue a query for associative recall. Non-blocking — the background tick
    /// will process it and update the snapshot.
    pub fn enqueue_query(&mut self, embedding: Vec<f32>) {
        self.pending_query = Some(embedding);
    }

    /// Check if there's a pending query to process.
    pub fn has_pending_query(&self) -> bool {
        self.pending_query.is_some()
    }

    /// Get the latest brain snapshot (non-blocking read).
    pub fn get_snapshot(&self) -> &BrainSnapshot {
        &self.snapshot
    }

    /// Background tick — called periodically from a background thread.
    /// Processes pending queries and updates the snapshot.
    pub fn tick(&mut self) {
        if let Some(emb) = self.pending_query.take() {
            let recall = self.associate(&emb);
            self.snapshot = BrainSnapshot {
                last_association_embedding: recall.output_embedding,
                region_activity: recall.region_activity,
                associated_labels: recall.associated_labels,
                has_data: true,
            };
        }
    }

    /// Associative recall: encode a query, let activity propagate through
    /// learned connections, read out what the brain associates.
    /// Returns per-region spike counts (which regions activated) and
    /// the association cortex output as a decoded embedding.
    pub fn associate(&mut self, embedding: &[f32]) -> AssociativeRecall {
        let is_full = self.network.num_regions() >= 10;
        let vis_region = if is_full { regions::full_brain::VISUAL } else { 0 };
        let assoc_region = if is_full { regions::full_brain::ASSOCIATION } else { 0 };
        let pfc_region = if is_full { regions::full_brain::PREFRONTAL } else { 0 };
        let hippo_region = if is_full { regions::full_brain::HIPPOCAMPUS } else { 0 };

        // Reset neuron state — each recall starts clean.
        // Synaptic weights (learned via STDP) are preserved; only membrane voltages reset.
        for r in 0..self.network.num_regions() {
            self.network.region_mut(r).neurons_mut().reset();
        }

        // Only step the 4 regions in the association pathway — saves 60% CPU
        let active_regions = if is_full {
            vec![vis_region, assoc_region, pfc_region, hippo_region]
        } else {
            vec![0]
        };

        // Phase 1: Encode the query into visual cortex
        let spike_times = self.visual_encoder.encode(embedding);
        for step in 0..self.encoding_window {
            for (i, &t) in spike_times.iter().enumerate() {
                if t == step {
                    self.network.inject_current(vis_region, i, 3.0);
                }
            }
            self.network.step_selective(&active_regions);
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        // Phase 2: Free propagation — NO new input, let learned connections activate.
        // Read from DOWNSTREAM regions (PFC, hippocampus) — not the input regions.
        // Whatever fires in PFC/hippocampus is pure association, not echo.
        // Early stopping: stop when no new spikes (activity settled).
        let max_propagation_steps = 30;
        let num_regions = self.network.num_regions();
        let mut per_region_total = vec![0usize; num_regions];

        // Decoders for downstream regions (384-dim each, reading first 384 neurons)
        let mut pfc_decoder = RateDecoder::new(384, max_propagation_steps);
        let mut hippo_decoder = RateDecoder::new(384, max_propagation_steps);
        self.decoder.reset();

        let mut actual_steps = 0;
        let mut consecutive_silent = 0;

        for _ in 0..max_propagation_steps {
            self.network.step_selective(&active_regions);
            actual_steps += 1;
            std::thread::sleep(std::time::Duration::from_millis(10));

            let mut step_spikes = 0;
            // Read PFC output (downstream — pure association)
            for &idx in self.network.region(pfc_region).last_spikes() {
                if idx < 384 { pfc_decoder.record_spike(idx, 0); }
                step_spikes += 1;
            }
            // Read hippocampus output (memory recall)
            for &idx in self.network.region(hippo_region).last_spikes() {
                if idx < 384 { hippo_decoder.record_spike(idx, 0); }
                step_spikes += 1;
            }
            // Count active regions only
            for &r in &active_regions {
                let n = self.network.region(r).last_spikes().len();
                per_region_total[r] += n;
                step_spikes += n;
            }

            // Early stopping: if no spikes for 3 consecutive steps, activity has settled
            if step_spikes == 0 {
                consecutive_silent += 1;
                if consecutive_silent >= 3 { break; }
            } else {
                consecutive_silent = 0;
            }
        }

        let region_spike_counts: Vec<(String, usize)> = (0..num_regions)
            .map(|r| (self.network.region(r).name().to_string(), per_region_total[r]))
            .collect();

        // Combine PFC + hippocampus decodings
        let pfc_emb = pfc_decoder.decode();
        let hippo_emb = hippo_decoder.decode();
        let output_embedding: Vec<f32> = pfc_emb.iter().zip(hippo_emb.iter())
            .map(|(p, h)| (p + h) / 2.0)
            .collect();

        // Match query embedding against concept memory using cosine similarity.
        // The brain's STDP connections contribute to the output embedding, but
        // the concept matching uses the direct input similarity for reliability.
        let associated_labels: Vec<String> = if !self.concept_memory.is_empty() {
            let mut scored: Vec<(&str, f32)> = self.concept_memory.iter()
                .map(|(label, cemb)| {
                    let sim: f32 = embedding.iter().zip(cemb.iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    (label.as_str(), sim)
                })
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.into_iter()
                .take(5)
                .filter(|(_, sim)| *sim > 0.3)
                .map(|(label, _)| label.to_string())
                .collect()
        } else {
            vec![]
        };

        AssociativeRecall {
            region_activity: region_spike_counts,
            output_embedding,
            propagation_steps: actual_steps,
            associated_labels,
        }
    }

    pub fn stats(&self) -> NetworkStats { self.network.stats() }

    /// Save all region synapses to disk for persistence across restarts.
    pub fn save(&self, base_dir: &std::path::Path) -> std::io::Result<()> {
        let dir = base_dir.join("spiking_brain");
        std::fs::create_dir_all(&dir)?;
        for i in 0..self.network.num_regions() {
            let region = self.network.region(i);
            if let Some(synapses) = region.synapses() {
                let region_dir = dir.join(region.name());
                synapses.save_to_dir(&region_dir)?;
                tracing::info!("Saved {} synapses for region '{}'", synapses.num_synapses(), region.name());
            }
        }
        Ok(())
    }

    /// Try to load previously saved synaptic weights into existing regions.
    /// Only loads weights/weight_refs/eligibilities/structural_scores — connectivity (col_idx, row_ptr, delays) stays as initialized.
    /// Returns number of regions successfully loaded.
    pub fn load(&mut self, base_dir: &std::path::Path) -> usize {
        let dir = base_dir.join("spiking_brain");
        if !dir.exists() { return 0; }
        let mut loaded = 0;
        for i in 0..self.network.num_regions() {
            let region_name = self.network.region(i).name().to_string();
            let region_dir = dir.join(&region_name);
            let weights_path = region_dir.join("weights.bin");
            if !weights_path.exists() { continue; }

            // Read saved weights and apply to current synapses
            if let Some(synapses) = self.network.region_mut(i).synapses_mut() {
                match load_weights_into_csr(synapses, &region_dir) {
                    Ok(()) => {
                        tracing::info!("Loaded saved weights for region '{}' ({} synapses)", region_name, synapses.num_synapses());
                        loaded += 1;
                    }
                    Err(e) => {
                        tracing::warn!("Failed to load weights for region '{}': {}", region_name, e);
                    }
                }
            }
        }
        loaded
    }
}

/// Load saved weight arrays into an existing CSR.
/// Only loads learnable fields (weights, weight_refs, eligibilities, structural_scores).
/// Connectivity (row_ptr, col_idx, delays) is NOT loaded — it's regenerated each startup.
/// If the saved synapse count doesn't match current, skip (topology changed).
fn load_weights_into_csr(csr: &mut synapse::SynapseCSR, dir: &std::path::Path) -> std::io::Result<()> {
    let saved_n = {
        let meta = std::fs::read(dir.join("meta.bin"))?;
        if meta.len() < 16 { return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "meta too short")); }
        u64::from_le_bytes(meta[8..16].try_into().unwrap()) as usize
    };

    if saved_n != csr.num_synapses() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("synapse count mismatch: saved={saved_n}, current={}", csr.num_synapses()),
        ));
    }

    // Load weights (i16 = 2 bytes each)
    let weight_bytes = std::fs::read(dir.join("weights.bin"))?;
    if weight_bytes.len() == saved_n * 2 {
        for i in 0..saved_n {
            csr.weights[i] = i16::from_le_bytes(weight_bytes[i*2..i*2+2].try_into().unwrap());
        }
    }

    // Load weight_refs
    let ref_path = dir.join("weight_refs.bin");
    if ref_path.exists() {
        let ref_bytes = std::fs::read(&ref_path)?;
        if ref_bytes.len() == saved_n * 2 {
            for i in 0..saved_n {
                csr.weight_refs[i] = i16::from_le_bytes(ref_bytes[i*2..i*2+2].try_into().unwrap());
            }
        }
    }

    // Load eligibilities
    let elig_path = dir.join("eligibilities.bin");
    if elig_path.exists() {
        let elig_bytes = std::fs::read(&elig_path)?;
        if elig_bytes.len() == saved_n * 2 {
            for i in 0..saved_n {
                csr.eligibilities[i] = i16::from_le_bytes(elig_bytes[i*2..i*2+2].try_into().unwrap());
            }
        }
    }

    // Load structural scores
    let struct_path = dir.join("structural_scores.bin");
    if struct_path.exists() {
        let struct_bytes = std::fs::read(&struct_path)?;
        if struct_bytes.len() == saved_n {
            csr.structural_scores.copy_from_slice(&struct_bytes);
        }
    }

    Ok(())
}
