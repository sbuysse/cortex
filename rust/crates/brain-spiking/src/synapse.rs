/// Convert f32 weight in [-1.0, 1.0] to i16.
#[inline]
pub fn weight_to_i16(w: f32) -> i16 {
    (w.clamp(-1.0, 1.0) * 32767.0) as i16
}

/// Convert i16 back to f32 weight.
#[inline]
pub fn weight_from_i16(q: i16) -> f32 {
    q as f32 / 32767.0
}

/// A single synapse as viewed from CSR iteration.
#[derive(Debug, Clone, Copy)]
pub struct SynapseView {
    pub target: u32,
    pub weight: i16,
    pub weight_ref: i16,
    pub delay: u8,
    pub eligibility: i16,
    pub structural_score: u8,
}

impl SynapseView {
    pub fn weight_f32(&self) -> f32 {
        weight_from_i16(self.weight)
    }
}

/// COO entry for building.
struct CooEntry {
    src: u32,
    tgt: u32,
    weight: i16,
    delay: u8,
}

/// Builder for synapse connectivity. Accumulates in COO format,
/// then freezes to CSR for simulation.
pub struct SynapseBuilder {
    entries: Vec<CooEntry>,
    num_neurons: usize,
}

impl SynapseBuilder {
    pub fn new(num_neurons: usize) -> Self {
        Self { entries: Vec::new(), num_neurons }
    }

    pub fn with_capacity(num_neurons: usize, estimated_synapses: usize) -> Self {
        Self { entries: Vec::with_capacity(estimated_synapses), num_neurons }
    }

    /// Add a synapse from src to tgt.
    pub fn add(&mut self, src: usize, tgt: usize, weight: f32, delay: u8) {
        self.entries.push(CooEntry {
            src: src as u32,
            tgt: tgt as u32,
            weight: weight_to_i16(weight),
            delay,
        });
    }

    /// Freeze to CSR format. Sorts by (src, tgt) for cache locality.
    pub fn freeze(mut self) -> SynapseCSR {
        self.entries.sort_unstable_by_key(|e| (e.src, e.tgt));

        let nnz = self.entries.len();
        let n = self.num_neurons;

        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx = Vec::with_capacity(nnz);
        let mut weights = Vec::with_capacity(nnz);
        let mut weight_refs = Vec::with_capacity(nnz);
        let mut delays = Vec::with_capacity(nnz);
        let mut eligibilities = Vec::with_capacity(nnz);

        let mut entry_idx = 0;
        for src in 0..n {
            row_ptr.push(entry_idx as u64);
            while entry_idx < nnz && self.entries[entry_idx].src == src as u32 {
                let e = &self.entries[entry_idx];
                col_idx.push(e.tgt);
                weights.push(e.weight);
                weight_refs.push(e.weight); // initial ref = initial weight
                delays.push(e.delay);
                eligibilities.push(0i16);
                entry_idx += 1;
            }
        }
        row_ptr.push(nnz as u64);

        let structural_scores = vec![0u8; nnz];

        SynapseCSR { row_ptr, col_idx, weights, weight_refs, delays, eligibilities, structural_scores, num_neurons: n }
    }
}

/// Compressed Sparse Row synapse storage.
pub struct SynapseCSR {
    pub row_ptr: Vec<u64>,
    pub col_idx: Vec<u32>,
    pub weights: Vec<i16>,
    pub weight_refs: Vec<i16>,
    pub delays: Vec<u8>,
    pub eligibilities: Vec<i16>,
    pub structural_scores: Vec<u8>,
    num_neurons: usize,
}

impl SynapseCSR {
    pub fn num_neurons(&self) -> usize { self.num_neurons }
    pub fn num_synapses(&self) -> usize { self.col_idx.len() }

    /// Get all outgoing synapses for presynaptic neuron `src`.
    pub fn targets_of(&self, src: usize) -> Vec<SynapseView> {
        let start = self.row_ptr[src] as usize;
        let end = self.row_ptr[src + 1] as usize;
        (start..end).map(|i| SynapseView {
            target: self.col_idx[i],
            weight: self.weights[i],
            weight_ref: self.weight_refs[i],
            delay: self.delays[i],
            eligibility: self.eligibilities[i],
            structural_score: self.structural_scores[i],
        }).collect()
    }

    #[inline]
    pub fn fanout(&self, src: usize) -> usize {
        (self.row_ptr[src + 1] - self.row_ptr[src]) as usize
    }

    /// Deliver spikes: for each fired neuron, add weighted current to targets.
    /// Clamps SYNAPTIC input per neuron to prevent activity cascades.
    /// External input (from encoders via inject_current) is NOT clamped.
    ///
    /// Optimized: sorts fired neurons for sequential CSR access,
    /// prefetches next row while processing current one.
    pub fn deliver_spikes(&self, fired: &[usize], current_buf: &mut [f32]) {
        if fired.is_empty() { return; }

        let n = current_buf.len();
        let mut synaptic = vec![0.0f32; n];

        // Sort fired neurons by index — sequential CSR row_ptr access
        let mut sorted_fired: Vec<usize> = fired.to_vec();
        sorted_fired.sort_unstable();

        for (fi, &src) in sorted_fired.iter().enumerate() {
            let start = self.row_ptr[src] as usize;
            let end = self.row_ptr[src + 1] as usize;

            // Prefetch next neuron's CSR data into cache while processing current
            if fi + 1 < sorted_fired.len() {
                let next_src = sorted_fired[fi + 1];
                let next_start = self.row_ptr[next_src] as usize;
                if next_start < self.col_idx.len() {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        core::arch::x86_64::_mm_prefetch(
                            self.col_idx.as_ptr().add(next_start) as *const i8,
                            core::arch::x86_64::_MM_HINT_T0);
                        core::arch::x86_64::_mm_prefetch(
                            self.weights.as_ptr().add(next_start) as *const i8,
                            core::arch::x86_64::_MM_HINT_T0);
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        // Fallback: volatile read pulls into cache
                        unsafe { let _ = std::ptr::read_volatile(&self.col_idx[next_start]); }
                    }
                }
            }

            // Process row with intra-row prefetching (4 cache lines = 64 entries ahead)
            const PREFETCH_DISTANCE: usize = 64;
            for i in start..end {
                // Prefetch ahead within this row
                if i + PREFETCH_DISTANCE < end {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        core::arch::x86_64::_mm_prefetch(
                            self.col_idx.as_ptr().add(i + PREFETCH_DISTANCE) as *const i8,
                            core::arch::x86_64::_MM_HINT_T0);
                        core::arch::x86_64::_mm_prefetch(
                            self.weights.as_ptr().add(i + PREFETCH_DISTANCE) as *const i8,
                            core::arch::x86_64::_MM_HINT_T0);
                    }
                }
                let tgt = self.col_idx[i] as usize;
                let w = weight_from_i16(self.weights[i]);
                synaptic[tgt] += w;
            }
        }

        // Clamp synaptic drive per neuron — prevents cascade.
        const MAX_SYNAPTIC_DRIVE: f32 = 0.5;
        for i in 0..n {
            if synaptic[i] != 0.0 {
                current_buf[i] += synaptic[i].clamp(-MAX_SYNAPTIC_DRIVE, MAX_SYNAPTIC_DRIVE);
            }
        }
    }
}
