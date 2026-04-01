use crate::config::NeuronParams;

/// Spike output bitmask for a single timestep.
pub struct SpikeOutput {
    bits: Vec<u64>,
    n: usize,
    spike_count: usize,
}

impl SpikeOutput {
    fn new(n: usize) -> Self {
        Self {
            bits: vec![0u64; (n + 63) / 64],
            n,
            spike_count: 0,
        }
    }

    #[inline]
    fn set(&mut self, idx: usize) {
        self.bits[idx / 64] |= 1u64 << (idx % 64);
        self.spike_count += 1;
    }

    #[inline]
    pub fn fired(&self, idx: usize) -> bool {
        (self.bits[idx / 64] >> (idx % 64)) & 1 == 1
    }

    pub fn count(&self) -> usize {
        self.spike_count
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn iter_fired(&self) -> impl Iterator<Item = usize> + '_ {
        let n = self.n;
        self.bits.iter().enumerate().flat_map(move |(word_idx, &word)| {
            let base = word_idx * 64;
            (0..64).filter_map(move |bit| {
                if (word >> bit) & 1 == 1 {
                    let idx = base + bit;
                    if idx < n { Some(idx) } else { None }
                } else {
                    None
                }
            })
        })
    }
}

/// Structure-of-Arrays neuron storage for ALIF neurons.
pub struct NeuronArray {
    pub v: Vec<f32>,
    pub a: Vec<f32>,
    pub i_ext: Vec<f32>,
    n: usize,
    params: NeuronParams,
}

impl NeuronArray {
    pub fn new(n: usize, params: &NeuronParams) -> Self {
        Self {
            v: vec![params.v_rest; n],
            a: vec![0.0; n],
            i_ext: vec![0.0; n],
            n,
            params: params.clone(),
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    #[inline]
    pub fn set_current(&mut self, idx: usize, current: f32) {
        self.i_ext[idx] = current;
    }

    #[inline]
    pub fn add_current(&mut self, idx: usize, current: f32) {
        self.i_ext[idx] += current;
    }

    #[inline]
    pub fn voltage(&self, idx: usize) -> f32 {
        self.v[idx]
    }

    pub fn step(&mut self) -> SpikeOutput {
        let mut spikes = SpikeOutput::new(self.n);
        let p = &self.params;

        for i in 0..self.n {
            self.v[i] = self.v[i] * p.v_decay + self.i_ext[i] - self.a[i];
            if self.v[i] >= p.v_threshold {
                spikes.set(i);
                self.v[i] = p.v_reset;
                self.a[i] += p.a_increment;
            }
            self.a[i] *= p.a_decay;
        }

        self.i_ext.fill(0.0);
        spikes
    }

    pub fn reset(&mut self) {
        self.v.fill(self.params.v_rest);
        self.a.fill(0.0);
        self.i_ext.fill(0.0);
    }
}
