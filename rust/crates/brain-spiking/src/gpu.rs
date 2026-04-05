//! GPU-accelerated spike delivery via tch (libtorch/CUDA).
//!
//! Stores synapse weights as a torch tensor on GPU.
//! Spike delivery = matrix @ spike_vector — single CUDA kernel.

use tch::{Device, Kind, Tensor};
use crate::synapse::{SynapseCSR, weight_from_i16};

/// GPU-backed synapse matrix for fast spike delivery.
pub struct GpuSynapses {
    /// Weight matrix on GPU: (num_neurons, num_neurons), sparse but stored as
    /// index tensors + value tensor for efficient sparse mv.
    row_indices: Tensor,
    col_indices: Tensor,
    values: Tensor,
    num_neurons: i64,
    num_synapses: usize,
    device: Device,
}

impl GpuSynapses {
    /// Create GPU synapse tensors from a CPU CSR.
    pub fn from_csr(csr: &SynapseCSR) -> Self {
        let device = if tch::Cuda::is_available() {
            tracing::info!("GPU available — using CUDA for spike delivery");
            Device::Cuda(0)
        } else {
            tracing::info!("No GPU — spike delivery stays on CPU");
            Device::Cpu
        };

        let n = csr.num_neurons() as i64;
        let nnz = csr.num_synapses();

        // Build COO arrays: (row, col, value) for each synapse
        let mut rows = Vec::with_capacity(nnz);
        let mut cols = Vec::with_capacity(nnz);
        let mut vals = Vec::with_capacity(nnz);

        for src in 0..csr.num_neurons() {
            let start = csr.row_ptr[src] as usize;
            let end = csr.row_ptr[src + 1] as usize;
            for i in start..end {
                // Note: CSR stores outgoing connections (src → tgt).
                // For spike delivery we need: for each fired src, add weight to each tgt.
                // This is matrix[tgt][src] = weight, so we can do: current = W @ spikes.
                rows.push(csr.col_idx[i] as i64);  // target
                cols.push(src as i64);               // source
                vals.push(weight_from_i16(csr.weights[i]));
            }
        }

        let row_indices = Tensor::from_slice(&rows).to(device);
        let col_indices = Tensor::from_slice(&cols).to(device);
        let values = Tensor::from_slice(&vals).to(device);

        tracing::info!("GPU synapses: {}x{}, {} non-zeros, device: {:?}", n, n, nnz, device);

        Self { row_indices, col_indices, values, num_neurons: n, num_synapses: nnz, device }
    }

    /// Deliver spikes via GPU scatter-add.
    /// For each fired neuron, adds its outgoing weights to target neurons.
    pub fn deliver_spikes(&self, fired: &[usize], current_buf: &mut [f32]) {
        if fired.is_empty() || self.num_synapses == 0 { return; }

        let n = self.num_neurons;

        // Create fired mask on GPU
        let fired_indices = Tensor::from_slice(
            &fired.iter().map(|&i| i as i64).collect::<Vec<_>>()
        ).to(self.device);

        // Sparse approach: find which synapses have their source in the fired set
        // col_indices contains source neuron for each synapse
        // Check membership: is col_indices[i] in fired_set?
        let spike_vec = Tensor::zeros([n], (Kind::Float, self.device));
        let ones = Tensor::ones([fired_indices.size()[0]], (Kind::Float, self.device));
        let spike_vec = spike_vec.scatter_add(0, &fired_indices, &ones);

        // For each synapse, get the spike value of its source neuron
        let source_spikes = spike_vec.index_select(0, &self.col_indices);

        // Multiply: active_weights = values * source_spikes (element-wise)
        let active_weights = &self.values * &source_spikes;

        // Scatter-add active weights to target neurons
        let result = Tensor::zeros([n], (Kind::Float, self.device));
        let result = result.scatter_add(0, &self.row_indices, &active_weights);

        // Clamp synaptic drive
        let clamped = result.clamp(-0.5, 0.5);

        // Copy back to CPU
        let cpu_result = Vec::<f32>::try_from(&clamped.to(Device::Cpu)).unwrap_or_default();
        for (i, &v) in cpu_result.iter().enumerate() {
            if i < current_buf.len() && v != 0.0 {
                current_buf[i] += v;
            }
        }
    }

    /// Check if running on GPU.
    pub fn is_cuda(&self) -> bool {
        matches!(self.device, Device::Cuda(_))
    }

    /// Device info.
    pub fn device(&self) -> Device {
        self.device
    }
}
