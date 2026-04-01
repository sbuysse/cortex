use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

use memmap2::{Mmap, MmapMut};

use crate::synapse::{SynapseCSR, weight_from_i16};

/// Save SynapseCSR arrays to individual files in a directory.
/// Files: row_ptr.bin, col_idx.bin, weights.bin, weight_refs.bin, delays.bin, eligibilities.bin, meta.bin
impl SynapseCSR {
    pub fn save_to_dir(&self, dir: &Path) -> std::io::Result<()> {
        fs::create_dir_all(dir)?;

        // meta: num_neurons (u64) + num_synapses (u64)
        let mut meta = File::create(dir.join("meta.bin"))?;
        meta.write_all(&(self.num_neurons() as u64).to_le_bytes())?;
        meta.write_all(&(self.num_synapses() as u64).to_le_bytes())?;

        // row_ptr: Vec<u64>
        let mut f = File::create(dir.join("row_ptr.bin"))?;
        for &v in &self.row_ptr {
            f.write_all(&v.to_le_bytes())?;
        }

        // col_idx: Vec<u32>
        let mut f = File::create(dir.join("col_idx.bin"))?;
        for &v in &self.col_idx {
            f.write_all(&v.to_le_bytes())?;
        }

        // weights: Vec<i16>
        let mut f = File::create(dir.join("weights.bin"))?;
        for &v in &self.weights {
            f.write_all(&v.to_le_bytes())?;
        }

        // weight_refs: Vec<i16>
        let mut f = File::create(dir.join("weight_refs.bin"))?;
        for &v in &self.weight_refs {
            f.write_all(&v.to_le_bytes())?;
        }

        // delays: Vec<u8>
        let mut f = File::create(dir.join("delays.bin"))?;
        f.write_all(&self.delays)?;

        // eligibilities: Vec<i16>
        let mut f = File::create(dir.join("eligibilities.bin"))?;
        for &v in &self.eligibilities {
            f.write_all(&v.to_le_bytes())?;
        }

        // structural_scores: Vec<u8>
        let mut f = File::create(dir.join("structural_scores.bin"))?;
        f.write_all(&self.structural_scores)?;

        Ok(())
    }
}

/// Memory-mapped CSR synapse storage.
/// Arrays are backed by mmap'd files -- OS page cache manages residency.
pub struct MmapSynapseCSR {
    num_neurons: usize,
    num_synapses: usize,
    row_ptr: Mmap,               // u64 array
    col_idx: Mmap,               // u32 array
    weights: MmapMut,            // i16 array (mutable for learning)
    weight_refs: MmapMut,        // i16 array (mutable for TACOS)
    delays: Mmap,                // u8 array (immutable)
    eligibilities: MmapMut,      // i16 array (mutable for learning)
    structural_scores: MmapMut,  // u8 array (mutable for structural plasticity)
}

impl MmapSynapseCSR {
    /// Open a previously saved CSR directory as mmap'd files.
    pub fn open(dir: &Path) -> std::io::Result<Self> {
        // Read meta
        let meta_bytes = fs::read(dir.join("meta.bin"))?;
        let num_neurons = u64::from_le_bytes(meta_bytes[0..8].try_into().unwrap()) as usize;
        let num_synapses = u64::from_le_bytes(meta_bytes[8..16].try_into().unwrap()) as usize;

        let row_ptr = unsafe { Mmap::map(&File::open(dir.join("row_ptr.bin"))?)? };
        let col_idx = unsafe { Mmap::map(&File::open(dir.join("col_idx.bin"))?)? };
        let delays = unsafe { Mmap::map(&File::open(dir.join("delays.bin"))?)? };

        // Mutable maps for learnable arrays
        let weights = unsafe {
            MmapMut::map_mut(
                &File::options()
                    .read(true)
                    .write(true)
                    .open(dir.join("weights.bin"))?,
            )?
        };
        let weight_refs = unsafe {
            MmapMut::map_mut(
                &File::options()
                    .read(true)
                    .write(true)
                    .open(dir.join("weight_refs.bin"))?,
            )?
        };
        let eligibilities = unsafe {
            MmapMut::map_mut(
                &File::options()
                    .read(true)
                    .write(true)
                    .open(dir.join("eligibilities.bin"))?,
            )?
        };

        // structural_scores: create file with zeros if it doesn't exist (backward compat)
        let scores_path = dir.join("structural_scores.bin");
        if !scores_path.exists() {
            let mut f = File::create(&scores_path)?;
            f.write_all(&vec![0u8; num_synapses])?;
        }
        let structural_scores = unsafe {
            MmapMut::map_mut(
                &File::options()
                    .read(true)
                    .write(true)
                    .open(&scores_path)?,
            )?
        };

        Ok(Self {
            num_neurons,
            num_synapses,
            row_ptr,
            col_idx,
            weights,
            weight_refs,
            delays,
            eligibilities,
            structural_scores,
        })
    }

    pub fn num_neurons(&self) -> usize {
        self.num_neurons
    }

    pub fn num_synapses(&self) -> usize {
        self.num_synapses
    }

    #[inline]
    fn row_ptr_at(&self, idx: usize) -> u64 {
        let offset = idx * 8;
        u64::from_le_bytes(self.row_ptr[offset..offset + 8].try_into().unwrap())
    }

    #[inline]
    fn col_idx_at(&self, idx: usize) -> u32 {
        let offset = idx * 4;
        u32::from_le_bytes(self.col_idx[offset..offset + 4].try_into().unwrap())
    }

    #[inline]
    fn weight_at(&self, idx: usize) -> i16 {
        let offset = idx * 2;
        i16::from_le_bytes(self.weights[offset..offset + 2].try_into().unwrap())
    }

    pub fn fanout(&self, src: usize) -> usize {
        (self.row_ptr_at(src + 1) - self.row_ptr_at(src)) as usize
    }

    /// Deliver spikes through mmap'd synapses.
    pub fn deliver_spikes(&self, fired: &[usize], current_buf: &mut [f32]) {
        for &src in fired {
            let start = self.row_ptr_at(src) as usize;
            let end = self.row_ptr_at(src + 1) as usize;
            for i in start..end {
                let tgt = self.col_idx_at(i) as usize;
                let w = weight_from_i16(self.weight_at(i));
                current_buf[tgt] += w;
            }
        }
    }

    /// Flush mutable maps to disk.
    pub fn flush(&self) -> std::io::Result<()> {
        self.weights.flush()?;
        self.weight_refs.flush()?;
        self.eligibilities.flush()?;
        self.structural_scores.flush()?;
        Ok(())
    }
}
