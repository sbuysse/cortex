use brain_spiking::synapse::SynapseBuilder;
use brain_spiking::synapse_mmap::MmapSynapseCSR;
use std::path::PathBuf;

use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir() -> PathBuf {
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "brain_spiking_test_{}_{id}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

#[test]
fn test_save_and_reopen() {
    let dir = temp_dir();

    // Build CSR in memory
    let mut builder = SynapseBuilder::new(10);
    builder.add(0, 1, 0.5, 1);
    builder.add(0, 3, -0.3, 2);
    builder.add(5, 7, 0.8, 0);
    let csr = builder.freeze();

    // Save to disk
    csr.save_to_dir(&dir).unwrap();

    // Reopen as mmap
    let mmap_csr = MmapSynapseCSR::open(&dir).unwrap();
    assert_eq!(mmap_csr.num_neurons(), 10);
    assert_eq!(mmap_csr.num_synapses(), 3);
    assert_eq!(mmap_csr.fanout(0), 2);
    assert_eq!(mmap_csr.fanout(5), 1);
    assert_eq!(mmap_csr.fanout(3), 0);

    // Clean up
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_mmap_spike_delivery() {
    let dir = temp_dir();

    let mut builder = SynapseBuilder::new(4);
    builder.add(0, 1, 0.5, 0);
    builder.add(0, 2, 0.3, 0);
    let csr = builder.freeze();
    csr.save_to_dir(&dir).unwrap();

    let mmap_csr = MmapSynapseCSR::open(&dir).unwrap();
    let mut currents = vec![0.0_f32; 4];
    mmap_csr.deliver_spikes(&[0], &mut currents);

    assert!((currents[1] - 0.5).abs() < 0.001);
    assert!((currents[2] - 0.3).abs() < 0.001);
    assert_eq!(currents[0], 0.0);
    assert_eq!(currents[3], 0.0);

    let _ = std::fs::remove_dir_all(&dir);
}
