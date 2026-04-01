use crate::config::RegionConfig;
use crate::network::{BrainRegionId, NetworkConfig, SpikingNetwork};
use rand::Rng;

use super::amygdala::amygdala_config;
use super::association::association_cortex_config;
use super::auditory::auditory_cortex_config;
use super::brainstem::brainstem_config;
use super::cerebellum::cerebellum_config;
use super::hippocampus::hippocampus_config;
use super::motor::motor_cortex_config;
use super::predictive::predictive_cortex_config;
use super::prefrontal::prefrontal_cortex_config;
use super::visual::visual_cortex_config;

/// Region indices in the network.
pub const VISUAL: BrainRegionId = 0;
pub const AUDITORY: BrainRegionId = 1;
pub const ASSOCIATION: BrainRegionId = 2;
pub const PREDICTIVE: BrainRegionId = 3;
pub const HIPPOCAMPUS: BrainRegionId = 4;
pub const PREFRONTAL: BrainRegionId = 5;
pub const AMYGDALA: BrainRegionId = 6;
pub const MOTOR: BrainRegionId = 7;
pub const BRAINSTEM: BrainRegionId = 8;
pub const CEREBELLUM: BrainRegionId = 9;

/// Build the complete 10-region brain.
/// `scale`: neuron count multiplier (0.01 = tiny test, 1.0 = full scale ~2M neurons)
/// `intra_prob`: intra-region connection probability
/// `inter_frac`: fraction of neurons in each region that project to connected regions
pub fn build_full_brain(scale: f32, intra_prob: f32, inter_frac: f32) -> SpikingNetwork {
    let configs: Vec<RegionConfig> = vec![
        visual_cortex_config(scale),      // 0
        auditory_cortex_config(scale),    // 1
        association_cortex_config(scale), // 2
        predictive_cortex_config(scale),  // 3
        hippocampus_config(scale),        // 4
        prefrontal_cortex_config(scale),  // 5
        amygdala_config(scale),           // 6
        motor_cortex_config(scale),       // 7
        brainstem_config(scale),          // 8
        cerebellum_config(scale),         // 9
    ];

    let region_sizes: Vec<usize> = configs.iter().map(|c| c.total_neurons()).collect();
    let config = NetworkConfig { regions: configs };
    let mut net = SpikingNetwork::new(config);
    let mut rng = rand::rng();

    // ── Intra-region random sparse connectivity ──
    for (region_id, &n) in region_sizes.iter().enumerate() {
        let n_exc = (n * 4) / 5; // approximate; actual ratio is in config
        for src in 0..n {
            for tgt in 0..n {
                if src == tgt {
                    continue;
                }
                if rng.random::<f32>() < intra_prob {
                    let is_inh = src >= n_exc;
                    let w = if is_inh {
                        -rng.random_range(0.1..0.5)
                    } else {
                        rng.random_range(0.01..0.3)
                    };
                    let delay = rng.random_range(0..5_u8);
                    net.connect_intra(region_id, src, tgt, w, delay);
                }
            }
        }
    }

    // ── Inter-region connectivity (biologically inspired) ──
    // Helper: connect a fraction of neurons from src_region to tgt_region
    let connect_regions = |net: &mut SpikingNetwork,
                           rng: &mut rand::rngs::ThreadRng,
                           src_r: usize,
                           tgt_r: usize,
                           frac: f32,
                           w_range: (f32, f32),
                           delay_range: (u8, u8)| {
        let src_n = region_sizes[src_r];
        let tgt_n = region_sizes[tgt_r];
        let n_proj = ((src_n as f32 * frac) as usize).max(1);
        for _ in 0..n_proj {
            let src = rng.random_range(0..src_n);
            let tgt = rng.random_range(0..tgt_n);
            let w = rng.random_range(w_range.0..w_range.1);
            let d = rng.random_range(delay_range.0..delay_range.1);
            net.connect_inter(src_r, src, tgt_r, tgt, w, d);
        }
    };

    // Feedforward sensory pathway
    connect_regions(&mut net, &mut rng, VISUAL, ASSOCIATION, inter_frac, (0.05, 0.3), (1, 5));
    connect_regions(&mut net, &mut rng, AUDITORY, ASSOCIATION, inter_frac, (0.05, 0.3), (1, 5));
    connect_regions(
        &mut net,
        &mut rng,
        ASSOCIATION,
        PREDICTIVE,
        inter_frac * 0.5,
        (0.05, 0.2),
        (1, 5),
    );
    connect_regions(
        &mut net,
        &mut rng,
        ASSOCIATION,
        PREFRONTAL,
        inter_frac * 0.5,
        (0.05, 0.2),
        (2, 8),
    );

    // Feedback from predictive to sensory
    connect_regions(
        &mut net,
        &mut rng,
        PREDICTIVE,
        VISUAL,
        inter_frac * 0.3,
        (0.02, 0.15),
        (2, 8),
    );
    connect_regions(
        &mut net,
        &mut rng,
        PREDICTIVE,
        AUDITORY,
        inter_frac * 0.3,
        (0.02, 0.15),
        (2, 8),
    );

    // Hippocampus bidirectional with association cortex
    connect_regions(
        &mut net,
        &mut rng,
        ASSOCIATION,
        HIPPOCAMPUS,
        inter_frac * 0.5,
        (0.05, 0.25),
        (2, 6),
    );
    connect_regions(
        &mut net,
        &mut rng,
        HIPPOCAMPUS,
        ASSOCIATION,
        inter_frac * 0.3,
        (0.03, 0.15),
        (3, 8),
    );

    // PFC bidirectional with association and hippocampus
    connect_regions(
        &mut net,
        &mut rng,
        PREFRONTAL,
        ASSOCIATION,
        inter_frac * 0.3,
        (0.02, 0.15),
        (3, 10),
    );
    connect_regions(
        &mut net,
        &mut rng,
        PREFRONTAL,
        HIPPOCAMPUS,
        inter_frac * 0.2,
        (0.02, 0.1),
        (3, 10),
    );

    // Amygdala connections (fast salience detection)
    connect_regions(
        &mut net,
        &mut rng,
        VISUAL,
        AMYGDALA,
        inter_frac * 0.3,
        (0.05, 0.3),
        (1, 3),
    );
    connect_regions(
        &mut net,
        &mut rng,
        AUDITORY,
        AMYGDALA,
        inter_frac * 0.3,
        (0.05, 0.3),
        (1, 3),
    );
    connect_regions(
        &mut net,
        &mut rng,
        AMYGDALA,
        PREFRONTAL,
        inter_frac * 0.3,
        (0.05, 0.2),
        (2, 5),
    );
    connect_regions(
        &mut net,
        &mut rng,
        PREFRONTAL,
        AMYGDALA,
        inter_frac * 0.2,
        (-0.2, 0.1),
        (3, 8),
    ); // inhibitory bias (emotion regulation)
    connect_regions(
        &mut net,
        &mut rng,
        AMYGDALA,
        HIPPOCAMPUS,
        inter_frac * 0.2,
        (0.05, 0.2),
        (2, 5),
    );

    // Motor output from PFC and association
    connect_regions(
        &mut net,
        &mut rng,
        PREFRONTAL,
        MOTOR,
        inter_frac * 0.5,
        (0.05, 0.25),
        (2, 5),
    );
    connect_regions(
        &mut net,
        &mut rng,
        ASSOCIATION,
        MOTOR,
        inter_frac * 0.3,
        (0.03, 0.15),
        (3, 8),
    );

    // Brainstem — receives from amygdala and PFC, projects broadly (neuromodulatory)
    connect_regions(
        &mut net,
        &mut rng,
        AMYGDALA,
        BRAINSTEM,
        inter_frac * 0.3,
        (0.1, 0.3),
        (1, 3),
    );
    connect_regions(
        &mut net,
        &mut rng,
        PREFRONTAL,
        BRAINSTEM,
        inter_frac * 0.2,
        (0.05, 0.2),
        (2, 5),
    );

    // Cerebellum — receives from cortex via pontine relay, outputs to motor/PFC
    connect_regions(
        &mut net,
        &mut rng,
        ASSOCIATION,
        CEREBELLUM,
        inter_frac * 0.3,
        (0.05, 0.2),
        (3, 8),
    );
    connect_regions(
        &mut net,
        &mut rng,
        MOTOR,
        CEREBELLUM,
        inter_frac * 0.3,
        (0.05, 0.2),
        (2, 5),
    );
    connect_regions(
        &mut net,
        &mut rng,
        CEREBELLUM,
        MOTOR,
        inter_frac * 0.2,
        (0.03, 0.15),
        (3, 8),
    );
    connect_regions(
        &mut net,
        &mut rng,
        CEREBELLUM,
        PREFRONTAL,
        inter_frac * 0.1,
        (0.02, 0.1),
        (5, 12),
    );

    net.finalize();
    net
}
