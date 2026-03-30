//! Train the best config and save M matrix + projected embeddings for brain voice.

use brain_experiment::config::ExperimentConfig;
use brain_experiment::cortex::project_all_embeddings;
use brain_experiment::embed_cache::CachedEmbeddings;
use brain_experiment::runner::run_experiment;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

fn save_f32_array(path: &std::path::Path, data: &[f32], shape: &[usize]) {
    let header = format!("{}x{}\n", shape[0], shape[1]);
    let mut bytes = header.into_bytes();
    for val in data {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    std::fs::write(path, &bytes).expect("Failed to write array");
}

fn main() {
    tracing_subscriber::fmt::init();

    let embed_path = std::env::var("EMBED_CACHE")
        .unwrap_or_else(|_| "/opt/brain/data/vggsound/.embed_cache/expanded_embeddings.safetensors".to_string());
    let output_dir = std::env::var("OUTPUT_DIR")
        .unwrap_or_else(|_| "/opt/brain/outputs/cortex/best_model".to_string());
    let skip_train = std::env::var("SKIP_TRAIN").is_ok();

    std::fs::create_dir_all(&output_dir).expect("Failed to create output dir");
    let out = std::path::Path::new(&output_dir);

    eprintln!("Loading embeddings from {embed_path}...");
    let cache = CachedEmbeddings::load(std::path::Path::new(&embed_path)).expect("Failed to load embeddings");
    eprintln!("Loaded {} clips (v={}, a={}, e={})",
        cache.v_emb.nrows(), cache.v_emb.ncols(), cache.a_emb.ncols(), cache.e_emb.ncols());

    // Project to embed_dim=512 (same seeds as server: v=100, a=200, e=300)
    eprintln!("Projecting embeddings to 512...");
    let projected = project_all_embeddings(&cache, 512, 36);
    let n = projected.v.nrows();

    // Save projected embeddings
    eprintln!("Saving projected embeddings ({} clips x 512)...", n);
    save_f32_array(&out.join("v_proj.bin"), projected.v.as_slice().unwrap(), &[n, 512]);
    save_f32_array(&out.join("a_proj.bin"), projected.a.as_slice().unwrap(), &[n, 512]);
    eprintln!("Saved v_proj.bin and a_proj.bin");

    if skip_train {
        eprintln!("SKIP_TRAIN set — skipping training, only saved projected embeddings.");
        return;
    }

    // Best known config from exp 20013 (v2a_MRR=0.53, a2v_MRR=0.54)
    let config = ExperimentConfig {
        embed_dim: 512,
        sparsity_k: 36,
        hebbian_lr: 0.06742183864116669,
        decay_rate: 0.992,
        temporal_decay: 0.86,
        trace_weight: 0.26,
        max_norm: 22.165306091308594,
        max_steps: 20000,
        batch_size: 2048,
        seed: 42,
        gradient_infonce: true,
        symmetric_infonce: true,
        use_infonce: true,
        infonce_temperature: 0.006394233554601669,
        eval_pool_size: 1000,
        output_dir: Some(output_dir.clone()),
        ..Default::default()
    };

    eprintln!("Training {} steps with gradient InfoNCE + symmetric...", config.max_steps);
    let shutdown = Arc::new(AtomicBool::new(false));
    let result = run_experiment(
        &config,
        &projected.v,
        &projected.a,
        &projected.e,
        projected.s.as_ref(),
        projected.p.as_ref(),
        0,
        shutdown,
        None,
    );

    eprintln!("Done! Metrics:");
    for (k, v) in &result.metrics {
        if k.contains("MRR") || k.contains("R@") || k.contains("MR") || k == "duration_seconds" {
            eprintln!("  {}: {:.6}", k, v);
        }
    }
    eprintln!("M matrix saved to {}/m_va.bin", output_dir);
}
