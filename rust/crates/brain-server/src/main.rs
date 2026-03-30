//! Brain server — axum REST API + Web UI + Cortex autonomous loop.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use brain_server::state::{BrainViz, InteractState};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "brain_server=info,brain_experiment=info,brain_cognition=info,brain_inference=info,tower_http=info".into()),
        )
        .init();

    let project_root = std::env::var("BRAIN_PROJECT_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/opt/brain"));

    let db_path = std::env::var("BRAIN_DB_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| project_root.join("outputs/cortex/knowledge.db"));

    let templates_dir = std::env::var("BRAIN_TEMPLATES_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| project_root.join("templates"));

    let output_dir = std::env::var("BRAIN_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| project_root.join("outputs/cortex"));

    let bind_addr = std::env::var("BRAIN_BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:443".into());

    let ssl_cert = std::env::var("BRAIN_SSL_CERT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| project_root.join("certs/cert.pem"));

    let ssl_key = std::env::var("BRAIN_SSL_KEY")
        .map(PathBuf::from)
        .unwrap_or_else(|_| project_root.join("certs/key.pem"));

    let embed_dir = std::env::var("BRAIN_EMBED_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| project_root.join("data/vggsound/.embed_cache"));

    let cortex_enabled = std::env::var("BRAIN_CORTEX_DISABLE").is_err();

    // Load embeddings once, share with both interactive UI and cortex
    let safetensors = find_safetensors(&embed_dir);
    let raw_embeddings = safetensors.and_then(|path| {
        tracing::info!(?path, "Loading embeddings");
        match brain_experiment::embed_cache::CachedEmbeddings::load(&path) {
            Ok(emb) => {
                tracing::info!(n_clips = emb.n_clips, "Embeddings loaded");
                Some(emb)
            }
            Err(e) => {
                tracing::error!(error = %e, "Failed to load embeddings");
                None
            }
        }
    });

    // Project embeddings and build interact state
    let (interact, cached_v, cached_a, cached_e, cached_s, cached_p, whitened) = if let Some(ref raw) = raw_embeddings {
        let projected = brain_experiment::cortex::project_all_embeddings(raw, 512, 36);

        // Load VGGSound labels
        let labels = load_vggsound_labels(&project_root, raw.n_clips);
        tracing::info!(n_labels = labels.len(), "Loaded clip labels");

        let interact = Arc::new(InteractState::new(
            projected.v.clone(),
            projected.a.clone(),
            projected.e.clone(),
            projected.s.clone(),
            projected.p.clone(),
            labels,
        ));
        let cs = projected.s.map(Arc::new);
        let cp = projected.p.map(Arc::new);

        // Load ZCA-whitened embeddings for v2 pipeline (if available)
        let whitening_path = embed_dir.join("whitening.safetensors");
        let whitened = if whitening_path.exists() {
            match brain_experiment::embed_cache::WhiteningTransforms::load(&whitening_path) {
                Ok(w) => {
                    tracing::info!(
                        v_dim = w.v_white.ncols(), a_dim = w.a_white.ncols(),
                        "Loaded ZCA whitened embeddings for v2 pipeline"
                    );
                    Some(brain_experiment::cortex::WhitenedEmbeddings {
                        v: Arc::new(w.v_white),
                        a: Arc::new(w.a_white),
                    })
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to load whitening — v2 pipeline disabled");
                    None
                }
            }
        } else {
            tracing::info!("No whitening.safetensors found — v2 pipeline disabled");
            None
        };

        (
            Some(interact),
            Some(Arc::new(projected.v)),
            Some(Arc::new(projected.a)),
            Some(Arc::new(projected.e)),
            cs,
            cp,
            whitened,
        )
    } else {
        (None, None, None, None, None, None, None)
    };

    // Build brain visualization from real embeddings
    let brain_viz = if let (Some(cv), Some(ca)) = (&cached_v, &cached_a) {
        tracing::info!("Computing brain visualization data...");
        let viz = compute_brain_viz(cv, ca, &cached_e, &cached_s, &cached_p,
            raw_embeddings.as_ref().map(|r| load_vggsound_labels(&project_root, r.n_clips)).unwrap_or_default());
        tracing::info!("Brain visualization ready");
        Some(Arc::new(viz))
    } else {
        None
    };

    // Build shared state and router
    let (app, state) = brain_server::app::build_app_with_state(
        &db_path,
        &templates_dir,
        None,
        project_root.clone(),
        output_dir,
        interact,
        brain_viz,
    );

    // Start cortex autonomous loop if enabled
    let shutdown = Arc::new(AtomicBool::new(false));
    if cortex_enabled {
        if let (Some(cv), Some(ca), Some(ce)) = (cached_v, cached_a, cached_e) {
            let shutdown_cortex = shutdown.clone();
            let db = state.db.clone();
            let live_tx = state.live_tx.clone();

            tokio::spawn(async move {
                tracing::info!(
                    n_clips = cv.nrows(),
                    has_speech = cached_s.is_some(),
                    has_props = cached_p.is_some(),
                    "Starting cortex autonomous loop"
                );
                brain_experiment::cortex::run_cortex(
                    db,
                    cv,
                    ca,
                    ce,
                    cached_s,
                    cached_p,
                    whitened,
                    live_tx,
                    shutdown_cortex,
                    {
                        let mut cfg = brain_experiment::cortex::CortexConfig::default();
                        let sweep_path = project_root.join("sweep_queue.json");
                        let sweep_path = sweep_path.to_string_lossy().to_string();
                        if std::path::Path::new(&sweep_path).exists() {
                            tracing::info!(path = %sweep_path, "Sweep queue found");
                            cfg.sweep_file = Some(sweep_path);
                            cfg.sweep_screen_steps = 20000;
                        }
                        cfg
                    },
                )
                .await;
            });
        } else {
            tracing::error!("No embeddings available — cortex disabled");
        }
    }

    // Start web server
    if ssl_cert.exists() && ssl_key.exists() {
        tracing::info!("Starting brain-server with TLS on {bind_addr}");
        let tls_config = axum_server::tls_rustls::RustlsConfig::from_pem_file(&ssl_cert, &ssl_key)
            .await
            .expect("Failed to load TLS config");

        let addr: std::net::SocketAddr = bind_addr.parse().expect("Invalid bind address");
        axum_server::bind_rustls(addr, tls_config)
            .serve(app.into_make_service())
            .await
            .expect("Server error");
    } else {
        tracing::info!("Starting brain-server (no TLS) on {bind_addr}");
        let listener = tokio::net::TcpListener::bind(&bind_addr).await.expect("Failed to bind");
        axum::serve(listener, app).await.expect("Server error");
    }

    shutdown.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Compute brain visualization data from real projected embeddings.
fn compute_brain_viz(
    v: &ndarray::Array2<f32>,
    a: &ndarray::Array2<f32>,
    e: &Option<Arc<ndarray::Array2<f32>>>,
    s: &Option<Arc<ndarray::Array2<f32>>>,
    p: &Option<Arc<ndarray::Array2<f32>>>,
    labels: Vec<String>,
) -> BrainViz {
    use ndarray::Array2;
    use std::collections::HashMap;

    let n = v.nrows();
    let d = v.ncols();

    // Sample up to 2000 clips for PCA scatter
    let sample_size = n.min(2000);
    let step = if n > sample_size { n / sample_size } else { 1 };
    let sample_indices: Vec<usize> = (0..n).step_by(step).take(sample_size).collect();

    // PCA: compute top-2 components from vision embeddings
    // Use power iteration for speed (no full SVD)
    let pca_v = compute_pca_2d(v, &sample_indices);
    let pca_a = compute_pca_2d(a, &sample_indices);

    // Sample labels and category mapping
    let pca_labels: Vec<String> = sample_indices.iter().map(|&i| {
        if i < labels.len() { labels[i].clone() } else { format!("clip #{i}") }
    }).collect();

    // Build category color index (top 30 categories by frequency + "other")
    let mut cat_counts: HashMap<String, usize> = HashMap::new();
    for l in &pca_labels {
        *cat_counts.entry(l.clone()).or_default() += 1;
    }
    let mut cat_sorted: Vec<_> = cat_counts.into_iter().collect();
    cat_sorted.sort_by(|a, b| b.1.cmp(&a.1));
    let top_cats: HashMap<String, u8> = cat_sorted.iter().take(30)
        .enumerate()
        .map(|(i, (k, _))| (k.clone(), i as u8))
        .collect();
    let pca_category_idx: Vec<u8> = pca_labels.iter()
        .map(|l| *top_cats.get(l).unwrap_or(&30))
        .collect();

    // Cross-correlation matrices: C_xy = (1/N) X^T @ Y, downsampled to 32x32
    let ds = 32; // downsample size
    let block = d / ds; // 512/32 = 16 elements per block
    let pairs_data: Vec<(&str, &Array2<f32>, &Array2<f32>)> = {
        let mut pairs: Vec<(&str, &Array2<f32>, &Array2<f32>)> = vec![
            ("M_va", v, a),
        ];
        if let Some(em) = e {
            pairs.push(("M_ve", v, em));
            pairs.push(("M_ae", a, em));
        }
        if let Some(sm) = s {
            pairs.push(("M_vs", v, sm));
            pairs.push(("M_as", a, sm));
            if let Some(em) = e {
                pairs.push(("M_es", em.as_ref(), sm));
            }
        }
        if let Some(pm) = p {
            pairs.push(("M_vp", v, pm));
            pairs.push(("M_ap", a, pm));
            if let Some(em) = e {
                pairs.push(("M_ep", em.as_ref(), pm));
            }
            if let Some(sm) = s {
                pairs.push(("M_sp", sm.as_ref(), pm));
            }
        }
        pairs
    };

    let inv_n = 1.0 / n as f32;
    let mut cross_corr = HashMap::new();
    let mut weight_histograms = HashMap::new();
    let mut global_min = 0.0f32;
    let mut global_max = 0.0f32;

    for &(name, x, y) in &pairs_data {
        // Compute full cross-correlation: C = X^T @ Y / N  [D, D]
        let mut cc = Array2::<f32>::zeros((d, d));
        ndarray::linalg::general_mat_mul(inv_n, &x.t(), &y.view(), 0.0f32, &mut cc);

        // Track global range for histogram
        for &val in cc.iter() {
            if val < global_min { global_min = val; }
            if val > global_max { global_max = val; }
        }

        // Downsample to 32x32 by block averaging
        let mut ds_matrix = vec![vec![0.0f32; ds]; ds];
        let cc_slice = cc.as_slice().unwrap();
        for bi in 0..ds {
            for bj in 0..ds {
                let mut sum = 0.0f32;
                for di in 0..block {
                    for dj in 0..block {
                        let row = bi * block + di;
                        let col = bj * block + dj;
                        sum += cc_slice[row * d + col];
                    }
                }
                ds_matrix[bi][bj] = sum / (block * block) as f32;
            }
        }
        cross_corr.insert(String::from(name), ds_matrix);

        // Weight histogram (50 bins)
        let n_bins = 50usize;
        let range = (global_max - global_min).max(1e-10);
        let mut hist = vec![0u32; n_bins];
        for &val in cc.iter() {
            let bin = (((val - global_min) / range) * (n_bins - 1) as f32) as usize;
            hist[bin.min(n_bins - 1)] += 1;
        }
        weight_histograms.insert(name.to_string(), hist);
    }

    // Per-clip similarity: v_i @ a_i (diagonal of V @ A^T)
    let sample_sims: Vec<f32> = sample_indices.iter().map(|&i| {
        v.row(i).iter().zip(a.row(i).iter()).map(|(a, b)| a * b).sum()
    }).collect();

    BrainViz {
        pca_v,
        pca_a,
        pca_labels,
        category_colors: top_cats,
        pca_category_idx,
        cross_corr,
        weight_histograms,
        hist_min: global_min,
        hist_max: global_max,
        sample_sims,
        n_clips: n,
    }
}

/// Fast 2D PCA via power iteration (2 components).
fn compute_pca_2d(data: &ndarray::Array2<f32>, indices: &[usize]) -> Vec<[f32; 2]> {
    let d = data.ncols();
    let n = indices.len();

    // Compute mean
    let mut mean = vec![0.0f32; d];
    for &i in indices {
        let row = data.row(i);
        for (j, &v) in row.iter().enumerate() {
            mean[j] += v;
        }
    }
    let inv_n = 1.0 / n as f32;
    for v in &mut mean { *v *= inv_n; }

    // Power iteration for top-2 eigenvectors of (centered data)^T @ (centered data)
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut components: Vec<Vec<f32>> = Vec::new();

    for _comp in 0..2 {
        // Random initial vector
        let mut w: Vec<f32> = (0..d).map(|_| rand::Rng::random::<f32>(&mut rng) - 0.5).collect();

        // 20 iterations of power method
        for _ in 0..20 {
            // Compute X^T @ X @ w  (without forming X^T @ X)
            // First: z = X @ w  [n]
            let z: Vec<f32> = indices.iter().map(|&i| {
                let row = data.row(i);
                row.iter().zip(mean.iter()).zip(w.iter())
                    .map(|((&x, &m), &wi)| (x - m) * wi).sum()
            }).collect();

            // Then: w_new = X^T @ z  [d]
            let mut w_new = vec![0.0f32; d];
            for (idx, &i) in indices.iter().enumerate() {
                let row = data.row(i);
                let zi = z[idx];
                for j in 0..d {
                    w_new[j] += (row[j] - mean[j]) * zi;
                }
            }

            // Remove previous components (deflation)
            for prev_w in &components {
                let dot: f32 = w_new.iter().zip(prev_w.iter()).map(|(a, b)| a * b).sum();
                for j in 0..d {
                    w_new[j] -= dot * prev_w[j];
                }
            }

            // Normalize
            let norm: f32 = w_new.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for v in &mut w_new { *v /= norm; }
            w = w_new;
        }
        components.push(w);
    }

    // Project samples onto 2 components
    indices.iter().map(|&i| {
        let row = data.row(i);
        let x: f32 = row.iter().zip(mean.iter()).zip(components[0].iter())
            .map(|((&x, &m), &w)| (x - m) * w).sum();
        let y: f32 = row.iter().zip(mean.iter()).zip(components[1].iter())
            .map(|((&x, &m), &w)| (x - m) * w).sum();
        [x, y]
    }).collect()
}

fn find_safetensors(dir: &std::path::Path) -> Option<PathBuf> {
    if !dir.exists() {
        return None;
    }
    // Prefer expanded_embeddings.safetensors (the main embeddings file)
    let preferred = dir.join("expanded_embeddings.safetensors");
    if preferred.exists() {
        return Some(preferred);
    }
    // Fallback: find any .safetensors that contains embeddings (skip whitening/labels)
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| {
            let path = e.path();
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            path.extension().is_some_and(|ext| ext == "safetensors")
                && !name.contains("whitening")
                && !name.contains("labels")
        })
        .collect();
    entries.sort_by_key(|e| std::cmp::Reverse(e.metadata().ok().and_then(|m| m.modified().ok())));
    entries.first().map(|e| e.path())
}

/// Load VGGSound labels from CSV, mapping embedding indices to labels.
///
/// The data_inventory table stores clip_ids in the same order as the embedding matrix.
/// We read clip_ids from the DB, then look up labels from the VGGSound CSV.
fn load_vggsound_labels(project_root: &std::path::Path, n_clips: usize) -> Vec<String> {
    let csv_path = project_root.join("data/vggsound/vggsound.csv");

    // Build clip_id → label map from VGGSound CSV
    // Also build expanded_id → label map for clips downloaded by expand_embeddings.py
    let mut id_to_label: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut all_csv_entries: Vec<(String, String)> = Vec::new(); // (clip_id, label)
    if let Ok(content) = std::fs::read_to_string(&csv_path) {
        for line in content.lines() {
            let parts: Vec<&str> = line.splitn(4, ',').collect();
            if parts.len() >= 3 {
                let yt_id = parts[0];
                let timestamp = parts[1].trim();
                let label = parts[2].trim_matches('"').to_string();
                let clip_id = format!("{yt_id}_{timestamp:0>6}");
                id_to_label.insert(clip_id.clone(), label.clone());
                all_csv_entries.push((clip_id, label));
            }
        }
        tracing::info!(csv_labels = id_to_label.len(), "Parsed VGGSound CSV");
    } else {
        tracing::warn!(?csv_path, "VGGSound CSV not found");
    }

    // Read clip_ids from data_inventory in insertion order
    let db_path = project_root.join("outputs/cortex/knowledge.db");
    if let Ok(db) = brain_db::KnowledgeBase::new(&db_path) {
        if let Ok(clip_ids) = db.get_clip_ids_ordered(n_clips as i64) {
            // For expanded clips, try to recover labels from the VGGSound CSV
            // The expand script processes CSV entries in order, so expanded_N
            // corresponds to roughly the Nth downloaded clip from the CSV.
            // We can't recover exact mapping, but we label them as their category.
            return clip_ids.iter().map(|cid| {
                if let Some(label) = id_to_label.get(cid) {
                    label.clone()
                } else if cid.starts_with("ravdess_") {
                    "speech/song (RAVDESS)".to_string()
                } else {
                    // Unknown expanded clip
                    format!("clip #{}", cid.trim_start_matches("expanded_"))
                }
            }).collect();
        }
    }

    // Fallback: unlabeled
    (0..n_clips).map(|i| format!("clip #{i}")).collect()
}
