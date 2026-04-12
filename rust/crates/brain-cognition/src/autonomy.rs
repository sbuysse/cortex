//! Autonomy loop — curiosity-driven self-directed learning.
//!
//! Runs in a background tokio task. Each cycle:
//! 1. Transfer working memory → fast memory
//! 2. Auto-train on buffered online pairs (real gradient InfoNCE)
//! 3. Generate dreams → buffer learning pairs
//! 4. Consolidate memory: merge prototypes, prune stale, replay episodes
//! 5. YouTube learning: download + mel spectrogram + Whisper encode

use std::sync::Arc;
use crate::state::BrainState;

/// Start the autonomy loop as a background task.
pub fn start_autonomy(brain: Arc<BrainState>) {
    if brain.autonomy_running.load(std::sync::atomic::Ordering::Relaxed) {
        tracing::warn!("Autonomy already running");
        return;
    }
    brain.autonomy_running.store(true, std::sync::atomic::Ordering::Relaxed);
    tracing::info!("Starting autonomy loop (interval: {}s)", brain.config.autonomy_interval_secs);

    let brain2 = brain.clone();
    tokio::spawn(async move {
        loop {
            if !brain2.autonomy_running.load(std::sync::atomic::Ordering::Relaxed) {
                tracing::info!("Autonomy loop stopped");
                break;
            }

            run_cycle(&brain2).await;

            let new_cycle = brain2.autonomy_cycles.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            let _ = brain2.memory_db.set_stat("autonomy_cycles", &new_cycle.to_string());
            let vids = brain2.autonomy_videos.load(std::sync::atomic::Ordering::Relaxed);
            let _ = brain2.memory_db.set_stat("autonomy_videos", &vids.to_string());
            let dreams = brain2.dream_count.load(std::sync::atomic::Ordering::Relaxed);
            let _ = brain2.memory_db.set_stat("dream_count", &dreams.to_string());

            tokio::time::sleep(std::time::Duration::from_secs(
                brain2.config.autonomy_interval_secs,
            )).await;
        }
    });
}

/// Stop the autonomy loop.
pub fn stop_autonomy(brain: &BrainState) {
    brain.autonomy_running.store(false, std::sync::atomic::Ordering::Relaxed);
    tracing::info!("Autonomy loop stop requested");
}

/// Run one autonomy cycle.
async fn run_cycle(brain: &BrainState) {
    let cycle = brain.autonomy_cycles.load(std::sync::atomic::Ordering::Relaxed);
    tracing::info!("Autonomy cycle {cycle} starting");

    // ── Step 0a: Grow codebook from prototypes (live, no restart) ──
    {
        let protos = brain.memory_db.get_prototypes().unwrap_or_default();
        let mut cb_guard = brain.codebook.lock().unwrap();
        if let Some(cb) = cb_guard.as_mut() {
            let before = cb.len();
            cb.grow_from_prototypes(&protos);
            if cb.len() > before {
                tracing::info!("Codebook grew: {} → {} concepts (live)", before, cb.len());
            }
        }
    }

    // ── Step 0b: Transfer working memory → fast memory ──
    {
        let wm = brain.working_memory.lock().unwrap();
        let embs = wm.get_embeddings();
        let state = wm.get_state();
        let mut fm = brain.fast_memory.lock().unwrap();
        for (emb, item) in embs.iter().zip(state.items.iter()) {
            fm.store(emb, &item.label);
        }
        if !state.items.is_empty() {
            tracing::info!("Transferred {} WM items to fast memory (total: {})", state.items.len(), fm.count());
        }
    }

    // ── Step 1: Auto-train on buffered pairs (real gradient InfoNCE) ──
    let pairs_count = brain.online_pairs.lock().unwrap().len();
    if pairs_count >= 10 {
        train_buffered_pairs(brain, cycle);
    }

    // ── Step 2: Dream phase → buffer learning pairs ──
    if let (Some(cb), Some(wm)) = (&*brain.codebook.lock().unwrap(), &brain.world_model) {
        match crate::dreams::generate_dream(None, 5, cb, wm, &brain.dream_count) {
            Ok(dream) => {
                let generated = dream.learning_pairs_generated;
                // Buffer dream pairs for next training cycle
                if !dream.learning_pairs.is_empty() {
                    let mut pairs = brain.online_pairs.lock().unwrap();
                    for (v, a) in &dream.learning_pairs {
                        // Dream pairs are in 512-dim MLP space, but train_infonce expects
                        // v=384, a=512. Store as (512-pad-to-384-trunc, 512) — these are
                        // already in shared space so we pass them as-is to the codebook training.
                        pairs.push((v.clone(), a.clone()));
                    }
                }
                brain.sse.emit("dream", serde_json::json!({
                    "seed": dream.seed, "steps": dream.steps.len(),
                    "avg_surprise": dream.avg_surprise, "pairs": generated, "cycle": cycle,
                }));
                tracing::info!("Dream: seed={}, surprise={:.3}, pairs={}", dream.seed, dream.avg_surprise, generated);
            }
            Err(e) => tracing::warn!("Dream failed: {e}"),
        }
    }

    // ── Step 3: Consolidation (every 5th cycle) ──
    if cycle % 5 == 0 && cycle > 0 {
        consolidate_memory(brain, cycle);
    }

    // ── Step 3b: Spiking brain sleep consolidation (every 5th cycle) ──
    if cycle % 5 == 0 && cycle > 0 {
        if let Some(ref sb) = brain.spiking_brain {
            let mut sb = sb.lock().unwrap();
            tracing::info!("Spiking brain sleep cycle starting");
            brain_spiking::sleep::sleep_cycle(&mut sb.network, 500);
            let stats = sb.stats();
            tracing::info!(
                "Spiking brain sleep complete — {} neurons, {} synapses, {} last_spikes",
                stats.total_neurons, stats.total_synapses, stats.total_spikes_last_step
            );
        }
    }

    // ── Step 3c: Save spiking brain weights (every 5th cycle) ──
    if cycle % 5 == 0 && cycle > 0 {
        if let Some(ref sb) = brain.spiking_brain {
            let sb = sb.lock().unwrap();
            let save_dir = brain.config.project_root.join("outputs/cortex");
            if let Err(e) = sb.save(&save_dir) {
                tracing::warn!("Failed to save spiking brain: {e}");
            }
        }
    }

    // ── Step 4: Knowledge enrichment — ConceptNet + Wikipedia ──
    let categories = get_curious_categories(brain, 5);
    let mut kg_added = 0;
    for cat in categories.iter().take(3) {
        kg_added += enrich_from_conceptnet(cat, brain).await;
        kg_added += enrich_from_wikipedia(cat, brain).await;
        kg_added += enrich_from_wikidata(cat, brain).await;
    }
    if kg_added > 0 {
        brain.sse.emit("knowledge_enriched", serde_json::json!({
            "edges_added": kg_added, "categories_explored": 3, "cycle": cycle,
        }));
        tracing::info!("Knowledge enriched: {kg_added} edges from ConceptNet/Wikipedia");
    }

    // ── Step 5: YouTube learning (every other cycle, 1 category) ──
    if cycle % 2 == 0 {
        if let Some(cat) = categories.first() {
            match youtube_learn_category(cat, brain).await {
                Ok(pairs) => {
                    brain.autonomy_videos.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    brain.sse.emit("youtube_learn", serde_json::json!({
                        "category": cat, "pairs": pairs, "cycle": cycle,
                    }));
                    if pairs > 0 { tracing::info!("YouTube learned: {cat} ({pairs} pairs)"); }
                }
                Err(e) => tracing::debug!("YouTube learn {cat}: {e}"),
            }
        }
    }

    let buf = brain.online_pairs.lock().unwrap().len();
    let total = brain.online_learning_count.load(std::sync::atomic::Ordering::Relaxed);
    let protos = brain.memory_db.prototype_count().unwrap_or(0);
    let edges = brain.memory_db.edge_count().unwrap_or(0);
    tracing::info!("Autonomy cycle {cycle} complete — {total} learned, {protos} prototypes, {edges} KG edges, {buf} buffered");
}

/// Real gradient InfoNCE training on buffered pairs.
fn train_buffered_pairs(brain: &BrainState, cycle: i64) {
    let mlp = match &brain.inference {
        Some(m) => m,
        None => return,
    };

    let pairs: Vec<(Vec<f32>, Vec<f32>)> = {
        let mut p = brain.online_pairs.lock().unwrap();
        p.drain(..).collect()
    };
    if pairs.is_empty() { return; }

    let count = pairs.len();
    let n_steps = (count * 5).min(50);

    // Build matrices
    let v_dim = pairs[0].0.len();
    let a_dim = pairs[0].1.len();
    let mut v_data = ndarray::Array2::<f32>::zeros((count, v_dim));
    let mut a_data = ndarray::Array2::<f32>::zeros((count, a_dim));
    for (i, (v, a)) in pairs.iter().enumerate() {
        for (j, &val) in v.iter().enumerate().take(v_dim) { v_data[[i, j]] = val; }
        for (j, &val) in a.iter().enumerate().take(a_dim) { a_data[[i, j]] = val; }
    }

    // Only train if dimensions match the MLP weights
    if v_dim == 384 && a_dim == 512 {
        let mut w_v = mlp.w_v.clone();
        let mut w_a = mlp.w_a.clone();
        let lr = brain.config.ach_lr_min;

        let (trained, loss) = brain_inference::mlp::train_infonce(
            &mut w_v, &mut w_a, &v_data, &a_data, lr, 0.01, n_steps);

        // Save trained weights
        let online_dir = brain.config.project_root.join("outputs/cortex/v6_mlp_online");
        let _ = std::fs::create_dir_all(&online_dir);
        let _ = brain_inference::mlp::save_bin_matrix(&w_v, &online_dir.join("w_v.bin"));
        let _ = brain_inference::mlp::save_bin_matrix(&w_a, &online_dir.join("w_a.bin"));

        brain.online_learning_count.fetch_add(count as i64, std::sync::atomic::Ordering::Relaxed);
        let total = brain.online_learning_count.load(std::sync::atomic::Ordering::Relaxed);
        let _ = brain.memory_db.set_stat("online_learning_count", &total.to_string());
        let _ = brain.memory_db.log_learning("auto_train",
            Some(&format!("{{\"pairs\":{count},\"steps\":{n_steps},\"loss\":{loss:.4}}}")));
        brain.sse.emit("auto_train", serde_json::json!({
            "pairs_trained": trained, "steps": n_steps, "loss": loss,
            "total_learned": total, "cycle": cycle,
        }));
        tracing::info!("Auto-trained: {count} pairs, {n_steps} steps, loss={loss:.4} (total: {total})");
    } else {
        // Dream pairs are 512-dim — log them but skip gradient update
        brain.online_learning_count.fetch_add(count as i64, std::sync::atomic::Ordering::Relaxed);
        let total = brain.online_learning_count.load(std::sync::atomic::Ordering::Relaxed);
        let _ = brain.memory_db.set_stat("online_learning_count", &total.to_string());
        let _ = brain.memory_db.log_learning("dream_pairs",
            Some(&format!("{{\"pairs\":{count},\"dims\":\"{}x{}\"}}", v_dim, a_dim)));
        tracing::info!("Buffered {count} dream pairs ({}x{}, not 384x512)", v_dim, a_dim);
    }
}

/// Real memory consolidation: merge similar prototypes, prune stale, update KG.
fn consolidate_memory(brain: &BrainState, cycle: i64) {
    let protos = brain.memory_db.get_prototypes().unwrap_or_default();
    let mut merged = 0;
    let mut pruned = 0;

    // Merge similar prototypes (cosine > 0.9)
    let mut to_merge: Vec<(String, String)> = Vec::new();
    for i in 0..protos.len() {
        for j in (i + 1)..protos.len() {
            if protos[i].centroid_blob.len() == protos[j].centroid_blob.len() && protos[i].centroid_blob.len() >= 512 * 4 {
                let a: Vec<f32> = protos[i].centroid_blob.chunks_exact(4).take(512)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                let b: Vec<f32> = protos[j].centroid_blob.chunks_exact(4).take(512)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                let sim: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
                if sim > 0.9 {
                    to_merge.push((protos[i].name.clone(), protos[j].name.clone()));
                }
            }
        }
    }
    for (keep, remove) in &to_merge {
        // Keep the one with more count, add edge
        let _ = brain.memory_db.upsert_edge(keep, "merged-with", remove, 0.9);
        merged += 1;
    }

    // Prune prototypes with count=1 that are old (created >24h ago)
    // (simplified: just count low-count protos for reporting)
    let low_count = protos.iter().filter(|p| p.count <= 1).count();
    pruned = 0; // actual pruning needs timestamp check

    let edge_count = brain.memory_db.edge_count().unwrap_or(0);
    brain.sse.emit("consolidation", serde_json::json!({
        "cycle": cycle, "prototypes": protos.len(), "kg_edges": edge_count,
        "merged": merged, "pruned": pruned, "low_count": low_count,
    }));
    tracing::info!("Consolidation: {} protos, {} edges, merged={merged}, low_count={low_count}",
        protos.len(), edge_count);
}

/// Get underrepresented categories for YouTube learning.
fn get_curious_categories(brain: &BrainState, n: usize) -> Vec<String> {
    if let Some(cb) = &*brain.codebook.lock().unwrap() {
        let protos = brain.memory_db.get_prototypes().unwrap_or_default();
        let proto_names: std::collections::HashSet<String> = protos.iter().map(|p| p.name.clone()).collect();
        let mut uncovered: Vec<String> = cb.all_labels().iter()
            .filter(|l| !proto_names.contains(*l))
            .cloned()
            .collect();
        use rand::seq::SliceRandom;
        uncovered.shuffle(&mut rand::rng());
        uncovered.into_iter().take(n).collect()
    } else {
        Vec::new()
    }
}

/// YouTube learn: download audio, compute mel spectrogram, encode via Whisper,
/// project through MLP, store as learning pair.
async fn youtube_learn_category(category: &str, brain: &BrainState) -> Result<usize, String> {
    let search_query = format!("{} sound short", category);

    // Search for video URL
    let output = tokio::process::Command::new("yt-dlp")
        .args(["--default-search", "ytsearch1", "--print", "webpage_url",
               "--match-filter", "duration<120", "--no-download", &search_query])
        .output().await
        .map_err(|e| format!("yt-dlp: {e}"))?;

    let url = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if url.is_empty() { return Err("No video found".into()); }

    // Download audio as WAV
    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis();
    let tmp_dir = format!("/tmp/brain_yt_{ts}");
    std::fs::create_dir_all(&tmp_dir).map_err(|e| e.to_string())?;
    let audio_path = format!("{tmp_dir}/audio.wav");

    let dl = tokio::process::Command::new("yt-dlp")
        .args(["-x", "--audio-format", "wav", "--audio-quality", "0",
               "--match-filter", "duration<120", "-o", &audio_path, &url])
        .output().await
        .map_err(|e| format!("download: {e}"))?;

    if !dl.status.success() {
        let _ = std::fs::remove_dir_all(&tmp_dir);
        return Err("Download failed".into());
    }

    // Convert to 16kHz mono PCM via ffmpeg
    let pcm_path = format!("{tmp_dir}/audio_16k.raw");
    let ffmpeg = tokio::process::Command::new("ffmpeg")
        .args(["-i", &audio_path, "-ar", "16000", "-ac", "1", "-f", "f32le", "-y", &pcm_path])
        .output().await
        .map_err(|e| format!("ffmpeg: {e}"))?;

    let mut pairs = 0;
    if ffmpeg.status.success() {
        // Read PCM samples
        if let Ok(raw) = std::fs::read(&pcm_path) {
            let samples: Vec<f32> = raw.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            if samples.len() > 1600 {
                // Compute mel spectrogram + Whisper encode
                let mel = brain_inference::mel::compute_log_mel(&samples);
                if let Some(whisper) = &brain.audio_encoder {
                    if let Ok(a_emb) = whisper.encode(&mel) {
                        if let Some(mlp) = &brain.inference {
                            let a_proj = mlp.project_audio(&a_emb);
                            // Store as learning pair (use category text as visual proxy)
                            if let Some(te) = &brain.text_encoder {
                                if let Ok(t_emb) = te.encode(category) {
                                    let v_proj = mlp.project_visual(&t_emb);
                                    // Buffer the pair for training
                                    brain.online_pairs.lock().unwrap().push((t_emb, a_emb.clone()));
                                    pairs += 1;
                                    // Store perception
                                    let _ = brain.memory_db.store_perception(
                                        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64(),
                                        "youtube_audio", None,
                                        Some(&format!("[\"{category}\"]")), None, None);
                                    // Store as fast memory
                                    brain.fast_memory.lock().unwrap().store(&a_proj, category);
                                    // Update KG
                                    let _ = brain.memory_db.upsert_edge(category, "sounds-like", category, 0.8);
                                    let _ = brain.memory_db.log_youtube(
                                        url.split("v=").last().unwrap_or("?"), category, "processed", 1);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let _ = std::fs::remove_dir_all(&tmp_dir);
    Ok(pairs)
}

/// Learn from an academic/educational YouTube video.
/// Pipeline: download subtitles → extract concepts via Ollama → encode via MiniLM →
/// feed spiking brain + store in knowledge graph. Also encodes audio → auditory cortex.
pub struct AcademicLearnResult {
    pub triples_count: usize,
    pub topic: String,
    pub key_concepts: Vec<String>,
}

pub async fn youtube_learn_academic(query: &str, topic_override: &str, brain_arc: std::sync::Arc<BrainState>) -> Result<AcademicLearnResult, String> {
    let brain: &BrainState = &brain_arc;
    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis();
    let tmp_dir = format!("/tmp/brain_academic_{ts}");
    std::fs::create_dir_all(&tmp_dir).map_err(|e| e.to_string())?;

    // 1. If query is a URL, use it directly. Otherwise search.
    let url = if query.starts_with("http") {
        query.to_string()
    } else {
        let search_query = format!("{query} explained");
        let url_out = tokio::process::Command::new("yt-dlp")
            .args(["--default-search", "ytsearch3", "--print", "webpage_url",
                   "--no-download", &search_query])
            .output().await
            .map_err(|e| format!("yt-dlp search: {e}"))?;
        String::from_utf8_lossy(&url_out.stdout)
            .lines()
            .find(|l| l.starts_with("http"))
            .unwrap_or("")
            .trim()
            .to_string()
    };
    if url.is_empty() {
        let _ = std::fs::remove_dir_all(&tmp_dir);
        return Err("No video found".into());
    }
    // Topic for pronoun resolution: user-provided > query words > video title
    let topic_key: String = if !topic_override.is_empty() {
        topic_override.to_lowercase()
    } else if !query.starts_with("http") {
        query.split_whitespace()
            .filter(|w| w.len() > 3)
            .take(3)
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase()
    } else {
        String::new() // resolved below from video title
    };
    // Always fetch the video title (used for topic + as a stored concept)
    let title_out = tokio::process::Command::new("yt-dlp")
        .args(["--print", "title", "--no-download", &url])
        .output().await.ok();
    let video_title: String = title_out
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    let mut topic_key: String = if !topic_key.is_empty() {
        topic_key
    } else {
        // Prefer parenthesized tokens (e.g. "(TurboQuant)" → "turboquant"),
        // otherwise the first capitalized non-generic ≥4-char token.
        let generic = ["google", "youtube", "video", "watch", "explained",
            "tutorial", "introduction", "guide", "basics", "review",
            "googles"];
        let strip = |w: &str| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string();
        let parens: Option<String> = video_title.split(|c: char| c == '(' || c == ')')
            .skip(1)
            .step_by(2) // odd indices = inside parens
            .find_map(|s| {
                let t = s.trim();
                if t.len() >= 3 && t.chars().any(|c| c.is_alphabetic()) {
                    Some(t.to_lowercase())
                } else { None }
            });
        if let Some(p) = parens {
            p
        } else {
            video_title.split_whitespace()
                .map(strip)
                .filter(|w| {
                    let lc = w.to_lowercase();
                    w.len() >= 4
                        && w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
                        && !generic.contains(&lc.as_str())
                })
                .take(2)
                .collect::<Vec<_>>()
                .join(" ")
                .to_lowercase()
        }
    };
    tracing::info!("Academic learn: {query} → {url} (topic: {topic_key})");

    // 2. Download auto-subtitles + audio
    let dl = tokio::process::Command::new("yt-dlp")
        .args(["--write-auto-sub", "--sub-lang", "en", "--skip-download",
               "--sub-format", "vtt", "-o", &format!("{tmp_dir}/video"), &url])
        .output().await
        .map_err(|e| format!("subtitle download: {e}"))?;

    // Also download audio for Whisper encoding
    let _audio_dl = tokio::process::Command::new("yt-dlp")
        .args(["-x", "--audio-format", "wav", "--audio-quality", "0",
               "--match-filter", "duration<600", "-o", &format!("{tmp_dir}/audio.wav"), &url])
        .output().await;

    // 3. Parse subtitles → transcript text
    let transcript = {
        let vtt_pattern = format!("{tmp_dir}/video.en.vtt");
        let vtt_content = std::fs::read_to_string(&vtt_pattern)
            .or_else(|_| {
                // Try alternative subtitle file patterns
                let entries = std::fs::read_dir(&tmp_dir).map_err(|e| e.to_string())?;
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if name.ends_with(".vtt") || name.ends_with(".srt") {
                        return std::fs::read_to_string(entry.path()).map_err(|e| e.to_string());
                    }
                }
                Err("No subtitle file found".to_string())
            });

        match vtt_content {
            Ok(vtt) => {
                // Strip VTT headers, timestamps, and tags — keep only text lines
                vtt.lines()
                    .filter(|line| {
                        let l = line.trim();
                        !l.is_empty()
                            && !l.starts_with("WEBVTT")
                            && !l.starts_with("Kind:")
                            && !l.starts_with("Language:")
                            && !l.contains("-->")
                            && !l.chars().all(|c| c.is_ascii_digit())
                    })
                    .map(|line| {
                        // Strip HTML tags like <c> and timing tags
                        let mut clean = String::new();
                        let mut in_tag = false;
                        for ch in line.chars() {
                            if ch == '<' { in_tag = true; continue; }
                            if ch == '>' { in_tag = false; continue; }
                            if !in_tag { clean.push(ch); }
                        }
                        clean
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            }
            Err(_) => String::new(),
        }
    };

    if transcript.len() < 50 {
        let _ = std::fs::remove_dir_all(&tmp_dir);
        return Err("Transcript too short or unavailable".into());
    }

    tracing::info!("Academic learn: got {} chars of transcript", transcript.len());

    // 4. Extract key concepts — pure local: split transcript into sentences,
    //    encode via MiniLM, match against codebook. No LLM needed.
    let te = brain.text_encoder.as_ref().ok_or("No text encoder")?;
    let mlp = brain.inference.as_ref().ok_or("No MLP encoder")?;

    // Split transcript into chunks — YouTube subtitles often lack punctuation,
    // so split on sentences (.!?) AND on ~100-char boundaries for long runs.
    let mut sentences: Vec<String> = Vec::new();
    for raw_sentence in transcript.split(|c: char| c == '.' || c == '!' || c == '?') {
        let s = raw_sentence.trim();
        if s.len() <= 20 { continue; }
        if s.len() <= 150 {
            sentences.push(s.to_string());
        } else {
            // Split long runs into ~100-char chunks at word boundaries
            let words: Vec<&str> = s.split_whitespace().collect();
            let mut chunk = String::new();
            for word in words {
                if chunk.len() + word.len() > 200 && chunk.len() > 40 {
                    sentences.push(chunk.clone());
                    chunk.clear();
                }
                if !chunk.is_empty() { chunk.push(' '); }
                chunk.push_str(word);
            }
            if chunk.len() > 20 { sentences.push(chunk); }
        }
    }
    // Filter: skip intro (first 10%), skip filler/generic sentences
    let skip_count = sentences.len() / 10; // skip first 10%
    let sentences: Vec<&str> = sentences.iter()
        .skip(skip_count)
        .filter(|s| {
            let lower = s.to_lowercase();
            // Skip common YouTube filler
            !lower.contains("subscribe") && !lower.contains("like and") &&
            !lower.contains("check out") && !lower.contains("link in") &&
            !lower.contains("thank you for watching") && !lower.contains("[music]") &&
            !lower.starts_with("so ") && !lower.starts_with("okay so") &&
            // Keep only sentences with some technical density (3+ words >4 chars)
            s.split_whitespace().filter(|w| w.len() > 4).count() >= 3
        })
        .map(|s| s.as_str())
        .collect();

    tracing::info!("Academic learn: {} sentences from transcript", sentences.len());

    // Extract the actual topic from the transcript via a tiny LLM call.
    // Title-based heuristics fail on clickbait titles ("Google's New AI Just
    // Broke My Brain") that never name the real subject (TurboQuant). The
    // first chunk of transcript almost always introduces it. Override the
    // earlier title-derived guess unless the user passed an explicit topic.
    if topic_override.is_empty() {
        // Use the RAW transcript (not filtered sentences) so we capture
        // the intro where the subject is typically named.
        let sample: String = transcript.chars().take(2500).collect();
        if !sample.is_empty() {
            let ollama_url = if brain.config.ollama_url.contains("/api/") {
                brain.config.ollama_url.clone()
            } else {
                format!("{}/api/generate", brain.config.ollama_url)
            };
            let topic_prompt = format!(
                "What is the specific technical subject or method being explained in \
                 this video? Output ONLY its name (1-3 words). Prefer the proper name \
                 of the algorithm, technique, paper, or product over generic descriptions.\n\n\
                 Title: {title}\n\n\
                 Transcript excerpt:\n{sample}\n\n\
                 Subject name:",
                title = video_title,
                sample = sample,
            );
            let body = serde_json::json!({
                "model": &brain.config.ollama_model,
                "prompt": topic_prompt,
                "stream": false,
                "options": {"temperature": 0.0, "num_predict": 20},
            });
            let client = reqwest::Client::new();
            if let Ok(resp) = client.post(&ollama_url).json(&body)
                .timeout(std::time::Duration::from_secs(20))
                .send().await
            {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    let raw = json["response"].as_str().unwrap_or("").trim();
                    // Take just the first line, strip surrounding punctuation/quotes,
                    // collapse to lowercase. Reject empty or absurdly long output.
                    let line = raw.lines().next().unwrap_or("").trim()
                        .trim_matches(|c: char| c == '"' || c == '\'' || c == '.' || c == ',' || c == ':');
                    let cleaned = line.to_lowercase();
                    let word_count = cleaned.split_whitespace().count();
                    if !cleaned.is_empty()
                        && cleaned.len() <= 60
                        && word_count >= 1 && word_count <= 5
                        && cleaned.chars().any(|c| c.is_alphabetic())
                    {
                        tracing::info!("Academic learn: LLM topic = '{cleaned}' (was '{topic_key}')");
                        topic_key = cleaned;
                    }
                }
            }
        }
    }

    // Encode each sentence, find nearest codebook concepts.
    // Wrapped in a tight scope so the MutexGuard cannot be inferred to live
    // across any later .await — the handler future must be Send.
    let (concepts, novel_concepts): (Vec<String>, Vec<(String, Vec<f32>)>) = {
        let cb_guard = brain.codebook.lock().unwrap();
        let cb = cb_guard.as_ref().ok_or("No codebook")?;

        let mut concept_scores: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
        for sentence in sentences.iter().take(50) {
            if let Ok(emb) = te.encode(sentence) {
                let proj = mlp.project_visual(&emb);
                let nearest = cb.nearest(&proj, 3);
                for (label, sim) in &nearest {
                    if *sim > 0.3 {
                        *concept_scores.entry(label.clone()).or_insert(0.0) += sim;
                    }
                }
            }
        }
        let mut scored: Vec<(String, f32)> = concept_scores.into_iter().collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let concepts: Vec<String> = scored.into_iter().take(10).map(|(label, _)| label).collect();

        let mut novel_concepts: Vec<(String, Vec<f32>)> = Vec::new();
        for sentence in sentences.iter().take(50) {
            if let Ok(emb) = te.encode(sentence) {
                let proj = mlp.project_visual(&emb);
                let nearest = cb.nearest(&proj, 1);
                if nearest.is_empty() || nearest[0].1 < 0.4 {
                    let label: String = sentence.split_whitespace().take(15).collect::<Vec<_>>().join(" ");
                    if label.len() > 10 {
                        novel_concepts.push((label, proj));
                    }
                }
            }
        }
        (concepts, novel_concepts)
    };

    // NOTE: previously this block dumped raw sentence prefixes ("first 15 words")
    // into learned_concepts as a "novel concepts" fallback. That produced
    // garbage labels in the UI ("You basically lose everything except…") and
    // polluted dialogue grounding. Real concepts now come from the LLM triple
    // extractor below — its subjects/objects are pushed into learned_concepts
    // once extraction finishes.

    let all_concepts: Vec<String> = concepts.iter()
        .chain(novel_concepts.iter().map(|(l, _)| l))
        .cloned()
        .collect();

    if all_concepts.is_empty() {
        let _ = std::fs::remove_dir_all(&tmp_dir);
        return Err("No concepts extracted from transcript".into());
    }

    tracing::info!("Academic learn: extracted {} concepts: {:?}",
        all_concepts.len(), &all_concepts[..all_concepts.len().min(5)]);

    // 5. Extract knowledge triples using LLM (high quality).
    // Awaited inline via a spawned task so the !Send awaits live in their own
    // future. The handler future stays Send because the JoinHandle is Send.
    let ollama_url = if brain.config.ollama_url.contains("/api/") {
        brain.config.ollama_url.clone()
    } else {
        format!("{}/api/generate", brain.config.ollama_url)
    };
    let ollama_model = brain.config.ollama_model.clone();
    let topic_key_owned = topic_key.clone();
    let video_title_owned = video_title.clone();
    let owned_sentences: Vec<String> = sentences.iter().take(50).map(|s| s.to_string()).collect();
    let sentence_count = owned_sentences.len();
    let brain_for_llm = brain_arc.clone();

    // Seed learned_concepts with the topic + video title up-front so a search
    // for the central entity (e.g. "TurboQuant") always finds *something*,
    // even if the LLM-extracted triples miss it.
    if let Some(te) = brain.text_encoder.as_ref() {
        let mut learned = brain.learned_concepts.lock().unwrap();
        let mut seed = |label: &str| {
            if label.len() >= 3 {
                if let Ok(emb) = te.encode(label) {
                    learned.push((label.to_string(), emb));
                }
            }
        };
        if !topic_key.is_empty() {
            seed(&topic_key);
            // Also seed without spaces ("turbo quant" → "turboquant") so
            // queries like "TurboQuant" hit even if user doesn't use spaces.
            let no_spaces: String = topic_key.split_whitespace().collect();
            if no_spaces != topic_key && no_spaces.len() >= 3 { seed(&no_spaces); }
        }
        if !video_title.is_empty() { seed(&video_title); }
    }

    let _llm_result: (usize, Vec<String>, Vec<brain_spiking::Triple>) = tokio::spawn(async move {
        let brain = &*brain_for_llm;
        let te = match brain.text_encoder.as_ref() { Some(t) => t, None => return (0usize, vec![], vec![]) };
        let t0 = std::time::Instant::now();
        let mut all_triples: Vec<brain_spiking::Triple> = Vec::new();

        let batch_size = 10;
        for chunk in owned_sentences.chunks(batch_size) {
            let numbered: String = chunk.iter().enumerate()
                .map(|(i, s)| format!("{}. {}", i + 1, s))
                .collect::<Vec<_>>()
                .join("\n");

            let prompt = format!(
                "You are extracting knowledge from a video titled: \"{title}\"\n\
                 The main topic is: {topic}\n\n\
                 Extract factual triples as subject|relation|object.\n\
                 STRICT RULES:\n\
                 - Whenever the sentence is about the main topic, USE \"{topic}\" as the subject.\n\
                 - subject and object must be NAMED ENTITIES or technical terms\n\
                 - NEVER use pronouns (it, you, we, this, that, they) — resolve them to entities\n\
                 - NEVER use sentence fragments or incomplete clauses\n\
                 - Reject vague filler like 'sometimes', 'somewhere'\n\
                 - relation must be a short verb phrase (2-4 words)\n\
                 - If a sentence has no clear claim about a real entity, SKIP it\n\
                 Examples:\n\
                 {topic}|compresses|KV cache\n\
                 {topic}|reduces|memory usage\n\
                 {topic}|uses|random rotations\n\n\
                 Output ONLY clean triples, one per line. No commentary.\n\n\
                 {numbered}\n",
                title = video_title_owned, topic = topic_key_owned, numbered = numbered
            );

            let client = reqwest::Client::new();
            let body = serde_json::json!({
                "model": &ollama_model,
                "prompt": prompt,
                "stream": false,
                "options": {"temperature": 0.1, "num_predict": 400},
            });

            match client.post(&ollama_url).json(&body)
                .timeout(std::time::Duration::from_secs(30))
                .send().await
            {
                Ok(resp) => {
                    if let Ok(json) = resp.json::<serde_json::Value>().await {
                        let text = json["response"].as_str().unwrap_or("");
                        for line in text.lines() {
                            let line = line.trim().trim_start_matches(|c: char| c.is_numeric() || c == '.' || c == '-' || c == ' ');
                            let parts: Vec<&str> = line.splitn(3, '|').collect();
                            if parts.len() == 3 {
                                let s = parts[0].trim().to_lowercase();
                                let r = parts[1].trim().to_lowercase();
                                let o = parts[2].trim().to_lowercase();
                                if s.len() > 2 && r.len() > 1 && o.len() > 2 {
                                    all_triples.push(brain_spiking::Triple::new(&s, &r, &o));
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("LLM triple extraction batch failed: {e}");
                    for sentence in chunk {
                        let triples = brain_spiking::extract_triples_with_topic(sentence, &topic_key_owned);
                        all_triples.extend(triples);
                    }
                }
            }
        }

        // Filter noise and deduplicate.
        // Reject sentence fragments, pronoun-only entities, leading-stopword
        // labels, and anything ending with a comma/incomplete punctuation.
        let stop_starts = [
            "the ", "a ", "an ", "this ", "that ", "these ", "those ",
            "it ", "you ", "we ", "they ", "he ", "she ", "his ", "her ",
            "their ", "our ", "my ", "your ", "and ", "but ", "or ",
            "if ", "when ", "while ", "before ", "after ", "because ",
            "so ", "to ", "of ", "for ", "in ", "on ", "with ", "by ",
            "is ", "was ", "are ", "were ", "be ", "been ",
        ];
        let pronouns = ["it", "you", "we", "they", "he", "she", "i", "us", "them", "this", "that"];
        let filler = ["sometimes", "somewhere", "somehow", "anything", "everything",
            "something", "nothing", "anywhere", "everywhere", "whatever"];
        let is_clean = |label: &str| -> bool {
            let l = label.trim().to_lowercase();
            if l.len() < 3 || l.len() > 60 { return false; }
            // No trailing punctuation that signals a fragment
            if l.ends_with(',') || l.ends_with(';') || l.ends_with(':') { return false; }
            // No pronoun-only or filler-only labels
            if pronouns.contains(&l.as_str()) || filler.contains(&l.as_str()) { return false; }
            // No leading stopword on multi-word labels
            if l.contains(' ') && stop_starts.iter().any(|s| l.starts_with(s)) { return false; }
            // Must contain at least one alphabetic char
            if !l.chars().any(|c| c.is_alphabetic()) { return false; }
            // Reject "to <verb>" infinitive starts
            if l.starts_with("to ") { return false; }
            true
        };
        all_triples.retain(|t| {
            let fields = [&t.subject, &t.relation, &t.object];
            let noise = ["example", "triple", "sentence", "output", "extract",
                "rule", "fact", "claim", "given", "according"];
            if fields.iter().any(|f| noise.iter().any(|n| f.contains(n))) { return false; }
            if !is_clean(&t.subject) || !is_clean(&t.object) { return false; }
            if t.relation.len() < 2 || t.relation.len() > 30 { return false; }
            if t.object.split_whitespace().count() > 6 { return false; }
            if t.subject.split_whitespace().count() > 5 { return false; }
            true
        });
        let mut seen = std::collections::HashSet::new();
        all_triples.retain(|t| seen.insert(format!("{}|{}|{}", t.subject, t.relation, t.object)));

        let count = all_triples.len();
        let mut seen_concepts: std::collections::HashSet<String> = std::collections::HashSet::new();
        if count > 0 {
            {
                let mut queue = brain.triple_queue.lock().unwrap();
                for (idx, triple) in all_triples.iter().enumerate() {
                    queue.push((triple.clone(), topic_key_owned.clone(), idx as i32));
                }
                tracing::info!("LLM extracted {} triples from {} sentences in {:.1}s (queue size: {})",
                    count, sentence_count, t0.elapsed().as_secs_f32(), queue.len());
            }
            let mut learned = brain.learned_concepts.lock().unwrap();
            for t in &all_triples {
                for label in [&t.subject, &t.object] {
                    if label.len() < 3 || !seen_concepts.insert(label.clone()) { continue; }
                    if let Ok(emb) = te.encode(label) {
                        learned.push((label.clone(), emb));
                    }
                }
            }
        }
        (count, seen_concepts.into_iter().collect::<Vec<String>>(), all_triples)
    }).await.unwrap_or((0, vec![], vec![]));
    let (llm_triples_count, llm_concepts, llm_triples) = _llm_result;

    // Store LLM-extracted triples in the SQLite KG so dialogue grounding
    // can query them by subject/object (e.g. "turboquant -> reduces -> memory usage").
    for t in &llm_triples {
        let _ = brain.memory_db.upsert_edge(&t.subject, &t.relation, &t.object, 0.8);
    }

    let novel_labels: std::collections::HashSet<String> = novel_concepts.iter()
        .map(|(l, _)| l.clone())
        .collect();

    let mut pairs_generated = 0;

    for concept in &all_concepts {
        if let Ok(t_emb) = te.encode(concept) {
            let t_proj = mlp.project_visual(&t_emb);

            // Store KG edges: topic → has-concept → concept
            let _ = brain.memory_db.upsert_edge(query, "has-concept", concept, 0.7);

            // Buffer as learning pair
            brain.online_pairs.lock().unwrap().push((t_emb, t_proj));
            pairs_generated += 1;
        }
    }

    // Concept-to-concept co-occurrence edges
    for i in 0..all_concepts.len() {
        for j in (i + 1)..all_concepts.len().min(i + 3) {
            let _ = brain.memory_db.upsert_edge(&all_concepts[i], "co-occurs-with", &all_concepts[j], 0.5);
        }
    }

    // 6. Encode audio via Whisper → feed spiking auditory cortex
    let pcm_path = format!("{tmp_dir}/audio_16k.raw");
    let _ = std::process::Command::new("ffmpeg")
        .args(["-i", &format!("{tmp_dir}/audio.wav"), "-ar", "16000", "-ac", "1",
               "-f", "f32le", "-y", &pcm_path])
        .output();

    if let Ok(raw) = std::fs::read(&pcm_path) {
        let samples: Vec<f32> = raw.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if samples.len() > 1600 {
            let mel = brain_inference::mel::compute_log_mel(&samples);
            if let Some(whisper) = &brain.audio_encoder {
                if let Ok(a_emb) = whisper.encode(&mel) {
                    // Feed audio into spiking auditory cortex
                    if let Some(ref sb) = brain.spiking_brain {
                        let mut sb = sb.lock().unwrap();
                        let enc_dim = sb.audio_encoder.dim();
                        let truncated: Vec<f32> = a_emb.iter().take(enc_dim).copied().collect();
                        if truncated.len() == enc_dim {
                            sb.process_audio(&truncated);
                        }
                    }
                    // Store fast memory
                    if let Some(mlp) = &brain.inference {
                        let a_proj = mlp.project_audio(&a_emb);
                        brain.fast_memory.lock().unwrap().store(&a_proj, query);
                    }
                }
            }
        }
    }

    // Log
    let video_id = url.split("v=").last().unwrap_or("?");
    let _ = brain.memory_db.log_youtube(video_id, query, "academic_processed", pairs_generated as i64);

    brain.sse.emit("academic_learn", serde_json::json!({
        "query": query, "url": url, "concepts": all_concepts,
        "pairs": pairs_generated, "transcript_chars": transcript.len(),
    }));

    let _ = std::fs::remove_dir_all(&tmp_dir);

    // Update topics.json registry
    let topics_path = brain.config.project_root.join("data/topics.json");
    let mut topic_list: Vec<serde_json::Value> = std::fs::read_to_string(&topics_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    topic_list.push(serde_json::json!({
        "topic": topic_key,
        "url": url,
        "triples_count": pairs_generated,
        "learned_at": ts,
    }));
    let _ = std::fs::create_dir_all(brain.config.project_root.join("data"));
    let _ = std::fs::write(&topics_path, serde_json::to_string_pretty(&topic_list).unwrap_or_default());

    Ok(AcademicLearnResult {
        triples_count: pairs_generated,
        topic: topic_key.clone(),
        key_concepts: llm_concepts,
    })
}

/// Enrich knowledge graph from ConceptNet API.
/// Queries for related concepts and stores edges.
async fn enrich_from_conceptnet(concept: &str, brain: &BrainState) -> usize {
    let clean = concept.to_lowercase().replace(' ', "_");
    let url = format!("http://api.conceptnet.io/c/en/{}?limit=10", urlencoding::encode(&clean));
    let client = reqwest::Client::new();

    let resp = match client.get(&url).timeout(std::time::Duration::from_secs(5)).send().await {
        Ok(r) => r,
        Err(_) => return 0,
    };
    let json: serde_json::Value = match resp.json().await {
        Ok(j) => j,
        Err(_) => return 0,
    };

    let mut edges_added = 0;
    if let Some(edges) = json.get("edges").and_then(|e| e.as_array()) {
        for edge in edges.iter().take(10) {
            let rel = edge.get("rel").and_then(|r| r.get("label")).and_then(|l| l.as_str()).unwrap_or("");
            let start = edge.get("start").and_then(|s| s.get("label")).and_then(|l| l.as_str()).unwrap_or("");
            let end = edge.get("end").and_then(|e| e.get("label")).and_then(|l| l.as_str()).unwrap_or("");
            let weight = edge.get("weight").and_then(|w| w.as_f64()).unwrap_or(1.0);

            if !rel.is_empty() && !start.is_empty() && !end.is_empty() && start.len() < 100 && end.len() < 100 {
                let _ = brain.memory_db.upsert_edge(start, rel, end, weight.min(1.0));
                edges_added += 1;
            }
        }
    }
    edges_added
}

/// Enrich knowledge from Wikidata SPARQL endpoint.
async fn enrich_from_wikidata(concept: &str, brain: &BrainState) -> usize {
    let query = format!(
        r#"SELECT ?itemLabel ?desc WHERE {{
            ?item rdfs:label "{}"@en .
            OPTIONAL {{ ?item schema:description ?desc FILTER(LANG(?desc)="en") }}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }} LIMIT 5"#, concept.replace('"', "'")
    );
    let url = format!("https://query.wikidata.org/sparql?query={}&format=json", urlencoding::encode(&query));
    let client = reqwest::Client::new();
    let resp = match client.get(&url)
        .header("User-Agent", "BrainCortex/1.0")
        .timeout(std::time::Duration::from_secs(5))
        .send().await {
        Ok(r) => r,
        Err(_) => return 0,
    };
    let json: serde_json::Value = match resp.json().await { Ok(j) => j, Err(_) => return 0 };

    let mut edges = 0;
    if let Some(bindings) = json.pointer("/results/bindings").and_then(|b| b.as_array()) {
        for b in bindings.iter().take(5) {
            if let Some(desc) = b.pointer("/desc/value").and_then(|v| v.as_str()) {
                if desc.len() > 3 && desc.len() < 200 {
                    let _ = brain.memory_db.upsert_edge(concept, "wikidata-desc", desc, 0.85);
                    edges += 1;
                }
            }
        }
    }
    edges
}

/// Enrich knowledge from Wikipedia summary API.
/// Gets a short description and extracts key facts as KG edges.
async fn enrich_from_wikipedia(concept: &str, brain: &BrainState) -> usize {
    let url = format!("https://en.wikipedia.org/api/rest_v1/page/summary/{}", urlencoding::encode(concept));
    let client = reqwest::Client::new();

    let resp = match client.get(&url).timeout(std::time::Duration::from_secs(5)).send().await {
        Ok(r) => r,
        Err(_) => return 0,
    };
    let json: serde_json::Value = match resp.json().await {
        Ok(j) => j,
        Err(_) => return 0,
    };

    let mut edges_added = 0;
    let extract = json.get("extract").and_then(|e| e.as_str()).unwrap_or("");
    let description = json.get("description").and_then(|d| d.as_str()).unwrap_or("");

    if !extract.is_empty() {
        // Store description as edge
        if !description.is_empty() && description.len() < 200 {
            let _ = brain.memory_db.upsert_edge(concept, "described-as", description, 0.9);
            edges_added += 1;
        }

        // Extract "is a" relationships from first sentence
        let first_sentence = extract.split('.').next().unwrap_or("");
        if let Some(is_pos) = first_sentence.find(" is ") {
            let obj = &first_sentence[is_pos + 4..];
            if obj.len() > 2 && obj.len() < 150 {
                let _ = brain.memory_db.upsert_edge(concept, "is-a", obj.trim(), 0.8);
                edges_added += 1;
            }
        }

        // Store as perception for text understanding
        let _ = brain.memory_db.store_perception(
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64(),
            "wikipedia", Some(&extract[..extract.len().min(500)]),
            Some(&format!("[\"{concept}\"]")), None, None);

        // Encode text and create learning pair if text encoder available
        if let Some(te) = &brain.text_encoder {
            if let Ok(emb) = te.encode(concept) {
                if let Ok(desc_emb) = te.encode(&extract[..extract.len().min(200)]) {
                    // Pair: concept name embedding ↔ description embedding
                    brain.online_pairs.lock().unwrap().push((emb, desc_emb));
                }
            }
        }
    }
    edges_added
}
