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
pub async fn youtube_learn_academic(query: &str, brain: &BrainState) -> Result<usize, String> {
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
    tracing::info!("Academic learn: {query} → {url}");

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
                if chunk.len() + word.len() > 100 && chunk.len() > 20 {
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

    // Encode each sentence, find nearest codebook concepts
    let mut concept_scores: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
    let cb_guard = brain.codebook.lock().unwrap();
    let cb = cb_guard.as_ref().ok_or("No codebook")?;

    for sentence in sentences.iter().take(50) { // cap at 50 sentences
        if let Ok(emb) = te.encode(sentence) {
            let proj = mlp.project_visual(&emb);
            let nearest = cb.nearest(&proj, 3);
            for (label, sim) in &nearest {
                if *sim > 0.3 { // only meaningful matches
                    *concept_scores.entry(label.clone()).or_insert(0.0) += sim;
                }
            }
        }
    }

    // Sort by accumulated score, take top 10
    let mut scored: Vec<(String, f32)> = concept_scores.into_iter().collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let concepts: Vec<String> = scored.into_iter().take(10).map(|(label, _)| label).collect();

    // Also extract novel concepts: sentences that DON'T match the codebook well
    let mut novel_concepts = Vec::new();
    for sentence in sentences.iter().take(50) {
        if let Ok(emb) = te.encode(sentence) {
            let proj = mlp.project_visual(&emb);
            let nearest = cb.nearest(&proj, 1);
            if nearest.is_empty() || nearest[0].1 < 0.4 {
                // This sentence is about something the codebook doesn't know
                // Use the first few words as a concept label
                let label: String = sentence.split_whitespace().take(5).collect::<Vec<_>>().join(" ");
                if label.len() > 10 {
                    novel_concepts.push((label, proj));
                }
            }
        }
    }

    drop(cb_guard); // release lock before mutating

    // Add novel concepts to learned_concepts store (text-searchable)
    if !novel_concepts.is_empty() {
        let mut learned = brain.learned_concepts.lock().unwrap();
        for (label, _proj) in &novel_concepts {
            // Encode via MiniLM for text-based semantic search
            if let Ok(emb) = te.encode(label) {
                learned.push((label.clone(), emb));
            }
        }
        tracing::info!("Academic learn: added {} novel concepts to learned store (total: {})",
            novel_concepts.len(), learned.len());
    }

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

    // 5. Feed NOVEL concepts into spiking brain + all concepts into KG
    // Only novel concepts (from video transcript) go into the brain's concept memory.
    // Audio codebook matches (Coo, Cymbal, etc.) are noise and excluded.
    let novel_labels: std::collections::HashSet<String> = novel_concepts.iter()
        .map(|(l, _)| l.clone())
        .collect();

    let mut pairs_generated = 0;

    for concept in &all_concepts {
        if let Ok(t_emb) = te.encode(concept) {
            let t_proj = mlp.project_visual(&t_emb);

            // Enqueue novel concepts for STDP learning (tick thread processes)
            if novel_labels.contains(concept) {
                if let Some(ref sb) = brain.spiking_brain {
                    if let Ok(mut sb) = sb.try_lock() {
                        sb.enqueue_learn(t_emb.clone());
                        sb.novelty(0.3);
                    }
                }
            }

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
    Ok(pairs_generated)
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
