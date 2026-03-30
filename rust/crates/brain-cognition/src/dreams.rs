//! Dream Engine — offline imagination with learning via world model chaining.
//!
//! Seeds a concept, chains predictions through the world model,
//! scores surprise at each step, generates learning pairs from gaps.

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct DreamStep {
    pub concept: String,
    pub step: usize,
    pub predicted_sim: f32,
    pub surprise: f32,
    pub kg_support: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Dream {
    pub id: i64,
    pub timestamp: f64,
    pub seed: String,
    pub steps: Vec<DreamStep>,
    pub avg_surprise: f32,
    pub max_surprise: f32,
    pub learning_pairs_generated: usize,
    pub multi_scale_pe: std::collections::HashMap<String, f32>,
    /// Learning pairs generated from high-surprise transitions (current → predicted).
    #[serde(skip)]
    pub learning_pairs: Vec<(Vec<f32>, Vec<f32>)>,
}

/// Generate a dream by chaining world model predictions through the concept codebook.
pub fn generate_dream(
    seed_concept: Option<&str>,
    max_steps: usize,
    codebook: &super::concepts::ConceptCodebook,
    world_model: &brain_inference::WorldModel,
    dream_count: &std::sync::atomic::AtomicI64,
) -> Result<Dream, String> {
    use std::sync::atomic::Ordering::Relaxed;

    if codebook.is_empty() {
        return Err("No concept codebook".into());
    }

    // 1. Seed selection
    let (seed_name, seed_emb) = if let Some(seed) = seed_concept {
        // Find matching concept
        let results = codebook.nearest(&vec![0.0f32; 512], codebook.len());
        let match_idx = results.iter().position(|(l, _)| l.to_lowercase().contains(&seed.to_lowercase()));
        if let Some(idx) = match_idx {
            let emb: Vec<f32> = codebook.centroids.row(idx).to_vec();
            (results[idx].0.clone(), emb)
        } else {
            // Use first concept as fallback
            (codebook.labels[0].clone(), codebook.centroids.row(0).to_vec())
        }
    } else {
        // Random concept
        let idx = rand::Rng::random_range(&mut rand::rng(), 0..codebook.len());
        (codebook.labels[idx].clone(), codebook.centroids.row(idx).to_vec())
    };

    // 2. Chain predictions
    let mut steps = vec![DreamStep {
        concept: seed_name.clone(),
        step: 0,
        predicted_sim: 1.0,
        surprise: 0.0,
        kg_support: false,
    }];

    let mut current_emb = seed_emb;
    let mut learning_pairs_count = 0;
    let mut learning_pairs: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
    let mut visited = std::collections::HashSet::new();
    visited.insert(seed_name.clone());
    let mut rng = rand::rng();

    for step_num in 1..=max_steps {
        // Predict next via world model
        let pred = match world_model.predict(&current_emb) {
            Ok(p) => p,
            Err(_) => break,
        };

        // Add exploration noise (increases with step to escape attractors)
        let noise_scale = 0.1 + step_num as f32 * 0.05;
        let mut noisy_pred = pred.clone();
        for v in noisy_pred.iter_mut() {
            *v += (rand::Rng::random::<f32>(&mut rng) - 0.5) * noise_scale;
        }
        // Re-normalize
        let norm: f32 = noisy_pred.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for v in noisy_pred.iter_mut() { *v /= norm; }

        // Find nearest concepts, skip already visited
        let candidates = codebook.nearest(&noisy_pred, 10);
        let chosen = candidates.iter()
            .find(|(name, _)| !visited.contains(name))
            .or_else(|| candidates.first());

        let (nearest_name, nearest_sim) = match chosen {
            Some(c) => c,
            None => break,
        };

        let surprise = (1.0 - nearest_sim).max(0.0);
        visited.insert(nearest_name.clone());

        steps.push(DreamStep {
            concept: nearest_name.clone(),
            step: step_num,
            predicted_sim: *nearest_sim,
            surprise,
            kg_support: false,
        });

        // High-surprise transitions generate learning pairs
        if surprise > 0.2 {
            learning_pairs_count += 1;
            learning_pairs.push((current_emb.clone(), pred.clone()));
        }

        // Move to chosen concept
        if let Some(idx) = codebook.labels.iter().position(|l| l == nearest_name) {
            current_emb = codebook.centroids.row(idx).to_vec();
        } else {
            current_emb = noisy_pred;
        }
    }

    let surprises: Vec<f32> = steps[1..].iter().map(|s| s.surprise).collect();
    let avg_surprise = if surprises.is_empty() { 0.0 } else {
        surprises.iter().sum::<f32>() / surprises.len() as f32
    };
    let max_surprise = surprises.iter().copied().fold(0.0f32, f32::max);

    // Multi-timescale PE
    let mut multi_scale_pe = std::collections::HashMap::new();
    if surprises.len() >= 2 {
        multi_scale_pe.insert("short".into(), surprises[..2.min(surprises.len())].iter().sum::<f32>() / 2.0);
    }
    if surprises.len() >= 4 {
        multi_scale_pe.insert("medium".into(), surprises[1..4.min(surprises.len())].iter().sum::<f32>() / 3.0);
    }
    if !surprises.is_empty() {
        multi_scale_pe.insert("long".into(), avg_surprise);
    }

    let id = dream_count.fetch_add(1, Relaxed) + 1;

    Ok(Dream {
        id,
        timestamp: now_secs(),
        seed: seed_name,
        steps,
        avg_surprise: (avg_surprise * 10000.0).round() / 10000.0,
        max_surprise: (max_surprise * 10000.0).round() / 10000.0,
        learning_pairs_generated: learning_pairs_count,
        multi_scale_pe,
        learning_pairs,
    })
}

fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}
