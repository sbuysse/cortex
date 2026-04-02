//! Text encoder — MiniLM sentence transformer via TorchScript.
//!
//! Encodes text to 384-dim embeddings. For tokenization in Rust,
//! we use the pre-encoded label embeddings + a simple tokenizer.
//! For runtime text encoding, we call the TorchScript model.

use std::path::Path;
use tch::{CModule, Device, Tensor};
use serde_json;

/// MiniLM text encoder loaded from TorchScript.
pub struct TextEncoder {
    model: CModule,
    tokenizer: Tokenizer,
    pub label_embeddings: Option<Vec<Vec<f32>>>,  // (310, 384)
    pub labels: Option<Vec<String>>,
    cache: std::sync::Mutex<std::collections::HashMap<String, Vec<f32>>>,
}

/// Simple wordpiece tokenizer loaded from HuggingFace tokenizer.json.
pub struct Tokenizer {
    vocab: std::collections::HashMap<String, i64>,
    unk_id: i64,
    cls_id: i64,
    sep_id: i64,
    pad_id: i64,
}

impl Tokenizer {
    /// Load from HuggingFace tokenizer.json format.
    pub fn load(dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let tokenizer_path = dir.join("tokenizer.json");
        let data: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&tokenizer_path)?)?;

        // Extract vocab from the model section
        let mut vocab = std::collections::HashMap::new();
        if let Some(model) = data.get("model") {
            if let Some(v) = model.get("vocab") {
                if let Some(obj) = v.as_object() {
                    for (key, val) in obj {
                        if let Some(id) = val.as_i64() {
                            vocab.insert(key.clone(), id);
                        }
                    }
                }
            }
        }

        let cls_id = *vocab.get("[CLS]").unwrap_or(&101);
        let sep_id = *vocab.get("[SEP]").unwrap_or(&102);
        let pad_id = *vocab.get("[PAD]").unwrap_or(&0);
        let unk_id = *vocab.get("[UNK]").unwrap_or(&100);

        tracing::info!("Tokenizer loaded: {} vocab entries", vocab.len());
        Ok(Self { vocab, unk_id, cls_id, sep_id, pad_id })
    }

    /// Simple tokenization: lowercase + split on whitespace + lookup.
    /// Not full WordPiece, but good enough for category labels.
    pub fn encode(&self, text: &str, max_len: usize) -> (Vec<i64>, Vec<i64>) {
        let mut ids = vec![self.cls_id];
        let lower = text.to_lowercase();

        for word in lower.split_whitespace() {
            let clean: String = word.chars().filter(|c| c.is_alphanumeric()).collect();
            if clean.is_empty() { continue; }
            let id = self.vocab.get(&clean).copied().unwrap_or(self.unk_id);
            ids.push(id);
            if ids.len() >= max_len - 1 { break; }
        }
        ids.push(self.sep_id);

        let len = ids.len();
        let mask = vec![1i64; len];
        // Pad to consistent length
        while ids.len() < max_len {
            ids.push(self.pad_id);
        }
        let mut full_mask = mask;
        while full_mask.len() < max_len {
            full_mask.push(0);
        }

        (ids, full_mask)
    }
}

impl TextEncoder {
    /// Load the TorchScript model + tokenizer + pre-encoded labels.
    pub fn load(dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let model_path = dir.join("minilm_ts.pt");
        let model = CModule::load(&model_path)?;
        tracing::info!("Text encoder (MiniLM) loaded from {:?}", model_path);

        let tokenizer = Tokenizer::load(&dir.join("tokenizer"))?;

        // Load pre-encoded label embeddings if available
        let label_emb_path = dir.join("label_embeddings.npy");
        let labels_path = dir.join("labels.json");
        let mut label_embeddings = None;
        let mut labels = None;

        if label_emb_path.exists() && labels_path.exists() {
            // Simple npy loader (assumes float32, 2D)
            let raw = std::fs::read(&label_emb_path)?;
            // Skip npy header (usually 128 bytes for simple arrays)
            let header_end = raw.iter().position(|&b| b == b'\n')
                .map(|p| p + 1).unwrap_or(128);
            let data_start = if raw.len() > 128 { 128 } else { header_end };
            // Actually parse the npy header to get shape
            // For simplicity, use known shape (310, 384)
            let labels_json: Vec<String> = serde_json::from_str(&std::fs::read_to_string(&labels_path)?)?;
            let n_labels = labels_json.len();
            let dim = 384;
            let float_data: Vec<f32> = raw[data_start..]
                .chunks_exact(4)
                .take(n_labels * dim)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            let embs: Vec<Vec<f32>> = float_data.chunks(dim).map(|c| c.to_vec()).collect();
            tracing::info!("Loaded {} pre-encoded label embeddings", embs.len());
            label_embeddings = Some(embs);
            labels = Some(labels_json);
        }

        Ok(Self { model, tokenizer, label_embeddings, labels,
            cache: std::sync::Mutex::new(std::collections::HashMap::new()) })
    }

    /// Encode text to 384-dim embedding via TorchScript model (cached).
    pub fn encode(&self, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Check cache
        {
            let cache = self.cache.lock().unwrap();
            if let Some(emb) = cache.get(text) {
                return Ok(emb.clone());
            }
        }
        // Compute
        let (ids, mask) = self.tokenizer.encode(text, 128);
        let ids_tensor = Tensor::from_slice(&ids).reshape(&[1, ids.len() as i64]).to_device(Device::Cpu);
        let mask_tensor = Tensor::from_slice(&mask).reshape(&[1, mask.len() as i64]).to_device(Device::Cpu);
        let output = self.model.forward_ts(&[ids_tensor, mask_tensor])?;
        let result: Vec<f32> = Vec::try_from(output.flatten(0, -1))?;
        // Cache (evict if too large)
        {
            let mut cache = self.cache.lock().unwrap();
            if cache.len() > 2000 { cache.clear(); }
            cache.insert(text.to_string(), result.clone());
        }
        Ok(result)
    }

    /// Semantic search: find the closest labels to a query text.
    pub fn semantic_search(&self, query: &str, top_k: usize) -> Result<Vec<(String, f32)>, Box<dyn std::error::Error>> {
        let q_emb = self.encode(query)?;
        let (embs, labels) = match (&self.label_embeddings, &self.labels) {
            (Some(e), Some(l)) => (e, l),
            _ => return Ok(Vec::new()),
        };

        let mut scored: Vec<(usize, f32)> = embs.iter().enumerate().map(|(i, emb)| {
            let sim: f32 = q_emb.iter().zip(emb).map(|(a, b)| a * b).sum();
            (i, sim)
        }).collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(scored.into_iter().take(top_k).map(|(i, sim)| {
            (labels[i].clone(), (sim * 10000.0).round() / 10000.0)
        }).collect())
    }

    /// Semantic search with a pre-computed embedding (384-dim).
    pub fn semantic_search_embedding(&self, query_emb: &[f32], top_k: usize) -> Result<Vec<(String, f32)>, Box<dyn std::error::Error>> {
        let (embs, labels) = match (&self.label_embeddings, &self.labels) {
            (Some(e), Some(l)) => (e, l),
            _ => return Ok(Vec::new()),
        };

        let mut scored: Vec<(usize, f32)> = embs.iter().enumerate().map(|(i, emb)| {
            let sim: f32 = query_emb.iter().zip(emb).map(|(a, b)| a * b).sum();
            (i, sim)
        }).collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(scored.into_iter().take(top_k).map(|(i, sim)| {
            (labels[i].clone(), (sim * 10000.0).round() / 10000.0)
        }).collect())
    }

    pub fn has_labels(&self) -> bool { self.labels.is_some() }
    pub fn label_count(&self) -> usize { self.labels.as_ref().map(|l| l.len()).unwrap_or(0) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_text_encoder_load_and_encode() {
        let dir = PathBuf::from("/opt/brain/outputs/cortex/text_encoder");
        if !dir.join("minilm_ts.pt").exists() { return; }

        let encoder = TextEncoder::load(&dir).unwrap();
        let emb = encoder.encode("thunder storm").unwrap();
        assert_eq!(emb.len(), 384);

        // Check L2 normalized
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "Should be ~L2 normalized, got {norm}");
    }

    #[test]
    fn test_semantic_search() {
        let dir = PathBuf::from("/opt/brain/outputs/cortex/text_encoder");
        if !dir.join("minilm_ts.pt").exists() { return; }

        let encoder = TextEncoder::load(&dir).unwrap();
        let results = encoder.semantic_search("dog barking", 5).unwrap();
        assert!(!results.is_empty());
        // "dog barking" should be in the top results
        assert!(results.iter().any(|(label, _)| label.contains("dog")),
                "Should find dog-related labels, got {:?}", results);
    }

    #[test]
    fn test_tokenizer() {
        let dir = PathBuf::from("/opt/brain/outputs/cortex/text_encoder/tokenizer");
        if !dir.exists() { return; }

        let tok = Tokenizer::load(&dir).unwrap();
        let (ids, mask) = tok.encode("hello world", 16);
        assert_eq!(ids.len(), 16); // padded to max_len
        assert!(ids[0] == tok.cls_id); // starts with [CLS]
        assert!(mask.iter().sum::<i64>() >= 3); // at least CLS + 2 words + SEP
    }
}
