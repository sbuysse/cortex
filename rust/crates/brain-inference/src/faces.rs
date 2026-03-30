//! Face detection and recognition.
//!
//! Uses DINOv2 embeddings for face recognition (same encoder, different use).
//! Stores face embeddings in personal_facts for known people.
//! Designed for low-power devices (no separate face model needed).

/// A recognized face with identity and confidence.
#[derive(Debug, Clone)]
pub struct FaceMatch {
    pub name: String,
    pub confidence: f32,
    pub embedding: Vec<f32>,
}

/// Simple face database — stores name→embedding pairs.
pub struct FaceDatabase {
    entries: Vec<(String, Vec<f32>)>,
}

impl FaceDatabase {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Register a face.
    pub fn register(&mut self, name: &str, embedding: &[f32]) {
        // Update if exists, otherwise add
        if let Some(entry) = self.entries.iter_mut().find(|(n, _)| n == name) {
            // Running average of face embeddings
            for (e, &new) in entry.1.iter_mut().zip(embedding) {
                *e = *e * 0.8 + new * 0.2; // exponential moving average
            }
        } else {
            self.entries.push((name.to_string(), embedding.to_vec()));
        }
    }

    /// Identify a face from its embedding.
    /// Returns best match if above threshold.
    pub fn identify(&self, embedding: &[f32], threshold: f32) -> Option<FaceMatch> {
        if self.entries.is_empty() { return None; }

        let mut best: Option<(usize, f32)> = None;
        for (i, (_, stored)) in self.entries.iter().enumerate() {
            let sim = cosine_sim(embedding, stored);
            if sim > threshold {
                if best.is_none() || sim > best.unwrap().1 {
                    best = Some((i, sim));
                }
            }
        }

        best.map(|(i, sim)| FaceMatch {
            name: self.entries[i].0.clone(),
            confidence: sim,
            embedding: self.entries[i].1.clone(),
        })
    }

    pub fn count(&self) -> usize { self.entries.len() }

    pub fn names(&self) -> Vec<&str> {
        self.entries.iter().map(|(n, _)| n.as_str()).collect()
    }
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() { return 0.0; }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b + 1e-12)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_face_database() {
        let mut db = FaceDatabase::new();
        let emb_a = vec![1.0, 0.0, 0.0, 0.0];
        let emb_b = vec![0.0, 1.0, 0.0, 0.0];

        db.register("Alice", &emb_a);
        db.register("Bob", &emb_b);
        assert_eq!(db.count(), 2);

        // Identify Alice
        let m = db.identify(&emb_a, 0.5).unwrap();
        assert_eq!(m.name, "Alice");
        assert!(m.confidence > 0.9);

        // Unknown face
        let unknown = vec![0.5, 0.5, 0.5, 0.5];
        let m = db.identify(&unknown, 0.9);
        assert!(m.is_none()); // below threshold
    }
}
