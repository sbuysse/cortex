use std::collections::HashMap;

/// A cell assembly: a dedicated population of neurons representing one concept.
#[derive(Debug, Clone)]
pub struct CellAssembly {
    /// Start neuron index within the concept region.
    pub start: usize,
    /// Number of neurons in this assembly.
    pub size: usize,
}

impl CellAssembly {
    pub fn neuron_range(&self) -> std::ops::Range<usize> {
        self.start..self.start + self.size
    }
}

/// Registry mapping concept strings to neuron populations.
/// Each concept gets a dedicated cell assembly of ~100 neurons.
/// The registry also tracks relation types (which are concepts too).
pub struct ConceptRegistry {
    /// Concept name → cell assembly location.
    concepts: HashMap<String, CellAssembly>,
    /// Next available neuron index for allocation.
    next_neuron: usize,
    /// Maximum neurons available in the concept region.
    max_neurons: usize,
    /// Neurons per concept assembly.
    assembly_size: usize,
}

impl ConceptRegistry {
    pub fn new(max_neurons: usize, assembly_size: usize) -> Self {
        Self {
            concepts: HashMap::new(),
            next_neuron: 0,
            max_neurons,
            assembly_size,
        }
    }

    /// Get or create a cell assembly for a concept.
    /// Returns the assembly and whether it was newly created.
    pub fn get_or_create(&mut self, concept: &str) -> Option<(CellAssembly, bool)> {
        if let Some(assembly) = self.concepts.get(concept) {
            return Some((assembly.clone(), false));
        }

        // Check capacity
        if self.next_neuron + self.assembly_size > self.max_neurons {
            tracing::warn!("Concept registry full ({} concepts, {} neurons used)",
                self.concepts.len(), self.next_neuron);
            return None;
        }

        let assembly = CellAssembly {
            start: self.next_neuron,
            size: self.assembly_size,
        };
        self.next_neuron += self.assembly_size;
        self.concepts.insert(concept.to_string(), assembly.clone());

        Some((assembly, true))
    }

    /// Look up an existing concept.
    pub fn get(&self, concept: &str) -> Option<&CellAssembly> {
        self.concepts.get(concept)
    }

    /// Number of registered concepts.
    pub fn len(&self) -> usize {
        self.concepts.len()
    }

    /// Total neurons allocated.
    pub fn neurons_used(&self) -> usize {
        self.next_neuron
    }

    /// Maximum concepts this registry can hold.
    pub fn capacity(&self) -> usize {
        self.max_neurons / self.assembly_size
    }

    /// Get all concept names.
    pub fn concept_names(&self) -> Vec<&str> {
        self.concepts.keys().map(|s| s.as_str()).collect()
    }

    /// Find which concept a neuron belongs to (if any).
    pub fn neuron_to_concept(&self, neuron_idx: usize) -> Option<&str> {
        for (name, assembly) in &self.concepts {
            if neuron_idx >= assembly.start && neuron_idx < assembly.start + assembly.size {
                return Some(name.as_str());
            }
        }
        None
    }
}

/// A knowledge triple: (subject, relation, object).
#[derive(Debug, Clone)]
pub struct Triple {
    pub subject: String,
    pub relation: String,
    pub object: String,
}

impl Triple {
    pub fn new(subject: &str, relation: &str, object: &str) -> Self {
        Self {
            subject: subject.to_string(),
            relation: relation.to_string(),
            object: object.to_string(),
        }
    }
}

/// Extract (subject, relation, object) triples from a sentence.
/// `topic`: the video/query topic used to resolve pronouns ("this" → topic).
pub fn extract_triples_with_topic(sentence: &str, topic: &str) -> Vec<Triple> {
    // Resolve pronouns: replace "this/it/that" at sentence start with the topic
    let pronouns = ["this", "it", "that", "they", "these", "those"];
    let resolved = {
        let first_word = sentence.split_whitespace().next().unwrap_or("").to_lowercase();
        if !topic.is_empty() && pronouns.contains(&first_word.as_str()) {
            format!("{} {}", topic, &sentence[first_word.len()..])
        } else {
            sentence.to_string()
        }
    };

    let words: Vec<&str> = resolved.split_whitespace().collect();
    if words.len() < 3 { return vec![]; }

    let mut triples = Vec::new();
    let sentence = &resolved;

    // Common relation verbs
    let relation_verbs = [
        "is", "are", "was", "were", "uses", "use", "using",
        "compresses", "compress", "compressing",
        "reduces", "reduce", "reducing",
        "enables", "enable", "enabling",
        "provides", "provide", "providing",
        "creates", "create", "creating",
        "converts", "convert", "converting",
        "stores", "store", "storing",
        "processes", "process", "processing",
        "improves", "improve", "improving",
        "requires", "require", "requiring",
        "replaces", "replace", "replacing",
        "achieves", "achieve", "achieving",
        "represents", "represent", "representing",
        "contains", "contain", "containing",
        "produces", "produce", "producing",
        "maintains", "maintain", "maintaining",
        "generates", "generate", "generating",
        "supports", "support", "supporting",
        "implements", "implement", "implementing",
        "optimizes", "optimize", "optimizing",
        "transforms", "transform", "transforming",
        "called", "known", "means", "works",
    ];

    // Find verb positions
    for (i, word) in words.iter().enumerate() {
        let lower = word.to_lowercase();
        let clean = lower.trim_matches(|c: char| !c.is_alphanumeric());

        if relation_verbs.contains(&clean.as_ref()) && i > 0 && i < words.len() - 1 {
            let stop_words = ["the", "a", "an", "of", "in", "on", "at", "to",
                "for", "by", "with", "from", "this", "that", "it", "its",
                "and", "or", "but", "so", "if", "as", "is", "was", "are",
                "very", "really", "just", "well", "also", "even", "about"];

            // Subject: words before the verb (last 1-3 meaningful words)
            let subject_words: Vec<&str> = words[..i].iter()
                .rev()
                .filter(|w| w.len() > 2 && !stop_words.contains(&w.to_lowercase().as_str()))
                .take(3)
                .copied()
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();

            // Object: words after the verb (first 1-4 meaningful words)
            let object_words: Vec<&str> = words[i+1..]
                .iter()
                .filter(|w| w.len() > 2 && !stop_words.contains(&w.to_lowercase().as_str()))
                .take(4)
                .copied()
                .collect();

            if !subject_words.is_empty() && !object_words.is_empty() {
                let subject = subject_words.join(" ").to_lowercase();
                let object = object_words.join(" ").to_lowercase();
                let relation = clean.to_string();

                // Skip trivial subjects/objects
                if subject.len() > 2 && object.len() > 2 {
                    triples.push(Triple::new(&subject, &relation, &object));
                }
            }
        }
    }

    triples
}

/// Extract triples without topic (backward compat).
pub fn extract_triples(sentence: &str) -> Vec<Triple> {
    extract_triples_with_topic(sentence, "")
}
