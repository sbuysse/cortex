use std::collections::{HashMap, HashSet};

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
    /// Concept name → set of topics that mention this concept.
    topic_provenance: HashMap<String, HashSet<String>>,
}

impl ConceptRegistry {
    pub fn new(max_neurons: usize, assembly_size: usize) -> Self {
        Self {
            concepts: HashMap::new(),
            next_neuron: 0,
            max_neurons,
            assembly_size,
            topic_provenance: HashMap::new(),
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

    /// Record that `concept` appeared in `topic`.
    pub fn add_topic(&mut self, concept: &str, topic: &str) {
        self.topic_provenance
            .entry(concept.to_string())
            .or_default()
            .insert(topic.to_string());
    }

    /// Topics that mention a given concept.
    pub fn get_topics(&self, concept: &str) -> Vec<String> {
        self.topic_provenance
            .get(concept)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// All distinct topics across all concepts.
    pub fn all_topics(&self) -> Vec<String> {
        let mut topics: HashSet<String> = HashSet::new();
        for set in self.topic_provenance.values() {
            topics.extend(set.iter().cloned());
        }
        let mut v: Vec<String> = topics.into_iter().collect();
        v.sort();
        v
    }

    /// Concepts that appear in two or more distinct topics ("bridge" concepts).
    pub fn bridge_concepts(&self) -> Vec<String> {
        let mut bridges: Vec<String> = self.topic_provenance
            .iter()
            .filter(|(_, topics)| topics.len() >= 2)
            .map(|(concept, _)| concept.clone())
            .collect();
        bridges.sort();
        bridges
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

    // Common relation verbs
    let relation_verbs: &[&str] = &[
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

    let stop_words: &[&str] = &["the", "a", "an", "of", "in", "on", "at", "to",
        "for", "by", "with", "from", "this", "that", "it", "its",
        "and", "or", "but", "so", "if", "as", "is", "was", "are",
        "very", "really", "just", "well", "also", "even", "about",
        "not", "can", "will", "would", "could", "should", "been",
        "have", "has", "had", "there", "here", "what", "when",
        "how", "why", "who", "which", "where", "then", "than",
        "more", "most", "some", "any", "all", "each", "every",
        "much", "many", "few", "only", "own", "same", "other"];

    // Words that indicate noise subjects (filler, channel intros, meta-commentary)
    let noise_starts: &[&str] = &["well", "true", "false", "yes", "no", "dear", "hello",
        "okay", "right", "sure", "look", "see", "now", "hey", "wow",
        "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "unless", "because", "although", "however", "actually", "basically",
        "obviously", "clearly", "literally", "number", "part", "moved",
        "caught", "news", "scholars", "fellow", "friends", "guys", "folks",
        "you", "they", "eye", "numbers", "relate", "mean", "means",
        "let", "think", "say", "know", "want", "need", "try", "make"];

    let topic_lower = topic.to_lowercase();

    // Split on sentence boundaries to handle multi-sentence chunks
    let sentences: Vec<&str> = resolved.split(|c: char| c == '.' || c == '!' || c == '?')
        .map(|s| s.trim())
        .filter(|s| s.split_whitespace().count() >= 4)
        .collect();

    let mut all_triples = Vec::new();
    for sent in &sentences {
        let sent_triples = extract_triples_from_sentence(sent, &topic_lower, stop_words, relation_verbs, noise_starts);
        all_triples.extend(sent_triples);
    }
    // Also try the whole resolved text as one pass
    if sentences.len() <= 1 {
        let words: Vec<&str> = resolved.split_whitespace().collect();
        if words.len() >= 4 {
            let sent_triples = extract_triples_from_sentence(&resolved, &topic_lower, stop_words, relation_verbs, noise_starts);
            all_triples.extend(sent_triples);
        }
    }

    // Topic-anchored extraction: if the text mentions the topic, extract key phrases
    // as (topic, "relates-to", phrase) triples. This catches information that SVO misses.
    if !topic_lower.is_empty() {
        let text_lower = resolved.to_lowercase();
        if text_lower.contains(&topic_lower) || {
            // Also match if first word was a pronoun that got resolved
            let first_word = sentence.split_whitespace().next().unwrap_or("").to_lowercase();
            pronouns.contains(&first_word.as_str())
        } {
            // Extract multi-word technical terms (2-4 word phrases not in stop words)
            let words: Vec<&str> = resolved.split_whitespace().collect();
            let mut i = 0;
            while i < words.len() {
                let mut phrase_words = Vec::new();
                let mut j = i;
                while j < words.len() && phrase_words.len() < 4 {
                    let w = words[j].to_lowercase();
                    let clean = w.trim_matches(|c: char| !c.is_alphanumeric());
                    if clean.len() > 2 && !stop_words.contains(&clean)
                        && !clean.eq_ignore_ascii_case(&topic_lower) {
                        phrase_words.push(clean.to_string());
                    } else if !phrase_words.is_empty() {
                        break;
                    }
                    j += 1;
                }
                if phrase_words.len() >= 2 {
                    let phrase = phrase_words.join(" ");
                    // Only keep phrases with at least one word > 4 chars (substantive)
                    if phrase_words.iter().any(|w| w.len() > 4) {
                        all_triples.push(Triple::new(&topic_lower, "relates-to", &phrase));
                    }
                }
                i = if j > i { j } else { i + 1 };
            }
        }
    }

    // Deduplicate
    let mut seen = std::collections::HashSet::new();
    all_triples.retain(|t| seen.insert(format!("{}|{}|{}", t.subject, t.relation, t.object)));
    all_triples
}

fn extract_triples_from_sentence(
    sentence: &str,
    topic_lower: &str,
    stop_words: &[&str],
    relation_verbs: &[&str],
    noise_starts: &[&str],
) -> Vec<Triple> {
    let words: Vec<&str> = sentence.split_whitespace().collect();
    let mut triples = Vec::new();

    for (i, word) in words.iter().enumerate() {
        let lower = word.to_lowercase();
        let clean = lower.trim_matches(|c: char| !c.is_alphanumeric());

        if relation_verbs.contains(&clean) && i > 0 && i < words.len() - 1 {
            // Subject: words before the verb (last 1-3 meaningful words)
            let subject_words: Vec<&str> = words[..i].iter()
                .rev()
                .filter(|w| {
                    let lw = w.to_lowercase();
                    let cleaned = lw.trim_matches(|c: char| !c.is_alphanumeric());
                    cleaned.len() > 2 && !stop_words.contains(&cleaned)
                })
                .take(3)
                .copied()
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();

            // Object: words after the verb (first 1-4 meaningful words)
            let object_words: Vec<&str> = words[i+1..]
                .iter()
                .filter(|w| {
                    let lw = w.to_lowercase();
                    let cleaned = lw.trim_matches(|c: char| !c.is_alphanumeric());
                    cleaned.len() > 2 && !stop_words.contains(&cleaned)
                })
                .take(4)
                .copied()
                .collect();

            if subject_words.is_empty() || object_words.is_empty() {
                continue;
            }

            let subject = subject_words.join(" ").to_lowercase()
                .trim_matches(|c: char| !c.is_alphanumeric() && c != ' ')
                .to_string();
            let object = object_words.join(" ").to_lowercase()
                .trim_matches(|c: char| !c.is_alphanumeric() && c != ' ')
                .to_string();
            let relation = clean.to_string();

            // Quality filters
            if subject.len() < 3 || object.len() < 3 { continue; }

            // Reject noise subjects
            let first_subj_word = subject.split_whitespace().next().unwrap_or("");
            if noise_starts.contains(&first_subj_word) { continue; }

            // Reject if subject or object contains commas (fragmented parse)
            if subject.contains(',') || object.contains(',') { continue; }

            // Reject if subject == object
            if subject == object { continue; }

            // Reject if subject or object is just a single common word
            let single_junk = ["thing", "stuff", "way", "lot", "kind", "sort",
                "bit", "point", "fact", "case", "time", "day", "minute",
                "people", "something", "someone", "anything", "everything",
                "paper", "video", "channel", "talk", "question", "answer",
                "gamechanger", "gamecher", "game", "huge", "total", "controversy",
                "points", "mostly", "along"];
            if single_junk.contains(&subject.as_str()) || single_junk.contains(&object.as_str()) {
                continue;
            }

            // Boost: if subject matches topic, always accept
            let topic_match = !topic_lower.is_empty() &&
                (subject.contains(topic_lower) || topic_lower.contains(&subject));

            // For non-topic subjects, require at least 2 meaningful words in object
            if !topic_match && object.split_whitespace().count() < 2 {
                continue;
            }

            triples.push(Triple::new(&subject, &relation, &object));
        }
    }

    triples
}

/// Extract triples without topic (backward compat).
pub fn extract_triples(sentence: &str) -> Vec<Triple> {
    extract_triples_with_topic(sentence, "")
}
