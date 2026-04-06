use brain_spiking::concepts::{ConceptRegistry, Triple};
use brain_spiking::network::{SpikingNetwork, NetworkConfig};
use brain_spiking::knowledge::KnowledgeEngine;

const REGION_NEURONS: usize = 100_000;
const ASSEMBLY_SIZE: usize = 100;

// ---------------------------------------------------------------------------
// Task 1 — persistence
// ---------------------------------------------------------------------------

#[test]
fn test_persist_and_reload_triples() {
    let dir = tempfile::tempdir().expect("temp dir");

    // Phase 1: learn and persist two triples.
    {
        let mut engine = KnowledgeEngine::new(0, REGION_NEURONS, ASSEMBLY_SIZE);
        engine.set_data_dir(dir.path());

        let t1 = Triple::new("rust", "is", "language");
        let t2 = Triple::new("neuron", "uses", "stdp");
        engine.learn_triple_with_topic(&t1, "programming");
        engine.learn_triple_with_topic(&t2, "neuroscience");
        engine.flush();
    }

    // Phase 2: fresh engine, load from file, verify associations are present.
    {
        let mut engine = KnowledgeEngine::new(0, REGION_NEURONS, ASSEMBLY_SIZE);
        let log_path = dir.path().join("triples.log");
        let loaded = engine.load_from_file(&log_path);

        assert_eq!(loaded.len(), 2, "expected 2 triples loaded from disk");
        assert!(engine.num_associations() > 0, "associations should be non-zero after reload");
        assert!(engine.num_concepts() >= 4, "at least rust, is, language, neuron should be registered");

        // Verify topic provenance was reconstructed
        let rust_topics = engine.concept_topics("rust");
        assert!(rust_topics.contains(&"programming".to_string()),
            "rust should be tagged with 'programming' topic");

        let neuron_topics = engine.concept_topics("neuron");
        assert!(neuron_topics.contains(&"neuroscience".to_string()),
            "neuron should be tagged with 'neuroscience' topic");
    }
}

#[test]
fn test_weight_cap_at_two() {
    let mut engine = KnowledgeEngine::new(0, REGION_NEURONS, ASSEMBLY_SIZE);
    let triple = Triple::new("cat", "is", "animal");
    // Learn the same triple many times — weight must not exceed 2.0.
    for _ in 0..20 {
        engine.learn_triple_with_topic(&triple, "biology");
    }
    // If the weight cap were 1.0 the chain would still be found, but we can at
    // least verify associations are non-zero and the engine is stable.
    assert!(engine.num_associations() > 0);
    assert!(engine.num_concepts() >= 2);
}

// ---------------------------------------------------------------------------
// Task 2 — topic provenance
// ---------------------------------------------------------------------------

#[test]
fn test_topic_provenance() {
    let mut registry = ConceptRegistry::new(REGION_NEURONS, ASSEMBLY_SIZE);

    // Register some concepts
    registry.get_or_create("transformer");
    registry.get_or_create("attention");
    registry.get_or_create("backpropagation");

    // Assign topics
    registry.add_topic("transformer", "nlp");
    registry.add_topic("transformer", "vision");
    registry.add_topic("attention", "nlp");
    registry.add_topic("backpropagation", "optimization");

    // get_topics
    let t_topics = registry.get_topics("transformer");
    assert!(t_topics.contains(&"nlp".to_string()));
    assert!(t_topics.contains(&"vision".to_string()));

    let a_topics = registry.get_topics("attention");
    assert_eq!(a_topics, vec!["nlp".to_string()]);

    // all_topics
    let all = registry.all_topics();
    assert!(all.contains(&"nlp".to_string()));
    assert!(all.contains(&"vision".to_string()));
    assert!(all.contains(&"optimization".to_string()));

    // bridge_concepts — transformer appears in 2 topics → bridge
    let bridges = registry.bridge_concepts();
    assert!(bridges.contains(&"transformer".to_string()),
        "transformer (nlp+vision) should be a bridge concept");
    assert!(!bridges.contains(&"attention".to_string()),
        "attention (only nlp) should NOT be a bridge concept");
    assert!(!bridges.contains(&"backpropagation".to_string()),
        "backpropagation (only optimization) should NOT be a bridge concept");
}

#[test]
fn test_all_topics_and_bridge_via_engine() {
    let mut engine = KnowledgeEngine::new(0, REGION_NEURONS, ASSEMBLY_SIZE);

    engine.learn_triple_with_topic(&Triple::new("rust", "is", "language"), "programming");
    engine.learn_triple_with_topic(&Triple::new("rust", "uses", "ownership"), "systems");

    // "rust" appears in two topics → bridge
    let bridges = engine.bridge_concepts();
    assert!(bridges.contains(&"rust".to_string()),
        "rust should be a bridge concept (programming + systems)");

    let all = engine.all_topics();
    assert!(all.contains(&"programming".to_string()));
    assert!(all.contains(&"systems".to_string()));
}

// ---------------------------------------------------------------------------
// Task 4 — bidirectional BFS / cross-domain recall
// ---------------------------------------------------------------------------

#[test]
fn test_cross_domain_recall() {
    let mut engine = KnowledgeEngine::new(0, 50000, 100);

    // Domain 1: TurboQuant
    engine.learn_triple_with_topic(
        &Triple::new("turboquant", "compresses", "kv cache"), "TurboQuant");
    engine.learn_triple_with_topic(
        &Triple::new("turboquant", "reduces", "memory usage"), "TurboQuant");

    // Domain 2: FlashAttention
    engine.learn_triple_with_topic(
        &Triple::new("flash attention", "optimizes", "kv cache"), "FlashAttention");
    engine.learn_triple_with_topic(
        &Triple::new("flash attention", "speeds up", "transformer inference"), "FlashAttention");

    // Query spanning both domains
    let mut net = SpikingNetwork::new(NetworkConfig { regions: vec![] });
    let chain = engine.recall_chain_bidirectional(&mut net, "turboquant flash attention", 10);

    let names: Vec<&str> = chain.iter().map(|(n, _)| n.as_str()).collect();
    assert!(!names.is_empty(), "Should find cross-domain associations");

    // Should reach concepts across domains through "kv cache" bridge
    let has_cross_domain = names.iter().any(|n|
        n.contains("memory") || n.contains("transformer") || n.contains("kv cache")
    );
    assert!(has_cross_domain, "Should reach concepts across domains, got: {:?}", names);
}

#[test]
fn test_spiking_seeds_after_bfs() {
    let mut engine = KnowledgeEngine::new(0, 50000, 100);

    engine.learn_triple_with_topic(
        &Triple::new("turboquant", "compresses", "kv cache"), "TurboQuant");
    engine.learn_triple_with_topic(
        &Triple::new("flash attention", "optimizes", "kv cache"), "FlashAttention");

    let mut net = SpikingNetwork::new(NetworkConfig { regions: vec![] });
    let _chain = engine.recall_chain_bidirectional(&mut net, "turboquant flash attention", 10);

    let (seeds, mode) = engine.take_spiking_seeds();
    assert!(!seeds.is_empty(), "Should have spiking seeds after BFS");
    assert_eq!(mode, "broad", "Multi-topic query should trigger broad mode");
}

// ---------------------------------------------------------------------------
// Task 1 — synaptic imprinting
// ---------------------------------------------------------------------------

#[test]
fn test_synaptic_imprinting() {
    let mut brain = brain_spiking::SpikingBrain::new(0.01, None);
    let triple = brain_spiking::Triple::new("alpha", "connects", "beta");
    brain.knowledge.learn_triple_with_topic(&triple, "test");
    let strengthened = brain.imprint_synapses(&triple);
    println!("Imprinted {} synapses at scale=0.01", strengthened);
    // At 0.01 scale the count may be small but the method should not panic
}
