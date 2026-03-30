"""Core function tests: MLP projection, sparse coding, grid cells, text extraction."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


class TestMLPProject:
    def test_shape_single(self, random_weight_v):
        from youtube_brain import mlp_project
        emb = np.random.randn(1, 384).astype(np.float32)
        result = mlp_project(emb, random_weight_v)
        assert result.shape == (1, 512)

    def test_shape_batch(self, random_weight_v):
        from youtube_brain import mlp_project
        emb = np.random.randn(10, 384).astype(np.float32)
        result = mlp_project(emb, random_weight_v)
        assert result.shape == (10, 512)

    def test_l2_normalized(self, random_weight_v):
        from youtube_brain import mlp_project
        emb = np.random.randn(5, 384).astype(np.float32)
        result = mlp_project(emb, random_weight_v)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_relu_nonnegative(self, random_weight_v):
        from youtube_brain import mlp_project
        emb = np.random.randn(5, 384).astype(np.float32)
        result = mlp_project(emb, random_weight_v)
        # After ReLU + normalization, values should be >= 0
        assert (result >= -1e-6).all()

    def test_sparse_k(self, random_weight_v):
        import youtube_brain
        old_k = youtube_brain.SPARSE_K
        youtube_brain.SPARSE_K = 50
        try:
            emb = np.random.randn(1, 384).astype(np.float32)
            result = youtube_brain.mlp_project(emb, random_weight_v)
            nonzero = np.count_nonzero(result[0])
            assert nonzero <= 50, f"Expected <=50 nonzero, got {nonzero}"
        finally:
            youtube_brain.SPARSE_K = old_k


class TestGridCells:
    def test_grid_encoder_fit(self, concept_codebook):
        from youtube_brain import GridCellEncoder
        codebook, labels = concept_codebook
        enc = GridCellEncoder()
        enc.fit(codebook)
        assert enc.projection is not None
        assert enc.projection.shape == (512, 2)

    def test_to_2d(self, concept_codebook):
        from youtube_brain import GridCellEncoder
        codebook, labels = concept_codebook
        enc = GridCellEncoder()
        enc.fit(codebook)
        pos = enc.to_2d(codebook[0])
        assert pos.shape == (2,)

    def test_to_2d_batch(self, concept_codebook):
        from youtube_brain import GridCellEncoder
        codebook, labels = concept_codebook
        enc = GridCellEncoder()
        enc.fit(codebook)
        pos = enc.to_2d(codebook)
        assert pos.shape == (10, 2)

    def test_grid_activation(self, concept_codebook):
        from youtube_brain import GridCellEncoder
        codebook, labels = concept_codebook
        enc = GridCellEncoder()
        enc.fit(codebook)
        pos = enc.to_2d(codebook[0])
        act = enc.grid_activation(pos)
        # 3 scales * 3 orientations * 2 (activation + phase) = 18
        assert act.shape == (18,)
        assert act.dtype == np.float32

    def test_grid_distance_self_zero(self, concept_codebook):
        from youtube_brain import GridCellEncoder
        codebook, _ = concept_codebook
        enc = GridCellEncoder()
        enc.fit(codebook)
        d = enc.grid_distance(codebook[0], codebook[0])
        assert d < 0.01, f"Self-distance should be ~0, got {d}"

    def test_find_nearby(self, concept_codebook):
        from youtube_brain import GridCellEncoder
        codebook, labels = concept_codebook
        enc = GridCellEncoder()
        enc.fit(codebook)
        results = enc.find_nearby(codebook[0], codebook, labels, top_k=5)
        assert len(results) == 5
        assert results[0]["distance"] < results[-1]["distance"]


class TestTextRelationExtraction:
    def test_causes(self):
        from youtube_brain import _extract_text_relations
        edges = _extract_text_relations("Lightning causes thunder")
        assert any(e["relation"] == "causes" for e in edges)

    def test_part_of(self):
        from youtube_brain import _extract_text_relations
        edges = _extract_text_relations("A dog is a type of animal")
        assert any(e["relation"] == "part-of" for e in edges)

    def test_sounds_like(self):
        from youtube_brain import _extract_text_relations
        edges = _extract_text_relations("Thunder sounds like a drum")
        assert any(e["relation"] == "sounds-like" for e in edges)

    def test_empty_text(self):
        from youtube_brain import _extract_text_relations
        edges = _extract_text_relations("")
        assert edges == []

    def test_no_relations(self):
        from youtube_brain import _extract_text_relations
        edges = _extract_text_relations("The quick brown fox jumps over the lazy dog")
        # May or may not find relations, but shouldn't crash
        assert isinstance(edges, list)


class TestKnowledgeGraph:
    def test_multi_hop_empty(self):
        from youtube_brain import _multi_hop_traverse
        results = _multi_hop_traverse("nonexistent_concept")
        assert isinstance(results, list)


class TestFastMemory:
    def test_store_retrieve(self):
        from youtube_brain import _fast_memory_store, _fast_memory_retrieve
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        _fast_memory_store(emb, "test_pattern")
        results = _fast_memory_retrieve(emb, top_k=1)
        assert len(results) >= 1
        assert results[0]["similarity"] > 0.9


class TestWorkingMemory:
    def test_theta_phase_advances(self):
        import youtube_brain
        old_phase = youtube_brain._wm_theta_phase
        emb = np.random.randn(512).astype(np.float32)
        youtube_brain._update_working_memory(emb, "test", "unit_test")
        new_phase = youtube_brain._wm_theta_phase
        assert new_phase != old_phase, "Theta phase should advance"

    def test_max_slots(self):
        import youtube_brain
        youtube_brain._working_memory = []
        for i in range(10):
            emb = np.random.randn(512).astype(np.float32)
            youtube_brain._update_working_memory(emb, f"item_{i}", "test")
        assert len(youtube_brain._working_memory) <= youtube_brain.WORKING_MEMORY_SLOTS
