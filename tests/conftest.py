"""Shared test fixtures for the Cortex brain test suite."""
import pytest
import numpy as np
import sys
import os

# Add scripts dir to path so we can import functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


@pytest.fixture
def random_embedding_384():
    """Random 384-dim L2-normalized embedding (DINOv2 visual)."""
    emb = np.random.randn(384).astype(np.float32)
    return emb / np.linalg.norm(emb)


@pytest.fixture
def random_embedding_512():
    """Random 512-dim L2-normalized embedding (MLP projected)."""
    emb = np.random.randn(512).astype(np.float32)
    return emb / np.linalg.norm(emb)


@pytest.fixture
def random_weight_v():
    """Random W_v matrix (384 x 512)."""
    return np.random.randn(384, 512).astype(np.float32) * 0.01


@pytest.fixture
def random_weight_a():
    """Random W_a matrix (512 x 512)."""
    return np.random.randn(512, 512).astype(np.float32) * 0.01


@pytest.fixture
def concept_codebook():
    """Small mock concept codebook (10 concepts x 512 dims)."""
    labels = ["thunder", "rain", "dog barking", "cat meowing", "piano",
              "guitar", "speech", "wind", "bird", "car"]
    codebook = np.random.randn(10, 512).astype(np.float32)
    codebook = codebook / np.linalg.norm(codebook, axis=1, keepdims=True)
    return codebook, labels
