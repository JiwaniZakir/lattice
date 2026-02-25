"""
Task embedding for routing decisions.

Computes dense vector representations of tasks for use by the
contextual bandit router and task classifier. Supports multiple
embedding strategies:

    1. TF-IDF with SVD projection (local, no API needed)
    2. Sentence-Transformer models (local GPU/CPU)
    3. OpenAI embeddings API (remote)

The TF-IDF approach provides a fast, dependency-free baseline that
works well for routing decisions.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class EmbeddingResult(BaseModel):
    """Result of embedding a task."""

    embedding: list[float]
    model: str
    dimension: int
    cached: bool = False


class TaskEmbedder:
    """
    Task embedder supporting multiple embedding strategies.

    The default strategy uses a lightweight TF-IDF + random projection
    approach that requires no external dependencies or API calls.
    This produces surprisingly effective embeddings for routing
    decisions where we need to distinguish task categories rather
    than capture fine-grained semantic similarity.

    For higher quality, configure an external embedding model.

    Args:
        dimension: Output embedding dimension.
        strategy: Embedding strategy ("tfidf", "hash", or "api").
        api_provider: Optional API provider for remote embeddings.
        cache_size: Maximum number of cached embeddings.
    """

    def __init__(
        self,
        dimension: int = 384,
        strategy: str = "tfidf",
        api_provider: Any = None,
        cache_size: int = 10_000,
    ) -> None:
        self._dimension = dimension
        self._strategy = strategy
        self._api_provider = api_provider
        self._cache: dict[str, np.ndarray] = {}
        self._cache_size = cache_size

        # IDF statistics for TF-IDF strategy
        self._document_freq: Counter[str] = Counter()
        self._total_documents = 0

        # Random projection matrix for dimensionality reduction.
        # Seeded for reproducibility.
        rng = np.random.RandomState(42)
        self._projection = rng.randn(10_000, dimension).astype(np.float32)
        self._projection /= np.linalg.norm(self._projection, axis=1, keepdims=True)

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> EmbeddingResult:
        """
        Compute an embedding for the given text.

        Args:
            text: Text to embed.

        Returns:
            EmbeddingResult with the embedding vector.
        """
        cache_key = hashlib.md5(text.encode()).hexdigest()

        if cache_key in self._cache:
            return EmbeddingResult(
                embedding=self._cache[cache_key].tolist(),
                model=self._strategy,
                dimension=self._dimension,
                cached=True,
            )

        if self._strategy == "api" and self._api_provider:
            embedding = await self._api_embed(text)
        elif self._strategy == "hash":
            embedding = self._hash_embed(text)
        else:
            embedding = self._tfidf_embed(text)

        # Cache the result
        if len(self._cache) >= self._cache_size:
            # Evict oldest entry
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[cache_key] = embedding

        return EmbeddingResult(
            embedding=embedding.tolist(),
            model=self._strategy,
            dimension=self._dimension,
            cached=False,
        )

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed multiple texts."""
        results = []
        for text in texts:
            result = await self.embed(text)
            results.append(result)
        return results

    def _tfidf_embed(self, text: str) -> np.ndarray:
        """
        Compute a TF-IDF embedding using random projection.

        Steps:
            1. Tokenize text into terms.
            2. Compute TF (term frequency) for each term.
            3. Compute IDF (inverse document frequency) from corpus stats.
            4. Build a sparse TF-IDF vector.
            5. Project into the target dimension using random projection.
            6. L2-normalize the result.
        """
        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(self._dimension, dtype=np.float32)

        # Update corpus statistics
        self._total_documents += 1
        unique_tokens = set(tokens)
        for token in unique_tokens:
            self._document_freq[token] += 1

        # TF
        tf: Counter[str] = Counter(tokens)
        max_tf = max(tf.values())

        # Build sparse TF-IDF vector using hash-based indexing
        sparse_dim = self._projection.shape[0]
        tfidf_vec = np.zeros(sparse_dim, dtype=np.float32)

        for token, count in tf.items():
            normalized_tf = 0.5 + 0.5 * (count / max_tf)
            df = self._document_freq.get(token, 1)
            idf = math.log(1 + self._total_documents / df)
            idx = hash(token) % sparse_dim
            tfidf_vec[idx] += normalized_tf * idf

        # Project to target dimension
        embedding = tfidf_vec @ self._projection

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm

        return embedding

    def _hash_embed(self, text: str) -> np.ndarray:
        """
        Feature hashing embedding (fast, fixed-dimension).

        Uses multiple hash functions to map tokens into a fixed-size
        vector. Less accurate than TF-IDF but very fast.
        """
        tokens = self._tokenize(text)
        embedding = np.zeros(self._dimension, dtype=np.float32)

        for token in tokens:
            for seed in range(3):
                h = int(hashlib.md5(f"{seed}:{token}".encode()).hexdigest(), 16)
                idx = h % self._dimension
                sign = 1.0 if (h >> 17) & 1 else -1.0
                embedding[idx] += sign

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm

        return embedding

    async def _api_embed(self, text: str) -> np.ndarray:
        """Compute embedding using an external API provider."""
        response = await self._api_provider.embed(text)
        embedding = np.array(response.embedding, dtype=np.float32)

        # Resize if needed
        if len(embedding) != self._dimension:
            if len(embedding) > self._dimension:
                embedding = embedding[: self._dimension]
            else:
                padded = np.zeros(self._dimension, dtype=np.float32)
                padded[: len(embedding)] = embedding
                embedding = padded

        return embedding

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Simple whitespace + punctuation tokenizer.

        Lowercases text, splits on whitespace and punctuation,
        and filters tokens shorter than 2 characters.
        """
        import re
        text = text.lower()
        tokens = re.findall(r"\b[a-z][a-z0-9]+\b", text)
        return tokens

    def clear_cache(self) -> int:
        """Clear the embedding cache. Returns count of cleared entries."""
        count = len(self._cache)
        self._cache.clear()
        return count
