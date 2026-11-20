"""Singleton local embedder for ``BAAI/bge-m3``.

Dense, L2-normalized 1024-dim vectors; fp16 on CUDA, CPU fallback with a
warning.  Documents and queries go through the exact same encoding path
(bge-m3 needs no instruction prefix).
"""

from __future__ import annotations

import asyncio
import threading
from typing import Sequence

import torch
from sentence_transformers import SentenceTransformer

from core.config import get_settings
from core.logging import get_logger

logger = get_logger(__name__)

EXPECTED_DIMENSION: int = 1024
MAX_SEQ_LENGTH: int = 512


class Embedder:
    """Thread-safe wrapper around a single loaded SentenceTransformer."""

    def __init__(self) -> None:
        settings = get_settings()
        cuda_available = torch.cuda.is_available()

        if settings.embedding_device == "cuda" and not cuda_available:
            logger.warning("cuda_unavailable_falling_back_to_cpu", extra={"requested_device": "cuda"})
            device = "cpu"
        else:
            device = settings.embedding_device

        self._model = SentenceTransformer(settings.embedding_model, device=device)
        self._model.max_seq_length = MAX_SEQ_LENGTH
        if device == "cuda":
            self._model.half()

        dimension = int(self._model.get_embedding_dimension())
        if dimension != EXPECTED_DIMENSION:
            raise RuntimeError(
                f"Embedding dimension mismatch: {settings.embedding_model} returned {dimension}, "
                f"expected {EXPECTED_DIMENSION}"
            )

        self.device: str = device
        self.dimension: int = dimension
        self.model_name: str = settings.embedding_model
        self.batch_size: int = settings.embedding_batch_size

        logger.info(
            "embedder_loaded",
            extra={
                "model": self.model_name,
                "device": device,
                "cuda_available": cuda_available,
                "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
                "dimension": dimension,
                "dtype": "float16" if device == "cuda" else "float32",
                "max_seq_length": MAX_SEQ_LENGTH,
                "batch_size": self.batch_size,
            },
        )

    def encode(self, texts: Sequence[str], batch_size: int | None = None) -> list[list[float]]:
        """Encode texts into normalized vectors (same path for docs and queries)."""
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size or self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [vector.tolist() for vector in vectors]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode a batch of chunk texts."""
        return self.encode(texts)

    def embed_query(self, text: str) -> list[float]:
        """Encode a single query text."""
        return self.encode([text], batch_size=1)[0]

    async def aembed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Async wrapper so blocking GPU work never stalls the event loop."""
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        """Async wrapper for a single query embedding."""
        return await asyncio.to_thread(self.embed_query, text)


_embedder: Embedder | None = None
_embedder_lock = threading.Lock()


def get_embedder() -> Embedder:
    """Return the process-wide embedder, loading the model on first use."""
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = Embedder()
    return _embedder
