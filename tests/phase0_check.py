"""Phase 0 acceptance check: CUDA GPU, bge-m3 dimension, and the ``rag`` schema.

Run with: ``python -m tests.phase0_check``
"""

from __future__ import annotations

import torch

from core.config import get_settings
from core.db import bootstrap_schema, check_database, close_pool
from core.embedder import get_embedder


def main() -> None:
    settings = get_settings()

    print("=== Phase 0 verification: CUDA ===")
    print(f"torch version: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
    print(f"EMBEDDING_MODEL (config): {settings.embedding_model}")
    print(f"EMBEDDING_DEVICE (config): {settings.embedding_device}")

    print("\n=== Phase 0 verification: embedder ===")
    embedder = get_embedder()
    print(f"embedder.model_name: {embedder.model_name}")
    print(f"embedder.device: {embedder.device}")
    print(f"embedding dimension: {embedder.dimension}")
    vector = embedder.embed_query("هل يمكنني التبرع بالدم؟")
    norm = sum(value * value for value in vector) ** 0.5
    print(f"sample vector length: {len(vector)}")
    print(f"sample vector L2 norm: {norm:.6f}")

    assert torch.cuda.is_available() is True, "CUDA must be available for Phase 0"
    assert embedder.device == "cuda", "Embedder must run on CUDA"
    assert embedder.dimension == 1024, "bge-m3 dimension must be 1024"
    assert len(vector) == 1024, "Encoded vector must have 1024 values"
    assert abs(norm - 1.0) < 1e-3, "Vectors must be L2-normalized"

    print("\n=== Phase 0 verification: PostgreSQL + pgvector ===")
    bootstrap_schema()
    info = check_database()
    for key, value in info.items():
        print(f"{key}: {value}")
    assert info["pgvector_version"] != "NOT INSTALLED", "pgvector extension is missing"
    assert info["chunks"] == 0, "rag.chunks must start empty before ingestion"

    print("\n=== Phase 0 verification PASSED ===")
    close_pool()


if __name__ == "__main__":
    main()
