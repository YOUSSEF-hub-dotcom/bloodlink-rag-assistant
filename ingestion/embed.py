"""Ingestion-side embedding helpers.

The actual model lives in :mod:`core.embedder` (singleton, loaded once).  This
module provides the small interface used by :mod:`ingestion.run` so the
ingestion pipeline can be tested or swapped without touching core code.
"""

from __future__ import annotations

from collections.abc import Sequence

from core.embedder import Embedder, get_embedder


def get_ingestion_embedder() -> Embedder:
    """Return the shared local embedder."""
    return get_embedder()


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Embed chunk texts with the same path used for documents and queries."""
    return get_embedder().embed_documents(texts)
