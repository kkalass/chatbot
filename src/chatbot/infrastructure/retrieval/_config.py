"""RetrieverConfig — construction-time config for the retrieval adapter."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RetrieverConfig:
    """Construction-time config for the retrieval adapter.

    Carries both retrieval parameters and the vector-store connection details,
    so the retriever can build its own store internally without depending on the
    ingestion infrastructure.
    """

    top_k: int
    score_threshold: float
    store_host: str
    store_port: int
    store_collection: str
    embedding_dim: int
    store_similarity: str = "cosine"
    store_backend: Literal["qdrant"] = "qdrant"
