"""Embedding backends.

The production encoder is a pinned local ONNX model (bge-small-en-v1.5 via fastembed):
free, offline after one download, reproducible. Everything downstream depends only on
the small protocol below, so tests inject deterministic fakes and never touch the model
(or the network), and the backend stays swappable.

bge models are *asymmetric*: queries are prepended with a fixed instruction sentence,
passages are encoded bare. Getting this wrong silently degrades retrieval quality, so
the prefix lives here as a constant next to the model id it belongs to.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

# Known bge v1.5 variants (all share the same query instruction).
KNOWN_MODELS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
}
MODEL_ID = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = KNOWN_MODELS[MODEL_ID]
# The instruction bge v1.5 models were trained to expect on the *query* side only.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

DEFAULT_MODEL_CACHE = Path("data/models")


class Encoder(Protocol):
    """What the index builder and retriever require of any embedding backend."""

    model_id: str
    dim: int

    def encode_passages(self, texts: list[str]) -> np.ndarray: ...  # (n, dim) unit-norm
    def encode_query(self, text: str) -> np.ndarray: ...  # (dim,) unit-norm


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return (matrix / np.maximum(norms, 1e-12)).astype(np.float32)


class FastEmbedEncoder:
    """A pinned local bge model. Imports fastembed lazily so offline tests never need it."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        *,
        cache_dir: Path | str = DEFAULT_MODEL_CACHE,
    ) -> None:
        from fastembed import TextEmbedding  # lazy: heavy import, model download

        if model_id not in KNOWN_MODELS:
            raise ValueError(f"Unknown model {model_id!r}; known: {sorted(KNOWN_MODELS)}")
        self.model_id = model_id
        self.dim = KNOWN_MODELS[model_id]
        self._model = TextEmbedding(model_name=model_id, cache_dir=str(cache_dir))

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        # Modest batch size: the default (256) with 512-token sequences allocates
        # multi-GB intermediates and thrashes memory — benchmarked 5x+ slower.
        vectors = np.array(list(self._model.embed(texts, batch_size=64)), dtype=np.float32)
        return _l2_normalize(vectors)  # defensive: normalization must be an invariant

    def encode_query(self, text: str) -> np.ndarray:
        (vector,) = list(self._model.embed([QUERY_PREFIX + text]))
        return _l2_normalize(np.asarray(vector, dtype=np.float32).reshape(1, -1))[0]
