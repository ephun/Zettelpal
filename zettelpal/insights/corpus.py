# corpus.py - Load every vault note with a unit-norm embedding for analysis.
#
# Reuses the shared embedding cache that linking maintains, so a warm cache
# means no model load and no recomputation. Only cache misses touch the
# embedding model, and any newly computed vectors are written back once.

import os

import numpy as np

from zettelpal import config, models
from zettelpal.log import get_logger
from zettelpal.vault import cache as vault_cache
from zettelpal.vault import notes

log = get_logger(__name__)


def _unit(vec) -> np.ndarray | None:
    """Return the L2-normalized vector, or None if it has no magnitude."""
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= 0:
        return None
    return arr / norm


def load_corpus(min_body_len: int = 1, save_cache: bool = True) -> list[dict]:
    """Load all vault notes (insight/system dirs excluded) as analysis records.

    Each record: {path, stem, title, body, created, tags, embedding} where
    embedding is a unit-norm float32 vector (so cosine similarity is a dot
    product). Notes without usable text or an embedding are skipped.
    """
    vault_root = config.settings.vault_root
    files = notes.find_all_markdown_files(vault_root)
    cache = vault_cache.load_embedding_cache()

    corpus: list[dict] = []
    cache_dirty = False
    model = None  # loaded lazily, only on a cache miss

    for path in files:
        note = notes.read_note(path, vault_root)
        body = note["body"]
        if len(body.strip()) < min_body_len:
            continue

        embedding = vault_cache.embedding_for_note(note, cache)  # normalized or None
        if embedding is None:
            if model is None:
                model = models.load_embedding_model()
            raw = models.get_embedding(body, model) if model is not None else None
            embedding = _unit(raw)
            if embedding is not None:
                relpath = os.path.relpath(path, vault_root)
                cache[relpath] = {
                    "embedding": embedding.tolist(),
                    "mtime": os.path.getmtime(path) if os.path.exists(path) else None,
                }
                cache_dirty = True

        if embedding is None:
            continue

        corpus.append({
            "path": path,
            "stem": note["stem"],
            "title": note["title"],
            "body": body,
            "created": note["created"],
            "tags": note["tags"],
            "embedding": embedding,
        })

    if cache_dirty and save_cache:
        vault_cache.save_embedding_cache(cache)

    log.info("[INSIGHTS] Loaded %d notes for analysis.", len(corpus))
    return corpus


def embedding_matrix(corpus: list[dict]) -> np.ndarray:
    """Stack corpus embeddings into an (n, d) matrix."""
    if not corpus:
        return np.zeros((0, 0), dtype=np.float32)
    return np.array([c["embedding"] for c in corpus], dtype=np.float32)
