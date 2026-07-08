# cache.py - Embedding cache persistence (single implementation).

import datetime
import json
import os

import numpy as np

from zettelpal import config


class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)


def load_embedding_cache() -> dict:
    """Loads the embedding cache from JSON file."""
    if os.path.exists(config.EMBEDDINGS_CACHE_FILE):
        try:
            with open(config.EMBEDDINGS_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    print("Warning: Cache file is not a dictionary. Creating new one.")
                    return {}
                return data
        except json.JSONDecodeError as e:
            print(f"Warning: Could not decode embedding cache: {e}")
            return {}
        except OSError as e:
            print(f"Warning: Error loading embedding cache: {e}")
            return {}
    return {}


def save_embedding_cache(cache: dict):
    """Saves the embedding cache to JSON file."""
    os.makedirs(os.path.dirname(config.EMBEDDINGS_CACHE_FILE), exist_ok=True)
    try:
        with open(config.EMBEDDINGS_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, cls=_NpEncoder)
    except OSError as e:
        print(f"Error saving embedding cache: {e}")


def update_embedding_cache(filepath: str, embedding: np.ndarray):
    """Updates a single entry in the embedding cache."""
    cache = load_embedding_cache()
    relative_path = os.path.relpath(filepath, config.OBSIDIAN_VAULT_ROOT)
    cache[relative_path] = {
        "embedding": embedding.tolist() if embedding is not None else None,
        "mtime": os.path.getmtime(filepath) if os.path.exists(filepath) else None,
    }
    save_embedding_cache(cache)


def embedding_for_note(note: dict, cache: dict) -> np.ndarray | None:
    """Returns the normalized cached embedding for a read_note() dict, or None."""
    entry = cache.get(note["relpath"]) or cache.get(note["relpath"].replace("/", "\\"))
    if not isinstance(entry, dict) or not isinstance(entry.get("embedding"), list):
        return None

    embedding = np.asarray(entry["embedding"], dtype=np.float32)
    norm = np.linalg.norm(embedding)
    if norm <= 0:
        return None
    return embedding / norm
