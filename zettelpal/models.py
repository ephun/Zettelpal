# models.py - Local model loading (Whisper, SentenceTransformers) and device detection.

import numpy as np

from zettelpal import config

_device = None
_whisper_model = None
_embedding_model = None


def get_device() -> str:
    """Detects the compute device once, on first use (not at import time)."""
    global _device
    if _device is not None:
        return _device

    import torch

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    if _device == "cuda":
        try:
            _ = torch.randn(1).to(_device)
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"Warning: CUDA available but GPU not usable: {e}. Falling back to CPU.")
            _device = "cpu"
    else:
        print("Using CPU (GPU recommended for faster processing)")
    return _device


def load_whisper_model(size: str = None):
    """Loads the local Whisper transcription model (cached after first load)."""
    global _whisper_model
    if size is None:
        size = config.LOCAL_WHISPER_MODEL_SIZE

    if _whisper_model is None:
        import whisper

        print(f"Loading Whisper model: {size}...")
        try:
            _whisper_model = whisper.load_model(size, device=get_device())
            print("Whisper model loaded.")
        except Exception as e:
            print(f"Error loading Whisper model '{size}': {e}")
            _whisper_model = None
    return _whisper_model


def load_embedding_model():
    """Loads the local SentenceTransformer embedding model (cached after first load)."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        print(f"Loading embedding model: {config.LOCAL_EMBEDDING_MODEL}...")
        try:
            _embedding_model = SentenceTransformer(
                config.LOCAL_EMBEDDING_MODEL, device=get_device()
            )
            print("Embedding model loaded.")
        except Exception as e:
            print(f"Error loading embedding model '{config.LOCAL_EMBEDDING_MODEL}': {e}")
            _embedding_model = None
    return _embedding_model


def get_embedding(text: str, model=None) -> np.ndarray | None:
    """Generates an embedding vector for the given text."""
    if not text or not text.strip():
        return None

    embedding_model = model if model is not None else load_embedding_model()
    if embedding_model is None:
        print("Error: Embedding model not loaded.")
        return None

    try:
        embedding = embedding_model.encode([text.strip()])[0]
        return np.array(embedding)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None
