# models.py - Local model loading (Whisper, SentenceTransformers) and device detection.

import numpy as np

from zettelpal import config
from zettelpal.log import get_logger

log = get_logger(__name__)

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
            log.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            log.warning(f"Warning: CUDA available but GPU not usable: {e}. Falling back to CPU.")
            _device = "cpu"
    else:
        log.info("Using CPU (GPU recommended for faster processing)")
    return _device


def load_whisper_model(size: str = None):
    """Loads the local Whisper transcription model (cached after first load)."""
    global _whisper_model
    if size is None:
        size = config.settings.whisper_model_size

    if _whisper_model is None:
        import whisper

        log.info(f"Loading Whisper model: {size}...")
        try:
            _whisper_model = whisper.load_model(size, device=get_device())
            log.info("Whisper model loaded.")
        except Exception as e:
            log.error(f"Error loading Whisper model '{size}': {e}")
            _whisper_model = None
    return _whisper_model


def load_embedding_model():
    """Loads the local SentenceTransformer embedding model (cached after first load)."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        log.info(f"Loading embedding model: {config.settings.embedding_model}...")
        try:
            _embedding_model = SentenceTransformer(
                config.settings.embedding_model, device=get_device()
            )
            log.info("Embedding model loaded.")
        except Exception as e:
            log.error(f"Error loading embedding model '{config.settings.embedding_model}': {e}")
            _embedding_model = None
    return _embedding_model


def get_embedding(text: str, model=None) -> np.ndarray | None:
    """Generates an embedding vector for the given text."""
    if not text or not text.strip():
        return None

    embedding_model = model if model is not None else load_embedding_model()
    if embedding_model is None:
        log.error("Error: Embedding model not loaded.")
        return None

    try:
        embedding = embedding_model.encode([text.strip()])[0]
        return np.array(embedding)
    except Exception as e:
        log.error(f"Error generating embedding: {e}")
        return None
