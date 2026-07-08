"""Shared fixtures: a sandboxed vault/config, plus fakes for the three
external systems (LLM, embedding model, Whisper) so tests never need network,
torch, or a GPU."""

import hashlib
import json
import os

import numpy as np
import pytest

from zettelpal import config


@pytest.fixture
def sandbox(tmp_path):
    """Point config at a temporary vault + data dir and restore afterwards."""
    vault = tmp_path / "vault"
    (vault / ".obsidian").mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    settings = config.settings
    saved = {
        "vault_root": settings.vault_root,
        "data_dir": settings.data_dir,
        "notes_subdirectory": settings.notes_subdirectory,
        "archive_dir": settings.archive_dir,
        "similarity_threshold": settings.similarity_threshold,
        "max_semantic_links_per_note": settings.max_semantic_links_per_note,
    }
    settings.vault_root = str(vault)
    settings.data_dir = str(data_dir)
    settings.notes_subdirectory = ""
    settings.archive_dir = str(tmp_path / "archive")
    try:
        yield tmp_path
    finally:
        for key, value in saved.items():
            setattr(settings, key, value)


def _deterministic_embedding(text: str, dims: int = 24) -> np.ndarray:
    """Bag-of-words hashed into a fixed space and normalized. Deterministic,
    and similar text yields similar vectors, so semantic linking is testable
    without a real model."""
    vec = np.zeros(dims, dtype=np.float32)
    for word in "".join(c.lower() if c.isalnum() else " " for c in text).split():
        h = int(hashlib.sha1(word.encode()).hexdigest(), 16)
        vec[h % dims] += 1.0
    norm = np.linalg.norm(vec)
    if norm == 0:
        vec[0] = 1.0
        return vec
    return vec / norm


class FakeLLM:
    """Routes prompts to canned responses by matching text from the real
    prompt templates. Records calls so tests can assert on them."""

    def __init__(self, classification="type/note", segments=None, tags=None):
        self.classification = classification
        self.segments = segments if segments is not None else [
            {"title": "First Thought", "emoji": "seedling", "content":
                "A note about family systems and memory and recurring patterns."},
            {"title": "Second Thought", "emoji": "brain", "content":
                "A separate note about clinical reasoning and reflective practice."},
        ]
        self.tags = tags if tags is not None else ["idea/testing"]
        self.calls = []

    def __call__(self, prompt, temperature=0.3, max_tokens=None, max_retries=3):
        self.calls.append(prompt)
        if "type/note" in prompt:  # classification prompt
            return self.classification
        if "four valid tag categories" in prompt:  # tagging prompt
            return json.dumps(self.tags)
        # segmentation prompt
        return json.dumps(self.segments)


@pytest.fixture
def fake_llm(monkeypatch):
    """Install a FakeLLM. classify/segment/create_notes all call the shared
    llm module's llm_chat at call time, so patching it once covers them."""
    fake = FakeLLM()
    from zettelpal import llm

    monkeypatch.setattr(llm, "llm_chat", fake)
    return fake


@pytest.fixture
def fake_embeddings(monkeypatch):
    """Replace the embedding model with a deterministic hashing embedder."""
    from zettelpal import models

    monkeypatch.setattr(models, "load_embedding_model", lambda *a, **k: object())
    monkeypatch.setattr(
        models, "get_embedding",
        lambda text, model=None: _deterministic_embedding(text) if text and text.strip() else None,
    )
    return _deterministic_embedding


@pytest.fixture
def fake_whisper(monkeypatch):
    """Replace transcription with one that writes a transcript + empty
    timestamp sidecar (so clip extraction is skipped)."""
    from zettelpal import transcribe

    def fake_transcribe(audio_filepath, output_filepath):
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write("This is the transcribed text of the recording.")
        json_path = os.path.splitext(output_filepath)[0] + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([], f)
        return os.path.abspath(output_filepath), os.path.abspath(json_path)

    monkeypatch.setattr(transcribe, "transcribe_audio_to_file", fake_transcribe)
    return fake_transcribe
