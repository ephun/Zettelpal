"""Tests for the insights module (theme clustering, RAG, digest, resurfacing).

Uses the shared sandbox + fake_embeddings fixtures; the LLM is faked per test
so nothing needs a network or a real model."""

import datetime
import json
import os

import pytest

from zettelpal import config
from zettelpal.insights import corpus as corpus_mod
from zettelpal.insights import digest, rag, resurfacing, themes, writer
from zettelpal.vault import notes


def _write_note(vault, name, body, created="2026-07-10", tags=("idea/testing",)):
    tag_lines = "".join(f"  - {t}\n" for t in tags)
    content = (
        "---\n"
        f"created: {created}\n"
        "tags:\n"
        f"{tag_lines}"
        "---\n\n"
        f"# {name}\n\n{body}\n"
    )
    path = os.path.join(vault, f"{name}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


@pytest.fixture
def fake_insight_llm(monkeypatch):
    """LLM that returns a theme-label JSON, digest prose, or a plain answer
    depending on which prompt it sees."""
    from zettelpal import llm

    def fake(prompt, temperature=0.3, max_tokens=None, max_retries=3):
        if "theme name" in prompt:
            return json.dumps({"label": "Test Theme", "summary": "About testing."})
        if "digest" in prompt.lower():
            return "You spent the week thinking about testing."
        return "Based on your notes, you focused on testing [1]."

    monkeypatch.setattr(llm, "llm_chat", fake)
    return fake


# --- corpus ---------------------------------------------------------------

def test_load_corpus_reads_notes(sandbox, fake_embeddings):
    vault = config.settings.vault_root
    _write_note(vault, "note-a", "family systems and recurring memory patterns")
    _write_note(vault, "note-b", "clinical reasoning and reflective practice")

    corpus = corpus_mod.load_corpus()

    assert len(corpus) == 2
    titles = {c["title"] for c in corpus}
    assert titles == {"note-a", "note-b"}
    for record in corpus:
        assert record["embedding"] is not None
        # Unit-norm embedding.
        import numpy as np
        assert abs(float(np.linalg.norm(record["embedding"])) - 1.0) < 1e-5


def test_corpus_excludes_insights_dir(sandbox, fake_embeddings):
    vault = config.settings.vault_root
    _write_note(vault, "real-entry", "an actual note the user wrote")
    # A generated insight note living in the insights subfolder.
    writer.write_insight("themes", "Themes", "some generated content", filename="Themes.md")

    corpus = corpus_mod.load_corpus()

    assert [c["title"] for c in corpus] == ["real-entry"]
    # The insight note is also invisible to the vault file scan.
    scanned = notes.find_all_markdown_files(vault)
    assert all("Insights" not in os.path.relpath(p, vault) for p in scanned)


# --- themes ---------------------------------------------------------------

def test_cluster_corpus_covers_all_notes(sandbox, fake_embeddings):
    vault = config.settings.vault_root
    for i in range(6):
        _write_note(vault, f"note-{i}", f"topic about subject number {i % 2}")
    corpus = corpus_mod.load_corpus()

    clusters = themes.cluster_corpus(corpus, k=2)

    assert len(clusters) == 2
    total = sum(len(c["members"]) for c in clusters)
    assert total == len(corpus)


def test_generate_themes_writes_note(sandbox, fake_embeddings, fake_insight_llm):
    vault = config.settings.vault_root
    for i in range(5):
        _write_note(vault, f"note-{i}", f"a thought about testing number {i}")

    path = themes.generate_themes(k=2)

    assert path is not None
    assert os.path.dirname(path) == config.settings.insights_dir
    with open(path, encoding="utf-8") as f:
        text = f.read()
    assert "Test Theme" in text
    assert "[[note-0|note-0]]" in text


def test_generate_themes_empty_vault(sandbox, fake_embeddings, fake_insight_llm):
    assert themes.generate_themes() is None


# --- rag ------------------------------------------------------------------

def test_ask_retrieves_relevant_note_first(sandbox, fake_embeddings):
    vault = config.settings.vault_root
    _write_note(vault, "cooking", "recipes for bread and pasta and sauce")
    _write_note(vault, "philosophy", "free will determinism and consciousness")
    corpus = corpus_mod.load_corpus()

    hits = rag.retrieve("bread and pasta recipes", corpus, k=1)

    assert len(hits) == 1
    assert hits[0]["title"] == "cooking"


def test_ask_returns_answer(sandbox, fake_embeddings, fake_insight_llm):
    vault = config.settings.vault_root
    _write_note(vault, "testing", "unit tests and coverage and assertions")

    result = rag.ask("what about testing?", k=3)

    assert "testing" in result["answer"]
    assert result["sources"]
    assert result["path"] is None


def test_ask_no_notes(sandbox, fake_embeddings, fake_insight_llm):
    result = rag.ask("anything?")
    assert result["sources"] == []


# --- digest ---------------------------------------------------------------

def test_digest_windows_by_date(sandbox, fake_embeddings, fake_insight_llm):
    vault = config.settings.vault_root
    _write_note(vault, "recent", "a fresh thought", created="2026-07-10")
    _write_note(vault, "old", "an ancient thought", created="2026-01-01")

    end = datetime.datetime(2026, 7, 11)
    body = digest.generate_digest(days=7, end=end, write=False)

    assert body is not None
    assert "recent" in body
    assert "old" not in body


def test_digest_empty_window(sandbox, fake_embeddings, fake_insight_llm):
    vault = config.settings.vault_root
    _write_note(vault, "old", "an ancient thought", created="2026-01-01")
    end = datetime.datetime(2026, 7, 11)
    assert digest.generate_digest(days=7, end=end) is None


# --- resurface ------------------------------------------------------------

def test_resurface_finds_related_older_note(sandbox, fake_embeddings):
    vault = config.settings.vault_root
    _write_note(vault, "recent", "quantum mechanics and wave functions",
                created="2026-07-10")
    _write_note(vault, "older-related", "quantum mechanics and wave functions",
                created="2026-01-01")
    _write_note(vault, "older-unrelated", "gardening tomatoes in spring",
                created="2026-01-01")
    corpus = corpus_mod.load_corpus()

    end = datetime.datetime(2026, 7, 11)
    items = resurfacing.find_resurfaced(corpus, days=7, threshold=0.5, end=end)

    stems = [item["note"]["stem"] for item in items]
    assert "older-related" in stems
    assert "older-unrelated" not in stems


def test_resurface_needs_both_recent_and_old(sandbox, fake_embeddings):
    vault = config.settings.vault_root
    _write_note(vault, "recent-1", "a thought", created="2026-07-10")
    _write_note(vault, "recent-2", "another thought", created="2026-07-10")
    corpus = corpus_mod.load_corpus()
    end = datetime.datetime(2026, 7, 11)
    assert resurfacing.find_resurfaced(corpus, days=7, end=end) == []
