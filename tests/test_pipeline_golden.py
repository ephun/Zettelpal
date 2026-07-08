"""End-to-end golden test: run the whole pipeline for one recording with the
LLM, embedding model, and Whisper all faked. Exercises transcribe -> classify
-> segment -> create notes -> integrity -> linking -> archive without any
external system."""

import os

from zettelpal import config
from zettelpal.vault import notes as vault_notes


def test_full_pipeline_creates_notes(sandbox, fake_llm, fake_embeddings, fake_whisper):
    from zettelpal import pipeline

    vault = sandbox / "vault"
    recording = sandbox / "07082601.m4a"
    recording.write_bytes(b"fake audio bytes" * 100)

    ok = pipeline.run_pipeline(str(recording), manual_tags="mood/calm")
    assert ok is True

    md_files = [f for f in os.listdir(vault) if f.endswith(".md")]
    assert len(md_files) == 2, md_files

    for name in md_files:
        note = vault_notes.read_note(str(vault / name), str(vault))
        assert note["frontmatter_error"] is None
        assert note["source"] == "07082601"
        assert note["created"] is not None
        assert "zettelpal" in note["tags"]
        assert "mood/calm" in note["tags"]           # manual tag
        assert "idea/testing" in note["tags"]        # from FakeLLM tagging
        assert note["body_len"] > 0
        assert note["has_link_markers"] is True      # link block written

    # Classification + segmentation + tagging all went through the FakeLLM.
    assert any("type/note" in p for p in fake_llm.calls)
    assert any("four valid tag categories" in p for p in fake_llm.calls)

    # Audio + transcript archived out of the working tree.
    assert not recording.exists()
    archive = config.settings.resolved_archive_dir
    archived = os.listdir(archive)
    assert any(f.endswith(".m4a") for f in archived)
    assert any(f.endswith(".txt") for f in archived)


def test_pipeline_rejects_bad_filename(sandbox, fake_llm, fake_embeddings, fake_whisper):
    from zettelpal import pipeline

    bad = sandbox / "not-a-recording.m4a"
    bad.write_bytes(b"x" * 100)
    assert pipeline.run_pipeline(str(bad)) is False


def test_pipeline_quarantines_duplicate_segments(
    sandbox, fake_llm, fake_embeddings, fake_whisper
):
    """If the LLM emits two identical segments, integrity should quarantine the
    duplicate rather than leaving two identical notes."""
    from zettelpal import pipeline

    dup_body = ("A single thought about the same subject repeated verbatim, long "
                "enough to clear the duplicate detector's minimum length floor.")
    fake_llm.segments = [
        {"title": "One", "emoji": "one", "content": dup_body},
        {"title": "Two", "emoji": "two", "content": dup_body},
    ]

    vault = sandbox / "vault"
    recording = sandbox / "07082602.m4a"
    recording.write_bytes(b"audio" * 100)

    assert pipeline.run_pipeline(str(recording)) is True
    md_files = [f for f in os.listdir(vault) if f.endswith(".md")]
    assert len(md_files) == 1, md_files
