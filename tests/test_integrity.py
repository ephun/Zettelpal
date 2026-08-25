import builtins
import datetime
import os

from zettelpal.vault import integrity


def _note(vault, name, source, created, body):
    path = vault / name
    path.write_text(
        "\n".join([
            "---",
            f"id: {created.strftime('%Y%m%d%H%M%S')}",
            f'title: "{name}"',
            f"created: {created.isoformat()}",
            f'source: "{source}"',
            "tags:",
            "  - zettelpal",
            "---",
            f"# {name}",
            "",
            body,
            "",
        ]),
        encoding="utf-8",
    )
    return path


def test_text_duplicate_score_identical():
    text = "a" * 200
    assert integrity.text_duplicate_score(text, text) == 1.0


def test_text_duplicate_score_short_text_ignored():
    assert integrity.text_duplicate_score("short", "short") == 0.0


def test_text_duplicate_score_substring():
    # Must clear the 100-char minimum-length floor.
    short = ("This is a sufficiently long note body about a very specific topic, "
             "written out at enough length to exceed the detector's floor.")
    assert len(short) >= 100
    longer = "Intro. " + short + " Outro."
    assert integrity.text_duplicate_score(short, longer) == 1.0


def test_quarantine_moves_later_duplicate(sandbox):
    vault = sandbox / "vault"
    base = datetime.datetime(2026, 1, 1, 12, 0, 0)
    body = ("This is a note about family systems, memory, recurring patterns, "
            "and old conversations returning in new forms during reflective writing.")

    _note(vault, "Alpha - 20260101120000.md", "01012601", base, body)
    dup = _note(vault, "Alpha-Dup - 20260101120100.md", "01012601",
                base + datetime.timedelta(minutes=1), body)

    moved = integrity.quarantine_duplicate_notes_for_source(str(vault), "01012601")
    assert len(moved) == 1
    assert not dup.exists()
    assert moved[0]["relative_path"] == "Alpha-Dup - 20260101120100.md"


def test_quarantine_keeps_distinct_notes(sandbox):
    vault = sandbox / "vault"
    base = datetime.datetime(2026, 1, 1, 12, 0, 0)
    a = _note(vault, "A - 20260101120000.md", "01012601", base,
              "A note that is entirely about the history of cartography and old maps.")
    b = _note(vault, "B - 20260101120100.md", "01012601",
              base + datetime.timedelta(minutes=1),
              "A completely different note about baking sourdough bread at high altitude.")
    moved = integrity.quarantine_duplicate_notes_for_source(str(vault), "01012601")
    assert moved == []
    assert a.exists() and b.exists()


def test_validate_source_integrity_clean_note(sandbox):
    vault = sandbox / "vault"
    base = datetime.datetime(2026, 1, 1, 12, 0, 0)
    _note(vault, "Good - 20260101120000.md", "01012601", base,
          "A perfectly well-formed note body with plenty of text in it.")
    assert integrity.validate_source_integrity(str(vault), "01012601") == []


def test_validate_source_integrity_flags_missing_key(sandbox):
    vault = sandbox / "vault"
    # Note with a source but no tags key.
    (vault / "NoTags - 20260101120000.md").write_text(
        "\n".join([
            "---",
            "id: 20260101120000",
            'title: "NoTags"',
            "created: 2026-01-01T12:00:00",
            'source: "01012601"',
            "---",
            "# NoTags",
            "",
            "Body text here.",
            "",
        ]),
        encoding="utf-8",
    )
    issues = integrity.validate_source_integrity(str(vault), "01012601")
    codes = {code for _, code in issues}
    assert "missing_tags" in codes


def test_integrity_survives_unreadable_note(sandbox, monkeypatch):
    """One unreadable file used to abort the whole source integrity step with
    OSError (seen as '[Errno 22] Invalid argument' on a network vault)."""
    vault = sandbox / "vault"
    base = datetime.datetime(2026, 1, 1, 12, 0, 0)
    _note(vault, "Good - 20260101120000.md", "01012601", base,
          "A perfectly well-formed note body with plenty of text in it.")
    bad = _note(vault, "Bad - 20260101120100.md", "01012601",
                base + datetime.timedelta(minutes=1), "Another note body entirely.")

    real_open = builtins.open

    def refuse_one(file, *args, **kwargs):
        if os.path.abspath(str(file)) == os.path.abspath(str(bad)):
            raise OSError(22, "Invalid argument")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", refuse_one)

    assert integrity.validate_source_integrity(str(vault), "01012601") == []
    assert integrity.quarantine_duplicate_notes_for_source(str(vault), "01012601") == []
