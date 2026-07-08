import re

from zettelpal import intake, pipeline


def test_process_folder_renames_and_dedupes(sandbox, monkeypatch):
    inbox = sandbox / "inbox"
    inbox.mkdir()
    calls = []

    def fake_run(path, tags=""):
        import os
        calls.append(os.path.basename(path))
        os.remove(path)  # real pipeline archives the audio away
        return True

    monkeypatch.setattr(pipeline, "run_pipeline", fake_run)

    (inbox / "Voice Memo.mp3").write_bytes(b"CONTENT-A" * 50)
    (inbox / "07082601.m4a").write_bytes(b"CONTENT-B" * 50)

    processed, skipped, failed = intake.process_folder(str(inbox))
    assert (processed, skipped, failed) == (2, 0, 0)
    assert "07082601.m4a" in calls
    renamed = [c for c in calls if c != "07082601.m4a"][0]
    assert re.match(r"^\d{8}\.mp3$", renamed)


def test_duplicate_content_skipped(sandbox, monkeypatch):
    inbox = sandbox / "inbox"
    inbox.mkdir()
    calls = []

    def fake_run(path, tags=""):
        import os
        calls.append(path)
        os.remove(path)
        return True

    monkeypatch.setattr(pipeline, "run_pipeline", fake_run)

    (inbox / "07082601.m4a").write_bytes(b"SAME-BYTES" * 50)
    intake.process_folder(str(inbox))
    assert len(calls) == 1

    # Same content, different name -> skipped as duplicate.
    (inbox / "07082602.m4a").write_bytes(b"SAME-BYTES" * 50)
    processed, skipped, failed = intake.process_folder(str(inbox))
    assert (processed, skipped, failed) == (0, 1, 0)
    assert len(calls) == 1


def test_scan_folder_ignores_non_audio(sandbox):
    inbox = sandbox / "inbox"
    inbox.mkdir()
    (inbox / "a.m4a").write_bytes(b"x")
    (inbox / "notes.txt").write_text("hi")
    (inbox / "b.mp3").write_bytes(b"y")
    found = {p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for p in intake.scan_folder(str(inbox))}
    assert found == {"a.m4a", "b.mp3"}
