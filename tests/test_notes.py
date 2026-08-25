from zettelpal import config
from zettelpal.vault import notes

FRONTMATTER = """---
id: 20260101120000
title: "A Test Note"
created: 2026-01-01T12:00:00
source: "01012601"
tags:
  - zettelpal
  - idea/testing
---
# A Test Note

This is the body of the note, long enough to be meaningful.
"""


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_read_note_parses_frontmatter(sandbox):
    vault = sandbox / "vault"
    path = _write(vault / "A Test Note - 20260101120000.md", FRONTMATTER)
    note = notes.read_note(path, str(vault))

    assert note["source"] == "01012601"
    assert note["title"] == "A Test Note"
    assert note["created"] is not None
    assert note["created"].year == 2026
    assert "zettelpal" in note["tags"]
    assert note["is_zettelpal"] is True
    assert note["frontmatter_error"] is None
    assert note["body_len"] > 0


def test_read_note_missing_frontmatter(sandbox):
    vault = sandbox / "vault"
    path = _write(vault / "plain.md", "Just some text, no frontmatter.\n")
    note = notes.read_note(path, str(vault))
    assert note["frontmatter_error"] == "missing_frontmatter"


def test_read_note_tolerates_bom(sandbox):
    vault = sandbox / "vault"
    path = vault / "bom.md"
    path.write_bytes(b"\xef\xbb\xbf" + FRONTMATTER.encode("utf-8"))
    note = notes.read_note(str(path), str(vault))
    assert note["frontmatter_error"] is None
    assert note["source"] == "01012601"


def test_split_body_and_generated_links_with_markers():
    body_text = "The body.\n"
    block = f"{config.LINK_BLOCK_START}\n[[Other - 123|Other]]\n{config.LINK_BLOCK_END}\n"
    content_lines = (body_text + "\n" + block).splitlines(keepends=True)
    body, link_lines, has_markers = notes.split_body_and_generated_links(content_lines)
    assert has_markers is True
    assert "The body." in body
    assert any("Other" in line for line in link_lines)


def test_generated_links_extracted(sandbox):
    vault = sandbox / "vault"
    text = FRONTMATTER + (
        f"\n{config.LINK_BLOCK_START}\n"
        "[[Neighbor - 20260101120100|Neighbor]]\n"
        f"{config.LINK_BLOCK_END}\n"
    )
    path = _write(vault / "A Test Note - 20260101120000.md", text)
    note = notes.read_note(path, str(vault))
    assert "Neighbor - 20260101120100" in note["generated_links"]
    assert note["has_link_markers"] is True


def test_read_note_unreadable_file_is_not_fatal(sandbox):
    """A file the OS refuses to open must not raise - callers scanning a whole
    vault would lose every other note to one bad file."""
    vault = sandbox / "vault"
    note = notes.read_note(str(vault / "Gone - 20260101120000.md"), str(vault))
    assert note["read_error"]
    assert note["body"] == ""
    assert note["source"] is None
