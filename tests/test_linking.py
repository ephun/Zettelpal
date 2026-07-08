import datetime
import json

from zettelpal import config
from zettelpal.vault import linking


def _note(vault, name, source, created, body):
    title = name.split(" - ")[0]
    (vault / name).write_text(
        "\n".join([
            "---",
            f"id: {created.strftime('%Y%m%d%H%M%S')}",
            f'title: "{title}"',
            f"created: {created.isoformat()}",
            f'source: "{source}"',
            "tags:",
            "  - zettelpal",
            "---",
            f"# {title}",
            "",
            body,
            "",
        ]),
        encoding="utf-8",
    )
    return vault / name


def _seed_cache(vault, entries):
    cache = {}
    for name, title, source, created, embedding in entries:
        cache[name] = {
            "display_title": title,
            "filename_stem": name[:-3],
            "source": source,
            "created": created.isoformat(),
            "mtime": (vault / name).stat().st_mtime,
            "embedding": embedding,
        }
    (vault / ".obsidian" / "zettelpal_embeddings_cache.json").write_text(
        json.dumps(cache), encoding="utf-8"
    )


def test_linking_writes_chronological_and_semantic_links(sandbox):
    vault = sandbox / "vault"
    base = datetime.datetime(2026, 1, 1, 12, 0, 0)

    alpha = _note(vault, "Alpha - 20260101120000.md", "01012601", base, "First body.")
    beta = _note(vault, "Beta - 20260101120100.md", "01012601",
                 base + datetime.timedelta(minutes=1), "Second body.")
    related = _note(vault, "Related - 20260101120200.md", "01012602",
                    base + datetime.timedelta(minutes=2), "Alpha-like body.")

    _seed_cache(vault, [
        ("Alpha - 20260101120000.md", "Alpha", "01012601", base, [1.0, 0.0, 0.0]),
        ("Beta - 20260101120100.md", "Beta", "01012601",
         base + datetime.timedelta(minutes=1), [0.0, 1.0, 0.0]),
        ("Related - 20260101120200.md", "Related", "01012602",
         base + datetime.timedelta(minutes=2), [0.99, 0.01, 0.0]),
    ])

    cache_path = config.settings.embeddings_cache_file
    all_notes, _ = linking.update_and_load_vault_embeddings(str(vault), cache_path, [])
    linking.update_all_notes_links(all_notes, config.settings.similarity_threshold,
                                   last_threshold=None, any_note_was_updated_or_created=True)

    alpha_text = alpha.read_text(encoding="utf-8")
    beta_text = beta.read_text(encoding="utf-8")
    related_text = related.read_text(encoding="utf-8")

    assert config.LINK_BLOCK_START in alpha_text
    # Chronological: Alpha <-> Beta share a source.
    assert "[[Beta - 20260101120100|Beta]]" in alpha_text
    assert "[[Alpha - 20260101120000|Alpha]]" in beta_text
    # Semantic: Alpha and Related have near-identical embeddings, different source.
    assert "[[Related - 20260101120200|Related]]" in alpha_text
    assert "[[Alpha - 20260101120000|Alpha]]" in related_text
    # No self-link: the body (before the link block) has no wikilink to itself.
    assert "[[Alpha - 20260101120000|Alpha]]" not in alpha_text.split(config.LINK_BLOCK_START)[0]
