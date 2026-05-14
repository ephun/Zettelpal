import datetime
import json
import os
import shutil
import tempfile
import wave

import numpy as np

import clip
import config
import link_notes
import vault_integrity


def write_note(vault, filename, title, source, created, body):
    path = os.path.join(vault, filename)
    with open(path, "w", encoding="utf-8") as file:
        file.write(
            "\n".join(
                [
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
                ]
            )
        )
    return path


def write_cache(vault, cache_path, notes):
    cache = {}
    for path, title, source, created, embedding in notes:
        relpath = os.path.relpath(path, vault).replace("\\", "/")
        cache[relpath] = {
            "display_title": title,
            "filename_stem": os.path.splitext(os.path.basename(path))[0],
            "source": source,
            "created": created.isoformat(),
            "mtime": os.path.getmtime(path),
            "embedding": embedding,
        }
    with open(cache_path, "w", encoding="utf-8") as file:
        json.dump(cache, file, indent=2)


def make_wav(path):
    with wave.open(path, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x00" * 16000 * 2)


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    temp_root = tempfile.mkdtemp(prefix="zettelpal_stress_")
    original_vault = config.OBSIDIAN_VAULT_ROOT
    original_cache = config.EMBEDDINGS_CACHE_FILE
    original_quarantine_root = config.ZETTELPAL_ROOT
    original_max_links = getattr(config, "MAX_SEMANTIC_LINKS_PER_NOTE", None)

    try:
        vault = os.path.join(temp_root, "vault")
        os.makedirs(os.path.join(vault, ".obsidian"), exist_ok=True)
        cache_path = os.path.join(vault, ".obsidian", "zettelpal_embeddings_cache.json")

        config.OBSIDIAN_VAULT_ROOT = vault
        config.EMBEDDINGS_CACHE_FILE = cache_path
        config.ZETTELPAL_ROOT = temp_root
        config.MAX_SEMANTIC_LINKS_PER_NOTE = 2

        source = "01012601"
        base_time = datetime.datetime(2026, 1, 1, 12, 0, 0)
        body_a = (
            "This is a note about family systems, memory, recurring patterns, "
            "and the way old conversations return in new forms during reflective writing."
        )
        body_b = (
            "This is a note about clinical reasoning, reflective practice, and "
            "the habit of slowing down before reaching for a conclusion."
        )
        body_c = (
            "This is a note about creative work, archives, technical projects, "
            "and the practical maintenance required to keep a system useful."
        )

        note_a = write_note(vault, "Alpha - 20260101120000.md", "Alpha", source, base_time, body_a)
        note_b = write_note(vault, "Beta - 20260101120100.md", "Beta", source, base_time + datetime.timedelta(minutes=1), body_b)
        note_c = write_note(vault, "Gamma - 20260101120200.md", "Gamma", source, base_time + datetime.timedelta(minutes=2), body_c)
        note_dup = write_note(vault, "Alpha-Duplicate - 20260101120300.md", "Alpha Duplicate", source, base_time + datetime.timedelta(minutes=3), body_a)

        other = write_note(
            vault,
            "Related - 20260101120400.md",
            "Related",
            "01012602",
            base_time + datetime.timedelta(minutes=4),
            "This related note also discusses family systems and recurring patterns.",
        )

        write_cache(
            vault,
            cache_path,
            [
                (note_a, "Alpha", source, base_time, [1.0, 0.0, 0.0]),
                (note_b, "Beta", source, base_time + datetime.timedelta(minutes=1), [0.0, 1.0, 0.0]),
                (note_c, "Gamma", source, base_time + datetime.timedelta(minutes=2), [0.0, 0.0, 1.0]),
                (note_dup, "Alpha Duplicate", source, base_time + datetime.timedelta(minutes=3), [1.0, 0.0, 0.0]),
                (other, "Related", "01012602", base_time + datetime.timedelta(minutes=4), [0.95, 0.05, 0.0]),
            ],
        )

        moved = vault_integrity.quarantine_duplicate_notes_for_source(vault, source)
        assert_true(len(moved) == 1, f"expected one duplicate moved, got {len(moved)}")
        assert_true(not os.path.exists(note_dup), "duplicate note should be quarantined")

        all_notes, _ = link_notes.update_and_load_vault_embeddings(vault, cache_path, [])
        link_notes.update_all_notes_links(
            all_notes,
            config.SIMILARITY_THRESHOLD,
            last_threshold=None,
            any_note_was_updated_or_created=True,
        )

        with open(note_a, "r", encoding="utf-8") as file:
            alpha_text = file.read()
        with open(note_b, "r", encoding="utf-8") as file:
            beta_text = file.read()

        assert_true(config.LINK_BLOCK_START in alpha_text, "generated link marker missing")
        assert_true("[[Beta - 20260101120100|Beta]]" in alpha_text, "Alpha should link to next chronological note")
        assert_true("[[Alpha - 20260101120000|Alpha]]" in beta_text, "Beta should link to previous chronological note")
        assert_true("[[Gamma - 20260101120200|Gamma]]" in beta_text, "Beta should link to next chronological note")
        assert_true("[[Alpha - 20260101120000|Alpha]]" not in alpha_text, "Alpha should not self-link")

        audio_source = os.path.join(temp_root, "source.wav")
        audio_clip = os.path.join(temp_root, "clip.wav")
        make_wav(audio_source)
        clip_ok = clip.extract_clip(audio_source, 0.0, 1.0, audio_clip)
        assert_true(clip_ok and os.path.exists(audio_clip), "audio clip extraction failed")

        print(json.dumps({"status": "ok", "temp_root": temp_root, "duplicate_moved": moved}, indent=2))
    finally:
        config.OBSIDIAN_VAULT_ROOT = original_vault
        config.EMBEDDINGS_CACHE_FILE = original_cache
        config.ZETTELPAL_ROOT = original_quarantine_root
        if original_max_links is None:
            delattr(config, "MAX_SEMANTIC_LINKS_PER_NOTE")
        else:
            config.MAX_SEMANTIC_LINKS_PER_NOTE = original_max_links
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
