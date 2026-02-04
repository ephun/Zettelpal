# create_notes.py — Create Obsidian notes + embeddings using local LLM

import os
import json
import re
import datetime
import config
import utils


def sanitize_filename(title: str) -> str:
    """Convert a title to a filesystem-safe filename stem."""
    s = title.lower()
    s = re.sub(r"[^a-z0-9\- ]+", "", s)
    s = s.replace(" ", "-").strip("-")
    return s or "untitled"


def format_yaml_frontmatter(title, emoji, segment_text, recording_id, transcript_type_tag, manual_tags):
    """Build YAML frontmatter."""
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")

    yaml = [
        "---",
        f'title: "{title}"',
        f'emoji: "{emoji}"',
        f"date: {date_str}",
        f"type: {transcript_type_tag}",
        f"recording: {recording_id}",
    ]

    # Manual tags
    if manual_tags:
        tags = [t.strip() for t in manual_tags.split(",") if t.strip()]
        if tags:
            yaml.append("manual-tags:")
            for t in tags:
                yaml.append(f"  - {t}")

    # Placeholder—semantic linking fills these later
    yaml.append("semantic-tags: []")

    yaml.append("---\n")
    return "\n".join(yaml)


def create_notes_from_segments(
    segmented_data: list,
    vault_root: str,
    subdir: str,
    recording_id: str,
    transcript_type_tag: str,
    manual_tags_str: str
):
    """
    Creates Obsidian markdown notes for each segment,
    stores embeddings (SentenceTransformer),
    returns list of dicts with note metadata.
    """

    output_dir = vault_root if not subdir else os.path.join(vault_root, subdir)
    os.makedirs(output_dir, exist_ok=True)

    created_notes_info = []

    for i, seg in enumerate(segmented_data):
        title = seg.get("title", f"Segment {i+1}")
        emoji = seg.get("emoji", "")
        text = seg.get("content", "")

        filename_stem = sanitize_filename(title)
        filename = f"{recording_id}_{filename_stem}.md"
        filepath = os.path.join(output_dir, filename)

        # Avoid overwriting
        counter = 1
        while os.path.exists(filepath):
            filepath = os.path.join(output_dir, f"{recording_id}_{filename_stem}-{counter}.md")
            counter += 1

        # Build note text
        frontmatter = format_yaml_frontmatter(title, emoji, text, recording_id, transcript_type_tag, manual_tags_str)
        full_note = f"{frontmatter}\n{text}\n"

        # Write note to disk
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(full_note)
        except Exception as e:
            print(f"[NOTE ERROR] Could not write {filepath}: {e}")
            continue

        print(f"[NOTE CREATED] {filepath}")

        # Compute embedding (local SentenceTransformer)
        print(f"[EMBED] Generating embedding for: {filepath}")
        embedding = utils.get_embedding(full_note)

        if embedding is None:
            print(f"[EMBED WARNING] Embedding failed for: {filename}")
            continue

        # Save embedding to cache
        utils.update_embedding_cache(filepath, embedding)

        created_notes_info.append({
            "filepath": filepath,
            "embedding": embedding.tolist(),
            "title": title
        })

    return created_notes_info
