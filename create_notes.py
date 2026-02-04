# create_notes.py - Create Obsidian Notes from Segmented Transcripts

import os
import re
import datetime
import config
import utils


def sanitize_filename(title: str) -> str:
    """Convert a title to a filesystem-safe filename stem."""
    s = title.lower()
    s = re.sub(r"[^a-z0-9\- ]+", "", s)
    s = s.replace(" ", "-").strip("-")
    s = re.sub(r'-+', '-', s)
    return s[:100] if s else "untitled"


def format_yaml_frontmatter(
    title: str,
    emoji: str,
    recording_id: str,
    transcript_type_tag: str,
    manual_tags: str = ""
) -> str:
    """Build YAML frontmatter for a note."""
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")

    # Escape title for YAML
    safe_title = title.replace('"', '\\"')

    yaml_lines = [
        "---",
        f'title: "{safe_title}"',
        f'emoji: "{emoji}"',
        f"date: {date_str}",
        f"type: {transcript_type_tag}",
        f"recording: {recording_id}",
    ]

    # Add manual tags if provided
    if manual_tags:
        tags = [t.strip() for t in manual_tags.split(",") if t.strip()]
        if tags:
            yaml_lines.append("tags:")
            for t in tags:
                yaml_lines.append(f"  - {t}")

    # Placeholder for semantic linking
    yaml_lines.append("semantic-tags: []")
    yaml_lines.append("---")
    yaml_lines.append("")

    return "\n".join(yaml_lines)


def create_notes_from_segments(
    segmented_data: list[dict],
    vault_root: str,
    subdir: str,
    recording_id: str,
    transcript_type_tag: str,
    manual_tags_str: str = ""
) -> list[dict]:
    """
    Creates Obsidian markdown notes for each segment.
    Generates embeddings and returns metadata for linking.

    Returns:
        List of dicts with filepath, embedding, title, and filename_stem.
    """
    output_dir = os.path.join(vault_root, subdir) if subdir else vault_root
    os.makedirs(output_dir, exist_ok=True)

    created_notes_info = []

    for i, seg in enumerate(segmented_data):
        title = seg.get("title", f"Segment {i + 1}")
        emoji = seg.get("emoji", "")
        content = seg.get("content", "")

        if not content.strip():
            print(f"[NOTE] Skipping empty segment: {title}")
            continue

        filename_stem = f"{recording_id}_{sanitize_filename(title)}"
        filename = f"{filename_stem}.md"
        filepath = os.path.join(output_dir, filename)

        # Avoid overwriting existing files
        counter = 1
        while os.path.exists(filepath):
            filename_stem = f"{recording_id}_{sanitize_filename(title)}-{counter}"
            filename = f"{filename_stem}.md"
            filepath = os.path.join(output_dir, filename)
            counter += 1

        # Build note content
        frontmatter = format_yaml_frontmatter(
            title, emoji, recording_id, transcript_type_tag, manual_tags_str
        )
        full_note = f"{frontmatter}\n{content}\n"

        # Write note to disk
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(full_note)
            print(f"[NOTE CREATED] {os.path.basename(filepath)}")
        except Exception as e:
            print(f"[NOTE ERROR] Could not write {filepath}: {e}")
            continue

        # Generate embedding
        print(f"[EMBED] Generating embedding for: {os.path.basename(filepath)}")
        embedding = utils.get_embedding(full_note)

        if embedding is None:
            print(f"[EMBED WARNING] Embedding failed for: {filename}")
            continue

        # Update cache
        utils.update_embedding_cache(filepath, embedding)

        created_notes_info.append({
            "filepath": filepath,
            "embedding": embedding.tolist(),
            "title": title,
            "filename_stem": filename_stem,
        })

    return created_notes_info


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Create Obsidian notes from segmented JSON.")
    parser.add_argument("segments_file", help="Path to the segmented JSON file.")
    parser.add_argument("--recording-id", required=True, help="Recording ID (e.g., 11222501)")
    parser.add_argument("--type", default="type/note", help="Type tag for notes")
    parser.add_argument("--tags", default="", help="Comma-separated manual tags")
    args = parser.parse_args()

    with open(args.segments_file, 'r', encoding='utf-8') as f:
        segments = json.load(f)

    notes = create_notes_from_segments(
        segments,
        config.OBSIDIAN_VAULT_ROOT,
        config.NOTES_SUBDIRECTORY_IN_VAULT,
        args.recording_id,
        args.type,
        args.tags
    )

    print(f"\nCreated {len(notes)} notes.")
