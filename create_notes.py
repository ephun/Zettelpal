# create_notes.py - Create Obsidian Notes from Segmented Transcripts

import os
import re
import datetime
import config
import utils


def sanitize_filename(title: str) -> str:
    """Convert a title to a filesystem-safe filename stem, preserving case."""
    # Remove characters that are invalid in filenames, but keep letters, numbers, spaces, hyphens
    s = re.sub(r'[<>:"/\\|?*]', '', title)
    # Replace spaces with hyphens
    s = s.replace(" ", "-")
    # Collapse multiple hyphens
    s = re.sub(r'-+', '-', s)
    # Strip leading/trailing hyphens
    s = s.strip("-")
    return s[:100] if s else "Untitled"


def generate_tags_for_content(content: str) -> list[str]:
    """
    Uses the local LLM to generate semantic tags for content.
    Returns a list of tags or empty list on failure.
    """
    if not content or not content.strip():
        return []

    prompt = config.GRANULAR_TAGGING_PROMPT_TEMPLATE.format(text_content=content[:3000])

    print("[TAGGING] Generating semantic tags...")
    response = utils.llm_chat(prompt, temperature=0.3, max_tokens=500)

    if not response:
        print("[TAGGING] No response from LLM")
        return []

    # Try to parse as JSON list
    parsed = utils.extract_json_from_text(response)
    if parsed and isinstance(parsed, list):
        # Filter to only valid string tags
        tags = [t for t in parsed if isinstance(t, str) and t.strip()]
        print(f"[TAGGING] Generated {len(tags)} tags")
        return tags

    print(f"[TAGGING] Could not parse tags from: {response[:100]}")
    return []


def get_placement(index: int, total: int) -> str:
    """Determine placement string based on position in sequence."""
    if total == 1:
        return "only"
    elif index == 0:
        return "first"
    elif index == total - 1:
        return "last"
    else:
        return "middle"


def format_yaml_frontmatter(
    title: str,
    emoji: str,
    recording_id: str,
    transcript_type_tag: str,
    placement: str,
    semantic_tags: list[str],
    manual_tags: str = "",
    note_id: str = None,
    created_time: datetime.datetime = None,
    audio_path: str = None
) -> str:
    """Build YAML frontmatter matching the original Zettelpal format."""
    if created_time is None:
        created_time = datetime.datetime.now()
    if note_id is None:
        note_id = created_time.strftime("%Y%m%d%H%M%S")
    created_iso = created_time.isoformat()

    # Escape title for YAML
    safe_title = title.replace('"', '\\"')

    yaml_lines = [
        "---",
        f"id: {note_id}",
        f'title: "{safe_title}"',
        f"created: {created_iso}",
        f'emoji: "{emoji}"',
        f'source: "{recording_id}"',
    ]

    if audio_path:
        safe_audio = audio_path.replace("\\", "/")
        yaml_lines.append(f'audio: "{safe_audio}"')

    yaml_lines.extend([
        f'placement: "{placement}"',
        "tags:",
    ])

    # Collect all tags
    all_tags = []

    # Add semantic tags from LLM
    all_tags.extend(semantic_tags)

    # Add manual tags
    if manual_tags:
        for t in manual_tags.split(","):
            t = t.strip()
            if t and t not in all_tags:
                all_tags.append(t)

    # Add the type tag
    if transcript_type_tag and transcript_type_tag not in all_tags:
        all_tags.append(transcript_type_tag)

    # Always add zettelpal tag
    if "zettelpal" not in all_tags:
        all_tags.append("zettelpal")

    # Write tags
    for tag in all_tags:
        yaml_lines.append(f"  - {tag}")

    yaml_lines.append("---")
    yaml_lines.append("")

    return "\n".join(yaml_lines)


def create_notes_from_segments(
    segmented_data: list[dict],
    vault_root: str,
    subdir: str,
    recording_id: str,
    transcript_type_tag: str,
    manual_tags_str: str = "",
    clip_paths: list[str | None] = None
) -> list[dict]:
    """
    Creates Obsidian markdown notes for each segment.
    Generates embeddings and returns metadata for linking.

    Args:
        clip_paths: Optional list parallel to segmented_data with vault-relative
                    audio clip paths (or None for segments without clips).

    Returns:
        List of dicts with filepath, embedding, title, and filename_stem.
    """
    output_dir = os.path.join(vault_root, subdir) if subdir else vault_root
    os.makedirs(output_dir, exist_ok=True)

    created_notes_info = []
    total_segments = len(segmented_data)

    for i, seg in enumerate(segmented_data):
        title = seg.get("title", f"Segment {i + 1}")
        emoji = seg.get("emoji", "")
        content = seg.get("content", "")

        if not content.strip():
            print(f"[NOTE] Skipping empty segment: {title}")
            continue

        # Generate timestamp for this note (matches id in frontmatter)
        now = datetime.datetime.now()
        note_id = now.strftime("%Y%m%d%H%M%S")

        # Filename format: "Title-With-Dashes - YYYYMMDDHHMMSS.md"
        sanitized_title = sanitize_filename(title)
        filename_stem = f"{sanitized_title} - {note_id}"
        filename = f"{filename_stem}.md"
        filepath = os.path.join(output_dir, filename)

        # Avoid overwriting existing files
        counter = 1
        while os.path.exists(filepath):
            filename_stem = f"{sanitized_title} - {note_id}-{counter}"
            filename = f"{filename_stem}.md"
            filepath = os.path.join(output_dir, filename)
            counter += 1

        # Generate semantic tags for this content
        semantic_tags = generate_tags_for_content(content)

        # Determine placement
        placement = get_placement(i, total_segments)

        # Get audio clip path for this segment (if available)
        audio_clip = None
        if clip_paths and i < len(clip_paths) and clip_paths[i]:
            # Convert to vault-relative path
            clip_abs = os.path.abspath(clip_paths[i])
            vault_dir = os.path.normpath(vault_root)
            if not vault_dir.endswith(os.sep):
                vault_dir += os.sep
            vault_abs = os.path.abspath(vault_dir)
            is_in_vault = os.path.normcase(clip_abs).startswith(os.path.normcase(vault_abs))

            if is_in_vault:
                audio_clip = os.path.relpath(clip_abs, vault_abs)
            else:
                audio_clip = clip_paths[i]

        # Build frontmatter
        frontmatter = format_yaml_frontmatter(
            title, emoji, recording_id, transcript_type_tag,
            placement, semantic_tags, manual_tags_str,
            note_id=note_id, created_time=now, audio_path=audio_clip
        )

        # Build note with H1 heading
        full_note = f"{frontmatter}# {title}\n\n{content}\n"

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
