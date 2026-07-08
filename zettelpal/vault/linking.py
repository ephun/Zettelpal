# linking.py - Semantic and Chronological Note Linking

import json
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from zettelpal import config, models
from zettelpal.vault import cache as vault_cache
from zettelpal.vault import notes


def update_and_load_vault_embeddings(
    vault_root: str,
    cache_filepath: str,
    new_notes_info: list[dict] = None
) -> tuple[list[dict], bool]:
    """
    Loads embeddings from cache, updates for new/modified/deleted notes.

    Returns:
        tuple: (list of valid notes with embeddings, whether any updates occurred)
    """
    print(f"Loading vault embeddings from: {cache_filepath}")
    cache = vault_cache.load_embedding_cache()
    updated_cache = cache.copy()
    update_needed = False

    new_note_paths = {n.get("filepath") for n in new_notes_info or []}
    any_updates = bool(new_note_paths)

    print(f"Scanning vault: {vault_root}")
    all_md_files = notes.find_all_markdown_files(vault_root)
    found_relative_paths = {os.path.relpath(f, vault_root) for f in all_md_files}

    print(f"Found {len(all_md_files)} markdown files.")

    current_notes = []

    for filepath in all_md_files:
        relative_path = os.path.relpath(filepath, vault_root)

        # Handle newly created notes
        if filepath in new_note_paths:
            new_note = next(n for n in new_notes_info if n["filepath"] == filepath)
            frontmatter, body, _ = notes.extract_frontmatter_and_body(filepath)

            source = notes.extract_source_from_frontmatter(frontmatter)
            created_str = notes.extract_created_timestamp_str_from_frontmatter(frontmatter)
            created_dt = notes.parse_created(created_str)

            embedding = np.array(new_note["embedding"])

            current_notes.append({
                "filepath": filepath,
                "display_title": new_note["title"],
                "filename_stem": new_note["filename_stem"],
                "source": source,
                "created": created_dt,
                "embedding": embedding,
            })

            updated_cache[relative_path] = {
                "display_title": new_note["title"],
                "filename_stem": new_note["filename_stem"],
                "source": source,
                "created": created_str,
                "mtime": os.path.getmtime(filepath),
                "embedding": new_note["embedding"],
            }
            update_needed = True
            continue

        # Handle existing notes
        try:
            mtime = os.path.getmtime(filepath)
        except FileNotFoundError:
            if relative_path in updated_cache:
                del updated_cache[relative_path]
                update_needed = True
            continue

        frontmatter, body, _ = notes.extract_frontmatter_and_body(filepath)

        filename_stem = os.path.splitext(os.path.basename(filepath))[0]
        title = notes.extract_title_from_filepath(filepath)
        source = notes.extract_source_from_frontmatter(frontmatter)
        created_str = notes.extract_created_timestamp_str_from_frontmatter(frontmatter)
        created_dt = notes.parse_created(created_str)

        cache_entry = cache.get(relative_path)
        is_cached = cache_entry is not None
        cache_current = (
            is_cached and
            cache_entry.get("mtime") == mtime and
            cache_entry.get("source") == source
        )

        should_recalc = not cache_current or cache_entry.get("embedding") is None

        embedding = None
        if not should_recalc and cache_entry:
            cached_emb = cache_entry.get("embedding")
            if cached_emb is not None:
                try:
                    embedding = np.array(cached_emb)
                except (TypeError, ValueError):
                    should_recalc = True

        if should_recalc:
            update_needed = True
            any_updates = True

            model = models.load_embedding_model()
            if model is not None and body and body.strip():
                embedding = models.get_embedding(body, model)

            updated_cache[relative_path] = {
                "display_title": title,
                "filename_stem": filename_stem,
                "source": source,
                "created": created_str,
                "mtime": mtime,
                "embedding": embedding.tolist() if embedding is not None else None,
            }

        current_notes.append({
            "filepath": filepath,
            "display_title": title,
            "filename_stem": filename_stem,
            "source": source,
            "created": created_dt,
            "embedding": embedding,
        })

    # Remove deleted files from cache
    for relative_path in list(updated_cache.keys()):
        if relative_path not in found_relative_paths:
            print(f"Removing deleted note from cache: {relative_path}")
            del updated_cache[relative_path]
            update_needed = True
            any_updates = True

    # Save cache if changed
    if update_needed:
        print("Saving updated cache...")
        vault_cache.save_embedding_cache(updated_cache)
    else:
        print("Cache is up-to-date.")

    # Filter notes that have valid embeddings
    valid_notes = [n for n in current_notes if n.get("embedding") is not None]

    return valid_notes, any_updates


def update_all_notes_links(
    all_notes: list[dict],
    threshold: float,
    last_threshold: float | None = None,
    any_note_was_updated_or_created: bool = False
):
    """
    Updates chronological and semantic links for notes.
    """
    print(f"Updating links (threshold: {threshold:.2f})...")
    block_start = getattr(config, "LINK_BLOCK_START", "<!-- zettelpal-links:start -->")
    block_end = getattr(config, "LINK_BLOCK_END", "<!-- zettelpal-links:end -->")
    max_semantic_links = getattr(config, "MAX_SEMANTIC_LINKS_PER_NOTE", 10)

    # Group notes by source for chronological linking
    notes_by_source = defaultdict(list)
    for note in all_notes:
        source = note.get("source")
        if source and note.get("created"):
            notes_by_source[source].append(note)

    # Sort notes within each source group
    source_sorted = {}
    filepath_to_index = {}
    for source, notes_list in notes_by_source.items():
        sorted_list = sorted(notes_list, key=lambda x: (x["created"], x["filename_stem"]))
        source_sorted[source] = sorted_list
        for i, note in enumerate(sorted_list):
            filepath_to_index[note["filepath"]] = (source, i)

    # Determine if full update needed
    force_update = False
    if last_threshold is None or abs(threshold - last_threshold) > 1e-6:
        print("Threshold changed. Full recalculation.")
        force_update = True
    elif any_note_was_updated_or_created:
        print("Notes changed. Full recalculation.")
        force_update = True

    notes_to_update = all_notes if force_update else []

    if not notes_to_update:
        print("No link updates needed.")
        return

    print(f"Updating links for {len(notes_to_update)} notes...")

    # Build similarity matrix
    notes_with_embeddings = [n for n in all_notes if n["embedding"] is not None]
    if not notes_with_embeddings:
        print("No notes with embeddings.")
        return

    print(f"[LINK DEBUG] {len(notes_with_embeddings)} notes have embeddings")
    print(f"[LINK DEBUG] Computing {len(notes_with_embeddings)}x{len(notes_with_embeddings)} similarity matrix...")

    all_embeddings = np.array([n["embedding"] for n in notes_with_embeddings])
    emb_filepath_to_idx = {n["filepath"]: i for i, n in enumerate(notes_with_embeddings)}

    sim_matrix = np.full((len(all_embeddings), len(all_embeddings)), -1.0)
    if len(all_embeddings) > 1:
        sim_matrix = cosine_similarity(all_embeddings)
        # Show similarity distribution
        upper_tri = sim_matrix[np.triu_indices(len(sim_matrix), k=1)]
        if len(upper_tri) > 0:
            print(f"[LINK DEBUG] Similarity stats: min={upper_tri.min():.3f}, max={upper_tri.max():.3f}, mean={upper_tri.mean():.3f}")
            above_threshold = np.sum(upper_tri >= threshold)
            print(f"[LINK DEBUG] {above_threshold} pairs above threshold {threshold:.2f}")

    # Process each note
    for note in notes_to_update:
        filepath = note["filepath"]

        frontmatter, body, _ = notes.extract_frontmatter_and_body(filepath)

        # Build fixed content (frontmatter + body)
        parts = []
        parts.extend(frontmatter)
        if frontmatter and frontmatter[-1].strip() == "---":
            parts.append("\n")
        if body:
            parts.append(body.rstrip())

        # Ensure blank line before links
        if parts:
            last = parts[-1]
            if last.endswith('\n'):
                parts.append('\n')
            elif not last.endswith('\n\n'):
                parts.append('\n\n')

        parts.append(f"{block_start}\n")

        # Get chronological neighbors
        chrono_links = []
        prev_note = None
        next_note = None

        source_info = filepath_to_index.get(filepath)
        if source_info:
            source_id, idx = source_info
            source_list = source_sorted[source_id]
            if len(source_list) > 1:
                if idx > 0:
                    prev_note = source_list[idx - 1]
                    chrono_links.append(
                        config.CHRONOLOGICAL_LINK_TEMPLATE.format(
                            link_target_filename_stem=prev_note["filename_stem"],
                            display_title=prev_note["display_title"]
                        )
                    )
                if idx < len(source_list) - 1:
                    next_note = source_list[idx + 1]
                    chrono_links.append(
                        config.CHRONOLOGICAL_LINK_TEMPLATE.format(
                            link_target_filename_stem=next_note["filename_stem"],
                            display_title=next_note["display_title"]
                        )
                    )

        # Write chronological links
        if chrono_links:
            for link in chrono_links:
                parts.append(f"{link}\n")
            parts.append("\n")

        # Get semantic links
        semantic_links = []
        current_emb = note.get("embedding")
        if current_emb is not None and filepath in emb_filepath_to_idx:
            current_idx = emb_filepath_to_idx[filepath]
            current_stem = note.get("filename_stem")

            for other_note in all_notes:
                if other_note["filepath"] == filepath:
                    continue
                if current_stem and other_note.get("filename_stem") == current_stem:
                    continue

                other_filepath = other_note["filepath"]
                other_emb = other_note.get("embedding")

                # Skip chronological neighbors
                is_chrono_neighbor = False
                if prev_note and other_filepath == prev_note["filepath"]:
                    is_chrono_neighbor = True
                if next_note and other_filepath == next_note["filepath"]:
                    is_chrono_neighbor = True
                if is_chrono_neighbor:
                    continue

                if other_emb is not None and other_filepath in emb_filepath_to_idx:
                    other_idx = emb_filepath_to_idx[other_filepath]
                    similarity = sim_matrix[current_idx, other_idx]

                    if similarity >= threshold:
                        semantic_links.append({
                            "display_title": other_note["display_title"],
                            "filename_stem": other_note["filename_stem"],
                            "similarity": similarity
                        })

            # Sort by similarity
            semantic_links.sort(key=lambda x: x["similarity"], reverse=True)
            if max_semantic_links is not None and max_semantic_links >= 0:
                semantic_links = semantic_links[:max_semantic_links]

        # Debug output
        note_name = os.path.basename(filepath)
        if semantic_links:
            print(f"[LINK] {note_name}: {len(semantic_links)} semantic links found")
            # Show top 3 matches
            for link_info in semantic_links[:3]:
                print(f"       -> {link_info['display_title'][:40]:<40} (sim: {link_info['similarity']:.3f})")
            if len(semantic_links) > 3:
                print(f"       ... and {len(semantic_links) - 3} more")
        else:
            print(f"[LINK] {note_name}: no semantic links above threshold {threshold:.2f}")

        # Write semantic links
        if semantic_links:
            for link_info in semantic_links:
                link = config.SEMANTIC_LINK_TEMPLATE.format(
                    link_target_filename_stem=link_info["filename_stem"],
                    display_title=link_info["display_title"]
                )
                parts.append(f"{link}\n")

        parts.append(f"{block_end}\n")

        # Write file
        try:
            final = "".join(parts).rstrip()
            if not final.endswith('\n'):
                final += '\n'

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(final)
        except OSError as e:
            print(f"Error writing {os.path.basename(filepath)}: {e}")


def run_linking_process(threshold: float, new_notes_info: list[dict] = None) -> bool:
    """
    Main function to run the linking process.
    """
    print(f"\n--- Linking Process (threshold: {threshold:.2f}) ---")

    if not os.path.exists(config.OBSIDIAN_VAULT_ROOT):
        print(f"ERROR: Vault not found: {config.OBSIDIAN_VAULT_ROOT}")
        return False

    try:
        # Load last threshold
        threshold_file = os.path.join(
            config.OBSIDIAN_VAULT_ROOT, ".obsidian", "zettelpal_last_threshold.json"
        )
        last_threshold = None
        if os.path.exists(threshold_file):
            try:
                with open(threshold_file, "r") as f:
                    last_threshold = json.load(f).get("threshold")
            except (OSError, json.JSONDecodeError):
                pass

        # Update cache and load embeddings
        all_notes, any_updates = update_and_load_vault_embeddings(
            os.path.abspath(config.OBSIDIAN_VAULT_ROOT),
            config.EMBEDDINGS_CACHE_FILE,
            new_notes_info
        )

        if not all_notes:
            print("No notes found for linking.")
            return True

        # Update links
        update_all_notes_links(
            all_notes,
            threshold,
            last_threshold,
            any_updates
        )

        # Save threshold
        try:
            with open(threshold_file, "w", encoding="utf-8") as f:
                json.dump({"threshold": threshold}, f)
        except OSError as e:
            print(f"Warning: Could not save threshold: {e}")

        print("--- Linking Complete ---")
        return True

    except Exception as e:
        print(f"ERROR: Linking failed: {e}")
        import traceback
        traceback.print_exc()
        return False
