# link_notes.py

import json
import os
import sys
import time
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity # Requires pip install scikit-learn
from collections import defaultdict # Added for grouping notes
import datetime # Import datetime for sorting and parsing timestamps

import utils # Imports local embedding model loader, embedding get, cache functions, find_files, extract_title, extract_frontmatter_and_body, extract_source_from_frontmatter, extract_created_timestamp_str_from_frontmatter, parse_created_timestamp_str
# Load config variables
from config import (
    OBSIDIAN_VAULT_ROOT, EMBEDDINGS_CACHE_FILE, SIMILARITY_THRESHOLD,
    LINK_SEPARATOR, SEMANTIC_LINK_TEMPLATE, CHRONOLOGICAL_LINK_TEMPLATE, SOURCE_METADATA_KEY
)


def update_and_load_vault_embeddings(vault_root: str, cache_filepath: str, new_notes_info: list[dict] = None) -> list[dict]:
    """
    Loads embeddings from cache, checks for new/modified/deleted notes in the vault,
    calculates new embeddings if needed, updates the cache, saves the cache,
    and returns a list of dictionaries for ALL valid notes in the vault
    (including filepath, embedding_vector, display_title, filename_stem, source, and created timestamp).
    This function now accepts a list of new notes to integrate more efficiently.
    
    Returns:
        tuple: (list of valid notes, boolean indicating if any embedding updates occurred)
    """
    print(f"Loading and updating vault embedding cache: {cache_filepath}...")
    cache = utils.load_embedding_cache()
    updated_cache = cache.copy() # Work on a copy to track changes before saving
    update_needed_in_cache = False # Flag for cache save
    
    new_note_paths = {n.get("filepath") for n in new_notes_info or []}
    
    # We will track if ANY notes in the vault had their embedding or metadata updated/created.
    # This is the flag needed for the linking stage.
    any_note_was_updated_or_created = bool(new_note_paths)

    print(f"Scanning Obsidian vault at {vault_root} for markdown files...")
    all_md_files = utils.find_all_markdown_files(vault_root)
    found_relative_paths = {os.path.relpath(f, vault_root) for f in all_md_files}
    
    current_vault_notes_data = []

    print(f"Found {len(all_md_files)} markdown files. Checking cache status...")

    # First pass: Check cache status for each file and gather data
    for count, filepath in enumerate(all_md_files):
        if (count + 1) % 100 == 0 or (count + 1) == len(all_md_files):
             print(f"  Processing cache status for {count + 1}/{len(all_md_files)} files...")

        relative_filepath = os.path.relpath(filepath, vault_root)
        
        # --- Handle Newly Created Notes (from pipeline) ---
        if filepath in new_note_paths:
             new_note_data = next(n for n in new_notes_info if n["filepath"] == filepath)
             
             # Extract metadata directly from file for integrity
             frontmatter_lines, body_content, _ = utils.extract_frontmatter_and_body(filepath)
             note_source = utils.extract_source_from_frontmatter(frontmatter_lines)
             note_created_timestamp_str = utils.extract_created_timestamp_str_from_frontmatter(frontmatter_lines)
             note_created_timestamp_dt = utils.parse_created_timestamp_str(note_created_timestamp_str)
             
             # Use the pre-calculated embedding (already done in create_notes)
             embedding_vector = np.array(new_note_data["embedding"])
             
             current_vault_notes_data.append({
                 "filepath": filepath,
                 "display_title": new_note_data["title"],
                 "filename_stem": new_note_data["filename_stem"],
                 "source": note_source,
                 "created": note_created_timestamp_dt,
                 "embedding": embedding_vector,
             })
             
             # Update the cache with this new information
             updated_cache[relative_filepath] = {
                 "display_title": new_note_data["title"],
                 "filename_stem": new_note_data["filename_stem"],
                 "source": note_source,
                 "created": note_created_timestamp_str,
                 "mtime": os.path.getmtime(filepath),
                 "embedding": new_note_data["embedding"],
             }
             update_needed_in_cache = True
             continue 
        # ----------------------------------------------------
             
        try:
             mtime = os.path.getmtime(filepath)
        except FileNotFoundError:
             # Handle files deleted externally
             if relative_filepath in updated_cache:
                  del updated_cache[relative_filepath]
                  update_needed_in_cache = True
             continue 

        # Read the file only once here to get frontmatter and body
        frontmatter_lines, body_content, _ = utils.extract_frontmatter_and_body(filepath)

        # Extract relevant metadata
        filename_stem = os.path.splitext(os.path.basename(filepath))[0]
        note_display_title = utils.extract_title_from_filepath(filepath)
        note_source = utils.extract_source_from_frontmatter(frontmatter_lines)
        note_created_timestamp_str = utils.extract_created_timestamp_str_from_frontmatter(frontmatter_lines)
        note_created_timestamp_dt = utils.parse_created_timestamp_str(note_created_timestamp_str)

        cache_entry = cache.get(relative_filepath)
        is_cached = cache_entry is not None
        
        # Check if cache entry is fully up-to-date (mtime, source string, created string)
        cache_is_current = is_cached and \
                           cache_entry.get("mtime") == mtime and \
                           cache_entry.get("source") == note_source and \
                           cache_entry.get("created") == note_created_timestamp_str
        
        should_recalculate = not cache_is_current or cache_entry.get("embedding") is None

        embedding_vector = None
        if not should_recalculate:
            # Load embedding from cache if available and up-to-date
            cached_embedding_list = cache_entry.get("embedding")
            if cached_embedding_list is not None:
                try:
                    embedding_vector = np.array(cached_embedding_list)
                except Exception as e:
                     print(f"  Error converting cached embedding to numpy for {os.path.basename(filepath)}: {e}. Marking for recalculation.")
                     should_recalculate = True

        if should_recalculate:
            update_needed_in_cache = True
            any_note_was_updated_or_created = True # Set flag for linking stage
            
            # Recalculate embedding for this note
            embedding_model = utils.load_embedding_model()
            if embedding_model is not None and body_content and body_content.strip():
                 embedding_vector = utils.get_embedding(body_content, embedding_model)
            else:
                 embedding_vector = None 

            # Update the cache entry with the new embedding (or None) and latest metadata
            updated_cache[relative_filepath] = {
                 "display_title": note_display_title,
                 "filename_stem": filename_stem,
                 "source": note_source,
                 "created": note_created_timestamp_str, # Store as string in cache
                 "mtime": mtime,
                 "embedding": embedding_vector.tolist() if embedding_vector is not None else None
            }
        
        # Add this note's data to the main list
        current_vault_notes_data.append({
             "filepath": filepath,
             "display_title": note_display_title,
             "filename_stem": filename_stem,
             "source": note_source,
             "created": note_created_timestamp_dt, # Store as datetime object
             "embedding": embedding_vector, # numpy array or None
        })

    # Remove deleted files from cache
    cached_relative_paths = list(updated_cache.keys())
    for relative_filepath in cached_relative_paths:
        if relative_filepath not in found_relative_paths:
            print(f"  Removing deleted note from cache: {relative_filepath}")
            del updated_cache[relative_filepath]
            update_needed_in_cache = True
            any_note_was_updated_or_created = True # A deletion is a change that affects links

    # Save cache if anything changed
    if update_needed_in_cache:
        print(f"\nSaving updated embeddings cache to {cache_filepath}...")
        utils.save_embedding_cache(updated_cache)
    else:
        print("\nEmbeddings cache is up-to-date.")

    valid_notes_for_linking = [
        n for n in current_vault_notes_data
        if not (n.get("source") and n.get("created") is None)
    ]
    
    return valid_notes_for_linking, any_note_was_updated_or_created


def update_all_notes_links(
    all_vault_notes_data: list[dict],
    similarity_threshold: float,
    last_known_threshold: float | None = None,
    any_note_was_updated_or_created: bool = False
):
    """
    Calculates and updates chronological and semantic links for notes that need it.
    Notes are updated if:
    1. The similarity threshold has changed.
    2. Any note was created, modified, or deleted since the last run (guaranteeing integrity).
    """
    print(f"\nCalculating and updating links for notes that require it...")

    # Group notes by source for chronological linking
    notes_by_source = defaultdict(list)
    for note_data in all_vault_notes_data:
         source = note_data.get("source")
         created_dt = note_data.get("created")
         if source and isinstance(source, str) and source.strip() and created_dt is not None:
              notes_by_source[source].append(note_data)

    # Sort notes within each source group chronologically
    source_sorted_notes_map = {}
    source_filepath_to_index_map = {}
    for source, notes_list in notes_by_source.items():
         sorted_list = sorted(notes_list, key=lambda x: (x["created"], x["filename_stem"]))
         source_sorted_notes_map[source] = sorted_list
         for i, note_data in enumerate(sorted_list):
              source_filepath_to_index_map[note_data["filepath"]] = (source, i)


    # --- FINALIZED LOGIC FOR FORCING FULL UPDATE ---
    force_full_update = False 

    # 1. Check for threshold change (highest priority)
    if last_known_threshold is None or abs(similarity_threshold - last_known_threshold) > 1e-6:
        print("  Similarity threshold has changed. Forcing full link recalculation for all notes.")
        force_full_update = True
    
    # 2. Check for any change in vault notes (new/modified/deleted)
    elif any_note_was_updated_or_created:
        print("  Detected new/modified/deleted notes. Forcing full link recalculation to update backlinks/remove broken links.")
        force_full_update = True
    
    # --- Determine final list to update ---
    if force_full_update:
        notes_to_update = all_vault_notes_data
    else:
        # If no full update is necessary, we can skip the link writing process.
        notes_to_update = []
        
    # ----------------------------------------
        
    if not notes_to_update:
        print("No notes require a link update.")
        return

    print(f"Updating links for {len(notes_to_update)} notes...")
    
    # Pre-calculate all pairwise similarities to avoid redundant calculations
    embeddings_with_data = [n for n in all_vault_notes_data if n["embedding"] is not None]
    all_embeddings = np.array([n["embedding"] for n in embeddings_with_data])
    filepath_to_index = {n["filepath"]: i for i, n in enumerate(embeddings_with_data)}
    
    similarity_matrix = np.full((len(all_embeddings), len(all_embeddings)), -1.0)
    if len(all_embeddings) > 1:
        # Only run cosine similarity if there's more than one embedding to compare
        similarity_matrix = cosine_similarity(all_embeddings)

    # Process each note to update its links
    for i, current_note_data in enumerate(notes_to_update):
        current_filepath = current_note_data["filepath"]
        
        # --- Read the existing note content and separate ---
        frontmatter_lines, body_content_string, existing_link_lines = utils.extract_frontmatter_and_body(current_filepath)

        # Reconstruct the "fixed" part of the note (frontmatter + body content)
        fixed_content_parts = []
        fixed_content_parts.extend(frontmatter_lines)
        
        # Ensure a blank line after frontmatter (if it exists)
        if len(frontmatter_lines) > 1 and frontmatter_lines[-1].strip() == "---":
            fixed_content_parts.append("\n")
        
        if body_content_string:
            fixed_content_parts.append(body_content_string.rstrip())
        
        # Ensure there's a blank line (two newlines) before adding ANY links section
        if fixed_content_parts:
            # Check the last element's content, adding newlines if needed
            last_part = fixed_content_parts[-1]
            if last_part.endswith('\n'):
                 # If it ends with one newline, add another for a blank line
                 fixed_content_parts.append('\n')
            elif not last_part.endswith('\n\n'):
                 # If it doesn't end with any newline, add two for a blank line
                 fixed_content_parts.append('\n\n')

        # --- Generate new chronological links (within source group) ---
        chronological_links_to_add = []
        source_info = source_filepath_to_index_map.get(current_filepath)
        prev_note_data_chrono = None
        next_note_data_chrono = None
        if source_info:
            source_id, current_source_index = source_info
            source_list = source_sorted_notes_map[source_id]
            if len(source_list) > 1:
                 # Previous note
                 if current_source_index > 0:
                     prev_note_data_chrono = source_list[current_source_index - 1]
                     chronological_links_to_add.append(
                         CHRONOLOGICAL_LINK_TEMPLATE.format(
                             link_target_filename_stem=prev_note_data_chrono["filename_stem"],
                             display_title=prev_note_data_chrono["display_title"]
                         )
                     )
                 # Next note
                 if current_source_index < len(source_list) - 1:
                     next_note_data_chrono = source_list[current_source_index + 1]
                     chronological_links_to_add.append(
                         CHRONOLOGICAL_LINK_TEMPLATE.format(
                             link_target_filename_stem=next_note_data_chrono["filename_stem"],
                             display_title=next_note_data_chrono["display_title"]
                         )
                     )
        
        # FIX: Write each chronological link on a new line
        if chronological_links_to_add:
            for link in chronological_links_to_add:
                fixed_content_parts.append(f"{link}\n")
            
            # Add an extra blank line to separate chronological from semantic links
            fixed_content_parts.append("\n")


        # --- Generate new semantic links ---
        semantic_links = []
        current_embedding = current_note_data.get("embedding")
        if current_embedding is not None and current_filepath in filepath_to_index:
            current_index = filepath_to_index[current_filepath]
            for other_note_data in all_vault_notes_data:
                if current_filepath == other_note_data["filepath"]:
                    continue
                
                other_filepath = other_note_data["filepath"]
                other_display_title = other_note_data["display_title"]
                other_filename_stem = other_note_data["filename_stem"]
                other_embedding = other_note_data.get("embedding")
                
                # Skip chronological neighbors from the *same source* to prevent link redundancy
                is_chronological_neighbor_in_source = False
                if source_info and len(source_sorted_notes_map[source_id]) > 1:
                     if prev_note_data_chrono and other_filepath == prev_note_data_chrono["filepath"]:
                          is_chronological_neighbor_in_source = True
                     if next_note_data_chrono and other_filepath == next_note_data_chrono["filepath"]:
                          is_chronological_neighbor_in_source = True
                if is_chronological_neighbor_in_source:
                     continue
                
                if other_embedding is not None and other_filepath in filepath_to_index:
                    other_index = filepath_to_index[other_filepath]
                    similarity = similarity_matrix[current_index, other_index]
                    
                    if similarity >= similarity_threshold:
                        semantic_links.append({
                            "display_title": other_display_title,
                            "filename_stem": other_filename_stem,
                            "similarity": similarity
                        })

            semantic_links.sort(key=lambda x: x["similarity"], reverse=True)

        # FIX: Write each semantic link on a new line
        if semantic_links:
            for link_info in semantic_links:
                link_text = SEMANTIC_LINK_TEMPLATE.format(
                    link_target_filename_stem=link_info["filename_stem"],
                    display_title=link_info["display_title"]
                )
                fixed_content_parts.append(f"{link_text}\n")


        # Write the updated note file
        try:
            # We rewrite the entire file content, replacing the original links section entirely.
            final_content = "".join(fixed_content_parts).rstrip() # Final rstrip to remove trailing newlines
            
            # Ensure final content ends with exactly one newline (Obsidian standard)
            if not final_content.endswith('\n'):
                 final_content += '\n'

            with open(current_filepath, "w", encoding="utf-8") as f:
                f.write(final_content)
            print(f"  Updated links for: {os.path.basename(current_filepath)}")
        except Exception as e:
            print(f"Error writing updated links to {os.path.basename(current_filepath)}: {e}")


def run_linking_process(threshold: float, new_notes_info: list[dict] = None):
    """
    Main function to run the linking process, designed to be called by the GUI.
    """
    print(f"\n--- Initiating Linking Process with threshold: {threshold:.2f} ---")

    if not os.path.exists(OBSIDIAN_VAULT_ROOT):
        print(f"ERROR: Obsidian vault root not found at {OBSIDIAN_VAULT_ROOT}")
        print("Please check your config.py and ensure OBSIDIAN_VAULT_ROOT is correct.")
        return False
    
    try:
        # Load the last known threshold from the cache if available
        last_threshold_file = os.path.join(OBSIDIAN_VAULT_ROOT, ".obsidian", "zettelpal_last_threshold.json")
        last_known_threshold = None
        if os.path.exists(last_threshold_file):
            try:
                with open(last_threshold_file, "r") as f:
                    data = json.load(f)
                    last_known_threshold = data.get("threshold")
            except (json.JSONDecodeError, KeyError):
                pass
        
        # Step 1: Update cache and load ALL vault embeddings
        all_vault_notes_data_embeddings, any_note_was_updated_or_created = update_and_load_vault_embeddings(
            os.path.abspath(OBSIDIAN_VAULT_ROOT),
            EMBEDDINGS_CACHE_FILE,
            new_notes_info
        )

        if not all_vault_notes_data_embeddings:
            print("No valid notes found in the vault with sufficient metadata for linking. Skipping linking.")
            return True

        # Step 2: Update links for all notes that require it
        # Pass the flag indicating if any changes occurred in the vault (new/modified/deleted notes).
        update_all_notes_links(
            all_vault_notes_data_embeddings, 
            threshold, 
            last_known_threshold,
            any_note_was_updated_or_created 
        )

        # Save the new threshold to the file
        try:
            with open(last_threshold_file, "w", encoding="utf-8") as f: # Use utf-8 for safety
                json.dump({"threshold": threshold}, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not save last known threshold to {last_threshold_file}: {e}")

        print("\n--- Linking Process Completed ---")
        return True

    except Exception as e:
        print(f"An error occurred during the linking process: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Update semantic links in Obsidian notes.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=SIMILARITY_THRESHOLD,
        help=f"Cosine similarity threshold for linking notes (default: {SIMILARITY_THRESHOLD})."
    )
    args = parser.parse_args()

    success = run_linking_process(args.threshold)
    sys.exit(0 if success else 1)