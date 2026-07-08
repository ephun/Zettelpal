# find_near_duplicates.py

import os
import sys
import argparse
import numpy as np
import datetime
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zettelpal import config, models
from zettelpal.log import setup_console_logging
from zettelpal.vault import notes

def find_near_duplicates(vault_root, similarity_threshold):
    """
    Finds and prompts to delete near-duplicate Markdown notes based on content similarity.

    Args:
        vault_root (str): The root path to the Obsidian vault.
        similarity_threshold (float): The cosine similarity score to consider notes as
                                      near-duplicates.
    """
    if not os.path.exists(vault_root):
        print(f"Error: Vault root not found at {vault_root}.")
        return

    print("--- Finding Near-Duplicate Notes ---")
    print(f"Scanning vault: {vault_root}")
    print(f"Using similarity threshold: {similarity_threshold}")
    
    # Use the same utility function from your pipeline to get all notes
    all_md_files = notes.find_all_markdown_files(vault_root)

    if len(all_md_files) < 2:
        print("Not enough notes to compare. Exiting.")
        return

    notes_data = []
    print(f"Found {len(all_md_files)} notes. Generating embeddings...")
    
    embedding_model = models.load_embedding_model()
    if embedding_model is None:
        print("Error: Embedding model could not be loaded. Cannot proceed.")
        return
    
    for i, filepath in enumerate(all_md_files):
        # Read file content and extract metadata
        frontmatter, body_content, _ = notes.extract_frontmatter_and_body(filepath)
        created_str = notes.extract_created_timestamp_str_from_frontmatter(frontmatter)
        created_dt = notes.parse_created(created_str) or datetime.datetime.fromtimestamp(os.path.getmtime(filepath))

        embedding = models.get_embedding(body_content, embedding_model)
        
        if embedding is not None:
            notes_data.append({
                "filepath": filepath,
                "content": body_content,
                "created": created_dt,
                "embedding": embedding
            })
        
        if (i + 1) % 50 == 0 or (i + 1) == len(all_md_files):
            print(f"  Processed {i + 1}/{len(all_md_files)} notes.")
    
    if len(notes_data) < 2:
        print("Not enough notes with valid content to compare. Exiting.")
        return

    print("\nCalculating content similarity...")
    embeddings = np.array([note["embedding"] for note in notes_data])
    
    # Calculate pairwise cosine similarity
    similarity_matrix = cosine_similarity(embeddings)
    
    near_duplicates = []
    processed_pairs = set()

    for i in range(len(notes_data)):
        for j in range(i + 1, len(notes_data)):
            if i == j:
                continue

            similarity = similarity_matrix[i, j]
            
            if similarity >= similarity_threshold:
                # Store the pair with their similarity and file info
                pair = tuple(sorted((notes_data[i]["filepath"], notes_data[j]["filepath"])))
                if pair not in processed_pairs:
                    near_duplicates.append((notes_data[i], notes_data[j], similarity))
                    processed_pairs.add(pair)
    
    if not near_duplicates:
        print("No near-duplicate notes found above the threshold.")
        return
        
    print(f"\nFound {len(near_duplicates)} potential near-duplicate pairs.")
    
    # Sort by similarity score, descending
    near_duplicates.sort(key=lambda x: x[2], reverse=True)

    for i, (note1, note2, similarity) in enumerate(near_duplicates):
        print("\n" + "="*50)
        print(f"PAIR {i+1}/{len(near_duplicates)} (Similarity: {similarity:.4f})")

        # Determine which note is older
        if note1["created"] < note2["created"]:
            older_note = note1
            newer_note = note2
        else:
            older_note = note2
            newer_note = note1

        print(f"\nOlder Note ({os.path.basename(older_note['filepath'])}):")
        print(f"  Created: {older_note['created'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Content (first 200 chars):\n  {older_note['content'][:200].strip()}...")
        
        print(f"\nNewer Note ({os.path.basename(newer_note['filepath'])}):")
        print(f"  Created: {newer_note['created'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Content (first 200 chars):\n  {newer_note['content'][:200].strip()}...")

        user_input = input("\nWould you like to delete the OLDER note? (yes/no/quit): ").strip().lower()

        if user_input == 'yes':
            try:
                os.remove(older_note['filepath'])
                print(f"  ✅ Deleted: {os.path.basename(older_note['filepath'])}")
            except OSError as e:
                print(f"  ❌ Error deleting file: {e}")
        elif user_input == 'quit':
            print("Exiting cleanup script.")
            return
        else:
            print("  Skipped deletion.")
            
    print("\n--- Cleanup process complete. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find and manage near-duplicate notes in an Obsidian vault.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Cosine similarity threshold for a near-duplicate match (default: 0.95). Lower for fuzzier matches."
    )
    args = parser.parse_args()

    setup_console_logging()

    # Use the vault root from your config file
    find_near_duplicates(config.settings.vault_root, args.threshold)