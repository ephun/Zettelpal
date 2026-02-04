# zettelpal_pipeline.py
# This is the main script to run the Zettelpal pipeline.

import argparse
import os
import sys
import time
import json
import datetime
import re
import shutil

# --- Import functions from other files ---
import config
import utils
import transcribe
import segment
import create_notes
import link_notes
import classify
import gui  # Updated import for the GUI


# --- Ensure required directories exist upfront ---
def ensure_directories():
    """Creates necessary Zettelpal directories."""
    os.makedirs(config.ZETTELPAL_ROOT, exist_ok=True)
    os.makedirs(config.RAW_TRANSCRIPTS_INTERMEDIATE_DIR, exist_ok=True)
    os.makedirs(config.SEGMENTED_OUTPUT_INTERMEDIATE_DIR, exist_ok=True)
    os.makedirs(config.ARCHIVE_DIR, exist_ok=True)

    if not os.path.exists(config.OBSIDIAN_VAULT_ROOT):
        print(f"ERROR: Obsidian vault root not found at {config.OBSIDIAN_VAULT_ROOT}")
        print("Please check your config.py and ensure OBSIDIAN_VAULT_ROOT is correct.")
        sys.exit(1)

    notes_subdir_path = os.path.join(config.OBSIDIAN_VAULT_ROOT, config.NOTES_SUBDIRECTORY_IN_VAULT)
    os.makedirs(notes_subdir_path, exist_ok=True)

    cache_dir = os.path.dirname(config.EMBEDDINGS_CACHE_FILE)
    os.makedirs(cache_dir, exist_ok=True)

    print("All necessary directories checked/created.")


# --- Main Pipeline Function ---
def run_pipeline(audio_filepath: str):
    """
    Runs the full Zettelpal pipeline for a single audio file.
    """
    print("--- Zettelpal Pipeline Started ---")

    audio_basename = os.path.basename(audio_filepath)
    audio_filename_stem = os.path.splitext(audio_basename)[0]
    audio_extension = os.path.splitext(audio_basename)[1].lstrip(".")

    if not re.match(r"^[0-9]{6,8}$", audio_filename_stem):
        print(
            f"ERROR: Audio filename stem '{audio_filename_stem}' does not match expected MMDDYYNN "
            f"format (6 to 8 digits)."
        )
        print("Please rename your audio file.")
        sys.exit(1)
    recording_id = audio_filename_stem

    timestamp_full = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    raw_transcript_filepath_intermediate = os.path.join(
        config.RAW_TRANSCRIPTS_INTERMEDIATE_DIR, f"{recording_id}.txt"
    )
    segmented_filepath_intermediate = os.path.join(
        config.SEGMENTED_OUTPUT_INTERMEDIATE_DIR, f"{recording_id}_segmented.json"
    )

    archived_audio_filepath = os.path.join(
        config.ARCHIVE_DIR, f"{recording_id}_{timestamp_full}.{audio_extension}"
    )
    archived_raw_transcript_filepath = os.path.join(
        config.ARCHIVE_DIR, f"{recording_id}_{timestamp_full}.txt"
    )

    print(f"Processing Recording ID: {recording_id}")
    print(f"  Original audio:          {audio_filepath}")
    print(f"  Raw transcript (int):    {raw_transcript_filepath_intermediate}")
    print(f"  Segmented JSON (int):    {segmented_filepath_intermediate}")
    print(f"  Archived audio:          {archived_audio_filepath}")
    print(f"  Archived raw transcript: {archived_raw_transcript_filepath}")

    # --- Step 1: Transcribe ---
    print("\n--- Step 1: Transcribing Audio ---")
    transcription_success = transcribe.transcribe_audio_to_file(
        audio_filepath, raw_transcript_filepath_intermediate
    )
    if transcription_success is None:
        print("ERROR: Transcription step failed. Aborting pipeline.")
        sys.exit(1)
    print("Step 1 Complete.")

    # --- Step 2: Classify ---
    print("\n--- Step 2: Classifying Transcript ---")
    try:
        with open(raw_transcript_filepath_intermediate, "r", encoding="utf-8") as f:
            raw_transcript_content = f.read()
    except Exception as e:
        print(f"ERROR: Could not re-read raw transcript file for classification: {e}")
        print("Aborting pipeline.")
        sys.exit(1)

    transcript_type_tag = classify.classify_transcript(raw_transcript_content)

    if transcript_type_tag is None:
        print("ERROR: Transcript could not be classified. Every note needs a type tag. Aborting pipeline.")
        if os.path.exists(raw_transcript_filepath_intermediate):
            try:
                os.remove(raw_transcript_filepath_intermediate)
                print(f"  Cleaned up intermediate raw transcript: {raw_transcript_filepath_intermediate}")
            except Exception as e:
                print(
                    f"Warning: Could not clean up intermediate raw transcript "
                    f"{raw_transcript_filepath_intermediate}: {e}"
                )
        sys.exit(1)

    print(f"Transcript classified as: {transcript_type_tag}")
    print("Step 2 Complete.")

    # --- Step 3: Segment ---
    print("\n--- Step 3: Segmenting Transcript ---")
    # Only require GOOGLE_API_KEY if using Gemini backend
    if config.LLM_BACKEND == "gemini" and config.GOOGLE_API_KEY is None:
        print("ERROR: GOOGLE_API_KEY environment variable is not set and LLM_BACKEND is 'gemini'.")
        print("Aborting pipeline.")
        if os.path.exists(raw_transcript_filepath_intermediate):
            try:
                os.remove(raw_transcript_filepath_intermediate)
                print(f"  Cleaned up intermediate raw transcript: {raw_transcript_filepath_intermediate}")
            except Exception as e:
                print(
                    f"Warning: Could not clean up intermediate raw transcript "
                    f"{raw_transcript_filepath_intermediate}: {e}"
                )
        sys.exit(1)

    segmented_data = segment.segment_and_title_transcript_gemini(raw_transcript_content)

    if not segmented_data:
        print("ERROR: Segmentation step failed or produced no segments. Aborting pipeline.")
        if os.path.exists(raw_transcript_filepath_intermediate):
            try:
                os.remove(raw_transcript_filepath_intermediate)
                print(f"  Cleaned up intermediate raw transcript: {raw_transcript_filepath_intermediate}")
            except Exception as e:
                print(
                    f"Warning: Could not clean up intermediate raw transcript "
                    f"{raw_transcript_filepath_intermediate}: {e}"
                )
        sys.exit(1)

    try:
        os.makedirs(os.path.dirname(segmented_filepath_intermediate), exist_ok=True)
        with open(segmented_filepath_intermediate, "w", encoding="utf-8") as f:
            json.dump(segmented_data, f, indent=4)
        print(f"Segmented data saved to {segmented_filepath_intermediate}")
    except Exception as e:
        print(f"Warning: Could not save segmented data to intermediate file: {e}")

    print("Step 3 Complete.")

    # --- Step 4: Create Obsidian Notes & Embed ---
    print("\n--- Step 4: Creating Obsidian Notes ---")
    notes_with_embeddings_info = create_notes.create_notes_from_segments(
        segmented_data,
        os.path.abspath(config.OBSIDIAN_VAULT_ROOT),
        config.NOTES_SUBDIRECTORY_IN_VAULT,
        recording_id,
        transcript_type_tag,
    )

    if not notes_with_embeddings_info:
        print("Warning: Note creation completed, but no notes had valid embeddings for semantic linking.")
        print("Step 4 Complete (No notes for linking).")
    else:
        print("Step 4 Complete. Notes created and initial embeddings calculated.")

    # --- Step 5: Perform Semantic Linking ---
    print("\n--- Step 5: Performing Semantic Linking ---")
    try:
        all_vault_notes_data_embeddings = link_notes.update_and_load_vault_embeddings(
            os.path.abspath(config.OBSIDIAN_VAULT_ROOT),
            config.EMBEDDINGS_CACHE_FILE,
        )

        if all_vault_notes_data_embeddings:
            link_notes.update_all_notes_links(
                all_vault_notes_data_embeddings,
                config.SIMILARITY_THRESHOLD,
            )
            print("Step 5 Complete: Semantic linking finished.")
        else:
            print("Step 5 skipped: No notes in vault or valid embeddings for linking.")
    except Exception as e:
        print(f"Warning: Semantic linking step encountered an error: {e}")

    # --- Step 6 & 7: Archiving and Cleanup ---
    print("\n--- Step 6 & 7: Archiving and Cleanup ---")

    print("Archiving original audio file...")
    try:
        shutil.move(audio_filepath, archived_audio_filepath)
        print(f"  Archived audio file to: {archived_audio_filepath}")
    except FileNotFoundError:
        print(f"Warning: Original audio file not found at {audio_filepath} for archiving.")
    except Exception as e:
        print(f"Warning: Failed to archive original audio file from {audio_filepath}: {e}")

    print("Archiving raw transcript text file...")
    try:
        shutil.move(raw_transcript_filepath_intermediate, archived_raw_transcript_filepath)
        print(f"  Archived raw transcript to: {archived_raw_transcript_filepath}")
    except FileNotFoundError:
        print(
            f"Warning: Intermediate raw transcript file not found at "
            f"{raw_transcript_filepath_intermediate} for archiving."
        )
    except Exception as e:
        print(
            f"Warning: Failed to archive raw transcript from "
            f"{raw_transcript_filepath_intermediate}: {e}"
        )

    print("Cleaning up intermediate segmented JSON file...")
    try:
        if os.path.exists(segmented_filepath_intermediate):
            os.remove(segmented_filepath_intermediate)
            print(f"  Cleaned up {segmented_filepath_intermediate}")
        else:
            print(f"  Intermediate segmented file not found: {segmented_filepath_intermediate} (already gone?)")
    except Exception as e:
        print(f"Warning: Failed to clean up segmented intermediate file {segmented_filepath_intermediate}: {e}")

    print("\n--- Zettelpal Pipeline Finished ---")
    print(f"Processed Recording ID: {recording_id}")
    print(
        f"New notes created in: "
        f"{os.path.join(config.OBSIDIAN_VAULT_ROOT, config.NOTES_SUBDIRECTORY_IN_VAULT)}"
    )
    print(f"Archived files located in: {config.ARCHIVE_DIR}")
    print(f"Embedding cache is updated: {config.EMBEDDINGS_CACHE_FILE}")


# --- Entry Point ---
if __name__ == "__main__":
    ensure_directories()
    app = gui.ZettelpalGUI()
    app.mainloop()
