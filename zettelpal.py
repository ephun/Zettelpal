# zettelpal.py - Main Entry Point
# Privacy-focused audio-to-Obsidian pipeline using local models only

import argparse
import os
import sys
import datetime
import re
import shutil
import json

import config
import utils
import transcribe
import segment
import create_notes
import link_notes
import classify
import clip
import vault_integrity


def ensure_directories():
    """Creates necessary Zettelpal directories."""
    os.makedirs(config.ZETTELPAL_ROOT, exist_ok=True)
    os.makedirs(config.RAW_TRANSCRIPTS_INTERMEDIATE_DIR, exist_ok=True)
    os.makedirs(config.SEGMENTED_OUTPUT_INTERMEDIATE_DIR, exist_ok=True)
    os.makedirs(config.ARCHIVE_DIR, exist_ok=True)
    os.makedirs(config.CLIPS_DIR, exist_ok=True)

    if not os.path.exists(config.OBSIDIAN_VAULT_ROOT):
        print(f"ERROR: Obsidian vault not found at {config.OBSIDIAN_VAULT_ROOT}")
        print("Please check your config.py and ensure OBSIDIAN_VAULT_ROOT is correct.")
        sys.exit(1)

    notes_subdir = os.path.join(config.OBSIDIAN_VAULT_ROOT, config.NOTES_SUBDIRECTORY_IN_VAULT)
    os.makedirs(notes_subdir, exist_ok=True)

    cache_dir = os.path.dirname(config.EMBEDDINGS_CACHE_FILE)
    os.makedirs(cache_dir, exist_ok=True)

    print("All directories ready.")


def run_pipeline(audio_filepath: str, manual_tags: str = "") -> bool:
    """
    Runs the full Zettelpal pipeline for a single audio file.

    Returns True on success, False on failure.
    """
    print("=" * 60)
    print("ZETTELPAL PIPELINE")
    print("=" * 60)

    audio_basename = os.path.basename(audio_filepath)
    audio_filename_stem = os.path.splitext(audio_basename)[0]
    audio_extension = os.path.splitext(audio_basename)[1].lstrip(".")

    # Validate filename format
    if not re.match(r"^[0-9]{6,8}$", audio_filename_stem):
        print(f"ERROR: Filename '{audio_filename_stem}' doesn't match MMDDYYNN format.")
        print("Please rename your audio file (e.g., 11222501.mp3)")
        return False

    recording_id = audio_filename_stem
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # Define file paths
    raw_transcript_path = os.path.join(
        config.RAW_TRANSCRIPTS_INTERMEDIATE_DIR, f"{recording_id}.txt"
    )
    segmented_path = os.path.join(
        config.SEGMENTED_OUTPUT_INTERMEDIATE_DIR, f"{recording_id}_segmented.json"
    )
    archived_audio_path = os.path.join(
        config.ARCHIVE_DIR, f"{recording_id}_{timestamp}.{audio_extension}"
    )
    archived_transcript_path = os.path.join(
        config.ARCHIVE_DIR, f"{recording_id}_{timestamp}.txt"
    )

    print(f"\nRecording ID: {recording_id}")
    print(f"Input: {audio_filepath}")

    # === STEP 1: TRANSCRIBE ===
    print("\n--- Step 1: Transcribing Audio ---")
    transcribe_result = transcribe.transcribe_audio_to_file(audio_filepath, raw_transcript_path)
    if transcribe_result is None:
        print("ERROR: Transcription failed.")
        return False
    _, json_sidecar_path = transcribe_result
    print("Transcription complete.")

    # Load whisper timestamp segments
    whisper_segments = None
    if json_sidecar_path and os.path.exists(json_sidecar_path):
        try:
            with open(json_sidecar_path, "r", encoding="utf-8") as f:
                whisper_segments = json.load(f)
            print(f"Loaded {len(whisper_segments)} Whisper timestamp segments.")
        except Exception as e:
            print(f"Warning: Could not load timestamps: {e}")

    # === STEP 2: CLASSIFY ===
    print("\n--- Step 2: Classifying Transcript ---")
    try:
        with open(raw_transcript_path, "r", encoding="utf-8") as f:
            raw_transcript = f.read()
    except Exception as e:
        print(f"ERROR: Could not read transcript: {e}")
        return False

    transcript_type = classify.classify_transcript(raw_transcript)
    if transcript_type is None:
        print("ERROR: Classification failed. Cleaning up.")
        _cleanup_file(raw_transcript_path)
        return False
    print(f"Classification: {transcript_type}")

    # === STEP 3: SEGMENT ===
    print("\n--- Step 3: Segmenting Transcript ---")
    segmented_data = segment.segment_transcript(raw_transcript, whisper_segments=whisper_segments)
    if not segmented_data:
        print("ERROR: Segmentation failed. Cleaning up.")
        _cleanup_file(raw_transcript_path)
        return False

    # Save segmented data
    try:
        os.makedirs(os.path.dirname(segmented_path), exist_ok=True)
        with open(segmented_path, "w", encoding="utf-8") as f:
            json.dump(segmented_data, f, indent=2)
        print(f"Segments saved: {len(segmented_data)} segments")
    except Exception as e:
        print(f"Warning: Could not save segments: {e}")

    # === STEP 3.5: EXTRACT AUDIO CLIPS ===
    clip_paths = None
    has_timestamps = any("audio_start" in seg for seg in segmented_data)
    if has_timestamps:
        print("\n--- Step 3.5: Extracting Audio Clips ---")
        clip_paths = clip.extract_all_clips(
            audio_filepath, segmented_data, config.CLIPS_DIR, recording_id
        )
    else:
        print("\n--- Step 3.5: Skipping clip extraction (no timestamps) ---")

    # === STEP 4: CREATE NOTES ===
    print("\n--- Step 4: Creating Obsidian Notes ---")
    notes_info = create_notes.create_notes_from_segments(
        segmented_data,
        os.path.abspath(config.OBSIDIAN_VAULT_ROOT),
        config.NOTES_SUBDIRECTORY_IN_VAULT,
        recording_id,
        transcript_type,
        manual_tags,
        clip_paths=clip_paths
    )

    if not notes_info:
        print("Warning: No notes created with embeddings.")
    else:
        print(f"Created {len(notes_info)} notes.")

    # === STEP 5: SOURCE INTEGRITY ===
    print("\n--- Step 5: Source Integrity Check ---")
    try:
        moved_duplicates = vault_integrity.quarantine_duplicate_notes_for_source(
            os.path.abspath(config.OBSIDIAN_VAULT_ROOT),
            recording_id,
        )
        if moved_duplicates:
            moved_paths = {item["relative_path"] for item in moved_duplicates}
            notes_info = [
                note for note in notes_info
                if os.path.relpath(note["filepath"], os.path.abspath(config.OBSIDIAN_VAULT_ROOT)).replace("\\", "/")
                not in moved_paths
            ]
            print(f"Quarantined {len(moved_duplicates)} duplicate note(s) for {recording_id}.")
        else:
            print("No duplicate notes found for this recording.")

        integrity_issues = vault_integrity.validate_source_integrity(
            os.path.abspath(config.OBSIDIAN_VAULT_ROOT),
            recording_id,
        )
        if integrity_issues:
            print(f"Warning: {len(integrity_issues)} integrity issue(s) found for {recording_id}:")
            for relpath, issue in integrity_issues[:10]:
                print(f"  - {relpath}: {issue}")
            if len(integrity_issues) > 10:
                print(f"  ... and {len(integrity_issues) - 10} more")
        else:
            print("Source integrity check passed.")
    except Exception as e:
        print(f"Warning: Source integrity check failed: {e}")

    # === STEP 6: SEMANTIC LINKING ===
    print("\n--- Step 6: Semantic Linking ---")
    try:
        all_notes, _ = link_notes.update_and_load_vault_embeddings(
            os.path.abspath(config.OBSIDIAN_VAULT_ROOT),
            config.EMBEDDINGS_CACHE_FILE,
            notes_info
        )

        if all_notes:
            link_notes.update_all_notes_links(
                all_notes,
                config.SIMILARITY_THRESHOLD,
                any_note_was_updated_or_created=True
            )
            print("Semantic linking complete.")
        else:
            print("Skipped: No notes for linking.")
    except Exception as e:
        print(f"Warning: Linking error: {e}")

    # === STEP 7: ARCHIVE & CLEANUP ===
    print("\n--- Step 7: Archive & Cleanup ---")

    # Archive audio
    try:
        shutil.move(audio_filepath, archived_audio_path)
        print(f"Archived audio: {os.path.basename(archived_audio_path)}")
    except Exception as e:
        print(f"Warning: Could not archive audio: {e}")

    # Archive transcript
    try:
        shutil.move(raw_transcript_path, archived_transcript_path)
        print(f"Archived transcript: {os.path.basename(archived_transcript_path)}")
    except Exception as e:
        print(f"Warning: Could not archive transcript: {e}")

    # Archive timestamps JSON sidecar
    if json_sidecar_path and os.path.exists(json_sidecar_path):
        archived_json_path = os.path.join(
            config.ARCHIVE_DIR, f"{recording_id}_{timestamp}.json"
        )
        try:
            shutil.move(json_sidecar_path, archived_json_path)
            print(f"Archived timestamps: {os.path.basename(archived_json_path)}")
        except Exception as e:
            print(f"Warning: Could not archive timestamps: {e}")

    # Cleanup segmented JSON
    _cleanup_file(segmented_path)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Notes created in: {os.path.join(config.OBSIDIAN_VAULT_ROOT, config.NOTES_SUBDIRECTORY_IN_VAULT)}")
    print(f"Archives in: {config.ARCHIVE_DIR}")

    return True


def _cleanup_file(filepath: str):
    """Remove a file if it exists."""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Cleaned up: {os.path.basename(filepath)}")
    except Exception as e:
        print(f"Warning: Could not clean up {filepath}: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Zettelpal - Privacy-focused audio to Obsidian notes"
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        help="Audio file to process (launches GUI if not provided)"
    )
    parser.add_argument(
        "--tags",
        default="",
        help="Comma-separated manual tags to add to notes"
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run in CLI mode only (requires audio_file)"
    )
    args = parser.parse_args()

    ensure_directories()

    if args.audio_file:
        # CLI mode
        if not os.path.exists(args.audio_file):
            print(f"ERROR: File not found: {args.audio_file}")
            sys.exit(1)
        success = run_pipeline(args.audio_file, args.tags)
        sys.exit(0 if success else 1)
    elif args.no_gui:
        print("ERROR: --no-gui requires an audio file argument.")
        sys.exit(1)
    else:
        # GUI mode
        import gui

        app = gui.ZettelpalGUI()
        app.mainloop()


if __name__ == "__main__":
    main()
