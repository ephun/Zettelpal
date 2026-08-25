# pipeline.py - The full audio-to-Obsidian pipeline for one recording.

import datetime
import json
import os
import re
import shutil
import sys

from zettelpal import classify, clip, config, create_notes, segment, transcribe
from zettelpal.log import get_logger
from zettelpal.vault import integrity as vault_integrity
from zettelpal.vault import linking

log = get_logger(__name__)


def ensure_directories():
    """Creates necessary Zettelpal directories."""
    os.makedirs(config.settings.data_dir, exist_ok=True)
    os.makedirs(config.settings.raw_transcripts_dir, exist_ok=True)
    os.makedirs(config.settings.segmented_output_dir, exist_ok=True)
    os.makedirs(config.settings.resolved_archive_dir, exist_ok=True)
    os.makedirs(config.settings.clips_dir, exist_ok=True)

    if not os.path.exists(config.settings.vault_root):
        log.error(f"ERROR: Obsidian vault not found at {config.settings.vault_root}")
        log.error("Set vault_root in zettelpal.toml or the ZETTELPAL_VAULT_ROOT environment variable.")
        sys.exit(1)

    notes_subdir = os.path.join(config.settings.vault_root, config.settings.notes_subdirectory)
    os.makedirs(notes_subdir, exist_ok=True)

    cache_dir = os.path.dirname(config.settings.embeddings_cache_file)
    os.makedirs(cache_dir, exist_ok=True)

    log.info("All directories ready.")


def run_pipeline(audio_filepath: str, manual_tags: str = "") -> bool:
    """
    Runs the full Zettelpal pipeline for a single audio file.

    Returns True on success, False on failure.
    """
    log.info("=" * 60)
    log.info("ZETTELPAL PIPELINE")
    log.info("=" * 60)

    audio_basename = os.path.basename(audio_filepath)
    audio_filename_stem = os.path.splitext(audio_basename)[0]
    audio_extension = os.path.splitext(audio_basename)[1].lstrip(".")

    # Validate filename format
    if not re.match(r"^[0-9]{6,8}$", audio_filename_stem):
        log.error(f"ERROR: Filename '{audio_filename_stem}' doesn't match MMDDYYNN format.")
        log.info("Please rename your audio file (e.g., 11222501.mp3)")
        return False

    recording_id = audio_filename_stem
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # Define file paths
    raw_transcript_path = os.path.join(
        config.settings.raw_transcripts_dir, f"{recording_id}.txt"
    )
    segmented_path = os.path.join(
        config.settings.segmented_output_dir, f"{recording_id}_segmented.json"
    )
    archived_audio_path = os.path.join(
        config.settings.resolved_archive_dir, f"{recording_id}_{timestamp}.{audio_extension}"
    )
    archived_transcript_path = os.path.join(
        config.settings.resolved_archive_dir, f"{recording_id}_{timestamp}.txt"
    )

    log.info(f"\nRecording ID: {recording_id}")
    log.info(f"Input: {audio_filepath}")

    # === STEP 1: TRANSCRIBE ===
    log.info("\n--- Step 1: Transcribing Audio ---")
    transcribe_result = transcribe.transcribe_audio_to_file(audio_filepath, raw_transcript_path)
    if transcribe_result is None:
        log.error("ERROR: Transcription failed.")
        return False
    _, json_sidecar_path = transcribe_result
    log.info("Transcription complete.")

    # Load whisper timestamp segments
    whisper_segments = None
    if json_sidecar_path and os.path.exists(json_sidecar_path):
        try:
            with open(json_sidecar_path, "r", encoding="utf-8") as f:
                whisper_segments = json.load(f)
            log.info(f"Loaded {len(whisper_segments)} Whisper timestamp segments.")
        except (OSError, json.JSONDecodeError) as e:
            log.warning(f"Warning: Could not load timestamps: {e}")

    # === STEP 2: CLASSIFY ===
    log.info("\n--- Step 2: Classifying Transcript ---")
    try:
        with open(raw_transcript_path, "r", encoding="utf-8") as f:
            raw_transcript = f.read()
    except OSError as e:
        log.error(f"ERROR: Could not read transcript: {e}")
        return False

    transcript_type = classify.classify_transcript(raw_transcript)
    if transcript_type is None:
        log.error("ERROR: Classification failed. Cleaning up.")
        _cleanup_file(raw_transcript_path)
        return False
    log.info(f"Classification: {transcript_type}")

    # === STEP 3: SEGMENT ===
    log.info("\n--- Step 3: Segmenting Transcript ---")
    segmented_data = segment.segment_transcript(raw_transcript, whisper_segments=whisper_segments)
    if not segmented_data:
        log.error("ERROR: Segmentation failed. Cleaning up.")
        _cleanup_file(raw_transcript_path)
        return False

    # Save segmented data
    try:
        os.makedirs(os.path.dirname(segmented_path), exist_ok=True)
        with open(segmented_path, "w", encoding="utf-8") as f:
            json.dump(segmented_data, f, indent=2)
        log.info(f"Segments saved: {len(segmented_data)} segments")
    except OSError as e:
        log.warning(f"Warning: Could not save segments: {e}")

    # === STEP 3.5: EXTRACT AUDIO CLIPS ===
    clip_paths = None
    has_timestamps = any("audio_start" in seg for seg in segmented_data)
    if has_timestamps:
        log.info("\n--- Step 3.5: Extracting Audio Clips ---")
        clip_paths = clip.extract_all_clips(
            audio_filepath, segmented_data, config.settings.clips_dir, recording_id
        )
    else:
        log.info("\n--- Step 3.5: Skipping clip extraction (no timestamps) ---")

    # === STEP 4: CREATE NOTES ===
    log.info("\n--- Step 4: Creating Obsidian Notes ---")
    notes_info = create_notes.create_notes_from_segments(
        segmented_data,
        os.path.abspath(config.settings.vault_root),
        config.settings.notes_subdirectory,
        recording_id,
        transcript_type,
        manual_tags,
        clip_paths=clip_paths
    )

    if not notes_info:
        log.warning("Warning: No notes created with embeddings.")
    else:
        log.info(f"Created {len(notes_info)} notes.")

    # === STEP 5: SOURCE INTEGRITY ===
    log.info("\n--- Step 5: Source Integrity Check ---")
    try:
        moved_duplicates = vault_integrity.quarantine_duplicate_notes_for_source(
            os.path.abspath(config.settings.vault_root),
            recording_id,
        )
        if moved_duplicates:
            moved_paths = {item["relative_path"] for item in moved_duplicates}
            notes_info = [
                note for note in notes_info
                if os.path.relpath(note["filepath"], os.path.abspath(config.settings.vault_root)).replace("\\", "/")
                not in moved_paths
            ]
            log.info(f"Quarantined {len(moved_duplicates)} duplicate note(s) for {recording_id}.")
        else:
            log.info("No duplicate notes found for this recording.")

        integrity_issues = vault_integrity.validate_source_integrity(
            os.path.abspath(config.settings.vault_root),
            recording_id,
        )
        if integrity_issues:
            log.warning(f"Warning: {len(integrity_issues)} integrity issue(s) found for {recording_id}:")
            for relpath, issue in integrity_issues[:10]:
                log.info(f"  - {relpath}: {issue}")
            if len(integrity_issues) > 10:
                log.info(f"  ... and {len(integrity_issues) - 10} more")
        else:
            log.info("Source integrity check passed.")
    except Exception as e:
        # exc_info: these handlers are wide on purpose (a failed check must not
        # cost the user their notes), but without the traceback the message
        # names neither the file nor the call that failed.
        log.warning(f"Warning: Source integrity check failed: {e}", exc_info=True)

    # === STEP 6: SEMANTIC LINKING ===
    log.info("\n--- Step 6: Semantic Linking ---")
    try:
        all_notes, _ = linking.update_and_load_vault_embeddings(
            os.path.abspath(config.settings.vault_root),
            config.settings.embeddings_cache_file,
            notes_info
        )

        if all_notes:
            linking.update_all_notes_links(
                all_notes,
                config.settings.similarity_threshold,
                any_note_was_updated_or_created=True
            )
            log.info("Semantic linking complete.")
        else:
            log.info("Skipped: No notes for linking.")
    except Exception as e:
        log.warning(f"Warning: Linking error: {e}", exc_info=True)

    # === STEP 7: ARCHIVE & CLEANUP ===
    log.info("\n--- Step 7: Archive & Cleanup ---")

    # Ensure the archive destination exists so a direct run_pipeline() call
    # doesn't strand the audio when ensure_directories() wasn't run first.
    os.makedirs(config.settings.resolved_archive_dir, exist_ok=True)

    # Archive audio
    try:
        shutil.move(audio_filepath, archived_audio_path)
        log.info(f"Archived audio: {os.path.basename(archived_audio_path)}")
    except (OSError, shutil.Error) as e:
        log.warning(f"Warning: Could not archive audio: {e}")

    # Archive transcript
    try:
        shutil.move(raw_transcript_path, archived_transcript_path)
        log.info(f"Archived transcript: {os.path.basename(archived_transcript_path)}")
    except (OSError, shutil.Error) as e:
        log.warning(f"Warning: Could not archive transcript: {e}")

    # Archive timestamps JSON sidecar
    if json_sidecar_path and os.path.exists(json_sidecar_path):
        archived_json_path = os.path.join(
            config.settings.resolved_archive_dir, f"{recording_id}_{timestamp}.json"
        )
        try:
            shutil.move(json_sidecar_path, archived_json_path)
            log.info(f"Archived timestamps: {os.path.basename(archived_json_path)}")
        except (OSError, shutil.Error) as e:
            log.warning(f"Warning: Could not archive timestamps: {e}")

    # Cleanup segmented JSON
    _cleanup_file(segmented_path)

    log.info("\n" + "=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info("=" * 60)
    log.info(f"Notes created in: {os.path.join(config.settings.vault_root, config.settings.notes_subdirectory)}")
    log.info(f"Archives in: {config.settings.resolved_archive_dir}")

    return True


def _cleanup_file(filepath: str):
    """Remove a file if it exists."""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            log.info(f"Cleaned up: {os.path.basename(filepath)}")
    except OSError as e:
        log.warning(f"Warning: Could not clean up {filepath}: {e}")
