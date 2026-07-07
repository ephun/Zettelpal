# backfill_audio.py - Retroactive Audio Clip Extraction for Existing Notes
#
# Scans the vault for existing Zettelpal notes, re-runs Whisper on archived audio,
# fuzzy-matches note content to timestamps, extracts clips, and inserts the
# `audio` field into frontmatter.
#
# Usage:
#   python backfill_audio.py --dry-run          # Preview changes
#   python backfill_audio.py                    # Run for real
#   python backfill_audio.py --no-backup        # Skip backups
#   python backfill_audio.py --recording-id 10142502  # Single recording

import argparse
import difflib
import json
import os
import re
import shutil
import sys
import time

import config
import clip
import utils


PROGRESS_FILE = os.path.join(config.ZETTELPAL_ROOT, "backfill_progress.json")
BACKUP_DIR = os.path.join(config.ARCHIVE_DIR, "backups")
AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".ogg", ".flac", ".webm", ".mp4", ".aac"}


def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"completed_recordings": []}


def save_progress(progress: dict):
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def scan_vault_notes() -> dict[str, list[dict]]:
    """
    Scans vault for .md files with zettelpal source frontmatter.
    Returns {recording_id: [note_info, ...]} where note_info has
    filepath, frontmatter_lines, body, link_lines, placement, created.
    """
    md_files = utils.find_all_markdown_files(config.OBSIDIAN_VAULT_ROOT)
    notes_by_recording = {}

    for filepath in md_files:
        fm_lines, body, link_lines = utils.extract_frontmatter_and_body(filepath)
        source = utils.extract_source_from_frontmatter(fm_lines)
        if not source:
            continue

        # Check if already has audio field
        has_audio = any(
            re.match(r"^\s*audio\s*:", line) for line in fm_lines
        )
        if has_audio:
            continue

        # Extract placement
        placement = None
        for line in fm_lines:
            m = re.match(r'^\s*placement:\s*["\']?(\w+)', line)
            if m:
                placement = m.group(1)
                break

        # Extract created timestamp
        created_str = utils.extract_created_timestamp_str_from_frontmatter(fm_lines)
        created_dt = utils.parse_created_timestamp_str(created_str) if created_str else None

        notes_by_recording.setdefault(source, []).append({
            "filepath": filepath,
            "frontmatter_lines": fm_lines,
            "body": body,
            "link_lines": link_lines,
            "placement": placement,
            "created": created_dt,
        })

    return notes_by_recording


def scan_archive_audio() -> dict[str, str]:
    """
    Scans archive dir for audio files.
    Returns {recording_id: audio_file_path}.
    Uses the first 8 chars (MMDDYYNN) as recording_id.
    """
    audio_map = {}
    if not os.path.exists(config.ARCHIVE_DIR):
        return audio_map

    for filename in os.listdir(config.ARCHIVE_DIR):
        _, ext = os.path.splitext(filename)
        if ext.lower() not in AUDIO_EXTENSIONS:
            continue

        # Extract recording_id: first part before _ or the stem if no _
        stem = os.path.splitext(filename)[0]
        # Archive format is typically MMDDYYNN_TIMESTAMP.ext
        match = re.match(r"^(\d{6,8})", stem)
        if match:
            rec_id = match.group(1)
            # Prefer the first (or only) audio file found
            if rec_id not in audio_map:
                audio_map[rec_id] = os.path.join(config.ARCHIVE_DIR, filename)

    return audio_map


def sort_notes_by_order(notes: list[dict]) -> list[dict]:
    """Sort notes by placement order (first→middle→last) then by created time."""
    placement_order = {"first": 0, "only": 0, "middle": 1, "last": 2}

    def sort_key(n):
        p = placement_order.get(n.get("placement"), 1)
        c = n.get("created") or __import__("datetime").datetime.max
        return (p, c)

    return sorted(notes, key=sort_key)


def transcribe_with_timestamps(audio_path: str) -> list[dict] | None:
    """Re-runs Whisper on audio to get timestamp segments."""
    model = utils.load_whisper_model()
    if model is None:
        print("[BACKFILL] ERROR: Whisper model failed to load.")
        return None

    print(f"[BACKFILL] Transcribing: {os.path.basename(audio_path)} (this may take a while)...")
    try:
        start_time = time.time()
        # verbose=True so Whisper prints each segment as it decodes — crucial feedback
        result = model.transcribe(audio_path, verbose=True, word_timestamps=True)
        elapsed = time.time() - start_time
        print(f"[BACKFILL] Transcription complete ({elapsed:.1f}s, {len(result.get('segments', []))} segments)")

        segments = [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
            for seg in result.get("segments", [])
        ]
        return segments
    except Exception as e:
        print(f"[BACKFILL] ERROR: Transcription failed: {e}")
        return None


def build_char_timestamp_map(whisper_segments: list[dict]) -> tuple[str, list[int]]:
    """
    Builds a full transcript string and a parallel list mapping each character
    position to a whisper segment index.
    """
    full_text = ""
    char_to_seg = []
    for idx, seg in enumerate(whisper_segments):
        text = seg["text"]
        for _ in text:
            char_to_seg.append(idx)
        full_text += text
    return full_text, char_to_seg


def fuzzy_match_note_to_transcript(
    note_body: str,
    full_text_lower: str,
    char_to_seg: list[int],
    whisper_segments: list[dict],
    search_start: int,
) -> tuple[float | None, float | None, float, int]:
    """
    Fuzzy-matches a note's body to a region in the transcript using
    SequenceMatcher.get_matching_blocks() for robust subsequence matching.

    Returns (audio_start, audio_end, confidence, new_search_start).
    audio_start/audio_end are None if confidence too low.
    """
    if not note_body.strip() or not full_text_lower:
        return None, None, 0.0, search_start

    # Clean the note body: remove H1 heading, strip whitespace
    body_clean = re.sub(r"^#\s+.*?\n", "", note_body).strip()
    if not body_clean:
        body_clean = note_body.strip()

    body_lower = body_clean.lower()

    # Search in remaining transcript with generous padding (4x note length)
    search_end = min(search_start + len(body_lower) * 4, len(full_text_lower))
    region = full_text_lower[search_start:search_end]

    if not region:
        return None, None, 0.0, search_start

    # Find all common subsequences between note text and transcript region
    matcher = difflib.SequenceMatcher(None, body_lower, region, autojunk=False)
    blocks = matcher.get_matching_blocks()

    # Coverage: fraction of note text found as matching subsequences
    total_matched = sum(b.size for b in blocks if b.size > 0)
    ratio = total_matched / len(body_lower) if body_lower else 0

    if ratio < 0.4:
        return None, None, ratio, search_start

    # Find the span in the transcript from first to last real match
    real_blocks = [b for b in blocks if b.size > 0]
    if not real_blocks:
        return None, None, 0.0, search_start

    region_match_start = real_blocks[0].b
    region_match_end = real_blocks[-1].b + real_blocks[-1].size

    abs_start = search_start + region_match_start
    abs_end = search_start + region_match_end

    # Map character range to whisper segment timestamps
    audio_start = None
    audio_end = None
    if abs_start < len(char_to_seg) and abs_end <= len(char_to_seg):
        start_idx = char_to_seg[min(abs_start, len(char_to_seg) - 1)]
        end_idx = char_to_seg[min(abs_end - 1, len(char_to_seg) - 1)]
        audio_start = max(0.0, whisper_segments[start_idx]["start"] - 0.5)
        audio_end = whisper_segments[end_idx]["end"] + 0.5

    return audio_start, audio_end, ratio, abs_end


def insert_audio_into_frontmatter(filepath: str, audio_path: str, dry_run: bool, no_backup: bool) -> bool:
    """
    Inserts `audio: "..."` into the frontmatter of a note file.
    Only adds the field — never modifies existing fields or body.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"[BACKFILL] ERROR reading {filepath}: {e}")
        return False

    # Find the closing --- of frontmatter
    if not content.startswith("---"):
        print(f"[BACKFILL] SKIP {os.path.basename(filepath)}: no frontmatter")
        return False

    # Find second ---
    second_fence = content.find("---", 3)
    if second_fence == -1:
        print(f"[BACKFILL] SKIP {os.path.basename(filepath)}: malformed frontmatter")
        return False

    safe_audio = audio_path.replace("\\", "/")
    audio_line = f'audio: "{safe_audio}"\n'

    # Insert before the closing ---
    new_content = content[:second_fence] + audio_line + content[second_fence:]

    if dry_run:
        print(f"[DRY RUN] Would add audio field to: {os.path.basename(filepath)}")
        print(f"          audio: \"{safe_audio}\"")
        return True

    # Backup
    if not no_backup:
        os.makedirs(BACKUP_DIR, exist_ok=True)
        backup_path = os.path.join(BACKUP_DIR, os.path.basename(filepath))
        # Add timestamp to avoid collisions
        if os.path.exists(backup_path):
            stem, ext = os.path.splitext(os.path.basename(filepath))
            backup_path = os.path.join(BACKUP_DIR, f"{stem}_{int(time.time())}{ext}")
        try:
            shutil.copy2(filepath, backup_path)
        except Exception as e:
            print(f"[BACKFILL] WARNING: backup failed for {filepath}: {e}")

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
        return True
    except Exception as e:
        print(f"[BACKFILL] ERROR writing {filepath}: {e}")
        return False


def process_recording(
    recording_id: str,
    notes: list[dict],
    audio_path: str,
    dry_run: bool,
    no_backup: bool,
) -> dict:
    """
    Processes a single recording: transcribe, match, extract clips, update notes.
    Returns stats dict.
    """
    stats = {"notes": len(notes), "matched": 0, "clips": 0, "updated": 0, "skipped": 0}

    print(f"\n{'=' * 50}")
    print(f"[BACKFILL] Recording: {recording_id} ({len(notes)} notes)")
    print(f"[BACKFILL] Audio: {os.path.basename(audio_path)}")
    print(f"{'=' * 50}")

    # Step 1: Transcribe with timestamps
    whisper_segments = transcribe_with_timestamps(audio_path)
    if not whisper_segments:
        print(f"[BACKFILL] SKIP {recording_id}: transcription failed")
        stats["skipped"] = len(notes)
        return stats

    # Step 2: Build character-to-timestamp mapping
    full_text, char_to_seg = build_char_timestamp_map(whisper_segments)
    full_text_lower = full_text.lower()

    # Step 3: Sort notes by placement order
    sorted_notes = sort_notes_by_order(notes)
    audio_ext = os.path.splitext(audio_path)[1] or ".m4a"
    audio_duration = whisper_segments[-1]["end"] if whisper_segments else 0.0

    # === Pass 1: Fuzzy matching ===
    search_start = 0
    match_results = []  # parallel to sorted_notes: (audio_start, audio_end, confidence) or None

    for i, note_info in enumerate(sorted_notes):
        body = note_info["body"]
        basename = os.path.basename(note_info["filepath"])

        print(f"[BACKFILL] Note {i + 1}/{len(sorted_notes)}: {basename}")

        audio_start, audio_end, confidence, new_search_start = fuzzy_match_note_to_transcript(
            body, full_text_lower, char_to_seg, whisper_segments, search_start
        )

        if audio_start is not None and audio_end is not None:
            match_results.append((audio_start, audio_end, confidence))
            search_start = new_search_start
            print(f"  -> MATCH: {audio_start:.1f}s - {audio_end:.1f}s (confidence: {confidence:.2f})")
            stats["matched"] += 1
        else:
            match_results.append(None)
            print(f"  -> UNMATCHED (confidence: {confidence:.2f}), will fill from neighbors")

    # === Pass 2: Fill gaps from neighboring matches ===
    # For unmatched notes, split the gap between the previous and next matched
    # notes proportionally by character length.
    for i in range(len(match_results)):
        if match_results[i] is not None:
            continue

        # Find preceding matched boundary
        prev_end = 0.0
        for j in range(i - 1, -1, -1):
            if match_results[j] is not None:
                prev_end = match_results[j][1]
                break

        # Find following matched boundary
        next_start = audio_duration
        for j in range(i + 1, len(match_results)):
            if match_results[j] is not None:
                next_start = match_results[j][0]
                break

        # Collect all consecutive unmatched notes in this gap
        gap_start_idx = i
        while gap_start_idx > 0 and match_results[gap_start_idx - 1] is None:
            gap_start_idx -= 1
        gap_end_idx = i
        while gap_end_idx < len(match_results) - 1 and match_results[gap_end_idx + 1] is None:
            gap_end_idx += 1

        # Get character lengths for proportional splitting
        gap_notes = list(range(gap_start_idx, gap_end_idx + 1))
        char_lengths = []
        for idx in gap_notes:
            body = sorted_notes[idx]["body"]
            clean = re.sub(r"^#\s+.*?\n", "", body).strip()
            char_lengths.append(max(len(clean), 1))
        total_chars = sum(char_lengths)

        # Split the time gap proportionally
        gap_duration = next_start - prev_end
        if gap_duration <= 0:
            continue

        cursor = prev_end
        for k, idx in enumerate(gap_notes):
            proportion = char_lengths[k] / total_chars
            seg_duration = gap_duration * proportion
            seg_start = cursor
            seg_end = cursor + seg_duration
            match_results[idx] = (seg_start, seg_end, 0.0)  # confidence 0 = gap-filled
            cursor = seg_end

        basename = os.path.basename(sorted_notes[i]["filepath"])
        filled_start, filled_end, _ = match_results[i]
        print(f"  [{basename}] GAP-FILLED: {filled_start:.1f}s - {filled_end:.1f}s")
        stats["matched"] += 1

    # === Pass 3: Extract clips and update frontmatter ===
    vault_abs = os.path.abspath(config.OBSIDIAN_VAULT_ROOT)

    for i, note_info in enumerate(sorted_notes):
        if match_results[i] is None:
            stats["skipped"] += 1
            continue

        audio_start, audio_end, confidence = match_results[i]
        filepath = note_info["filepath"]
        basename = os.path.basename(filepath)

        clip_filename = f"{recording_id}_{i + 1:02d}{audio_ext}"
        clip_path = os.path.join(config.CLIPS_DIR, clip_filename)

        clip_abs = os.path.abspath(clip_path)
        if clip_abs.startswith(vault_abs):
            vault_relative = os.path.relpath(clip_abs, vault_abs)
        else:
            vault_relative = clip_path

        if dry_run:
            label = "GAP-FILL" if confidence == 0.0 else f"conf:{confidence:.2f}"
            print(f"  -> [DRY RUN] {basename}: clip {clip_filename} ({audio_start:.1f}s - {audio_end:.1f}s, {label})")
        else:
            os.makedirs(config.CLIPS_DIR, exist_ok=True)
            success = clip.extract_clip(audio_path, audio_start, audio_end, clip_path)
            if not success:
                print(f"  -> ERROR: clip extraction failed for {basename}")
                stats["skipped"] += 1
                continue
            stats["clips"] += 1
            print(f"  -> CLIP: {clip_filename} created")

        if insert_audio_into_frontmatter(filepath, vault_relative, dry_run, no_backup):
            stats["updated"] += 1
            if not dry_run:
                print(f"  -> UPDATED frontmatter for {basename}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Backfill audio clips for existing Zettelpal notes."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview changes without modifying anything."
    )
    parser.add_argument(
        "--no-backup", action="store_true",
        help="Skip backing up notes before modifying."
    )
    parser.add_argument(
        "--recording-id",
        help="Process only this recording ID (e.g., 10142502)."
    )
    parser.add_argument(
        "--reset-progress", action="store_true",
        help="Reset progress tracking and reprocess all recordings."
    )
    args = parser.parse_args()

    if args.dry_run:
        print("[BACKFILL] === DRY RUN MODE — no files will be modified ===\n")

    # Load progress
    progress = load_progress()
    if args.reset_progress:
        progress = {"completed_recordings": []}
        save_progress(progress)
        print("[BACKFILL] Progress reset.\n")

    # Step 1: Scan vault
    print("[BACKFILL] Scanning vault for notes...")
    notes_by_recording = scan_vault_notes()
    print(f"[BACKFILL] Found {sum(len(v) for v in notes_by_recording.values())} notes across {len(notes_by_recording)} recordings.\n")

    # Step 2: Scan archive for audio files
    print("[BACKFILL] Scanning archive for audio files...")
    audio_map = scan_archive_audio()
    print(f"[BACKFILL] Found {len(audio_map)} audio files.\n")

    # Step 3: Filter to recordings that have both notes and audio
    if args.recording_id:
        target_ids = [args.recording_id]
    else:
        target_ids = sorted(set(notes_by_recording.keys()) & set(audio_map.keys()))

    # Skip already completed
    completed = set(progress.get("completed_recordings", []))
    pending_ids = [rid for rid in target_ids if rid not in completed]

    print(f"[BACKFILL] {len(pending_ids)} recordings to process ({len(completed)} already done).\n")

    if not pending_ids:
        print("[BACKFILL] Nothing to do.")
        return

    # Step 4: Process each recording
    total_stats = {"notes": 0, "matched": 0, "clips": 0, "updated": 0, "skipped": 0}
    run_start_time = time.time()

    for idx, rid in enumerate(pending_ids):
        # Overall progress header
        elapsed_total = time.time() - run_start_time
        if idx > 0:
            avg_per_recording = elapsed_total / idx
            remaining = avg_per_recording * (len(pending_ids) - idx)
            eta_min = remaining / 60
            print(f"\n[PROGRESS] Recording {idx + 1}/{len(pending_ids)} | "
                  f"Elapsed: {elapsed_total / 60:.1f}min | ETA: ~{eta_min:.0f}min remaining")
        else:
            print(f"\n[PROGRESS] Recording {idx + 1}/{len(pending_ids)}")

        if rid not in notes_by_recording:
            print(f"[BACKFILL] SKIP {rid}: no notes found in vault")
            continue
        if rid not in audio_map:
            print(f"[BACKFILL] SKIP {rid}: no audio file in archive")
            continue

        rec_start = time.time()
        stats = process_recording(
            rid, notes_by_recording[rid], audio_map[rid],
            args.dry_run, args.no_backup
        )
        rec_elapsed = time.time() - rec_start
        print(f"[BACKFILL] Recording {rid} done in {rec_elapsed:.1f}s "
              f"({stats['matched']} matched, {stats['clips']} clips, {stats['skipped']} skipped)")

        for k in total_stats:
            total_stats[k] += stats[k]

        # Save progress (even in dry-run, to track what was reviewed)
        if not args.dry_run:
            progress["completed_recordings"].append(rid)
            save_progress(progress)

    # Summary
    total_elapsed = time.time() - run_start_time
    print(f"\n{'=' * 60}")
    print("[BACKFILL] COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total time:     {total_elapsed / 60:.1f} minutes")
    print(f"  Recordings:     {len(pending_ids)}")
    print(f"  Notes scanned:  {total_stats['notes']}")
    print(f"  Matched:        {total_stats['matched']}")
    print(f"  Clips created:  {total_stats['clips']}")
    print(f"  Notes updated:  {total_stats['updated']}")
    print(f"  Skipped:        {total_stats['skipped']}")

    if args.dry_run:
        print("\n  (Dry run — no files were modified)")


if __name__ == "__main__":
    main()
