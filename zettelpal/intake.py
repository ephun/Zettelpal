# intake.py - Recording intake: validate, rename, dedupe, and process
# audio files from a folder (one-shot or continuous watch).
#
# Duplicate prevention is content-based: a SHA-256 ledger of every audio file
# already run through the pipeline lives in the data dir, so re-uploading the
# same recording under any filename is skipped.

import hashlib
import json
import os
import time

from zettelpal import config, naming
from zettelpal.log import get_logger

log = get_logger(__name__)

AUDIO_EXTENSIONS = {
    ".mp3", ".m4a", ".wav", ".ogg", ".flac", ".webm", ".mp4", ".aac", ".wma",
}

# Give up on a file after this many failed pipeline runs within one watch session.
MAX_ATTEMPTS = 3

READY = "ready"
DUPLICATE = "duplicate"
ERROR = "error"


def ledger_path() -> str:
    return os.path.join(config.settings.data_dir, "zettelpal_intake_ledger.json")


def load_ledger() -> dict:
    """Ledger maps content hash -> info about the processed recording."""
    try:
        with open(ledger_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as e:
        log.warning(f"[INTAKE] Could not read intake ledger: {e}")
        return {}


def save_ledger(ledger: dict) -> None:
    try:
        os.makedirs(os.path.dirname(ledger_path()), exist_ok=True)
        with open(ledger_path(), "w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=2)
    except OSError as e:
        log.warning(f"[INTAKE] Could not save intake ledger: {e}")


def file_sha256(filepath: str) -> str:
    digest = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_audio_file(filepath: str) -> bool:
    return os.path.splitext(filepath)[1].lower() in AUDIO_EXTENSIONS


def scan_folder(folder: str) -> list[str]:
    """Audio files directly in the folder, oldest first."""
    try:
        entries = [
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if is_audio_file(name)
        ]
    except OSError as e:
        log.error(f"[INTAKE] Could not list {folder}: {e}")
        return []
    entries = [p for p in entries if os.path.isfile(p)]
    entries.sort(key=lambda p: os.path.getmtime(p))
    return entries


def intake_file(filepath: str, ledger: dict) -> tuple[str, str | None, str | None]:
    """
    Prepare one audio file for the pipeline: reject duplicates by content
    hash and rename to the MMDDYYNN scheme if needed.

    Returns (status, ready_filepath, content_hash) where status is READY,
    DUPLICATE, or ERROR; the path and hash are only set when READY.
    """
    try:
        digest = file_sha256(filepath)
    except OSError as e:
        log.error(f"[INTAKE] Could not read {filepath}: {e}")
        return ERROR, None, None

    if digest in ledger:
        previous = ledger[digest]
        log.warning(
            f"[INTAKE] Skipping duplicate {os.path.basename(filepath)}: "
            f"same audio already processed as {previous.get('recording_id', '?')} "
            f"on {previous.get('processed_at', '?')}"
        )
        return DUPLICATE, None, None

    if not naming.is_valid_zettelpal_filename(os.path.basename(filepath)):
        log.info(f"[INTAKE] Renaming {os.path.basename(filepath)} to Zettelpal format...")
        renamed = naming.rename_audio_file_to_zettelpal_format(filepath)
        if not renamed:
            log.error(f"[INTAKE] Could not rename {os.path.basename(filepath)}. Skipping.")
            return ERROR, None, None
        filepath = renamed

    return READY, filepath, digest


def mark_processed(ledger: dict, digest: str, recording_id: str, original_name: str) -> None:
    ledger[digest] = {
        "recording_id": recording_id,
        "original_name": original_name,
        "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_ledger(ledger)


def process_file(filepath: str, manual_tags: str = "") -> bool:
    """Intake and run the pipeline for a single file. Returns True on success;
    a previously-processed duplicate counts as a clean no-op."""
    from zettelpal import pipeline  # deferred: pulls in the model stack

    ledger = load_ledger()
    original_name = os.path.basename(filepath)
    status, ready_path, digest = intake_file(filepath, ledger)
    if status == DUPLICATE:
        return True
    if status == ERROR:
        return False

    recording_id = os.path.splitext(os.path.basename(ready_path))[0]
    success = pipeline.run_pipeline(ready_path, manual_tags)
    if success:
        mark_processed(ledger, digest, recording_id, original_name)
    return success


def process_folder(folder: str, manual_tags: str = "") -> tuple[int, int, int]:
    """
    One-shot intake of every audio file in a folder (n8n-friendly).

    Returns (processed, skipped_duplicates, failed).
    """
    from zettelpal import pipeline

    processed = skipped = failed = 0
    ledger = load_ledger()

    for filepath in scan_folder(folder):
        original_name = os.path.basename(filepath)
        status, ready_path, digest = intake_file(filepath, ledger)
        if status == DUPLICATE:
            skipped += 1
            continue
        if status == ERROR:
            failed += 1
            continue

        recording_id = os.path.splitext(os.path.basename(ready_path))[0]
        if pipeline.run_pipeline(ready_path, manual_tags):
            mark_processed(ledger, digest, recording_id, original_name)
            processed += 1
        else:
            failed += 1

    log.info(
        f"[INTAKE] Folder pass complete: {processed} processed, "
        f"{skipped} duplicate(s) skipped, {failed} failed."
    )
    return processed, skipped, failed


def watch_folder(folder: str, interval: int = 30, manual_tags: str = "") -> None:
    """
    Poll a folder (e.g. a Nextcloud sync dir) and process new recordings.

    Polling is used instead of filesystem events because network shares and
    sync clients deliver events unreliably. A file is only picked up once its
    size is unchanged between two consecutive scans, so half-uploaded files
    are left alone.
    """
    from zettelpal import pipeline

    log.info(f"[WATCH] Watching {folder} every {interval}s. Ctrl+C to stop.")
    previous_sizes: dict[str, int] = {}
    attempts: dict[str, int] = {}

    while True:
        current_sizes: dict[str, int] = {}
        for filepath in scan_folder(folder):
            try:
                current_sizes[filepath] = os.path.getsize(filepath)
            except OSError:
                continue

        ledger = load_ledger()
        for filepath, size in current_sizes.items():
            if previous_sizes.get(filepath) != size:
                continue  # new or still uploading; check again next scan
            if attempts.get(filepath, 0) >= MAX_ATTEMPTS:
                continue

            original_name = os.path.basename(filepath)
            status, ready_path, digest = intake_file(filepath, ledger)
            if status == DUPLICATE:
                # Remember so the same file isn't re-hashed every scan.
                attempts[filepath] = MAX_ATTEMPTS
                continue
            if status == ERROR:
                attempts[filepath] = attempts.get(filepath, 0) + 1
                continue

            recording_id = os.path.splitext(os.path.basename(ready_path))[0]
            if pipeline.run_pipeline(ready_path, manual_tags):
                mark_processed(ledger, digest, recording_id, original_name)
                attempts.pop(filepath, None)
            else:
                attempts[filepath] = attempts.get(filepath, 0) + 1
                if attempts[filepath] >= MAX_ATTEMPTS:
                    log.error(
                        f"[WATCH] Giving up on {original_name} after "
                        f"{MAX_ATTEMPTS} failed attempts."
                    )

        previous_sizes = current_sizes
        time.sleep(interval)
