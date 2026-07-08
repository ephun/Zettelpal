# naming.py - Recording filename scheme (MMDDYYNN) and filename sanitizing.

import datetime
import os
import re

from zettelpal import config
from zettelpal.log import get_logger

log = get_logger(__name__)


def sanitize_filename(title: str) -> str:
    """Convert a title to a filesystem-safe filename stem, preserving case."""
    if not title:
        return "Untitled"
    # Remove characters that are invalid in filenames, but keep letters, numbers, spaces, hyphens
    s = re.sub(r'[<>:"/\\|?*]', '', title)
    # Replace spaces with hyphens
    s = s.replace(" ", "-")
    # Collapse multiple hyphens
    s = re.sub(r'-+', '-', s)
    # Strip leading/trailing hyphens
    s = s.strip("-")
    return s[:100] if s else "Untitled"


def is_valid_zettelpal_filename(filename: str) -> bool:
    """Checks if filename matches MMDDYYNN.ext format."""
    return re.match(r"^\d{6,8}\..+$", filename) is not None


def rename_audio_file_to_zettelpal_format(filepath: str) -> str | None:
    """
    Renames an audio file to MMDDYYNN.ext format based on modification time.
    """
    if not os.path.exists(filepath):
        log.error(f"Error: File not found: {filepath}")
        return None

    try:
        mtime = os.path.getmtime(filepath)
        dt = datetime.datetime.fromtimestamp(mtime)
        date_prefix = dt.strftime("%m%d%y")

        # Find next available NN
        next_nn = _get_next_available_nn(date_prefix)
        if len(next_nn) > 2:
            log.error(f"Error: Too many recordings for date {date_prefix}")
            return None

        _, ext = os.path.splitext(filepath)
        new_filename = f"{date_prefix}{next_nn}{ext.lower()}"
        new_filepath = os.path.join(os.path.dirname(filepath), new_filename)

        if os.path.exists(new_filepath):
            log.error(f"Error: Target file already exists: {new_filename}")
            return None

        os.rename(filepath, new_filepath)
        log.info(f"Renamed to: {new_filename}")
        return os.path.abspath(new_filepath)

    except Exception as e:
        log.error(f"Error renaming file: {e}")
        return None


def _get_next_available_nn(date_prefix: str) -> str:
    """Finds next available sequential number for a date prefix."""
    max_nn = -1

    dirs_to_check = [config.settings.raw_transcripts_dir, config.settings.resolved_archive_dir]

    for check_dir in dirs_to_check:
        if os.path.exists(check_dir):
            try:
                for filename in os.listdir(check_dir):
                    match = re.match(rf"^{re.escape(date_prefix)}(\d{{2}}).*\..*$", filename)
                    if match:
                        try:
                            nn = int(match.group(1))
                            if 0 <= nn <= 99 and nn > max_nn:
                                max_nn = nn
                        except ValueError:
                            pass
            except OSError:
                pass

    return f"{max_nn + 1:02d}"
