# clip.py - Audio Clip Extraction using ffmpeg

import os
import subprocess

from zettelpal.log import get_logger

log = get_logger(__name__)


def extract_clip(audio_path: str, start_sec: float, end_sec: float, output_path: str) -> bool:
    """
    Extracts an audio clip from start_sec to end_sec using ffmpeg.

    Args:
        audio_path: Path to the source audio file.
        start_sec: Start time in seconds.
        end_sec: End time in seconds.
        output_path: Path for the output clip.

    Returns:
        True if successful, False otherwise.
    """
    duration = end_sec - start_sec
    if duration <= 0:
        log.warning(f"[CLIP] Invalid duration ({duration:.1f}s), skipping.")
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.3f}",
        "-i", audio_path,
        "-t", f"{duration:.3f}",
        "-c", "copy",
        output_path,
    ]

    try:
        subprocess.run(
            cmd, check=True, capture_output=True, text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        # Fallback: re-encode if stream copy fails (e.g., codec mismatch)
        cmd_reencode = [
            "ffmpeg", "-y",
            "-ss", f"{start_sec:.3f}",
            "-i", audio_path,
            "-t", f"{duration:.3f}",
            output_path,
        ]
        try:
            subprocess.run(
                cmd_reencode, check=True, capture_output=True, text=True
            )
            return True
        except subprocess.CalledProcessError as e2:
            log.error(f"[CLIP ERROR] ffmpeg failed: {e2.stderr[:200]}")
            return False
    except FileNotFoundError:
        log.error("[CLIP ERROR] ffmpeg not found. Please install ffmpeg.")
        return False


def extract_all_clips(
    audio_path: str,
    segments: list[dict],
    output_dir: str,
    recording_id: str,
) -> list[str | None]:
    """
    Extracts audio clips for all segments that have audio_start/audio_end.

    Args:
        audio_path: Path to the source audio file.
        segments: List of segment dicts (with optional audio_start/audio_end).
        output_dir: Directory to write clips into.
        recording_id: Recording ID for clip naming.

    Returns:
        List parallel to segments: clip path or None for each.
    """
    ext = os.path.splitext(audio_path)[1] or ".m4a"
    os.makedirs(output_dir, exist_ok=True)
    clip_paths = []

    for i, seg in enumerate(segments):
        start = seg.get("audio_start")
        end = seg.get("audio_end")

        if start is None or end is None:
            clip_paths.append(None)
            continue

        clip_filename = f"{recording_id}_{i + 1:02d}{ext}"
        clip_path = os.path.join(output_dir, clip_filename)

        log.info(f"[CLIP] Extracting {clip_filename} ({start:.1f}s - {end:.1f}s)")
        if extract_clip(audio_path, start, end, clip_path):
            clip_paths.append(clip_path)
        else:
            clip_paths.append(None)

    extracted = sum(1 for p in clip_paths if p is not None)
    log.info(f"[CLIP] Extracted {extracted}/{len(segments)} clips.")
    return clip_paths


if __name__ == "__main__":
    import argparse
    import sys

    from zettelpal.log import setup_console_logging

    setup_console_logging()

    parser = argparse.ArgumentParser(description="Extract an audio clip using ffmpeg.")
    parser.add_argument("audio_file", help="Source audio file.")
    parser.add_argument("start", type=float, help="Start time in seconds.")
    parser.add_argument("end", type=float, help="End time in seconds.")
    parser.add_argument("output", help="Output clip path.")
    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"ERROR: File not found: {args.audio_file}")
        sys.exit(1)

    success = extract_clip(args.audio_file, args.start, args.end, args.output)
    sys.exit(0 if success else 1)
