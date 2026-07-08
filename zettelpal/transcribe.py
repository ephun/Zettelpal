# transcribe.py - Audio Transcription using Local Whisper

import json
import os
import time

from zettelpal import models


def transcribe_audio_to_file(audio_filepath: str, output_filepath: str) -> tuple[str, str] | None:
    """
    Transcribes an audio file using the local Whisper model.

    Args:
        audio_filepath: Path to the input audio file.
        output_filepath: Path where the transcript text will be saved.

    Returns:
        A tuple of (text_path, json_path) if successful, None otherwise.
        The JSON sidecar contains Whisper's segment-level timestamps.
    """
    model = models.load_whisper_model()
    if model is None:
        print("ERROR: Whisper model failed to load.")
        return None

    print(f"Transcribing: {os.path.basename(audio_filepath)}")

    try:
        start_time = time.time()
        result = model.transcribe(audio_filepath, verbose=False)
        raw_transcript = result["text"]
        elapsed = time.time() - start_time

        print(f"Transcription complete ({elapsed:.1f}s)")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

        # Save plain text transcript
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(raw_transcript)

        # Save timestamps JSON sidecar
        json_path = os.path.splitext(output_filepath)[0] + ".json"
        whisper_segments = [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
            for seg in result.get("segments", [])
        ]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(whisper_segments, f, indent=2)

        print(f"Saved to: {output_filepath}")
        print(f"Timestamps saved to: {json_path}")
        return (os.path.abspath(output_filepath), os.path.abspath(json_path))

    except Exception as e:
        print(f"ERROR: Transcription failed: {e}")
        return None
