# transcribe.py - Audio Transcription using Local Whisper

import os
import time
import utils


def transcribe_audio_to_file(audio_filepath: str, output_filepath: str) -> str | None:
    """
    Transcribes an audio file using the local Whisper model.

    Args:
        audio_filepath: Path to the input audio file.
        output_filepath: Path where the transcript will be saved.

    Returns:
        The absolute path to the output file if successful, None otherwise.
    """
    model = utils.load_whisper_model()
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

        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(raw_transcript)

        print(f"Saved to: {output_filepath}")
        return os.path.abspath(output_filepath)

    except Exception as e:
        print(f"ERROR: Transcription failed: {e}")
        return None


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Transcribe audio using local Whisper.")
    parser.add_argument("audio_file", help="Path to the audio file.")
    parser.add_argument("output_file", help="Path for the transcript output.")
    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"ERROR: File not found: {args.audio_file}")
        sys.exit(1)

    result = transcribe_audio_to_file(args.audio_file, args.output_file)
    sys.exit(0 if result else 1)
