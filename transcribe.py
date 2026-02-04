# transcribe.py

import argparse
import os
import sys
import time # Use for performance timing if needed

import utils # Imports load_whisper_model

# removed imports: datetime (handled by shell), AUDIO_TRANSCRIPTS_DIR (output path arg)

def transcribe_audio_to_file(audio_filepath: str, output_filepath: str) -> str | None:
    """
    Transcribes an audio file using the local Whisper model and saves to a specified file.

    Args:
        audio_filepath: Path to the input audio file.
        output_filepath: Path where the raw transcript text will be saved.

    Returns:
        The absolute path to the output file if successful, None otherwise.
    """
    model = utils.load_whisper_model()
    if model is None:
        print("Error: Whisper model failed to load. Cannot transcribe.")
        return None

    print(f"Transcribing audio file: {os.path.basename(audio_filepath)} -> {output_filepath}...")
    try:
        # The whisper library should handle various audio formats.
        # Add error handling or format conversion (e.g., with pydub) if needed.
        # Example with pydub (requires pip install pydub ffmpeg-python):
        # from pydub import AudioSegment
        # try:
        #     audio = AudioSegment.from_file(audio_filepath)
        #     # Optional: Convert to a standard format like wav if needed, but whisper is usually flexible
        #     # audio.export("temp.wav", format="wav")
        #     # Use the path of the temporary or original file
        #     processed_audio_filepath = audio_filepath # Or "temp.wav"
        # except Exception as e:
        #     print(f"Error processing audio file with pydub: {e}")
        #     return None


        start_time = time.time()
        # Use the loaded local model to transcribe the audio file directly by path
        result = model.transcribe(audio_filepath, verbose=False) # verbose=False suppresses detailed progress
        raw_transcript = result["text"]
        end_time = time.time()
        print(f"Transcription complete ({end_time - start_time:.2f} seconds).")

        # Ensure directory exists before writing the file
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(raw_transcript)

        print(f"Raw transcript saved successfully to: {output_filepath}")
        return os.path.abspath(output_filepath) # Return absolute path

    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe an audio file using a local Whisper model.")
    parser.add_argument("audio_filepath", help="Path to the input audio file (e.g., MMDDYYNN.ext).")
    parser.add_argument("output_filepath", help="Path to save the raw transcript text file (e.g., path/to/raw_transcripts/MMDDYYNN.txt).")
    args = parser.parse_args()

    if not os.path.exists(args.audio_filepath):
        print(f"Error: Audio input file not found at {args.audio_filepath}")
        sys.exit(1) # Exit with error code

    # Call the transcription function
    output_file = transcribe_audio_to_file(args.audio_filepath, args.output_filepath)

    # Indicate success or failure for the shell script
    if output_file:
        print(f"SUCCESS:{output_file}") # Use marker to pass the output path
        sys.exit(0) # Exit with success code
    else:
        print("FAIL")
        sys.exit(1) # Exit with error code