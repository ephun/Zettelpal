# segment.py - Transcript Segmentation using Local LLM

import time
import config
import utils


def chunk_transcript(text: str, max_chars: int = 4000) -> list[str]:
    """Splits large transcripts into chunks for LLM processing."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings near the chunk boundary
            for sep in ['. ', '! ', '? ', '\n']:
                last_sep = text[start:end].rfind(sep)
                if last_sep > max_chars * 0.7:  # At least 70% through
                    end = start + last_sep + len(sep)
                    break
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if c]


def segment_transcript(transcript: str) -> list[dict] | None:
    """
    Uses the local LLM to segment a transcript into titled segments.

    Returns a list of dicts with 'title', 'emoji', and 'content' keys,
    or None on failure.
    """
    if not transcript or not transcript.strip():
        print("[SEGMENTER] Empty transcript provided.")
        return None

    all_segments = []
    chunks = chunk_transcript(transcript)

    print(f"[SEGMENTER] Transcript split into {len(chunks)} chunk(s).")

    for i, chunk in enumerate(chunks):
        print(f"[SEGMENTER] Processing chunk {i + 1}/{len(chunks)}...")

        prompt = config.SEGMENTATION_PROMPT_TEMPLATE.format(transcript_chunk=chunk)

        llm_response = utils.llm_chat(
            prompt,
            temperature=config.SEGMENTATION_LLM_TEMPERATURE,
            max_tokens=config.SEGMENTATION_MAX_TOKENS
        )

        if llm_response is None:
            print(f"[SEGMENTER ERROR] LLM did not return a response for chunk {i + 1}.")
            return None

        parsed = utils.extract_json_from_text(llm_response)
        if parsed is None:
            print(f"[SEGMENTER ERROR] Failed to parse JSON for chunk {i + 1}:")
            print(llm_response[:500] + "..." if len(llm_response) > 500 else llm_response)
            return None

        if not isinstance(parsed, list):
            print(f"[SEGMENTER ERROR] LLM did not return JSON array for chunk {i + 1}.")
            return None

        # Validate each segment has required fields
        for seg in parsed:
            if not isinstance(seg, dict):
                continue
            if 'title' not in seg:
                seg['title'] = 'Untitled Segment'
            if 'emoji' not in seg:
                seg['emoji'] = ''
            if 'content' not in seg:
                seg['content'] = ''

        all_segments.extend(parsed)

        # Small delay between chunks to avoid overwhelming the LLM
        if i < len(chunks) - 1:
            time.sleep(0.3)

    print(f"[SEGMENTER] Total segments created: {len(all_segments)}")
    return all_segments if all_segments else None


# Legacy alias for backward compatibility
segment_and_title_transcript_local = segment_transcript
segment_and_title_transcript_gemini = segment_transcript


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Segment a transcript using local LLM.")
    parser.add_argument("input_file", help="Path to the transcript text file.")
    parser.add_argument("-o", "--output", help="Output JSON file (optional)")
    args = parser.parse_args()

    with open(args.input_file, 'r', encoding='utf-8') as f:
        transcript = f.read()

    segments = segment_transcript(transcript)

    if segments:
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(segments, f, indent=2)
            print(f"Segments saved to {args.output}")
        else:
            print(json.dumps(segments, indent=2))
    else:
        print("Segmentation failed.")
