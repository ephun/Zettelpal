# segment.py — Local LLM segmentation for Zettelpal

import json
import math
import time
import config
import utils


# --------------------------------------------------------------------
#  Chunk large transcripts into smaller pieces for the local LLM
# --------------------------------------------------------------------

def chunk_transcript(text: str, max_chars: int = 8000):
    """
    Splits large transcripts into multiple chunks so the LLM can handle them.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks


# --------------------------------------------------------------------
#  Main segmentation function
# --------------------------------------------------------------------

def segment_and_title_transcript_local(transcript: str):
    """
    Uses the local LLM (OpenAI-compatible backend) to segment a transcript into JSON.
    Follows the template in config.SEGMENTATION_PROMPT_TEMPLATE.
    """

    all_segments = []
    chunks = chunk_transcript(transcript)

    print(f"[SEGMENTER] Transcript split into {len(chunks)} chunk(s).")

    for i, chunk in enumerate(chunks):
        print(f"[SEGMENTER] Processing chunk {i+1}/{len(chunks)}...")

        prompt = config.SEGMENTATION_PROMPT_TEMPLATE.format(
            transcript_chunk=chunk
        )

        llm_response = utils.local_llm_chat(
            prompt,
            temperature=config.SEGMENTATION_LLM_TEMPERATURE
        )

        if llm_response is None:
            print(f"[SEGMENTER ERROR] LLM did not return a response for chunk {i+1}.")
            return None

        parsed = utils.extract_json_from_text(llm_response)
        if parsed is None:
            print(f"[SEGMENTER ERROR] Failed to parse JSON for chunk {i+1}:")
            print(llm_response)
            return None

        if not isinstance(parsed, list):
            print(f"[SEGMENTER ERROR] LLM did not return JSON array for chunk {i+1}.")
            return None

        all_segments.extend(parsed)

        time.sleep(0.3)

    print(f"[SEGMENTER] Combined total segments: {len(all_segments)}")
    return all_segments
