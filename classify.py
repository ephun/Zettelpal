# classify.py - Transcript Classification using Local LLM

import config
import utils


# Maximum characters to send for classification (truncate long transcripts)
MAX_CLASSIFICATION_CHARS = 4000


def classify_transcript(transcript_text: str) -> str | None:
    """
    Classifies a transcript into one of:
        type/note
        type/journal
        type/thought

    Uses the local LLM via utils.llm_chat().
    Long transcripts are truncated to avoid overwhelming the model.

    Returns the classification tag or None on failure.
    """
    if not transcript_text or not transcript_text.strip():
        print("[CLASSIFY] Empty transcript provided.")
        return None

    # Truncate if necessary
    if len(transcript_text) > MAX_CLASSIFICATION_CHARS:
        truncated = transcript_text[:MAX_CLASSIFICATION_CHARS]
        truncated += "\n\n[Transcript truncated for classification]\n"
    else:
        truncated = transcript_text

    # Build prompt
    prompt = config.CLASSIFICATION_PROMPT_TEMPLATE.format(transcript_content=truncated)

    print(f"\n[CLASSIFY] Sending transcript to local LLM ({len(truncated)} chars)...")

    # Call LLM
    response_text = utils.llm_chat(
        prompt,
        temperature=0.2,
        max_tokens=config.CLASSIFICATION_MAX_TOKENS
    )

    # Log raw response for debugging
    print(f"[CLASSIFY] Raw response: {repr(response_text)}")

    # Validate response
    if not response_text:
        print("[CLASSIFY] ERROR: Empty response from LLM")
        return None

    cleaned = response_text.strip().lower()

    # Exact match
    if cleaned.startswith("type/"):
        return cleaned

    # Fuzzy detection
    if "type/note" in cleaned or cleaned == "note":
        return "type/note"
    if "type/journal" in cleaned or cleaned == "journal":
        return "type/journal"
    if "type/thought" in cleaned or cleaned == "thought":
        return "type/thought"

    print(f"[CLASSIFY] Unrecognized classifier output: {cleaned}")
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classify a transcript using local LLM.")
    parser.add_argument("input_file", help="Path to the transcript text file.")
    args = parser.parse_args()

    with open(args.input_file, 'r', encoding='utf-8') as f:
        transcript = f.read()

    result = classify_transcript(transcript)
    if result:
        print(f"Classification: {result}")
    else:
        print("Classification failed.")
