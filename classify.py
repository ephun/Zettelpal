# classify.py — LOCAL LLM VERSION (Option C: Truncate input before classification)

import json
import config
import utils


# How many characters of the transcript to use for classification
MAX_CLASSIFICATION_CHARS = 4000     # You can raise/lower this as needed


def classify_transcript(transcript_text: str) -> str:
    """
    Classify transcript into:
        type/note
        type/journal
        type/thought

    Using a local LLM via utils.local_llm_chat().
    This version truncates long transcripts to avoid overwhelming the model.
    """

    # --- STEP 1: Truncate input to safe size ---
    if len(transcript_text) > MAX_CLASSIFICATION_CHARS:
        truncated = transcript_text[:MAX_CLASSIFICATION_CHARS]
        truncated += "\n\n[Transcript truncated for classification]\n"
    else:
        truncated = transcript_text

    # --- STEP 2: Build prompt ---
    prompt = config.CLASSIFICATION_PROMPT_TEMPLATE.format(
        transcript_content=truncated
    )

    print("\n[CLASSIFY] Sending truncated transcript to local LLM...")
    print(f"[CLASSIFY] Input size: {len(truncated)} characters")

    # --- STEP 3: Call LLM ---
    response_text = utils.local_llm_chat(
        prompt,
        temperature=0.2,
        max_tokens=20
    )

    # Log raw response for debugging ------------------
    print("\n[CLASSIFY] RAW LLM OUTPUT:")
    print(repr(response_text))
    print("--------------------------------------------------")

    # --- STEP 4: Validate ---
    if not response_text:
        print("[CLASSIFY] ERROR: Empty response from LLM")
        return None

    cleaned = response_text.strip().lower()

    # Exact expected tags
    if cleaned.startswith("type/"):
        return cleaned

    # Fuzzy detection
    if "type/note" in cleaned:
        return "type/note"
    if "type/journal" in cleaned:
        return "type/journal"
    if "type/thought" in cleaned:
        return "type/thought"

    print(f"[CLASSIFY] Unrecognized classifier output: {cleaned}")
    return None
