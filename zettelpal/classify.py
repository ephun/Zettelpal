# classify.py - Transcript Classification via the configured LLM backend

from zettelpal import config, llm


# Maximum characters to send for classification (truncate long transcripts)
MAX_CLASSIFICATION_CHARS = 4000


def classify_transcript(transcript_text: str) -> str | None:
    """
    Classifies a transcript into one of:
        type/note
        type/journal
        type/thought

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

    prompt = config.CLASSIFICATION_PROMPT_TEMPLATE.format(transcript_content=truncated)

    print(f"\n[CLASSIFY] Sending transcript to LLM ({len(truncated)} chars)...")

    response_text = llm.llm_chat(
        prompt,
        temperature=0.2,
        max_tokens=config.CLASSIFICATION_MAX_TOKENS
    )

    print(f"[CLASSIFY] Raw response: {repr(response_text)}")

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
