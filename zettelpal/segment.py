# segment.py - Transcript Segmentation via the configured LLM backend

import difflib
import time

from zettelpal import config, llm


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


def align_segments_to_timestamps(
    llm_segments: list[dict], whisper_segments: list[dict]
) -> list[dict]:
    """
    Fuzzy-matches each LLM segment's content to Whisper timestamp segments
    to determine audio_start and audio_end times.

    Args:
        llm_segments: List of dicts with 'content' (and 'title', 'emoji').
        whisper_segments: List of dicts with 'start', 'end', 'text' from Whisper.

    Returns:
        The llm_segments list with 'audio_start' and 'audio_end' added to each.
    """
    if not whisper_segments:
        return llm_segments

    # Build a full transcript from whisper segments with character offset tracking
    full_text = ""
    char_to_segment_idx = []  # maps each char position to whisper segment index
    for idx, wseg in enumerate(whisper_segments):
        seg_text = wseg["text"]
        for _ in seg_text:
            char_to_segment_idx.append(idx)
        full_text += seg_text

    full_text_lower = full_text.lower()
    search_start = 0

    for llm_seg in llm_segments:
        content = llm_seg.get("content", "")
        if not content.strip():
            continue

        content_lower = content.lower()

        # Search in remaining transcript with generous padding
        search_end = min(search_start + len(content_lower) * 4, len(full_text_lower))
        region = full_text_lower[search_start:search_end]

        if not region:
            continue

        # Use get_matching_blocks to find all common subsequences
        matcher = difflib.SequenceMatcher(None, content_lower, region, autojunk=False)
        blocks = matcher.get_matching_blocks()

        # Calculate coverage: fraction of note text found in transcript
        total_matched = sum(b.size for b in blocks if b.size > 0)
        ratio = total_matched / len(content_lower) if content_lower else 0

        if ratio < 0.4:
            continue

        # Find span in the transcript from first to last real match
        real_blocks = [b for b in blocks if b.size > 0]
        if not real_blocks:
            continue

        region_match_start = real_blocks[0].b
        region_match_end = real_blocks[-1].b + real_blocks[-1].size

        abs_start = search_start + region_match_start
        abs_end = search_start + region_match_end

        # Map character range to whisper segment timestamps
        if abs_start < len(char_to_segment_idx) and abs_end <= len(char_to_segment_idx):
            start_wseg_idx = char_to_segment_idx[min(abs_start, len(char_to_segment_idx) - 1)]
            end_wseg_idx = char_to_segment_idx[min(abs_end - 1, len(char_to_segment_idx) - 1)]
            llm_seg["audio_start"] = max(0.0, whisper_segments[start_wseg_idx]["start"] - 0.5)
            llm_seg["audio_end"] = whisper_segments[end_wseg_idx]["end"] + 0.5

        # Advance search window past this match
        search_start = abs_end

    # Gap-fill: assign unmatched segments the time between their neighbors,
    # split proportionally by content character length.
    audio_duration = whisper_segments[-1]["end"] if whisper_segments else 0.0
    for i, seg in enumerate(llm_segments):
        if "audio_start" in seg:
            continue

        content = seg.get("content", "").strip()
        if not content:
            continue

        # Find preceding boundary
        prev_end = 0.0
        for j in range(i - 1, -1, -1):
            if "audio_end" in llm_segments[j]:
                prev_end = llm_segments[j]["audio_end"]
                break

        # Find following boundary
        next_start = audio_duration
        for j in range(i + 1, len(llm_segments)):
            if "audio_start" in llm_segments[j]:
                next_start = llm_segments[j]["audio_start"]
                break

        # Collect consecutive unmatched segments in this gap
        gap_start_idx = i
        while gap_start_idx > 0 and "audio_start" not in llm_segments[gap_start_idx - 1]:
            if not llm_segments[gap_start_idx - 1].get("content", "").strip():
                break
            gap_start_idx -= 1
        gap_end_idx = i
        while gap_end_idx < len(llm_segments) - 1 and "audio_start" not in llm_segments[gap_end_idx + 1]:
            if not llm_segments[gap_end_idx + 1].get("content", "").strip():
                break
            gap_end_idx += 1

        gap_indices = list(range(gap_start_idx, gap_end_idx + 1))
        char_lengths = [max(len(llm_segments[idx].get("content", "")), 1) for idx in gap_indices]
        total_chars = sum(char_lengths)

        gap_duration = next_start - prev_end
        if gap_duration <= 0:
            continue

        cursor = prev_end
        for k, idx in enumerate(gap_indices):
            proportion = char_lengths[k] / total_chars
            llm_segments[idx]["audio_start"] = cursor
            cursor += gap_duration * proportion
            llm_segments[idx]["audio_end"] = cursor

    return llm_segments


def segment_transcript(transcript: str, whisper_segments: list[dict] = None) -> list[dict] | None:
    """
    Uses the configured LLM to segment a transcript into titled segments.

    Args:
        transcript: The raw transcript text.
        whisper_segments: Optional list of Whisper segments with timestamps.

    Returns a list of dicts with 'title', 'emoji', and 'content' keys
    (plus 'audio_start'/'audio_end' if whisper_segments provided),
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

        llm_response = llm.llm_chat(
            prompt,
            temperature=config.SEGMENTATION_LLM_TEMPERATURE,
            max_tokens=config.SEGMENTATION_MAX_TOKENS
        )

        if llm_response is None:
            print(f"[SEGMENTER ERROR] LLM did not return a response for chunk {i + 1}.")
            return None

        parsed = llm.extract_json_from_text(llm_response)
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

    # Align segments to audio timestamps if available
    if all_segments and whisper_segments:
        print("[SEGMENTER] Aligning segments to audio timestamps...")
        all_segments = align_segments_to_timestamps(all_segments, whisper_segments)
        aligned = sum(1 for s in all_segments if "audio_start" in s)
        print(f"[SEGMENTER] Aligned {aligned}/{len(all_segments)} segments to timestamps.")

    return all_segments if all_segments else None
