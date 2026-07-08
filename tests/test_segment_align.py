from zettelpal import segment


def test_align_segments_to_timestamps():
    whisper = [
        {"start": 0.0, "end": 2.0, "text": "The first part about mornings. "},
        {"start": 2.0, "end": 4.0, "text": "The second part about evenings. "},
    ]
    llm_segments = [
        {"title": "Mornings", "emoji": "", "content": "The first part about mornings."},
        {"title": "Evenings", "emoji": "", "content": "The second part about evenings."},
    ]
    aligned = segment.align_segments_to_timestamps(llm_segments, whisper)

    assert "audio_start" in aligned[0] and "audio_end" in aligned[0]
    assert aligned[0]["audio_start"] < aligned[0]["audio_end"]
    # First segment should map near the start of the audio.
    assert aligned[0]["audio_start"] <= 0.5
    assert "audio_start" in aligned[1]
    assert aligned[1]["audio_end"] >= aligned[1]["audio_start"]


def test_align_no_whisper_segments_is_passthrough():
    llm_segments = [{"title": "x", "emoji": "", "content": "some content"}]
    assert segment.align_segments_to_timestamps(llm_segments, []) == llm_segments


def test_chunk_transcript_splits_large_text():
    text = ". ".join(f"sentence number {i}" for i in range(500))
    chunks = segment.chunk_transcript(text, max_chars=200)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
