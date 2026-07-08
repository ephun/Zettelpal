import shutil
import wave

import pytest

from zettelpal import clip

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed"
)


def _make_wav(path, seconds=2):
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x00" * 16000 * seconds)


def test_extract_clip(tmp_path):
    source = tmp_path / "source.wav"
    out = tmp_path / "clip.wav"
    _make_wav(source)
    assert clip.extract_clip(str(source), 0.0, 1.0, str(out)) is True
    assert out.exists() and out.stat().st_size > 0


def test_extract_clip_invalid_duration(tmp_path):
    source = tmp_path / "source.wav"
    out = tmp_path / "clip.wav"
    _make_wav(source)
    assert clip.extract_clip(str(source), 1.0, 1.0, str(out)) is False
