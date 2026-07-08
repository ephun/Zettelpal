import os

from zettelpal import naming


def test_sanitize_preserves_case_and_strips_invalid():
    assert naming.sanitize_filename('My Note: "Draft"/v2') == "My-Note-Draftv2"


def test_sanitize_empty_is_untitled():
    assert naming.sanitize_filename("") == "Untitled"
    assert naming.sanitize_filename("///") == "Untitled"


def test_sanitize_truncates_to_100():
    assert len(naming.sanitize_filename("a" * 200)) == 100


def test_valid_zettelpal_filename():
    assert naming.is_valid_zettelpal_filename("07082601.mp3")
    assert naming.is_valid_zettelpal_filename("070826.wav")
    assert not naming.is_valid_zettelpal_filename("Voice Memo.mp3")
    assert not naming.is_valid_zettelpal_filename("1234.m4a")  # too short


def test_rename_to_zettelpal_format(sandbox):
    src = sandbox / "Voice Memo.mp3"
    src.write_bytes(b"x" * 100)
    result = naming.rename_audio_file_to_zettelpal_format(str(src))
    assert result is not None
    name = os.path.basename(result)
    assert naming.is_valid_zettelpal_filename(name)
    assert name.endswith(".mp3")
    assert not src.exists()
