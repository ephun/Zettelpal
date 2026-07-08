from zettelpal import llm


def test_extract_plain_json_array():
    assert llm.extract_json_from_text('[{"a": 1}]') == [{"a": 1}]


def test_extract_json_from_code_fence():
    text = '```json\n[{"title": "x"}]\n```'
    assert llm.extract_json_from_text(text) == [{"title": "x"}]


def test_extract_json_with_surrounding_prose():
    text = 'Here is the result: [1, 2, 3] hope that helps!'
    assert llm.extract_json_from_text(text) == [1, 2, 3]


def test_extract_json_object():
    text = 'preamble {"k": "v"} trailer'
    assert llm.extract_json_from_text(text) == {"k": "v"}


def test_extract_json_none_on_garbage():
    assert llm.extract_json_from_text("no json here at all") is None
    assert llm.extract_json_from_text(None) is None


def test_llm_chat_routes_to_backend(monkeypatch):
    from zettelpal import config

    monkeypatch.setattr(llm, "local_llm_chat", lambda *a, **k: "LOCAL")
    monkeypatch.setattr(llm, "gemini_chat", lambda *a, **k: "GEMINI")

    saved = config.settings.llm_backend
    try:
        config.settings.llm_backend = "local"
        assert llm.llm_chat("hi") == "LOCAL"
        config.settings.llm_backend = "gemini"
        assert llm.llm_chat("hi") == "GEMINI"
    finally:
        config.settings.llm_backend = saved
