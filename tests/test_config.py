import os

from zettelpal import config


def test_derived_paths_follow_vault_root():
    settings = config.settings
    saved = settings.vault_root
    try:
        settings.vault_root = "C:/somevault"
        assert settings.embeddings_cache_file.replace("\\", "/") == \
            "C:/somevault/.obsidian/zettelpal_embeddings_cache.json"
        assert settings.resolved_archive_dir.replace("\\", "/") == "C:/somevault/archive"
        assert settings.clips_dir.replace("\\", "/") == "C:/somevault/archive/clips"
    finally:
        settings.vault_root = saved


def test_env_override(monkeypatch):
    monkeypatch.setenv("ZETTELPAL_LLM_BACKEND", "gemini")
    monkeypatch.setenv("ZETTELPAL_SIMILARITY_THRESHOLD", "0.42")
    s = config.Settings()
    assert s.llm_backend == "gemini"
    assert abs(s.similarity_threshold - 0.42) < 1e-9


def test_gemini_key_alias(monkeypatch):
    monkeypatch.delenv("ZETTELPAL_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ZETTELPAL_GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "from-google-env")
    assert config.Settings().gemini_api_key == "from-google-env"


def test_toml_override(tmp_path, monkeypatch):
    toml = tmp_path / "zp.toml"
    toml.write_text('llm_backend = "gemini"\nmax_semantic_links_per_note = 3\n', encoding="utf-8")

    class TomlSettings(config.Settings):
        model_config = dict(config.Settings.model_config, toml_file=str(toml))

    s = TomlSettings()
    assert s.llm_backend == "gemini"
    assert s.max_semantic_links_per_note == 3


def test_env_beats_toml(tmp_path, monkeypatch):
    toml = tmp_path / "zp.toml"
    toml.write_text('llm_backend = "gemini"\n', encoding="utf-8")
    monkeypatch.setenv("ZETTELPAL_LLM_BACKEND", "local")

    class TomlSettings(config.Settings):
        model_config = dict(config.Settings.model_config, toml_file=str(toml))

    assert TomlSettings().llm_backend == "local"
