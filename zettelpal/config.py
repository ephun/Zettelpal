# config.py - Zettelpal Configuration
# Privacy-focused: defaults to local models (Whisper, SentenceTransformers, local LLM).
#
# Settings load in priority order:
#   1. Environment variables prefixed ZETTELPAL_ (e.g. ZETTELPAL_VAULT_ROOT)
#   2. zettelpal.toml next to the app root (repo root, or the exe when frozen)
#   3. Built-in defaults below
# See zettelpal.example.toml for a documented template.

import os
import sys
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


def app_root() -> str:
    """Directory the app runs from: the repo root in a checkout, or the
    executable's directory in a frozen (PyInstaller) build, where __file__
    points at a temp extraction dir."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


CONFIG_FILE = os.path.join(app_root(), "zettelpal.toml")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ZETTELPAL_",
        toml_file=CONFIG_FILE,
        extra="ignore",
    )

    # --- Paths ---
    # Obsidian vault root directory.
    vault_root: str = "//PHUNRAID/ephunism"
    # Subdirectory within the vault where notes are created ("" = vault root).
    notes_subdirectory: str = ""
    # Where intermediate files and the quarantine live.
    data_dir: str = Field(default_factory=app_root)
    # Where processed recordings are archived ("" = <vault_root>/archive).
    archive_dir: str = ""
    # Subdirectory (within the vault) where generated insight notes are written.
    # The vault root stays your own entries only; insights live here and are
    # excluded from linking, scanning, and integrity checks.
    insights_subdirectory: str = "Insights"

    # --- LLM backend ---
    # "local" keeps everything on your machines; "gemini" sends transcript
    # text to Google. Nothing leaves the network unless you opt in here.
    llm_backend: Literal["local", "gemini"] = "local"
    # Whisper model for transcription: tiny/base/small/medium/large/large-v3.
    whisper_model_size: str = "medium"
    # SentenceTransformers model for embeddings.
    embedding_model: str = "all-MiniLM-L6-v2"
    # Local LLM (any OpenAI-compatible API).
    local_llm_base_url: str = "http://100.93.31.109:8888/v1"
    local_llm_model: str = "gpt-oss:120b"
    # Gemini (cloud) backend.
    gemini_model: str = "gemini-2.0-flash"
    gemini_api_key: str = Field(
        "",
        validation_alias=AliasChoices(
            "ZETTELPAL_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"
        ),
    )

    # --- LLM parameters ---
    segmentation_temperature: float = 0.3
    segmentation_max_tokens: int = 8192
    # Needs headroom for model reasoning before the one-line answer.
    classification_max_tokens: int = 100

    # --- Embeddings & linking ---
    # Cosine similarity threshold for semantic linking (0.0 to 1.0).
    similarity_threshold: float = Field(0.6, ge=0.0, le=1.0)
    # Chronological prev/next links do not count toward this limit.
    max_semantic_links_per_note: int = 10
    # Vault directories that are never scanned or rewritten.
    excluded_vault_dirs: set[str] = {
        ".git",
        ".obsidian",
        ".trash",
        "zettelpal_quarantine",
    }

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )

    # --- Derived paths (follow the fields above, including runtime changes) ---

    @property
    def notes_dir(self) -> str:
        if self.notes_subdirectory:
            return os.path.join(self.vault_root, self.notes_subdirectory)
        return self.vault_root

    @property
    def resolved_archive_dir(self) -> str:
        return self.archive_dir or os.path.join(self.vault_root, "archive")

    @property
    def insights_dir(self) -> str:
        return os.path.join(self.vault_root, self.insights_subdirectory)

    @property
    def clips_dir(self) -> str:
        return os.path.join(self.resolved_archive_dir, "clips")

    @property
    def raw_transcripts_dir(self) -> str:
        return os.path.join(self.data_dir, "zettelpal_intermediate", "raw_transcripts")

    @property
    def segmented_output_dir(self) -> str:
        return os.path.join(self.data_dir, "zettelpal_intermediate", "segmented_output")

    @property
    def quarantine_dir(self) -> str:
        return os.path.join(self.data_dir, "zettelpal_quarantine")

    @property
    def embeddings_cache_file(self) -> str:
        return os.path.join(
            self.vault_root, ".obsidian", "zettelpal_embeddings_cache.json"
        )

    @property
    def last_threshold_file(self) -> str:
        return os.path.join(
            self.vault_root, ".obsidian", "zettelpal_last_threshold.json"
        )


settings = Settings()

# User-editable settings that the GUI exposes and persists to zettelpal.toml.
# The Gemini API key is deliberately excluded — keep it in the environment.
USER_EDITABLE_KEYS = (
    "vault_root",
    "notes_subdirectory",
    "insights_subdirectory",
    "llm_backend",
    "local_llm_base_url",
    "local_llm_model",
    "gemini_model",
    "whisper_model_size",
    "embedding_model",
    "similarity_threshold",
    "max_semantic_links_per_note",
)


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def write_user_settings(values: dict) -> str:
    """Persist the user-editable settings to zettelpal.toml and apply them to
    the live `settings` object. Returns the path written."""
    lines = [
        "# Zettelpal settings - written by the app's Settings tab.",
        "# Environment variables prefixed ZETTELPAL_ still override these.",
        "",
    ]
    for key in USER_EDITABLE_KEYS:
        if key not in values:
            continue
        value = values[key]
        setattr(settings, key, value)  # apply live
        lines.append(f"{key} = {_toml_value(value)}")
    text = "\n".join(lines) + "\n"
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write(text)
    return CONFIG_FILE

# =============================================================================
# INTERNAL CONSTANTS (not user configuration)
# =============================================================================

# Metadata key for source recording ID
SOURCE_METADATA_KEY = "source"

# Link formatting
LINK_SEPARATOR = ", "
CHRONOLOGICAL_LINK_TEMPLATE = "[[{link_target_filename_stem}|{display_title}]]"
SEMANTIC_LINK_TEMPLATE = "[[{link_target_filename_stem}|{display_title}]]"

# Generated link block markers. Only content between these markers is replaced.
LINK_BLOCK_START = "<!-- zettelpal-links:start -->"
LINK_BLOCK_END = "<!-- zettelpal-links:end -->"

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

SEGMENTATION_PROMPT_TEMPLATE = """Your task is to break the following stream-of-consciousness text into distinct segments based on topic shifts or natural breaks in thought.

For EACH segment, you must provide a concise title, a single relevant unicode emoji, and then the exact text from the transcript that corresponds to that segment.

Minor edits to remove filler language, correct grammar/spelling, italicize titles of works, fix quotations, or capitalize proper nouns are permissible.

The output MUST be a JSON array where each element is an object with "title", "emoji", and "content" keys.

Do NOT include any text before '[' and do NOT include anything after the closing ']'.

Example:
[
  {{
    "title": "Morning Reflections",
    "emoji": "sunrise",
    "content": "Exact text for segment 1."
  }}
]

Here is the transcript to segment:
{transcript_chunk}"""

CLASSIFICATION_PROMPT_TEMPLATE = """Your task is to categorize the following transcript based on its overall content and purpose.

Respond with ONLY ONE of these exact tags:
type/note
type/journal
type/thought

No punctuation. No explanations.

Transcript:
{transcript_content}"""

GRANULAR_TAGGING_PROMPT_TEMPLATE = """Your task is to provide a comprehensive list of tags for the following text. The tags should be detailed, specific, and relevant.

Tags should be lowercase and hyphenated, e.g., person/alan-turing, idea/machine-learning.

Return ONLY a JSON list of strings, no commentary.

There are only four valid tag categories: "person", "place", "thing", and "idea".

Example:
["person/susan", "idea/personal-growth"]

You may NOT nest tags further ("person/teacher/susan" or "place/utah/salt-lake-city" are *INVALID* tags)

Here is the text to tag:
{text_content}"""
