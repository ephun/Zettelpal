# config.py - Zettelpal Configuration
# Privacy-focused: Uses only local models (Whisper, SentenceTransformers, and local LLM)

import os

# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================

# Your Obsidian vault root directory
OBSIDIAN_VAULT_ROOT = "//PHUNRAID/ephunism"

# Zettelpal working directory (for intermediate files)
ZETTELPAL_ROOT = os.path.expanduser(
    "C:/Users/ethan/Documents/R & Python Projects & Vibecoding/Zettelpal"
)

# Subdirectory within the vault where notes will be created (empty string = vault root)
NOTES_SUBDIRECTORY_IN_VAULT = ""

# Intermediate file directories
INTERMEDIATE_DIR = "zettelpal_intermediate"
RAW_TRANSCRIPTS_INTERMEDIATE_DIR = os.path.join(ZETTELPAL_ROOT, INTERMEDIATE_DIR, "raw_transcripts")
SEGMENTED_OUTPUT_INTERMEDIATE_DIR = os.path.join(ZETTELPAL_ROOT, INTERMEDIATE_DIR, "segmented_output")

# Archive directory for processed files
ARCHIVE_DIR = os.path.join("//PHUNRAID/ephunism/archive")

# Clips directory for extracted audio segments
CLIPS_DIR = os.path.join(ARCHIVE_DIR, "clips")

# =============================================================================
# LLM BACKEND SELECTION
# =============================================================================

# Options: "local" or "gemini"
LLM_BACKEND = "gemini"

# =============================================================================
# LOCAL MODEL CONFIGURATION (Privacy-Focused)
# =============================================================================

# Whisper model for local transcription
# Options: tiny, base, small, medium, large, large-v2, large-v3
LOCAL_WHISPER_MODEL_SIZE = "medium"

# SentenceTransformers model for local embeddings
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Local LLM configuration (OpenAI-compatible API)
LOCAL_LLM_BASE_URL = "http://100.93.31.109:8888/v1"
LOCAL_LLM_MODEL = "gpt-oss:120b"

# =============================================================================
# GEMINI CONFIGURATION (Cloud)
# =============================================================================

# Set via environment variable GEMINI_API_KEY or GOOGLE_API_KEY
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"

# =============================================================================
# LLM PARAMETERS (shared)
# =============================================================================

SEGMENTATION_LLM_TEMPERATURE = 0.3
SEGMENTATION_MAX_TOKENS = 8192
CLASSIFICATION_MAX_TOKENS = 100  # Needs extra tokens for model reasoning

# =============================================================================
# EMBEDDINGS & LINKING CONFIGURATION
# =============================================================================

# Cache file for storing note embeddings
EMBEDDINGS_CACHE_FILE = os.path.join(
    OBSIDIAN_VAULT_ROOT, ".obsidian", "zettelpal_embeddings_cache.json"
)

# Cosine similarity threshold for semantic linking (0.0 to 1.0)
SIMILARITY_THRESHOLD = 0.6

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
