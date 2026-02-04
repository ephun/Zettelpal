import os

# --- Root Directories (adjust as needed) ---
OBSIDIAN_VAULT_ROOT = "C:/Users/ethan/Documents/ephunism"

ZETTELPAL_ROOT = os.path.expanduser(
    "C:/Users/ethan/Documents/R & Python Projects & Vibecoding/Zettelpal"
)

# --- Subdirectories ---
NOTES_SUBDIRECTORY_IN_VAULT = ""

INTERMEDIATE_DIR = "zettelpal_intermediate"
RAW_TRANSCRIPTS_INTERMEDIATE_DIR = os.path.join(ZETTELPAL_ROOT, INTERMEDIATE_DIR, "raw_transcripts")
SEGMENTED_OUTPUT_INTERMEDIATE_DIR = os.path.join(ZETTELPAL_ROOT, INTERMEDIATE_DIR, "segmented_output")

# --- Archive directories ---
ARCHIVE_DIR = os.path.join("C:/Users/ethan/Documents/ephunism/archive")
AUDIO_ARCHIVE_DIR = os.path.join(ARCHIVE_DIR, "audio")
TRANSCRIPTS_ARCHIVE_DIR = os.path.join(ARCHIVE_DIR, "transcripts")

# --- Whisper Configuration ---
LOCAL_WHISPER_MODEL_SIZE = "medium"  # Options: tiny, base, small, medium, large, etc.

# --- Embedding Model (SentenceTransformers) ---
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Local LLM (your 120B GPT-OSS) ---
LOCAL_LLM_BASE_URL = "http://100.93.31.109:8888/v1"
LOCAL_LLM_MODEL = "gpt-oss:120b"

# --- Prompt Templates ---

SEGMENTATION_PROMPT_TEMPLATE = """
Your task is to break the following stream-of-consciousness text into distinct segments based on topic shifts or natural breaks in thought.

For EACH segment, you must provide a concise title, a single, relevant unicode emoji, and then the exact text from the transcript that corresponds to that segment.

Minor edits to remove filler language, correct grammar/spelling, italicize titles of works, fix quotations, or capitalize proper nouns are permissable.

The output MUST be a JSON array where each element is an object with "title", "emoji", and "content" keys.

Do NOT include any text before '[' and do NOT include anything after the closing ']'.

Example:
[
  {
    "title": "Thought Title 1",
    "emoji": "🤔",
    "content": "Exact text for segment 1."
  }
]

Here is the transcript to segment:
{transcript_chunk}
"""

GRANULAR_TAGGING_PROMPT_TEMPLATE = """
Your task is to provide a comprehensive list of tags for the following text. The tags should be detailed, specific, and relevant.

Tags should be lowercase and hyphenated, e.g., person/alan-turing, idea/machine-learning.

Return ONLY a JSON list of strings, no commentary.

Example:
["person/susan", "idea/personal-growth"]

Here is the text to tag:
{text_content}
"""

CLASSIFICATION_PROMPT_TEMPLATE = """
Your task is to categorize the following transcript based on its overall content and purpose.

Respond with ONLY ONE of these exact tags:
type/note
type/journal
type/thought

No punctuation. No explanations.

Transcript:
{transcript_content}
"""

# --- Embeddings Cache ---
EMBEDDINGS_CACHE_FILE = os.path.join(
    OBSIDIAN_VAULT_ROOT, ".obsidian", "zettelpal_embeddings_cache.json"
)

# --- Linking Configuration ---
SIMILARITY_THRESHOLD = 0.6  # default cosine-similarity cutoff

SOURCE_METADATA_KEY = "source"
SOURCE_METADATA_VALUE_FORMAT = "{recording_id}"

LINK_SEPARATOR = ", "
CHRONOLOGICAL_LINK_TEMPLATE = "[[{link_target_filename_stem}|{display_title}]]"
SEMANTIC_LINK_TEMPLATE = "[[{link_target_filename_stem}|{display_title}]]"
