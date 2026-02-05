# utils.py - Zettelpal Utilities
# All local operations: Whisper, SentenceTransformers, local LLM

import os
import re
import json
import time
import datetime
import requests
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import whisper

import config

# =============================================================================
# DEVICE DETECTION
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    try:
        _ = torch.randn(1).to(DEVICE)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"Warning: CUDA available but GPU not usable: {e}. Falling back to CPU.")
        DEVICE = "cpu"
else:
    print("Using CPU (GPU recommended for faster processing)")

# =============================================================================
# MODEL CACHING
# =============================================================================

_whisper_model = None
_embedding_model = None

def load_whisper_model(size: str = None):
    """Loads the local Whisper transcription model."""
    global _whisper_model
    if size is None:
        size = config.LOCAL_WHISPER_MODEL_SIZE

    if _whisper_model is None:
        print(f"Loading Whisper model: {size}...")
        try:
            _whisper_model = whisper.load_model(size, device=DEVICE)
            print("Whisper model loaded.")
        except Exception as e:
            print(f"Error loading Whisper model '{size}': {e}")
            _whisper_model = None
    return _whisper_model


def load_embedding_model():
    """Loads the local SentenceTransformer embedding model."""
    global _embedding_model
    if _embedding_model is None:
        print(f"Loading embedding model: {config.LOCAL_EMBEDDING_MODEL}...")
        try:
            _embedding_model = SentenceTransformer(config.LOCAL_EMBEDDING_MODEL, device=DEVICE)
            print("Embedding model loaded.")
        except Exception as e:
            print(f"Error loading embedding model '{config.LOCAL_EMBEDDING_MODEL}': {e}")
            _embedding_model = None
    return _embedding_model


def get_embedding(text: str, model=None) -> np.ndarray | None:
    """Generates an embedding vector for the given text."""
    if not text or not text.strip():
        return None

    embedding_model = model if model is not None else load_embedding_model()
    if embedding_model is None:
        print("Error: Embedding model not loaded.")
        return None

    try:
        embedding = embedding_model.encode([text.strip()])[0]
        return np.array(embedding)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


# =============================================================================
# LLM CLIENTS
# =============================================================================

def local_llm_chat(prompt: str, max_retries: int = 3, temperature: float = 0.3, max_tokens: int = None) -> str | None:
    """
    Sends a chat prompt to the local LLM using OpenAI-compatible API.
    Returns the text response or None on failure.
    """
    url = f"{config.LOCAL_LLM_BASE_URL}/chat/completions"

    payload = {
        "model": config.LOCAL_LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    for attempt in range(max_retries):
        try:
            # Large local models (120B+) can take 10+ minutes per request
            resp = requests.post(url, json=payload, timeout=900)
            if resp.status_code != 200:
                print(f"[LLM ERROR] Status {resp.status_code}: {resp.text}")
                time.sleep(2 ** (attempt + 1))
                continue

            data = resp.json()
            msg = data["choices"][0]["message"]["content"]
            return msg.strip()

        except requests.exceptions.Timeout:
            print(f"[LLM TIMEOUT] Request timed out (attempt {attempt + 1}/{max_retries})")
            time.sleep(2 ** (attempt + 1))
        except Exception as e:
            print(f"[LLM Exception] {e}")
            time.sleep(2 ** (attempt + 1))

    print("[LLM FAILURE] Local LLM did not respond after retries.")
    return None


def gemini_chat(prompt: str, max_retries: int = 3, temperature: float = 0.3, max_tokens: int = None) -> str | None:
    """
    Sends a chat prompt to Google Gemini API.
    Returns the text response or None on failure.
    """
    if not config.GOOGLE_API_KEY:
        print("[GEMINI ERROR] GOOGLE_API_KEY not set")
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.GEMINI_MODEL}:generateContent?key={config.GOOGLE_API_KEY}"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
        }
    }

    if max_tokens is not None:
        payload["generationConfig"]["maxOutputTokens"] = max_tokens

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=120)
            if resp.status_code != 200:
                print(f"[GEMINI ERROR] Status {resp.status_code}: {resp.text[:200]}")
                time.sleep(2 ** (attempt + 1))
                continue

            data = resp.json()
            # Extract text from Gemini response
            candidates = data.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if parts:
                    return parts[0].get("text", "").strip()

            print("[GEMINI ERROR] No content in response")
            return None

        except requests.exceptions.Timeout:
            print(f"[GEMINI TIMEOUT] Request timed out (attempt {attempt + 1}/{max_retries})")
            time.sleep(2 ** (attempt + 1))
        except Exception as e:
            print(f"[GEMINI Exception] {e}")
            time.sleep(2 ** (attempt + 1))

    print("[GEMINI FAILURE] Gemini API did not respond after retries.")
    return None


def llm_chat(prompt: str, max_retries: int = 3, temperature: float = 0.3, max_tokens: int = None) -> str | None:
    """
    Unified LLM chat function that routes to local or Gemini based on config.LLM_BACKEND.
    """
    backend = getattr(config, 'LLM_BACKEND', 'local').lower()

    if backend == "gemini":
        return gemini_chat(prompt, max_retries, temperature, max_tokens)
    else:
        return local_llm_chat(prompt, max_retries, temperature, max_tokens)


# =============================================================================
# JSON UTILITIES
# =============================================================================

def extract_json_from_text(text: str):
    """
    Attempts to extract a valid JSON array or object from LLM response.
    Returns a Python object or None.
    """
    if text is None:
        return None

    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        # Remove closing fence
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()

    # Try exact JSON first
    try:
        return json.loads(text)
    except:
        pass

    # Try to find JSON array
    try:
        start = text.index("[")
        end = text.rindex("]") + 1
        return json.loads(text[start:end])
    except:
        pass

    # Try to find JSON object
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except:
        pass

    print("[JSON ERROR] Could not extract valid JSON from LLM response.")
    return None


# =============================================================================
# EMBEDDING CACHE MANAGEMENT
# =============================================================================

def load_embedding_cache() -> dict:
    """Loads the embedding cache from JSON file."""
    if os.path.exists(config.EMBEDDINGS_CACHE_FILE):
        try:
            with open(config.EMBEDDINGS_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    print(f"Warning: Cache file is not a dictionary. Creating new one.")
                    return {}
                return data
        except json.JSONDecodeError as e:
            print(f"Warning: Could not decode embedding cache: {e}")
            return {}
        except Exception as e:
            print(f"Warning: Error loading embedding cache: {e}")
            return {}
    return {}


def save_embedding_cache(cache: dict):
    """Saves the embedding cache to JSON file."""
    os.makedirs(os.path.dirname(config.EMBEDDINGS_CACHE_FILE), exist_ok=True)
    try:
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, datetime.datetime):
                    return obj.isoformat()
                return json.JSONEncoder.default(self, obj)

        with open(config.EMBEDDINGS_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, cls=NpEncoder)
    except Exception as e:
        print(f"Error saving embedding cache: {e}")


def update_embedding_cache(filepath: str, embedding: np.ndarray):
    """Updates a single entry in the embedding cache."""
    cache = load_embedding_cache()
    relative_path = os.path.relpath(filepath, config.OBSIDIAN_VAULT_ROOT)
    cache[relative_path] = {
        "embedding": embedding.tolist() if embedding is not None else None,
        "mtime": os.path.getmtime(filepath) if os.path.exists(filepath) else None,
    }
    save_embedding_cache(cache)


# =============================================================================
# FILENAME UTILITIES
# =============================================================================

def sanitize_filename(title: str) -> str:
    """Sanitizes a string to be used as a filename."""
    if not title:
        return "untitled"

    # Remove illegal characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '', title)
    # Replace spaces with hyphens
    sanitized = sanitized.replace(' ', '-')
    # Remove multiple hyphens
    sanitized = re.sub(r'-+', '-', sanitized)
    # Strip hyphens and dots from ends
    sanitized = sanitized.strip(' -.').lower()
    # Limit length
    sanitized = sanitized[:200]

    return sanitized if sanitized else "untitled"


def is_valid_zettelpal_filename(filename: str) -> bool:
    """Checks if filename matches MMDDYYNN.ext format."""
    return re.match(r"^\d{6,8}\..+$", filename) is not None


def rename_audio_file_to_zettelpal_format(filepath: str) -> str | None:
    """
    Renames an audio file to MMDDYYNN.ext format based on modification time.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return None

    try:
        mtime = os.path.getmtime(filepath)
        dt = datetime.datetime.fromtimestamp(mtime)
        date_prefix = dt.strftime("%m%d%y")

        # Find next available NN
        next_nn = _get_next_available_nn(date_prefix)
        if len(next_nn) > 2:
            print(f"Error: Too many recordings for date {date_prefix}")
            return None

        _, ext = os.path.splitext(filepath)
        new_filename = f"{date_prefix}{next_nn}{ext.lower()}"
        new_filepath = os.path.join(os.path.dirname(filepath), new_filename)

        if os.path.exists(new_filepath):
            print(f"Error: Target file already exists: {new_filename}")
            return None

        os.rename(filepath, new_filepath)
        print(f"Renamed to: {new_filename}")
        return os.path.abspath(new_filepath)

    except Exception as e:
        print(f"Error renaming file: {e}")
        return None


def _get_next_available_nn(date_prefix: str) -> str:
    """Finds next available sequential number for a date prefix."""
    max_nn = -1

    dirs_to_check = [config.RAW_TRANSCRIPTS_INTERMEDIATE_DIR, config.ARCHIVE_DIR]

    for check_dir in dirs_to_check:
        if os.path.exists(check_dir):
            try:
                for filename in os.listdir(check_dir):
                    match = re.match(rf"^{re.escape(date_prefix)}(\d{{2}}).*\..*$", filename)
                    if match:
                        try:
                            nn = int(match.group(1))
                            if 0 <= nn <= 99 and nn > max_nn:
                                max_nn = nn
                        except ValueError:
                            pass
            except Exception:
                pass

    return f"{max_nn + 1:02d}"


# =============================================================================
# FILE UTILITIES
# =============================================================================

def find_all_markdown_files(directory: str) -> list[str]:
    """Recursively finds all .md files in a directory."""
    md_files = []
    if not os.path.exists(directory):
        return []

    cache_realpath = None
    if config.EMBEDDINGS_CACHE_FILE and os.path.exists(os.path.dirname(config.EMBEDDINGS_CACHE_FILE)):
        try:
            cache_realpath = os.path.realpath(config.EMBEDDINGS_CACHE_FILE)
        except:
            pass

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".md"):
                filepath = os.path.join(root, file)
                # Skip cache file
                if cache_realpath:
                    try:
                        if os.path.realpath(filepath) == cache_realpath:
                            continue
                    except:
                        pass
                md_files.append(filepath)

    return md_files


def extract_frontmatter_and_body(filepath: str) -> tuple[list[str], str, list[str]]:
    """
    Extracts frontmatter lines, body content, and link lines from a Markdown file.

    Returns:
        tuple: (frontmatter_lines, body_content_string, link_lines_list)
    """
    frontmatter_lines = []
    body_lines = []
    link_lines = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return [], "", []

    # Extract frontmatter
    frontmatter_end_index = 0
    if all_lines and all_lines[0].strip() == '---':
        frontmatter_lines.append(all_lines[0])
        for i in range(1, len(all_lines)):
            frontmatter_lines.append(all_lines[i])
            if all_lines[i].strip() == '---':
                frontmatter_end_index = i + 1
                break

    # Separate body and link lines
    content_lines = all_lines[frontmatter_end_index:]
    link_pattern = re.compile(r'^\s*(\[\[.*?\]\]\s*[,;-]?\s*)+$')

    link_start = -1
    for i, line in enumerate(content_lines):
        if line.strip() and link_pattern.match(line.strip()):
            link_start = i
            break

    if link_start != -1:
        body_lines = content_lines[:link_start]
        link_lines = content_lines[link_start:]
    else:
        body_lines = content_lines

    body_content = "".join(body_lines).strip()
    return frontmatter_lines, body_content, link_lines


def extract_title_from_filepath(filepath: str) -> str:
    """Extracts display title from an Obsidian note."""
    if not os.path.exists(filepath):
        return os.path.splitext(os.path.basename(filepath))[0]

    try:
        frontmatter_lines, body_content, _ = extract_frontmatter_and_body(filepath)

        # Try H1 title
        h1_match = re.search(r"^\s*#\s+(.+)", body_content, re.MULTILINE)
        if h1_match:
            title = h1_match.group(1).strip()
            title = re.sub(r'\s+[#^][\w-]+$', '', title)
            if title:
                return title

        # Try frontmatter title
        if frontmatter_lines and frontmatter_lines[0].strip() == '---':
            title_pattern = re.compile(r"^\s*title:\s*[\"']?(.*?)\s*[\"']?$", re.IGNORECASE)
            for line in frontmatter_lines[1:-1]:
                match = title_pattern.match(line)
                if match and match.group(1).strip():
                    return match.group(1).strip()

        # Fallback to filename
        return os.path.splitext(os.path.basename(filepath))[0]

    except Exception:
        return os.path.splitext(os.path.basename(filepath))[0]


def extract_source_from_frontmatter(frontmatter_lines: list[str]) -> str | None:
    """Extracts the source/recording ID from frontmatter."""
    if not frontmatter_lines or frontmatter_lines[0].strip() != '---':
        return None

    source_pattern = re.compile(
        rf"^\s*{re.escape(config.SOURCE_METADATA_KEY)}\s*:\s*[\"']?(.*?)\s*[\"']?$",
        re.IGNORECASE
    )

    for line in frontmatter_lines[1:-1]:
        match = source_pattern.match(line)
        if match:
            value = match.group(1).strip()
            return value if value else None

    return None


def extract_created_timestamp_str_from_frontmatter(frontmatter_lines: list[str]) -> str | None:
    """Extracts the 'created' or 'date' timestamp string from frontmatter."""
    if not frontmatter_lines or frontmatter_lines[0].strip() != '---':
        return None

    for key in ['created', 'date']:
        pattern = re.compile(rf"^\s*{key}:\s*[\"']?(.*?)\s*[\"']?$", re.IGNORECASE)
        for line in frontmatter_lines[1:-1]:
            match = pattern.match(line)
            if match:
                value = match.group(1).strip()
                if value:
                    return value

    return None


def parse_created_timestamp_str(timestamp_str: str) -> datetime.datetime | None:
    """Parses a timestamp string into a datetime object."""
    if not timestamp_str:
        return None

    try:
        return datetime.datetime.fromisoformat(timestamp_str)
    except (ValueError, TypeError):
        # Try common date formats
        for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d']:
            try:
                return datetime.datetime.strptime(timestamp_str, fmt)
            except:
                continue
        return None
