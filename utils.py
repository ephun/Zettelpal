# utils.py

import os
import re
import json
import time
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import whisper

import config


# ============================================================
#  Local LLM Client (OpenAI-compatible)
# ============================================================

def local_llm_chat(prompt: str, max_retries: int = 3, temperature: float = 0.3, max_tokens: int = None):
    """
    Sends a chat prompt to the local LLM using OpenAI-compatible /v1/chat/completions.
    Returns the text response (string) or None on failure.
    """

    url = f"{config.LOCAL_LLM_BASE_URL}/chat/completions"

    payload = {
        "model": config.LOCAL_LLM_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
    }

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=60)
            if resp.status_code != 200:
                print(f"[LLM ERROR] Status {resp.status_code}: {resp.text}")
                time.sleep(2 ** (attempt + 1))
                continue

            data = resp.json()
            msg = data["choices"][0]["message"]["content"]
            return msg.strip()

        except Exception as e:
            print(f"[LLM Exception] {e}")
            time.sleep(2 ** (attempt + 1))

    print("[LLM FAILURE] Local LLM did not respond successfully after retries.")
    return None


# ============================================================
#  JSON Cleanup Utilities
# ============================================================

def extract_json_from_text(text: str):
    """
    Attempts to extract a valid JSON array or object from an LLM response.
    Returns a Python object or None.
    """

    if text is None:
        return None

    # Try exact JSON first
    try:
        return json.loads(text)
    except:
        pass

    # Attempt to locate first '[' and last ']'
    try:
        start = text.index("[")
        end = text.rindex("]") + 1
        cleaned = text[start:end]
        return json.loads(cleaned)
    except:
        pass

    # Attempt object `{}` mode
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        cleaned = text[start:end]
        return json.loads(cleaned)
    except:
        pass

    print("[JSON ERROR] Could not extract valid JSON from LLM response.")
    return None


# ============================================================
#  Embedding Model Loader and Wrapper
# ============================================================

_embedding_model_cache = None

def load_embedding_model():
    global _embedding_model_cache

    if _embedding_model_cache is not None:
        return _embedding_model_cache

    try:
        print("Loading SentenceTransformer embedding model:", config.LOCAL_EMBEDDING_MODEL)
        _embedding_model_cache = SentenceTransformer(config.LOCAL_EMBEDDING_MODEL)
        return _embedding_model_cache
    except Exception as e:
        print(f"Error loading embedding model '{config.LOCAL_EMBEDDING_MODEL}': {e}")
        return None


def get_embedding(text: str, model=None):
    """
    Returns a numpy embedding vector for the given text.
    """

    if model is None:
        model = load_embedding_model()
        if model is None:
            return None

    try:
        emb = model.encode([text])[0]
        return np.array(emb)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None



# ============================================================
#  Whisper Loader
# ============================================================

_whisper_cache = {}

def load_whisper_model(size: str = None):
    if size is None:
        size = config.LOCAL_WHISPER_MODEL_SIZE

    if size in _whisper_cache:
        return _whisper_cache[size]

    try:
        print(f"Loading Whisper model: {size}")
        model = whisper.load_model(size)
        _whisper_cache[size] = model
        return model
    except Exception as e:
        print(f"Error loading Whisper model '{size}': {e}")
        return None



# ============================================================
#  Filename Utilities
# ============================================================

def sanitize_filename(name: str) -> str:
    """
    Removes illegal filename characters.
    """
    return re.sub(r'[^A-Za-z0-9\- ]+', '', name).strip().replace(" ", "-")


def is_valid_zettelpal_filename(filename: str) -> bool:
    """
    Must match MMDDYYNN or MMDDYY.
    """
    stem = os.path.splitext(filename)[0]
    return re.fullmatch(r"\d{6,8}", stem) is not None


def rename_audio_file_to_zettelpal_format(filepath: str):
    """
    If audio file is not named MMDDYYNN.ext, try renaming using 
    file creation/modification time.
    """

    try:
        directory = os.path.dirname(filepath)
        ext = os.path.splitext(filepath)[1]
        timestamp = int(os.path.getmtime(filepath))
        dt = time.localtime(timestamp)
        new_stem = f"{dt.tm_mon:02d}{dt.tm_mday:02d}{str(dt.tm_year)[2:4]}"
        new_name = new_stem + ext
        new_path = os.path.join(directory, new_name)

        os.rename(filepath, new_path)
        return new_path
    except Exception as e:
        print(f"Error renaming '{filepath}': {e}")
        return None
