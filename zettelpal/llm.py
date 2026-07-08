# llm.py - LLM backends (local OpenAI-compatible endpoint, Google Gemini)
# and the unified dispatcher the pipeline uses.

import json
import time

import requests

from zettelpal import config
from zettelpal.log import get_logger

log = get_logger(__name__)


def local_llm_chat(prompt: str, max_retries: int = 3, temperature: float = 0.3, max_tokens: int = None) -> str | None:
    """
    Sends a chat prompt to the local LLM using OpenAI-compatible API.
    Returns the text response or None on failure.
    """
    url = f"{config.settings.local_llm_base_url}/chat/completions"

    payload = {
        "model": config.settings.local_llm_model,
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
                log.warning("[LLM ERROR] Status %s: %s", resp.status_code, resp.text)
                time.sleep(2 ** (attempt + 1))
                continue

            data = resp.json()
            msg = data["choices"][0]["message"]["content"]
            return msg.strip()

        except requests.exceptions.Timeout:
            log.warning("[LLM TIMEOUT] Request timed out (attempt %d/%d)", attempt + 1, max_retries)
            time.sleep(2 ** (attempt + 1))
        except Exception as e:
            log.warning("[LLM Exception] %s", e)
            time.sleep(2 ** (attempt + 1))

    log.error("[LLM FAILURE] Local LLM did not respond after retries.")
    return None


def gemini_chat(prompt: str, max_retries: int = 3, temperature: float = 0.3, max_tokens: int = None) -> str | None:
    """
    Sends a chat prompt to Google Gemini API.
    Returns the text response or None on failure.
    """
    if not config.settings.gemini_api_key:
        log.error("[GEMINI ERROR] No API key set (GEMINI_API_KEY or GOOGLE_API_KEY)")
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.settings.gemini_model}:generateContent?key={config.settings.gemini_api_key}"

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
                log.warning("[GEMINI ERROR] Status %s: %s", resp.status_code, resp.text[:200])
                time.sleep(2 ** (attempt + 1))
                continue

            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if parts:
                    return parts[0].get("text", "").strip()

            log.error("[GEMINI ERROR] No content in response")
            return None

        except requests.exceptions.Timeout:
            log.warning("[GEMINI TIMEOUT] Request timed out (attempt %d/%d)", attempt + 1, max_retries)
            time.sleep(2 ** (attempt + 1))
        except Exception as e:
            log.warning("[GEMINI Exception] %s", e)
            time.sleep(2 ** (attempt + 1))

    log.error("[GEMINI FAILURE] Gemini API did not respond after retries.")
    return None


def llm_chat(prompt: str, max_retries: int = 3, temperature: float = 0.3, max_tokens: int = None) -> str | None:
    """
    Unified LLM chat function that routes to local or Gemini based on config.settings.llm_backend.
    """
    backend = config.settings.llm_backend.lower()

    if backend == "gemini":
        return gemini_chat(prompt, max_retries, temperature, max_tokens)
    else:
        return local_llm_chat(prompt, max_retries, temperature, max_tokens)


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
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()

    # Try exact JSON first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON array
    try:
        start = text.index("[")
        end = text.rindex("]") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        pass

    # Try to find JSON object
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        pass

    log.error("[JSON ERROR] Could not extract valid JSON from LLM response.")
    return None
