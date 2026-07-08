# notes.py - Canonical Obsidian note parsing for the vault.
#
# This is the single implementation of frontmatter/body/link-block parsing.
# Two access styles are provided because callers need different shapes:
#   - read_note(): structured dict view (integrity checks, reporting)
#   - extract_frontmatter_and_body(): raw-lines view (linking rewrites files
#     and must preserve the original frontmatter lines byte-for-byte)

import datetime
import os
import re

from zettelpal import config
from zettelpal.log import get_logger

log = get_logger(__name__)

FRONTMATTER_KEY_RE = re.compile(r"^\s*([A-Za-z0-9_-]+)\s*:\s*(.*)\s*$")
WIKILINK_RE = re.compile(r"\[\[(.*?)\]\]")
LEGACY_LINK_LINE_RE = re.compile(r"^\s*(\[\[.*?\]\]\s*[,;-]?\s*)+$")


def clean_scalar(value) -> str:
    return str(value).strip().strip('"').strip("'")


def parse_created(value) -> datetime.datetime | None:
    """Parses a frontmatter timestamp value (ISO or common date formats)."""
    if not value:
        return None

    value = clean_scalar(value)
    try:
        return datetime.datetime.fromisoformat(value)
    except (ValueError, TypeError):
        for fmt in ('%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d'):
            try:
                return datetime.datetime.strptime(value, fmt)
            except ValueError:
                continue
        return None


def normalize_body(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def extract_wikilink_targets(text: str) -> list[str]:
    targets = []
    for match in WIKILINK_RE.findall(text):
        target = re.split(r"[#|]", match, maxsplit=1)[0].strip()
        if target:
            targets.append(target)
    return targets


def split_frontmatter(lines: list[str]) -> tuple[dict, int, str | None]:
    """
    Parses YAML-ish frontmatter from a note's lines.

    Returns:
        (frontmatter_dict, body_start_index, error_code_or_None)
    """
    if not lines or lines[0].strip() != "---":
        return {}, 0, "missing_frontmatter"

    frontmatter = {}
    current_list_key = None
    for index in range(1, len(lines)):
        line = lines[index]
        if line.strip() == "---":
            return frontmatter, index + 1, None

        match = FRONTMATTER_KEY_RE.match(line)
        if match:
            key, value = match.group(1), match.group(2).strip()
            current_list_key = key if value == "" else None
            frontmatter[key] = [] if value == "" else clean_scalar(value)
            continue

        if current_list_key and line.lstrip().startswith("- "):
            frontmatter[current_list_key].append(clean_scalar(line.lstrip()[2:]))

    return frontmatter, len(lines), "unterminated_frontmatter"


def split_body_and_generated_links(content_lines: list[str]) -> tuple[str, list[str], bool]:
    """
    Separates note body from the generated link block. Prefers explicit
    markers; falls back to the legacy trailing-wikilink heuristic.

    Returns:
        (body_text, link_lines, has_link_markers)
    """
    block_start = config.LINK_BLOCK_START
    block_end = config.LINK_BLOCK_END

    marker_start = -1
    marker_end = -1
    for index, line in enumerate(content_lines):
        stripped = line.strip()
        if stripped == block_start:
            marker_start = index
        elif marker_start != -1 and stripped == block_end:
            marker_end = index
            break

    if marker_start != -1 and marker_end != -1:
        return (
            "".join(content_lines[:marker_start]).strip(),
            content_lines[marker_start: marker_end + 1],
            True,
        )

    link_start = -1
    for index, line in enumerate(content_lines):
        if line.strip() and LEGACY_LINK_LINE_RE.match(line.strip()):
            link_start = index
            break

    if link_start != -1:
        return (
            "".join(content_lines[:link_start]).strip(),
            content_lines[link_start:],
            False,
        )

    return "".join(content_lines).strip(), [], False


def read_note(filepath: str, vault_root: str) -> dict:
    """Reads a note into a structured dict (see keys below)."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as file:
        lines = file.readlines()

    frontmatter, body_start, frontmatter_error = split_frontmatter(lines)
    body, generated_link_lines, has_link_markers = split_body_and_generated_links(
        lines[body_start:]
    )

    relpath = os.path.relpath(filepath, vault_root).replace("\\", "/")
    filename = os.path.basename(filepath)
    stem = os.path.splitext(filename)[0]

    raw_tags = frontmatter.get("tags") or []
    tags = raw_tags if isinstance(raw_tags, list) else [raw_tags]

    source = frontmatter.get("source")
    created_raw = frontmatter.get("created") or frontmatter.get("date")
    is_zettelpal = bool(source) or "zettelpal" in tags

    title = frontmatter.get("title")
    if not title:
        h1_match = re.search(r"^\s*#\s+(.+?)\s*$", body, re.MULTILINE)
        title = h1_match.group(1).strip() if h1_match else stem

    return {
        "path": filepath,
        "relpath": relpath,
        "filename": filename,
        "stem": stem,
        "frontmatter": frontmatter,
        "frontmatter_error": frontmatter_error,
        "body": body,
        "body_norm": normalize_body(body),
        "body_len": len(body.strip()),
        "generated_links": extract_wikilink_targets("".join(generated_link_lines)),
        "all_links": extract_wikilink_targets("".join(lines)),
        "has_link_markers": has_link_markers,
        "source": source,
        "created": parse_created(created_raw),
        "created_raw": created_raw,
        "title": title,
        "tags": tags,
        "is_zettelpal": is_zettelpal,
    }


def find_all_markdown_files(directory: str) -> list[str]:
    """Recursively finds all .md files in a directory, skipping excluded dirs."""
    md_files = []
    if not os.path.exists(directory):
        return []

    excluded_dirs = set(config.settings.excluded_vault_dirs)

    cache_realpath = None
    if config.settings.embeddings_cache_file and os.path.exists(os.path.dirname(config.settings.embeddings_cache_file)):
        try:
            cache_realpath = os.path.realpath(config.settings.embeddings_cache_file)
        except OSError:
            pass

    for root, dirs, files in os.walk(directory):
        dirs[:] = [
            d for d in dirs
            if d not in excluded_dirs and not d.startswith(".trash")
        ]

        for file in files:
            if file.endswith(".md"):
                filepath = os.path.join(root, file)
                if cache_realpath:
                    try:
                        if os.path.realpath(filepath) == cache_realpath:
                            continue
                    except OSError:
                        pass
                md_files.append(filepath)

    return md_files


# =============================================================================
# Raw-lines view (used by linking, which rewrites note files and must keep
# the original frontmatter lines intact)
# =============================================================================

def extract_frontmatter_and_body(filepath: str) -> tuple[list[str], str, list[str]]:
    """
    Extracts raw frontmatter lines, body content, and generated link lines.

    Returns:
        tuple: (frontmatter_lines, body_content_string, link_lines_list)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
    except OSError as e:
        log.error(f"Error reading file {filepath}: {e}")
        return [], "", []

    frontmatter_lines = []
    frontmatter_end_index = 0
    if all_lines and all_lines[0].strip() == '---':
        frontmatter_lines.append(all_lines[0])
        for i in range(1, len(all_lines)):
            frontmatter_lines.append(all_lines[i])
            if all_lines[i].strip() == '---':
                frontmatter_end_index = i + 1
                break

    body_content, link_lines, _ = split_body_and_generated_links(
        all_lines[frontmatter_end_index:]
    )
    return frontmatter_lines, body_content, link_lines


def extract_title_from_filepath(filepath: str) -> str:
    """Extracts display title from an Obsidian note (H1, then frontmatter, then filename)."""
    if not os.path.exists(filepath):
        return os.path.splitext(os.path.basename(filepath))[0]

    try:
        frontmatter_lines, body_content, _ = extract_frontmatter_and_body(filepath)

        h1_match = re.search(r"^\s*#\s+(.+)", body_content, re.MULTILINE)
        if h1_match:
            title = h1_match.group(1).strip()
            title = re.sub(r'\s+[#^][\w-]+$', '', title)
            if title:
                return title

        if frontmatter_lines and frontmatter_lines[0].strip() == '---':
            title_pattern = re.compile(r"^\s*title:\s*[\"']?(.*?)\s*[\"']?$", re.IGNORECASE)
            for line in frontmatter_lines[1:-1]:
                match = title_pattern.match(line)
                if match and match.group(1).strip():
                    return match.group(1).strip()

        return os.path.splitext(os.path.basename(filepath))[0]

    except Exception:
        return os.path.splitext(os.path.basename(filepath))[0]


def extract_source_from_frontmatter(frontmatter_lines: list[str]) -> str | None:
    """Extracts the source/recording ID from raw frontmatter lines."""
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
    """Extracts the 'created' or 'date' timestamp string from raw frontmatter lines."""
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
