import datetime
import hashlib
import json
import os
import re
import shutil
from collections import defaultdict

import numpy as np

import config


FRONTMATTER_KEY_RE = re.compile(r"^\s*([A-Za-z0-9_-]+)\s*:\s*(.*)\s*$")
WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[#|][^\]]*)?\]\]")
LEGACY_LINK_LINE_RE = re.compile(r"^\s*(\[\[.*?\]\]\s*[,;-]?\s*)+$")


def canonical_dir(path):
    path = os.path.normpath(path)
    if not path.endswith(os.sep):
        path += os.sep
    return os.path.abspath(path)


def is_relative_to(child_path, parent_dir):
    child = os.path.normcase(os.path.abspath(child_path))
    parent = os.path.normcase(canonical_dir(parent_dir))
    return child.startswith(parent)


def clean_scalar(value):
    return str(value).strip().strip('"').strip("'")


def parse_created(value):
    if not value:
        return None
    try:
        return datetime.datetime.fromisoformat(clean_scalar(value))
    except ValueError:
        return None


def normalize_body(text):
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _split_frontmatter(lines):
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


def _split_body_and_generated_links(content_lines):
    block_start = getattr(config, "LINK_BLOCK_START", "<!-- zettelpal-links:start -->")
    block_end = getattr(config, "LINK_BLOCK_END", "<!-- zettelpal-links:end -->")

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
            content_lines[marker_start : marker_end + 1],
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


def read_note(filepath, vault_root):
    with open(filepath, "r", encoding="utf-8", errors="replace") as file:
        lines = file.readlines()

    frontmatter, body_start, frontmatter_error = _split_frontmatter(lines)
    body, generated_link_lines, has_link_markers = _split_body_and_generated_links(
        lines[body_start:]
    )
    tags = frontmatter.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]

    relpath = os.path.relpath(filepath, vault_root).replace("\\", "/")
    stem = os.path.splitext(os.path.basename(filepath))[0]
    return {
        "path": filepath,
        "relpath": relpath,
        "stem": stem,
        "frontmatter": frontmatter,
        "frontmatter_error": frontmatter_error,
        "source": frontmatter.get("source"),
        "created": parse_created(frontmatter.get("created") or frontmatter.get("date")),
        "body": body,
        "body_norm": normalize_body(body),
        "body_len": len(body.strip()),
        "generated_links": WIKILINK_RE.findall("".join(generated_link_lines)),
        "has_link_markers": has_link_markers,
        "tags": tags,
    }


def _load_embedding_cache():
    if not os.path.exists(config.EMBEDDINGS_CACHE_FILE):
        return {}
    try:
        with open(config.EMBEDDINGS_CACHE_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_embedding_cache(cache):
    os.makedirs(os.path.dirname(config.EMBEDDINGS_CACHE_FILE), exist_ok=True)
    with open(config.EMBEDDINGS_CACHE_FILE, "w", encoding="utf-8") as file:
        json.dump(cache, file, indent=2)


def _embedding_for_note(note, cache):
    entry = cache.get(note["relpath"]) or cache.get(note["relpath"].replace("/", "\\"))
    if not isinstance(entry, dict) or not isinstance(entry.get("embedding"), list):
        return None

    embedding = np.asarray(entry["embedding"], dtype=np.float32)
    norm = np.linalg.norm(embedding)
    if norm <= 0:
        return None
    return embedding / norm


def _iter_source_notes(vault_root, recording_id):
    notes = []
    for root, _, files in os.walk(vault_root):
        if os.path.basename(root) == ".obsidian":
            continue
        for filename in files:
            if not filename.lower().endswith(".md"):
                continue
            filepath = os.path.join(root, filename)
            note = read_note(filepath, vault_root)
            if note["source"] == recording_id:
                notes.append(note)
    return notes


class _DisjointSet:
    def __init__(self):
        self.parents = {}

    def find(self, item):
        self.parents.setdefault(item, item)
        if self.parents[item] != item:
            self.parents[item] = self.find(self.parents[item])
        return self.parents[item]

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parents[right_root] = left_root


def quarantine_duplicate_notes_for_source(
    vault_root,
    recording_id,
    similarity_threshold=0.92,
):
    """Move later duplicate notes for a recording into a quarantine folder."""
    vault_root = canonical_dir(vault_root)
    source_notes = _iter_source_notes(vault_root, recording_id)
    if len(source_notes) < 2:
        return []

    cache = _load_embedding_cache()
    dsu = _DisjointSet()

    exact_groups = defaultdict(list)
    for note in source_notes:
        if note["body_norm"] and len(note["body_norm"]) >= 100:
            digest = hashlib.sha1(note["body_norm"].encode("utf-8")).hexdigest()
            exact_groups[digest].append(note)

    for group in exact_groups.values():
        if len(group) > 1:
            first = group[0]["relpath"]
            for note in group[1:]:
                dsu.union(first, note["relpath"])

    embedded_notes = []
    for note in source_notes:
        embedding = _embedding_for_note(note, cache)
        if embedding is not None:
            embedded_notes.append((note, embedding))

    if len(embedded_notes) > 1:
        embeddings = np.stack([embedding for _, embedding in embedded_notes])
        similarities = embeddings @ embeddings.T
        for row in range(len(embedded_notes)):
            for col in range(row + 1, len(embedded_notes)):
                if similarities[row, col] >= similarity_threshold:
                    dsu.union(
                        embedded_notes[row][0]["relpath"],
                        embedded_notes[col][0]["relpath"],
                    )

    components = defaultdict(list)
    for relpath in dsu.parents:
        components[dsu.find(relpath)].append(relpath)

    notes_by_relpath = {note["relpath"]: note for note in source_notes}
    to_quarantine = []
    for relpaths in components.values():
        if len(relpaths) < 2:
            continue
        notes = [notes_by_relpath[relpath] for relpath in relpaths]
        notes.sort(key=lambda note: (note["created"] or datetime.datetime.max, note["relpath"]))
        to_quarantine.extend(notes[1:])

    if not to_quarantine:
        return []

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    quarantine_root = os.path.join(
        config.ZETTELPAL_ROOT,
        "zettelpal_quarantine",
        f"pipeline_duplicates_{recording_id}_{timestamp}",
    )
    os.makedirs(quarantine_root, exist_ok=True)

    moved = []
    for note in sorted(to_quarantine, key=lambda item: item["relpath"]):
        source_path = os.path.abspath(note["path"])
        if not is_relative_to(source_path, vault_root):
            raise RuntimeError(f"Refusing to move outside vault: {source_path}")

        target_path = os.path.abspath(
            os.path.join(quarantine_root, note["relpath"].replace("/", os.sep))
        )
        if not is_relative_to(target_path, quarantine_root):
            raise RuntimeError(f"Refusing to move outside quarantine: {target_path}")

        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.move(source_path, target_path)
        cache.pop(note["relpath"], None)
        cache.pop(note["relpath"].replace("/", "\\"), None)
        moved.append(
            {
                "relative_path": note["relpath"],
                "quarantine_path": target_path,
            }
        )

    _save_embedding_cache(cache)
    manifest_path = os.path.join(quarantine_root, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "recording_id": recording_id,
                "similarity_threshold": similarity_threshold,
                "moved": moved,
            },
            file,
            indent=2,
        )

    return moved


def validate_source_integrity(vault_root, recording_id):
    """Return source-scoped integrity issues without modifying the vault."""
    vault_root = canonical_dir(vault_root)
    source_notes = _iter_source_notes(vault_root, recording_id)
    issues = []

    for note in source_notes:
        if note["frontmatter_error"]:
            issues.append((note["relpath"], note["frontmatter_error"]))
        for key in ["id", "title", "created", "source", "tags"]:
            if note["frontmatter"].get(key) in ("", [], None):
                issues.append((note["relpath"], f"missing_{key}"))
        if not note["created"]:
            issues.append((note["relpath"], "invalid_created"))
        if note["body_len"] == 0:
            issues.append((note["relpath"], "empty_body"))
        if note["stem"] in note["generated_links"]:
            issues.append((note["relpath"], "self_link"))

    return issues
