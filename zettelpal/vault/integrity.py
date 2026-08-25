# integrity.py - Vault integrity: duplicate quarantine and source validation.

import datetime
import difflib
import hashlib
import json
import os
import shutil
from collections import defaultdict

from zettelpal import config
from zettelpal.vault import cache as vault_cache
from zettelpal.vault.notes import find_all_markdown_files, read_note


def canonical_dir(path: str) -> str:
    path = os.path.normpath(path)
    if not path.endswith(os.sep):
        path += os.sep
    return os.path.abspath(path)


def is_relative_to(child_path: str, parent_dir: str) -> bool:
    child = os.path.normcase(os.path.abspath(child_path))
    parent = os.path.normcase(canonical_dir(parent_dir))
    return child.startswith(parent)


def _iter_source_notes(vault_root: str, recording_id: str) -> list[dict]:
    """Every readable note in the vault that came from this recording.

    Uses the shared vault walk so excluded directories (.git, .obsidian,
    .trash, the quarantine folder, generated insights) are pruned here exactly
    as they are everywhere else.
    """
    notes = []
    for filepath in find_all_markdown_files(vault_root):
        note = read_note(filepath, vault_root)
        if note["read_error"]:
            continue
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


def text_duplicate_score(left: str, right: str) -> float:
    if not left or not right:
        return 0.0

    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    if len(shorter) < 100:
        return 0.0

    if shorter in longer:
        return 1.0

    matcher = difflib.SequenceMatcher(None, shorter, longer, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    containment = matched / len(shorter)
    ratio = matcher.ratio()
    return max(containment, ratio)


def quarantine_duplicate_notes_for_source(
    vault_root: str,
    recording_id: str,
    text_similarity_threshold: float = 0.90,
) -> list[dict]:
    """Move later duplicate notes for a recording into a quarantine folder."""
    vault_root = canonical_dir(vault_root)
    source_notes = _iter_source_notes(vault_root, recording_id)
    if len(source_notes) < 2:
        return []

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

    for row in range(len(source_notes)):
        for col in range(row + 1, len(source_notes)):
            left = source_notes[row]
            right = source_notes[col]
            score = text_duplicate_score(left["body_norm"], right["body_norm"])
            if score >= text_similarity_threshold:
                dsu.union(left["relpath"], right["relpath"])

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

    cache = vault_cache.load_embedding_cache()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    quarantine_root = os.path.join(
        config.settings.quarantine_dir,
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

    vault_cache.save_embedding_cache(cache)
    manifest_path = os.path.join(quarantine_root, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "recording_id": recording_id,
                "text_similarity_threshold": text_similarity_threshold,
                "moved": moved,
            },
            file,
            indent=2,
        )

    return moved


def validate_source_integrity(vault_root: str, recording_id: str) -> list[tuple[str, str]]:
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
