import json
import os
import re
import datetime
from collections import defaultdict

import config
import vault_integrity


WIKILINK_RE = re.compile(r"\[\[(.*?)\]\]")
BLOCK_START = getattr(config, "LINK_BLOCK_START", "<!-- zettelpal-links:start -->")
BLOCK_END = getattr(config, "LINK_BLOCK_END", "<!-- zettelpal-links:end -->")


def normalize_stem(value):
    value = re.split(r"[#|]", value, maxsplit=1)[0].strip()
    value = re.sub(r"[^a-z0-9\s-]", " ", value.lower())
    value = value.replace("-", " ")
    return re.sub(r"\s+", " ", value).strip()


def load_report():
    with open("vault_integrity_report.json", "r", encoding="utf-8") as file:
        return json.load(file)


def load_cache(vault_root):
    with open(config.EMBEDDINGS_CACHE_FILE, "r", encoding="utf-8") as file:
        cache = json.load(file)

    notes = []
    for relpath, entry in cache.items():
        if not isinstance(entry, dict) or not relpath.lower().endswith(".md"):
            continue
        parts = relpath.replace("\\", "/").split("/")
        if any(part in getattr(config, "EXCLUDED_VAULT_DIRS", set()) for part in parts):
            continue
        if any("trash" in part.lower() or "quarantine" in part.lower() for part in parts):
            continue
        filepath = os.path.join(vault_root, relpath.replace("/", os.sep))
        if not os.path.exists(filepath):
            continue
        notes.append(
            {
                "relpath": relpath.replace("\\", "/"),
                "filepath": filepath,
                "stem": os.path.splitext(os.path.basename(relpath))[0],
                "source": entry.get("source"),
                "created": vault_integrity.parse_created(entry.get("created")),
                "title": entry.get("display_title"),
            }
        )
    return notes


def repair_dead_links(vault_root, report, cache_notes):
    stems = {note["stem"] for note in cache_notes}
    stems_by_norm = defaultdict(list)
    for note in cache_notes:
        stems_by_norm[normalize_stem(note["stem"])].append(note["stem"])

    dead_files = sorted(
        {
            issue["file"]
            for issue in report["issues"]
            if issue["code"] == "dead_wikilink"
        }
    )
    changed = []
    replacements = []
    removed_self_links = []

    for relpath in dead_files:
        filepath = os.path.join(vault_root, relpath.replace("/", os.sep))
        if not os.path.exists(filepath):
            continue
        current_stem = os.path.splitext(os.path.basename(relpath))[0]
        current_norm = normalize_stem(current_stem)
        with open(filepath, "r", encoding="utf-8", errors="replace") as file:
            original = file.read()

        def replace_link(match):
            inner = match.group(1)
            target = re.split(r"[#|]", inner, maxsplit=1)[0].strip()
            if target in stems:
                return match.group(0)

            target_norm = normalize_stem(target)
            candidates = stems_by_norm.get(target_norm, [])
            if not candidates:
                return match.group(0)

            if target_norm == current_norm:
                removed_self_links.append((relpath, target))
                return ""

            replacement = sorted(candidates)[0]
            replacements.append((relpath, target, replacement))
            return "[[" + inner.replace(target, replacement, 1) + "]]"

        updated = WIKILINK_RE.sub(replace_link, original)
        updated = re.sub(r"\n{3,}", "\n\n", updated)
        if updated != original:
            with open(filepath, "w", encoding="utf-8", newline="") as file:
                file.write(updated)
            changed.append(relpath)

    return changed, replacements, removed_self_links


def ensure_chronological_links(vault_root, cache_notes, sources):
    by_source = defaultdict(list)
    for note in cache_notes:
        if note["source"] in sources:
            by_source[note["source"]].append(note)

    changed = []
    for source, notes in by_source.items():
        ordered = sorted(
            notes,
            key=lambda note: (note["created"] or datetime.datetime.min, note["stem"]),
        )
        for index, note in enumerate(ordered):
            expected = []
            if index > 0:
                expected.append(ordered[index - 1])
            if index < len(ordered) - 1:
                expected.append(ordered[index + 1])
            if not expected:
                continue

            filepath = note["filepath"]
            with open(filepath, "r", encoding="utf-8", errors="replace") as file:
                original = file.read()

            existing_targets = set(vault_integrity.extract_wikilink_targets(original))
            links_to_add = []
            for neighbor in expected:
                if neighbor["stem"] not in existing_targets:
                    display = neighbor["title"] or neighbor["stem"]
                    links_to_add.append(f"[[{neighbor['stem']}|{display}]]")
            if not links_to_add:
                continue

            insertion = "\n".join(links_to_add) + "\n"
            if BLOCK_START in original and BLOCK_END in original:
                updated = original.replace(BLOCK_START + "\n", BLOCK_START + "\n" + insertion, 1)
            else:
                updated = original.rstrip() + f"\n\n{BLOCK_START}\n{insertion}{BLOCK_END}\n"

            with open(filepath, "w", encoding="utf-8", newline="") as file:
                file.write(updated)
            changed.append(note["relpath"])

    return changed


def main():
    vault_root = vault_integrity.canonical_dir(config.OBSIDIAN_VAULT_ROOT)
    report = load_report()
    cache_notes = load_cache(vault_root)

    dead_changed, replacements, removed_self_links = repair_dead_links(
        vault_root, report, cache_notes
    )
    chrono_changed = ensure_chronological_links(
        vault_root, cache_notes, {"05082601", "09062502"}
    )

    print(
        json.dumps(
            {
                "cache_notes": len(cache_notes),
                "dead_link_files_changed": len(dead_changed),
                "dead_link_replacements": len(replacements),
                "self_links_removed": len(removed_self_links),
                "chronological_files_changed": len(chrono_changed),
                "dead_link_files": dead_changed,
                "replacements": replacements,
                "removed_self_links": removed_self_links,
                "chronological_files": chrono_changed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
