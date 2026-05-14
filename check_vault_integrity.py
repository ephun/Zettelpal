import argparse
import datetime
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

import config


WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[#|][^\]]*)?\]\]")
FRONTMATTER_KEY_RE = re.compile(r"^\s*([A-Za-z0-9_-]+)\s*:\s*(.*)\s*$")
LEGACY_LINK_LINE_RE = re.compile(r"^\s*(\[\[.*?\]\]\s*[,;-]?\s*)+$")


def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message):
    print(f"[{stamp()}] {message}", flush=True)


def clean_scalar(value):
    return str(value).strip().strip('"').strip("'")


def parse_created(value):
    if not value:
        return None

    value = clean_scalar(value)
    try:
        return datetime.datetime.fromisoformat(value)
    except ValueError:
        return None


def normalize_body(text):
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def split_frontmatter(lines):
    if not lines or lines[0].strip() != "---":
        return None, 0, "missing_frontmatter"

    frontmatter = {}
    current_list_key = None
    for index in range(1, len(lines)):
        line = lines[index]
        if line.strip() == "---":
            return frontmatter, index + 1, None

        key_match = FRONTMATTER_KEY_RE.match(line)
        if key_match:
            key, value = key_match.group(1), key_match.group(2).strip()
            current_list_key = key if value == "" else None
            frontmatter[key] = [] if value == "" else clean_scalar(value)
            continue

        if current_list_key and line.lstrip().startswith("- "):
            frontmatter[current_list_key].append(clean_scalar(line.lstrip()[2:]))

    return frontmatter, len(lines), "unterminated_frontmatter"


def split_body_and_generated_links(content_lines):
    block_start = getattr(config, "LINK_BLOCK_START", "<!-- zettelpal-links:start -->")
    block_end = getattr(config, "LINK_BLOCK_END", "<!-- zettelpal-links:end -->")

    start_index = -1
    end_index = -1
    for index, line in enumerate(content_lines):
        stripped = line.strip()
        if stripped == block_start:
            start_index = index
        elif start_index != -1 and stripped == block_end:
            end_index = index
            break

    if start_index != -1 and end_index != -1:
        return (
            "".join(content_lines[:start_index]).strip(),
            content_lines[start_index : end_index + 1],
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

    frontmatter, body_start, frontmatter_error = split_frontmatter(lines)
    content_lines = lines[body_start:]
    body, generated_link_lines, has_link_markers = split_body_and_generated_links(
        content_lines
    )

    relpath = os.path.relpath(filepath, vault_root).replace("\\", "/")
    filename = os.path.basename(filepath)
    stem = os.path.splitext(filename)[0]
    tags = []
    if isinstance(frontmatter, dict):
        raw_tags = frontmatter.get("tags") or []
        tags = raw_tags if isinstance(raw_tags, list) else [raw_tags]

    source = frontmatter.get("source") if isinstance(frontmatter, dict) else None
    is_zettelpal = bool(source) or "zettelpal" in tags
    title = None
    if isinstance(frontmatter, dict):
        title = frontmatter.get("title")
    if not title:
        h1_match = re.search(r"^\s*#\s+(.+?)\s*$", body, re.MULTILINE)
        title = h1_match.group(1).strip() if h1_match else stem

    generated_text = "".join(generated_link_lines)
    return {
        "path": filepath,
        "relpath": relpath,
        "filename": filename,
        "stem": stem,
        "frontmatter": frontmatter or {},
        "frontmatter_error": frontmatter_error,
        "body": body,
        "body_norm": normalize_body(body),
        "body_len": len(body.strip()),
        "generated_links": WIKILINK_RE.findall(generated_text),
        "all_links": WIKILINK_RE.findall("".join(lines)),
        "has_link_markers": has_link_markers,
        "source": source,
        "created": parse_created(
            frontmatter.get("created") or frontmatter.get("date")
            if isinstance(frontmatter, dict)
            else None
        ),
        "created_raw": (
            frontmatter.get("created") or frontmatter.get("date")
            if isinstance(frontmatter, dict)
            else None
        ),
        "title": title,
        "tags": tags,
        "is_zettelpal": is_zettelpal,
    }


def add_issue(issues, severity, code, relpath, detail):
    issues.append(
        {
            "severity": severity,
            "code": code,
            "file": relpath,
            "detail": detail,
        }
    )


def main():
    parser = argparse.ArgumentParser(
        description="Read-only integrity checker for the Zettelpal Obsidian vault."
    )
    parser.add_argument(
        "--vault",
        default=config.OBSIDIAN_VAULT_ROOT,
        help="Vault root to scan.",
    )
    parser.add_argument(
        "--json",
        default="vault_integrity_report.json",
        help="Path for the full JSON report.",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Exit non-zero for warnings as well as errors.",
    )
    args = parser.parse_args()

    vault_root = os.path.abspath(args.vault)
    log(f"Scanning vault: {vault_root}")

    notes = []
    for root, _, files in os.walk(vault_root):
        if os.path.basename(root) == ".obsidian":
            continue
        for filename in files:
            if filename.lower().endswith(".md"):
                filepath = os.path.join(root, filename)
                notes.append(read_note(filepath, vault_root))

    issues = []
    by_stem = defaultdict(list)
    by_norm_body = defaultdict(list)
    by_source = defaultdict(list)

    for note in notes:
        by_stem[note["stem"]].append(note)
        if note["is_zettelpal"] and note["body_norm"]:
            digest = hashlib.sha1(note["body_norm"].encode("utf-8")).hexdigest()
            by_norm_body[(note["source"], digest)].append(note)
        if note["is_zettelpal"] and note["source"]:
            by_source[note["source"]].append(note)

        if note["frontmatter_error"]:
            add_issue(
                issues,
                "error",
                note["frontmatter_error"],
                note["relpath"],
                "Frontmatter is missing or not closed.",
            )

        if note["is_zettelpal"]:
            required = ["id", "title", "created", "source", "tags"]
            for key in required:
                if key not in note["frontmatter"] or note["frontmatter"].get(key) in (
                    "",
                    [],
                    None,
                ):
                    add_issue(
                        issues,
                        "error",
                        "missing_required_frontmatter",
                        note["relpath"],
                        f"Missing required key: {key}",
                    )

            if not note["created"]:
                add_issue(
                    issues,
                    "error",
                    "invalid_created_timestamp",
                    note["relpath"],
                    f"Invalid created value: {note['created_raw']}",
                )

            if note["body_len"] == 0:
                add_issue(
                    issues,
                    "error",
                    "empty_body",
                    note["relpath"],
                    "Zettelpal note has no body text outside generated links.",
                )
            elif note["body_len"] < 40:
                add_issue(
                    issues,
                    "warning",
                    "very_short_body",
                    note["relpath"],
                    f"Body is only {note['body_len']} characters.",
                )

            if note["stem"] in note["generated_links"]:
                add_issue(
                    issues,
                    "error",
                    "self_link",
                    note["relpath"],
                    "Generated link block links to the note itself.",
                )

            if note["generated_links"] and not note["has_link_markers"]:
                add_issue(
                    issues,
                    "warning",
                    "legacy_unmarked_link_block",
                    note["relpath"],
                    "Generated links are not wrapped in Zettelpal link markers.",
                )

    for stem, group in by_stem.items():
        if len(group) > 1:
            for note in group:
                add_issue(
                    issues,
                    "error",
                    "duplicate_filename_stem",
                    note["relpath"],
                    f"{len(group)} notes share stem {stem!r}.",
                )

    for (_, _), group in by_norm_body.items():
        if len(group) > 1:
            files = [note["relpath"] for note in group]
            for note in group:
                add_issue(
                    issues,
                    "error",
                    "duplicate_body_same_source",
                    note["relpath"],
                    "Same normalized body text appears in: " + "; ".join(files),
                )

    all_stems = set(by_stem)
    for note in notes:
        for link_target in note["all_links"]:
            if link_target not in all_stems:
                add_issue(
                    issues,
                    "error",
                    "dead_wikilink",
                    note["relpath"],
                    f"Missing target: [[{link_target}]]",
                )

    for source, group in by_source.items():
        ordered = sorted(group, key=lambda n: (n["created"] or datetime.datetime.min, n["stem"]))
        stems = {note["stem"] for note in ordered}
        for index, note in enumerate(ordered):
            expected = set()
            if len(ordered) > 1:
                if index > 0:
                    expected.add(ordered[index - 1]["stem"])
                if index < len(ordered) - 1:
                    expected.add(ordered[index + 1]["stem"])

            actual_same_source = {
                link for link in note["generated_links"] if link in stems
            }
            missing = sorted(expected - actual_same_source)
            extra = sorted(actual_same_source - expected)
            if missing:
                add_issue(
                    issues,
                    "error",
                    "missing_chronological_link",
                    note["relpath"],
                    f"Source {source} missing chronological links: {missing}",
                )
            if extra:
                add_issue(
                    issues,
                    "warning",
                    "extra_same_source_generated_link",
                    note["relpath"],
                    f"Source {source} has non-neighbor generated links: {extra}",
                )

    counts = Counter(issue["code"] for issue in issues)
    severity_counts = Counter(issue["severity"] for issue in issues)
    report = {
        "vault": vault_root,
        "scanned_notes": len(notes),
        "zettelpal_notes": sum(1 for note in notes if note["is_zettelpal"]),
        "source_count": len(by_source),
        "severity_counts": dict(severity_counts),
        "issue_counts": dict(counts),
        "issues": issues,
    }

    with open(args.json, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    log(f"Scanned notes: {report['scanned_notes']}")
    log(f"Zettelpal notes: {report['zettelpal_notes']}")
    log(f"Sources: {report['source_count']}")
    log(f"Issues by severity: {dict(severity_counts)}")
    log(f"Issues by code: {dict(counts)}")
    log(f"Wrote report: {os.path.abspath(args.json)}")

    if severity_counts.get("error", 0) > 0:
        return 1
    if args.fail_on_warning and severity_counts.get("warning", 0) > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
