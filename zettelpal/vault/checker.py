# checker.py - Read-only full-vault integrity linter with JSON report.

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict

from zettelpal import config
from zettelpal.vault.notes import read_note


def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message):
    print(f"[{stamp()}] {message}", flush=True)


def add_issue(issues, severity, code, relpath, detail):
    issues.append(
        {
            "severity": severity,
            "code": code,
            "file": relpath,
            "detail": detail,
        }
    )


def run_check(
    vault_root: str,
    json_path: str = "vault_integrity_report.json",
    fail_on_warning: bool = False,
) -> int:
    """Scan the vault, write the JSON report, and return an exit code."""
    vault_root = os.path.abspath(vault_root)
    log(f"Scanning vault: {vault_root}")

    notes = []
    excluded_dirs = set(config.settings.excluded_vault_dirs)
    for root, dirs, files in os.walk(vault_root):
        dirs[:] = [
            dirname for dirname in dirs
            if dirname not in excluded_dirs
            and "trash" not in dirname.lower()
            and "quarantine" not in dirname.lower()
        ]
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
            if missing:
                add_issue(
                    issues,
                    "error",
                    "missing_chronological_link",
                    note["relpath"],
                    f"Source {source} missing chronological links: {missing}",
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

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    log(f"Scanned notes: {report['scanned_notes']}")
    log(f"Zettelpal notes: {report['zettelpal_notes']}")
    log(f"Sources: {report['source_count']}")
    log(f"Issues by severity: {dict(severity_counts)}")
    log(f"Issues by code: {dict(counts)}")
    log(f"Wrote report: {os.path.abspath(json_path)}")

    if severity_counts.get("error", 0) > 0:
        return 1
    if fail_on_warning and severity_counts.get("warning", 0) > 0:
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Read-only integrity checker for the Zettelpal Obsidian vault."
    )
    parser.add_argument(
        "--vault",
        default=config.settings.vault_root,
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
    return run_check(args.vault, args.json, args.fail_on_warning)


if __name__ == "__main__":
    sys.exit(main())
