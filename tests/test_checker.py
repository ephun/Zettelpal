import json

from zettelpal import config
from zettelpal.vault import checker

FRONTMATTER = """---
id: 20260101120000
title: "{title}"
created: 2026-01-01T12:00:00
source: "01012601"
tags:
  - zettelpal
---
# {title}

A note body with enough text in it to be meaningful.
"""


def _write(path, title):
    path.write_text(FRONTMATTER.format(title=title), encoding="utf-8")
    return path


def test_run_check_scans_only_user_entries(sandbox, tmp_path):
    """The linter must use the same walk as the rest of the pipeline: generated
    insight notes and excluded directories are not user entries."""
    vault = sandbox / "vault"
    _write(vault / "Entry - 20260101120000.md", "Entry")

    insights = vault / config.settings.insights_subdirectory
    insights.mkdir()
    _write(insights / "Themes.md", "Themes")

    (vault / ".obsidian" / "Plugin Readme.md").write_text("# readme\n", encoding="utf-8")

    report_path = tmp_path / "report.json"
    checker.run_check(str(vault), str(report_path))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["scanned_notes"] == 1
    flagged = {issue["file"] for issue in report["issues"]}
    assert not any(path.startswith(config.settings.insights_subdirectory) for path in flagged)
    assert not any(path.startswith(".obsidian") for path in flagged)


def test_run_check_reports_unreadable_file(sandbox, tmp_path, monkeypatch):
    """An unreadable note is a reported issue, not a crashed linter."""
    import builtins
    import os

    vault = sandbox / "vault"
    good = _write(vault / "Entry - 20260101120000.md", "Entry")
    bad = _write(vault / "Broken - 20260101120100.md", "Broken")

    real_open = builtins.open

    def refuse_one(file, *args, **kwargs):
        if os.path.abspath(str(file)) == os.path.abspath(str(bad)):
            raise OSError(22, "Invalid argument")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", refuse_one)
    report_path = tmp_path / "report.json"
    checker.run_check(str(vault), str(report_path))
    monkeypatch.undo()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["scanned_notes"] == 1
    unreadable = [issue for issue in report["issues"] if issue["code"] == "unreadable_file"]
    assert len(unreadable) == 1
    assert unreadable[0]["file"] == "Broken - 20260101120100.md"
    assert good.exists()
