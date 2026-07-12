# writer.py - Write generated insight notes into the vault's insights folder.
#
# The vault root stays your own entries; everything here lands in
# config.settings.insights_dir, which vault scanning/linking excludes.

import datetime
import os
import re

from zettelpal import config
from zettelpal.log import get_logger

log = get_logger(__name__)


def _slug(text: str) -> str:
    s = re.sub(r'[<>:"/\\|?*]', "", text or "").strip().replace(" ", "-")
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:80] or "insight"


def _wikilink(note: dict) -> str:
    """A '[[stem|title]]' link to a corpus note (navigable in Obsidian)."""
    title = (note.get("title") or note.get("stem") or "").replace("]", "")
    return f"[[{note['stem']}|{title}]]"


def write_insight(
    kind: str,
    title: str,
    body_md: str,
    filename: str | None = None,
    now: datetime.datetime | None = None,
) -> str:
    """Write a Markdown insight note and return its path.

    kind is a short slug used in the `insight/<kind>` tag. A stable `filename`
    (e.g. "Themes.md") overwrites in place; otherwise a dated name is used.
    """
    now = now or datetime.datetime.now()
    os.makedirs(config.settings.insights_dir, exist_ok=True)

    if filename is None:
        filename = f"{now.strftime('%Y-%m-%d')}-{_slug(title)}.md"
    path = os.path.join(config.settings.insights_dir, filename)

    frontmatter = (
        "---\n"
        f'title: "{title}"\n'
        f"created: {now.isoformat(timespec='seconds')}\n"
        "tags:\n"
        "  - insight\n"
        f"  - insight/{kind}\n"
        "---\n\n"
    )
    content = frontmatter + f"# {title}\n\n" + body_md.rstrip() + "\n"

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    log.info("[INSIGHTS] Wrote %s", path)
    return path
