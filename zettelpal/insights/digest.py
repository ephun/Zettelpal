# digest.py - Summarize a recent window of notes into a reflective digest.
#
# Gathers notes whose `created` timestamp falls in the window, has the local
# LLM write a short prose digest, and appends light stats + links.

import datetime
from collections import Counter

from zettelpal import llm
from zettelpal.insights import corpus as corpus_mod
from zettelpal.insights import writer
from zettelpal.log import get_logger

log = get_logger(__name__)

DIGEST_PROMPT = """Below are excerpts from one person's personal notes written over a single period of time.

Write a warm, perceptive digest of 2 to 4 short paragraphs describing what they were thinking about, the recurring threads, and any shifts in mood or focus. Address them directly as "you". Do not invent anything that is not supported by the notes.

Notes:
{items}"""

MAX_DIGEST_ITEMS = 40
SNIPPET_CHARS = 220


def _window(corpus: list[dict], start, end) -> list[dict]:
    within = [c for c in corpus if c["created"] and start <= c["created"] <= end]
    within.sort(key=lambda c: c["created"])
    return within


def _render(window: list[dict], prose: str, start, end) -> str:
    tag_counts = Counter(t for c in window for t in c["tags"] if t)
    top_tags = ", ".join(f"{t} ({n})" for t, n in tag_counts.most_common(8))

    lines = [
        f"*{len(window)} notes from {start:%b %d} to {end:%b %d, %Y}.*",
        "",
        prose.strip() or "_(No summary was generated.)_",
        "",
        "## Notes",
        "",
    ]
    for note in window:
        stamp = note["created"].strftime("%b %d") if note["created"] else ""
        lines.append(f"- {stamp} — {writer._wikilink(note)}")
    if top_tags:
        lines += ["", "## Tags", "", top_tags]
    return "\n".join(lines)


def generate_digest(days: int = 7, end: datetime.datetime | None = None,
                    write: bool = True):
    """Summarize the last `days` of notes. Returns the written path
    (write=True) or the rendered body (write=False); None if the window is
    empty."""
    corpus = corpus_mod.load_corpus()
    end = end or datetime.datetime.now()
    start = end - datetime.timedelta(days=days)

    window = _window(corpus, start, end)
    if not window:
        log.info("[INSIGHTS] No notes in the last %d days.", days)
        return None

    items = "\n".join(
        f"- {c['title']}: {' '.join(c['body'].split())[:SNIPPET_CHARS]}"
        for c in window[:MAX_DIGEST_ITEMS]
    )
    prose = llm.llm_chat(DIGEST_PROMPT.format(items=items),
                         temperature=0.4, max_tokens=800) or ""

    body = _render(window, prose, start, end)
    if not write:
        return body

    filename = f"Digest-{start:%Y%m%d}-{end:%Y%m%d}.md"
    title = f"Digest {start:%b %d} – {end:%b %d, %Y}"
    return writer.write_insight("digest", title, body, filename=filename, now=end)
