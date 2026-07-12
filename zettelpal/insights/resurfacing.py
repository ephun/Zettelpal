# resurface.py - Surface older notes related to what you've written recently.
#
# For each recent note, find the most similar older notes; collect the best of
# them into a "worth revisiting" list, each shown with what recent note pulled
# it back up. Pure embedding math, no LLM required.

import datetime

import numpy as np

from zettelpal.insights import corpus as corpus_mod
from zettelpal.insights import writer
from zettelpal.log import get_logger

log = get_logger(__name__)

DEFAULT_DAYS = 7
DEFAULT_PER_NOTE = 3
DEFAULT_THRESHOLD = 0.5


def find_resurfaced(corpus: list[dict], days: int = DEFAULT_DAYS,
                    per_note: int = DEFAULT_PER_NOTE,
                    threshold: float = DEFAULT_THRESHOLD,
                    end: datetime.datetime | None = None) -> list[dict]:
    """Return older notes worth revisiting, best-similarity first.

    Each item: {"note": old_note, "sim": float, "trigger": recent_note}.
    """
    end = end or datetime.datetime.now()
    start = end - datetime.timedelta(days=days)

    recent = [c for c in corpus if c["created"] and c["created"] >= start]
    older = [c for c in corpus if c["created"] and c["created"] < start]
    if not recent or not older:
        return []

    older_matrix = np.array([o["embedding"] for o in older], dtype=np.float32)

    # Keep the strongest trigger per older note, deduped by stem.
    picks: dict[str, dict] = {}
    for note in recent:
        sims = older_matrix @ note["embedding"]
        for j in np.argsort(-sims)[:per_note]:
            score = float(sims[j])
            if score < threshold:
                continue
            old = older[j]
            existing = picks.get(old["stem"])
            if existing is None or score > existing["sim"]:
                picks[old["stem"]] = {"note": old, "sim": score, "trigger": note}

    return sorted(picks.values(), key=lambda p: -p["sim"])


def _render(items: list[dict], days: int) -> str:
    lines = [
        f"*Older notes related to what you wrote in the last {days} days.*",
        "",
    ]
    for item in items:
        note, trigger, sim = item["note"], item["trigger"], item["sim"]
        lines.append(f"- {writer._wikilink(note)}  ·  {sim:.2f}")
        lines.append(f"  brought back by {writer._wikilink(trigger)}")
    return "\n".join(lines)


def resurface(days: int = DEFAULT_DAYS, per_note: int = DEFAULT_PER_NOTE,
              threshold: float = DEFAULT_THRESHOLD,
              end: datetime.datetime | None = None, write: bool = True):
    """Find and (by default) write resurfaced notes. Returns the written path,
    or the list when write=False; None if nothing qualifies."""
    corpus = corpus_mod.load_corpus()
    items = find_resurfaced(corpus, days, per_note, threshold, end)
    if not items:
        log.info("[INSIGHTS] Nothing to resurface (need recent and older notes).")
        return None

    if not write:
        return items
    return writer.write_insight(
        "resurface", "Resurfaced", _render(items, days), filename="Resurfaced.md"
    )
