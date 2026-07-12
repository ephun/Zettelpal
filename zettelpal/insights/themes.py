# themes.py - Cluster the vault's notes into recurring themes and label them.
#
# KMeans over note embeddings groups what you keep circling back to; the local
# LLM names each cluster. Output is a single "Themes.md" note in the insights
# folder, with each theme's member notes as navigable wikilinks.

import math

import numpy as np

from zettelpal import llm
from zettelpal.insights import corpus as corpus_mod
from zettelpal.insights import writer
from zettelpal.log import get_logger

log = get_logger(__name__)

THEME_LABEL_PROMPT = """These are excerpts from a set of one person's personal notes that were grouped together because they are semantically similar.

Give the group a short theme name and a one-sentence description of what ties these notes together.

Respond with ONLY a JSON object, no other text:
{{"label": "<3 to 6 word theme name>", "summary": "<one sentence>"}}

Notes:
{items}"""

MAX_LABEL_ITEMS = 8
SNIPPET_CHARS = 200


def choose_k(n: int, k: int | None = None) -> int:
    """Pick a cluster count. Explicit k wins; otherwise ~sqrt(n/2), clamped."""
    if k is not None:
        return max(1, min(k, n))
    if n < 4:
        return 1
    return max(2, min(12, int(round(math.sqrt(n / 2)))))


def cluster_corpus(corpus: list[dict], k: int | None = None) -> list[dict]:
    """Group corpus notes into clusters, each members-first-by-centrality.

    Returns a list of {"members": [note, ...]} ordered by cluster size.
    """
    n = len(corpus)
    if n == 0:
        return []

    k = choose_k(n, k)
    embeddings = corpus_mod.embedding_matrix(corpus)

    if k <= 1 or n < 2:
        labels = np.zeros(n, dtype=int)
        centroids = np.array([embeddings.mean(axis=0)])
    else:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(embeddings)
        centroids = km.cluster_centers_

    grouped: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        grouped.setdefault(int(label), []).append(idx)

    clusters = []
    for label, idxs in grouped.items():
        centroid = centroids[label]
        cnorm = float(np.linalg.norm(centroid)) or 1.0
        unit_centroid = centroid / cnorm
        # Most central notes first: highest cosine to the cluster centroid.
        idxs.sort(key=lambda i: -float(np.dot(corpus[i]["embedding"], unit_centroid)))
        clusters.append({"members": [corpus[i] for i in idxs]})

    clusters.sort(key=lambda c: -len(c["members"]))
    return clusters


def label_cluster(members: list[dict]) -> tuple[str, str]:
    """Ask the local LLM for a theme name + summary for a cluster."""
    items = []
    for note in members[:MAX_LABEL_ITEMS]:
        snippet = " ".join(note["body"].split())[:SNIPPET_CHARS]
        items.append(f"- {note['title']}: {snippet}")

    prompt = THEME_LABEL_PROMPT.format(items="\n".join(items))
    response = llm.llm_chat(prompt, temperature=0.3, max_tokens=200)
    data = llm.extract_json_from_text(response) or {}

    label = str(data.get("label") or "").strip() or "Untitled theme"
    summary = str(data.get("summary") or "").strip()
    return label, summary


def _render(clusters: list[dict]) -> str:
    total = sum(len(c["members"]) for c in clusters)
    lines = [
        f"*{len(clusters)} themes across {total} notes.*",
        "",
    ]
    for cluster in clusters:
        members = cluster["members"]
        lines.append(f"## {cluster['label']}  ({len(members)})")
        if cluster.get("summary"):
            lines.append(f"{cluster['summary']}")
        lines.append("")
        for note in members:
            lines.append(f"- {writer._wikilink(note)}")
        lines.append("")
    return "\n".join(lines)


def generate_themes(k: int | None = None, write: bool = True):
    """Cluster and label the vault. Returns the written path (write=True) or
    the labeled clusters (write=False); None if there are no notes."""
    corpus = corpus_mod.load_corpus()
    if not corpus:
        log.info("[INSIGHTS] No notes to cluster.")
        return None

    clusters = cluster_corpus(corpus, k)
    log.info("[INSIGHTS] Labeling %d themes...", len(clusters))
    for cluster in clusters:
        cluster["label"], cluster["summary"] = label_cluster(cluster["members"])

    if not write:
        return clusters
    return writer.write_insight("themes", "Themes", _render(clusters), filename="Themes.md")
