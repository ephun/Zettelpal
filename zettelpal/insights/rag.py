# rag.py - Ask your own vault a question, answered from your notes.
#
# Embeds the question, retrieves the most similar notes by cosine, and has the
# local LLM answer using only those notes (with citations). Local end to end.

import numpy as np

from zettelpal import llm, models
from zettelpal.insights import corpus as corpus_mod
from zettelpal.insights import writer
from zettelpal.log import get_logger

log = get_logger(__name__)

ANSWER_PROMPT = """You are helping someone reflect on their own personal notes. Answer their question using ONLY the notes provided below. Cite the notes you use by their bracketed number, like [2]. If the notes do not address the question, say so plainly rather than guessing.

Question: {question}

Notes:
{context}

Answer:"""

DEFAULT_K = 8
SNIPPET_CHARS = 500


def _query_embedding(question: str) -> np.ndarray | None:
    model = models.load_embedding_model()
    raw = models.get_embedding(question, model) if model is not None else None
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 0 else None


def retrieve(question: str, corpus: list[dict], k: int = DEFAULT_K) -> list[dict]:
    """Return the k notes most similar to the question, best first."""
    query = _query_embedding(question)
    if query is None or not corpus:
        return []
    scored = sorted(corpus, key=lambda c: -float(np.dot(c["embedding"], query)))
    return scored[:k]


def ask(question: str, k: int = DEFAULT_K, corpus: list[dict] | None = None,
        write: bool = False) -> dict:
    """Answer a question from the vault.

    Returns {"answer": str, "sources": [note, ...], "path": str|None}.
    """
    corpus = corpus if corpus is not None else corpus_mod.load_corpus()
    hits = retrieve(question, corpus, k)
    if not hits:
        return {"answer": "There are no notes available to answer from.",
                "sources": [], "path": None}

    context = []
    for i, note in enumerate(hits, 1):
        snippet = " ".join(note["body"].split())[:SNIPPET_CHARS]
        context.append(f"[{i}] {note['title']}\n{snippet}")

    prompt = ANSWER_PROMPT.format(question=question, context="\n\n".join(context))
    answer = llm.llm_chat(prompt, temperature=0.3, max_tokens=800) or "(no response)"

    path = None
    if write:
        path = _write_qa(question, answer, hits)
    return {"answer": answer, "sources": hits, "path": path}


def _write_qa(question: str, answer: str, hits: list[dict]) -> str:
    lines = [f"**Question:** {question}", "", answer.strip(), "", "## Sources", ""]
    for note in hits:
        lines.append(f"- {writer._wikilink(note)}")
    title = question.strip().rstrip("?")[:60] or "Question"
    return writer.write_insight("ask", title, "\n".join(lines))
