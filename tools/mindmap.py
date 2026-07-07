#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mindmap.py — Overnight “Internal Conflict Map” for a Zettelpal/Obsidian vault

What it does
------------
1) Scans your Obsidian vault for .md notes (skips _mindmap/ outputs + archive/)
2) Parses YAML frontmatter + body (light cleanup)
3) Computes (and caches) semantic embeddings (SentenceTransformers)
4) Clusters notes into emergent “nodes” (UMAP -> HDBSCAN; no preset k)
5) Computes valence per note (VADER), aggregates per cluster
6) Labels clusters:
   - local phrase-based labels (TF-IDF unigrams/bigrams)
   - optional Gemini refinement (uses *only* excerpts + phrases)
7) Builds edges:
   - similarity edges (cluster centroid cosine similarity)
   - tension edges (high semantic similarity + high valence difference)
   - contradiction edges (Gemini NLI between *atomic claims* extracted per cluster)
8) Writes outputs to: <vault>/_mindmap/YYYY-MM-DD/
   - nodes.csv (Gephi)
   - edges.csv (Gephi)
   - report.md (Obsidian)
   - run.json (metadata)

Design philosophy
-----------------
- Treat outputs as hypothesis/visualization. Don’t treat it as “truth.”
- Contradictions are *candidate* contradictions: useful to review, not definitive.
- Very verbose console output by design.

Gemini API key
--------------
Expects one of these environment variables:
- GEMINI_API_KEY
- GOOGLE_API_KEY
- GOOGLE_AI_API_KEY

Usage
-----
PowerShell:
  python mindmap.py --vault "C:\\Users\\ethan\\Documents\\ephunism" --contradictions

Local-only (no Gemini calls):
  python mindmap.py --vault "C:\\Users\\ethan\\Documents\\ephunism" --local-only

Include tags in embeddings (optional; OFF by default to avoid “tag-dominated blobs”):
  python mindmap.py --vault "..." --embed-tags

Use a light “dimension checklist” (optional; nudges claim extraction toward contradiction-friendly stances):
  python mindmap.py --vault "..." --contradictions --claim-dimensions

Write-back (OFF by default; writes cluster info into frontmatter):
  python mindmap.py --vault "..." --writeback

Notes
-----
- This script intentionally avoids injecting frameworks (utilitarian/Kant/DBT/etc.) unless those words appear
  in your excerpts. The optional --claim-dimensions flag is *not* frameworks; it’s a neutral checklist
  of common dimensions (responsibility, self-worth, obligation, etc.) and is only used “if clearly present.”
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Super-verbose logging
# ============================================================

START_TS = time.time()

def ts() -> str:
    return f"{time.time() - START_TS:9.2f}s"

def log(msg: str) -> None:
    try:
        print(f"[{ts()}] {msg}", flush=True)
    except UnicodeEncodeError:
        print(f"[{ts()}] {msg.encode('ascii', 'replace').decode('ascii')}", flush=True)

def log_kv(title: str, kv: Dict[str, Any]) -> None:
    log(title)
    for k, v in kv.items():
        log(f"  - {k}: {v}")

def hr(char: str = "=", n: int = 72) -> None:
    try:
        print(char * n, flush=True)
    except UnicodeEncodeError:
        print("=" * n, flush=True)

def die(msg: str, code: int = 1) -> None:
    log(f"FATAL: {msg}")
    sys.exit(code)


# ============================================================
# Dependency bootstrap (installs missing packages)
# ============================================================

def ensure_deps() -> None:
    """
    Installs required packages if missing.
    Uses pip in the current Python environment.
    """
    import importlib.util
    import subprocess

    required = [
        ("pyyaml", "yaml"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("sentence-transformers", "sentence_transformers"),
        ("umap-learn", "umap"),
        ("hdbscan", "hdbscan"),
        ("vaderSentiment", "vaderSentiment"),
        # New official Gemini SDK:
        ("google-genai", "google.genai"),
    ]

    missing_pip: List[str] = []
    for pip_name, import_name in required:
        try:
            if importlib.util.find_spec(import_name) is None:
                missing_pip.append(pip_name)
        except (ModuleNotFoundError, ImportError, ValueError):
            missing_pip.append(pip_name)

    if not missing_pip:
        log("[deps] All dependencies already installed ✅")
        return

    hr()
    log("[deps] Missing packages detected. Will install via pip:")
    for p in missing_pip:
        log(f"  - {p}")
    hr()

    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", *missing_pip]
    log(f"[deps] Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

    log("[deps] Install complete ✅")
    hr()


# ============================================================
# Data model + parsing
# ============================================================

@dataclass
class Note:
    path: Path
    rel_path: str
    file_hash: str
    mtime: float
    note_id: str
    title: str
    created: str
    tags: List[str]
    text_for_embedding: str
    text_for_labeling: str
    raw_body: str


FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def parse_frontmatter(md: str) -> Tuple[Dict[str, Any], str]:
    import yaml
    m = FRONTMATTER_RE.match(md)
    if not m:
        return {}, md
    fm_raw = m.group(1)
    body = md[m.end():]
    try:
        fm = yaml.safe_load(fm_raw) or {}
        if not isinstance(fm, dict):
            fm = {}
    except Exception:
        fm = {}
    return fm, body

def normalize_tags(fm: Dict[str, Any]) -> List[str]:
    tags = fm.get("tags", [])
    if tags is None:
        return []
    if isinstance(tags, str):
        return [tags]
    if isinstance(tags, list):
        out: List[str] = []
        for t in tags:
            if isinstance(t, str):
                out.append(t)
        return out
    return []

def clean_body_for_embedding(body: str) -> str:
    """
    Light cleanup to reduce Obsidian noise while keeping semantics.
    """
    txt = body.strip()

    # wikilinks -> keep display-ish text
    txt = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r" \2 ", txt)
    txt = re.sub(r"\[\[([^\]]+)\]\]", r" \1 ", txt)

    # code blocks
    txt = re.sub(r"`{3}.*?`{3}", " ", txt, flags=re.DOTALL)

    # inline code
    txt = re.sub(r"`[^`]+`", " ", txt)

    # collapse whitespace
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def strip_tag_tokens(text: str) -> str:
    """
    Removes 'tag:...' tokens that we optionally embed, so they don't dominate TF-IDF/claim prompts.
    """
    # remove whole tag tokens
    text = re.sub(r"\btag:[^\s]+\b", " ", text)
    # collapse whitespace
    return re.sub(r"\s+", " ", text).strip()

def extract_note_fields(
    path: Path,
    vault: Path,
    raw_bytes: bytes,
    embed_chars: int,
    embed_tags: bool,
) -> Note:
    md = raw_bytes.decode("utf-8", errors="replace")
    fm, body = parse_frontmatter(md)

    rel_path = str(path.relative_to(vault)).replace("\\", "/")
    title = str(fm.get("title") or path.stem)
    note_id = rel_path  # unique/stable
    created = str(fm.get("created") or "")
    tags = normalize_tags(fm)

    body_clean = clean_body_for_embedding(body)
    body_trunc = body_clean[:embed_chars]

    # Embedding text: optionally include tags, but keep them out of labeling text by default.
    if embed_tags:
        tags_str = " ".join([f"tag:{t}" for t in tags[:80]])
        text_for_embedding = f"title: {title}\n{tags_str}\n\n{body_trunc}".strip()
    else:
        text_for_embedding = f"title: {title}\n\n{body_trunc}".strip()

    # Labeling text: NEVER include tag tokens (helps avoid 'tag tag idea' labels)
    text_for_labeling = strip_tag_tokens(text_for_embedding)

    mtime = path.stat().st_mtime
    file_hash = sha256_bytes(raw_bytes)

    return Note(
        path=path,
        rel_path=rel_path,
        file_hash=file_hash,
        mtime=mtime,
        note_id=note_id,
        title=title,
        created=created,
        tags=tags,
        text_for_embedding=text_for_embedding,
        text_for_labeling=text_for_labeling,
        raw_body=body,
    )


# ============================================================
# Vault scanning + filtering
# ============================================================

def should_include_note(
    note: Note,
    min_chars: int,
    include_tag_prefixes: List[str],
    exclude_tag_prefixes: List[str],
) -> bool:
    if len(note.text_for_labeling) < min_chars:
        return False

    tags = note.tags

    if include_tag_prefixes:
        ok = any(any(t.startswith(pref) for pref in include_tag_prefixes) for t in tags)
        if not ok:
            return False

    if exclude_tag_prefixes:
        bad = any(any(t.startswith(pref) for pref in exclude_tag_prefixes) for t in tags)
        if bad:
            return False

    return True

def scan_vault(
    vault: Path,
    min_chars: int,
    include_tag_prefixes: List[str],
    exclude_tag_prefixes: List[str],
    embed_chars: int,
    embed_tags: bool,
) -> List[Note]:
    hr()
    log("[scan] Scanning vault for markdown notes...")
    log(f"[scan] Vault: {vault}")
    hr()

    md_files = [p for p in vault.rglob("*.md") if p.is_file()]
    log(f"[scan] Found .md files (including possible exclusions): {len(md_files)}")

    notes: List[Note] = []
    skipped_read = 0
    skipped_filter = 0
    skipped_mindmap = 0

    for idx, p in enumerate(md_files, start=1):
        # skip output folder + archives (matches your current behavior)
        if "_mindmap" in p.parts or "archive" in p.parts:
            skipped_mindmap += 1
            continue

        try:
            raw = p.read_bytes()
        except Exception:
            skipped_read += 1
            continue

        note = extract_note_fields(p, vault, raw, embed_chars=embed_chars, embed_tags=embed_tags)
        if should_include_note(note, min_chars, include_tag_prefixes, exclude_tag_prefixes):
            notes.append(note)
        else:
            skipped_filter += 1

        if idx % 1000 == 0:
            log(f"[scan] Progress: {idx}/{len(md_files)} files processed... included so far: {len(notes)}")

    hr()
    log_kv("[scan] Scan summary:", {
        "total_md_files": len(md_files),
        "included_notes": len(notes),
        "skipped_mindmap_or_archive": skipped_mindmap,
        "skipped_unreadable": skipped_read,
        "skipped_by_filters": skipped_filter,
    })
    hr()
    return notes


# ============================================================
# Embeddings cache (by file hash, robust to note reorder)
# ============================================================

def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text("utf-8"))

def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), "utf-8")

def load_or_compute_embeddings(
    notes: List[Note],
    cache_dir: Path,
    model_name: str,
    batch_size: int,
) -> Tuple[List[str], "np.ndarray"]:
    """
    Returns (note_ids_in_order, embeddings ndarray)

    Cache:
      cache_dir/embeddings_meta.json
      cache_dir/embeddings.npy

    Cache meta keys:
      - model
      - items: {note_id: {hash}}
      - order: [note_id, ...]
    """
    import numpy as np
    from sentence_transformers import SentenceTransformer

    hr()
    log("[embed] Preparing embeddings (with caching)...")
    log(f"[embed] Cache dir: {cache_dir}")
    log(f"[embed] Model: {model_name}")
    hr()

    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "embeddings_meta.json"
    vecs_path = cache_dir / "embeddings.npy"

    meta = load_json(meta_path) or {}
    cached_model = meta.get("model")
    items = meta.get("items", {})
    order = meta.get("order", [])

    vecs = None
    if vecs_path.exists() and cached_model == model_name and isinstance(order, list):
        try:
            vecs = np.load(vecs_path)
            log(f"[embed] Loaded cached vectors: {vecs.shape}")
        except Exception:
            log("[embed] Failed to load cached vectors; will recompute.")
            vecs = None

    new_order = [n.note_id for n in notes]
    id_to_note = {n.note_id: n for n in notes}

    to_compute: List[Note] = []
    reused = 0

    if vecs is None or cached_model != model_name:
        log("[embed] No valid cache for this model; will compute ALL embeddings.")
        to_compute = notes[:]
        items = {}
        vecs = None
    else:
        for nid in new_order:
            n = id_to_note[nid]
            rec = items.get(nid)
            if rec and rec.get("hash") == n.file_hash:
                reused += 1
            else:
                to_compute.append(n)

    log_kv("[embed] Cache decision:", {
        "notes_total": len(notes),
        "reused_from_cache": reused,
        "to_compute": len(to_compute),
    })

    model = SentenceTransformer(model_name)
    emb_dim = model.get_sentence_embedding_dimension()
    log(f"[embed] Embedding dimension: {emb_dim}")

    if vecs is None:
        out = np.zeros((len(new_order), emb_dim), dtype=np.float32)
        old_index: Dict[str, int] = {}
    else:
        out = vecs
        old_index = {nid: i for i, nid in enumerate(order)}
        if out.shape[1] != emb_dim:
            log("[embed] Cached dim mismatch; recomputing all embeddings.")
            out = np.zeros((len(new_order), emb_dim), dtype=np.float32)
            old_index = {}
            to_compute = notes[:]
            items = {}
        elif out.shape[0] != len(new_order):
            log(f"[embed] Note count changed (cache {out.shape[0]} -> now {len(new_order)}). Resizing.")
            resized = np.zeros((len(new_order), emb_dim), dtype=np.float32)
            copied = 0
            for i, nid in enumerate(new_order):
                oi = old_index.get(nid)
                if oi is not None and oi < out.shape[0]:
                    resized[i] = out[oi]
                    copied += 1
            log(f"[embed] Copied {copied} vectors from old cache into resized matrix.")
            out = resized

    idx_map = {nid: i for i, nid in enumerate(new_order)}

    if to_compute:
        hr()
        log(f"[embed] Computing {len(to_compute)} embeddings (batch_size={batch_size})...")
        hr()
        texts = [n.text_for_embedding for n in to_compute]

        new_vecs = model.encode(
            texts,
            show_progress_bar=True,
            batch_size=batch_size,
            normalize_embeddings=True,
        )
        log("[embed] Finished encoding. Writing into embedding matrix...")

        for n, v in zip(to_compute, new_vecs):
            out[idx_map[n.note_id]] = v.astype(np.float32)
            items[n.note_id] = {"hash": n.file_hash}

        log("[embed] Embed matrix updated for computed notes.")

    meta_out = {"model": model_name, "items": items, "order": new_order}
    np.save(vecs_path, out)
    save_json(meta_path, meta_out)
    log(f"[embed] Saved cache: {vecs_path.name} + {meta_path.name}")

    hr()
    return new_order, out


# ============================================================
# Valence scoring (VADER)
# ============================================================

def compute_valence(notes_by_id: Dict[str, Note]) -> Dict[str, float]:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    hr()
    log("[valence] Scoring valence (VADER compound in [-1,1])...")
    hr()

    analyzer = SentimentIntensityAnalyzer()
    out: Dict[str, float] = {}
    for i, (nid, n) in enumerate(notes_by_id.items(), start=1):
        score = analyzer.polarity_scores(n.text_for_labeling).get("compound", 0.0)
        out[nid] = float(score)
        if i % 2000 == 0:
            log(f"[valence] Scored {i}/{len(notes_by_id)} notes...")

    log("[valence] Done.")
    hr()
    return out


# ============================================================
# Clustering (UMAP -> HDBSCAN)
# ============================================================

def cluster_embeddings(
    embeddings: "np.ndarray",
    n_neighbors: int,
    min_dist: float,
    min_cluster_size: int,
    random_state: int,
    umap_components: int,
    cluster_selection_method: str,
) -> Tuple["np.ndarray", "np.ndarray"]:
    import umap
    import hdbscan

    hr()
    log("[cluster] Running UMAP dimensionality reduction (for clustering)...")
    log_kv("[cluster] UMAP params:", {
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "metric": "cosine",
        "n_components": umap_components,
        "random_state": random_state,
    })
    hr()

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=umap_components,
        metric="cosine",
        random_state=random_state,
    )
    reduced = reducer.fit_transform(embeddings)
    log(f"[cluster] UMAP done. reduced shape={reduced.shape}")

    hr()
    log("[cluster] Running HDBSCAN clustering...")
    log_kv("[cluster] HDBSCAN params:", {
        "min_cluster_size": min_cluster_size,
        "metric": "euclidean",
        "cluster_selection_method": cluster_selection_method,
    })
    hr()

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method=cluster_selection_method,
        prediction_data=False,
    )
    labels = clusterer.fit_predict(reduced)

    log("[cluster] HDBSCAN done.")
    hr()
    return reduced, labels


# ============================================================
# Simple phrase extraction for label grounding
# ============================================================

def top_phrases_for_cluster(texts: List[str], top_k: int = 12) -> List[str]:
    """
    TF-IDF over unigrams/bigrams; returns top terms by mean TF-IDF in this cluster.
    NOTE: These texts should already be stripped of 'tag:' tokens.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    if not texts:
        return []

    vec = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
        lowercase=True,
    )
    X = vec.fit_transform(texts)
    scores = np.asarray(X.mean(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())

    if scores.size == 0:
        return []

    idx = np.argsort(scores)[::-1][:top_k]
    out: List[str] = []
    for t in terms[idx]:
        t = t.strip()
        if len(t) < 3:
            continue
        # extra guard: avoid "tag"
        if t == "tag" or t.startswith("tag "):
            continue
        out.append(t)
    return out


# ============================================================
# Gemini helpers (labeling + claim extraction + NLI)
# ============================================================

def get_gemini_key() -> Optional[str]:
    for k in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_AI_API_KEY"]:
        v = os.getenv(k)
        if v:
            return v
    return None

def gemini_client():
    """
    google-genai client.
    """
    key = get_gemini_key()
    if not key:
        return None
    try:
        from google import genai
    except Exception:
        return None
    return genai.Client(api_key=key)

def gemini_generate_text(
    model_name: str,
    prompt: str,
    temperature: float = 0.2,
    max_retries: int = 3,
) -> str:
    """
    Minimal wrapper with retries. Returns text (may be empty).
    """
    client = gemini_client()
    if client is None:
        return ""

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "temperature": temperature,
                },
            )
            txt = (getattr(resp, "text", "") or "").strip()
            return txt
        except Exception as e:
            log(f"[gemini] ERROR attempt {attempt}/{max_retries}: {e}")
            time.sleep(1.0 * attempt)
    return ""

def label_cluster(
    excerpts: List[str],
    phrases: List[str],
    local_only: bool,
    model_name: str,
) -> str:
    fallback = " ".join([p.title() for p in phrases[:4]]) if phrases else "Cluster"
    if local_only or not get_gemini_key():
        return fallback

    prompt = (
        "You are labeling an UNSUPERVISED text cluster.\n"
        "Rules:\n"
        "- Use ONLY language/terms that appear in the phrases or excerpts.\n"
        "- Output a short neutral label, 2–5 words.\n"
        "- Do NOT introduce philosophical/therapy frameworks unless those words appear in excerpts.\n"
        "- No quotes. No punctuation-heavy titles.\n\n"
        f"Top phrases: {phrases}\n\n"
        "Excerpts:\n"
        + "\n---\n".join(excerpts[:12])
        + "\n\nLabel:"
    )

    txt = gemini_generate_text(model_name, prompt, temperature=0.2)
    label = re.sub(r"[\r\n\t]+", " ", txt).strip()
    label = " ".join(label.split()[:6]).strip()
    return label if label else fallback


def _extract_first_json(txt: str) -> Optional[str]:
    """
    Try to extract the first JSON array/object from a text blob.
    """
    if not txt:
        return None
    # Prefer an array
    m = re.search(r"\[\s*\{.*\}\s*\]", txt, flags=re.DOTALL)
    if m:
        return m.group(0)
    # Fallback to object
    m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
    if m:
        return m.group(0)
    return None


def extract_cluster_claims(
    excerpts: List[str],
    local_only: bool,
    model_name: str,
    fallback_phrases: List[str],
    use_dimensions: bool,
) -> List[Dict[str, Any]]:
    """
    Returns a list of atomic claims (3–7) as dicts:
      {topic, claim, polarity, confidence, evidence:[...]}
    """
    if local_only or not get_gemini_key():
        # offline fallback: one bland “claim-like” item
        if fallback_phrases:
            return [{
                "topic": "theme",
                "claim": f"This cluster focuses on {', '.join(fallback_phrases[:3])}.",
                "polarity": "affirm",
                "confidence": 0.1,
                "evidence": [],
            }]
        return [{
            "topic": "theme",
            "claim": "This cluster expresses a recurring stance in the corpus.",
            "polarity": "affirm",
            "confidence": 0.1,
            "evidence": [],
        }]

    dim_block = ""
    if use_dimensions:
        dim_block = (
            "\nTry to include claims in these dimensions ONLY if clearly present in the excerpts:\n"
            "1) morality/obligation\n"
            "2) responsibility/blame\n"
            "3) self-worth/value\n"
            "4) agency/free will vs determinism\n"
            "5) relationships/attachment\n"
            "6) epistemic certainty/doubt\n"
        )

    prompt = (
        "You are extracting ATOMIC claims from excerpts of personal notes.\n"
        "Goal: produce claims that could CONTRADICT each other across clusters.\n\n"
        "Rules:\n"
        "- Return 3 to 7 claims.\n"
        "- Each claim must be a single, falsifiable assertion.\n"
        "- NO hedging words like: sometimes, maybe, can be, might, uncertain, it depends.\n"
        "  If the excerpt is hedged, convert it into TWO conditional claims with explicit conditions.\n"
        "- Use ONLY information present in the excerpts.\n"
        "- Claims should reflect the author's stance, not generic truths.\n"
        "- Do NOT introduce philosophical/therapy frameworks unless those words appear in excerpts.\n"
        "- Include short topic labels.\n"
        "- Evidence snippets must be direct short phrases from the excerpts (<=140 chars each).\n"
        f"{dim_block}\n"
        "Return JSON ONLY (no markdown) in this schema:\n"
        "[\n"
        "  {\n"
        "    \"topic\": \"...\",\n"
        "    \"claim\": \"...\",\n"
        "    \"polarity\": \"affirm\" or \"deny\",\n"
        "    \"confidence\": 0.0-1.0,\n"
        "    \"evidence\": [\"<=140 chars\", \"...\"]\n"
        "  }\n"
        "]\n\n"
        "Excerpts:\n"
        + "\n---\n".join(excerpts[:12])
        + "\n\nJSON:"
    )

    txt = gemini_generate_text(model_name, prompt, temperature=0.2)
    jtxt = _extract_first_json(txt)
    if not jtxt:
        return [{
            "topic": "theme",
            "claim": "This cluster expresses a recurring stance in the corpus.",
            "polarity": "affirm",
            "confidence": 0.1,
            "evidence": [],
        }]

    try:
        obj = json.loads(jtxt)
        if not isinstance(obj, list):
            raise ValueError("claims_json_not_list")
        cleaned: List[Dict[str, Any]] = []
        for it in obj[:10]:
            if not isinstance(it, dict):
                continue
            topic = str(it.get("topic", "")).strip()[:80] or "topic"
            claim = str(it.get("claim", "")).strip()
            pol = str(it.get("polarity", "affirm")).strip().lower()
            conf = float(it.get("confidence", 0.0))
            ev = it.get("evidence", [])
            if not isinstance(ev, list):
                ev = []
            ev2 = []
            for e in ev[:4]:
                if isinstance(e, str) and e.strip():
                    ev2.append(e.strip()[:140])
            if pol not in {"affirm", "deny"}:
                pol = "affirm"
            conf = max(0.0, min(1.0, conf))

            # hard guard: keep it “claim-like”
            if len(claim) < 8:
                continue

            cleaned.append({
                "topic": topic,
                "claim": claim,
                "polarity": pol,
                "confidence": conf,
                "evidence": ev2,
            })

        # ensure at least 1
        if not cleaned:
            cleaned = [{
                "topic": "theme",
                "claim": "This cluster expresses a recurring stance in the corpus.",
                "polarity": "affirm",
                "confidence": 0.1,
                "evidence": [],
            }]
        return cleaned[:7]
    except Exception:
        return [{
            "topic": "theme",
            "claim": "This cluster expresses a recurring stance in the corpus.",
            "polarity": "affirm",
            "confidence": 0.1,
            "evidence": [],
        }]


def classify_relation_nli(
    claim_a: str,
    claim_b: str,
    local_only: bool,
    model_name: str,
) -> Dict[str, Any]:
    """
    Gemini-powered NLI: contradiction/supports/independent.
    """
    if local_only or not get_gemini_key():
        return {"relation": "independent", "confidence": 0.0, "rationale": "local-only/no-key"}

    prompt = (
        "You are a neutral classifier.\n"
        "Given claim A and claim B, classify the relationship as exactly one of:\n"
        "- contradiction (cannot both be true as stated)\n"
        "- supports (one implies or strengthens the other)\n"
        "- independent (no direct logical relation)\n\n"
        "Return JSON ONLY in this exact schema:\n"
        "{\"relation\":\"...\",\"confidence\":0.0,\"rationale\":\"<=20 words\"}\n\n"
        f"Claim A: {claim_a}\n"
        f"Claim B: {claim_b}\n"
        "JSON:"
    )

    txt = gemini_generate_text(model_name, prompt, temperature=0.0)
    jtxt = _extract_first_json(txt)
    if not jtxt:
        return {"relation": "independent", "confidence": 0.0, "rationale": "no-json"}

    try:
        obj = json.loads(jtxt)
        rel = str(obj.get("relation", "independent")).strip().lower()
        conf = float(obj.get("confidence", 0.0))
        rat = str(obj.get("rationale", "")).strip()
        if rel not in {"contradiction", "supports", "independent"}:
            rel = "independent"
        conf = max(0.0, min(1.0, conf))
        return {"relation": rel, "confidence": conf, "rationale": rat[:120]}
    except Exception:
        return {"relation": "independent", "confidence": 0.0, "rationale": "json-parse-failed"}


# ============================================================
# Graph construction
# ============================================================

def cosine_sim(a: "np.ndarray", b: "np.ndarray") -> float:
    import numpy as np
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float((a @ b) / denom)

def build_cluster_centroids(note_ids: List[str], embeddings: "np.ndarray", labels: "np.ndarray") -> Dict[int, "np.ndarray"]:
    import numpy as np
    clusters: Dict[int, List[int]] = {}
    for idx, lab in enumerate(labels):
        lab = int(lab)
        if lab == -1:
            continue
        clusters.setdefault(lab, []).append(idx)

    centroids: Dict[int, "np.ndarray"] = {}
    for lab, idxs in clusters.items():
        c = embeddings[idxs].mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-9)
        centroids[lab] = c
    return centroids

def build_cluster_valence(note_ids: List[str], labels: "np.ndarray", valence_by_note: Dict[str, float]) -> Dict[int, Dict[str, float]]:
    import numpy as np
    clusters: Dict[int, List[float]] = {}
    for i, lab in enumerate(labels):
        lab = int(lab)
        if lab == -1:
            continue
        nid = note_ids[i]
        clusters.setdefault(lab, []).append(valence_by_note.get(nid, 0.0))

    out: Dict[int, Dict[str, float]] = {}
    for lab, vals in clusters.items():
        arr = np.array(vals, dtype=float)
        out[lab] = {
            "valence_mean": float(arr.mean()) if arr.size else 0.0,
            "valence_std": float(arr.std()) if arr.size else 0.0,
            "n": int(arr.size),
        }
    return out

def neighbor_lists_by_similarity(centroids: Dict[int, "np.ndarray"]) -> Dict[int, List[Tuple[int, float]]]:
    labs = sorted(centroids.keys())
    neigh: Dict[int, List[Tuple[int, float]]] = {}
    for i in labs:
        lst: List[Tuple[int, float]] = []
        for j in labs:
            if i == j:
                continue
            s = cosine_sim(centroids[i], centroids[j])
            lst.append((j, s))
        lst.sort(key=lambda x: x[1], reverse=True)
        neigh[i] = lst
    return neigh

def build_edges_similarity_and_tension(
    centroids: Dict[int, "np.ndarray"],
    cluster_stats: Dict[int, Dict[str, float]],
    top_k_neighbors: int,
    tension_valence_threshold: float,
    min_similarity_for_tension: float,
) -> List[Dict[str, Any]]:
    hr()
    log("[edges] Building similarity + tension edges...")
    log_kv("[edges] Params:", {
        "top_k_neighbors": top_k_neighbors,
        "tension_valence_threshold": tension_valence_threshold,
        "min_similarity_for_tension": min_similarity_for_tension,
    })
    hr()

    neigh = neighbor_lists_by_similarity(centroids)
    labs = sorted(centroids.keys())
    edges: List[Dict[str, Any]] = []

    sim_count = 0
    for i in labs:
        for j, s in neigh[i][:top_k_neighbors]:
            edges.append({
                "Source": f"C{i}",
                "Target": f"C{j}",
                "Type": "Undirected",
                "Relationship": "similarity",
                "Weight": round(float(s), 6),
                "Meta": "",
            })
            sim_count += 1
    log(f"[edges] Raw similarity edges (pre-dedupe): {sim_count}")

    tension_count = 0
    for i in labs:
        vi = cluster_stats[i]["valence_mean"]
        for j, s in neigh[i][:max(top_k_neighbors, 12)]:
            vj = cluster_stats[j]["valence_mean"]
            dv = abs(vi - vj)
            if s >= min_similarity_for_tension and dv >= tension_valence_threshold:
                tension = s * dv
                edges.append({
                    "Source": f"C{i}",
                    "Target": f"C{j}",
                    "Type": "Undirected",
                    "Relationship": "tension",
                    "Weight": round(float(tension), 6),
                    "Meta": f"sim={s:.3f}, dv={dv:.3f}",
                })
                tension_count += 1
    log(f"[edges] Raw tension edges (pre-dedupe): {tension_count}")

    dedup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for e in edges:
        a, b = e["Source"], e["Target"]
        key = (min(a, b), max(a, b), e["Relationship"])
        prev = dedup.get(key)
        if prev is None or float(e["Weight"]) > float(prev["Weight"]):
            dedup[key] = e

    out = list(dedup.values())
    log(f"[edges] Deduped edges count: {len(out)} (similarity+tension)")
    hr()
    return out


def _topic_key(topic: str) -> str:
    t = (topic or "").strip().lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:80] if t else "topic"

def _claim_signature(c: Dict[str, Any]) -> str:
    topic = _topic_key(str(c.get("topic", "")))
    claim = str(c.get("claim", "")).strip().lower()
    claim = re.sub(r"\s+", " ", claim)
    return f"{topic}::{claim[:180]}"

def build_edges_contradictions(
    centroids: Dict[int, "np.ndarray"],
    cluster_claims: Dict[int, List[Dict[str, Any]]],
    local_only: bool,
    model_name: str,
    contradiction_neighbors: int,
    contradiction_threshold: float,
    weight_mode: str,
    topic_match_required: bool,
    max_claim_pairs_per_cluster_pair: int,
    min_similarity_to_test: float,
) -> List[Dict[str, Any]]:
    """
    Builds contradiction edges via Gemini NLI on *atomic claims*.
    Tests each cluster against its top-N semantic neighbors to control API usage.

    Strategy:
    - For each cluster pair (i, j) in neighbor list:
      - Candidate claim pairs filtered by:
          - (optional) topic match
          - basic lexical overlap heuristic (cheap)
      - Run NLI on up to max_claim_pairs_per_cluster_pair best candidates
      - If any contradictions exceed threshold, add one edge with best weight
    """
    hr()
    log("[edges] Building CONTRADICTION edges (Gemini NLI on atomic claims)...")
    log_kv("[edges] Contradiction params:", {
        "enabled": True,
        "local_only": local_only,
        "model": model_name,
        "neighbors_tested_per_cluster": contradiction_neighbors,
        "threshold": contradiction_threshold,
        "weight_mode": weight_mode,
        "topic_match_required": topic_match_required,
        "max_claim_pairs_per_cluster_pair": max_claim_pairs_per_cluster_pair,
        "min_similarity_to_test": min_similarity_to_test,
    })
    if local_only or not get_gemini_key():
        log("[edges] NOTE: local-only or missing API key -> contradiction edges will be empty or trivial.")
    hr()

    neigh = neighbor_lists_by_similarity(centroids)
    labs = sorted(centroids.keys())

    edges: List[Dict[str, Any]] = []
    tested_pairs = 0
    kept_edges = 0

    for i in labs:
        claims_i = cluster_claims.get(i) or []
        if not claims_i:
            continue

        for j, sim in neigh[i][:contradiction_neighbors]:
            if sim < min_similarity_to_test:
                continue
            claims_j = cluster_claims.get(j) or []
            if not claims_j:
                continue

            # Build candidate claim pairs
            cand: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
            for ci in claims_i:
                ti = _topic_key(str(ci.get("topic", "")))
                ai = str(ci.get("claim", "")).strip()
                if not ai:
                    continue
                ai_l = ai.lower()

                for cj in claims_j:
                    tj = _topic_key(str(cj.get("topic", "")))
                    aj = str(cj.get("claim", "")).strip()
                    if not aj:
                        continue
                    aj_l = aj.lower()

                    if topic_match_required and ti != tj:
                        continue

                    # Cheap lexical overlap score (0..1)
                    wi = set(re.findall(r"[a-z0-9]{3,}", ai_l))
                    wj = set(re.findall(r"[a-z0-9]{3,}", aj_l))
                    if not wi or not wj:
                        continue
                    overlap = len(wi & wj) / max(1, min(len(wi), len(wj)))

                    # Combine: prioritize topical overlap, then similarity between clusters
                    score = 0.65 * overlap + 0.35 * float(sim)
                    if score < 0.08:
                        continue

                    cand.append((score, ci, cj))

            if not cand:
                tested_pairs += 1
                continue

            cand.sort(key=lambda x: x[0], reverse=True)
            cand = cand[:max_claim_pairs_per_cluster_pair]

            best_edge: Optional[Dict[str, Any]] = None
            best_w = -1.0
            best_meta = ""

            for _, ci, cj in cand:
                a = str(ci.get("claim", "")).strip()
                b = str(cj.get("claim", "")).strip()
                if not a or not b:
                    continue

                res = classify_relation_nli(a, b, local_only=local_only, model_name=model_name)
                if res["relation"] != "contradiction":
                    continue
                conf = float(res.get("confidence", 0.0))
                if conf < contradiction_threshold:
                    continue

                if weight_mode == "confidence_x_similarity":
                    w = conf * float(sim)
                else:
                    w = conf

                if w > best_w:
                    best_w = w
                    # include minimal meta, plus topics to make review easier
                    best_meta = (
                        f"conf={conf:.3f}, sim={sim:.3f}, "
                        f"topic_i={ci.get('topic','')}, topic_j={cj.get('topic','')}, "
                        f"why={res.get('rationale','')}"
                    )
                    best_edge = {
                        "Source": f"C{i}",
                        "Target": f"C{j}",
                        "Type": "Undirected",
                        "Relationship": "contradiction",
                        "Weight": round(float(w), 6),
                        "Meta": best_meta[:500],
                    }

            tested_pairs += 1
            if best_edge is not None:
                edges.append(best_edge)
                kept_edges += 1

            if tested_pairs % 50 == 0:
                log(f"[edges] Contradiction progress: tested {tested_pairs} cluster-pairs, kept {kept_edges}")

    # dedupe undirected contradiction edges, keep max weight
    dedup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for e in edges:
        a, b = e["Source"], e["Target"]
        key = (min(a, b), max(a, b), e["Relationship"])
        prev = dedup.get(key)
        if prev is None or float(e["Weight"]) > float(prev["Weight"]):
            dedup[key] = e

    out = list(dedup.values())
    log_kv("[edges] Contradiction summary:", {
        "cluster_pairs_tested": tested_pairs,
        "raw_edges_kept": kept_edges,
        "deduped_edges": len(out),
    })
    hr()
    return out


# ============================================================
# Output helpers
# ============================================================

def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    log(f"[io] Writing CSV: {path} (rows={len(rows)})")
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

def generate_report_md(
    out_path: Path,
    run_date: str,
    included_notes: int,
    noise_notes: int,
    n_clusters: int,
    nodes_rows: List[Dict[str, Any]],
    edges_rows: List[Dict[str, Any]],
) -> None:
    log(f"[io] Writing report.md: {out_path}")
    nodes_sorted = sorted(nodes_rows, key=lambda r: int(r["Size"]), reverse=True)

    def top_edges(rel: str, n: int) -> List[Dict[str, Any]]:
        xs = [e for e in edges_rows if e["Relationship"] == rel]
        xs.sort(key=lambda e: float(e["Weight"]), reverse=True)
        return xs[:n]

    tensions = top_edges("tension", 12)
    contradictions = top_edges("contradiction", 12)

    md: List[str] = []
    md.append(f"# Mind Map — {run_date}")
    md.append("")
    md.append("## Summary")
    md.append("")
    md.append(f"- Notes scanned (included): **{included_notes}**")
    md.append(f"- Noise/unclustered notes: **{noise_notes}**")
    md.append(f"- Clusters (nodes): **{n_clusters}**")
    md.append(f"- Edges: **{len(edges_rows)}** (similarity/tension/contradiction)")
    md.append("")
    md.append("## Biggest clusters")
    md.append("")
    for r in nodes_sorted[:20]:
        md.append(
            f"- **{r['Id']}** — {r['Label']} "
            f"(n={r['Size']}, valence={r['Mean_Valence']}, std={r['Valence_Std']})"
        )
        if r.get("Top_Claims"):
            md.append(f"  - top claims: _{r['Top_Claims']}_")
    md.append("")
    md.append("## Strongest tension links (semantic neighbors + big valence difference)")
    md.append("")
    if tensions:
        for e in tensions:
            md.append(f"- {e['Source']} ↔ {e['Target']} (w={e['Weight']}) — {e.get('Meta','')}")
    else:
        md.append("- (none)")
    md.append("")
    md.append("## Strongest contradiction links (Gemini NLI between atomic claims)")
    md.append("")
    if contradictions:
        for e in contradictions:
            md.append(f"- {e['Source']} ↔ {e['Target']} (w={e['Weight']}) — {e.get('Meta','')}")
    else:
        md.append("- (none or not enabled)")
    md.append("")
    md.append("## Files")
    md.append("")
    md.append("- nodes.csv")
    md.append("- edges.csv")
    md.append("- run.json")
    md.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), "utf-8")

def writeback_frontmatter_cluster(
    note_path: Path,
    cluster_id: str,
    cluster_label: str,
    valence: float,
    run_date: str,
) -> bool:
    """
    Adds/updates frontmatter keys:
      mindmap_cluster, mindmap_label, mindmap_valence, mindmap_run

    Returns True if file changed.
    """
    import yaml

    raw = note_path.read_text("utf-8", errors="replace")
    fm, body = parse_frontmatter(raw)
    if not fm:
        fm = {}

    changed = False
    def set_key(k, v):
        nonlocal changed
        if fm.get(k) != v:
            fm[k] = v
            changed = True

    set_key("mindmap_cluster", cluster_id)
    set_key("mindmap_label", cluster_label)
    set_key("mindmap_valence", round(float(valence), 6))
    set_key("mindmap_run", run_date)

    if not changed:
        return False

    fm_dump = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True).strip()
    new_text = f"---\n{fm_dump}\n---\n\n{body.lstrip()}"
    note_path.write_text(new_text, "utf-8")
    return True


# ============================================================
# Main pipeline
# ============================================================

def main() -> None:
    ensure_deps()

    import numpy as np

    ap = argparse.ArgumentParser(description="mindmap.py — Zettelpal internal conflict graph generator")
    ap.add_argument("--vault", required=True, help=r'Path to Obsidian vault, e.g. C:\Users\ethan\Documents\ephunism')

    # Filtering
    ap.add_argument("--min-chars", type=int, default=200, help="Minimum length of labeling/embedding text to include")
    ap.add_argument("--embed-chars", type=int, default=4000, help="Max characters from body used in embeddings")
    ap.add_argument("--include-tag-prefix", action="append", default=[], help="Include notes with tags starting with prefix (repeat)")
    ap.add_argument("--exclude-tag-prefix", action="append", default=["type/inbox"], help="Exclude notes with tags starting with prefix (repeat)")

    # Embedding content control
    ap.add_argument("--embed-tags", action="store_true", help="Include tag:tokens in embeddings (OFF by default)")
    ap.add_argument("--no-embed-tags", action="store_true", help="Explicitly disable tag tokens in embeddings (default behavior)")

    # Embeddings model
    ap.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformers model name")
    ap.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")

    # Clustering
    ap.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors")
    ap.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist")
    ap.add_argument("--min-cluster", type=int, default=20, help="HDBSCAN min_cluster_size")
    ap.add_argument("--umap-components", type=int, default=15, help="UMAP n_components used for clustering (NOT 2D viz)")
    ap.add_argument("--hdbscan-selection", choices=["eom", "leaf"], default="eom", help="HDBSCAN cluster_selection_method")

    # Edges
    ap.add_argument("--top-k-neighbors", type=int, default=6, help="Similarity edges per cluster")
    ap.add_argument("--tension-threshold", type=float, default=0.65, help="Abs(mean valence diff) threshold for tension edges")
    ap.add_argument("--tension-min-sim", type=float, default=0.25, help="Minimum centroid similarity for tension edges")

    # Gemini / labeling
    ap.add_argument("--local-only", action="store_true", help="Disable Gemini calls; local phrase labels only")
    ap.add_argument("--gemini-model", default="gemini-2.0-flash", help="Gemini model name")

    # Contradictions
    ap.add_argument("--contradictions", action="store_true", help="Add contradiction edges via Gemini NLI between atomic claims")
    ap.add_argument("--contradiction-threshold", type=float, default=0.70, help="NLI contradiction confidence threshold")
    ap.add_argument("--contradiction-neighbors", type=int, default=8, help="Only test top-N semantic neighbors per cluster for contradictions")
    ap.add_argument("--contradiction-weight-mode", choices=["confidence", "confidence_x_similarity"], default="confidence_x_similarity")
    ap.add_argument("--contradiction-topic-match", action="store_true", help="Require matching topic labels between claims (reduces API calls; may miss some)")
    ap.add_argument("--contradiction-max-claim-pairs", type=int, default=8, help="Max claim-pairs to NLI-test per cluster-pair")
    ap.add_argument("--contradiction-min-sim", type=float, default=0.15, help="Skip contradiction testing for cluster pairs below this centroid similarity")

    # Claim extraction shaping (framework-lite, optional)
    ap.add_argument("--claim-dimensions", action="store_true", help="Use neutral dimension checklist during claim extraction (optional)")

    # Writeback (off by default)
    ap.add_argument("--writeback", action="store_true", help="Write cluster results back into note frontmatter (OFF by default)")

    args = ap.parse_args()

    vault = Path(args.vault).expanduser().resolve()
    if not vault.exists():
        die(f"Vault not found: {vault}")

    run_date = dt.date.today().isoformat()
    out_dir = vault / "_mindmap" / run_date
    cache_dir = vault / "_mindmap" / "_cache"

    # Determine embed_tags behavior
    embed_tags = bool(args.embed_tags)
    if args.no_embed_tags:
        embed_tags = False

    hr()
    log("mindmap.py starting 🚀")
    log_kv("Run configuration:", {
        "vault": str(vault),
        "run_date": run_date,
        "output_dir": str(out_dir),
        "cache_dir": str(cache_dir),
        "local_only": args.local_only,
        "contradictions": args.contradictions,
        "writeback": args.writeback,
        "embed_tags": embed_tags,
        "claim_dimensions": args.claim_dimensions,
        "gemini_key_present": bool(get_gemini_key()),
    })
    hr()

    # 1) Scan vault
    notes = scan_vault(
        vault=vault,
        min_chars=args.min_chars,
        include_tag_prefixes=args.include_tag_prefix,
        exclude_tag_prefixes=args.exclude_tag_prefix,
        embed_chars=args.embed_chars,
        embed_tags=embed_tags,
    )
    if len(notes) < 50:
        log("[warn] Very small corpus included. Consider lowering --min-chars or adjusting tag filters.")
    notes_by_id = {n.note_id: n for n in notes}

    # 2) Embeddings (cached)
    note_ids, embeddings = load_or_compute_embeddings(
        notes=notes,
        cache_dir=cache_dir,
        model_name=args.model,
        batch_size=args.batch_size,
    )
    log(f"[embed] Final embedding matrix shape: {embeddings.shape}")

    # 3) Valence per note
    valence_by_note = compute_valence(notes_by_id)

    # 4) Clustering
    reduced, labels = cluster_embeddings(
        embeddings=embeddings,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        min_cluster_size=args.min_cluster,
        random_state=42,
        umap_components=args.umap_components,
        cluster_selection_method=args.hdbscan_selection,
    )

    # Cluster summary
    label_counts: Dict[int, int] = {}
    for lab in labels:
        lab = int(lab)
        label_counts[lab] = label_counts.get(lab, 0) + 1

    noise_notes = label_counts.get(-1, 0)
    cluster_ids = sorted([k for k in label_counts.keys() if k != -1])

    hr()
    log_kv("[cluster] Cluster summary:", {
        "included_notes": len(notes),
        "clusters": len(cluster_ids),
        "noise_notes": noise_notes,
        "noise_pct": f"{(noise_notes / max(1,len(notes)))*100:.1f}%",
        "min_cluster_size": args.min_cluster,
        "umap_components": args.umap_components,
        "hdbscan_selection": args.hdbscan_selection,
    })
    if len(cluster_ids) <= 0:
        die("No clusters found (all notes are noise). Try lowering --min-cluster or adjusting filters.")
    hr()

    # 5) Centroids + cluster valence stats
    centroids = build_cluster_centroids(note_ids, embeddings, labels)
    cluster_stats = build_cluster_valence(note_ids, labels, valence_by_note)

    # 6) Label clusters + extract claims
    hr()
    log("[nodes] Building node table (cluster labels, sizes, valence stats, atomic claims)...")
    hr()

    cluster_label_map: Dict[int, str] = {}
    cluster_claims_map: Dict[int, List[Dict[str, Any]]] = {}
    nodes_rows: List[Dict[str, Any]] = []

    # Precompute per-cluster note indices
    cluster_to_note_idxs: Dict[int, List[int]] = {}
    for i, lab in enumerate(labels):
        lab = int(lab)
        if lab == -1:
            continue
        cluster_to_note_idxs.setdefault(lab, []).append(i)

    for c in cluster_ids:
        idxs = cluster_to_note_idxs.get(c, [])
        if not idxs:
            continue

        # Representative selection: choose top by similarity to centroid
        cvec = centroids[c]
        sims: List[Tuple[int, float]] = []
        for i in idxs:
            s = cosine_sim(embeddings[i], cvec)
            sims.append((i, s))
        sims.sort(key=lambda x: x[1], reverse=True)

        rep_idxs = [i for i, _ in sims[:30]]

        # Use tag-stripped text for labeling/phrases/claims (prevents “tag tag idea” domination)
        rep_texts = [notes_by_id[note_ids[i]].text_for_labeling for i in rep_idxs]

        # Excerpts for Gemini (short)
        excerpts: List[str] = []
        for t in rep_texts[:12]:
            snippet = t.replace("\n", " ").strip()
            excerpts.append(snippet[:240])

        phrases = top_phrases_for_cluster(rep_texts, top_k=12)

        # Label
        label = label_cluster(
            excerpts=excerpts,
            phrases=phrases,
            local_only=args.local_only,
            model_name=args.gemini_model,
        )
        cluster_label_map[c] = label

        stats = cluster_stats.get(c, {"valence_mean": 0.0, "valence_std": 0.0, "n": len(idxs)})

        # Atomic claims for contradictions
        claims: List[Dict[str, Any]] = []
        if args.contradictions:
            claims = extract_cluster_claims(
                excerpts=excerpts,
                local_only=args.local_only,
                model_name=args.gemini_model,
                fallback_phrases=phrases,
                use_dimensions=args.claim_dimensions,
            )
            cluster_claims_map[c] = claims

        # Human-friendly short claim preview
        top_claims_preview = ""
        if claims:
            top_claims_preview = " | ".join([str(x.get("claim", "")).strip() for x in claims[:2] if x.get("claim")])

        nodes_rows.append({
            "Id": f"C{c}",
            "Label": label,
            "Size": int(stats["n"]),
            "Mean_Valence": round(float(stats["valence_mean"]), 6),
            "Valence_Std": round(float(stats["valence_std"]), 6),
            "Top_Claims": top_claims_preview,
            "Claims_JSON": json.dumps(claims, ensure_ascii=False) if claims else "",
            "Top_Phrases": "; ".join(phrases[:8]),
        })

        log(f"[nodes] C{c}: n={int(stats['n'])} val={stats['valence_mean']:.3f} std={stats['valence_std']:.3f} label='{label}'")
        if phrases:
            log(f"[nodes]   phrases: {', '.join(phrases[:8])}")
        if claims:
            log(f"[nodes]   claims_extracted: {len(claims)}")
            for k, cc in enumerate(claims[:3], start=1):
                log(f"[nodes]     - ({k}) [{cc.get('topic','topic')}] {cc.get('claim','')}")
        elif args.contradictions:
            log("[nodes]   claims_extracted: 0 (fallback/parse issue)")

    hr()
    log(f"[nodes] Built nodes: {len(nodes_rows)}")
    hr()

    # 7) Build edges (similarity + tension)
    edges_rows = build_edges_similarity_and_tension(
        centroids=centroids,
        cluster_stats=cluster_stats,
        top_k_neighbors=args.top_k_neighbors,
        tension_valence_threshold=args.tension_threshold,
        min_similarity_for_tension=args.tension_min_sim,
    )

    # 8) Contradiction edges
    if args.contradictions:
        contr_edges = build_edges_contradictions(
            centroids=centroids,
            cluster_claims=cluster_claims_map,
            local_only=args.local_only,
            model_name=args.gemini_model,
            contradiction_neighbors=args.contradiction_neighbors,
            contradiction_threshold=args.contradiction_threshold,
            weight_mode=args.contradiction_weight_mode,
            topic_match_required=args.contradiction_topic_match,
            max_claim_pairs_per_cluster_pair=args.contradiction_max_claim_pairs,
            min_similarity_to_test=args.contradiction_min_sim,
        )
        edges_rows.extend(contr_edges)
        log(f"[edges] Total edges after adding contradictions: {len(edges_rows)}")

    # 9) Write outputs
    hr()
    log("[io] Writing output bundle...")
    hr()

    out_dir.mkdir(parents=True, exist_ok=True)

    # nodes.csv
    write_csv(
        out_dir / "nodes.csv",
        nodes_rows,
        ["Id", "Label", "Size", "Mean_Valence", "Valence_Std", "Top_Claims", "Claims_JSON", "Top_Phrases"],
    )

    # edges.csv
    write_csv(
        out_dir / "edges.csv",
        edges_rows,
        ["Source", "Target", "Type", "Relationship", "Weight", "Meta"],
    )

    # run.json metadata
    run_bundle = {
        "run_date": run_date,
        "vault": str(vault),
        "args": vars(args),
        "included_notes": len(notes),
        "clusters": len(cluster_ids),
        "noise_notes": int(noise_notes),
        "gemini_key_present": bool(get_gemini_key()),
        "embed_tags": embed_tags,
        "cluster_label_map": cluster_label_map,
        "claims_present": bool(cluster_claims_map),
    }
    save_json(out_dir / "run.json", run_bundle)
    log("[io] Wrote run.json")

    # report.md
    generate_report_md(
        out_path=out_dir / "report.md",
        run_date=run_date,
        included_notes=len(notes),
        noise_notes=int(noise_notes),
        n_clusters=len(cluster_ids),
        nodes_rows=nodes_rows,
        edges_rows=edges_rows,
    )
    log("[io] Wrote report.md")

    # 10) Optional writeback
    if args.writeback:
        hr()
        log("[writeback] Writing cluster fields into frontmatter (this modifies your notes!)")
        hr()
        changed = 0
        for i, nid in enumerate(note_ids):
            lab = int(labels[i])
            if lab == -1:
                continue
            note = notes_by_id.get(nid)
            if not note:
                continue
            cid = f"C{lab}"
            clabel = cluster_label_map.get(lab, f"cluster {lab}")
            v = valence_by_note.get(nid, 0.0)
            try:
                if writeback_frontmatter_cluster(note.path, cid, clabel, v, run_date):
                    changed += 1
            except Exception as e:
                log(f"[writeback] ERROR updating {note.rel_path}: {e}")
        log(f"[writeback] Updated notes: {changed}")
        hr()

    # Done
    hr()
    log("DONE ✅")
    log_kv("Outputs:", {
        "output_dir": str(out_dir),
        "report_md": str(out_dir / "report.md"),
        "nodes_csv": str(out_dir / "nodes.csv"),
        "edges_csv": str(out_dir / "edges.csv"),
    })
    log("Gephi import tips:")
    log("  - Import nodes.csv as Nodes table")
    log("  - Import edges.csv as Edges table")
    log("  - In Gephi, run Modularity + ForceAtlas2 (or your favorite layout)")
    hr()


def entrypoint():
    try:
        main()
    except KeyboardInterrupt:
        die("Interrupted by user (Ctrl+C).", code=130)
    except Exception as e:
        hr("!")
        log("UNHANDLED EXCEPTION ❌")
        log(str(e))
        traceback.print_exc()
        hr("!")
        sys.exit(1)

if __name__ == "__main__":
    entrypoint()
