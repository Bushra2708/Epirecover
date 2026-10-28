"""
src/evidence_retrieval/retriever.py
------------------------------------
Evidence retrieval engine for EpiRecover.

Retrieval tiers (in order of preference):
  1. Exact match on record 'id' or 'transcription'
  2. Fuzzy character-level match using rapidfuzz (or stdlib Levenshtein fallback)
  3. FAISS + sentence-transformer semantic search (optional; requires extra deps)

Every retrieved result includes its source record ID so provenance is traceable.

Data-leakage prevention:
  Records whose 'id' matches an entry in `exclude_ids` are excluded from
  retrieval results. This ensures evaluation inscription IDs are not
  accidentally used as supporting evidence for themselves.

EpiRecover, 2026.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

from src.evidence_retrieval.corpus_adapter import CorpusAdapter

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional rapidfuzz import; fallback to stdlib
# ---------------------------------------------------------------------------
try:
    from rapidfuzz import fuzz as _rfuzz
    def _similarity(a: str, b: str) -> float:
        return _rfuzz.token_set_ratio(a.lower(), b.lower()) / 100.0
except ImportError:
    log.info("rapidfuzz not installed; using slower stdlib Levenshtein for fuzzy match.")
    def _levenshtein(a: str, b: str) -> int:
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            curr = [i] + [0] * len(b)
            for j, cb in enumerate(b, 1):
                cost = 0 if ca == cb else 1
                curr[j] = min(curr[j-1]+1, prev[j]+1, prev[j-1]+cost)
            prev = curr
        return prev[-1]

    def _similarity(a: str, b: str) -> float:
        if not a and not b:
            return 1.0
        return 1.0 - _levenshtein(a.lower(), b.lower()) / max(len(a), len(b), 1)


# ---------------------------------------------------------------------------
# Optional FAISS import
# ---------------------------------------------------------------------------
_HAS_FAISS = False
try:
    import faiss as _faiss
    _HAS_FAISS = True
except ImportError:
    pass

_HAS_SENTENCE_TRANSFORMERS = False
try:
    from sentence_transformers import SentenceTransformer as _ST
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class CorpusRetriever:
    """
    Retrieves supporting inscription records for a query string.

    Loads the corpus from the configured corpus directory on first use.
    """

    def __init__(
        self,
        corpus_dir: str,
        cfg: Optional[Dict] = None,
        enable_faiss: Optional[bool] = None,
    ) -> None:
        self._corpus_dir = corpus_dir
        self._cfg = cfg or {}
        self._adapter = CorpusAdapter()
        self._adapter.load_directory(corpus_dir)

        # FAISS index (optional)
        retrieval_cfg = self._cfg.get("corpus", {})
        self._enable_faiss = (
            enable_faiss
            if enable_faiss is not None
            else retrieval_cfg.get("enable_faiss", False)
        )
        self._faiss_index = None
        self._faiss_ids: List[str] = []
        self._embedder = None

        if self._enable_faiss:
            self._init_faiss(retrieval_cfg)

        log.info(
            "CorpusRetriever: %d records loaded (faiss=%s)",
            self._adapter.count, self._enable_faiss and self._faiss_index is not None,
        )

    def _init_faiss(self, retrieval_cfg: Dict) -> None:
        """Initialise FAISS semantic index if dependencies are available."""
        if not (_HAS_FAISS and _HAS_SENTENCE_TRANSFORMERS):
            log.warning(
                "FAISS semantic retrieval requested but faiss-cpu and/or "
                "sentence-transformers are not installed. Falling back to fuzzy match."
            )
            self._enable_faiss = False
            return

        index_path = retrieval_cfg.get("faiss_index_path", "models/corpus_faiss.index")
        model_name = retrieval_cfg.get(
            "embedding_model",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        )

        try:
            self._embedder = _ST(model_name)
            records = self._adapter.records
            texts = [r["_text_index"] for r in records]
            if not texts:
                return
            embeddings = self._embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            dim = embeddings.shape[1]
            index = _faiss.IndexFlatL2(dim)
            index.add(embeddings)
            self._faiss_index = index
            self._faiss_ids = [r["id"] for r in records]
            if index_path:
                os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
                _faiss.write_index(index, index_path)
            log.info("FAISS index built with %d vectors (dim=%d).", len(texts), dim)
        except Exception as e:
            log.warning("FAISS index build failed: %s — falling back to fuzzy.", e)
            self._enable_faiss = False

    # ------------------------------------------------------------------ #
    # Retrieval methods                                                    #
    # ------------------------------------------------------------------ #

    def _exact_match(self, query: str, records: List[Dict]) -> List[Tuple[Dict, float]]:
        results = []
        q = query.strip().lower()
        for rec in records:
            if q == rec.get("transcription", "").lower() or q == rec.get("id", "").lower():
                results.append((rec, 1.0))
        return results

    def _fuzzy_match(
        self, query: str, records: List[Dict], top_k: int, min_score: float
    ) -> List[Tuple[Dict, float]]:
        scored = []
        q = query.lower()
        for rec in records:
            idx = rec.get("_text_index", "")
            score = _similarity(q, idx[:500])  # cap to avoid quadratic on long texts
            scored.append((rec, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(r, s) for r, s in scored[:top_k] if s >= min_score]

    def _faiss_search(self, query: str, top_k: int) -> List[Tuple[Dict, float]]:
        if not (self._faiss_index and self._embedder):
            return []
        try:
            vec = self._embedder.encode([query], convert_to_numpy=True)
            distances, indices = self._faiss_index.search(vec, top_k)
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0:
                    continue
                rec_id = self._faiss_ids[idx]
                rec = self._adapter.get_by_id(rec_id)
                if rec:
                    # Convert L2 distance to a 0-1 score (heuristic)
                    score = float(max(0.0, 1.0 - dist / 10.0))
                    results.append((rec, score))
            return results
        except Exception as e:
            log.warning("FAISS search error: %s", e)
            return []

    # ------------------------------------------------------------------ #
    # Tamil Unicode check                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_tamil_query(query: str) -> bool:
        """
        Return True if the query contains enough Tamil Unicode characters
        to be useful for text-based retrieval (>=10% of non-space chars).
        A non-Tamil query (e.g. '44 4' from broken OCR) will produce
        near-zero fuzzy similarity against Tamil corpus records.
        """
        stripped = query.replace(" ", "")
        if not stripped:
            return False
        tamil_count = sum(1 for c in stripped if "\u0b80" <= c <= "\u0bff")
        return (tamil_count / len(stripped)) >= 0.10

    # ------------------------------------------------------------------ #
    # Metadata-guided fallback retrieval                                   #
    # ------------------------------------------------------------------ #

    def _metadata_fallback_retrieve(
        self,
        records: List[Dict],
        metadata: Optional[Any],
        top_k: int,
    ) -> List[Tuple[Dict, float]]:
        """
        When OCR query is non-Tamil or empty, retrieve corpus records that
        share metadata with the query inscription (period, region, monument).

        Metadata-based matches receive max score 0.50 to distinguish them
        from text-similarity matches (which can reach 1.0).
        This prevents metadata fallback from inflating R scores unfairly.
        """
        if metadata is None:
            return []

        target_period = (getattr(metadata, "historical_period", None) or "").lower()
        target_region = (getattr(metadata, "region", None) or "").lower()
        target_monument = (getattr(metadata, "monument", None) or "").lower()

        if not any([target_period, target_region, target_monument]):
            log.info(
                "CorpusRetriever: metadata fallback skipped — no period/region/monument supplied."
            )
            return []

        scored: List[Tuple[Dict, float]] = []
        for rec in records:
            rec_period = (rec.get("historical_period") or "").lower()
            rec_region = (rec.get("region") or "").lower()
            rec_monument = (rec.get("monument") or "").lower()

            score_parts: List[float] = []
            # Period match: fuzzy (any keyword overlap)
            if target_period and rec_period:
                overlap = sum(
                    1 for w in target_period.split()
                    if len(w) > 3 and w in rec_period
                )
                score_parts.append(min(1.0, overlap * 0.5))
            elif target_period:
                score_parts.append(0.0)  # target provided but record has none

            # Region match
            if target_region and rec_region:
                overlap = sum(
                    1 for w in target_region.split()
                    if len(w) > 3 and w in rec_region
                )
                score_parts.append(min(1.0, overlap * 0.5))
            elif target_region:
                score_parts.append(0.0)

            # Monument match
            if target_monument and rec_monument:
                overlap = sum(
                    1 for w in target_monument.split()
                    if len(w) > 3 and w in rec_monument
                )
                score_parts.append(min(1.0, overlap * 0.5))
            elif target_monument:
                score_parts.append(0.0)

            if not score_parts:
                continue
            raw_score = sum(score_parts) / len(score_parts)
            # Cap at 0.50 to distinguish metadata match from text match.
            meta_score = min(0.50, raw_score)
            if meta_score > 0.0:
                scored.append((rec, meta_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        log.info(
            "CorpusRetriever: metadata fallback returned %d candidate(s) (period='%s', region='%s').",
            len(scored[:top_k]), target_period, target_region,
        )
        return scored[:top_k]

    def retrieve(
        self,
        query: str,
        script: Optional[str] = None,
        language: Optional[str] = None,
        exclude_ids: Optional[List[str]] = None,
        top_k: int = 5,
        min_score: float = 0.25,
        metadata: Optional[Any] = None,
    ) -> List[Dict]:
        """
        Retrieve supporting evidence records for `query`.

        Parameters
        ----------
        query : str
            OCR or mapped text to match against the corpus.
        script : str, optional
            If provided, prefer records with matching script ('tamil' or 'tamil-brahmi').
        language : str, optional
            Target language filter. Defaults to None (all languages in corpus).
        exclude_ids : list of str, optional
            Record IDs to exclude (data-leakage prevention for evaluation).
        top_k : int
            Maximum number of records to return.
        min_score : float
            Minimum similarity score (0-1) to include a record.
        metadata : InscriptionMetadata, optional
            Researcher-supplied metadata used for fallback retrieval when the
            OCR query is non-Tamil or empty.

        Returns
        -------
        List of record dicts, each with an added 'score' field.
        Never contains records with IDs in exclude_ids.
        """
        query = (query or "").strip()

        # Data leakage prevention: strictly filter out any record in exclude_ids
        exclude = set(exclude_ids or [])
        all_records = [r for r in self._adapter.records if r["id"] not in exclude]

        # Filter by language if specified
        if language:
            lang_records = [
                r for r in all_records
                if (r.get("language") or "").lower() == language.lower()
            ]
            if lang_records:
                all_records = lang_records

        # Optionally filter to matching script (prefer matching script pool)
        if script:
            script_records = [
                r for r in all_records
                if script.lower() in (r.get("script") or "").lower()
            ]
            candidate_pool = script_records if script_records else all_records
        else:
            candidate_pool = all_records

        if not candidate_pool:
            log.info("CorpusRetriever: candidate pool is empty; returning no evidence.")
            return []

        # Determine whether the query is usable for text-based retrieval.
        query_is_tamil = self._is_tamil_query(query)
        if language and language.lower() == "tamil":
            query_is_usable = query_is_tamil and len(query) >= 3
        else:
            query_is_usable = len(query) >= 3

        results: List[Tuple[Dict, float]] = []

        if query_is_usable:
            # Tier 1: exact match
            results = self._exact_match(query, candidate_pool)

            # Tier 2/3: semantic or fuzzy
            if self._enable_faiss and self._faiss_index:
                semantic = self._faiss_search(query, top_k * 2)
                results.extend(semantic)
            else:
                fuzzy = self._fuzzy_match(query, candidate_pool, top_k * 2, min_score)
                results.extend(fuzzy)

            if not results and metadata is not None:
                results = self._metadata_fallback_retrieve(candidate_pool, metadata, top_k * 2)
        else:
            # Text query is non-Tamil or too short (e.g. broken OCR produced '44 4').
            # Fall back to metadata-guided retrieval so R can be non-zero when
            # the researcher supplied period/region/monument context.
            log.info(
                "CorpusRetriever: query '%s' is non-Tamil or too short; "
                "using metadata-guided fallback retrieval.",
                query[:40],
            )
            results = self._metadata_fallback_retrieve(candidate_pool, metadata, top_k * 2)

        # Deduplicate by record id, keep highest score
        seen: Dict[str, float] = {}
        deduped: List[Tuple[Dict, float]] = []
        for rec, score in results:
            rid = rec["id"]
            if rid not in seen or score > seen[rid]:
                seen[rid] = score
                deduped = [(r, s) for r, s in deduped if r["id"] != rid]
                deduped.append((rec, score))

        deduped.sort(key=lambda x: x[1], reverse=True)
        top = deduped[:top_k]

        output = []
        for rec, score in top:
            if score < min_score:
                continue
            r = dict(rec)
            r["score"] = round(score, 4)
            r["retrieval_method"] = "text" if query_is_usable else "metadata_fallback"
            # Strip internal index field before returning
            r.pop("_text_index", None)
            output.append(r)

        log.info(
            "CorpusRetriever: query='%s...' -> %d results (pool=%d excluded=%d method=%s)",
            query[:40], len(output), len(candidate_pool), len(exclude),
            "text" if query_is_usable else "metadata_fallback",
        )
        return output
