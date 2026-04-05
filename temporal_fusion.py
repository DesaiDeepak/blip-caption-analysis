"""
Temporal Fusion Module  (Option B)
────────────────────────────────────────────────────────────────────────────
Provides two complementary post-processing steps for video captioning:

1. **Semantic deduplication** (SBERT + cosine similarity)
   - Removes *near-duplicate* captions that differ only in wording but
     describe the same scene (e.g. "a dog runs" vs "a dog is running").
   - Uses ``sentence-transformers/all-MiniLM-L6-v2`` for fast CPU inference.
   - Threshold 0.85 by default (tune higher to keep more, lower to prune more).

2. **Temporal summarisation** (T5-small)
   - Takes the semantically-deduplicated captions and generates a single
     coherent paragraph that captures the temporal progression of the video.
   - Falls back to the old connector-join method if ``transformers`` is
     unavailable or the model fails to load.

Public API
──────────
    from temporal_fusion import TemporalFusion

    tf = TemporalFusion()                      # lazy-loads models on first use
    deduped = tf.semantic_dedup(captions)      # list[str] → list[str]
    summary = tf.summarise(deduped)            # list[str] → str
    summary = tf.fuse(captions)                # convenience: dedup + summarise
"""

from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger("temporal_fusion")

# ── Connector fallback (used when T5 is unavailable) ─────────────────────────
_CONNECTORS = [
    "Initially", "Then", "After that", "Subsequently",
    "Next", "Later", "Following that", "Afterwards", "Eventually", "Finally",
]


def _connector_join(captions: List[str]) -> str:
    """Simple connector-join fallback (original method)."""
    if not captions:
        return ""
    if len(captions) == 1:
        return captions[0].capitalize()
    parts = []
    n = len(captions)
    for i, cap in enumerate(captions):
        cap = cap.strip().rstrip(".")
        if i == 0:
            conn = _CONNECTORS[0]
        elif i == n - 1:
            conn = _CONNECTORS[-1]
        else:
            conn = _CONNECTORS[((i - 1) % (len(_CONNECTORS) - 2)) + 1]
        if cap:
            cap = cap[0].lower() + cap[1:]
        parts.append(f"{conn}, {cap}")
    return ". ".join(parts) + "."


class TemporalFusion:
    """
    Lazy-loading wrapper around SBERT semantic dedup and T5 summarisation.

    Both models are only loaded on first call so the import is zero-cost.

    Parameters
    ----------
    sbert_model : str
        HuggingFace model ID for the sentence-embedding model.
    summarizer_model : str
        HuggingFace model ID for the summarisation model.
    dedup_threshold : float
        Cosine-similarity threshold above which two captions within the
        sliding window are considered duplicates (default 0.85).
    dedup_window : int
        How many of the most-recently kept captions to compare against.
        A window of 5 prevents over-collapsing on long videos while still
        removing repeated consecutive shots (default 5).
    max_summary_tokens : int
        Maximum tokens the summariser may generate (default 80).
    min_summary_tokens : int
        Minimum tokens the summariser must generate (default 20).
    """

    def __init__(
        self,
        sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        summarizer_model: str = "t5-small",
        dedup_threshold: float = 0.85,
        dedup_window: int = 5,
        max_summary_tokens: int = 120,
        min_summary_tokens: int = 20,
    ):
        self._sbert_model_name = sbert_model
        self._summarizer_model_name = summarizer_model
        self.dedup_threshold = dedup_threshold
        self.dedup_window = dedup_window  # sliding window size for dedup
        self.max_summary_tokens = max_summary_tokens
        self.min_summary_tokens = min_summary_tokens

        # Lazy-loaded — None until first use
        self._sbert: Optional[object] = None
        self._summarizer: Optional[object] = None
        self._sbert_failed = False
        self._summarizer_failed = False

    # ── Lazy loaders ─────────────────────────────────────────────────────────

    def _load_sbert(self):
        if self._sbert is not None or self._sbert_failed:
            return
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SBERT model: {self._sbert_model_name}")
            self._sbert = SentenceTransformer(self._sbert_model_name)
            logger.info("SBERT model loaded.")
        except Exception as exc:
            logger.warning(f"Could not load SBERT model: {exc}. Falling back to exact-string dedup.")
            self._sbert_failed = True

    def _load_summarizer(self):
        if self._summarizer is not None or self._summarizer_failed:
            return
        try:
            import torch
            from transformers import T5ForConditionalGeneration, AutoTokenizer
            logger.info(f"Loading T5 summariser: {self._summarizer_model_name}")
            tokenizer = AutoTokenizer.from_pretrained(self._summarizer_model_name)
            model = T5ForConditionalGeneration.from_pretrained(
                self._summarizer_model_name
            )
            model.eval()
            # Store as a callable dict for use in summarise()
            self._summarizer = {"model": model, "tokenizer": tokenizer, "torch": torch}
            logger.info("T5 summariser loaded.")
        except Exception as exc:
            logger.warning(
                f"Could not load T5 summariser: {exc}. "
                "Falling back to connector-join."
            )
            self._summarizer_failed = True

    # ── Public methods ────────────────────────────────────────────────────────

    def semantic_dedup(self, captions: List[str]) -> List[str]:
        """
        Remove semantically near-duplicate captions using a sliding window.

        Each caption is compared against the last ``dedup_window`` kept
        captions (not all of them).  This prevents global collapse on long
        videos where many nature/outdoor scenes share high cosine similarity,
        while still dropping truly repetitive consecutive shots.

        Falls back to exact-string dedup if SBERT is unavailable.
        """
        if not captions:
            return []
        if len(captions) == 1:
            return list(captions)

        self._load_sbert()

        if self._sbert_failed or self._sbert is None:
            # Exact-string fallback
            unique = [captions[0]]
            for cap in captions[1:]:
                if cap.lower().strip() != unique[-1].lower().strip():
                    unique.append(cap)
            return unique

        # SBERT path — sliding-window dedup.
        # Compare each caption against only the last `dedup_window` kept
        # captions so long videos don't over-collapse.
        try:
            from sentence_transformers import util
            embeddings = self._sbert.encode(captions, convert_to_tensor=True)
            unique: List[str] = [captions[0]]
            unique_embeds = [embeddings[0]]

            for i in range(1, len(captions)):
                # Only look at the most recent `dedup_window` kept embeddings
                window = unique_embeds[-self.dedup_window:]
                sims = [util.cos_sim(embeddings[i], kept).item() for kept in window]
                max_sim = max(sims)
                if max_sim < self.dedup_threshold:
                    unique.append(captions[i])
                    unique_embeds.append(embeddings[i])
                else:
                    logger.debug(
                        f"Dropped near-duplicate (max_sim={max_sim:.2f}): '{captions[i]}'"
                    )
            logger.info(
                f"Semantic dedup: {len(captions)} → {len(unique)} captions "
                f"(threshold={self.dedup_threshold}, window={self.dedup_window})"
            )
            return unique
        except Exception as exc:
            logger.warning(f"SBERT dedup error: {exc}. Falling back to exact-string.")
            unique = [captions[0]]
            for cap in captions[1:]:
                if cap.lower().strip() != unique[-1].lower().strip():
                    unique.append(cap)
            return unique

    def summarise(self, captions: List[str]) -> str:
        """
        Produce a single coherent paragraph from a list of captions using T5.

        Falls back to connector-join if the summariser is unavailable or the
        joined text is too short to summarise.
        """
        if not captions:
            return ""
        if len(captions) == 1:
            return captions[0].capitalize()

        self._load_summarizer()

        joined = " ".join(captions)

        # T5 needs enough input tokens to summarise — short inputs get connector-join
        word_count = len(joined.split())
        if self._summarizer_failed or self._summarizer is None or word_count < 15:
            logger.info("Using connector-join (summariser unavailable or input too short).")
            return _connector_join(captions)

        try:
            model = self._summarizer["model"]
            tokenizer = self._summarizer["tokenizer"]
            torch = self._summarizer["torch"]

            # T5 requires a task prefix.  We use a temporally-aware prompt
            # so the model preserves the chronological progression of events
            # instead of producing a generic one-line summary.
            numbered = " ".join(
                f"({i+1}) {c.strip().rstrip('.')}." for i, c in enumerate(captions)
            )
            t5_input = (
                "summarize the following sequence of video frame descriptions "
                "into a coherent temporal paragraph: " + numbered
            )
            inputs = tokenizer(
                t5_input, return_tensors="pt", truncation=True, max_length=512
            )
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.max_summary_tokens,
                    min_new_tokens=self.min_summary_tokens,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )
            summary = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            logger.info(f"T5 summary: '{summary[:80]}'")
            return summary
        except Exception as exc:
            logger.warning(f"Summariser error: {exc}. Falling back to connector-join.")
            return _connector_join(captions)

    def fuse(self, captions: List[str]) -> str:
        """
        Convenience method: semantic dedup → summarise → return paragraph.

        This is the main entry point for the video captioning pipeline.
        """
        deduped = self.semantic_dedup(captions)
        return self.summarise(deduped)
