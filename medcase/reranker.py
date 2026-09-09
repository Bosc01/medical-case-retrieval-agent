"""Cross-encoder reranking for the Medical Case Retrieval Agent.

A bi-encoder (what FAISS searches over) embeds the query and the document
independently, so it never sees the two together and can only compare fixed
vectors. A cross-encoder concatenates them into one sequence and runs full
attention across the pair, which is what lets it catch the distinctions a
single embedding flattens -- negation, laterality, whether a finding belongs
to the patient or the differential.

That power costs a forward pass per candidate, so it cannot run over the
corpus. It runs over what FAISS already narrowed down: retrieve deep (top 50),
rerank, hand the top 5 to the generator.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


@contextmanager
def _quiet_load():
    """Silence the weight-loading progress bar; it corrupts eval output."""
    prev = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    try:
        from transformers.utils import logging as hf_logging

        level = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            yield
        finally:
            hf_logging.set_verbosity(level)
    finally:
        if prev is None:
            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        else:
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = prev


# Trained on 255M PubMed search logs (Jin et al., NCBI/NLM). Single-logit
# BertForSequenceClassification -- domain-matched to a PubMed case corpus.
DEFAULT_MODEL = "ncbi/MedCPT-Cross-Encoder"

# General-domain baseline. Worth running as a control: if the biomedical model
# does not beat this on your eval set, the domain match is not doing the work.
BASELINE_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class ScoredDocument:
    """A candidate with both scores kept, so reordering stays auditable."""

    document: Any
    text: str
    rerank_score: float
    retrieval_score: float | None = None
    retrieval_rank: int | None = None
    rerank_rank: int | None = None

    @property
    def rank_delta(self) -> int | None:
        """Positions gained by reranking. Positive = promoted."""
        if self.retrieval_rank is None or self.rerank_rank is None:
            return None
        return self.retrieval_rank - self.rerank_rank


@dataclass
class RerankMetrics:
    """Per-call telemetry. Reranking adds latency; measure it, don't assume it."""

    candidates: int = 0
    scored: int = 0
    cache_hits: int = 0
    batches: int = 0
    latency_ms: float = 0.0
    truncated: int = 0
    padding_tokens: int = 0
    real_tokens: int = 0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "candidates": self.candidates,
            "scored": self.scored,
            "cache_hits": self.cache_hits,
            "batches": self.batches,
            "latency_ms": round(self.latency_ms, 2),
            "ms_per_candidate": round(self.latency_ms / self.candidates, 2)
            if self.candidates
            else 0.0,
            "truncated": self.truncated,
            "padding_waste": round(
                self.padding_tokens / (self.padding_tokens + self.real_tokens), 4
            )
            if (self.padding_tokens + self.real_tokens)
            else 0.0,
        }


class CrossEncoderReranker:
    """Scores (query, document) pairs jointly and reorders by that score.

    The model loads lazily on first use so importing this module stays cheap
    and a process that never reranks never pays for the weights.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 16,
        cache_size: int = 4096,
        normalize: bool = False,
        torch_dtype: torch.dtype | None = None,
        sort_by_length: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = _resolve_device(device)
        self.max_length = max_length
        self.batch_size = batch_size
        self.normalize = normalize
        self.torch_dtype = torch_dtype
        self.sort_by_length = sort_by_length
        self._tokenizer = None
        self._model = None
        self._cache: dict[str, float] = {}
        self._cache_size = cache_size
        self.last_metrics = RerankMetrics()

    # -- model lifecycle ------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return
        t0 = time.perf_counter()
        with _quiet_load():
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        kwargs = {"dtype": self.torch_dtype} if self.torch_dtype else {}
        with _quiet_load():
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, **kwargs
            )
        self._model = model.to(self.device).eval()
        logger.info(
            "loaded %s on %s in %.2fs", self.model_name, self.device, time.perf_counter() - t0
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def warmup(self) -> None:
        """Load weights and run one pass, so the first real query is not slow."""
        self._load()
        self.score("warmup query", ["warmup document"])

    # -- scoring --------------------------------------------------------

    @staticmethod
    def _cache_key(query: str, text: str) -> str:
        return hashlib.blake2b(
            f"{query}\x00{text}".encode("utf-8"), digest_size=16
        ).hexdigest()

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Relevance logits for one query against many documents.

        Higher is more relevant. Raw logits are unbounded and only comparable
        within a single query -- do not threshold them across queries unless
        you set ``normalize=True``, which maps them through a sigmoid.
        """
        if not texts:
            return []
        self._load()

        metrics = RerankMetrics(candidates=len(texts))
        t0 = time.perf_counter()

        scores: list[float | None] = [None] * len(texts)
        pending: list[int] = []
        for i, text in enumerate(texts):
            key = self._cache_key(query, text)
            hit = self._cache.get(key)
            if hit is None:
                pending.append(i)
            else:
                scores[i] = hit
                metrics.cache_hits += 1

        # Every sequence in a batch is padded to the longest one in it, so a
        # batch mixing a 300-character abstract with a 2,500-character one does
        # most of its work on padding. Grouping similar lengths together means
        # the padding is only ever as wide as the spread inside one batch.
        # Scores are written back by original index, so order is unaffected.
        order = pending
        if self.sort_by_length:
            order = sorted(pending, key=lambda i: len(texts[i]), reverse=True)

        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            batch = [texts[i] for i in idx]
            # only_second protects the query: a long abstract gets cut, the
            # clinical question never does.
            encoded = self._tokenizer(
                [query] * len(batch),
                batch,
                truncation="only_second",
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            metrics.truncated += sum(
                1
                for ids in encoded["input_ids"]
                if int(ids.ne(self._tokenizer.pad_token_id).sum()) >= self.max_length
            )
            mask = encoded["attention_mask"]
            metrics.real_tokens += int(mask.sum())
            metrics.padding_tokens += int(mask.numel() - mask.sum())
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.inference_mode():
                logits = self._model(**encoded).logits
            # num_labels==1 -> relevance regression head (MedCPT).
            # num_labels==2 -> take the positive class.
            if logits.shape[-1] == 1:
                batch_scores = logits.squeeze(-1)
            else:
                batch_scores = logits[:, -1]
            if self.normalize:
                batch_scores = torch.sigmoid(batch_scores)
            for i, s in zip(idx, batch_scores.float().tolist()):
                scores[i] = s
                self._remember(self._cache_key(query, texts[i]), s)
            metrics.batches += 1
            metrics.scored += len(idx)

        metrics.latency_ms = (time.perf_counter() - t0) * 1000
        self.last_metrics = metrics
        return [float(s) for s in scores]  # type: ignore[arg-type]

    def _remember(self, key: str, value: float) -> None:
        if len(self._cache) >= self._cache_size:
            # Cheap FIFO eviction; ordinary dicts preserve insertion order.
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = value

    def clear_cache(self) -> None:
        self._cache.clear()

    # -- reranking ------------------------------------------------------

    def rerank(
        self,
        query: str,
        documents: Sequence[Any],
        *,
        top_k: int | None = None,
        text_of: Any = None,
        retrieval_scores: Sequence[float] | None = None,
    ) -> list[ScoredDocument]:
        """Reorder ``documents`` by cross-encoder relevance to ``query``.

        ``text_of`` extracts the scoreable text from whatever object the
        retriever returns -- a LangChain ``Document``, a dict, a dataclass.
        Defaults to ``page_content`` if present, else ``str(doc)``.
        """
        if not documents:
            return []
        extract = text_of or _default_text_of
        texts = [extract(d) for d in documents]
        scores = self.score(query, texts)

        scored = [
            ScoredDocument(
                document=doc,
                text=text,
                rerank_score=score,
                retrieval_score=retrieval_scores[i] if retrieval_scores else None,
                retrieval_rank=i,
            )
            for i, (doc, text, score) in enumerate(zip(documents, texts, scores))
        ]
        scored.sort(key=lambda s: s.rerank_score, reverse=True)
        for new_rank, item in enumerate(scored):
            item.rerank_rank = new_rank
        return scored[:top_k] if top_k else scored


def _default_text_of(doc: Any) -> str:
    for attr in ("page_content", "text", "content", "abstract"):
        value = getattr(doc, attr, None)
        if isinstance(value, str):
            return value
        if isinstance(doc, dict) and isinstance(doc.get(attr), str):
            return doc[attr]
    return str(doc)
