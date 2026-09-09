"""MedCPT bi-encoder embeddings -- the vectors FAISS actually searches.

MedCPT is *asymmetric*: a clinical question and a case report go through two
different towers into one shared space. That asymmetry is the point. A query
is a short, under-specified vignette; an article is a titled abstract. Training
them separately lets each tower specialise while contrastive training keeps the
outputs comparable, so a query vector can be dotted against an article vector.

Consequences that are easy to get wrong, and that silently degrade recall
rather than raising anything:

* Use the right tower for the right text. Encoding a query with the article
  encoder produces a well-formed 768-d vector that simply lands in the wrong
  neighbourhood.
* Pooling is the ``[CLS]`` token, not a mean over tokens. MedCPT's contrastive
  loss was applied to ``[CLS]``; mean pooling reads a representation nothing
  ever trained.
* An article is a *text pair* -- ``title [SEP] abstract`` -- not a concatenated
  string. The pair form sets ``token_type_ids``, which the model uses to tell
  the two fields apart.

This module owns only the vectors. Building and searching the index lives
elsewhere; reranking the shortlist lives in ``reranker.py``.
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.utils import logging as hf_logging

from .reranker import _resolve_device

logger = logging.getLogger(__name__)

QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"

# MedCPT's own truncation limits. Queries are short by construction, so 64 is
# not a compromise; articles get the full BERT window.
QUERY_MAX_LENGTH = 64
ARTICLE_MAX_LENGTH = 512


@contextlib.contextmanager
def _quiet_load() -> Iterator[None]:
    """Silence the weight-loading progress bar for the duration of a load.

    Scoped and restoring rather than a global disable at import, so importing
    this module does not decide how the rest of the process reports progress.
    """
    was_on = hf_logging.is_progress_bar_enabled()
    hf_logging.disable_progress_bar()
    try:
        yield
    finally:
        if was_on:
            hf_logging.enable_progress_bar()


class MedCPTEmbedder:
    """Encodes queries and articles into MedCPT's shared 768-d space.

    Each tower loads lazily and independently: a serving process that only
    ever embeds queries never pays for the article encoder's weights, and
    indexing does not pay for the query encoder.
    """

    def __init__(
        self,
        query_model: str = QUERY_MODEL,
        article_model: str = ARTICLE_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 16,
        query_max_length: int = QUERY_MAX_LENGTH,
        article_max_length: int = ARTICLE_MAX_LENGTH,
        normalize: bool = False,
    ) -> None:
        """
        ``normalize`` L2-normalises every vector, turning the dot product into
        cosine similarity. It defaults to False because MedCPT was trained and
        evaluated on the dot product of raw ``[CLS]`` vectors: normalising
        discards magnitude, which that objective uses, so rankings shift. Turn
        it on only for a deliberate cosine setup, and turn it on for *both*
        sides -- and for the FAISS index -- or the two halves of a similarity
        stop being comparable.
        """
        self.query_model = query_model
        self.article_model = article_model
        self.device = _resolve_device(device)
        self.batch_size = batch_size
        self.query_max_length = query_max_length
        self.article_max_length = article_max_length
        self.normalize = normalize
        self._query_tok = None
        self._query_enc = None
        self._article_tok = None
        self._article_enc = None
        self._dim: int | None = None

    # -- model lifecycle ------------------------------------------------

    def _load(self, which: str) -> tuple[Any, Any]:
        name = self.query_model if which == "query" else self.article_model
        tok = self._query_tok if which == "query" else self._article_tok
        enc = self._query_enc if which == "query" else self._article_enc
        if enc is not None:
            return tok, enc

        t0 = time.perf_counter()
        with _quiet_load():
            tok = AutoTokenizer.from_pretrained(name)
            enc = AutoModel.from_pretrained(name).to(self.device).eval()
        logger.info(
            "loaded %s on %s in %.2fs", name, self.device, time.perf_counter() - t0
        )
        if which == "query":
            self._query_tok, self._query_enc = tok, enc
        else:
            self._article_tok, self._article_enc = tok, enc
        return tok, enc

    @property
    def is_loaded(self) -> bool:
        """True once both towers are resident."""
        return self._query_enc is not None and self._article_enc is not None

    @property
    def embedding_dim(self) -> int:
        """Width of the shared space, read from config so an empty batch can
        be shaped without paying to load half a gigabyte of weights."""
        if self._dim is None:
            q = AutoConfig.from_pretrained(self.query_model).hidden_size
            a = AutoConfig.from_pretrained(self.article_model).hidden_size
            if q != a:
                raise ValueError(
                    f"encoder widths disagree ({self.query_model}={q}, "
                    f"{self.article_model}={a}); these towers do not share a space"
                )
            self._dim = int(a)
        return self._dim

    def warmup(self) -> None:
        """Load both towers and run one pass each, so the first real call is
        not paying for weight loading and lazy kernel compilation."""
        self.encode_queries(["warmup query"])
        self.encode_articles([("warmup title", "warmup abstract")])

    # -- encoding -------------------------------------------------------

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        """Encode clinical questions. Returns ``(len(queries), 768)`` float32."""
        if isinstance(queries, str):
            raise TypeError("encode_queries takes a list of strings, not one string")
        if not queries:
            return self._empty()
        tok, enc = self._load("query")
        return self._forward(
            tok, enc, list(queries), self.query_max_length, truncation=True
        )

    def encode_articles(self, records: Sequence[Any]) -> np.ndarray:
        """Encode case reports. Returns ``(len(records), 768)`` float32.

        Accepts ``CaseRecord`` objects (anything with ``.title``/``.abstract``)
        or ``(title, abstract)`` pairs. Both become a tokenizer text pair, so
        the model sees ``[CLS] title [SEP] abstract [SEP]``.
        """
        if not records:
            return self._empty()
        pairs = [_as_pair(r) for r in records]
        tok, enc = self._load("article")
        # longest_first is what MedCPT's reference implementation uses. With a
        # short title and a long abstract it trims the abstract only, so the
        # title -- the densest signal in a case report -- survives intact.
        return self._forward(
            tok, enc, pairs, self.article_max_length, truncation="longest_first"
        )

    def _forward(
        self,
        tokenizer: Any,
        model: Any,
        inputs: list[Any],
        max_length: int,
        *,
        truncation: bool | str,
    ) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for start in range(0, len(inputs), self.batch_size):
            batch = inputs[start : start + self.batch_size]
            encoded = tokenizer(
                batch,
                truncation=truncation,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.inference_mode():
                hidden = model(**encoded).last_hidden_state
            # [CLS] pooling: position 0. Not a mean over tokens -- see module
            # docstring.
            pooled = hidden[:, 0, :]
            if self.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            chunks.append(pooled.float().cpu().numpy())
        return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)

    def _empty(self) -> np.ndarray:
        return np.zeros((0, self.embedding_dim), dtype=np.float32)


def _as_pair(record: Any) -> list[str]:
    """Normalise a record into ``[title, abstract]``."""
    title = getattr(record, "title", None)
    abstract = getattr(record, "abstract", None)
    if isinstance(title, str) and isinstance(abstract, str):
        return [title, abstract]
    if isinstance(record, dict) and "title" in record:
        return [str(record.get("title", "")), str(record.get("abstract", ""))]
    # Checked before the generic-sequence branch: a 2-character string is a
    # two-element sequence and would otherwise be split into title/abstract.
    if isinstance(record, str):
        raise TypeError(
            "encode_articles needs a title and an abstract; got a bare string. "
            "Pass a CaseRecord or a (title, abstract) pair."
        )
    if isinstance(record, (tuple, list)) and len(record) == 2:
        return [str(record[0]), str(record[1])]
    raise TypeError(
        f"cannot read a (title, abstract) pair from {type(record).__name__}"
    )
