"""FAISS index over the case corpus.

This layer owns the vectors *and* the records they came from, in the same
order. FAISS only ever hands back row numbers; if those rows lived somewhere
else, a drifted corpus file would silently turn row 7 into the wrong PMID and
every citation downstream would be fiction. Keeping both here means a hit can
always be traced to a real article, and a mismatch is an error rather than a
plausible-looking wrong answer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np

from .corpus import CaseRecord, load_corpus, save_corpus

logger = logging.getLogger(__name__)

INDEX_FILE = "index.faiss"
RECORDS_FILE = "records.jsonl"
META_FILE = "meta.json"

DEFAULT_DIM = 768  # MedCPT bi-encoder hidden size


def _as_matrix(vectors: Any, *, dim: int | None = None) -> np.ndarray:
    """FAISS wants a C-contiguous float32 2-D array; torch/np give many shapes."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D array of vectors, got shape {arr.shape}")
    if dim is not None and arr.shape[1] != dim:
        raise ValueError(
            f"embedding dim mismatch: index is {dim}-d, got {arr.shape[1]}-d vectors"
        )
    return np.ascontiguousarray(arr)


def _encode_articles(
    embedder: Any, records: Sequence[CaseRecord], batch_size: int
) -> np.ndarray:
    """Embed records with the article tower at a caller-chosen batch size.

    The forward-pass batch lives on the embedder, so honouring ``build``'s
    ``batch_size`` means setting it there and putting it back. Chunking the
    records out here instead would leave the real batch untouched and make the
    argument a lie.
    """
    previous = getattr(embedder, "batch_size", None)
    if previous is not None and batch_size:
        embedder.batch_size = int(batch_size)
    try:
        return _as_matrix(embedder.encode_articles(list(records)))
    finally:
        if previous is not None:
            embedder.batch_size = previous


class CaseIndex:
    """Exact inner-product search over case reports, paired with their records.

    ``IndexFlatIP`` is a brute-force scan. Inner product is the metric MedCPT
    was trained on, so that part is not a choice. Flat is: at this corpus size
    -- tens to thousands of case reports -- an exact scan returns true
    nearest neighbours, while HNSW/IVF would approximate them and add tuning
    knobs (and IVF a training step) in exchange. Recall here is the thing being
    evaluated, so give up none of it until a measurement says the scan is the
    bottleneck.
    """

    def __init__(
        self,
        index: faiss.Index | None = None,
        records: Sequence[CaseRecord] | None = None,
        embedding_dim: int = DEFAULT_DIM,
    ) -> None:
        self.index = index if index is not None else faiss.IndexFlatIP(embedding_dim)
        self._records: list[CaseRecord] = list(records or [])
        self._check_aligned()

    # -- invariants -----------------------------------------------------

    def _check_aligned(self) -> None:
        """One record per vector, or the row-number -> PMID mapping is a lie."""
        if self.index.ntotal != len(self._records):
            raise ValueError(
                f"index/record mismatch: {self.index.ntotal} vectors but "
                f"{len(self._records)} records"
            )

    @property
    def records(self) -> list[CaseRecord]:
        """Records in index-row order. Row ``i`` from FAISS is ``records[i]``."""
        return self._records

    @property
    def embedding_dim(self) -> int:
        # The index is the single source of truth for dimensionality.
        return self.index.d

    def __len__(self) -> int:
        return self.index.ntotal

    def __repr__(self) -> str:
        return f"CaseIndex(vectors={len(self)}, dim={self.embedding_dim})"

    # -- construction ---------------------------------------------------

    @classmethod
    def build(
        cls,
        records: Sequence[CaseRecord],
        embedder: Any,
        batch_size: int = 32,
    ) -> "CaseIndex":
        """Embed ``records`` with the article tower and index them.

        Whole records are handed to the embedder, not ``rec.text``: MedCPT
        wants ``title [SEP] abstract`` as a text pair, and flattening it to one
        string throws away the ``token_type_ids`` the model uses to tell the
        two fields apart.
        """
        records = list(records)
        if not records:
            raise ValueError("cannot build an index from zero records")

        vectors = _encode_articles(embedder, records, batch_size)
        if vectors.shape[0] != len(records):
            raise ValueError(
                f"embedder returned {vectors.shape[0]} vectors for "
                f"{len(records)} records"
            )

        # Vectors go in exactly as the embedder produced them. Normalizing here
        # would quietly turn dot-product scores into cosine ones; that choice
        # belongs to the embedder, which knows how the model was trained. An
        # embedder set to normalize still works: on unit vectors inner product
        # *is* cosine, so the same flat index serves both.
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        logger.info("indexed %d records at dim %d", index.ntotal, index.d)
        return cls(index=index, records=records)

    # -- search ---------------------------------------------------------

    def search_vectors(
        self, vecs: np.ndarray, k: int
    ) -> list[list[tuple[CaseRecord, float]]]:
        """Batched search. One result list per input row, best-first."""
        queries = _as_matrix(vecs, dim=self.embedding_dim)

        # FAISS pads short result rows with id -1 when k exceeds ntotal. Those
        # are not documents; clamping first and filtering after keeps a bogus
        # record from ever being constructed.
        k = min(max(int(k), 0), len(self))
        if k == 0:
            return [[] for _ in range(queries.shape[0])]

        scores, ids = self.index.search(queries, k)
        results: list[list[tuple[CaseRecord, float]]] = []
        for row_ids, row_scores in zip(ids, scores):
            results.append(
                [
                    (self._records[int(i)], float(s))
                    for i, s in zip(row_ids, row_scores)
                    if i != -1
                ]
            )
        return results

    def search(
        self, query: str, embedder: Any, k: int
    ) -> list[tuple[CaseRecord, float]]:
        """Top-``k`` records for one clinical question, best-first."""
        # Query tower, not the article tower: the wrong one still returns a
        # well-formed 768-d vector, it just lands in the wrong neighbourhood.
        vec = embedder.encode_queries([query])
        return self.search_vectors(vec, k)[0]

    # -- persistence ----------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """Write vectors, records, and shape metadata to ``directory``."""
        self._check_aligned()
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out / INDEX_FILE))
        save_corpus(self._records, out / RECORDS_FILE)
        meta = {
            "count": len(self),
            "embedding_dim": self.embedding_dim,
            "metric": "inner_product",
            "index_type": type(self.index).__name__,
        }
        (out / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("saved %d vectors to %s", len(self), out)

    @classmethod
    def load(cls, directory: str | Path) -> "CaseIndex":
        """Read back an index saved by :meth:`save`."""
        src = Path(directory)
        index_path = src / INDEX_FILE
        if not index_path.exists():
            raise FileNotFoundError(
                f"No FAISS index at {index_path}. Run scripts/build_index.py first."
            )

        index = faiss.read_index(str(index_path))
        records = load_corpus(src / RECORDS_FILE)

        meta_path = src / META_FILE
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            expected_dim = meta.get("embedding_dim")
            if expected_dim is not None and expected_dim != index.d:
                raise ValueError(
                    f"{meta_path} records dim {expected_dim} but the index is "
                    f"{index.d}-d; the two files are from different builds"
                )

        # __init__ raises if the vector count and the record count disagree --
        # a half-updated directory must fail here, not at citation time.
        return cls(index=index, records=records)
