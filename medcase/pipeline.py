"""The retrieval pipeline: FAISS -> cross-encoder -> generation.

The whole point of the reranker sits in the gap between two numbers. The
bi-encoder compresses a case into one 768-dim vector and MedCPT's query encoder
only reads the first 64 tokens of the query, so a paragraph-long presentation is
mostly discarded before the search even runs. The cross-encoder reads the query
and the candidate together, up to 512 tokens, and can therefore use the detail
the first stage threw away.

That is why retrieve-deep-then-rerank works: stage one needs only to get the
right case somewhere into the top 50, and stage two decides the top 5 that
actually reach the model.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .corpus import CaseRecord
from .index import CaseIndex
from .reranker import CrossEncoderReranker, ScoredDocument

logger = logging.getLogger(__name__)


@dataclass
class RetrievalTrace:
    """Where the time went and what the reranker changed.

    Kept per-query because "the reranker helps" is only checkable if you can
    see which cases it promoted and what that cost in latency.
    """

    query: str
    n_candidates: int = 0
    n_returned: int = 0
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    generate_ms: float = 0.0
    reranked: bool = True
    promotions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_ms(self) -> float:
        return self.retrieve_ms + self.rerank_ms + self.generate_ms

    def summary(self) -> str:
        bits = [
            f"retrieve {self.retrieve_ms:.0f}ms",
            f"rerank {self.rerank_ms:.0f}ms" if self.reranked else "rerank off",
        ]
        if self.generate_ms:
            bits.append(f"generate {self.generate_ms:.0f}ms")
        return f"{self.n_candidates}->{self.n_returned} | " + " | ".join(bits)


@dataclass
class CaseSearchResult:
    query: str
    documents: list[ScoredDocument]
    trace: RetrievalTrace
    answer: Any | None = None

    @property
    def records(self) -> list[CaseRecord]:
        return [d.document for d in self.documents]

    @property
    def citations(self) -> list[str]:
        return [r.pmid for r in self.records]


class MedicalCaseAgent:
    """Retrieve, rerank, and optionally answer.

    ``retrieve_k`` is the depth handed to the reranker and ``top_k`` is what
    reaches the generator. Depth is the knob that matters: reranking cannot
    recover a case FAISS never returned, so recall@retrieve_k is the ceiling on
    everything downstream.
    """

    def __init__(
        self,
        index: CaseIndex,
        embedder: Any,
        reranker: CrossEncoderReranker | None = None,
        generator: Any | None = None,
        *,
        retrieve_k: int = 50,
        top_k: int = 5,
    ) -> None:
        self.index = index
        self.embedder = embedder
        self.reranker = reranker
        self.generator = generator
        self.retrieve_k = retrieve_k
        self.top_k = top_k

    def retrieve(self, query: str, k: int | None = None) -> list[tuple[CaseRecord, float]]:
        """Stage one: approximate, cheap, and the recall ceiling for stage two."""
        return self.index.search(query, self.embedder, k or self.retrieve_k)

    def search(
        self,
        query: str,
        *,
        retrieve_k: int | None = None,
        top_k: int | None = None,
        rerank: bool = True,
        generate: bool = False,
    ) -> CaseSearchResult:
        retrieve_k = retrieve_k or self.retrieve_k
        top_k = top_k or self.top_k
        trace = RetrievalTrace(query=query, reranked=rerank and self.reranker is not None)

        t0 = time.perf_counter()
        hits = self.retrieve(query, retrieve_k)
        trace.retrieve_ms = (time.perf_counter() - t0) * 1000
        trace.n_candidates = len(hits)

        if not hits:
            trace.n_returned = 0
            return CaseSearchResult(query=query, documents=[], trace=trace)

        records = [r for r, _ in hits]
        scores = [s for _, s in hits]

        if trace.reranked:
            t0 = time.perf_counter()
            docs = self.reranker.rerank(
                query,
                records,
                top_k=top_k,
                text_of=lambda r: r.text,
                retrieval_scores=scores,
            )
            trace.rerank_ms = (time.perf_counter() - t0) * 1000
            trace.promotions = [
                {
                    "pmid": d.document.pmid,
                    "from_rank": d.retrieval_rank,
                    "to_rank": d.rerank_rank,
                    "gained": d.rank_delta,
                }
                for d in docs
                if d.rank_delta and d.rank_delta > 0
            ]
        else:
            # Baseline arm: embedding order, untouched.
            docs = [
                ScoredDocument(
                    document=r,
                    text=r.text,
                    rerank_score=s,
                    retrieval_score=s,
                    retrieval_rank=i,
                    rerank_rank=i,
                )
                for i, (r, s) in enumerate(zip(records, scores))
            ][:top_k]

        trace.n_returned = len(docs)

        answer = None
        if generate:
            if self.generator is None:
                raise ValueError("generate=True but no generator was configured")
            t0 = time.perf_counter()
            answer = self.generator.generate(query, docs, max_cases=top_k)
            trace.generate_ms = (time.perf_counter() - t0) * 1000

        return CaseSearchResult(query=query, documents=docs, trace=trace, answer=answer)

    def ask(self, query: str, **kwargs) -> CaseSearchResult:
        """Search and generate in one call."""
        return self.search(query, generate=True, **kwargs)

    @classmethod
    def from_directory(
        cls,
        index_dir: str,
        *,
        embedder: Any = None,
        reranker: CrossEncoderReranker | None = None,
        generator: Any | None = None,
        **kwargs,
    ) -> "MedicalCaseAgent":
        from .embed import MedCPTEmbedder

        return cls(
            index=CaseIndex.load(index_dir),
            embedder=embedder or MedCPTEmbedder(),
            reranker=reranker if reranker is not None else CrossEncoderReranker(),
            generator=generator,
            **kwargs,
        )
