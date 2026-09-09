"""Reranker behaviour that must hold regardless of which model is loaded.

These use a stub scorer rather than a real cross-encoder: the ordering,
caching and truncation logic is what breaks in practice, and pinning it to a
model's actual logits would make the tests a weather report on the weights.
The one real-model test is marked slow.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medcase.corpus import CaseRecord
from medcase.reranker import CrossEncoderReranker, ScoredDocument, _default_text_of


class StubReranker(CrossEncoderReranker):
    """Scores by term overlap so expectations are exactly computable."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.score_calls = 0

    def score(self, query, texts):
        if not texts:
            return []
        self.score_calls += 1
        q = set(query.lower().split())
        return [float(len(q & set(t.lower().split()))) for t in texts]


def rec(pmid, title, abstract=""):
    return CaseRecord(pmid=pmid, title=title, abstract=abstract or title)


def test_rerank_orders_by_score_descending():
    rr = StubReranker()
    docs = [rec("1", "pancreatitis"), rec("2", "myocarditis fever chest"), rec("3", "fever")]
    out = rr.rerank("fever chest pain", docs)
    assert [d.document.pmid for d in out] == ["2", "3", "1"]
    assert out[0].rerank_score > out[1].rerank_score >= out[2].rerank_score


def test_rerank_records_both_ranks_and_delta():
    rr = StubReranker()
    docs = [rec("1", "unrelated"), rec("2", "fever chest")]
    out = rr.rerank("fever chest", docs)
    top = out[0]
    assert top.retrieval_rank == 1 and top.rerank_rank == 0
    assert top.rank_delta == 1  # promoted one place


def test_rank_delta_is_none_without_both_ranks():
    assert ScoredDocument(document=None, text="", rerank_score=0.0).rank_delta is None


def test_top_k_truncates_after_sorting_not_before():
    rr = StubReranker()
    # The best document sits last; a naive implementation that slices first loses it.
    docs = [rec(str(i), "filler") for i in range(5)] + [rec("best", "fever chest pain")]
    out = rr.rerank("fever chest pain", docs, top_k=1)
    assert len(out) == 1 and out[0].document.pmid == "best"


def test_empty_documents_returns_empty():
    assert StubReranker().rerank("q", []) == []
    assert StubReranker().score("q", []) == []


def test_single_document():
    out = StubReranker().rerank("fever", [rec("1", "fever")])
    assert len(out) == 1 and out[0].rerank_rank == 0


def test_retrieval_scores_are_preserved():
    rr = StubReranker()
    docs = [rec("1", "a"), rec("2", "fever")]
    out = rr.rerank("fever", docs, retrieval_scores=[0.9, 0.1])
    by_pmid = {d.document.pmid: d for d in out}
    assert by_pmid["1"].retrieval_score == 0.9
    assert by_pmid["2"].retrieval_score == 0.1


def test_text_of_selects_the_scored_field():
    rr = StubReranker()
    docs = [{"page_content": "fever chest"}, {"page_content": "nothing"}]
    out = rr.rerank("fever chest", docs, text_of=lambda d: d["page_content"])
    assert out[0].document["page_content"] == "fever chest"


def test_default_text_of_handles_common_shapes():
    assert _default_text_of(rec("1", "T", "A")) == "T\n\nA"

    class Obj:
        page_content = "pc"

    assert _default_text_of(Obj()) == "pc"
    assert _default_text_of("plain") == "plain"


def test_unicode_and_long_text_do_not_crash():
    rr = StubReranker()
    docs = [rec("1", "fièvre — 38°C αβγ"), rec("2", "x " * 5000)]
    out = rr.rerank("fièvre", docs)
    assert len(out) == 2


@pytest.mark.slow
def test_real_cross_encoder_separates_relevant_from_irrelevant():
    """The only test that asserts on real weights, and only on sign/order."""
    rr = CrossEncoderReranker(batch_size=4)
    q = "myasthenia gravis presenting with ptosis and diplopia"
    docs = [
        rec("irrelevant", "Acute pancreatitis following ERCP in a diabetic patient."),
        rec("relevant", "Ocular myasthenia gravis presenting with ptosis and diplopia."),
    ]
    out = rr.rerank(q, docs)
    assert out[0].document.pmid == "relevant"
    assert out[0].rerank_score > out[1].rerank_score
    assert rr.last_metrics.candidates == 2


def test_cache_key_is_order_sensitive_and_collision_safe():
    k = CrossEncoderReranker._cache_key
    assert k("a", "b") != k("b", "a")
    # A naive f"{q}{d}" key would make these collide.
    assert k("ab", "c") != k("a", "bc")
    assert k("a", "b") == k("a", "b")


def test_cache_eviction_is_bounded():
    rr = CrossEncoderReranker(cache_size=3)
    for i in range(5):
        rr._remember(f"k{i}", float(i))
    assert len(rr._cache) == 3
    assert "k0" not in rr._cache and "k4" in rr._cache  # FIFO


def test_clear_cache_empties_it():
    rr = CrossEncoderReranker()
    rr._remember("k", 1.0)
    rr.clear_cache()
    assert rr._cache == {}


def test_model_is_not_loaded_until_used():
    assert CrossEncoderReranker().is_loaded is False
