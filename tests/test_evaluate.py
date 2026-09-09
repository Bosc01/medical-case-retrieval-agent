"""Metric maths pinned to values worked out by hand.

Every expected number below is derived in a comment from the definition, not
read back out of the implementation -- a test that asserts whatever the code
already returns pins nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medcase.evaluate import (  # noqa: E402
    EvalResult,
    NoRelevantDocuments,
    average_precision,
    compare,
    evaluate_run,
    format_comparison,
    format_table,
    judged_at_k,
    mrr,
    ndcg_at_k,
    paired_bootstrap_p,
    precision_at_k,
    recall_at_k,
    sign_test_p,
)

# ---------------------------------------------------------------------
# Fixture A -- one ranking, binary qrels, used for P / R / MRR / AP / nDCG.
#
#   ranked   : d1 d2 d3 d4 d5 d6 d7 d8 d9 d10
#   relevant : {d2, d3, d7, d11}      (4 relevant; d11 is never retrieved)
#   hits at 1-indexed ranks 2, 3, 7
# ---------------------------------------------------------------------
RANKED_A = ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10"]
RELEVANT_A = {"d2", "d3", "d7", "d11"}


def test_precision_at_k_hand_computed():
    # P@k = (relevant in top k) / k
    #   P@1  = 0/1  = 0.0        (d1 is not relevant)
    #   P@3  = 2/3  = 0.6666...  (d2, d3)
    #   P@5  = 2/5  = 0.4
    #   P@10 = 3/10 = 0.3        (d2, d3, d7)
    assert precision_at_k(RANKED_A, RELEVANT_A, 1) == 0.0
    assert precision_at_k(RANKED_A, RELEVANT_A, 3) == pytest.approx(2 / 3)
    assert precision_at_k(RANKED_A, RELEVANT_A, 5) == pytest.approx(0.4)
    assert precision_at_k(RANKED_A, RELEVANT_A, 10) == pytest.approx(0.3)


def test_precision_divides_by_k_not_by_documents_returned():
    # Only 2 documents returned, 2 of them relevant, but P@5 divides by 5:
    #   P@5 = 2/5 = 0.4, NOT 2/2 = 1.0
    assert precision_at_k(["d2", "d3"], RELEVANT_A, 5) == pytest.approx(0.4)


def test_recall_at_k_hand_computed():
    # R@k = (relevant in top k) / |relevant| , |relevant| = 4
    #   R@3  = 2/4 = 0.5
    #   R@5  = 2/4 = 0.5
    #   R@10 = 3/4 = 0.75   (d11 was never retrieved and still counts against)
    assert recall_at_k(RANKED_A, RELEVANT_A, 3) == pytest.approx(0.5)
    assert recall_at_k(RANKED_A, RELEVANT_A, 5) == pytest.approx(0.5)
    assert recall_at_k(RANKED_A, RELEVANT_A, 10) == pytest.approx(0.75)


def test_mrr_hand_computed():
    # First relevant document is d2 at 1-indexed rank 2 -> 1/2 = 0.5
    assert mrr(RANKED_A, RELEVANT_A) == pytest.approx(0.5)


def test_mrr_is_zero_when_nothing_relevant_was_retrieved():
    # Relevant documents exist (d99), the run returned none of them.
    # This 0.0 is an earned failure, not an unjudged query.
    assert mrr(RANKED_A, {"d99"}) == 0.0


def test_average_precision_hand_computed():
    # AP = (sum of P@rank at each retrieved relevant doc) / |relevant|
    #   d2 at rank 2 -> 1/2 = 0.5
    #   d3 at rank 3 -> 2/3 = 0.666666...
    #   d7 at rank 7 -> 3/7 = 0.428571...
    #   d11 never retrieved -> contributes 0
    #   AP = (0.5 + 0.6666666667 + 0.4285714286) / 4
    #      = 1.5952380952 / 4 = 0.3988095238095238
    assert average_precision(RANKED_A, RELEVANT_A) == pytest.approx(0.3988095238095238)


def test_ndcg_at_5_binary_hand_computed():
    # Binary grades, linear gain, log2(i+2) discount (i is 0-indexed).
    #   DCG@5  = 0/log2(2) + 1/log2(3) + 1/log2(4) + 0/log2(5) + 0/log2(6)
    #          = 0.6309297535714575 + 0.5
    #          = 1.1309297535714575
    #   ideal ordering of the qrels grades = [1, 1, 1, 1], truncated to 5
    #   IDCG@5 = 1/log2(2) + 1/log2(3) + 1/log2(4) + 1/log2(5)
    #          = 1.0 + 0.6309297535714575 + 0.5 + 0.4306765580733931
    #          = 2.5616063116448506
    #   nDCG@5 = 1.1309297535714575 / 2.5616063116448506 = 0.44149241373678083
    assert ndcg_at_k(RANKED_A, RELEVANT_A, 5) == pytest.approx(0.44149241373678083)


def test_ndcg_at_10_binary_hand_computed():
    #   DCG@10 = 1/log2(3) + 1/log2(4) + 1/log2(8)   (ranks 2, 3, 7)
    #          = 0.6309297535714575 + 0.5 + 0.3333333333333333
    #          = 1.4642630869047908
    #   only 4 grades exist, so IDCG@10 == IDCG@5 = 2.5616063116448506
    #   nDCG@10 = 1.4642630869047908 / 2.5616063116448506 = 0.5716190970674814
    assert ndcg_at_k(RANKED_A, RELEVANT_A, 10) == pytest.approx(0.5716190970674814)


# ---------------------------------------------------------------------
# Fixture B -- graded qrels.
#
#   grades : d1=3, d2=2, d3=1   (d0 is unjudged -> gain 0)
#   ranked : d3 d0 d1 d2
# ---------------------------------------------------------------------
GRADES_B = {"d1": 3, "d2": 2, "d3": 1}
RANKED_B = ["d3", "d0", "d1", "d2"]


def test_ndcg_graded_linear_gain_hand_computed():
    #   DCG@4  = 1/log2(2) + 0/log2(3) + 3/log2(4) + 2/log2(5)
    #          = 1.0 + 0 + 1.5 + 0.8613531161467861
    #          = 3.3613531161467862
    #   ideal grades sorted descending = [3, 2, 1]
    #   IDCG@4 = 3/log2(2) + 2/log2(3) + 1/log2(4)
    #          = 3.0 + 1.261859507142915 + 0.5
    #          = 4.7618595071429155
    #   nDCG@4 = 3.3613531161467862 / 4.7618595071429155 = 0.7058908628246313
    assert ndcg_at_k(RANKED_B, GRADES_B, 4) == pytest.approx(0.7058908628246313)


def test_ndcg_graded_exponential_gain_hand_computed():
    # gain = 2**g - 1
    #   DCG@4  = (2-1)/log2(2) + 0 + (8-1)/log2(4) + (4-1)/log2(5)
    #          = 1.0 + 3.5 + 1.2920296742201799
    #          = 5.79202967422018
    #   IDCG@4 = 7/log2(2) + 3/log2(3) + 1/log2(4)
    #          = 7.0 + 1.8927892607143724 + 0.5
    #          = 9.392789260714373
    #   nDCG@4 = 5.79202967422018 / 9.392789260714373 = 0.6166463990037038
    got = ndcg_at_k(RANKED_B, GRADES_B, 4, gain="exponential")
    assert got == pytest.approx(0.6166463990037038)
    # The two schemes are genuinely different scales, not a rounding apart.
    assert got != pytest.approx(ndcg_at_k(RANKED_B, GRADES_B, 4, gain="linear"))


def test_ndcg_idcg_comes_from_qrels_not_from_what_was_retrieved():
    # Three relevant documents judged; the run returned exactly one, at rank 1.
    # Normalising against the retrieved list would give 1.0. The correct IDCG
    # is over the qrels:
    #   DCG@3  = 1/log2(2) = 1.0
    #   IDCG@3 = 1/log2(2) + 1/log2(3) + 1/log2(4) = 2.1309297535714578
    #   nDCG@3 = 1.0 / 2.1309297535714578 = 0.46927872602275644
    got = ndcg_at_k(["a"], {"a": 1, "b": 1, "c": 1}, 3)
    assert got == pytest.approx(0.46927872602275644)
    assert got < 1.0


def test_perfect_ranking():
    # All three relevant documents at the top, in order.
    ranked = ["r1", "r2", "r3", "x1", "x2"]
    relevant = {"r1", "r2", "r3"}
    assert mrr(ranked, relevant) == 1.0
    assert average_precision(ranked, relevant) == 1.0  # (1/1 + 2/2 + 3/3)/3
    assert precision_at_k(ranked, relevant, 1) == 1.0
    assert precision_at_k(ranked, relevant, 3) == 1.0
    # P@5 cannot reach 1.0: only 3 relevant documents exist, so 3/5 = 0.6.
    assert precision_at_k(ranked, relevant, 5) == pytest.approx(0.6)
    assert recall_at_k(ranked, relevant, 3) == 1.0
    assert recall_at_k(ranked, relevant, 5) == 1.0
    for k in (1, 3, 5, 10):
        # DCG == IDCG at every cutoff when the ideal order is the actual order.
        assert ndcg_at_k(ranked, relevant, k) == pytest.approx(1.0)


def test_graded_perfect_ranking_is_one_for_both_gain_schemes():
    ranked = ["d1", "d2", "d3"]
    for scheme in ("linear", "exponential"):
        assert ndcg_at_k(ranked, GRADES_B, 3, gain=scheme) == pytest.approx(1.0)


def test_judged_at_k_hand_computed():
    # qrels judge d2, d3 (relevant) and d1, d4 (judged, grade 0).
    # Top 5 of RANKED_A = d1 d2 d3 d4 d5 -> 4 of 5 judged = 0.8
    judgements = {"d1": 0, "d2": 1, "d3": 1, "d4": 0}
    assert judged_at_k(RANKED_A, judgements, 5) == pytest.approx(0.8)
    # Top 10 -> still 4 judged, denominator is min(10, 10) = 10 -> 0.4
    assert judged_at_k(RANKED_A, judgements, 10) == pytest.approx(0.4)


# -- refusals ---------------------------------------------------------


@pytest.mark.parametrize("qrels", [{}, {"d1": 0, "d2": 0}, set()])
def test_primitives_refuse_a_query_with_no_relevant_documents(qrels):
    # Undefined, not 0.0: an unjudged query must never be mistaken for a
    # query the system failed.
    with pytest.raises(NoRelevantDocuments):
        precision_at_k(RANKED_A, qrels, 5)
    with pytest.raises(NoRelevantDocuments):
        recall_at_k(RANKED_A, qrels, 5)
    with pytest.raises(NoRelevantDocuments):
        mrr(RANKED_A, qrels)
    with pytest.raises(NoRelevantDocuments):
        ndcg_at_k(RANKED_A, qrels, 5)
    with pytest.raises(NoRelevantDocuments):
        average_precision(RANKED_A, qrels)


def test_duplicate_doc_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate doc ids"):
        precision_at_k(["d2", "d2", "d3"], RELEVANT_A, 3)


def test_non_positive_k_is_rejected():
    with pytest.raises(ValueError, match="k must be positive"):
        precision_at_k(RANKED_A, RELEVANT_A, 0)


# -- evaluate_run -----------------------------------------------------


def test_evaluate_run_excludes_queries_with_no_positive_judgement():
    runs = {
        "q1": ["a", "b", "c"],  # a is relevant, at rank 1 -> P@1 = 1.0
        "q2": ["a", "b", "c"],  # no qrels entry at all -> unknown, not wrong
        "q3": ["a", "b", "c"],  # judged, but every grade is 0
    }
    qrels = {"q1": {"a": 1}, "q3": {"a": 0, "b": 0}}
    res = evaluate_run(runs, qrels, ks=(1,), name="run")

    assert res.n_queries == 1
    assert res.skipped_no_qrels == ["q2", "q3"]
    assert res.n_skipped == 2
    assert set(res.per_query) == {"q1"}
    # The mean is over the 1 scorable query (1.0), NOT over 3 queries
    # with two zeros folded in (which would be 1/3).
    assert res.metrics["P@1"] == pytest.approx(1.0)
    assert res.metrics["P@1"] != pytest.approx(1 / 3)
    assert res.metrics["MRR"] == pytest.approx(1.0)


def test_evaluate_run_counts_judged_queries_the_run_did_not_answer():
    runs = {"q1": ["a", "b"]}
    qrels = {"q1": {"a": 1}, "q2": {"z": 1}}
    res = evaluate_run(runs, qrels, ks=(1,))
    assert res.n_queries == 1
    assert res.missing_run == ["q2"]
    assert "q2" not in res.per_query


def test_evaluate_run_produces_no_metrics_when_nothing_is_scorable():
    res = evaluate_run({"q1": ["a"]}, {"q1": {"a": 0}}, ks=(1, 5))
    assert res.n_queries == 0
    # Empty, not a dict of zeros -- never print a metric the run did not make.
    assert res.metrics == {}
    assert res.per_query == {}
    assert "P@1" not in format_table([res])


def test_evaluate_run_metric_names_follow_the_requested_cutoffs():
    res = evaluate_run({"q1": ["a", "b"]}, {"q1": {"a": 1}}, ks=(1, 2))
    assert set(res.metrics) == {
        "MRR",
        "MAP",
        "nDCG@1",
        "nDCG@2",
        "P@1",
        "P@2",
        "R@1",
        "R@2",
        "judged@1",
        "judged@2",
    }
    assert res.ks == (1, 2)


def test_evaluate_run_per_query_matches_the_primitives():
    runs = {"qA": RANKED_A}
    qrels = {"qA": {d: 1 for d in RELEVANT_A}}
    res = evaluate_run(runs, qrels, ks=(5,))
    # Same hand-computed values as above, reached through the harness.
    assert res.per_query["qA"]["P@5"] == pytest.approx(0.4)
    assert res.per_query["qA"]["MRR"] == pytest.approx(0.5)
    assert res.per_query["qA"]["MAP"] == pytest.approx(0.3988095238095238)
    assert res.per_query["qA"]["nDCG@5"] == pytest.approx(0.44149241373678083)


# -- significance -----------------------------------------------------


def test_sign_test_exact_hand_computed_values():
    # Two-sided exact sign test, ties dropped: p = 2 * sum_{i<=m} C(n,i) / 2**n
    # 10 queries, treatment wins all: m = 0
    #   p = 2 * C(10,0) / 2**10 = 2/1024 = 0.001953125
    assert sign_test_p([0.0] * 10, [1.0] * 10) == pytest.approx(0.001953125)
    # 3 queries, treatment wins all: p = 2 * 1 / 8 = 0.25.
    # A clean sweep of three queries is not evidence, however big the deltas.
    assert sign_test_p([0.0] * 3, [1.0] * 3) == pytest.approx(0.25)
    # 8 non-tied queries, 7 wins 1 loss: m = 1
    #   p = 2 * (C(8,0) + C(8,1)) / 2**8 = 2 * 9 / 256 = 0.0703125
    assert sign_test_p([0.0] * 8, [1.0] * 7 + [-1.0]) == pytest.approx(0.0703125)


def test_sign_test_all_ties_is_p_one():
    # No non-tied query -> n = 0 -> no evidence of any difference.
    assert sign_test_p([0.3, 0.4, 0.5], [0.3, 0.4, 0.5]) == 1.0


def test_sign_test_drops_ties_before_counting():
    # 2 wins, 0 losses, 3 ties -> n = 2 -> p = 2 * C(2,0) / 4 = 0.5
    a = [0.0, 0.0, 0.5, 0.5, 0.5]
    b = [1.0, 1.0, 0.5, 0.5, 0.5]
    assert sign_test_p(a, b) == pytest.approx(0.5)


def test_bootstrap_on_identical_runs_is_p_one():
    # Every difference is 0, so every re-centred resample ties the observed
    # mean of 0: p = (B + 1) / (B + 1) = 1.0
    vals = [0.1, 0.9, 0.4, 0.4, 0.7]
    assert paired_bootstrap_p(vals, vals, n_resamples=500) == 1.0


def test_bootstrap_is_deterministic_and_bounded():
    a = [0.1, 0.2, 0.3, 0.2, 0.1, 0.4, 0.2, 0.3, 0.1, 0.2]
    b = [0.5, 0.6, 0.7, 0.5, 0.6, 0.8, 0.6, 0.7, 0.5, 0.6]
    p1 = paired_bootstrap_p(a, b, n_resamples=2000, seed=7)
    p2 = paired_bootstrap_p(a, b, n_resamples=2000, seed=7)
    assert p1 == p2
    # Floor of the estimator with the +1 correction; it can never report 0.
    assert 1 / 2001 <= p1 <= 1.0
    assert p1 < 0.05  # consistent, sizeable improvement on all 10 queries


def test_bootstrap_on_noise_is_not_significant():
    a = [0.5, 0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4]
    b = [0.4, 0.2, 0.8, 0.4, 0.6, 0.3, 0.7, 0.5]  # alternating +-0.1, mean 0
    assert paired_bootstrap_p(a, b, n_resamples=2000, seed=3) > 0.5


def test_paired_tests_reject_mismatched_lengths():
    with pytest.raises(ValueError):
        sign_test_p([0.1, 0.2], [0.1])
    with pytest.raises(ValueError):
        paired_bootstrap_p([0.1, 0.2], [0.1])


# -- compare ----------------------------------------------------------


def _two_runs(n=10):
    """Baseline puts the one relevant doc at rank 2, reranked at rank 1."""
    qrels = {f"q{i}": {"good": 1} for i in range(n)}
    base = {f"q{i}": ["bad", "good", "filler"] for i in range(n)}
    rer = {f"q{i}": ["good", "bad", "filler"] for i in range(n)}
    return (
        evaluate_run(base, qrels, ks=(1,), name="faiss"),
        evaluate_run(rer, qrels, ks=(1,), name="medcpt"),
    )


def test_compare_deltas_and_sign_test_hand_computed():
    base, rer = _two_runs(10)
    # MRR: baseline 1/2 on every query, reranked 1/1 on every query.
    assert base.metrics["MRR"] == pytest.approx(0.5)
    assert rer.metrics["MRR"] == pytest.approx(1.0)
    cmp = compare(base, rer, n_resamples=500)
    entry = cmp["metrics"]["MRR"]
    assert entry["delta"] == pytest.approx(0.5)
    assert entry["relative_delta"] == pytest.approx(1.0)  # 0.5 / 0.5
    assert (entry["wins"], entry["losses"], entry["ties"]) == (10, 0, 0)
    # 10 wins, 0 losses -> 2/1024
    assert entry["p_sign"] == pytest.approx(0.001953125)
    assert cmp["n_paired"] == 10


def test_compare_reports_judged_coverage_alongside_effectiveness():
    # In this fixture "bad" is absent from qrels, i.e. unjudged. The baseline
    # puts it at rank 1, so its judged@1 is 0.0 while the reranked run's is
    # 1.0: the baseline's top result is being scored 0 on an assumption, not
    # on a judgement. This is the diagnostic that says how much of a
    # comparison rests on unjudged-as-irrelevant.
    base, rer = _two_runs(10)
    assert base.metrics["judged@1"] == 0.0
    assert rer.metrics["judged@1"] == 1.0
    cmp = compare(base, rer, n_resamples=500)
    assert cmp["metrics"]["judged@1"]["delta"] == pytest.approx(1.0)


def test_compare_relative_delta_is_none_when_baseline_is_zero():
    qrels = {f"q{i}": {"good": 1} for i in range(4)}
    base = evaluate_run(
        {f"q{i}": ["bad", "filler"] for i in range(4)}, qrels, ks=(1,), name="base"
    )
    rer = evaluate_run(
        {f"q{i}": ["good", "bad"] for i in range(4)}, qrels, ks=(1,), name="rer"
    )
    assert base.metrics["MRR"] == 0.0
    entry = compare(base, rer, n_resamples=200)["metrics"]["MRR"]
    assert entry["delta"] == pytest.approx(1.0)
    # Undefined, not inf and not 0.0.
    assert entry["relative_delta"] is None


def test_compare_warns_when_the_sample_is_too_small_to_be_significant():
    base, rer = _two_runs(3)
    cmp = compare(base, rer, n_resamples=200)
    assert cmp["metrics"]["MRR"]["p_sign"] == pytest.approx(0.25)
    assert any("3 paired queries" in w for w in cmp["warnings"])


def test_compare_warns_when_query_sets_differ():
    qrels = {"q1": {"good": 1}, "q2": {"good": 1}}
    base = evaluate_run({"q1": ["good"], "q2": ["bad", "good"]}, qrels, ks=(1,))
    rer = evaluate_run({"q1": ["good"]}, qrels, ks=(1,))
    cmp = compare(base, rer, n_resamples=200)
    assert cmp["n_paired"] == 1
    assert any("query sets differ" in w for w in cmp["warnings"])


def test_compare_reports_no_delta_for_a_metric_only_one_run_produced():
    qrels = {f"q{i}": {"good": 1} for i in range(3)}
    runs = {f"q{i}": ["good", "bad"] for i in range(3)}
    a = evaluate_run(runs, qrels, ks=(1,), name="a")
    b = evaluate_run(runs, qrels, ks=(1, 2), name="b")
    entry = compare(a, b, n_resamples=200)["metrics"]["P@2"]
    assert entry["baseline"] is None
    assert "delta" not in entry
    assert "no delta" in entry["note"]


def test_compare_of_a_run_with_itself_shows_nothing():
    base, _ = _two_runs(6)
    cmp = compare(base, base, n_resamples=200)
    for entry in cmp["metrics"].values():
        assert entry["delta"] == 0.0
        assert entry["p_sign"] == 1.0
        assert entry["p_bootstrap"] == 1.0


# -- formatting -------------------------------------------------------


def test_format_table_marks_absent_metrics_without_inventing_a_number():
    qrels = {"q1": {"good": 1}}
    runs = {"q1": ["good", "bad", "other"]}
    a = evaluate_run(runs, qrels, ks=(1,), name="a")
    b = evaluate_run(runs, qrels, ks=(1, 3), name="b")
    table = format_table([a, b])
    lines = {ln.split()[0]: ln for ln in table.splitlines() if ln.strip()}

    assert "P@3" in lines
    # a never computed P@3: the cell is a dash, not 0.0 and not a copy of P@1.
    cells = lines["P@3"].split()
    assert cells[1] == "-"
    assert cells[2] == "0.3333"  # b: 1 relevant doc in the top 3 -> 1/3
    assert "queries scored" in table
    assert "skipped (unjudged)" in table


def test_format_table_columns_are_aligned():
    a, b = _two_runs(4)
    lines = [ln for ln in format_table([a, b]).splitlines() if ln.strip()]
    # Fixed-width rendering: every line, rules included, is the same length.
    assert len({len(ln) for ln in lines}) == 1
    # And every numeric cell ends at the same column as the header it is under.
    header = lines[0]
    for line in lines:
        if set(line) <= {"-", " "}:
            continue
        assert line.index(line.strip()[0]) == header.index("metric")


def test_format_table_handles_no_results():
    assert format_table([]) == "no results"


def test_format_comparison_surfaces_warnings():
    base, rer = _two_runs(3)
    text = format_comparison(compare(base, rer, n_resamples=200))
    assert "WARNING" in text
    assert "uncorrected" in text
    assert "MRR" in text


def test_eval_result_summary_mentions_skipped_queries():
    res = EvalResult(
        name="x", n_queries=2, metrics={}, per_query={}, skipped_no_qrels=["q9"]
    )
    assert "2 queries scored" in res.summary()
    assert "1 skipped" in res.summary()
