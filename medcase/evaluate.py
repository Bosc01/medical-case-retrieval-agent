"""Retrieval evaluation for the Medical Case Retrieval Agent.

This module decides whether reranking actually helped, so it is written to
disappoint rather than to flatter.

Two conventions matter more than the arithmetic:

**A query with no positively judged document is skipped, not scored 0.0.**
Precision, recall and nDCG are all undefined when the qrels contain nothing
relevant to find. Averaging a 0.0 in would let an unjudged query masquerade as
a system failure, and would drag both systems toward whatever fraction of the
query set happens to be judged. The primitives raise ``NoRelevantDocuments``;
``evaluate_run`` filters those queries out up front and reports how many.

**A missing qrels entry means UNKNOWN, not irrelevant.** Every metric here
nonetheless has to treat an unjudged document as gain 0, because there is no
third option -- and that assumption is not neutral between systems. Judgements
are usually pooled from the runs of whichever systems existed when the pool was
built. A reranker earns its keep by surfacing documents the first-stage
retriever never returned; those documents are exactly the ones least likely to
have been pooled, so they score 0 whatever they actually are. Unjudged-as-zero
therefore biases *against* the reranker. ``judged@k`` is reported alongside the
effectiveness metrics for this reason: if the reranked run's judged@k is
materially lower than the baseline's, its other metrics are being measured on
thinner evidence and the comparison is not clean.

Pure python + numpy; the significance tests are implemented here rather than
pulled from scipy/sklearn, neither of which is installed.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

DEFAULT_KS = (1, 3, 5, 10)

# Two-sided sign-test p-values cannot go below 2/2**n, so a "win on every
# query" result over a handful of queries is arithmetically incapable of
# reaching 0.05. compare() warns below this many paired queries.
_SMALL_SAMPLE = 20


class NoRelevantDocuments(ValueError):
    """A query whose qrels contain no positive grade.

    Its score is undefined, not zero, and callers must not average it in.
    """


# -- input normalisation ------------------------------------------------


def _grade_map(relevance: Mapping[str, Any] | Iterable[str]) -> dict[str, float]:
    """Accept either a graded mapping or a bare collection of relevant ids."""
    if isinstance(relevance, Mapping):
        return {str(d): float(g) for d, g in relevance.items()}
    return {str(d): 1.0 for d in relevance}


def _positives(grades: Mapping[str, float]) -> set[str]:
    return {d for d, g in grades.items() if g > 0}


def _require_relevant(grades: Mapping[str, float], where: str) -> None:
    if not any(g > 0 for g in grades.values()):
        raise NoRelevantDocuments(
            f"{where}: no document with a positive grade. This query has no "
            "defined score -- exclude it, do not record 0.0."
        )


def _check_ranked(ranked: Sequence[Any], where: str = "ranked") -> list[str]:
    """A duplicated doc id inflates every metric here, so refuse to score it."""
    out = [str(d) for d in ranked]
    if len(set(out)) != len(out):
        dupes = sorted({d for d, n in Counter(out).items() if n > 1})
        raise ValueError(f"{where}: duplicate doc ids in ranked list: {dupes}")
    return out


def _check_k(k: int) -> int:
    k = int(k)
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    return k


# -- metrics ------------------------------------------------------------


def precision_at_k(
    ranked: Sequence[Any], relevant: Mapping[str, Any] | Iterable[str], k: int
) -> float:
    """Fraction of the top ``k`` that is relevant.

    Divides by ``k``, not by the number of documents actually returned: a run
    that returns 3 documents when 10 were asked for has not earned the
    precision of a full page. Grades > 0 count as relevant.
    """
    k = _check_k(k)
    grades = _grade_map(relevant)
    _require_relevant(grades, "precision_at_k")
    rel = _positives(grades)
    hits = sum(1 for d in _check_ranked(ranked)[:k] if d in rel)
    return hits / k


def recall_at_k(
    ranked: Sequence[Any], relevant: Mapping[str, Any] | Iterable[str], k: int
) -> float:
    """Fraction of all known relevant documents found in the top ``k``.

    The denominator is every positively judged document, including ones the
    run never returned -- which is what makes recall the metric that punishes
    a shallow candidate pool.
    """
    k = _check_k(k)
    grades = _grade_map(relevant)
    _require_relevant(grades, "recall_at_k")
    rel = _positives(grades)
    hits = sum(1 for d in _check_ranked(ranked)[:k] if d in rel)
    return hits / len(rel)


def mrr(ranked: Sequence[Any], relevant: Mapping[str, Any] | Iterable[str]) -> float:
    """Reciprocal rank of the first relevant document, 1-indexed.

    0.0 when no relevant document appears anywhere in the list. That zero is
    earned -- relevant documents exist and the run missed them all -- unlike
    the zero this module refuses to record for an unjudged query.
    """
    grades = _grade_map(relevant)
    _require_relevant(grades, "mrr")
    rel = _positives(grades)
    for i, doc in enumerate(_check_ranked(ranked)):
        if doc in rel:
            return 1.0 / (i + 1)
    return 0.0


def _gain(grade: float, scheme: str) -> float:
    # Negative grades (some collections mark junk as -1) are clamped to 0
    # rather than allowed to subtract from the DCG.
    g = max(0.0, float(grade))
    if scheme == "linear":
        return g
    if scheme == "exponential":
        return (2.0**g) - 1.0
    raise ValueError(f"unknown gain scheme {scheme!r}; use 'linear' or 'exponential'")


def _dcg(gains: Iterable[float]) -> float:
    return float(sum(g / math.log2(i + 2) for i, g in enumerate(gains)))


def ndcg_at_k(
    ranked: Sequence[Any],
    relevance_grades: Mapping[str, Any] | Iterable[str],
    k: int,
    *,
    gain: str = "linear",
) -> float:
    """Graded nDCG@k with a log2 discount.

    Divides by the IDCG of the ideal ordering of the *qrels* -- every judged
    grade sorted descending and truncated to k -- not by the ideal ordering of
    the retrieved candidates, which would let a run that retrieved nothing good
    normalise its way to a high score.

    ``gain='linear'`` uses the raw grade (Jarvelin & Kekalainen);
    ``gain='exponential'`` uses 2**g - 1 (Burges et al.). They coincide on
    binary qrels. Whichever you pick must be held fixed across the systems you
    compare -- the two are not on the same scale.

    Documents absent from ``relevance_grades`` contribute gain 0. See the module
    docstring for why that is not a neutral assumption.
    """
    k = _check_k(k)
    grades = _grade_map(relevance_grades)
    _require_relevant(grades, "ndcg_at_k")
    docs = _check_ranked(ranked)[:k]

    dcg = _dcg(_gain(grades.get(d, 0.0), gain) for d in docs)
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = _dcg(_gain(g, gain) for g in ideal)
    # idcg > 0 is guaranteed: _require_relevant found a positive grade, and it
    # sorts into the first k slots.
    return dcg / idcg


def average_precision(
    ranked: Sequence[Any], relevant: Mapping[str, Any] | Iterable[str]
) -> float:
    """Mean of the precisions measured at each relevant document retrieved.

    Divided by the total number of known relevant documents, so relevant
    documents the run never returned contribute 0 -- the standard TREC AP,
    which is a recall-sensitive number despite the name.
    """
    grades = _grade_map(relevant)
    _require_relevant(grades, "average_precision")
    rel = _positives(grades)
    hits = 0
    total = 0.0
    for i, doc in enumerate(_check_ranked(ranked)):
        if doc in rel:
            hits += 1
            total += hits / (i + 1)
    return total / len(rel)


def judged_at_k(
    ranked: Sequence[Any], judgements: Mapping[str, Any] | Iterable[str], k: int
) -> float:
    """Fraction of the returned top-``k`` that carries any judgement.

    Not an effectiveness metric -- a coverage diagnostic. Every other number
    here treats unjudged as irrelevant, and this one says how much of the run
    that assumption was applied to.

    Unlike ``precision_at_k`` this divides by the number of documents actually
    returned, not by ``k``: the question is what fraction of what the system
    showed was judged, and padding the denominator would blame a short result
    list for a shallow judgement pool.
    """
    k = _check_k(k)
    judged = set(_grade_map(judgements))
    docs = _check_ranked(ranked)[:k]
    if not docs:
        return 0.0
    return sum(1 for d in docs if d in judged) / len(docs)


# -- results ------------------------------------------------------------

_FAMILY_ORDER = ("MRR", "MAP", "nDCG@", "P@", "R@", "judged@")


def _metric_sort_key(name: str) -> tuple[int, float, str]:
    for rank, family in enumerate(_FAMILY_ORDER):
        if name == family or (family.endswith("@") and name.startswith(family)):
            suffix = name[len(family) :]
            try:
                return (rank, float(suffix), name)
            except ValueError:
                return (rank, -1.0, name)
    return (len(_FAMILY_ORDER), -1.0, name)


def _sorted_metrics(names: Iterable[str]) -> list[str]:
    return sorted(set(names), key=_metric_sort_key)


@dataclass
class EvalResult:
    """Scores for one run, with the bookkeeping needed to trust them.

    ``metrics`` holds only metrics that were actually computed; a run that
    scored nothing gets an empty dict, never a dict of zeros.
    """

    name: str
    n_queries: int
    metrics: dict[str, float]
    per_query: dict[str, dict[str, float]]
    skipped_no_qrels: list[str] = field(default_factory=list)
    missing_run: list[str] = field(default_factory=list)
    ks: tuple[int, ...] = ()
    gain: str = "linear"

    @property
    def n_skipped(self) -> int:
        return len(self.skipped_no_qrels)

    def summary(self) -> str:
        """One line of provenance to print next to any number from this run."""
        bits = [f"{self.name or 'run'}: {self.n_queries} queries scored"]
        if self.skipped_no_qrels:
            bits.append(f"{len(self.skipped_no_qrels)} skipped (no positive judgement)")
        if self.missing_run:
            n = len(self.missing_run)
            bits.append(
                f"{n} judged quer{'y' if n == 1 else 'ies'} absent from the run"
            )
        return ", ".join(bits)


def evaluate_run(
    runs: Mapping[str, Sequence[Any]],
    qrels: Mapping[str, Mapping[str, Any]],
    ks: Sequence[int] = DEFAULT_KS,
    name: str = "",
    *,
    gain: str = "linear",
) -> EvalResult:
    """Score a run against graded qrels.

    ``runs`` maps query id -> ranked doc ids, best first. ``qrels`` maps query
    id -> {doc id: grade}, where 0 means judged-and-not-relevant and an absent
    doc id means unjudged.

    Only queries that appear in both ``runs`` and ``qrels`` *and* have at least
    one positive grade are scored. The rest are counted, not scored:
    ``skipped_no_qrels`` for queries the run answered but the judgements cannot
    score, ``missing_run`` for judged queries the run did not answer. Neither
    group contributes a 0.0 to any mean.
    """
    ks = tuple(sorted({_check_k(k) for k in ks}))
    if not ks:
        raise ValueError("ks must contain at least one cutoff")

    scorable = {
        q: _grade_map(g) for q, g in qrels.items() if any(float(v) > 0 for v in g.values())
    }
    evaluated = sorted(set(runs) & set(scorable))
    skipped = sorted(set(runs) - set(scorable))
    missing = sorted(set(scorable) - set(runs))

    per_query: dict[str, dict[str, float]] = {}
    for qid in evaluated:
        grades = scorable[qid]
        ranked = _check_ranked(runs[qid], where=f"runs[{qid!r}]")
        scores: dict[str, float] = {
            "MRR": mrr(ranked, grades),
            "MAP": average_precision(ranked, grades),
        }
        for k in ks:
            scores[f"nDCG@{k}"] = ndcg_at_k(ranked, grades, k, gain=gain)
            scores[f"P@{k}"] = precision_at_k(ranked, grades, k)
            scores[f"R@{k}"] = recall_at_k(ranked, grades, k)
            scores[f"judged@{k}"] = judged_at_k(ranked, grades, k)
        per_query[qid] = scores

    metrics: dict[str, float] = {}
    if per_query:
        names = _sorted_metrics(next(iter(per_query.values())))
        for metric in names:
            metrics[metric] = float(
                np.mean([per_query[q][metric] for q in evaluated])
            )

    return EvalResult(
        name=name,
        n_queries=len(evaluated),
        metrics=metrics,
        per_query=per_query,
        skipped_no_qrels=skipped,
        missing_run=missing,
        ks=ks,
        gain=gain,
    )


# -- significance -------------------------------------------------------


def sign_test_p(
    baseline: Sequence[float], treatment: Sequence[float], *, tie_eps: float = 1e-12
) -> float:
    """Exact two-sided sign test on paired per-query scores.

    Ties are dropped, which is the conventional handling and the conservative
    one: dropping them shrinks n, and the smallest reachable p-value is
    2/2**n. Ten wins and no losses gives 0.001953125; three wins and no losses
    gives 0.25, however large the deltas look.
    """
    a = np.asarray(baseline, dtype=float)
    b = np.asarray(treatment, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired inputs must match: {a.shape} vs {b.shape}")
    d = b - a
    pos = int(np.sum(d > tie_eps))
    neg = int(np.sum(d < -tie_eps))
    n = pos + neg
    if n == 0:
        return 1.0
    m = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(m + 1))
    return min(1.0, 2.0 * tail / (2**n))


def paired_bootstrap_p(
    baseline: Sequence[float],
    treatment: Sequence[float],
    *,
    n_resamples: int = 10000,
    seed: int = 0,
) -> float:
    """Two-sided paired bootstrap p-value over per-query differences.

    Resamples queries with replacement from the differences re-centred on zero,
    and asks how often that null produces a mean at least as extreme as the one
    observed. The +1 in numerator and denominator keeps the estimate away from
    an unsupportable p = 0.
    """
    a = np.asarray(baseline, dtype=float)
    b = np.asarray(treatment, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired inputs must match: {a.shape} vs {b.shape}")
    d = b - a
    n = d.size
    if n == 0:
        raise ValueError("paired bootstrap needs at least one query")
    observed = float(np.mean(d))
    centred = d - observed
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_resamples), n))
    means = centred[idx].mean(axis=1)
    # The tolerance rounds borderline resamples into the tail, i.e. upward.
    extreme = int(np.sum(np.abs(means) >= abs(observed) - 1e-12))
    return (extreme + 1) / (int(n_resamples) + 1)



def benjamini_hochberg(
    p_values: Mapping[str, float], alpha: float = 0.05
) -> dict[str, Any]:
    """Control the false discovery rate across a family of tests.

    Bonferroni controls the chance of ANY false positive and is brutal when the
    metrics are correlated, which P@5, P@10 and nDCG@5 obviously are. BH instead
    controls the expected PROPORTION of discoveries that are false, which is the
    right question when reporting a table of related metrics.

    Sort the p-values ascending, find the largest rank i where p_i <= (i/m)*alpha,
    and reject everything up to it. Returns adjusted q-values (the step-up
    monotone version) alongside the reject flags.
    """
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    if m == 0:
        return {"alpha": alpha, "n_tests": 0, "results": {}, "n_significant": 0}

    # Step-up: enforce monotonicity from the largest p downwards so a q-value
    # can never exceed one computed at a higher rank.
    q_raw = [(name, p, min(1.0, p * m / (i + 1))) for i, (name, p) in enumerate(items)]
    q_adj: list[tuple[str, float, float]] = []
    running = 1.0
    for name, p, q in reversed(q_raw):
        running = min(running, q)
        q_adj.append((name, p, running))
    q_adj.reverse()

    crit = 0
    for i, (_, p) in enumerate(items, start=1):
        if p <= (i / m) * alpha:
            crit = i

    results = {
        name: {
            "p": p,
            "q": q,
            "rank": i + 1,
            "reject": (i + 1) <= crit,
        }
        for i, (name, p, q) in enumerate(q_adj)
    }
    return {
        "alpha": alpha,
        "n_tests": m,
        "critical_rank": crit,
        "results": results,
        "n_significant": crit,
    }


def compare(
    baseline: EvalResult,
    reranked: EvalResult,
    *,
    n_resamples: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    """Per-metric deltas plus paired significance tests.

    Deltas come from the reported means; the p-values come from the queries
    both runs actually scored, and ``paired_delta`` is the mean difference over
    exactly those queries. When the two runs scored different query sets, the
    reported means are not comparable and that is recorded in ``warnings``
    rather than smoothed over.

    A p-value per metric is a multiple-comparison machine: testing eight
    metrics at 0.05 finds something roughly a third of the time on noise alone.
    Pick one primary metric before looking.
    """
    warnings: list[str] = []
    paired_ids = sorted(set(baseline.per_query) & set(reranked.per_query))
    only_b = len(set(baseline.per_query) - set(paired_ids))
    only_r = len(set(reranked.per_query) - set(paired_ids))
    if only_b or only_r:
        warnings.append(
            f"query sets differ: {only_b} scored only by {baseline.name or 'baseline'}, "
            f"{only_r} only by {reranked.name or 'reranked'}; reported means are "
            "over different queries and the delta between them is not clean"
        )
    if 0 < len(paired_ids) < _SMALL_SAMPLE:
        floor = 2.0 / (2 ** len(paired_ids))
        warnings.append(
            f"only {len(paired_ids)} paired queries: the sign test cannot return "
            f"below p={floor:.4g} even if every query improves"
        )
    if not paired_ids:
        warnings.append("no queries scored by both runs; no significance test possible")
    for res in (baseline, reranked):
        if res.skipped_no_qrels:
            warnings.append(
                f"{res.name or 'run'}: {len(res.skipped_no_qrels)} queries skipped for "
                "having no positively judged document"
            )
    if baseline.gain != reranked.gain:
        warnings.append(
            f"nDCG gain schemes differ ({baseline.gain} vs {reranked.gain}); "
            "the nDCG columns are not on the same scale"
        )

    metrics: dict[str, dict[str, Any]] = {}
    for metric in _sorted_metrics([*baseline.metrics, *reranked.metrics]):
        b_mean = baseline.metrics.get(metric)
        r_mean = reranked.metrics.get(metric)
        entry: dict[str, Any] = {"baseline": b_mean, "reranked": r_mean}

        if b_mean is None or r_mean is None:
            entry["note"] = "metric absent from one run; no delta computed"
            metrics[metric] = entry
            continue

        delta = r_mean - b_mean
        entry["delta"] = delta
        # A zero baseline makes the relative change undefined, not infinite
        # and not zero.
        entry["relative_delta"] = (delta / b_mean) if b_mean != 0 else None

        pairs = [
            (baseline.per_query[q][metric], reranked.per_query[q][metric])
            for q in paired_ids
            if metric in baseline.per_query[q] and metric in reranked.per_query[q]
        ]
        entry["n_paired"] = len(pairs)
        if pairs:
            b_vals = [p[0] for p in pairs]
            r_vals = [p[1] for p in pairs]
            diffs = np.asarray(r_vals) - np.asarray(b_vals)
            entry["paired_delta"] = float(np.mean(diffs))
            entry["wins"] = int(np.sum(diffs > 1e-12))
            entry["losses"] = int(np.sum(diffs < -1e-12))
            entry["ties"] = int(np.sum(np.abs(diffs) <= 1e-12))
            entry["p_sign"] = sign_test_p(b_vals, r_vals)
            entry["p_bootstrap"] = paired_bootstrap_p(
                b_vals, r_vals, n_resamples=n_resamples, seed=seed
            )
            # Aliases for callers that read the conventional key names. p_value
            # is the sign test, not the bootstrap: its floor of 2/2**n keeps a
            # small query set from reporting a p it cannot support.
            entry["relative"] = entry["relative_delta"]
            entry["p_value"] = entry["p_sign"]
        metrics[metric] = entry

    return {
        "n_resamples": int(n_resamples),
        "baseline": baseline.name or "baseline",
        "reranked": reranked.name or "reranked",
        "n_queries": {
            "baseline": baseline.n_queries,
            "reranked": reranked.n_queries,
        },
        "n_paired": len(paired_ids),
        "metrics": metrics,
        "warnings": warnings,
    }


# -- formatting ---------------------------------------------------------

_MISSING = "-"


def _fmt(value: float | None, places: int = 4) -> str:
    return _MISSING if value is None else f"{value:.{places}f}"


def _fmt_p(value: float | None) -> str:
    """Small p-values render as a bound, never as 0.0000."""
    if value is None:
        return _MISSING
    return "<0.0001" if value < 1e-4 else f"{value:.4f}"


def format_table(results: list[EvalResult]) -> str:
    """Plain-text table: one row per metric, one column per run.

    A metric a run did not produce prints as ``-``. Nothing is filled in.
    """
    if not results:
        return "no results"

    names = [r.name or f"run{i}" for i, r in enumerate(results)]
    metric_names = _sorted_metrics([m for r in results for m in r.metrics])

    rows: list[list[str]] = [["metric", *names]]
    for metric in metric_names:
        rows.append([metric, *[_fmt(r.metrics.get(metric)) for r in results]])
    rule_after = len(rows) - 1
    rows.append(["queries scored", *[str(r.n_queries) for r in results]])
    rows.append(["skipped (unjudged)", *[str(r.n_skipped) for r in results]])
    rows.append(["judged qs missing", *[str(len(r.missing_run)) for r in results]])

    widths = [max(len(row[c]) for row in rows) for c in range(len(rows[0]))]

    def render(row: list[str]) -> str:
        cells = [row[0].ljust(widths[0])]
        cells += [row[c].rjust(widths[c]) for c in range(1, len(row))]
        return "  ".join(cells).rstrip()

    sep = "  ".join("-" * w for w in widths)
    out = [render(rows[0]), sep]
    out += [render(r) for r in rows[1 : rule_after + 1]]
    out.append(sep)
    out += [render(r) for r in rows[rule_after + 1 :]]
    if not metric_names:
        out.append("(no metrics computed)")
    return "\n".join(out)


def format_comparison(comparison: Mapping[str, Any]) -> str:
    """Plain-text rendering of a :func:`compare` dict, warnings included."""
    header = [
        "metric",
        str(comparison["baseline"]),
        str(comparison["reranked"]),
        "delta",
        "rel",
        "W/L/T",
        "p(sign)",
        "p(boot)",
    ]
    rows = [header]
    for metric, e in comparison["metrics"].items():
        rel = e.get("relative_delta")
        wlt = (
            f"{e['wins']}/{e['losses']}/{e['ties']}"
            if "wins" in e
            else _MISSING
        )
        rows.append(
            [
                metric,
                _fmt(e.get("baseline")),
                _fmt(e.get("reranked")),
                _fmt(e.get("delta")) if e.get("delta") is not None else _MISSING,
                f"{rel * 100:+.1f}%" if rel is not None else _MISSING,
                wlt,
                _fmt_p(e.get("p_sign")),
                _fmt_p(e.get("p_bootstrap")),
            ]
        )

    widths = [max(len(row[c]) for row in rows) for c in range(len(header))]

    def render(row: list[str]) -> str:
        cells = [row[0].ljust(widths[0])]
        cells += [row[c].rjust(widths[c]) for c in range(1, len(row))]
        return "  ".join(cells).rstrip()

    out = [render(rows[0]), "  ".join("-" * w for w in widths)]
    out += [render(r) for r in rows[1:]]
    out.append("")
    out.append(f"paired queries: {comparison['n_paired']}")
    resamples = comparison.get("n_resamples")
    if resamples:
        # Without this the floor reads like a precise p rather than "as small
        # as this many resamples can resolve".
        out.append(
            f"bootstrap: {resamples} resamples, so p(boot) cannot go below "
            f"{1 / (resamples + 1):.2g}"
        )
    out.append(
        "p-values are per-metric and uncorrected; choose one primary metric first."
    )
    for warning in comparison["warnings"]:
        out.append(f"WARNING: {warning}")
    return "\n".join(out)
