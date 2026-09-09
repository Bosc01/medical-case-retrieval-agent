"""Does a deeper first stage actually produce a better top 5?

The recall sweep shows depth 50 captures only about half the relevant cases it
could. That raises the reranker's ceiling but does not by itself improve the
answer: more candidates give the cross-encoder more chances to be fooled as
well as more chances to be right. This settles it by measurement.

Scoring is done ONCE at the maximum depth. The depth-50 candidate set is a
prefix of the depth-200 set, so every shallower arm is a subset of scores
already computed, and the comparison costs one pass rather than three.
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medcase.embed import MedCPTEmbedder
from medcase.evaluate import compare, evaluate_run, format_comparison, format_table
from medcase.index import CaseIndex
from medcase.reranker import CrossEncoderReranker

DEPTHS = [50, 100, 200]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    evalset = json.loads((root / "data" / "evalset.json").read_text())
    queries = evalset["queries"][: args.limit] if args.limit else evalset["queries"]
    index = CaseIndex.load(root / "data" / "index")
    embedder = MedCPTEmbedder()
    reranker = CrossEncoderReranker(batch_size=32)
    reranker.warmup()

    maxd = max(DEPTHS)
    print(f"{len(queries)} queries | corpus {len(index)} | scoring once at depth {maxd}\n")

    qrels = {q["qid"]: q["qrels"] for q in queries}
    # ordered[qid] = FAISS-ranked pmids; scores[qid][pmid] = cross-encoder score
    ordered, scores = {}, {}
    t0 = time.perf_counter()
    for i, q in enumerate(queries, 1):
        qid = q["qid"]
        hits = index.search(q["query"], embedder, maxd + 1)
        hits = [(r, s) for r, s in hits if r.pmid != qid][:maxd]
        recs = [r for r, _ in hits]
        ordered[qid] = [r.pmid for r in recs]
        pair_scores = reranker.score(q["query"], [r.text for r in recs])
        scores[qid] = dict(zip(ordered[qid], pair_scores))
        if i % 20 == 0:
            el = time.perf_counter() - t0
            print(f"  {i}/{len(queries)}  ({el:.0f}s, {el/i:.1f}s/query)")
    total = time.perf_counter() - t0
    print(f"scored in {total:.0f}s ({total/len(queries)*1000:.0f} ms/query at depth {maxd})\n")

    runs = {}
    for d in DEPTHS:
        # Baseline arm: FAISS order truncated to this depth.
        runs[f"faiss@{d}"] = {q: ordered[q][:d] for q in ordered}
        # Reranked arm: same candidates, reordered by cached cross-encoder score.
        runs[f"rerank@{d}"] = {
            q: sorted(ordered[q][:d], key=lambda p: scores[q][p], reverse=True)
            for q in ordered
        }

    results = [evaluate_run(r, qrels, name=n) for n, r in runs.items()]
    print(format_table(results))

    by_name = {r.name: r for r in results}
    print(f"\n{'=' * 70}\nEffect of retrieval depth on the reranked top {args.top_k}")
    print(f"{'=' * 70}")
    base = by_name[f"rerank@{DEPTHS[0]}"]
    for d in DEPTHS[1:]:
        print(f"\n--- rerank@{d} vs rerank@{DEPTHS[0]} ---")
        print(format_comparison(compare(base, by_name[f"rerank@{d}"])))

    out = root / "data" / "depth_results.json"
    out.write_text(json.dumps({
        "depths": DEPTHS,
        "n_queries": len(queries),
        "ms_per_query_at_max_depth": total / len(queries) * 1000,
        "results": {r.name: r.metrics for r in results},
    }, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
