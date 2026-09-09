"""Measure what the reranker actually changes.

Every arm sees the SAME 50 FAISS candidates and differs only in how it orders
them. That isolates the reranker: any delta is reordering, not better recall.
Recall@50 is reported separately because it is the hard ceiling -- the reranker
cannot surface a case the first stage never returned.

The ms-marco arm is a control. If a general-domain reranker matches the
biomedical one, the domain match is not what is doing the work.
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medcase.embed import MedCPTEmbedder
from medcase.evaluate import (
    compare, evaluate_run, format_comparison, format_table, recall_at_k,
)
from medcase.index import CaseIndex
from medcase.reranker import BASELINE_MODEL, DEFAULT_MODEL, CrossEncoderReranker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieve-k", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-control", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    evalset = json.loads((root / "data" / "evalset.json").read_text())
    queries = evalset["queries"][: args.limit] if args.limit else evalset["queries"]
    index = CaseIndex.load(root / "data" / "index")
    embedder = MedCPTEmbedder()

    print(f"{len(queries)} queries | corpus {len(index)} | retrieve_k {args.retrieve_k}")

    # --- stage one, once. Both arms reorder this same candidate set. ---
    candidates, qrels, retrieve_ms = {}, {}, []
    t_start = time.perf_counter()
    for q in queries:
        qid = q["qid"]
        t0 = time.perf_counter()
        hits = index.search(q["query"], embedder, args.retrieve_k + 1)
        retrieve_ms.append((time.perf_counter() - t0) * 1000)
        # The query IS a corpus record; it would otherwise sit at rank 1 as a
        # non-relevant self-match and distort every metric.
        hits = [(r, s) for r, s in hits if r.pmid != qid][: args.retrieve_k]
        candidates[qid] = hits
        qrels[qid] = q["qrels"]
    print(f"retrieval: {sum(retrieve_ms)/len(retrieve_ms):.1f} ms/query "
          f"({time.perf_counter()-t_start:.0f}s total)")

    ceiling = sum(
        recall_at_k([r.pmid for r, _ in candidates[qid]], qrels[qid], args.retrieve_k)
        for qid in candidates
    ) / len(candidates)
    print(f"recall@{args.retrieve_k} (reranker ceiling): {ceiling:.4f}\n")

    runs = {"faiss-baseline": {q: [r.pmid for r, _ in c] for q, c in candidates.items()}}
    timings = {}

    arms = [("rerank-medcpt", DEFAULT_MODEL)]
    if not args.skip_control:
        arms.append(("rerank-msmarco", BASELINE_MODEL))

    for name, model in arms:
        rr = CrossEncoderReranker(model, batch_size=32)
        rr.warmup()
        ranked, ms = {}, []
        t0 = time.perf_counter()
        for i, (qid, hits) in enumerate(candidates.items(), 1):
            recs = [r for r, _ in hits]
            out = rr.rerank(q_text(queries, qid), recs, text_of=lambda r: r.text)
            ranked[qid] = [d.document.pmid for d in out]
            ms.append(rr.last_metrics.latency_ms)
            if i % 30 == 0:
                print(f"  {name}: {i}/{len(candidates)}")
        runs[name] = ranked
        timings[name] = {
            "ms_per_query": sum(ms) / len(ms),
            "total_s": time.perf_counter() - t0,
        }
        print(f"  {name}: {timings[name]['ms_per_query']:.0f} ms/query "
              f"({timings[name]['total_s']:.0f}s total)\n")

    results = [evaluate_run(r, qrels, name=n) for n, r in runs.items()]
    print(format_table(results))

    base = results[0]
    comparisons = {}
    for res in results[1:]:
        print(f"\n=== {res.name} vs {base.name} ===")
        cmp = compare(base, res)
        comparisons[res.name] = cmp
        print(format_comparison(cmp))

    (root / "data" / "eval_runs.json").write_text(json.dumps(
        {"runs": runs, "qrels": qrels}, indent=1))

    out = root / "data" / "eval_results.json"
    out.write_text(json.dumps({
        "n_queries": len(queries),
        "retrieve_k": args.retrieve_k,
        "corpus_size": len(index),
        "recall_ceiling": ceiling,
        "retrieval_ms_per_query": sum(retrieve_ms) / len(retrieve_ms),
        "rerank_timings": timings,
        "results": {r.name: r.metrics for r in results},
        "comparisons": comparisons,
    }, indent=2))
    print(f"\n-> {out}")


def q_text(queries, qid):
    for q in queries:
        if q["qid"] == qid:
            return q["query"]
    raise KeyError(qid)


if __name__ == "__main__":
    main()
