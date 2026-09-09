"""Score every (query, candidate) pair once and save the raw scores.

Every downstream question -- which rerank depth, which relevance definition,
which correction -- is a re-sort or a re-label of these same numbers. Dumping
them once turns each of those from an 11-minute GPU run into a replay, and
guarantees the arms are compared on identical scores rather than on two runs
that happened to differ.
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medcase.embed import MedCPTEmbedder
from medcase.index import CaseIndex
from medcase.reranker import CrossEncoderReranker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="data/scores_depth100.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    evalset = json.loads((root / "data" / "evalset.json").read_text())
    queries = evalset["queries"][: args.limit] if args.limit else evalset["queries"]
    index = CaseIndex.load(root / "data" / "index")
    embedder = MedCPTEmbedder()
    reranker = CrossEncoderReranker(batch_size=32)
    reranker.warmup()

    print(f"{len(queries)} queries | depth {args.depth}")
    out = {}
    t0 = time.perf_counter()
    for i, q in enumerate(queries, 1):
        qid = q["qid"]
        hits = index.search(q["query"], embedder, args.depth + 1)
        hits = [(r, s) for r, s in hits if r.pmid != qid][: args.depth]
        recs = [r for r, _ in hits]
        ce = reranker.score(q["query"], [r.text for r in recs])
        out[qid] = {
            "faiss_order": [r.pmid for r in recs],
            "faiss_scores": [float(s) for _, s in hits],
            "ce_scores": {r.pmid: float(c) for r, c in zip(recs, ce)},
        }
        if i % 20 == 0:
            el = time.perf_counter() - t0
            print(f"  {i}/{len(queries)} ({el:.0f}s, {el/i:.1f}s/query)")

    total = time.perf_counter() - t0
    path = root / args.out
    path.write_text(json.dumps({
        "depth": args.depth,
        "n_queries": len(queries),
        "ms_per_query": total / len(queries) * 1000,
        "scores": out,
    }))
    print(f"scored in {total:.0f}s -> {path}")


if __name__ == "__main__":
    main()
