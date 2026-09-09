"""Does the optimised configuration return the same answers as the reference?

Two changes stack: fp16 weights, and reranking only the top 30 so the work fits
in one batch. Both are only worth having if P@5 is unchanged, so this measures
the full pipeline end to end rather than trusting a spot check on one query.
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medcase.embed import MedCPTEmbedder
from medcase.evaluate import compare, evaluate_run, format_table
from medcase.index import CaseIndex
from medcase.reranker import CrossEncoderReranker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieve-k", type=int, default=50)
    ap.add_argument("--budget", type=int, default=30)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    evalset = json.loads((root / "data" / "evalset.json").read_text())
    queries = evalset["queries"]
    qrels = {q["qid"]: q["qrels"] for q in queries}
    index = CaseIndex.load(root / "data" / "index")
    embedder = MedCPTEmbedder()

    candidates = {}
    for q in queries:
        hits = index.search(q["query"], embedder, args.retrieve_k + 1)
        candidates[q["qid"]] = [
            (r, s) for r, s in hits if r.pmid != q["qid"]
        ][: args.retrieve_k]

    runs, timings = {}, {}
    runs["faiss-baseline"] = {q: [r.pmid for r, _ in c] for q, c in candidates.items()}

    configs = [
        ("reference fp32 top-50", None, args.retrieve_k),
        ("fast fp16 top-30", torch.float16, args.budget),
    ]
    for name, dtype, budget in configs:
        rr = CrossEncoderReranker(batch_size=32, torch_dtype=dtype)
        rr.warmup()
        ranked, ms = {}, []
        for q in queries:
            qid = q["qid"]
            hits = candidates[qid]
            head = [r for r, _ in hits[:budget]]
            tail = [r.pmid for r, _ in hits[budget:]]
            t0 = time.perf_counter()
            out = rr.rerank(q["query"], head, text_of=lambda r: r.text)
            ms.append((time.perf_counter() - t0) * 1000)
            # The tail is still returned, so metrics past the budget stay honest.
            ranked[qid] = [d.document.pmid for d in out] + tail
        runs[name] = ranked
        timings[name] = sum(ms) / len(ms)
        print(f"{name:<24} {timings[name]:>8.0f} ms/query")

    results = [evaluate_run(r, qrels, name=n) for n, r in runs.items()]
    print()
    print(format_table(results))

    by = {r.name: r for r in results}
    ref, fast = by["reference fp32 top-50"], by["fast fp16 top-30"]
    print(f"\n=== fast vs reference ===")
    for m in ("P@5", "P@10", "MAP", "nDCG@5"):
        c = compare(ref, fast)["metrics"][m]
        print(f"  {m:<8}{c['baseline']:>9.4f} -> {c['reranked']:<9.4f}"
              f"{c['delta']:>+9.4f}  p={c['p_bootstrap']:.4f}")
    speed = timings["reference fp32 top-50"] / timings["fast fp16 top-30"]
    print(f"\nspeedup: {speed:.2f}x")

    out = root / "data" / "fast_path_results.json"
    out.write_text(json.dumps({
        "timings_ms": timings, "speedup": speed,
        "results": {r.name: r.metrics for r in results},
    }, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
