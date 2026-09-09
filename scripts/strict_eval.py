"""Re-score the saved rankings under a harder relevance definition.

The main eval counts two cases as relevant when they share any specific MeSH
descriptor, which yields about 30 relevant documents per query out of 763. That
is a generous pool, and a generous pool inflates Precision@5 for every arm. The
question this answers is whether the reranker's gain is an artifact of that
generosity.

Strict relevance keeps only pairs where the shared descriptor is a major topic
of BOTH cases AND is rare in the corpus, which is the case-matching a clinician
actually wants. No model is run: the rankings from scripts/run_eval.py are
replayed against tighter labels, so this is a labelling change and nothing else.
"""

from __future__ import annotations

import argparse, json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medcase.corpus import load_corpus
from medcase.evaluate import compare, evaluate_run, format_comparison, format_table

MAX_DF = 15  # descriptor must be rare to count as a strong clinical match


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-df", type=int, default=MAX_DF)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    saved = json.loads((root / "data" / "eval_runs.json").read_text())
    evalset = json.loads((root / "data" / "evalset.json").read_text())
    records = {r.pmid: r for r in load_corpus(root / "data" / "corpus.jsonl")}

    df = Counter(m for r in records.values() for m in (r.mesh_major or []))
    rare = {m for m, n in df.items() if n <= args.max_df}

    strict = {}
    for q in evalset["queries"]:
        qid = q["qid"]
        src = records.get(qid)
        if src is None:
            continue
        anchors = [a for a in q["anchors"] if a in rare]
        if not anchors:
            continue
        rel = {}
        for pmid, rec in records.items():
            if pmid == qid:
                continue
            # Major topic of BOTH cases, on a rare descriptor.
            if any(a in (rec.mesh_major or []) for a in anchors):
                rel[pmid] = 2
        if rel:
            strict[qid] = rel

    n_rel = [len(v) for v in strict.values()]
    corpus_n = len(records) - 1
    density = sum(n / corpus_n for n in n_rel) / len(n_rel)
    print(f"strict relevance: descriptor df <= {args.max_df}, major topic in both")
    print(f"queries retained: {len(strict)}/{len(evalset['queries'])}")
    print(f"relevant per query: mean {sum(n_rel)/len(n_rel):.1f} "
          f"(median {sorted(n_rel)[len(n_rel)//2]}, max {max(n_rel)})")
    print(f"random-ranking P@5 on this definition: {density:.4f}\n")

    runs = {n: {q: r for q, r in run.items() if q in strict}
            for n, run in saved["runs"].items()}
    results = [evaluate_run(r, strict, name=n) for n, r in runs.items()]
    print(format_table(results))

    by = {r.name: r for r in results}
    base = by["faiss-baseline"]
    for name in [n for n in by if n != "faiss-baseline"]:
        print(f"\n=== {name} vs faiss-baseline (strict) ===")
        print(format_comparison(compare(base, by[name])))

    out = root / "data" / "strict_results.json"
    out.write_text(json.dumps({
        "max_df": args.max_df,
        "n_queries": len(strict),
        "mean_relevant_per_query": sum(n_rel) / len(n_rel),
        "random_p_at_5": density,
        "results": {r.name: r.metrics for r in results},
    }, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
