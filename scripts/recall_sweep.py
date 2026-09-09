"""How deep does stage one have to go before the reranker has something to work with?

Reranking cannot surface a case FAISS never returned, so recall at the retrieval
depth is a hard ceiling on the whole pipeline. This sweeps that depth. If recall
is still climbing steeply at 50, the cheapest available win is a deeper first
stage, not a better second one.

Reported at two relevance thresholds because they answer different questions:
grade>=1 counts any shared MeSH descriptor, grade>=2 counts only cases NLM
indexed the concept as a major topic of. The second is the stricter, more
clinically meaningful target.
"""

from __future__ import annotations

import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medcase.embed import MedCPTEmbedder
from medcase.index import CaseIndex

DEPTHS = [5, 10, 20, 50, 100, 200, 400]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    evalset = json.loads((root / "data" / "evalset.json").read_text())
    queries = evalset["queries"]
    index = CaseIndex.load(root / "data" / "index")
    embedder = MedCPTEmbedder()
    maxd = max(DEPTHS)

    rankings = {}
    for q in queries:
        hits = index.search(q["query"], embedder, maxd + 1)
        rankings[q["qid"]] = [r.pmid for r, _ in hits if r.pmid != q["qid"]][:maxd]

    print(f"{len(queries)} queries | corpus {len(index)}\n")
    print(f"{'depth':>6}  {'recall(g>=1)':>13}  {'ceiling':>8}  "
          f"{'recall(g>=2)':>13}  {'ceiling':>8}")
    print("-" * 58)
    for d in DEPTHS:
        row = [d]
        for threshold in (1, 2):
            rec, cap = [], []
            for q in queries:
                rel = {p for p, g in q["qrels"].items() if g >= threshold}
                if not rel:
                    continue  # unknown, not zero
                got = sum(1 for p in rankings[q["qid"]][:d] if p in rel)
                rec.append(got / len(rel))
                cap.append(min(d, len(rel)) / len(rel))
            row += [sum(rec) / len(rec), sum(cap) / len(cap)]
        print(f"{row[0]:>6}  {row[1]:>13.4f}  {row[2]:>8.4f}  "
              f"{row[3]:>13.4f}  {row[4]:>8.4f}")

    n2 = sum(1 for q in queries if any(g >= 2 for g in q["qrels"].values()))
    print(f"\nqueries with at least one grade-2 relevant case: {n2}/{len(queries)}")
    print("ceiling = best achievable recall at that depth, given how many "
          "relevant cases exist")


if __name__ == "__main__":
    main()
