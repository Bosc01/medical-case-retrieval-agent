"""Does query style decide whether the reranker works?

MedCPT was trained on PubMed search logs, which are short keyword queries. The
eval in this repository uses narrative case presentations, which are not that.
A spot check found the cross-encoder assigning every candidate a score inside a
0.23 band for a narrative query, which is no discrimination at all, while the
same target under a keyword query separated by 22.5.

If that generalises, the headline result was measured with the reranker close to
its noise floor. This compares two query styles over the same cases, same
candidates and same labels. Titles stand in for keyword-style queries: condensed
and diagnosis-forward, closer to how PubMed is actually searched, and not the
label, which the MeSH anchors would be.
"""

from __future__ import annotations

import argparse, json, statistics, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medcase.corpus import load_corpus
from medcase.embed import MedCPTEmbedder
from medcase.evaluate import compare, evaluate_run
from medcase.index import CaseIndex
from medcase.reranker import CrossEncoderReranker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieve-k", type=int, default=50)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    evalset = json.loads((root / "data" / "evalset.json").read_text())
    records = {r.pmid: r for r in load_corpus(root / "data" / "corpus.jsonl")}
    index = CaseIndex.load(root / "data" / "index")
    embedder = MedCPTEmbedder()
    rr = CrossEncoderReranker(batch_size=32, torch_dtype=torch.float16)
    rr.warmup()

    styles = {
        "narrative (presentation)": lambda q: q["query"],
        "keyword (title)": lambda q: records[q["qid"]].title,
    }

    out = {}
    for label, pick in styles.items():
        qrels, faiss_run, rr_run, spreads = {}, {}, {}, []
        for q in evalset["queries"]:
            qid = q["qid"]
            if qid not in records:
                continue
            text = pick(q)
            if not text or len(text) < 10:
                continue
            hits = index.search(text, embedder, args.retrieve_k + 1)
            hits = [(r, s) for r, s in hits if r.pmid != qid][: args.retrieve_k]
            recs = [r for r, _ in hits]
            scores = rr.score(text, [r.text for r in recs])
            spreads.append(max(scores) - min(scores))
            qrels[qid] = q["qrels"]
            faiss_run[qid] = [r.pmid for r in recs]
            rr_run[qid] = [
                p for p, _ in sorted(
                    zip(faiss_run[qid], scores), key=lambda kv: kv[1], reverse=True
                )
            ]

        base = evaluate_run(faiss_run, qrels, name="faiss")
        rer = evaluate_run(rr_run, qrels, name="rerank")
        c = compare(base, rer)["metrics"]
        med_spread = statistics.median(spreads)

        print(f"\n=== {label} ===")
        print(f"  queries: {len(qrels)} | median score spread across 50 candidates: "
              f"{med_spread:.2f}")
        for m in ("P@5", "P@10", "nDCG@5"):
            d = c[m]
            print(f"  {m:<8}{d['baseline']:.4f} -> {d['reranked']:.4f}"
                  f"  ({d['delta']:+.4f}, {d['relative_delta']:+.1%}, "
                  f"p={d['p_bootstrap']:.4f})")
        out[label] = {
            "n": len(qrels), "median_spread": med_spread,
            "metrics": {m: c[m] for m in ("P@5", "P@10", "nDCG@5")},
        }

    path = root / "data" / "query_style_results.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
