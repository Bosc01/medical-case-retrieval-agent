"""Where does the reranker's gain actually stop?

The broad definition (any shared specific descriptor) shows a gain. The strict
one (rare descriptor, major topic in both) does not. Reporting only those two
says "it works on one definition and not another", which is not an actionable
statement. Sweeping the definition between them locates the boundary and turns
it into a claim about which queries the reranker helps.

The knob is the document frequency of the shared descriptor: how many cases in
the corpus carry the concept. Low df means a rare, specific diagnosis; high df
means a common one. Everything replays the saved rankings, so only the labels
change.
"""

from __future__ import annotations

import argparse, json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medcase.corpus import load_corpus
from medcase.evaluate import benjamini_hochberg, compare, evaluate_run

THRESHOLDS = [10, 15, 20, 25, 30, 45]


def build_qrels(records, evalset, max_df, rare, df):
    qrels = {}
    for q in evalset["queries"]:
        qid = q["qid"]
        if qid not in records:
            continue
        anchors = [a for a in q["anchors"] if df.get(a, 10**9) <= max_df]
        if not anchors:
            continue
        rel = {
            pmid: 2
            for pmid, rec in records.items()
            if pmid != qid and any(a in (rec.mesh_major or []) for a in anchors)
        }
        if rel:
            qrels[qid] = rel
    return qrels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="P@5")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    saved = json.loads((root / "data" / "eval_runs.json").read_text())
    evalset = json.loads((root / "data" / "evalset.json").read_text())
    records = {r.pmid: r for r in load_corpus(root / "data" / "corpus.jsonl")}
    df = Counter(m for r in records.values() for m in (r.mesh_major or []))

    print(f"metric: {args.metric}   (major topic in both cases throughout)\n")
    print(f"{'max df':>7}{'queries':>9}{'rel/query':>11}{'baseline':>10}"
          f"{'reranked':>10}{'delta':>9}{'p(boot)':>10}")
    print("-" * 66)

    rows, pvals = [], {}
    for t in THRESHOLDS:
        qrels = build_qrels(records, evalset, t, None, df)
        if len(qrels) < 10:
            continue
        runs = {
            n: {q: r for q, r in run.items() if q in qrels}
            for n, run in saved["runs"].items()
        }
        base = evaluate_run(runs["faiss-baseline"], qrels, name="base")
        rr = evaluate_run(runs["rerank-medcpt"], qrels, name="rr")
        c = compare(base, rr)["metrics"][args.metric]
        n_rel = sum(len(v) for v in qrels.values()) / len(qrels)
        rows.append((t, len(qrels), n_rel, c))
        pvals[f"df<={t}"] = c["p_bootstrap"]
        print(f"{t:>7}{len(qrels):>9}{n_rel:>11.1f}{c['baseline']:>10.4f}"
              f"{c['reranked']:>10.4f}{c['delta']:>+9.4f}"
              f"{c['p_bootstrap']:>10.4f}")

    bh = benjamini_hochberg(pvals)
    print(f"\nafter Benjamini-Hochberg across these {len(pvals)} definitions:")
    for name in pvals:
        r = bh["results"][name]
        print(f"  {name:<10} q={r['q']:.4f}  {'significant' if r['reject'] else 'not significant'}")

    out = root / "data" / "relevance_sweep.json"
    out.write_text(json.dumps({
        "metric": args.metric,
        "thresholds": THRESHOLDS,
        "rows": [
            {"max_df": t, "n_queries": n, "mean_relevant": r,
             "baseline": c["baseline"], "reranked": c["reranked"],
             "delta": c["delta"], "p_bootstrap": c["p_bootstrap"]}
            for t, n, r, c in rows
        ],
        "benjamini_hochberg": bh,
    }, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
