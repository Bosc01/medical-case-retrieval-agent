"""Does retrieving 100 instead of 50 improve the answer, on any definition?

The recall sweep says depth 50 leaves half the achievable recall on the table,
which makes a deeper first stage look like the obvious win. The broad-relevance
depth experiment said it is not. This checks the remaining possibility: that
deeper retrieval helps where recall is genuinely scarce, which is the strict
rare-descriptor definition, even though it does not help on the loose one.

Replays the depth-100 score dump, so every arm uses identical scores.
"""

from __future__ import annotations

import argparse, json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medcase.corpus import load_corpus
from medcase.evaluate import compare, evaluate_run

DEFINITIONS = [("strict", 15), ("middle", 25), ("broad", 45)]
METRICS = ["P@5", "P@10", "MAP"]


def build_qrels(records, evalset, max_df, df):
    qrels = {}
    for q in evalset["queries"]:
        qid = q["qid"]
        if qid not in records:
            continue
        anchors = [a for a in q["anchors"] if df.get(a, 10**9) <= max_df]
        if not anchors:
            continue
        rel = {p: 2 for p, r in records.items()
               if p != qid and any(a in (r.mesh_major or []) for a in anchors)}
        if rel:
            qrels[qid] = rel
    return qrels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="data/scores_depth100.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    dump = json.loads((root / args.scores).read_text())["scores"]
    evalset = json.loads((root / "data" / "evalset.json").read_text())
    records = {r.pmid: r for r in load_corpus(root / "data" / "corpus.jsonl")}
    df = Counter(m for r in records.values() for m in (r.mesh_major or []))

    def arm(depth, rerank):
        out = {}
        for qid, d in dump.items():
            order = d["faiss_order"][:depth]
            if rerank:
                ce = d["ce_scores"]
                order = sorted(order, key=lambda p: ce[p], reverse=True)
            out[qid] = order
        return out

    runs = {
        "faiss@50": arm(50, False), "rerank@50": arm(50, True),
        "faiss@100": arm(100, False), "rerank@100": arm(100, True),
    }

    for label, max_df in DEFINITIONS:
        qrels = build_qrels(records, evalset, max_df, df)
        sub = {n: {q: r for q, r in run.items() if q in qrels}
               for n, run in runs.items()}
        res = {n: evaluate_run(r, qrels, name=n) for n, r in sub.items()}
        n_rel = sum(len(v) for v in qrels.values()) / len(qrels)

        print(f"\n{'=' * 74}")
        print(f"{label} relevance (df <= {max_df}) | {len(qrels)} queries | "
              f"{n_rel:.1f} relevant/query")
        print(f"{'=' * 74}")
        print(f"{'metric':<8}{'faiss@50':>10}{'rr@50':>9}{'faiss@100':>11}"
              f"{'rr@100':>9}{'rr100-rr50':>12}{'p(boot)':>10}")
        print("-" * 69)
        for m in METRICS:
            c = compare(res["rerank@50"], res["rerank@100"])["metrics"][m]
            print(f"{m:<8}{res['faiss@50'].metrics[m]:>10.4f}"
                  f"{res['rerank@50'].metrics[m]:>9.4f}"
                  f"{res['faiss@100'].metrics[m]:>11.4f}"
                  f"{res['rerank@100'].metrics[m]:>9.4f}"
                  f"{c['delta']:>+12.4f}{c['p_bootstrap']:>10.4f}")

    out = root / "data" / "depth100_results.json"
    out.write_text(json.dumps({"definitions": DEFINITIONS}, indent=2))
    print(f"\n-> scores replayed from {args.scores}")


if __name__ == "__main__":
    main()
