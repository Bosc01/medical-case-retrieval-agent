"""How few candidates can the cross-encoder score before quality drops?

Reranking cost is linear in candidates: 50 candidates means 50 BERT-base
forward passes. If the cases that end up in the top five were already near the
top of the FAISS ranking, scoring the tail is wasted work.

This needs no GPU. Reranking a subset of a scored candidate set is just sorting
that subset by the same scores, so every budget is derived from the rankings
already saved by scripts/run_eval.py. Each arm therefore uses identical scores
rather than a fresh run that might differ.
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medcase.evaluate import compare, evaluate_run, format_table

BUDGETS = [8, 10, 12, 15, 20, 25, 30, 50]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-latency-ms", type=float, default=2272.0,
                    help="measured cost of reranking the full candidate set")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="must match the reranker's batch size; cost is per batch")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    saved = json.loads((root / "data" / "eval_runs.json").read_text())
    qrels = saved["qrels"]
    faiss = saved["runs"]["faiss-baseline"]
    reranked = saved["runs"]["rerank-medcpt"]
    full = max(len(v) for v in faiss.values())

    runs = {"faiss-only": faiss}
    for b in BUDGETS:
        # Rerank the FAISS top-b, then append the untouched tail. A real system
        # would still return the tail; dropping it would silently change recall
        # at depths beyond b and make shallow budgets look better than they are.
        run = {}
        for qid, order in faiss.items():
            head, tail = order[:b], order[b:]
            rank = {p: i for i, p in enumerate(reranked[qid])}
            run[qid] = sorted(head, key=lambda p: rank.get(p, 10**9)) + tail
        runs[f"rerank-top{b}"] = run

    results = [evaluate_run(r, qrels, name=n) for n, r in runs.items()]
    print(format_table(results))

    by = {r.name: r for r in results}
    full_arm = by[f"rerank-top{full}"] if f"rerank-top{full}" in by else by[f"rerank-top{BUDGETS[-1]}"]

    # Cost is per BATCH, not per candidate. Every sequence in a batch is padded
    # to the longest one and the whole batch goes through the model together, so
    # 12 candidates and 30 candidates both cost exactly one forward pass at
    # batch_size 32. The only budget that saves anything is one that removes a
    # batch. Modelling this as linear in candidates, which an earlier version of
    # this script did, overstates the saving of every sub-batch budget.
    import math
    bs = args.batch_size
    full_batches = math.ceil(full / bs)
    per_batch = args.full_latency_ms / full_batches

    print(f"\n{'=' * 76}")
    print(f"Cost against quality, batch size {bs}.")
    print(f"Latency is per batch: {full} candidates is {full_batches} batches at "
          f"{per_batch:.0f} ms each.")
    print(f"{'=' * 76}\n")
    print(f"{'budget':<12}{'batches':>9}{'P@5':>8}{'vs full':>10}"
          f"{'p(boot)':>10}{'est ms':>9}{'saving':>9}")
    print("-" * 67)
    base_p5 = full_arm.metrics.get("P@5")
    for b in BUDGETS:
        arm = by[f"rerank-top{b}"]
        p5 = arm.metrics["P@5"]
        batches = math.ceil(b / bs)
        est = per_batch * batches
        if b == BUDGETS[-1]:
            pv = "-"
        else:
            c = compare(full_arm, arm)
            pv = f"{c['metrics']['P@5']['p_bootstrap']:.4f}"
        print(f"{'top-' + str(b):<12}{batches:>9}{p5:>8.4f}"
              f"{p5 - base_p5:>+10.4f}{pv:>10}{est:>9.0f}"
              f"{args.full_latency_ms / est:>8.1f}x")

    print(f"\nfaiss-only P@5: {by['faiss-only'].metrics['P@5']:.4f} (no rerank cost)")

    out = root / "data" / "budget_results.json"
    out.write_text(json.dumps({
        "budgets": BUDGETS,
        "full_candidates": full,
        "batch_size": bs,
        "full_latency_ms": args.full_latency_ms,
        "cost_model": "per batch, ceil(budget / batch_size)",
        "results": {r.name: r.metrics for r in results},
    }, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
