"""Reranking latency, and what actually costs the time.

Correctness is checked before any timing: length-sorted batching must return
exactly the same scores as unsorted, because it only changes which sequences
share a batch. If that assertion fails the speedup is meaningless.
"""

from __future__ import annotations

import argparse, json, statistics, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medcase.corpus import load_corpus
from medcase.reranker import CrossEncoderReranker


def bench(texts, query, *, sort, batch_size, repeats):
    rr = CrossEncoderReranker(batch_size=batch_size, sort_by_length=sort)
    rr.warmup()
    runs, waste = [], None
    for _ in range(repeats):
        rr.clear_cache()  # otherwise run 2 is a cache read, not a forward pass
        t0 = time.perf_counter()
        scores = rr.score(query, texts)
        runs.append((time.perf_counter() - t0) * 1000)
        waste = rr.last_metrics.as_dict()["padding_waste"]
    return scores, min(runs), statistics.median(runs), waste


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="candidates per query")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    records = load_corpus(root / "data" / "corpus.jsonl")[: args.n]
    texts = [r.text for r in records]
    query = ("A 34-year-old woman presented with progressive proximal muscle "
             "weakness, ptosis and diplopia worsening through the day.")

    lens = sorted(len(t) for t in texts)
    print(f"{len(texts)} candidates | abstract chars: min {lens[0]} "
          f"median {lens[len(lens)//2]} max {lens[-1]}")
    print(f"batch {args.batch_size}, {args.repeats} repeats, reporting best-of\n")

    unsorted_scores, u_min, u_med, u_waste = bench(
        texts, query, sort=False, batch_size=args.batch_size, repeats=args.repeats)
    sorted_scores, s_min, s_med, s_waste = bench(
        texts, query, sort=True, batch_size=args.batch_size, repeats=args.repeats)

    # Correctness gate.
    worst = max(abs(a - b) for a, b in zip(unsorted_scores, sorted_scores))
    print(f"max score difference sorted vs unsorted: {worst:.2e}")
    assert worst < 1e-2, f"length sorting changed scores by {worst}"
    print("scores match, so the comparison below is like for like\n")

    print(f"{'variant':<22}{'best ms':>9}{'median ms':>11}{'padding waste':>15}")
    print("-" * 57)
    print(f"{'batch order':<22}{u_min:>9.0f}{u_med:>11.0f}{u_waste:>14.1%}")
    print(f"{'length-sorted':<22}{s_min:>9.0f}{s_med:>11.0f}{s_waste:>14.1%}")
    speedup = u_min / s_min if s_min else 0
    print(f"\nspeedup: {speedup:.2f}x  ({u_min - s_min:+.0f} ms on {len(texts)} candidates)")

    out = root / "data" / "bench_results.json"
    out.write_text(json.dumps({
        "n_candidates": len(texts), "batch_size": args.batch_size,
        "unsorted_ms": u_min, "sorted_ms": s_min,
        "unsorted_padding_waste": u_waste, "sorted_padding_waste": s_waste,
        "speedup": speedup,
    }, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
