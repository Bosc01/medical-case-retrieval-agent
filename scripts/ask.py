"""Ask the agent a clinical question and show what reranking changed."""
from __future__ import annotations

import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medcase.embed import MedCPTEmbedder
from medcase.generate import CaseAnswerGenerator
from medcase.index import CaseIndex
from medcase.pipeline import MedicalCaseAgent
from medcase.reranker import CrossEncoderReranker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+")
    ap.add_argument("--retrieve-k", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-generate", action="store_true")
    args = ap.parse_args()
    query = " ".join(args.query)

    root = Path(__file__).resolve().parents[1]
    agent = MedicalCaseAgent(
        index=CaseIndex.load(root / "data" / "index"),
        embedder=MedCPTEmbedder(),
        reranker=CrossEncoderReranker(),
        generator=CaseAnswerGenerator(),
        retrieve_k=args.retrieve_k,
        top_k=args.top_k,
    )

    result = agent.search(
        query, rerank=not args.no_rerank, generate=not args.no_generate
    )

    print(f"\nQUERY: {query}\n{'=' * 78}")
    for i, d in enumerate(result.documents, 1):
        rec = d.document
        moved = ""
        if d.rank_delta:
            moved = f"  [FAISS #{d.retrieval_rank + 1} -> #{d.rerank_rank + 1}]"
        print(f"\n{i}. {rec.title[:88]}")
        print(f"   {rec.citation()}  score {d.rerank_score:.2f}{moved}")
        print(f"   {rec.url}")

    print(f"\n{'-' * 78}\n{result.trace.summary()}")
    if result.trace.promotions:
        print(f"reranker promoted {len(result.trace.promotions)} case(s) into the top "
              f"{args.top_k} that FAISS ranked lower")

    if result.answer is not None:
        a = result.answer
        print(f"\n{'=' * 78}\nANSWER ({a.model}{', fallback' if a.used_fallback else ''})\n")
        print(a.text)
        print(f"\ncitations: {', '.join(a.citations)}")


if __name__ == "__main__":
    main()
