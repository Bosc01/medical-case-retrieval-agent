"""Build a case-based retrieval eval set with expert relevance labels.

The labels are NOT authored here. Each query is a real case report's own
presentation text, and two cases count as relevant when NLM's indexers assigned
them the same specific MeSH descriptor. So the ground truth comes from human
indexing that is independent of both the embedder and the reranker being
compared -- neither system had any hand in producing it.

Graded: 2 when the shared descriptor is a major topic of BOTH cases (the
article is substantively about it), 1 when shared but minor for either.
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medcase.corpus import load_corpus

SEED = 42
N_QUERIES = 120
# A descriptor shared by half the corpus says nothing about relevance; one held
# by a handful of cases is a real clinical signal. Bound it at both ends.
MIN_DF, MAX_DF = 4, 45
MIN_RELEVANT = 3


def presentation(record) -> str:
    """The part a clinician would actually type: what the patient presented with.

    Structured abstracts often lead with a BACKGROUND section about the disease
    in general, which is not the case at hand -- prefer the CASE section.
    """
    abstract = record.abstract
    m = re.search(
        r"(?:CASE (?:PRESENTATION|REPORT|DESCRIPTION|SUMMARY)|PATIENT CONCERNS)\s*:\s*(.+)",
        abstract,
        flags=re.IGNORECASE,
    )
    body = m.group(1) if m else abstract
    sentences = re.split(r"(?<=[.!?])\s+", body)
    return " ".join(sentences[:3]).strip()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    records = load_corpus(root / "data" / "corpus.jsonl")
    by_pmid = {r.pmid: r for r in records}

    df = Counter(m for r in records for m in (r.mesh_major or []))
    specific = {m for m, n in df.items() if MIN_DF <= n <= MAX_DF}
    print(f"{len(records)} records | {len(specific)} specific descriptors "
          f"(df {MIN_DF}-{MAX_DF})")

    postings: dict[str, list[str]] = {}
    for r in records:
        for m in r.mesh_major or []:
            if m in specific:
                postings.setdefault(m, []).append(r.pmid)

    candidates = []
    for r in records:
        anchors = [m for m in (r.mesh_major or []) if m in specific]
        if not anchors:
            continue
        text = presentation(r)
        if len(text) < 120:  # too thin to be a meaningful query
            continue
        qrels: dict[str, int] = {}
        for a in anchors:
            for pmid in postings[a]:
                if pmid == r.pmid:
                    continue
                other = by_pmid[pmid]
                major_both = a in (other.mesh_major or [])
                grade = 2 if major_both else 1
                qrels[pmid] = max(qrels.get(pmid, 0), grade)
        # Cases sharing only a minor descriptor are weaker evidence; keep them
        # graded rather than dropped so nDCG can use the distinction.
        for a in anchors:
            for pmid, other in by_pmid.items():
                if pmid == r.pmid or pmid in qrels:
                    continue
                if a in (other.mesh_terms or []):
                    qrels[pmid] = 1
        if len(qrels) < MIN_RELEVANT:
            continue
        candidates.append({
            "qid": r.pmid,
            "query": text,
            "anchors": anchors,
            "source_title": r.title,
            "qrels": qrels,
        })

    rng = random.Random(SEED)
    rng.shuffle(candidates)
    chosen = candidates[:N_QUERIES]

    out = {
        "seed": SEED,
        "n_records": len(records),
        "min_df": MIN_DF,
        "max_df": MAX_DF,
        "queries": chosen,
    }
    path = root / "data" / "evalset.json"
    path.write_text(json.dumps(out, indent=2))

    rel_counts = [len(q["qrels"]) for q in chosen]
    grade2 = [sum(1 for g in q["qrels"].values() if g == 2) for q in chosen]
    print(f"{len(candidates)} eligible queries, kept {len(chosen)}")
    print(f"relevant per query: min {min(rel_counts)} median "
          f"{sorted(rel_counts)[len(rel_counts)//2]} max {max(rel_counts)}")
    print(f"grade-2 per query: median {sorted(grade2)[len(grade2)//2]}")
    print(f"-> {path}")
    print(f"\nexample query (PMID {chosen[0]['qid']}, anchors {chosen[0]['anchors']}):")
    print("  " + chosen[0]["query"][:220] + "...")


if __name__ == "__main__":
    main()
