"""Fetch a real PubMed case-report corpus. Run once; the JSONL is the artifact."""
import logging, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medcase.corpus import build_corpus, save_corpus

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Spread across systems so queries have near-misses to be confused by --
# a corpus of one specialty makes reranking look better than it is.
TOPICS = [
    "myocarditis", "pulmonary embolism", "acute pancreatitis",
    "temporal arteritis", "Guillain-Barre syndrome", "pheochromocytoma",
    "systemic lupus erythematosus", "thyroid storm", "aortic dissection",
    "bacterial meningitis", "diabetic ketoacidosis", "sarcoidosis",
    "adrenal insufficiency", "infective endocarditis", "myasthenia gravis",
    "amyloidosis", "tuberculosis miliary", "hemophagocytic lymphohistiocytosis",
]

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "corpus.jsonl"
    records = build_corpus(TOPICS, per_topic=45)
    save_corpus(records, out)
    print(f"\n{len(records)} records -> {out}")
    print(f"topics: {len(TOPICS)}")
    if records:
        r = records[0]
        print(f"sample: PMID {r.pmid} | {r.title[:70]}... | abstract {len(r.abstract)} chars")
