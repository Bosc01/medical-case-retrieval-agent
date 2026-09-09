"""Embed the corpus and write the FAISS index."""
import logging, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medcase.corpus import load_corpus
from medcase.embed import MedCPTEmbedder
from medcase.index import CaseIndex

logging.basicConfig(level=logging.INFO, format="%(message)s")

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    records = load_corpus(root / "data" / "corpus.jsonl")
    print(f"embedding {len(records)} records...")
    t0 = time.perf_counter()
    idx = CaseIndex.build(records, MedCPTEmbedder())
    dt = time.perf_counter() - t0
    idx.save(root / "data" / "index")
    print(f"built {idx!r} in {dt:.1f}s ({dt/len(records)*1000:.1f} ms/record)")
