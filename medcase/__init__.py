"""Medical Case Retrieval Agent: MedCPT bi-encoder -> FAISS -> cross-encoder.

The public API is re-exported here, but it is resolved lazily through
``__getattr__`` rather than imported at module load. That is not stylistic.
Measured on this machine, importing ``medcase.corpus`` costs 0.04s and
``medcase.evaluate`` 0.04s, while ``medcase.embed`` costs 2.81s and
``medcase.finetune`` 2.85s because they pull in torch, transformers and faiss.
Eager re-exports would put that cost -- and half a gigabyte of resident
libraries -- on every process that only wanted to parse a corpus file or score
a run, including the eval tests. Lazy resolution keeps ``import medcase`` at
the cost of the submodule you actually touch.

Submodule names resolve here too, so ``import medcase`` followed by
``medcase.evaluate.compare(...)`` works without a separate import line.

Two names are deliberately NOT in the flat namespace, because exporting them
would silently pick one of several meanings:

* ``DEFAULT_MODEL`` is defined three times with different values --
  ``reranker.DEFAULT_MODEL`` is the MedCPT cross-encoder,
  ``generate.DEFAULT_MODEL`` is the OpenAI chat model, and ``finetune`` re-exports
  the reranker's. Import it from the module you mean.
* the ``finetune`` *function* would shadow the ``finetune`` *module*, so
  ``medcase.finetune`` here is the module; the trainer is
  ``medcase.finetune.finetune``.
"""

from __future__ import annotations

import importlib
from typing import Any

_SUBMODULES = (
    "corpus",
    "embed",
    "evaluate",
    "finetune",
    "generate",
    "index",
    "pipeline",
    "reranker",
)

# name -> submodule that defines it.
_EXPORTS: dict[str, str] = {
    # corpus
    "CaseRecord": "corpus",
    "build_corpus": "corpus",
    "fetch_records": "corpus",
    "load_corpus": "corpus",
    "save_corpus": "corpus",
    "search_pmids": "corpus",
    # embed
    "ARTICLE_MODEL": "embed",
    "MedCPTEmbedder": "embed",
    "QUERY_MODEL": "embed",
    # index
    "CaseIndex": "index",
    # reranker
    "BASELINE_MODEL": "reranker",
    "CrossEncoderReranker": "reranker",
    "RerankMetrics": "reranker",
    "ScoredDocument": "reranker",
    # pipeline
    "CaseSearchResult": "pipeline",
    "MedicalCaseAgent": "pipeline",
    "RetrievalTrace": "pipeline",
    # generate
    "AnswerResult": "generate",
    "CaseAnswerGenerator": "generate",
    "GenerationError": "generate",
    "SYSTEM_PROMPT": "generate",
    # evaluate
    "EvalResult": "evaluate",
    "NoRelevantDocuments": "evaluate",
    "average_precision": "evaluate",
    "compare": "evaluate",
    "evaluate_run": "evaluate",
    "format_comparison": "evaluate",
    "format_table": "evaluate",
    "judged_at_k": "evaluate",
    "mrr": "evaluate",
    "ndcg_at_k": "evaluate",
    "paired_bootstrap_p": "evaluate",
    "precision_at_k": "evaluate",
    "recall_at_k": "evaluate",
    "sign_test_p": "evaluate",
    # finetune (the module-shadowing `finetune` function is left qualified)
    "FineTuneConfig": "finetune",
    "TrainPair": "finetune",
    "mine_hard_negatives": "finetune",
    "save_model": "finetune",
}

# Submodules resolve as attributes but stay out of ``__all__``: a star-import
# that bound ``index`` and ``generate`` would shadow two of the most common
# local names in a caller's own code.
__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        value = getattr(importlib.import_module(f".{_EXPORTS[name]}", __name__), name)
    elif name in _SUBMODULES:
        value = importlib.import_module(f".{name}", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # Cached in the module dict, so the lazy path runs once per name.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__, *_SUBMODULES})
