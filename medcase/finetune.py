"""Fine-tuning the cross-encoder on mined hard negatives.

Read this part before running it. With a few dozen labelled queries -- the
usual situation here -- an off-the-shelf domain cross-encoder will normally
beat anything fine-tuned on them. MedCPT already saw 255M PubMed search logs;
a few hundred local pairs are more than enough to move its weights and nowhere
near enough to move them somewhere better. The failure mode is quiet: training
loss drops, the model memorises the training queries, and the general PubMed
relevance signal it arrived with erodes.

Fine-tuning starts to pay when the labels are real -- thousands of judged
pairs from someone who knows the domain, plus a held-out set to prove the gain
on. Until then this module is machinery that is ready when the labels are, and
the base model is the one that should be serving.

What makes it work when the labels do exist is where the negatives come from.
A randomly drawn document is trivially irrelevant and teaches the model
nothing it does not already know. The negatives worth training on are the ones
the bi-encoder ranks highly and gets wrong, because those are exactly the
confusions the cross-encoder is there to fix. ``mine_hard_negatives`` takes
them out of the FAISS index rather than out of a random sample.

Nothing in this module reports a ranking improvement. It measures losses, and
a lower training loss is not a better ranker -- that claim needs the eval
harness on held-out queries.
"""

from __future__ import annotations

import inspect
import json
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Reuse the reranker's device pick and text extraction so training sees the
# documents exactly as inference will.
from .reranker import DEFAULT_MODEL, ScoredDocument, _default_text_of, _resolve_device

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "models/cross-encoder-finetuned"


@dataclass
class TrainPair:
    """One (query, document) example. ``label`` is 1.0 relevant, 0.0 not.

    Under the BCE objective a label anywhere in [0, 1] is meaningful -- a soft
    target. Outside it the objective is not: BCE with a target above 1 has a
    gradient that never changes sign, so the loss is unbounded below and a
    graded qrel of 2 trains the model to run the logit to infinity while
    printing an ever-more-negative "improving" loss. Rescale graded judgements
    into [0, 1] before building pairs; ``finetune`` rejects anything else. The
    2-label fallback binarises at 0.5 instead.
    """

    query: str
    doc_text: str
    label: float
    doc_id: str = ""


# -- hard negative mining ----------------------------------------------

# The index and embedder are owned by another module, so hits are normalised
# defensively here rather than against one assumed return shape.
_QUERY_ENCODERS = (
    "encode_queries",
    "embed_queries",
    "encode_query",
    "embed_query",
    "encode",
    "embed",
    "__call__",
)
_CORPUS_ATTRS = ("records", "documents", "docs", "corpus", "items")
_ID_ATTRS = ("pmid", "doc_id", "docid", "id", "identifier")
_VECTOR_ARGS = ("vec", "vecs", "vector", "vectors", "x", "query_vector", "embedding")


def _as_matrix(out: Any) -> np.ndarray:
    if isinstance(out, torch.Tensor):
        out = out.detach().cpu().numpy()
    arr = np.asarray(out, dtype=np.float32)
    return arr.reshape(1, -1) if arr.ndim == 1 else arr


def _encode_query(embedder: Any, query: str) -> np.ndarray:
    for name in _QUERY_ENCODERS:
        fn = getattr(embedder, name, None)
        if not callable(fn):
            continue
        try:
            return _as_matrix(fn([query]))
        except TypeError:
            pass
        try:
            return _as_matrix(fn(query))
        except TypeError:
            continue
    raise TypeError(
        f"embedder {type(embedder).__name__} exposes none of {_QUERY_ENCODERS}"
    )


def _is_numeric_array(x: Any) -> bool:
    if isinstance(x, (np.ndarray, torch.Tensor)):
        return True
    return isinstance(x, (list, tuple)) and all(
        isinstance(v, (int, float, np.integer, np.floating)) for v in x
    )


def _normalize_hits(raw: Any) -> list[Any]:
    if isinstance(raw, tuple) and len(raw) == 2:
        a, b = raw
        if _is_numeric_array(a) and _is_numeric_array(b):
            # Bare faiss: (distances, ids), batch of one, -1 pads short results.
            ids = np.asarray(b)
            if ids.ndim == 2:
                ids = ids[0]
            return [int(i) for i in ids.tolist() if int(i) >= 0]
        if _is_numeric_array(a):
            return list(b)  # (scores, hits)
        if _is_numeric_array(b):
            return list(a)  # (hits, scores)
        # Neither half is a score array, so this is not a parallel pair -- it
        # is two hits. Splitting it would drop one of them.
    return list(raw)


def _corpus_of(index: Any) -> Sequence[Any] | None:
    for attr in _CORPUS_ATTRS:
        seq = getattr(index, attr, None)
        if isinstance(seq, Sequence) and not isinstance(seq, (str, bytes)):
            return seq
    return None


def _doc_id_of(obj: Any) -> str:
    for attr in _ID_ATTRS:
        value = getattr(obj, attr, None)
        if isinstance(value, (str, int)):
            return str(value)
        if isinstance(obj, Mapping) and isinstance(obj.get(attr), (str, int)):
            return str(obj[attr])
    return ""


def _resolve_hit(index: Any, hit: Any) -> tuple[str, str]:
    """(doc_id, doc_text) for one retrieved hit. Empty id means unresolvable."""
    if isinstance(hit, ScoredDocument):
        hit = hit.document
    elif isinstance(hit, (tuple, list)) and len(hit) == 2:
        first, second = hit
        hit = second if isinstance(first, (int, float, np.number)) else first
    if isinstance(hit, (int, np.integer)):
        corpus = _corpus_of(index)
        if corpus is None or not 0 <= int(hit) < len(corpus):
            return "", ""
        hit = corpus[int(hit)]
    return _doc_id_of(hit), _default_text_of(hit)


def _search_call(index: Any, embedder: Any, query: str, k: int) -> Any:
    """Call whichever ``search`` the retriever exposes.

    ``CaseIndex.search`` takes (query, embedder, k); other wrappers take a
    vector, or a bare string and embed internally. The signature picks, so a
    TypeError raised *inside* a working search is never mistaken for a
    calling-convention mismatch.
    """
    search = getattr(index, "search", None)
    if not callable(search):
        raise TypeError(f"index {type(index).__name__} has no .search()")
    names: list[str] = []
    try:
        names = [
            p.name
            for p in inspect.signature(search).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
    except (TypeError, ValueError):  # C bindings expose no signature
        pass

    if embedder is None:
        return search(query, k)
    if "embedder" in names:
        return search(query, embedder, k)
    if names and names[0] in _VECTOR_ARGS:
        return search(_encode_query(embedder, query), k)
    try:
        return search(_encode_query(embedder, query), k)
    except TypeError:
        return search(query, k)


def _retrieve(index: Any, embedder: Any, query: str, k: int) -> list[tuple[str, str]]:
    raw = _search_call(index, embedder, query, k)
    return [_resolve_hit(index, hit) for hit in _normalize_hits(raw)]


def _normalize_queries(queries: Any) -> list[tuple[Any, str]]:
    """(query_id, text) pairs. The id keeps its original type on purpose.

    Coercing it to ``str`` here would make an int-keyed qrels file -- the
    ordinary TREC shape -- miss on every lookup, and a query whose judgements
    cannot be found is discarded as unjudged rather than raising.
    """
    if isinstance(queries, Mapping):
        return [(k, str(v)) for k, v in queries.items()]
    out: list[tuple[Any, str]] = []
    for q in queries:
        if isinstance(q, str):
            out.append((q, q))
        elif isinstance(q, (tuple, list)) and len(q) == 2:
            out.append((q[0], str(q[1])))
        else:
            raise TypeError(f"cannot read a (query_id, text) pair from {q!r}")
    return out


def _qrels_get(qrels: Mapping[Any, Any], key: Any) -> Any:
    """Judgements for ``key``, tolerating an id stored as int on one side."""
    for candidate in (key, str(key)):
        try:
            judged = qrels.get(candidate)
        except TypeError:  # unhashable query id
            continue
        if judged is not None:
            return judged
    return None


def _relevant_ids(qrels: Mapping[Any, Any], key: Any) -> set[str]:
    judged = _qrels_get(qrels, key)
    if judged is None:
        return set()
    if isinstance(judged, Mapping):
        return {str(d) for d, grade in judged.items() if float(grade) > 0}
    if isinstance(judged, (str, bytes)):
        return {str(judged)}
    return {str(d) for d in judged}


def _text_lookup(documents: Any) -> dict[str, str]:
    if documents is None:
        return {}
    if isinstance(documents, Mapping):
        return {
            str(k): v if isinstance(v, str) else _default_text_of(v)
            for k, v in documents.items()
        }
    table: dict[str, str] = {}
    for doc in documents:
        doc_id = _doc_id_of(doc)
        if doc_id:
            table[doc_id] = _default_text_of(doc)
    return table


def mine_hard_negatives(
    queries: Any,
    qrels: Mapping[Any, Any],
    index: Any,
    embedder: Any,
    n_candidates: int = 50,
    n_negatives: int = 4,
    skip_top: int = 0,
    *,
    documents: Any = None,
    include_positives: bool = True,
) -> list[TrainPair]:
    """Build training pairs whose negatives are what FAISS actually confuses.

    Retrieves ``n_candidates`` per query, drops everything ``qrels`` marks
    relevant, and keeps the highest-ranked survivors as negatives -- the
    documents the bi-encoder ranked above the answer.

    ``skip_top`` discards that many of the hardest survivors first. In a
    sparsely judged pool the top non-relevant hits are frequently unjudged
    positives, and training on those teaches the model to demote documents
    that were in fact good. Skipping them trades away the most informative
    negatives to avoid that; with exhaustive judgements, leave it at 0.

    ``documents`` (a mapping of doc_id to text, or a sequence of records with
    a pmid) lets relevant documents that were never retrieved still be emitted
    as positives.
    """
    lookup = _text_lookup(documents)
    # Materialised once: ``queries`` may be a one-shot iterable, and counting
    # it a second time for the log would report 0 for a run that trained on
    # every one of them.
    items = _normalize_queries(queries)
    pairs: list[TrainPair] = []
    skipped_unjudged = 0
    unresolved_hits = 0
    missing_positives = 0

    for key, text in items:
        relevant = _relevant_ids(qrels, key)
        if not relevant:
            # No judgements means no way to tell a negative from a positive.
            skipped_unjudged += 1
            continue

        hits = _retrieve(index, embedder, text, n_candidates)
        seen_positive: set[str] = set()
        negatives: list[tuple[str, str]] = []
        for doc_id, doc_text in hits:
            if not doc_id:
                # Unmatchable against qrels, so it cannot be safely called a
                # negative.
                unresolved_hits += 1
                continue
            if doc_id in relevant:
                seen_positive.add(doc_id)
                if include_positives and doc_text:
                    pairs.append(TrainPair(text, doc_text, 1.0, doc_id))
            else:
                negatives.append((doc_id, doc_text))

        for doc_id, doc_text in negatives[skip_top : skip_top + n_negatives]:
            if doc_text:
                pairs.append(TrainPair(text, doc_text, 0.0, doc_id))

        if include_positives:
            for doc_id in sorted(relevant - seen_positive):
                doc_text = lookup.get(doc_id, "")
                if doc_text:
                    pairs.append(TrainPair(text, doc_text, 1.0, doc_id))
                else:
                    missing_positives += 1

    n_pos = sum(1 for p in pairs if p.label > 0)
    logger.info(
        "mined %d pairs (%d positive, %d negative) from %d queries; "
        "%d queries had no judgements, %d hits unresolvable, %d positives had no text",
        len(pairs),
        n_pos,
        len(pairs) - n_pos,
        len(items),
        skipped_unjudged,
        unresolved_hits,
        missing_positives,
    )
    return pairs


# -- fine-tuning -------------------------------------------------------


@dataclass
class FineTuneConfig:
    model_name: str = DEFAULT_MODEL
    epochs: int = 2
    lr: float = 2e-5
    batch_size: int = 8
    max_length: int = 512
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    seed: int = 42
    eval_every: int | None = None
    output_dir: str = DEFAULT_OUTPUT_DIR
    device: str | None = None
    max_grad_norm: float = 1.0
    num_labels: int | None = None


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _load_for_training(config: FineTuneConfig) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    kwargs: dict[str, Any] = {}
    if config.num_labels is not None:
        kwargs = {"num_labels": config.num_labels, "ignore_mismatched_sizes": True}
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name, **kwargs
    )
    return tokenizer, model


def _param_groups(model: Any, weight_decay: float) -> list[dict[str, Any]]:
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    decayed, plain = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (plain if any(n in name for n in no_decay) else decayed).append(param)
    return [
        {"params": decayed, "weight_decay": weight_decay},
        {"params": plain, "weight_decay": 0.0},
    ]


def _linear_schedule(optimizer: Any, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return step / max(1, warmup_steps)
        remaining = total_steps - warmup_steps
        return max(0.0, (total_steps - step) / max(1, remaining))

    return LambdaLR(optimizer, lr_lambda)


def _check_bce_labels(pairs: Sequence[TrainPair], what: str) -> None:
    """Reject labels BCE cannot express, loudly.

    A target above 1 gives BCE a gradient that never changes sign: the loss
    falls without bound as the logit grows, so a graded qrel of 2 would train
    the model to saturate and report a large negative "train loss" that looks
    like a breakthrough. Caught here rather than written into the metrics file.
    """
    bad = sorted({p.label for p in pairs if not 0.0 <= float(p.label) <= 1.0})
    if bad:
        raise ValueError(
            f"{what} carry labels outside [0, 1]: {bad}. The single-logit head "
            "trains with BCE, which is unbounded below for targets above 1 -- "
            "rescale graded judgements (e.g. grade/max_grade) first."
        )


def _batch_loss(
    model: Any,
    tokenizer: Any,
    batch: Sequence[TrainPair],
    device: str,
    max_length: int,
    loss_fn: Any,
    binary: bool,
) -> torch.Tensor:
    encoded = tokenizer(
        [p.query for p in batch],
        [p.doc_text for p in batch],
        truncation="only_second",  # the abstract gets cut, never the query
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    logits = model(**encoded).logits
    if binary:
        target = torch.tensor(
            [float(p.label) for p in batch], dtype=torch.float32, device=device
        )
        return loss_fn(logits.squeeze(-1), target)
    # A 2-class head has no room for a graded label, so it has to binarise.
    target = torch.tensor(
        [1 if p.label >= 0.5 else 0 for p in batch], dtype=torch.long, device=device
    )
    return loss_fn(logits, target)


@torch.no_grad()
def _eval_loss(
    model: Any,
    tokenizer: Any,
    pairs: Sequence[TrainPair],
    device: str,
    config: FineTuneConfig,
    loss_fn: Any,
    binary: bool,
) -> float:
    was_training = model.training
    model.eval()
    total, seen = 0.0, 0
    for start in range(0, len(pairs), config.batch_size):
        batch = pairs[start : start + config.batch_size]
        loss = _batch_loss(
            model, tokenizer, batch, device, config.max_length, loss_fn, binary
        )
        total += float(loss) * len(batch)
        seen += len(batch)
    if was_training:
        model.train()
    return total / seen if seen else float("nan")


def finetune(
    pairs: Sequence[TrainPair],
    config: FineTuneConfig,
    eval_pairs: Sequence[TrainPair] | None = None,
    *,
    save: bool = True,
) -> dict[str, Any]:
    """Train the cross-encoder on ``pairs`` and return the measured losses.

    The returned dict carries only what was actually computed: per-epoch mean
    train loss, and eval loss per epoch (plus per ``eval_every`` steps) when
    ``eval_pairs`` is given. No ranking metric is produced here -- a loss curve
    says the optimiser worked, not that retrieval improved.

    The saved directory reloads straight into the reranker:
    ``CrossEncoderReranker(model_name=result["output_dir"])``.
    """
    # Checked before the weights load, so a bad config fails in milliseconds
    # rather than half a gigabyte later.
    if not pairs:
        raise ValueError("no training pairs")
    if config.epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {config.epochs}")
    if config.batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {config.batch_size}")

    _set_seeds(config.seed)
    device = _resolve_device(config.device)
    tokenizer, model = _load_for_training(config)
    model = model.to(device)
    model.train()

    num_labels = int(model.config.num_labels)
    binary = num_labels == 1
    loss_fn = torch.nn.BCEWithLogitsLoss() if binary else torch.nn.CrossEntropyLoss()
    if binary:
        _check_bce_labels(pairs, "pairs")
        if eval_pairs:
            _check_bce_labels(eval_pairs, "eval_pairs")

    steps_per_epoch = math.ceil(len(pairs) / config.batch_size)
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    optimizer = torch.optim.AdamW(
        _param_groups(model, config.weight_decay), lr=config.lr
    )
    scheduler = _linear_schedule(optimizer, warmup_steps, total_steps)

    logger.info(
        "fine-tuning %s on %s: %d pairs, %d steps, objective=%s",
        config.model_name,
        device,
        len(pairs),
        total_steps,
        "bce" if binary else "cross_entropy",
    )

    rng = random.Random(config.seed)
    order = list(range(len(pairs)))
    train_losses: list[float] = []
    eval_losses: list[float] = []
    eval_history: list[dict[str, float]] = []
    global_step = 0
    t0 = time.perf_counter()

    for epoch in range(config.epochs):
        rng.shuffle(order)
        running, seen = 0.0, 0
        for start in range(0, len(order), config.batch_size):
            batch = [pairs[i] for i in order[start : start + config.batch_size]]
            loss = _batch_loss(
                model, tokenizer, batch, device, config.max_length, loss_fn, binary
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            running += float(loss.detach()) * len(batch)
            seen += len(batch)
            global_step += 1

            if eval_pairs and config.eval_every and global_step % config.eval_every == 0:
                step_loss = _eval_loss(
                    model, tokenizer, eval_pairs, device, config, loss_fn, binary
                )
                eval_history.append({"step": global_step, "eval_loss": step_loss})
                logger.info("step %d eval_loss %.4f", global_step, step_loss)

        epoch_loss = running / seen
        train_losses.append(epoch_loss)
        if eval_pairs:
            eval_losses.append(
                _eval_loss(model, tokenizer, eval_pairs, device, config, loss_fn, binary)
            )
            logger.info(
                "epoch %d train_loss %.4f eval_loss %.4f",
                epoch + 1,
                epoch_loss,
                eval_losses[-1],
            )
        else:
            logger.info("epoch %d train_loss %.4f", epoch + 1, epoch_loss)

    result: dict[str, Any] = {
        "model_name": config.model_name,
        "device": device,
        "objective": "bce" if binary else "cross_entropy",
        "num_labels": num_labels,
        "train_pairs": len(pairs),
        "eval_pairs": len(eval_pairs) if eval_pairs else 0,
        "positive_train_pairs": sum(1 for p in pairs if p.label > 0),
        "epochs": config.epochs,
        "steps": global_step,
        "warmup_steps": warmup_steps,
        "train_loss_per_epoch": train_losses,
        "eval_loss_per_epoch": eval_losses if eval_pairs else None,
        "eval_history": eval_history,
        "final_train_loss": train_losses[-1],
        "final_eval_loss": eval_losses[-1] if eval_losses else None,
        "seconds": round(time.perf_counter() - t0, 2),
        "output_dir": None,
    }

    if save and config.output_dir:
        out = save_model(model, tokenizer, config.output_dir)
        result["output_dir"] = str(out)
        (out / "finetune_metrics.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    return result


def save_model(model: Any, tokenizer: Any, output_dir: str | Path) -> Path:
    """Write weights and tokenizer where ``from_pretrained`` can find them."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    logger.info("saved fine-tuned model to %s", out)
    return out
