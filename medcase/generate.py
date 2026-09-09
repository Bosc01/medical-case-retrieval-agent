"""Grounded answer generation for the Medical Case Retrieval Agent.

Retrieval and reranking decide which cases a clinician sees. This stage decides
what is said about them, and it is the only stage that can invent a clinical
claim. So the contract is narrow: every sentence traces to a numbered case that
was actually retrieved, and the PMIDs that come back in the answer are checked
against the PMIDs that went into the prompt.

Two paths, and which one ran is always visible on ``AnswerResult``:

* an OpenAI chat-completions call, when a key is configured;
* an extractive fallback that copies sentences out of the retrieved abstracts,
  used when no key is set. It deliberately does not imitate a model answer --
  ``used_fallback=True`` means the text was assembled by string handling, so it
  reads like excerpts because it is excerpts.

The HTTP call is stdlib ``urllib`` so the pipeline carries no client
dependency. Swapping providers means rewriting ``_chat_completion`` and
nothing else.
"""

from __future__ import annotations

import json
import logging
import os
import re
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Sequence

try:
    # A TLS-intercepting proxy breaks this call exactly as it breaks
    # E-utilities, so share corpus.py's trust store instead of rebuilding it.
    from .corpus import _ssl_context
except ImportError:  # imported as a loose module, or corpus.py restructured

    def _ssl_context() -> ssl.SSLContext:
        return ssl.create_default_context()


logger = logging.getLogger(__name__)

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o"
FALLBACK_MODEL = "extractive-fallback"

_TRUNC_MARK = " [case text truncated to fit context budget]"

# ``PMIDs?`` because a "Cases used" line is normally written "PMIDs 1, 2", and
# reading only "PMID" there drops every citation on it. The trailing \b makes an
# over-long digit run match nothing rather than silently yielding its first nine
# digits as a PMID that was never written.
_PMID_RE = re.compile(r"PMIDs?[:\s#=-]*([0-9]{4,9})\b", re.IGNORECASE)

# Continues a comma-separated run after one PMID marker. 7 digits minimum, not
# 4: a short number after a comma is far more often a year than a PMID, and a
# citation invented by the parser is worse than one it missed.
_PMID_TAIL_RE = re.compile(
    r"\s*(?:,|;|/|&|\band\b)\s*(?:PMIDs?[:\s#=-]*)?([0-9]{7,9})\b", re.IGNORECASE
)

# Sentence starts are "not lowercase ASCII" rather than "A-Z" so an accented or
# CJK opening word still ends a sentence; "8.2 ng/mL" stays intact because the
# following character is lowercase. CJK stops usually carry no trailing space.
_SENTENCE_RE = re.compile(r"(?:(?<=[.!?])\s+|(?<=[。！？])\s*)(?=[(\[]|[^\Wa-z\d_])")
_WORD_RE = re.compile(r"[^\W_]{3,}")

# Words that match everything and therefore rank nothing.
_STOP = frozenset(
    """a an and are as at be but by for from had has have how in into is it its of on or
    that the their then there these this to was were what when which who why with
    patient patient's case report reports""".split()
)

SYSTEM_PROMPT = """You are a literature synthesis assistant for clinicians. You are \
given a clinical question and a numbered list of published case reports retrieved from \
PubMed.

Rules, in priority order:
1. Use ONLY the numbered cases supplied in this prompt. Do not draw on background
   knowledge, and do not add any clinical detail that is not stated in a case.
2. Cite inline for every clinical claim, as (PMID 12345678). A sentence that carries a
   clinical claim and no PMID is a defect. Cite only PMIDs that appear in the list.
3. If the retrieved cases do not support an answer, say so plainly, name the part that
   is unsupported, and stop. Do not fill the gap from memory.
4. Do NOT diagnose, and do NOT give a treatment or management recommendation for any
   real patient. What you produce is a synthesis of published literature for review by
   a qualified clinician, not clinical advice.
5. Where cases disagree, say so rather than averaging them. These are case reports:
   they establish that something has been observed, never how often.

Format: a short synthesis (3-6 sentences), then a "Cases used" line listing the PMIDs
you actually cited, each written in full as "PMID 12345678"."""


class GenerationError(RuntimeError):
    """The model call failed. Raised instead of returning invented text."""


@dataclass
class AnswerResult:
    """An answer plus the evidence trail needed to audit it.

    ``grounded`` is a provenance check, not an entailment check: it is True when
    every PMID cited in ``text`` was one of the cases put into the prompt. It
    does not verify that each claim is actually supported by its source.
    """

    text: str
    citations: list[str] = field(default_factory=list)
    model: str = ""
    grounded: bool = False
    prompt_tokens_estimate: int = 0
    used_fallback: bool = False


def _chat_completion(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    url: str | None = None,
    timeout: float = 60.0,
) -> str:
    """POST one chat completion and return the assistant message content.

    Kept deliberately small and free of pipeline concepts so another provider
    can be dropped in here. ``url`` resolves at call time rather than in the
    signature, so pointing the module at a different endpoint actually works.
    """
    url = url or OPENAI_URL
    payload = json.dumps(
        {"model": model, "messages": messages, "temperature": temperature}
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise GenerationError(f"{url} returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise GenerationError(f"could not reach {url}: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise GenerationError(f"{url} returned a non-JSON body: {exc}") from exc

    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise GenerationError(f"unexpected response shape: {body!r:.500}") from exc
    if not isinstance(content, str) or not content.strip():
        raise GenerationError("model returned an empty message")
    return content.strip()


def _pmid_of(doc: Any) -> str:
    value = getattr(doc, "pmid", None)
    if value is None and isinstance(doc, dict):
        value = doc.get("pmid")
    return str(value) if value else ""


def _title_of(doc: Any) -> str:
    value = getattr(doc, "title", None)
    if value is None and isinstance(doc, dict):
        value = doc.get("title")
    return str(value) if value else ""


def _citation_of(doc: Any) -> str:
    cite = getattr(doc, "citation", None)
    if callable(cite):
        return str(cite())
    pmid = _pmid_of(doc)
    return f"PMID {pmid}" if pmid else "unknown source"


def _body_of(scored: Any) -> str:
    """Abstract alone when the record has one, else the full scored text.

    Used only for pulling quotes: the title already sits on the header line, so
    quoting it back spends one of the two sentences the reader gets per case.
    """
    doc = scored.document
    abstract = getattr(doc, "abstract", None)
    if abstract is None and isinstance(doc, dict):
        abstract = doc.get("abstract")
    if isinstance(abstract, str) and abstract.strip():
        return abstract
    return scored.text


def _case_block(number: int, scored: Any) -> str:
    """One numbered case as it appears in the prompt.

    ``ScoredDocument.text`` already carries title + abstract, so the header adds
    only the identifier the model is required to cite.
    """
    return f"[{number}] {_citation_of(scored.document)}\n{scored.text}".strip()


def _truncate_block(block: str, budget: int) -> str:
    """Cut a case block's body, never its header line.

    The header carries the PMID the answer has to cite. Slicing the block as one
    string throws that away first at small budgets, leaving a numbered case with
    no identifier attached to it -- evidence the reader cannot trace. So the
    header survives even when it alone exceeds the budget.
    """
    header, _, body = block.partition("\n")
    room = budget - len(header) - len(_TRUNC_MARK)
    if room <= 0:
        return header + _TRUNC_MARK
    return f"{header}\n{body[:room]}{_TRUNC_MARK}"


def _fit_to_budget(blocks: list[str], budget: int) -> tuple[list[str], int]:
    """Drop whole cases from the bottom until the context fits.

    Lowest-ranked first, because the reranker already put the least relevant
    case there. A single case larger than the whole budget is truncated rather
    than dropped, with a marker so the cut is visible to the model and in logs.
    """
    kept = list(blocks)
    sep = 2  # blank line between blocks
    while len(kept) > 1 and sum(len(b) + sep for b in kept) > budget:
        kept.pop()
    dropped = len(blocks) - len(kept)
    if kept and len(kept[0]) > budget:
        kept[0] = _truncate_block(kept[0], budget)
    return kept, dropped


def _sentences(text: str) -> list[str]:
    """Split on paragraphs first, then on sentence punctuation.

    ``CaseRecord.text`` joins title and abstract with a blank line, and a title
    carries no terminal punctuation -- splitting on punctuation alone quotes the
    title welded to the abstract's first sentence.
    """
    out: list[str] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        out.extend(s.strip() for s in _SENTENCE_RE.split(block.strip()) if s.strip())
    return out


def _keywords(query: str) -> set[str]:
    return {w for w in _WORD_RE.findall(query.lower()) if w not in _STOP}


def _cited_pmids(text: str) -> list[str]:
    """PMIDs written in ``text``, in order of first appearance.

    List continuations are followed as well as standalone markers: the answer
    format asks for a closing "Cases used" line, and matching only the marker
    would silently return fewer citations than the answer actually made.
    """
    seen: list[str] = []
    pos = 0
    while (match := _PMID_RE.search(text, pos)) is not None:
        pos = match.end()
        if match.group(1) not in seen:
            seen.append(match.group(1))
        while (tail := _PMID_TAIL_RE.match(text, pos)) is not None:
            pos = tail.end()
            if tail.group(1) not in seen:
                seen.append(tail.group(1))
    return seen


class CaseAnswerGenerator:
    """Turns reranked cases into an answer that cites them.

    Without an API key the object is still usable: ``generate`` returns an
    extractive summary of the real abstracts rather than a simulated model
    answer, so the pipeline stays runnable and stays honest about what produced
    the text.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_context_chars: int = 12000,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or ""
        self.max_context_chars = max_context_chars
        self.temperature = temperature

    @property
    def available(self) -> bool:
        """True when a key is configured, i.e. when the model path can run."""
        return bool(self.api_key)

    # -- prompt ---------------------------------------------------------

    def build_prompt(self, query: str, cases: Sequence[Any]) -> tuple[str, int]:
        """User message for ``cases``, plus how many cases were dropped."""
        blocks = [_case_block(i + 1, sc) for i, sc in enumerate(cases)]
        kept, dropped = _fit_to_budget(blocks, self.max_context_chars)
        if dropped:
            logger.info(
                "dropped %d lowest-ranked case(s) to fit %d chars",
                dropped,
                self.max_context_chars,
            )
        body = "\n\n".join(kept)
        prompt = (
            f"Clinical question:\n{query.strip()}\n\n"
            f"Retrieved case reports ({len(kept)}):\n\n{body}\n\n"
            "Answer the question using only these cases, citing PMIDs inline."
        )
        return prompt, dropped

    # -- generation -----------------------------------------------------

    def generate(
        self, query: str, scored_docs: Sequence[Any], max_cases: int = 5
    ) -> AnswerResult:
        """Answer ``query`` from the top ``max_cases`` reranked documents.

        ``scored_docs`` is expected best-first, as ``CrossEncoderReranker.rerank``
        returns it. Raises ``GenerationError`` if the model call fails; callers
        that would rather degrade than fail can catch it and call
        ``extractive_answer``.
        """
        if max_cases < 0:
            # [:-1] would quietly drop the last case instead of failing.
            raise ValueError(f"max_cases must be >= 0, got {max_cases}")
        cases = list(scored_docs)[:max_cases]
        if not cases:
            return AnswerResult(
                text=(
                    "Retrieval returned no cases for this question, so there is "
                    "nothing to synthesise. No answer is given."
                ),
                citations=[],
                model="none",
                grounded=True,
                prompt_tokens_estimate=0,
                used_fallback=True,
            )

        if not self.available:
            logger.info("no OPENAI_API_KEY configured; using extractive fallback")
            return self.extractive_answer(query, cases)

        prompt, _ = self.build_prompt(query, cases)
        text = _chat_completion(
            api_key=self.api_key,
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )

        supplied = {_pmid_of(sc.document) for sc in cases} - {""}
        cited = _cited_pmids(text)
        unknown = [p for p in cited if p not in supplied]
        if not cited:
            # grounded stays True -- no cited PMID is out of context -- but the
            # flag is vacuous here, and that has to be visible to the caller.
            logger.warning("model answer cites no PMIDs; nothing to trace")
        if unknown:
            # A PMID that was never in the prompt came from model memory.
            logger.warning("answer cites PMIDs not in context: %s", ", ".join(unknown))
        return AnswerResult(
            text=text,
            citations=[p for p in cited if p in supplied],
            model=self.model,
            grounded=not unknown,
            prompt_tokens_estimate=self._estimate_tokens(SYSTEM_PROMPT + prompt),
            used_fallback=False,
        )

    def extractive_answer(self, query: str, cases: Sequence[Any]) -> AnswerResult:
        """Quote the retrieved abstracts instead of writing about them.

        Sentences are copied verbatim and picked by word overlap with the query,
        so nothing here is generated: it is retrieval output reformatted, which
        is the only thing that can be said truthfully with no model in the loop.
        """
        blocks = [_case_block(i + 1, sc) for i, sc in enumerate(cases)]
        kept, dropped = _fit_to_budget(blocks, self.max_context_chars)
        used = list(cases)[: len(kept)]

        keys = _keywords(query)
        parts = [
            "No language model was called (no API key configured). Below are "
            "sentences copied verbatim from the retrieved case reports, ranked by "
            "the cross-encoder and selected by overlap with the question. Nothing "
            "is synthesised, and no clinical claim is made beyond what each "
            "abstract states.",
            "",
            f"Question: {query.strip()}",
            "",
        ]
        citations: list[str] = []
        for i, scored in enumerate(used, start=1):
            doc = scored.document
            pmid = _pmid_of(doc)
            if pmid:
                citations.append(pmid)
            title = _title_of(doc) or "(no title)"
            parts.append(f"[{i}] {title} -- {_citation_of(doc)}")
            for sentence in self._salient_sentences(_body_of(scored), keys):
                parts.append(f'    "{sentence}" (PMID {pmid})' if pmid else f'    "{sentence}"')
            parts.append("")

        if dropped:
            parts.append(
                f"{dropped} lower-ranked case(s) omitted to stay within "
                f"{self.max_context_chars} characters of context."
            )
        parts.append(
            "These are case reports: they show that something has been observed, "
            "not how often. Review by a qualified clinician is required; this is "
            "not a diagnosis or a recommendation for any real patient."
        )
        text = "\n".join(parts).strip()
        return AnswerResult(
            text=text,
            citations=citations,
            model=FALLBACK_MODEL,
            grounded=True,
            prompt_tokens_estimate=self._estimate_tokens(text),
            used_fallback=True,
        )

    @staticmethod
    def _salient_sentences(text: str, keys: set[str], limit: int = 2) -> list[str]:
        """Highest query-overlap sentences, kept in their original order."""
        sentences = _sentences(text)
        if not sentences:
            return []
        scored = sorted(
            enumerate(sentences),
            key=lambda pair: (-len(keys & _keywords(pair[1])), pair[0]),
        )
        chosen = sorted(idx for idx, _ in scored[:limit])
        return [sentences[i] for i in chosen]

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough chars/4 heuristic -- an estimate, not a tokenizer count."""
        return len(text) // 4
