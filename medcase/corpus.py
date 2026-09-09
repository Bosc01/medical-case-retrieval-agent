"""PubMed case-report corpus: fetch, parse, cache.

Uses NCBI E-utilities, which is public and needs no key at low volume. NCBI
asks for <=3 requests/second unencrypted, so requests are paced. Set
``NCBI_API_KEY`` to raise that to 10/s.

Every record keeps its PMID, so any claim the agent makes downstream can be
traced back to a real article rather than to model memory.
"""

from __future__ import annotations

import json
import logging
import os
import ssl
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_TOOL = "medcase-retrieval-agent"


@dataclass
class CaseRecord:
    """One PubMed case report."""

    pmid: str
    title: str
    abstract: str
    journal: str = ""
    year: str = ""
    mesh_terms: list[str] | None = None
    mesh_major: list[str] | None = None  # descriptors NLM flagged MajorTopicYN=Y

    @property
    def text(self) -> str:
        """Title + abstract, the field the encoders and reranker score."""
        return f"{self.title}\n\n{self.abstract}".strip()

    @property
    def url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"

    def citation(self) -> str:
        bits = [b for b in (self.journal, self.year) if b]
        suffix = f" ({', '.join(bits)})" if bits else ""
        return f"PMID {self.pmid}{suffix}"


def _ssl_context() -> ssl.SSLContext:
    """Trust store that survives a TLS-intercepting proxy.

    Python verifies against certifi, which does not know a corporate proxy's
    self-signed root even when the OS keychain trusts it -- so curl succeeds
    while urllib raises CERTIFICATE_VERIFY_FAILED. Point MEDCASE_CA_BUNDLE (or
    SSL_CERT_FILE) at a PEM that includes that root; scripts/export_ca_roots.sh
    writes one from the macOS keychain. Verification stays ON either way.
    """
    bundle = os.getenv("MEDCASE_CA_BUNDLE") or os.getenv("SSL_CERT_FILE")
    if not bundle:
        local = Path(__file__).resolve().parents[1] / "data" / "ca-roots.pem"
        if local.exists():
            bundle = str(local)
    if bundle and Path(bundle).exists():
        return ssl.create_default_context(cafile=bundle)
    return ssl.create_default_context()


def _get(url: str, *, retries: int = 3) -> bytes:
    key = os.getenv("NCBI_API_KEY")
    if key:
        url = f"{url}&api_key={urllib.parse.quote(key)}"
    delay = 0.34 if not key else 0.11  # respect NCBI rate limits
    last: Exception | None = None
    for attempt in range(retries):
        try:
            time.sleep(delay)
            req = urllib.request.Request(url, headers={"User-Agent": _TOOL})
            with urllib.request.urlopen(req, timeout=30, context=_ssl_context()) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 - network flakiness is expected
            last = exc
            logger.warning("E-utilities attempt %d failed: %s", attempt + 1, exc)
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"E-utilities request failed after {retries} attempts: {last}")


def search_pmids(term: str, retmax: int = 50) -> list[str]:
    """PMIDs for a query, newest first."""
    url = (
        f"{EUTILS}/esearch.fcgi?db=pubmed&term={urllib.parse.quote(term)}"
        f"&retmax={retmax}&retmode=json&tool={_TOOL}"
    )
    payload = json.loads(_get(url))
    return payload.get("esearchresult", {}).get("idlist", [])


def _text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return " ".join("".join(node.itertext()).split())


def fetch_records(pmids: list[str], *, chunk: int = 100) -> list[CaseRecord]:
    """Full records for PMIDs. Articles with no abstract are dropped -- a
    title alone is too thin for a cross-encoder to judge."""
    out: list[CaseRecord] = []
    for start in range(0, len(pmids), chunk):
        batch = pmids[start : start + chunk]
        url = (
            f"{EUTILS}/efetch.fcgi?db=pubmed&id={','.join(batch)}"
            f"&retmode=xml&tool={_TOOL}"
        )
        root = ET.fromstring(_get(url))
        for art in root.findall(".//PubmedArticle"):
            pmid = _text(art.find(".//PMID"))
            title = _text(art.find(".//ArticleTitle"))
            # Structured abstracts split across labelled sections.
            parts = []
            for seg in art.findall(".//Abstract/AbstractText"):
                label = seg.get("Label")
                body = _text(seg)
                if not body:
                    continue
                parts.append(f"{label}: {body}" if label else body)
            abstract = " ".join(parts)
            if not pmid or not abstract:
                continue
            out.append(
                CaseRecord(
                    pmid=pmid,
                    title=title,
                    abstract=abstract,
                    journal=_text(art.find(".//Journal/ISOAbbreviation")),
                    year=_text(art.find(".//JournalIssue/PubDate/Year")),
                    mesh_terms=[
                        _text(m) for m in art.findall(".//MeshHeading/DescriptorName")
                    ],
                    # Major topics carry NLM's judgement that the article is
                    # substantively about the concept, not merely mentions it.
                    # That distinction is what makes graded relevance possible.
                    mesh_major=[
                        _text(m)
                        for m in art.findall(".//MeshHeading/DescriptorName")
                        if m.get("MajorTopicYN") == "Y"
                    ],
                )
            )
    return out


def build_corpus(
    topics: list[str],
    per_topic: int = 25,
    *,
    medline_only: bool = True,
) -> list[CaseRecord]:
    """Fetch case reports across topics, de-duplicated by PMID.

    ``medline_only`` restricts to MEDLINE-indexed citations. NLM assigns MeSH
    months to years after publication, so the newest case reports carry no
    descriptors at all -- and MeSH is what the eval harness uses for relevance
    labels. Without this filter most of the corpus is unjudgeable.
    """
    seen: dict[str, CaseRecord] = {}
    for topic in topics:
        term = f'({topic}) AND "case reports"[Publication Type]'
        if medline_only:
            term += " AND medline[sb]"
        pmids = search_pmids(term, retmax=per_topic)
        logger.info("%s -> %d pmids", topic, len(pmids))
        for rec in fetch_records(pmids):
            seen.setdefault(rec.pmid, rec)
    return list(seen.values())


def save_corpus(records: list[CaseRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(asdict(rec)) + "\n")


def load_corpus(path: str | Path) -> list[CaseRecord]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No corpus at {path}. Run scripts/build_index.py first."
        )
    with path.open(encoding="utf-8") as fh:
        return [CaseRecord(**json.loads(line)) for line in fh if line.strip()]
