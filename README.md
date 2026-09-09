# Medical Case Retrieval Agent

Retrieval over PubMed case reports with a cross-encoder reranker between the
vector search and the language model.

## The change

The original design was a single retrieval stage: embed the query, search FAISS,
hand the top hits to GPT-4. The problem with that shape is what a bi-encoder
throws away. It compresses a case report into one 768-dimensional vector before
it has seen the query, so the query and the document are never compared
directly. MedCPT's query encoder also truncates at 64 tokens, and a clinical
presentation is longer than 64 tokens, so most of the question is discarded
before the search runs.

A cross-encoder does not have that limitation. It reads the query and one
candidate together, in a single sequence of up to 512 tokens, with full
attention across the pair. It can therefore use the detail the first stage
dropped. The cost is one forward pass per candidate, which is why it cannot run
over a corpus and has to sit behind a cheaper stage.

So the pipeline is now two stages:

```
query
  -> MedCPT bi-encoder  ->  FAISS IndexFlatIP  ->  top 50 candidates
  -> MedCPT cross-encoder scores each (query, case) pair
  -> reorder, keep top 5
  -> generation, grounded and cited
```

Stage one only has to get the right case somewhere into the top 50. Stage two
decides which 5 reach the model. Recall at 50 is the ceiling on the whole
system, because reranking cannot surface a case that FAISS never returned.

## Findings

Four results, including the two that went against the change:

1. Reranking improves the top five under a broad relevance definition. P@5 rises
   9.2%, from 0.5417 to 0.5917, at p=0.0044 on 120 queries.
2. That gain does not survive a stricter relevance definition. Under rare-
   descriptor, major-topic-in-both labels, P@5 goes from 0.2316 to 0.2211 with
   p=0.61. No detectable effect.
3. A general-domain reranker gains nothing on the same setup, moving P@5 by
   exactly 0.0000. The biomedical model is doing the work, not reranking as a
   technique.
4. Retrieving deeper does not help, despite recall@50 of 0.5432 against a 0.9709
   ceiling suggesting it should. Depth 200 costs 4.8 times the latency and makes
   P@10 significantly worse.

Reranking costs 2272 ms per query against 17.1 ms for the retrieval it corrects.
Every number here is reproducible from the scripts in this repository, and the
JSON each one wrote is committed alongside it.


## Models

| Role | Model | Why |
|---|---|---|
| Query encoder | `ncbi/MedCPT-Query-Encoder` | Trained on 255M PubMed search logs |
| Article encoder | `ncbi/MedCPT-Article-Encoder` | Asymmetric partner to the query encoder |
| Reranker | `ncbi/MedCPT-Cross-Encoder` | Domain matched to a PubMed corpus |
| Control | `cross-encoder/ms-marco-MiniLM-L6-v2` | General domain baseline, run as a check |

The ms-marco arm exists to keep the biomedical claim honest. If a general
purpose reranker matches the biomedical one, then the domain match is not what
is doing the work.

## Layout

```
medcase/
  corpus.py      PubMed E-utilities fetch, parse, cache
  embed.py       MedCPT bi-encoders, CLS pooling, dot product space
  index.py       FAISS IndexFlatIP, owns vectors and records together
  reranker.py    cross-encoder scoring, batching, caching, telemetry
  pipeline.py    retrieve, rerank, generate, with a per-query trace
  generate.py    grounded answer synthesis, GPT-4 or extractive fallback
  evaluate.py    Precision, Recall, MRR, nDCG, significance testing
  finetune.py    hard negative mining and cross-encoder training
scripts/
  fetch_corpus.py    build the corpus from PubMed
  build_index.py     embed and write the FAISS index
  build_evalset.py   derive queries and relevance labels
  run_eval.py        baseline against reranked, same candidates
  ask.py             ask a question, see what reranking moved
```

## Setup

```bash
python3.12 -m venv venv
./venv/bin/pip install -r requirements.txt
```

Two environment notes, both specific to this machine rather than to the code:

TLS interception. Python verifies against certifi, which does not know a
corporate proxy root even when the operating system trusts it, so `curl`
succeeds while `urllib` raises `CERTIFICATE_VERIFY_FAILED`. Run
`scripts/export_ca_roots.sh` to write the macOS trust store to
`data/ca-roots.pem`, which `corpus.py` picks up automatically. Verification
stays on.

OpenMP. `torch` and `faiss-cpu` each ship a copy of `libomp.dylib`, and loading
both aborts the process the first time FAISS runs a search alongside a torch
model. Both are LLVM builds with the same ABI, so point one at the other:

```bash
ln -sf "$PWD/venv/lib/python3.12/site-packages/torch/lib/libomp.dylib" \
       venv/lib/python3.12/site-packages/faiss/.dylibs/libomp.dylib
```

## Run

```bash
./venv/bin/python scripts/fetch_corpus.py
./venv/bin/python scripts/build_index.py
./venv/bin/python scripts/build_evalset.py
./venv/bin/python scripts/run_eval.py
```

Ask a question and see what the reranker moved:

```bash
./venv/bin/python scripts/ask.py "progressive proximal weakness with ptosis worsening through the day"
```

## How the evaluation is built

The weak point in any reranking claim is the relevance labels. If the labels
come from the same embedding model being evaluated, the result is circular.

So no labels are authored here, and none come from a model. Each query is a real
case report's own presentation text, taken from the abstract. Two cases count as
relevant when NLM's human indexers assigned them the same specific MeSH
descriptor. Relevance is graded: 2 when the shared descriptor is a major topic of
both cases, 1 when it is shared but minor for either.

Descriptors are bounded at both ends. One held by half the corpus says nothing
about relevance, and one held by a single case cannot be retrieved at all, so
only descriptors covering 4 to 45 records are used. The query case is removed
from its own result list, since otherwise it sits at rank 1 as a non-relevant
self match and distorts every metric.

Both arms receive the same 50 FAISS candidates and differ only in ordering. Any
delta is therefore reordering, not better recall.

Queries with no relevant documents are excluded from the mean and counted
separately, never scored as zero. Unjudged documents mean unknown, not
irrelevant, which matters here: scoring unjudged as zero biases against whichever
system surfaces documents the other never did, and that is precisely the
reranker.

## Results

Measured on 120 queries over 764 case reports, retrieval depth 50, on an M-series
laptop using MPS. Every number below comes from `data/eval_results.json`, written
by `scripts/run_eval.py`. Reproduce with `./venv/bin/python scripts/run_eval.py`.

| metric | FAISS only | reranked | delta | p (sign) | p (bootstrap) |
|---|---|---|---|---|---|
| P@5 | 0.5417 | 0.5917 | +9.2% | 0.0119 | 0.0044 |
| P@10 | 0.4875 | 0.5442 | +11.6% | 0.0344 | <0.0001 |
| MAP | 0.3408 | 0.3623 | +6.3% | 0.0011 | 0.0188 |
| nDCG@5 | 0.5610 | 0.5975 | +6.5% | 0.0693 | 0.0595 |
| P@1 | 0.6750 | 0.6583 | -2.5% | 0.8450 | 0.7632 |
| MRR | 0.7361 | 0.7338 | -0.3% | 0.7660 | 0.9444 |

P@5 is the primary metric, chosen because the pipeline hands exactly five cases
to the generator. It improves by 9.2%, winning on 39 queries and losing on 19.

Four things in that table deserve to be read carefully.

The gain is in the body of the ranking, not at the top. P@1 and MRR do not move,
and their p-values are around 0.8, so the small negative numbers are noise rather
than a regression: 94 of 120 queries have the same document at rank 1 under both
systems. The cross-encoder is not finding a better single best case. It is
clearing out the weak cases sitting behind it.

nDCG@5 is not significant at 0.05 by either test. The precision gains are real
and the graded-gain version of the same measurement is not yet distinguishable
from noise at this sample size. Both are reported.

The p-values are uncorrected across 18 metrics. Testing that many at 0.05 finds
something roughly a third of the time on noise alone. P@10 survives a Bonferroni
correction. P@5 at 0.0044 does not quite clear the corrected 0.0028 threshold, so
it should be read as suggestive rather than settled.

The control arm earns its place. `ms-marco-MiniLM`, a general-domain reranker,
moves P@5 by exactly 0.0000 and makes MRR worse. Reranking as a technique is not
what produces the gain here. The biomedical model is.

### The gain does not survive a stricter definition

The headline result counts two cases as relevant when they share any specific
MeSH descriptor. That is a broad notion of relevance, roughly 30 documents per
query, and a broad pool inflates precision for every arm. So the same rankings
were re-scored against a harder definition: the shared descriptor must be rare
in the corpus (document frequency 15 or less) and a major topic of both cases.
That is closer to the case-matching a clinician actually wants. No model is
re-run, only the labels change.

Under that definition, 76 of the 120 queries qualify, with 9.3 relevant
documents per query and a random-ranking P@5 of 0.0122.

| metric | FAISS only | reranked | delta | p (sign) | p (bootstrap) |
|---|---|---|---|---|---|
| P@5 | 0.2316 | 0.2211 | -4.5% | 0.7201 | 0.6051 |
| P@1 | 0.3816 | 0.3026 | -20.7% | 0.2632 | 0.2068 |
| MAP | 0.2143 | 0.2039 | -4.9% | 0.4704 | 0.4294 |
| nDCG@5 | 0.2733 | 0.2532 | -7.4% | 0.4709 | 0.3788 |
| MRR | 0.4833 | 0.4131 | -14.5% | 0.1189 | 0.1225 |

Every metric moves slightly negative and not one is significant. The correct
reading is no detectable effect, not that reranking hurts: the sample falls to
76 queries, which costs statistical power, and every p-value is far from the
threshold in both directions.

What this says about the system is specific and worth stating plainly. The
cross-encoder is better at broad topical precision, at clearing loosely related
cases out of the top five. It is not measurably better at the harder task of
finding the case that shares a rare diagnosis as its main subject. Those are
different capabilities, and only the first one is supported by evidence here.

The +9.2% headline is therefore real but definition-dependent, and any use of it
has to carry that condition. A reranker that improves broad topical precision is
still worth having in front of a generator, because the five cases the model
reads are less padded with weak matches. It is not the case-matching improvement
the framing might otherwise suggest.


### Cost

| stage | latency |
|---|---|
| FAISS retrieval | 17.1 ms/query |
| MedCPT reranking, 50 candidates | 2272 ms/query |
| ms-marco reranking, 50 candidates | 425 ms/query |

Reranking is roughly 130 times the cost of the retrieval it corrects, because it
runs one BERT-base forward pass per candidate instead of one vector comparison.
Buying 9.2% P@5 for 2.3 seconds per query is a defensible trade for an offline or
clinician-in-the-loop tool, and a poor one for anything interactive. Batch size,
a shorter `max_length`, a distilled model, or fp16 are the levers if that matters.

An earlier version of these numbers was measured while other processes were
competing for the GPU, and reported 4761 ms/query and 33.6 ms/query for the two
stages. Those figures were wrong. The table above is from an uncontended run.

### Where the headroom actually is

`scripts/recall_sweep.py` measures recall at increasing retrieval depth:

| depth | recall | best achievable at that depth |
|---|---|---|
| 10 | 0.1861 | 0.4567 |
| 20 | 0.3104 | 0.7017 |
| 50 | 0.5432 | 0.9709 |
| 100 | 0.6823 | 1.0000 |
| 200 | 0.8144 | 1.0000 |
| 400 | 0.9246 | 1.0000 |

At depth 50 the bi-encoder returns only 56% of the relevant cases it could have.
Recall is still climbing steeply there and does not flatten until roughly 200, so
the first stage, not the reranker, is the binding constraint on this corpus.

That is a statement about the ceiling, not about the answer, and the difference
turned out to matter. `scripts/depth_experiment.py` scores all 200 candidates
once and re-sorts subsets of them, so each depth is compared on identical
scores:

| retrieval depth | P@5 | P@10 | MAP | rerank latency |
|---|---|---|---|---|
| 50 | 0.5917 | 0.5442 | 0.3623 | 2272 ms |
| 100 | 0.5967 | 0.5275 | 0.4018 | ~5400 ms |
| 200 | 0.5867 | 0.5167 | 0.4003 | 10859 ms |

Going deeper does not improve the top five. Depth 100 moves P@5 by +0.0050
(p=0.80, 9 wins against 7 losses and 104 ties) and depth 200 moves it by -0.0050
(p=0.68). Depth 200 makes P@10 significantly worse, -5.1% at p=0.0057, while
costing 4.8 times the latency.

MAP does improve with depth, +10.5% at depth 200 with p=0.0011, but MAP rewards
finding more relevant documents anywhere in the ranking. This pipeline shows
five. The metric that improves is not the metric the product uses.

So the recall ceiling was a misleading signal, and the hypothesis it suggested
was wrong: raising the ceiling did not raise the answer. Depth 50 is the right
default, and the extra candidates give the cross-encoder more chances to be
fooled at roughly the rate they give it more chances to be right.

Note also that depth 400 is 52% of this 764-document corpus, so the tail of the
recall curve is close to retrieving the whole collection and will not transfer
to a corpus of realistic size.

### Making the reranker cheaper

Every sequence in a batch is padded to the longest one in it, so a batch mixing
a 581-character abstract with a 2,728-character one does much of its work on
padding. Sorting candidates by length before batching narrows that spread.
Scores are written back by original index, so the output order is unchanged.

`scripts/bench_rerank.py` asserts the scores are identical before timing
anything, then measures, on 50 candidates at batch size 32, best of 9 runs:

| variant | best | median | padding waste |
|---|---|---|---|
| batch order | 1855 ms | 1923 ms | 31.4% |
| length-sorted | 1624 ms | 1765 ms | 23.4% |

Scores match exactly, to a maximum difference of 0.00e+00.

The padding reduction is deterministic and substantial, 31.4% down to 23.4%.
The wall-clock gain is modest, about 1.14 times, because these abstracts have a
median length near 450 tokens against a 512-token cap, so most sequences are
already close to full and there is not much padding left to remove. A corpus
with more variable document lengths would gain more.

One process note, since it changes what the number means. An earlier run with
3 repeats reported 1.40 times. That was measurement noise, and 9 repeats put it
at 1.14. The larger figure was not reproducible and is not the result.


## These numbers are not the published numbers

The paper this project came from reports Precision@5 improving from 0.11 to 0.24
across three releases. This README reports Precision@5 of 0.5417 rising to
0.5917. Those two pairs of numbers must not be compared, added, or presented as
one result, and the larger number is not an improvement on the smaller one.

They measure different tasks:

| | published work | this repository |
|---|---|---|
| Corpus | 50+ PubMed records | 764 PubMed case reports |
| Queries | rubric-based harness | 120 held-out case presentations |
| Relevance | rubric judgement | shared specific MeSH descriptor |
| Relevant per query | not comparable | 30.4 on average, 4.0% of the corpus |
| What changed between arms | three releases of the whole system | one component, same candidates |

Precision@5 is not a portable score. It moves with the density of the relevance
pool, so it only means something next to the baseline it was measured against.
On this eval a random ranker scores 0.0398, because roughly 30 of 763 documents
are relevant to any given query. The FAISS baseline of 0.5417 is 13.6 times
random, and the reranked 0.5917 is a 9.2% relative gain over that specific
baseline. A score of 0.24 measured against a stricter rubric can represent a far
harder task than 0.54 measured against MeSH co-indexing.

So the honest framing is two separate results. The published one belongs to the
paper. The one here belongs to this reranker, and it is quoted with its own
baseline attached or not quoted at all.

The looseness of the relevance definition cuts against this repository, not for
it. `scripts/strict_eval.py` re-runs the comparison under a tighter definition,
and the result is reported below under "The gain does not survive a stricter
definition". It does not survive.


## Fine-tuning

`finetune.py` mines hard negatives from the FAISS index rather than sampling
random documents, because the negatives that teach a reranker anything are the
ones the first stage actually confuses. Training uses binary cross-entropy on
the single logit, which matches the MedCPT head.

It is worth being blunt about when this pays off. With a few dozen labelled
queries, an off-the-shelf domain cross-encoder will usually beat a fine-tune,
and fine-tuning on that little data mostly overfits. The script is there for
when real labelled data exists.

## What is verified and what is not

Verified by running it:

- Corpus fetch, index build, retrieval, reranking, and the evaluation harness.
- The extractive fallback path in `generate.py`.
- The metric implementations, pinned in tests against hand-computed values.

Not verified:

- The GPT-4 generation path. No API key was available on this machine, so the
  OpenAI branch of `generate.py` has never executed. It is written but untested.
- Fine-tuning beyond a one-epoch smoke test on synthetic pairs with a tiny
  random model. That confirms the loop runs. It is not evidence that
  fine-tuning improves ranking.

## Safety posture

The generator is instructed to use only the supplied cases, to cite a PMID for
every clinical claim, to say plainly when the retrieved cases do not support an
answer, and not to diagnose or recommend treatment for a real patient. Output is
framed as literature synthesis for review by a qualified clinician.

The `grounded` flag on a result is a provenance check, not an entailment check.
It confirms that every PMID cited was among the cases put into the prompt. It
does not confirm that the text is supported by those cases.

Case reports establish that something has been observed. They never establish
how often.
