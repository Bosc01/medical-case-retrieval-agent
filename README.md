# Medical Case Retrieval Agent

[![CI](https://github.com/Bosc01/medical-case-retrieval-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Bosc01/medical-case-retrieval-agent/actions/workflows/ci.yml)

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

Seven results, including the three that went against the change:

1. Reranking improves the top five under a broad relevance definition. P@5 rises
   9.2%, from 0.5417 to 0.5917, and survives Benjamini-Hochberg correction
   across 14 metrics at q=0.0257.
2. The gain is bounded by how common the diagnosis is. Sweeping the relevance
   definition by descriptor rarity, the effect is negative below a document
   frequency of about 20 and grows monotonically above it. The reranker helps on
   common presentations and does not help on rare ones.
3. A general-domain reranker gains nothing on the same setup, moving P@5 by
   exactly 0.0000 and surviving 0 of 14 corrected tests. The biomedical model is
   doing the work, not reranking as a technique.
4. Retrieving deeper does not help on any relevance definition, despite
   recall@50 of 0.5432 against a 0.9709 ceiling suggesting it should. Depth 100
   leaves P@5 unchanged and makes P@10 significantly worse on all three
   definitions.
5. Reranking cost is a step function of batch count, not linear in candidates.
   The top-30 budget is the largest that still fits one batch, and it matches
   full-depth P@5 exactly.
6. fp16 weights plus that budget run 4.70 times faster than the reference for a
   P@5 delta of exactly 0.0000, at the cost of 2.6% MAP.
7. Query style matters. Keyword-style queries gain 12.4% P@5 against 9.2% for
   narrative case presentations, and the cross-encoder can collapse to a score
   spread near zero on some narrative queries, where it degrades a correct
   ranking. The reranker now falls back to embedding order when that happens,
   which changes none of the measured results.

Reranking costs 290 ms per query on the fast path, against 17.1 ms for the
retrieval it corrects. Every number here is reproducible from the scripts in
this repository, and the JSON each one wrote is committed alongside it.

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

`requirements.txt` carries lower bounds so the package installs on 3.11 and
3.12. To reproduce the measured numbers exactly, install
`requirements-lock.txt` instead, which pins the versions every figure in this
README was produced under.

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

The p-values in this table are uncorrected. Testing many metrics at 0.05 finds
something on noise alone, so the correction is done properly under "After
multiple-comparison correction" below, where P@5 survives Benjamini-Hochberg.

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

### After multiple-comparison correction

Fourteen correlated metrics tested at 0.05 will turn up something on noise alone
about half the time, so the p-values above need correcting. `judged@k` is
excluded from the family because every retrieved document here is judged, making
it numerically identical to `P@k`; counting both would pad the family with an
exact duplicate.

Benjamini-Hochberg controls the expected proportion of reported findings that
are false, which is the right control for a table of metrics that move together.
Bonferroni controls the chance of any false positive at all and is shown for
reference.

| metric | delta | p | BH q | survives BH | survives Bonferroni |
|---|---|---|---|---|---|
| P@10 | +0.0567 | 0.0001 | 0.0014 | yes | yes |
| P@5 | +0.0500 | 0.0044 | 0.0257 | yes | no |
| nDCG@10 | +0.0447 | 0.0055 | 0.0257 | yes | no |
| MAP | +0.0215 | 0.0188 | 0.0658 | no | no |
| nDCG@5 | +0.0364 | 0.0595 | 0.1388 | no | no |

Three of fourteen survive, and the headline P@5 result is one of them at
q=0.0257. It does not survive Bonferroni, which is the stricter and less
appropriate test here. The ms-marco control survives nothing: zero of fourteen,
with its best q-value at 0.6780.

So the claim holds. Reranking improves P@5 by 9.2% under this relevance
definition, at a false discovery rate of 0.05 across the whole metric table.

### Which queries it helps

Reporting a gain on one relevance definition and no gain on another does not
say anything actionable. Sweeping the definition between them does.

The knob is document frequency: how many cases in the corpus carry the shared
descriptor. Low df is a rare, specific diagnosis. High df is a common one.
Relevance requires the descriptor to be a major topic of both cases throughout,
so only the rarity threshold changes.

| max df | queries | relevant/query | baseline | reranked | delta | p |
|---|---|---|---|---|---|---|
| 10 | 63 | 6.2 | 0.2222 | 0.2000 | -0.0222 | 0.2749 |
| 15 | 76 | 9.3 | 0.2316 | 0.2211 | -0.0105 | 0.6051 |
| 20 | 82 | 11.5 | 0.2634 | 0.2732 | +0.0098 | 0.6518 |
| 25 | 90 | 14.7 | 0.3244 | 0.3356 | +0.0111 | 0.6019 |
| 30 | 94 | 16.3 | 0.3617 | 0.3745 | +0.0128 | 0.5393 |
| 45 | 120 | 27.1 | 0.5150 | 0.5633 | +0.0483 | 0.0068 |

The boundary sits near df 20. Below it the reranker is neutral to slightly
negative; above it the gain grows with the commonness of the shared concept.
Only the loosest definition reaches significance after correction (q=0.0408),
and it is the only one that does.

The delta increases monotonically across all six thresholds. That is suggestive
rather than a formal result, because the definitions are nested and therefore
share most of their queries, so adjacent rows are not independent tests.

The mechanism this points to is worth stating. MedCPT was trained on PubMed
search logs, which are dominated by common clinical queries. The pattern here is
consistent with the reranker being strong inside that distribution and no better
than the bi-encoder outside it. For a rare-disease case-matching tool, which is
arguably the most valuable version of this product, the reranker as configured
does not help.

### Reranking fewer candidates

Cost is per batch, not per candidate. Every sequence in a batch is padded to the
longest one and the batch goes through the model together, so at batch size 32
a budget of 12 and a budget of 30 cost exactly one forward pass. Only a budget
that removes a batch saves anything.

| budget | batches | P@5 | vs full | p | latency | saving |
|---|---|---|---|---|---|---|
| top-10 | 1 | 0.5750 | -0.0167 | 0.2854 | 1136 ms | 2.0x |
| top-20 | 1 | 0.5900 | -0.0017 | 0.9406 | 1136 ms | 2.0x |
| top-30 | 1 | 0.5917 | +0.0000 | 1.0000 | 1136 ms | 2.0x |
| top-50 | 2 | 0.5917 | - | - | 2272 ms | 1.0x |

Reranking the top 30 rather than all 50 halves the cost for a P@5 delta of
exactly 0.0000. Cutting further saves nothing at this batch size and starts
costing quality, so 30 is the right budget: it is the largest one that still
fits in a single batch.

An earlier version of this analysis modelled latency as linear in candidates and
reported a 2.5x saving at top-20. That was wrong. Padding makes cost a step
function of batch count, and the corrected model is in the table above.


### Query style, and a failure case

MedCPT was trained on PubMed search logs, which are short keyword queries. The
eval here uses narrative case presentations, which are not that. Running both
styles over the same cases, candidates and labels, with titles standing in for
keyword-style queries:

| style | P@5 | delta | p | nDCG@5 | p |
|---|---|---|---|---|---|
| narrative presentation | 0.5417 to 0.5917 | +9.2% | 0.0044 | +6.5% | 0.0604 |
| keyword title | 0.5633 to 0.6333 | +12.4% | 0.0004 | +13.8% | 0.0003 |

Keyword-style queries get more out of the reranker. The P@5 gain is larger, and
nDCG@5 moves from not significant to clearly significant. Query formulation is
therefore a real lever here, and the headline number was measured on the harder
of the two styles.

There is a failure mode underneath this that is worth knowing about. For a
textbook endocarditis vignette, "Middle-aged man with fever, new murmur and
splinter haemorrhages after a dental procedure", the cross-encoder scored all 30
candidates between -15.35 and -16.01, a spread of 0.66. With no discrimination
to offer it reordered essentially at random, and it demoted the endocarditis
cases that FAISS had correctly ranked 1, 4, 6, 7 and 9. The same target under
the keyword query "infective endocarditis" separated relevant from irrelevant by
22.5.

That query is not representative. The median spread across all 120 narrative
queries is 16.07, so the reranker usually does discriminate, and an early
reading that it was operating near its noise floor throughout was wrong. But the
failure is real, it is silent, and it is the kind that degrades a correct
ranking rather than merely failing to improve it. A production system should
detect a collapsed score spread and fall back to embedding order rather than
trusting the reordering.


### The fast path

Two changes compose. Reranking the top 30 rather than all 50 fits the work into
one batch, and fp16 weights halve the memory traffic per pass. Measured over all
120 queries, end to end:

| configuration | latency | P@5 | P@10 | nDCG@5 | MAP |
|---|---|---|---|---|---|
| fp32, top-50 | 1361 ms | 0.5917 | 0.5442 | 0.5975 | 0.3623 |
| fp16, top-30 | 290 ms | 0.5917 | 0.5475 | 0.5997 | 0.3530 |

4.70 times faster. P@5 is unchanged to four decimal places, a delta of exactly
0.0000 at p=1.0000, and P@10 and nDCG@5 both drift very slightly upward.

MAP drops 2.6%, and that one is significant at p=0.0081. It is the deliberate
cost of the budget: documents at ranks 31 to 50 keep their embedding order
rather than being rescored, and MAP is sensitive to the whole ranking while the
pipeline only shows five. If a use case needs the full ranking ordered well,
raise the budget to 50 and keep fp16, which is still roughly 2.4 times faster
than the reference.

fp16 changes individual scores by up to 0.023, which is enough to permute
documents deep in the ranking but never reordered the top five in testing. The
verification is the table above rather than that observation.


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

The obvious objection is that this was measured on the loose relevance
definition, where recall is plentiful and extra candidates cannot help much.
Deeper retrieval ought to pay off precisely where relevant cases are scarce. So
`scripts/depth100_eval.py` re-runs depth 50 against depth 100 on all three
definitions, replaying one score dump so the arms are identical:

| definition | P@5 at depth 50 | P@5 at depth 100 | delta | p | P@10 delta | p |
|---|---|---|---|---|---|---|
| strict (df<=15) | 0.2211 | 0.2211 | +0.0000 | 1.0000 | -0.0197 | 0.0077 |
| middle (df<=25) | 0.3356 | 0.3400 | +0.0044 | 0.6285 | -0.0189 | 0.0039 |
| broad (df<=45) | 0.5633 | 0.5717 | +0.0083 | 0.3192 | -0.0150 | 0.0285 |

It does not pay off anywhere. P@5 is flat on every definition, and under the
strict one it is identical to four decimal places: the extra fifty candidates
contributed nothing at all to the top five. P@10 is significantly worse on all
three, which is the clearer signal. Candidates from ranks 51 to 100 are
occasionally scored highly by the cross-encoder and displace better cases,
because its scores are not calibrated well enough across a deeper pool for the
extra reach to be free.

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
and the result is reported above under "The gain does not survive a stricter
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
