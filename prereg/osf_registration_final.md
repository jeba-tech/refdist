# REFDIST: A Causal Reference-Swap Audit of False Positives in AI Text Detectors

**Pre-registration**

Author: Independent researcher (fawjiajeba@gmail.com)
Code: https://github.com/jeba-tech/refdist (the commit tagged `prereg` is the one this document describes)
Date registered: October 4, 2026

---

## Where things stand at the moment of registering

I want to be exact about what has and has not happened, because the value of a pre-registration depends on it.

The corpus is built. Building it meant downloading, cleaning, de-duplicating and length-matching human-written text. No detector was involved, and nothing about the outcomes can be learned from that step.

I have not yet scored a single study document with any detector or with any of the language models used in the analysis. A few models have run on other material, and only to test the code:

- GPT-2 small (124M, not part of the study) on 20 paragraphs of *Pride and Prejudice*, to check the scoring maths.
- The eight language models the study uses (the perplexity model, the five reference models and the two Binoculars instruction-tuned models), on those same 20 paragraphs plus 20 machine-written continuations. This measured whether half precision (fp16) gives the same scores as full precision (fp32) on a free Kaggle GPU. It does: every model agreed to within 2.1% of the score spread, so these eight run in fp16. The three trained classifiers are small, so they simply run in full precision and need no such check.
- Qwen2.5-1.5B-Instruct, which wrote the 500 machine texts for the positive control (Section 5.6). Writing those texts involves no scoring.

Every threshold and confirmation rule below is also written into the code (`refdist/config.py` and `refdist/analysis/hypotheses.py`). The paper will be judged against what is here and in that commit.

---

## 1. What I want to find out

AI text detectors regularly flag human writing as machine-generated. The common explanation is that some people just "write like an AI". I think the actual mechanism is narrower and easier to test. A detector compares text against a reference: for zero-shot detectors that is a scoring language model, and for trained classifiers it is their training data. My claim is that detectors flag text that is *predictable to that reference*, whoever wrote it.

If that is right, two things should follow:

1. If I keep the text fixed and swap the reference model, the list of populations that get wrongly flagged should change, not only the overall rate.
2. If I make small rule-based edits to human text that make it more or less predictable, detector scores should move with the change in predictability. It shouldn't matter which kind of edit caused it.

I test both on 12 collections of human-written English, all written in 2021 or earlier.

---

## 2. Hypotheses

All six are primary. I will report every one of them, whatever the result.

A note on direction, since it matters for every test below. My predictor is **log perplexity**, the average negative log-likelihood per token. Low perplexity means predictable text. My claim is that predictable text gets flagged more, so **I expect every perplexity coefficient to be negative.**

### H1: One perplexity score explains false positives across populations and detectors

I compute perplexity with SmolLM2-360M, a model that none of the detectors use. That choice is deliberate. If I used a detector's own model, a correlation between its perplexity and its scores would be built in, and the test would mean nothing. The code refuses to run if SmolLM2-360M ever appears in a detector.

I test this in two ways:

- **Across cells** (one detector on one population), I fit `FPR ~ log PPL_indep + length + (1 | instance)` as a linear mixed model and measure how much the marginal R² rises when perplexity is added (ΔR²).
- **Across documents**, I fit a logistic regression, `flagged ~ log PPL_indep + length + C(instance)`, with standard errors clustered by document. I fit it once without and once with population fixed effects. If the perplexity coefficient survives adding the populations, perplexity predicts flags within a population and not just between them. I measure that as attenuation = 1 − (coefficient with populations / coefficient without).

**Confirmed** if the cell-level coefficient is negative with p < 0.05, ΔR² is at least 0.10, and attenuation is below 30%.
**Rejected** if the coefficient is not significantly negative, or ΔR² is below 0.05. Anything in between I will report as partial.

### H2: Changing the reference changes who gets flagged (the central test)

This uses Fast-DetectGPT with all five reference models.

- I fit `FPR(r, k) ~ log PPL_r(k) + length + C(stratum) + C(reference)` with HC3 standard errors. Because populations and references both get fixed effects, the only thing left to explain is how unusually predictable a given reference finds a given population. That is exactly the effect I'm claiming.
- I also run a permutation test on whether the pattern of flag rates across populations differs between references. I shuffle the reference labels within each document 2,000 times and compare the population × reference interaction with that null.
- I report Kendall's τ between the population rankings of each pair of references, as a description only.

GPT-Neo-1.3B (R2) and Pythia-1.4B (R3) were both trained on The Pile, so any difference between them can't come from different training data. I report that pair separately with the same tests.

**Confirmed** if the coefficient is negative with p < 0.05 and the permutation test gives p < 0.05. **Rejected** if the permutation test gives p ≥ 0.05.

As a robustness check, I add an indicator for whether a population's source appears in the reference model's published training data (Section 6).

### H3: The same holds for trained classifiers

Trained classifiers (D3–D5) have no scoring language model, so if H1 holds for them too, the result can't be a side effect of how zero-shot scores are computed. I repeat H1 on them alone. I also run the H2 permutation test across the three classifiers, since each was trained on a different corpus. I'll report H3 in its own section whichever way it comes out.

### H4: Perplexity can forecast a detector's false-positive rate on a new population

I leave each population out in turn, fit `FPR ~ log PPL_indep + length + C(instance)` on the other 11, and predict the one left out.

Low error by itself would prove little. If false-positive rates barely varied, any model would look accurate. So I compare against a baseline that knows each detector's average rate but nothing about perplexity, `FPR ~ length + C(instance)`, and compute skill = 1 − (MAE of my model / MAE of the baseline).

**Confirmed** if MAE is at most 0.10 (ten percentage points) and skill is at least 0.25. **Rejected** if MAE is above 0.15 or skill is zero or below.

### H5: Editing the text moves the scores in step with perplexity

I take forum answers from the calibration population (S6) and apply six kinds of rule-based edit (Section 5.5) at three strengths. At strength *p*, a document with *n* places where an edit could go gets the first ⌈p·n⌉ of them, in a fixed random order for that document. That way the lighter edits are always a subset of the heavier ones.

Some edits apply to almost every answer and some to only a minority. For each edit type I therefore use up to 300 S6 answers that have at least one place for it, chosen with a fixed seed, or all of them if fewer qualify.

For each detector I fit `Δz ~ Δlog PPL_indep + (1 + Δlog PPL_indep | transform)`. Here Δz is the change in the detector's score against the unedited document, divided by the detector's score spread on S6.

**Confirmed for a detector** if the overall slope is negative with p < 0.05, and the spread (SD) of the six edit-specific slopes is no more than a quarter of the overall slope. Because this test is repeated for every detector, the slope p-values are Holm-corrected across detectors before counting. **H5 as a whole is confirmed** if this holds for at least half of the detectors that pass the positive control.

No language model rewrites anything. Every single edit is recorded and can be undone.

### H6: Detectors with the same reference agree with each other

For every pair of detectors I compute Cohen's κ on which documents they flag. Two detectors count as sharing a reference when their scoring models have the same pretraining corpus (R2 and R3 share The Pile), or, for trained classifiers, the same training corpus. The test statistic is mean κ for same-reference pairs minus mean κ for different-reference pairs. I compare it against 5,000 shuffles of the reference labels.

**Confirmed** if the gap is positive with p < 0.05.

---

## 3. A forecast I commit to before seeing the data

Before any detector touches the last three populations (S10 TwitterAAE, S11 legal opinions, S12 meeting transcripts), I fit `FPR ~ log PPL_indep + length + C(instance)` on S1–S9 and predict every detector's false-positive rate on S10–S12 from perplexity alone.

The predictions go into `prereg/forecast_S10_S12.json`. I will post the file's SHA-256 hash as a public comment on this registration *before* scoring S10–S12. The code enforces this: it won't run a detector on those three populations until the forecast file exists. Only the perplexity model may read them first, because its scores are the forecast's input. When I reveal the results, the code first checks the hash, then compares predictions with what actually happened. I'll report the error however large it turns out to be.

---

## 4. Design

The study is observational, with two interventions: swapping the reference model while keeping the text fixed (H2), and editing the text while keeping the author fixed (H5).

The only machine-written text anywhere in the study is the positive control. It's stored separately and never enters a hypothesis test.

---

## 5. Materials

### 5.1 The 12 populations

Each population has 1,000 documents of 150–400 tokens, all written in 2021 or earlier. I measure length with the GPT-2 tokenizer for every population, so "150–400 tokens" means the same thing everywhere. Documents are sampled with a fixed seed and spread evenly across five length bands. When a source has too few documents in a band, the nearest bands make up the difference, and the build log records how many.

| ID | Population | Source and selection |
|----|------------|----------------------|
| S1 | Literary prose, 1850–1899 | Project Gutenberg, English/American literature (LoCC PR/PS), single-author books, author died 1855–1899 |
| S2 | Literary prose, 1900–1921 | Same, author born 1865–1895 |
| S3 | Scientific abstracts, 1995–2005 | arXiv, by submission date, two random weeks per archive per year; equal quotas for astro-ph, cond-mat, hep-th, math, quant-ph and cs |
| S4 | Scientific abstracts, 2015–2021 | Same archives and quotas as S3 |
| S5 | Forum answers, 2011–2015 | Stack Exchange: English Language & Usage, Travel, Cooking |
| S6 | Forum answers, 2018–2021 | Same sites (the calibration population) |
| S7 | Learner essays | PELIC, final draft of each student's response to each prompt, non-English first language |
| S8 | Learner essays | ICNALE Written Essays, native-speaker group excluded |
| S9 | US school essays | ASAP-AES |
| S10 | African American English | TwitterAAE, AA-aligned release (Blodgett et al., 2016) |
| S11 | Legal opinions | US court opinions from CourtListener, via pile-of-law |
| S12 | Spoken transcripts | AMI Meeting Corpus, manual transcripts |

S3/S4 and S5/S6 are pairs from the same source, so comparing them isolates the era. To make sure that era is the *only* difference, the second member of each pair copies its partner's exact counts across length band × field (the arXiv archive, or the Stack Exchange site). Matching on length alone wasn't enough. Older astrophysics abstracts pass the 150-token minimum much more often than old maths or computer-science ones, so without this step the era comparison would also have been a length and field comparison.

How I handled each source:

- **Dates.** Gutenberg doesn't record when a book was written, so I bound it by the author's lifetime; for S2 I use birth year + 35, kept between 1900 and 1921. CourtListener only records when it added an opinion, which can't be earlier than the opinion itself, so I require that date to be 2021 or earlier. For ASAP I use 2012, the competition year, as an upper bound. For ICNALE I use 2013, since Ishikawa (2013) already describes all 5,600 essays in the module.
- **Gutenberg.** Poetry and drama are out: any book whose catalogue subjects, title or bookshelves mention poetry, drama, plays, verse, hymns, songs, ballads or sonnets. I use one random book per author.
- **Stack Exchange.** Answers only. I strip code, preformatted text and block quotes, since quotes are usually someone else's words. Community-wiki posts and anything edited after 2021 are excluded.
- **ASAP.** The anonymisation tags (@PERSON1 and so on) are replaced using a fixed list. Essays with more than five tags are dropped. So are essays containing the transcribers' marks for unreadable handwriting ("???", "illegible", "not legible"). The competition terms don't allow sharing the essays, so only summary statistics from S9 will ever be published.
- **ICNALE.** These essays were written in 20–40 minutes on two set topics, without dictionaries. Each student wrote on both topics, so the per-source cap applies per student.
- **TwitterAAE.** I keep tweets with an AA posterior of at least 0.8 and leave out retweets. URLs are removed and mentions become `@user`. Each document is one user's tweets in posting order, up to the length limit.
- **Legal opinions.** I drop paragraphs with more than two case citations, lists of judges or counsel, and page markers such as "*228" or "{¶ 40}".
- **AMI.** I build each speaker's utterances separately, splitting at pauses longer than 1.5 seconds, then interleave speakers by start time.
- **Everything.** HTML entities are decoded fully, however many times they were escaped. Windows-1252 characters that were decoded wrongly are repaired. Exact and near duplicates are removed (MinHash, Jaccard ≥ 0.8).
- **Caps per source.** At most 3 documents per writer, user or court case. Gutenberg allows 4 per author, because only about 360 prose authors died between 1855 and 1899. AMI allows 8 per meeting, because the corpus has only about 170 meetings.

S6 sets every detector's threshold: each one is calibrated to flag at most 5% of S6 (in practice between 4.7% and 5%, depending on ties among the 1,000 scores). S6 is locked, and the build code won't rebuild it.

### 5.2 Reference models

| ID | Model | Size | Trained on | Year |
|----|-------|------|------------|------|
| R1 | GPT-2 Large | 774M | WebText | 2019 |
| R2 | GPT-Neo-1.3B | 1.3B | The Pile | 2021 |
| R3 | Pythia-1.4B | 1.4B | The Pile | 2023 |
| R4 | OLMo-2-0425-1B | 1B | OLMo 2 mix | 2025 |
| R5 | Qwen2.5-0.5B | 0.5B | undisclosed | 2024 |

Every model in the study, including detectors, the perplexity model and the control generator, is pinned to an exact Hugging Face commit in `refdist/config.py`.

### 5.3 Detectors

| ID | Detector | Type | Reference |
|----|----------|------|-----------|
| D1 | Fast-DetectGPT (analytic version) | zero-shot | each of R1–R5 |
| D2 | Binoculars | zero-shot | R4 and R5 |
| D3 | openai-community/roberta-base-openai-detector | trained | GPT-2 outputs |
| D4 | Hello-SimpleAI/chatgpt-detector-roberta | trained | HC3 |
| D5 | TrustSafeAI/RADAR-Vicuna-7B (a RoBERTa-large detector, despite the name) | trained | RADAR's adversarial data |

Binoculars needs a base model plus an instruction-tuned version of the same model. Only R4 and R5 have one under 1.5B parameters, so Binoculars runs on those two only, and Fast-DetectGPT covers all five references. I'll list this as a limitation.

### 5.4 Positive control

Qwen2.5-1.5B-Instruct wrote 500 forum-style answers, each prompted with the site and opening sentence of one S6 answer. The raw outputs often contained markdown (bold text, headers, bullet points) that the human answers don't have, because their HTML was flattened to plain text. Left in, a detector could have told the two apart by the asterisks alone. So these answers got exactly the same treatment as the human text: markdown removed, then cut into the same 150–400 token windows. The raw outputs are kept for reference.

A detector stays in the study only if it separates these 500 from S6 with an AUROC of at least 0.80. Otherwise I drop it and say so.

RADAR's documentation doesn't say which of its two outputs means "AI". For RADAR only, if its AUROC comes out below 0.5, I flip its output once, record that I did, and it then has to clear the same 0.80 bar.

### 5.5 The six edits (H5)

| ID | Edit | Direction |
|----|------|-----------|
| T1 | Contractions | expand: "don't" → "do not" |
| T2 | Word frequency | rarer word → commoner word, from a hand-made plain-English list ("utilize" → "use"). The code checks every pair really is commoner using `wordfreq` |
| T3 | Sentence splitting | ", and it" → ". And it" |
| T4 | Discourse markers | add "Moreover," and similar at the start of a sentence |
| T5 | Spelling | American → British (a hand-made list plus checked -ize/-ise pairs) |
| T6 | Optional "that" | "I think it works" → "I think that it works" |

For T2 I first tried picking synonyms automatically from WordNet. I dropped it before any scoring, because it produced swaps that changed the meaning ("category" → "family"). The perplexity change would then have reflected broken sentences rather than predictability.

---

## 6. Variables

- **Outcome.** A detector's false-positive rate on a population: the share of its 1,000 human documents scoring above that detector's locked threshold.
- **Main predictor.** `log PPL_indep`: average negative log-likelihood per token under SmolLM2-360M.
- **Reference perplexity (H2).** The same measure under each reference model.
- **Covariate.** Document length in tokens.
- **Robustness covariate (H2).** Whether the population's source appears in the reference model's published training data. The Pile, for example, includes arXiv, Stack Exchange, Gutenberg (PG-19) and CourtListener. I coded this from each model's documentation before scoring, with the evidence recorded in `prereg/source_overlap.csv`. Where I couldn't verify it, I coded "unknown", and those cells are left out of that one model only. This separates "the reference finds it predictable" from "the reference was literally trained on it".

---

## 7. Analysis

- **Checks done before registering.** The analytic Fast-DetectGPT terms match a Monte-Carlo estimate. Scoring documents in batches gives the same result as scoring them one by one (to 10⁻⁵ in full precision), and the Binoculars code matches a direct implementation to 10⁻⁷. I also ran the whole analysis twice on synthetic data: with an effect planted, all six hypotheses came out confirmed, and with no effect, all came out rejected. That no-effect run is where I found that H4 could pass with a useless model, which is why the baseline comparison is there.
- **Thresholds** are set once on S6 and locked in `thresholds.json`. The code won't refit them.
- **Negative control.** Every detector's false-positive rate on S6 must come out at 5% ± 0.5%.
- **Length check.** Within S6, false-positive rates must not differ by more than 3 points across length quartiles.
- **Uncertainty.** Each false-positive rate gets a Wilson 95% interval. The headline H1 and H2 coefficients get 95% intervals from a bootstrap over documents (1,000 resamples, resampling within each population so every resample keeps 1,000 documents per population). Holm correction is applied where one test is repeated across detectors, which is H5.
- **Fixed settings.** The random seed is 42 throughout. There are 10 detector instances: Fast-DetectGPT with each of the five references, Binoculars with R4 and R5, and the three trained classifiers. Software: Python 3.11, PyTorch, transformers 4.56 or later, statsmodels 0.15, and wordfreq 3.1.1 (pinned, because T2 and T5 depend on its frequency tables).
- **Primary and exploratory.** H1–H6 and the forecast are primary. Anything else I report will be labelled exploratory, and I won't promote it afterwards.
- **Stopping rules.** If no detector passes the positive control, the study stops and I report that. If H1 and H2 both fail, I write up the null result.

---

## 8. How I'll judge the overall claim

The claim holds if **at least two of H1, H2 and H5 are confirmed.** One confirmed out of three I'll call suggestive. If all three fail, the claim is rejected. H3, H4, H6 and the forecast are reported regardless. I won't change this rule.

---

## 9. Limitations I already know about

1. Binoculars runs on R4 and R5 only.
2. English only.
3. This study is about false positives. It doesn't measure how well detectors catch real AI text.
4. Dates for Gutenberg, CourtListener, ASAP and ICNALE are upper bounds, not exact years.
5. Some sources don't have enough very long, or very short, documents to fill every length band exactly. The shortfall is recorded for each population.
6. In H5, each edit type uses a different set of documents, because each edit only applies to some documents.
7. Qwen2.5's training data isn't public, so its overlap entries are all "unknown".
8. Everything runs on Kaggle's free GPUs. Floating-point differences between machines might change the last decimal places, but they shouldn't change any sign or significance.

---

## 10. Data sharing

All code will be public. Several sources can't be redistributed: PELIC, ICNALE and ASAP have their own terms, and TwitterAAE contains tweets. For those I'll share only derived statistics (scores, perplexities and false-positive rates per population), and the code to rebuild each population from its original source. The openly licensed populations (Gutenberg, arXiv abstracts, Stack Exchange, the legal opinions and the AMI transcripts) can be shared as built, with their licences.

---

## 11. Ethics

The study only uses existing, publicly documented corpora; I collect no new data from people. I make no claim about any individual writer or document. Every result is a rate for a population. Twitter user IDs are used only to group a user's tweets and never appear in any output, and mentions are replaced with `@user`. In line with the competition terms, no ASAP essay is quoted or shared anywhere. Some populations (learner English, African American English) are exactly the groups a biased detector can harm, so I report their results with the same care as the rest and frame them as properties of the detectors, never of the writers.

---

## 12. Changes after registration

If I have to change anything, I'll post a dated public amendment explaining what changed and why. I won't amend the hypotheses, the confirmation criteria or the rule in Section 8.
