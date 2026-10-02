# What Do AI Detectors Actually Measure?

### A two-week, zero-cost experimental protocol testing whether detector false positives track distance from the detector's reference distribution

**Status:** executable protocol, v1
**Date:** 2026-09-22
**Scope:** 14 days of experiments (write-up excluded). Free-tier compute only.
**Companion document:** `alignment-detectability-proposal.md` (separate, larger study — not a dependency)

---

## 0. Plain-language summary

AI detectors work roughly like a phone keyboard's autocomplete. They ask: *would my guessing machine have predicted these words?* If yes, the text is called AI-written, because AI picks predictable words.

The problem is that some humans also write predictably — second-language writers, children, anyone following an essay formula. They get falsely accused. This is documented: roughly 6 in 10 TOEFL essays by Chinese students were flagged as AI, against about 1 in 20 for US students.

This study tests a different explanation for *all* such failures. The guessing machine only knows what it has read. Our claim is that detectors do not measure "was this written by AI" — they measure **"is this unfamiliar to my reference model."** Second-language writing, 19th-century prose, dialect and unusual registers all look unfamiliar, and all get flagged, for one shared reason.

We test this from two directions.

**Test one — change the machine, keep the writing.** Collect writing we know is human, record who gets wrongly accused, then swap the detector's reference model and look again. If the wrongly-accused group *changes* when the reference changes, the accusation was never really about the writer.

**Test two — change the writing, keep the machine.** Take one human text and edit it by fixed rules: swap in simpler words, split long sentences, expand contractions. One property at a time, same text, no AI involved. If the detector's verdict moves in step with how predictable the text became — no matter *which* edit caused it — then predictability is the only thing it was ever reading.

Together these pin the claim from both sides. Either one on its own could be explained away; both agreeing is hard to dismiss.

The payoff is a pre-deployment audit: a school could check, in advance, how badly a detector will misfire on *its own* students — instead of finding out after someone is punished. We release that check as a tool, not just a recommendation.

---

## 1. Problem addressed

The literature contains a set of disconnected false-positive findings: non-native writers, archaic text, dialectal varieties, and certain registers are each independently documented as over-flagged. There is no unifying account, and no way to predict which population a given detector will harm before deploying it.

**This study does not improve detection accuracy, fix the bias, or address whether detection is viable.** It is diagnostic. State this explicitly in the abstract.

**Intended deliverable:** a prospective audit procedure — measure a population's distance from a detector's reference distribution, obtain an advance estimate of the false-positive rate that detector will produce on it.

---

## 2. Hypotheses (pre-registered)

**H1 — Unification.** Across human-only strata, false-positive rate at a fixed threshold is predicted by the text's perplexity under an **independent** reference model. Stratum identity adds little explanatory power once perplexity is conditioned on.
*Primary estimand:* ΔR² between `FPR ~ PPL_indep` and `FPR ~ PPL_indep + stratum`.
*Falsified if* ΔR² is large — i.e. stratum membership carries substantial variance perplexity does not explain.

**H2 — Reference-shift (the causal test, load-bearing).** The *profile* of which strata are over-flagged shifts when the detector's reference distribution is swapped, and shifts in the direction predicted by the new reference's own familiarity with each stratum.
*Primary estimand:* pooled regression `FPR(r, k) ~ PPL_r(k)` across reference models r and strata k, with random intercepts for r and k; plus Kendall's τ between stratum-FPR rankings across reference pairs.
*Falsified if* τ ≈ 1 across all reference pairs (profile is fixed) or the pooled regression does not fit.

**H3 — Generalisation beyond zero-shot.** H1 and H2 hold for **trained** detectors, whose reference distribution is a training corpus rather than a scorer LM.
*Falsified if* effects appear only for perplexity-based detectors. **This outcome reduces the paper to a methodological note** — see §9.

**H4 — Audit validity (applied).** FPR on a held-out stratum is predictable from that stratum's perplexity percentile, using a model fitted on the other strata.
*Estimand:* leave-one-stratum-out prediction error, reported as mean absolute error in FPR points.

**H5 — Text-side intervention (second causal test).** Deterministic, rule-based edits to a *fixed* human text move its detector score in proportion to the perplexity change they induce, regardless of *which* linguistic property was edited.
*Primary estimand:* slope of Δ(detector score) on Δ(PPL_indep), pooled across transformation types, with transformation type as a random effect. The mechanism predicts a **single common slope** — the transform's identity should not matter once its perplexity effect is accounted for.
*Falsified if* different transformations produce systematically different slopes, i.e. detectors respond to specific surface features rather than to predictability as such.

**H6 — Agreement structure.** Detectors that share a reference distribution falsely flag the *same documents* more often than detectors with different references.
*Primary estimand:* pairwise Cohen's κ on false-flag indicators, regressed on a binary same-reference-family indicator.
*Falsified if* agreement is unrelated to reference sharing.

### Why H5 and H6 matter

H2 changes the **reference** and holds the text fixed. H5 changes the **text** and holds the reference fixed. Together they pin the mechanism from both directions — a 2×2 rather than a single intervention, which is materially harder to dismiss than either alone.

H6 is a third test that costs **no additional compute**: it is a prediction of the same theory, evaluated on scores already collected for H1–H2. A theory that makes an unexpected, verified side-prediction reads very differently from one that only fits the data it was built on.

---

## 3. Two mandatory design constraints

These are not refinements. Without both, the study is unpublishable.

### 3.1 The predictor must be independent of the detector

Predicting Fast-DetectGPT-with-Neo's flags using Neo's own perplexity is a mechanical correlation and will be rejected as such.

**Rule:** the H1 predictor is perplexity under a model used in **no** detector in the study. Fixed in advance: **SmolLM2-360M**. Never substitute mid-study.

H2 is different — there, the scorer's *own* perplexity is the predictor, and that is the point of the experiment. Keep the two analyses clearly separated in the write-up.

### 3.2 Trained detectors are required, not optional

The anticipated critical review is: *"true by construction — zero-shot detectors are functions of scorer log-probability."* That objection is correct for Fast-DetectGPT and Binoculars.

The defence is H3. A trained classifier computes no perplexity; its reference distribution is its training corpus. If the effect holds there, the claim is substantive rather than tautological.

**Restate the hypothesis accordingly throughout:** distance from the detector's *reference distribution*, whether that is a scorer LM or a training corpus.

---

## 4. Materials

### 4.1 Corpora — human-only, no generated text required

Target **1,000 documents per stratum**, length-matched to 150–400 tokens. All sources free; verify licence before use.

| # | Stratum | Source | Access | Purpose |
|---|---|---|---|---|
| S1 | Historical prose, 1850s–1900s | Project Gutenberg | public domain | temporal extreme |
| S2 | Historical prose, 1900s–1950s | Project Gutenberg | public domain | temporal |
| S3 | Academic abstracts, 1995–2005 | arXiv bulk (Kaggle) | free | temporal, within-source |
| S4 | Academic abstracts, 2015–2021 | arXiv bulk | free | temporal anchor, same source as S3 |
| S5 | Forum prose, 2011–2015 | Stack Exchange dumps | free | temporal, within-source |
| S6 | Forum prose, 2018–2021 | Stack Exchange dumps | free | **reference stratum for threshold calibration** |
| S7 | L2 learner writing, by L1 and proficiency | **PELIC** | free (GitHub) | proficiency — carries L1 + level labels |
| S8 | L2 learner writing, Asian L1s | ICNALE | free, registration | proficiency replication |
| S9 | L1 student essays | ASAP-AES (Kaggle) | free | native-speaker student baseline |
| S10 | Dialectal variety | TwitterAAE | research access | variety — see §10 |
| S11 | Legal register | US court opinions / EUR-Lex | free | register |
| S12 | Spoken transcripts | TED / Santa Barbara Corpus | free | register |

**Contamination control:** every stratum has a hard cutoff at **2021** or earlier. S1–S2 are safe by construction.

**Within-source pairs** (S3/S4, S5/S6) are the confound-resistant core: same venue, same genre, same register, differing only in era. Report these separately from cross-source comparisons; if the effect appears only across sources, era is confounded with genre and the temporal claim fails.

PELIC is the highest-value single resource here — free, pre-LLM, with L1 background and proficiency level attached, permitting within-stratum dose-response on proficiency.

### 4.2 Detectors

| ID | Detector | Type | Reference distribution |
|---|---|---|---|
| D1 | Fast-DetectGPT | zero-shot | scorer LM (swappable) |
| D2 | Binoculars | zero-shot | observer/performer pair (swappable) |
| D3 | OpenAI RoBERTa detector | trained | GPT-2 output / WebText |
| D4 | ChatGPT-detector-roberta (HC3) | trained | HC3 |
| D5 | RADAR / MAGE-trained detector | trained | different corpus again |
| D6 | One commercial free tier | black box | unknown | 

D3–D5 provide the **trained-detector reference swap** — different training corpora, same task. D6 runs on a 100-document subsample per stratum only; treat as illustrative, not as a primary result.

### 4.3 Reference models for the swap (H2)

| ID | Model | Era | Data | Role |
|---|---|---|---|---|
| R1 | GPT-2 | 2019 | WebText | pre-LLM anchor |
| R2 | GPT-Neo-1.3B | 2021 | Pile | **matched-data pair with R3** |
| R3 | Pythia-1.4B | 2023 | Pile | **matched-data pair with R2** — isolates recipe from data |
| R4 | OLMo-2-1B | 2024–25 | Dolma | contemporary |
| R5 | Qwen2.5-0.5B | 2024 | multilingual-heavy | data-mix contrast |

R2/R3 is the cleanest contrast in the design: same corpus, different date and recipe. Report it as the primary H2 pair and the rest as the extended sweep.

**Known confound:** era, size, architecture and multilinguality are entangled across R1–R5. The R2/R3 matched-data pair is the defence. A domain-swap variant was considered and **cut** to make room for §4.4 — noted in §7 as the first thing to restore if time appears.

### 4.4 Text transformations (H5)

Deterministic, **rule-based** edits applied to a fixed subsample of 300 documents drawn from S6. No LLM is used to rewrite — that would inject machine text into a human-only design and destroy the study's main methodological advantage.

| ID | Transformation | Direction | Implementation |
|---|---|---|---|
| T1 | Contraction expansion / contraction | both | rule list (`don't` ↔ `do not`) |
| T2 | Lexical frequency substitution | both | WordNet synonyms filtered by a frequency list; move toward higher- or lower-frequency alternatives |
| T3 | Sentence splitting / joining | both | punctuation and conjunction rules; directly manipulates burstiness |
| T4 | Discourse-marker insertion / removal | both | fixed marker list (`however`, `moreover`, `in addition`) |
| T5 | Orthographic variant | both | British ↔ American spelling map |
| T6 | Function-word density | both | permitted optional-word insertion/deletion (`that`-complementiser, articles where optional) |

Each transformation is applied at graded intensity (0%, 25%, 50%, 100% of eligible sites) to give a **dose-response curve per transform**. Every variant is verified meaning-preserving by rule construction; a 50-document manual spot check is logged.

This yields, per document, a set of controlled minimal pairs differing in exactly one measurable property — the text-side analogue of the reference swap.

**Why this is the strongest addition:** the stratum analysis is correlational by nature (populations differ in many ways at once). T1–T6 produce variation in *one* property at a time, on the *same* text, with no confounds of topic, author or era. If detector scores track the induced perplexity change with a common slope across all six transforms, the mechanism is demonstrated rather than inferred.

### 4.5 Positive control (validity check — not optional)

The design uses human-only text, which is elegant but creates an obvious hole: a reviewer will ask how we know the detectors work at all in this harness rather than emitting noise.

**Control set:** 500 machine-generated documents from a small open model (Qwen2.5-1.5B-Instruct), prompted to match the S6 genre and length distribution.

**Pass criterion:** each detector must achieve AUROC ≥ 0.80 separating this control set from S6. A detector failing this is excluded from the study and the exclusion is reported.

This set is used **only** to validate the pipeline. It plays no part in H1–H6 and must not be described as a detection benchmark.

---

## 5. Measures

| Quantity | Definition |
|---|---|
| **FPR(d, r, k)** | Share of stratum k flagged by detector d under reference r, at a threshold calibrated to **5% FPR on S6** |
| **PPL_indep(x)** | Mean token log-perplexity under SmolLM2-360M (H1 predictor) |
| **PPL_r(x)** | Mean token log-perplexity under reference model r (H2 predictor) |
| **Fragmentation** | Tokens per word under each reference's tokenizer — secondary predictor |
| **Score shift** | Stratum mean detector score in SD units of the S6 distribution |
| **τ(r, r′)** | Kendall's τ between stratum-FPR rankings under two references |
| **Δscore, ΔPPL** | Within-document change induced by a transformation at a given intensity (H5) |
| **κ(d, d′)** | Pairwise Cohen's κ on false-flag indicators between two detectors (H6) |

All texts truncated to a common token budget; report raw and length-matched throughout. Length enters every model as a covariate.

---

## 6. Analysis plan

Primary analyses fixed before data collection. Everything else is labelled exploratory.

1. **H1.** Mixed-effects: `FPR ~ PPL_indep + length + (1 | detector)`. Then add `stratum`. Report ΔR², with bootstrap CIs (1,000 resamples).
2. **H2 — primary.** Pooled across reference models and strata: `FPR(r,k) ~ PPL_r(k) + length + (1 | r) + (1 | k)`. Report fit and the R2/R3 matched-data contrast separately. Report τ for all reference pairs.
3. **H3.** Repeat 1–2 on D3–D5 alone. **Pre-commit to reporting zero-shot and trained results separately, whichever way they fall.**
4. **H4.** Leave-one-stratum-out prediction of FPR; report mean absolute error in FPR points.
5. **H5.** Within-document: `Δscore ~ ΔPPL_indep + (1 + ΔPPL_indep | transform)`. The test is whether the random slope variance across transforms is near zero — a **common slope** supports the mechanism; transform-specific slopes refute it. Report dose-response curves per transform.
6. **H6.** `κ(d, d′) ~ same_reference_family`, over all detector pairs, with permutation test.
7. **Multiplicity.** Holm correction across the detector × reference grid. Primary tests are declared above; nothing else may be promoted to primary after seeing results.

### 6.1 Pre-registered out-of-sample forecast

Before scoring strata S10–S12, fit the H1/H4 model on S1–S9 and **commit predicted FPRs for S10–S12, with intervals, to a timestamped public record** (OSF registration or a signed git commit) — then score them.

This converts H4 from cross-validation into a genuine prediction. It costs no compute and almost nothing in time, and it is rare enough in this literature to be worth a sentence in the abstract. For an unaffiliated researcher it also does real work: a timestamped correct forecast is credibility that does not depend on an institution's name.

---

## 7. Fourteen-day schedule

| Day | Work | Output |
|---|---|---|
| 1 | **Novelty verification** (§8). Corpus acquisition begins. **OSF pre-registration posted** | Go / no-go decision; timestamped protocol |
| 2 | Corpus assembly, cleaning, length matching, cutoff enforcement | 12 strata × 1,000 docs, frozen |
| 3 | Scoring pipeline D1–D5, batched and resumable. **Positive control (§4.5)** | Harness validated; failing detectors excluded |
| 4 | PPL_indep over all strata (GPU) **in parallel with** building the T1–T6 transform library (CPU) | PPL_indep table; transform library |
| 5 | Main run — all detectors at default reference | FPR(d, k) table |
| 6 | **H1 analysis.** **Commit S10–S12 forecast** (§6.1) before scoring them | ΔR² with CIs; timestamped prediction |
| 7 | **DECISION GATE** (§9) | Continue / pivot / stop |
| 8–9 | Reference swap R1–R5 for D1–D2 | FPR(d, r, k) grid |
| 10 | Trained-detector swap D3–D5; **H6 agreement analysis** (no new compute) | H3 data; κ matrix |
| 11 | **H2 analysis**; R2/R3 matched-data contrast | Pooled fit, τ matrix |
| 12 | **Transform run (H5)** — 300 docs × 6 transforms × 4 intensities | Δscore/ΔPPL dataset |
| 13 | **H5 analysis**; H4 leave-one-out; forecast reveal; famous-texts figure (§12.5) | Common-slope test; audit validity |
| 14 | Buffer — reruns only, no new analyses | Frozen results |

Day 14 is buffer by design. Something will fail; the schedule assumes it.

**What this costs.** The domain-swap variant is cut to make room for H5. If days 8–11 run clean and time appears, restore it — but H5 is the better use of the same two days, because it intervenes on the text rather than adding a second variant of the reference intervention.

**Do not let corpus assembly slip past day 2.** That is the standard failure mode, and it eats days 8–12, which carry the paper.

**Day 4 is deliberately parallel.** Transform-library construction is CPU work and should run while the GPU does the perplexity pass — this is how H5 fits without extending the schedule.

---

## 8. Day-1 novelty verification (go / no-go)

Before any compute. Confirm nobody has stated the reference-distribution claim:

- Search: "scorer model choice detector bias", "reference model perplexity false positive AI detection", "detector bias profile scorer", "training corpus bias AI text detector"
- Check forward citations of Liang et al. 2023 (*GPT detectors are biased against non-native English writers*)
- Check the Binoculars and Fast-DetectGPT papers' scorer ablations — confirm they ablate for **accuracy**, not for **who is falsely flagged**
- Check *Temporal Flattening in LLM-Generated Text* and *Hitting a Moving Target: Test-Time Adaptation under Continual Distribution Shift* for overlap

**Known adjacent work that does not pre-empt this:** scorer ablations optimise accuracy; Liang documents one population; industry reports note old text being flagged. None states that the *bias profile is a function of the reference distribution*, nor tests it by swapping.

If that claim is already published: stop and reconsider. Do not proceed on the assumption it is novel.

---

## 9. Decision gate (day 7) and kill criteria

| Observation at day 7 | Action |
|---|---|
| H1 holds — PPL_indep explains most cross-stratum variance | Proceed to reference swap as planned |
| ΔR² large — stratum identity dominates after conditioning | **Unification claim is false.** Pivot to a short descriptive report; do not run the full swap |
| Effects present only for zero-shot detectors (early D3–D5 signal) | Tautology objection stands. Reframe as a methodological note on detector calibration |
| Reference swap (day 11) shows no profile shift | Load-bearing result absent on the reference side — **but H5 is still ahead.** Continue to day 12; decide after |
| Reference swap fails **and** H5 shows transform-specific slopes (day 13) | Mechanism refuted from both sides. Write the negative result; it is a real contribution |
| Positive control fails for a detector (day 3) | Exclude that detector, report the exclusion, continue |

Each of these is a publishable outcome at some level. None is a reason to keep running.

**H5 de-risks the project.** Before it was added, a failed reference swap on day 11 left nothing causal. Now the two interventions fail independently: either one surviving yields a mechanism result, and both failing is a clean, interesting refutation. This is the practical reason to accept the schedule cost, separate from the novelty argument.

---

## 10. Ethics

- **S10 (TwitterAAE)** concerns harm to a specific speaker community. Check current access terms, follow them, and write the section deliberately: the finding is that detectors misfire on these speakers, which is a statement about the tools, not the writers.
- **No individual-level claims.** Report stratum-level rates only. Do not publish per-document or per-author scores for any identifiable writer.
- **No re-identification.** Aggregate before release.
- **Framing obligation.** The conclusion — that detectors respond to unfamiliarity rather than authorship — must be stated so it cannot be read as endorsing detector deployment with a tweak. It is evidence about a class of failure.
- Verify and record the licence of every corpus; release the scoring code and aggregate results, not redistributed corpora.

---

## 11. Compute and infrastructure

Free-tier only. **Kaggle Notebooks** — 30 GPU-h/week on T4 16GB, 12-hour sessions, non-decaying quota. Colab free as overflow.

| Item | Estimate |
|---|---|
| Documents | ~12,000 (12 strata × 1,000) |
| Zero-shot scoring, 5 references × 2 detectors | ~10 T4-h |
| Trained detectors D3–D5 | ~2 T4-h |
| Independent predictor pass | ~1 T4-h |
| Reruns and overhead | ~7 T4-h |
| **Total** | **~20 T4-h — under one week of quota** |

Compute is not the constraint; session management is:

- Make every script resumable from step 0; write results incrementally to disk.
- Cache model weights as a Kaggle dataset rather than re-downloading each session.
- 20GB working disk per session — do not hold all strata in memory.
- Commit intermediate score tables to HF Hub or Kaggle datasets after each stratum.

---

## 12. Deliverables

1. Short empirical paper answering H1–H6.

2. **`detector-audit` — a released, runnable tool.** Not a procedure described in prose: a notebook or CLI where an institution pastes in ~50 samples of its own known-human writing, picks a detector, and receives an estimated false-positive rate with a confidence interval.

   This is the highest-leverage deliverable for an unaffiliated researcher. A paper gets cited by researchers; a working tool gets used by universities, journalists and student advocates, and that is a different and broader kind of reach. Build it during write-up, outside the 14 experiment days — it needs no new compute, only the fitted H4 model.

3. **Scoring harness** — reference-swappable, length-controlled, resumable. The absence of length control in this literature makes this reusable on its own.

4. **Transform library (T1–T6)** — deterministic, meaning-preserving, released. Reusable by anyone probing detector sensitivity without injecting machine text.

5. **One memorable figure.** Detector AI-probability scores for famous, unambiguously human, public-domain texts — the US Constitution, Frederick Douglass, Austen, the King James Bible, MLK's *Letter from Birmingham Jail*. Near-zero compute.

   This is a communication device, not evidence, and must be labelled as such in a clearly marked sidebar. Handled carelessly it reads as a gimmick; handled well it is the figure that carries the paper beyond the field. Keep it out of the main claims.

6. **Timestamped forecast record** (§6.1) and OSF pre-registration.

7. Aggregate score tables per stratum × detector × reference.

8. Negative-result report if any decision gate fails.

---

## 13. Anticipated reviewer objections

| # | Objection | Severity | Prepared response |
|---|---|---|---|
| R1 | True by construction for perplexity-based detectors | **Critical** | H3 — trained detectors have no scorer. Reported separately, pre-committed |
| R2 | Predictor and detector share a model; correlation is mechanical | **Critical** | §3.1 — independent predictor fixed in advance |
| R3 | Strata confound era with topic, genre, orthography | High | Within-source pairs S3/S4, S5/S6; length matching; orthography-normalisation ablation. Concede residual confounding |
| R4 | Reference models differ in size and architecture, not just era | High | R2/R3 matched-data pair is primary; domain-swap variant; stated as the design's main weakness |
| R5 | How is "human" text verified? | Medium | ≤2021 cutoff; S1–S2 safe by construction; residual risk stated |
| R6 | Power and multiple comparisons | Medium | Pre-registered primary tests; Holm correction; bootstrap CIs |
| R7 | Does it hold for Turnitin / GPTZero? | Medium | D6 subsample; Turnitin inaccessible. Limitation |
| R8 | Actionable takeaway beyond "don't use detectors"? | Medium | §12.2 audit procedure with worked example |
| R9 | Novelty over Liang 2023? | High | One sentence: Liang documents *one* population; this predicts *which* populations from a measurable property, and shows the profile moves when the reference moves |
| R10 | Dialect section framing | Medium | §10 |
| R11 | "Your transformations change meaning, so the comparison is invalid" | High | T1–T6 are rule-based and meaning-preserving by construction; 50-document manual spot check logged and reported. No LLM rewriting anywhere |
| R12 | "How do we know the detectors work in your hands at all?" | High | §4.5 positive control, with a pre-declared AUROC ≥ 0.80 pass criterion and reported exclusions |
| R13 | "Transform effects are just a length artefact" | Medium | T3 changes sentence count at fixed token count; length is a covariate in every H5 model; report at matched length |
| R14 | "Six transforms × four intensities × six detectors — fishing" | Medium | H5's primary test is the *common-slope* variance, declared in advance, not per-transform significance |

---

## 14. Scope limits — state these in the paper

This study does **not**:
- improve detection accuracy;
- propose a debiasing method;
- establish that detection is or is not viable;
- make claims about any individual document or writer;
- cover non-English text (excluded by design);
- cover closed commercial detectors beyond one illustrative subsample.

---

## 15. References

- Liang, W. et al. (2023). GPT detectors are biased against non-native English writers. *Patterns*. arXiv:2304.02819
- Mitchell, E. et al. (2023). DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature. ICML. arXiv:2301.11305
- Bao, G. et al. (2024). Fast-DetectGPT. ICLR. arXiv:2310.05130
- Hans, A. et al. (2024). Spotting LLMs with Binoculars. ICML. arXiv:2401.12070
- Reinhart, A. et al. (2025). Do LLMs write like humans? *PNAS* 122(8). arXiv:2410.16107
- Sadasivan, V. S. et al. (2023). Can AI-Generated Text be Reliably Detected? arXiv:2303.11156
- Wu, J. et al. (2025). A Survey on LLM-Generated Text Detection. *Computational Linguistics* 51(1).
- Juzek, T. & Ward, Z. (2025) / related work on temporal misalignment and detector false positives — verify exact citation on day 1
- *Temporal Flattening in LLM-Generated Text.* arXiv:2604.12097 — check overlap
- *Hitting a Moving Target: Test-Time Adaptation for AI Text Detection under Continual Distribution Shift.* arXiv:2606.25152 — check overlap
- PELIC: Pitt English Language Institute Corpus — L2 writing with L1 and proficiency labels
- Blodgett, S. L. et al. (2016). Demographic Dialectal Variation in Social Media (TwitterAAE). EMNLP

---

## 16. Open decisions

- **Venue.** Workshop (realistic if H2 holds) vs short paper at a main venue (needs H3 clean). Decide after day 11, not before.
- **arXiv endorsement.** cs.CL requires endorsement for unaffiliated first-time submitters. Arrange during week 1 — email an author of a cited paper. Fallbacks: OpenReview workshop submission, Zenodo/OSF for a DOI.
- **Domain-swap variant** (§4.3) — run if days 11–12 allow; it is the easier confound story and may become the primary H2 evidence.
- Whether to add a short-text length sweep as a secondary result, or keep scope tight. Default: keep it tight.
