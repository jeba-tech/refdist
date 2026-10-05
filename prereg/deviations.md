# Deviations from the registered code (https://osf.io/3a7hs, commit a5e7600)

Every change to the code after registration is listed here: what changed, why,
when, and whether it touches any hypothesis, threshold or decision rule.

---

## 1. Column-name clash in the analysis tables (2026-10-04)

**What happened.** Phase 1 completed all scoring on Kaggle, then stopped at the
validity step with `KeyError: 'n_tokens'`. Score tables carry each scoring
model's own token count in a column called `n_tokens`; the document table
carries the registered length measure (the GPT-2 ruler count) under the same
name. Merging the two renamed both to `n_tokens_x` / `n_tokens_y`, so the
length-invariance check (and three later steps) could not find the column. The
synthetic dry run had not caught it because its fake score table lacked that
column.

**Change.** `refdist/analysis/tables.py::load` renames the scorer's count to
`scorer_n_tokens` when scores are loaded, so `n_tokens` always means the
registered GPT-2 ruler length. The synthetic dry run now includes the column,
and passes with unchanged verdicts.

**Effect on the registered analysis.** None. No hypothesis, model formula,
threshold, confirmation criterion or decision rule changed. Length in every
model is the registered measure, as before.

**Where the remaining Phase 1 steps ran.** The validity checks and the forecast
commit need no GPU, so they were run locally on the Phase 1 outputs downloaded
from Kaggle, with the fixed code. Thresholds had already been frozen on Kaggle
before the crash; re-running calibration reproduced them exactly (the code
refuses to refit a frozen threshold if any value would change).

---

## Recorded outcomes of the validity gates (as registered, no deviation)

- **Positive control** (AUROC ≥ 0.80): six instances pass (D1@R3 0.827,
  D1@R4 0.867, D2@R4 0.997, D2@R5 0.817, D4 0.961, D5 0.913). Four are
  excluded by the registered rule: D1@R1 0.788, D1@R2 0.792, D1@R5 0.779,
  D3 0.502. RADAR (D5) needed no label flip. Consequence, reported as such:
  the H2 reference swap retains only R3 and R4, and the R2/R3 matched-data
  pair cannot be tested under the registered rule. Any analysis that includes
  the excluded instances will be labelled exploratory.
- **Negative control:** every included instance flags 4.9% of S6 (registered
  range 4.5–5.5%). Pass.
- **Length check** (FPR spread across S6 length quartiles ≤ 3 points): D1@R4
  2.1 and D2@R4 2.7 pass; D2@R5 3.1, D4 3.1 and D1@R3 3.4 exceed it narrowly;
  D5 (RADAR) exceeds it clearly at 14.7. The registration attaches no exclusion
  to this check, so it is reported, not acted on. Note for the paper: with 250
  documents per quartile, chance alone produces spreads of about this size at
  a 5% base rate, so the 3-point margin was stricter than sampling noise
  allows; RADAR's length dependence is substantive.

## Forecast commit

- `prereg/forecast_S10_S12.json`, SHA-256
  `231444d74ae0fcfc3cfdced3e8c1ee69d5d08b5136cd9d787b135b568f54279f`,
  committed 2026-10-04 before any detector scored S10–S12.

---

## 2. Phase 2 input loading (2026-10-04)

**What happened.** Kaggle refused to attach the Phase 1 kernel as an input to
Phase 2, because that run had ended in the error described in section 1. On
the second attempt, with the Phase 1 outputs uploaded as a private Kaggle
dataset (`refdist-phase1-out`) instead, the input step picked up a cache file
that is also called `variants.parquet` in place of the H5 corpus, and stopped.

**Change.** `scripts/kaggle_phase.py` also accepts earlier outputs that come in
as a dataset. Corpus files are now read only from the top level of the code
bundle, and each one is checked for its expected columns before anything runs.
`kaggle/launch.py` attaches the Phase 1 dataset to Phase 2.

**Effect on the registered analysis.** None. This is plumbing only. The same
Phase 1 outputs (scores, frozen thresholds, committed forecast) went into
Phase 2.

## 3. H2 label when a reference pair is missing (2026-10-05)

**What happened.** The R2/R3 matched-data test of H2 needs both R2 and R3. The
positive-control gate excluded D1@R2, so only R3 was left, and the code fell
through to a REJECTED verdict on a test that could not be run.

**Change.** `refdist/analysis/hypotheses.py::h2` returns **NOT ESTIMABLE**
(with the reason) when fewer than two references remain.

**Effect on the registered analysis.** It corrects a label and changes no
number. The registration already says the matched pair "cannot be tested"
when a reference is excluded (see the validity outcomes above). The decision
rule uses H1, H2 and H5. The main H2 test (R3 and R4) is unaffected, and so is
the decision (SUGGESTIVE).

## 4. Faster bootstrap (2026-10-05)

**Change.** `refdist/analysis/bootstrap.py` selects resampled rows by position
instead of by merging tables (4.65 s down to 0.17 s per resample). The seed,
resampling scheme (documents within strata, 1,000 resamples) and models are
unchanged.

**Effect.** None on results: the old and new code give identical coefficients
for the same seed. This was checked before the switch.

## 5. Local re-run of the registered report (2026-10-05)

The full report (`python -m refdist.analysis.report`) was re-run locally on the
Phase 2 outputs with the code in sections 3 and 4. 305 of 307 reported values
match the Kaggle run exactly. The original Kaggle files are kept beside the new
ones (`results_kaggle_original.json` and `audit_model_kaggle_original.json`).
The two differences are p-values in the H2 pretraining-robustness model. That
model is close to saturated with only two references, so its p-values are
numerically unstable between machines. Its verdict, every other value and the
decision are the same.

## 6. Exploratory analysis with all ten instances (2026-10-05)

`scripts/exploratory_all_instances.py` re-runs H1–H6 with all ten detector
instances, ignoring the positive-control gate. Thresholds are calibrated on S6
by the same rule, in a scratch copy, so the frozen registered thresholds are
untouched. Its output (`exploratory_all_instances.json`, figure
`figS1_h2_all_references_EXPLORATORY`) is **exploratory**. It does not feed the
registered decision and will be reported in the paper only under that label.
