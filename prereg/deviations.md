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
