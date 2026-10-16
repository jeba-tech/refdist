# REFDIST derived-statistics release

Derived statistics for the pre-registered REFDIST study
(registration: https://osf.io/3a7hs; code: https://github.com/jeba-tech/refdist).
This is the release promised in Section 10 of the registration.

**No text from any source is included.** Several sources (PELIC, ICNALE, ASAP-AES,
TwitterAAE) cannot be redistributed, so this release holds only numbers computed from
the texts: document lengths, perplexities, detector scores, flags and rates. The
code in the repository rebuilds every population from its original source.

Built by `scripts/make_release.py`. `SHA256SUMS` lists a checksum for every file.

## Files

| File | Rows | What it is |
|---|---|---|
| `document_scores.csv.gz` | 12,000 | One row per human document (12 populations × 1,000) |
| `edit_scores.csv.gz` | 5,085 | One row per rule-edited S6 document used in H5 |
| `control_scores.csv.gz` | 500 | Scores of the machine-written positive-control texts |
| `population_fpr.csv` | 120 | False-positive rate per detector instance × population, with Wilson 95% interval |
| `population_summary.csv` | 12 | Per population: mean length, mean perplexities, documents per length band |
| `forecast_reveal.csv` | 18 | The pre-committed S10–S12 forecast against the observed rates |
| `results/results.json` | | All registered results (H1–H6, validity checks, decision) |
| `results/thresholds.json` | | The frozen thresholds (5% FPR on S6) for the six included instances |
| `results/validity.json` | | Positive control, negative control and length check |
| `results/precision.json` | | fp16 vs fp32 check run before registration |
| `results/audit_model.json` | | Fitted model behind the audit tool |
| `results/exploratory_all_instances.json` | | Exploratory re-analysis with all ten instances (not registered) |
| `results/followup_entropy.json` | | Registered follow-up E (https://osf.io/rd6jg): which part of the score carries the reference swap |
| `results/ledger.jsonl` | | Scoring log with UTC timestamps for every scoring step |

## Columns

**`document_scores.csv.gz`**
- `doc_id`: population ID plus the first 12 hex digits of the SHA-256 of the
  document's text. Anyone who rebuilds a population from its source can match their
  documents to these rows; the ID reveals nothing about the text or its writer.
- `population` (S1–S12), `source`, `year` (an upper bound for some sources, see the
  registration), `license` (of the source text, which is not included here).
- `n_tokens_gpt2`: length in GPT-2 tokens, the length measure used in every model.
- `log_ppl_indep`: mean negative log-likelihood per token under SmolLM2-360M, the
  independent predictor.
- `log_ppl_R1` … `log_ppl_R5`: the same under each reference model (R1 GPT-2 Large,
  R2 GPT-Neo-1.3B, R3 Pythia-1.4B, R4 OLMo-2-0425-1B, R5 Qwen2.5-0.5B).
- `D1@R1` … `D5`: detector scores, oriented so that higher means more likely to be
  flagged. D1 = Fast-DetectGPT (analytic) with each reference; D2 = Binoculars with
  R4 and R5; D3 = RoBERTa GPT-2 output detector; D4 = RoBERTa trained on HC3;
  D5 = RADAR.
- `flag_<instance>`: 1 if the score exceeds the frozen threshold. Given only for the
  six instances that passed the registered positive control (D1@R3, D1@R4, D2@R4,
  D2@R5, D4, D5).

**`edit_scores.csv.gz`**: `variant_id`, the unedited `doc_id` (in S6), `transform`
(T1–T6), `intensity` (share of eligible sites edited), `n_sites_eligible`,
`n_sites_applied`, `n_tokens_gpt2`, then the same score columns as above. The edited
texts and the edit snippets are not included; the code regenerates them exactly from S6.

**`control_scores.csv.gz`**: scores of the 500 positive-control texts written by
Qwen2.5-1.5B-Instruct. They are used only for the positive control and never in a
hypothesis test.

**`population_fpr.csv`**: `instance`, `population`, `n`, `n_flagged`, `fpr`,
`wilson_lo`, `wilson_hi`, `status`. Rows marked *exploratory* belong to the four
instances that failed the positive control. Their rates use a threshold set by the
same rule in a separate exploratory run, and they are not part of any registered result.

## The audit tool

`results/audit_model.json` is the fitted model behind the audit tool in the code
repository. To estimate a detector's false-positive rate on your own human-written
texts:

```
pip install -e .
refdist-audit --texts ./my_texts/ --detector all --model release/results/audit_model.json
```

It reports a rate for the whole set of texts, with a 95% interval. The intervals are
wide, and the tool says nothing about any individual text.

## Licence

These derived statistics are released under CC BY 4.0. The source texts are not
included and remain under their own terms (see the `license` column).
