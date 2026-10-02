# REFDIST

**A Causal Reference-Swap Audit of False Positives in AI Text Detectors**

Do AI-text detectors flag human writing because of *who wrote it*, or because the text is *predictable to the detector's reference distribution*? REFDIST tests this on 12,000 human-written documents from 12 populations (Victorian novels, arXiv abstracts, forum answers, L2 learner essays, African American English tweets, court opinions, meeting transcripts) with two causal interventions:

- **Reference swap:** hold the text fixed, change the detector's scoring model, and see whether the population profile of false positives moves with that model's perplexity.
- **Text-side edits:** apply deterministic, rule-based edits to human text and see whether detector scores follow the resulting perplexity change, whatever kind of edit caused it.

The study is pre-registered: [`prereg/osf_registration.md`](prereg/osf_registration.md). Every threshold and confirmation rule in that document is also enforced in code (`refdist/config.py`, `refdist/analysis/hypotheses.py`).

## Layout

| Path | What |
|---|---|
| `refdist/config.py` | Frozen registries: strata, models (pinned SHAs), detectors, thresholds. Asserts that the H1 predictor is used by no detector. |
| `refdist/corpora/` | One adapter per source, plus `build.py`: year cutoff, length ruler, dedup, source caps, length matching |
| `refdist/transforms/` | T1–T6 rule-based edits (H5); `apply.py` builds the variants |
| `refdist/scoring/` | Cached per-token statistics, Fast-DetectGPT, Binoculars, trained classifiers, resumable runner, positive control |
| `refdist/analysis/` | Validity gates, H1–H6, forecast, bootstrap, report |
| `notebooks/refdist_kaggle.ipynb` | GPU stages on a free Kaggle T4 |
| `scripts/milestone0_local.py` | Proves the scoring maths (analytic vs Monte-Carlo, batching, orientation) |
| `scripts/dry_run_analysis.py` | Runs the full analysis on synthetic data with and without a planted effect |

## Runbook

```bash
pip install -e .
pytest tests/                               # 38 tests
python scripts/milestone0_local.py          # scoring maths, CPU, ~2 min
python scripts/dry_run_analysis.py          # analysis recovers planted effect, rejects null

# 1. corpus (CPU, local): S8 and S9 need manual downloads, see below
python -m refdist.corpora.build             # all strata; built strata are frozen
python -m refdist.transforms.apply          # H5 variants

# 2. upload data/processed/{documents,variants}.parquet as Kaggle dataset "refdist-data"
# 3. run notebooks/refdist_kaggle.ipynb on a T4. It stops after committing the forecast:
#    post the printed SHA-256 on OSF, then continue.
```

**Manual sources:** S8 ICNALE needs free registration at https://language.sakura.ne.jp/icnale/; extract the Written Essays into `data/raw/icnale/`. S9 ASAP needs you to accept the rules at https://www.kaggle.com/competitions/asap-aes, then place `training_set_rel3.tsv` in `data/raw/asap/`.

## Integrity rules enforced in code

- The H1 predictor (SmolLM2-360M) is asserted absent from every detector. The code will not import if this is violated.
- The calibration stratum S6 cannot be rebuilt, and frozen thresholds cannot be refitted.
- Detector stages refuse the held-out strata S10–S12 until the forecast file exists. The forecast's hash is checked before it is revealed.
- Generated text exists only in `control.parquet`, a separate file that never enters a hypothesis test.
