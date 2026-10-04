"""
FROZEN registries for the REFDIST study.

Nothing in this file may change after the OSF registration date
(2026-10-04) without a logged public amendment. The registries here define
every hypothesis test; silently editing one invalidates the pre-registration.

The independence assertion at the bottom of this file is the single most
important line in the codebase: it enforces that the H1 predictor model
appears in no detector. That property is what defends the study against the
"mechanical correlation" review.
"""

from __future__ import annotations

import hashlib
import json

# ─────────────────────────────────────────────────────────────────────────────
# Independent predictor (H1, H5)
# ─────────────────────────────────────────────────────────────────────────────
# PPL_indep is computed with this model and nothing else. It must appear in
# ZERO detectors. Enforced by assertion at module load.
INDEPENDENT_PREDICTOR = {
    "model_id": "HuggingFaceTB/SmolLM2-360M",
    "revision": "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
    "params_M": 360,
}

# ─────────────────────────────────────────────────────────────────────────────
# Reference models R1-R5
# ─────────────────────────────────────────────────────────────────────────────
# `binoculars_pair` is None where no instruction-tuned sibling of the same base
# exists under the 1.5B ceiling. This is a real coverage gap, reported as a
# limitation rather than hidden: Fast-DetectGPT carries the full H2 era sweep,
# Binoculars replicates it only where pairs exist.
REFERENCE_MODELS = {
    "R1": {
        "model_id": "openai-community/gpt2-large",
        "revision": "32b71b12589c2f8d625668d2335a01cac3249519",
        "era": 2019,
        "corpus": "WebText",
        "params_M": 774,
        "binoculars_pair": None,
    },
    "R2": {
        "model_id": "EleutherAI/gpt-neo-1.3B",
        "revision": "dbe59a7f4a88d01d1ba9798d78dbe3fe038792c8",
        "era": 2021,
        "corpus": "Pile",
        "params_M": 1300,
        "binoculars_pair": None,
        "note": "matched-data pair with R3 (both Pile); primary H2 contrast",
    },
    "R3": {
        "model_id": "EleutherAI/pythia-1.4b",
        "revision": "fedc38a16eea3bd36a96b906d78d11d2ce18ed79",
        "era": 2023,
        "corpus": "Pile",
        "params_M": 1400,
        "binoculars_pair": None,
        "note": "matched-data pair with R2 (both Pile); primary H2 contrast",
    },
    "R4": {
        "model_id": "allenai/OLMo-2-0425-1B",
        "revision": "a1847dff35000b4271fa70afc5db10fd29fedbdf",
        "era": 2025,
        "corpus": "OLMo 2 mix",  # olmo-mix-1124 + dolmino-mix-1124 (verified from the dataset cards)
        "params_M": 1000,
        "binoculars_pair": "allenai/OLMo-2-0425-1B-Instruct",
        "binoculars_pair_revision": "48d788eca847d4d7548f375ad03d3c9312f6139e",
    },
    "R5": {
        "model_id": "Qwen/Qwen2.5-0.5B",
        "revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
        "era": 2024,
        "corpus": "multilingual web",
        "params_M": 500,
        "binoculars_pair": "Qwen/Qwen2.5-0.5B-Instruct",
        "binoculars_pair_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
    },
}

# R2 vs R3: same training corpus (Pile), different recipe and date. Any FPR
# profile difference between them cannot be explained by training data content.
MATCHED_DATA_PAIR = ("R2", "R3")

# ─────────────────────────────────────────────────────────────────────────────
# Detectors D1-D5
# ─────────────────────────────────────────────────────────────────────────────
# Zero-shot detectors have a scorer LM (the reference). Trained classifiers do
# not -- their reference is a training corpus. H3 tests whether the mechanism
# survives on the trained family, which is what defeats the "true by
# construction" objection.
DETECTORS = {
    "D1": {
        "name": "fast_detectgpt",
        "family": "zero_shot",
        "reference_kind": "single_model",
        "references": ["R1", "R2", "R3", "R4", "R5"],
    },
    "D2": {
        "name": "binoculars",
        "family": "zero_shot",
        "reference_kind": "model_pair",
        "references": ["R4", "R5"],
    },
    "D3": {
        "name": "roberta_openai",
        "family": "trained",
        "reference_kind": "training_corpus",
        "model_id": "openai-community/roberta-base-openai-detector",
        "revision": "6cba99c003b711c7fe94f8a3aa2be35a792cb6fa",
        "ai_label_index": 0,  # id2label {0: Fake, 1: Real}
        "training_corpus": "gpt2_outputs_vs_webtext",
        "references": [],
    },
    "D4": {
        "name": "chatgpt_detector_roberta",
        "family": "trained",
        "reference_kind": "training_corpus",
        "model_id": "Hello-SimpleAI/chatgpt-detector-roberta",
        "revision": "d2b342c61775d5dd0221808a79983ed3b86ffd86",
        "ai_label_index": 1,  # id2label {0: Human, 1: ChatGPT}
        "training_corpus": "hc3",
        "references": [],
    },
    "D5": {
        "name": "radar",
        "family": "trained",
        "reference_kind": "training_corpus",
        # RoBERTa-large detector (~355M) despite the name; "Vicuna-7B" is the
        # generator it was adversarially trained against.
        "model_id": "TrustSafeAI/RADAR-Vicuna-7B",
        "revision": "4ff1f23a69a36aa1df47b0933be6279f1b896c9b",
        # Not documented in config.json. Index 0 follows the authors' demo code;
        # the positive-control gate rejects the detector if this is inverted.
        "ai_label_index": 0,
        "training_corpus": "radar_adversarial",
        "references": [],
    },
}

TRAINED_DETECTORS = [k for k, v in DETECTORS.items() if v["family"] == "trained"]
ZERO_SHOT_DETECTORS = [k for k, v in DETECTORS.items() if v["family"] == "zero_shot"]



def reference_family(detector: str, reference: str | None) -> str:
    """
    The distribution a detector instance measures distance from (H6).

    Zero-shot instances inherit their scorer's pretraining corpus, so R2 and R3
    (both Pile) share a family. Trained classifiers are keyed by training corpus.
    """
    det = DETECTORS[detector]
    if det["family"] == "trained":
        return det["training_corpus"]
    return REFERENCE_MODELS[reference]["corpus"]

# ─────────────────────────────────────────────────────────────────────────────
# Strata S1-S12
# ─────────────────────────────────────────────────────────────────────────────
# Every document is pre-2022. S3/S4 and S5/S6 are within-source pairs: same
# venue and genre, only era differs. Those two pairs are the confound-resistant
# core of the design, because a difference between them cannot be attributed to
# topic or register.
STRATA = {
    "S1":  {"name": "Gutenberg 1850-1899",       "source": "gutenberg",     "year_min": 1850, "year_max": 1899},
    "S2":  {"name": "Gutenberg 1900-1921",       "source": "gutenberg",     "year_min": 1900, "year_max": 1921},
    "S3":  {"name": "arXiv 1995-2005",           "source": "arxiv",         "year_min": 1995, "year_max": 2005},
    "S4":  {"name": "arXiv 2015-2021",           "source": "arxiv",         "year_min": 2015, "year_max": 2021},
    "S5":  {"name": "StackExchange 2011-2015",   "source": "stackexchange", "year_min": 2011, "year_max": 2015},
    "S6":  {"name": "StackExchange 2018-2021",   "source": "stackexchange", "year_min": 2018, "year_max": 2021},
    "S7":  {"name": "PELIC L2 English",          "source": "pelic",         "year_min": 2006, "year_max": 2020},
    "S8":  {"name": "ICNALE L2 English",         "source": "icnale",        "year_min": 2008, "year_max": 2020},
    "S9":  {"name": "ASAP-AES student essays",   "source": "asap",          "year_min": 1999, "year_max": 2012},
    "S10": {"name": "TwitterAAE",                "source": "twitteraae",    "year_min": 2013, "year_max": 2018},
    "S11": {"name": "Legal opinions",            "source": "legal",         "year_min": 1990, "year_max": 2021},
    "S12": {"name": "Spoken transcripts",        "source": "spoken",        "year_min": 1990, "year_max": 2021},
}

WITHIN_SOURCE_PAIRS = [("S3", "S4"), ("S5", "S6")]

# Held out for the pre-registered out-of-sample forecast. The H1 model is fit
# on FIT_STRATA only, predictions for FORECAST_STRATA are written and hashed,
# and only then are these three scored.
FORECAST_STRATA = ["S10", "S11", "S12"]
FIT_STRATA = [s for s in STRATA if s not in FORECAST_STRATA]

# All detector thresholds are calibrated to TARGET_FPR on this stratum. It is
# built first and never resampled: re-drawing it would move every threshold and
# silently invalidate every FPR in the study.
CALIBRATION_STRATUM = "S6"
TARGET_FPR = 0.05

# ─────────────────────────────────────────────────────────────────────────────
# Corpus construction
# ─────────────────────────────────────────────────────────────────────────────
CORPUS = {
    "docs_per_stratum": 1000,
    "min_tokens": 150,
    "max_tokens": 400,
    "seed": 42,
    "hard_cutoff_year": 2021,
    "length_match_bins": 5,
    "minhash_threshold": 0.8,
    "candidates_per_stratum": 2500,
    "max_windows_per_source": 3,
    # A single tokenizer is the length ruler for every stratum, so "150-400
    # tokens" means the same thing everywhere. GPT-2 BPE: stable and widely used.
    "length_tokenizer": {
        "model_id": "openai-community/gpt2",
        "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
    },
}

# Per-source rules. Each is declared in the OSF registration.
SOURCES = {
    "gutenberg": {
        "catalog": "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv",
        "text_url": "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
        "locc_prefixes": ["PR", "PS"],  # English and American literature
        # Original publication year is not in Gutenberg metadata; author
        # lifespan bounds when the text was written.
        "S1": {"death_min": 1855, "death_max": 1899},
        "S2": {"birth_min": 1865, "birth_max": 1895},
        "max_books": 900,
        "request_delay_s": 1.0,
        # Only ~360 prose authors died 1855-1899 once poetry and drama are
        # excluded; a cap of 3 windows each cannot reliably reach 1,000.
        "max_windows_per_source": 4,
    },
    "arxiv": {
        "api": "http://export.arxiv.org/api/query",
        # Archives present in both eras, sampled with equal quotas so S3 and S4
        # differ in era, not in field mix.
        "archives": ["astro-ph*", "cond-mat*", "hep-th*", "math.*", "quant-ph*", "cs.*"],
        "request_delay_s": 3.1,
        "max_latex_dollars": 4,
    },
    "stackexchange": {
        "dump": "https://archive.org/download/stackexchange/{site}.stackexchange.com.7z",
        "sites": ["english", "travel", "cooking"],
        "post_type": "answer",
        # Long answers are rare; a small pool cannot fill the 350-400 token bin.
        "reservoir_per_site": 4000,
        # Pre-LLM guarantee: exclude any post edited after this date.
        "max_last_edit": "2021-12-31",
    },
    "pelic": {
        "url": "https://media.githubusercontent.com/media/ELI-Data-Mining-Group/PELIC-dataset/master/PELIC_compiled.csv",
        "version": "final",  # one version per (writer, question): drafts are near-duplicates
    },
    "icnale": {"manual_dir": "icnale"},   # registration required; user places files
    "asap": {"manual_file": "asap/training_set_rel3.tsv", "max_placeholders": 5},
    "twitteraae": {
        "zip": "http://slanglab.cs.umass.edu/TwitterAAE/TwitterAAE-full-v1.zip",
        "member": "TwitterAAE-full-v1/twitteraae_all_aa",
        "exclude_retweets": True,
    },
    "legal": {
        "shard": "https://huggingface.co/datasets/pile-of-law/pile-of-law/resolve/main/data/train.courtlisteneropinions.{n}.jsonl.xz",
        # created_timestamp is when CourtListener ingested the record, which is
        # never earlier than authorship, so <=2021 is a conservative bound.
        "max_created_year": 2021,
        "max_citations_per_paragraph": 2,
    },
    "spoken": {
        "zip": "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip",
        "year": 2005,
        # ~170 meetings: a cap of 3 windows per meeting cannot reach 1,000 docs.
        "max_windows_per_source": 8,
        "turn_gap_s": 1.5,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Transforms T1-T6 (H5)
# ─────────────────────────────────────────────────────────────────────────────
# Rule-based and deterministic. No LLM rewriting anywhere: injecting machine
# text into the corpus would forfeit the human-only design that is this study's
# main methodological advantage.
TRANSFORMS = {
    "T1": {"name": "contractions",   "mechanism": "contraction expand/contract"},
    "T2": {"name": "lexfreq",        "mechanism": "frequency-directed synonym substitution"},
    "T3": {"name": "sentence_split", "mechanism": "sentence split/join (burstiness at fixed tokens)"},
    "T4": {"name": "discourse",      "mechanism": "discourse marker insert/remove"},
    "T5": {"name": "orthography",    "mechanism": "British/American spelling map"},
    "T6": {"name": "function_words", "mechanism": "optional complementiser/article"},
}

TRANSFORM_INTENSITIES = [0.0, 0.25, 0.50, 1.00]
TRANSFORM_SOURCE_STRATUM = "S6"
TRANSFORM_N_DOCS = 300

# ─────────────────────────────────────────────────────────────────────────────
# Positive control (validity gate, firewalled from H1-H6)
# ─────────────────────────────────────────────────────────────────────────────
# The only place any text is generated. Its sole purpose is to prove the
# scoring harness works. A detector below MIN_AUROC is excluded from the study
# and the exclusion is reported. This set is never concatenated into the human
# corpus and is never described as a detection benchmark.
CONTROL = {
    "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
    "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "n_docs": 500,
    "min_auroc": 0.80,
    "match_stratum": "S6",
}

# ─────────────────────────────────────────────────────────────────────────────
# Analysis thresholds (pre-declared)
# ─────────────────────────────────────────────────────────────────────────────
# These are the confirmation criteria from the OSF registration, in code so
# they cannot drift from the registered document.
ANALYSIS = {
    "alpha": 0.05,
    "bootstrap_resamples": 1000,
    "h1_min_delta_r2": 0.10,
    "h1_fail_delta_r2": 0.05,
    "h1_max_coef_attenuation": 0.30,
    "h4_max_mae_confirm": 0.10,
    "h4_max_mae_partial": 0.15,
    # Skill over a perplexity-blind baseline. Without it, H4 "confirms" whenever
    # FPR barely varies -- the synthetic null dry run showed exactly that.
    "h4_min_skill": 0.25,
    # SD of transform-specific slopes / |common slope|. 0.25: transforms may
    # deviate from the common slope by a quarter of its size and still count
    # as one mechanism; 0.05 would demand agreement no measurement can show.
    "h5_max_random_slope_ratio": 0.25,
    "length_quartile_fpr_margin": 0.03,
}

# ─────────────────────────────────────────────────────────────────────────────
# Study metadata
# ─────────────────────────────────────────────────────────────────────────────
STUDY = {
    "name": "REFDIST",
    "title": "REFDIST: A Causal Reference-Swap Audit of False Positives in AI Text Detectors",
    "github": "https://github.com/jeba-tech/refdist",
    "osf": "",     # fill after registration
    "arxiv": "",   # fill after posting
    "registration_date": "2026-10-04",
}

# ─────────────────────────────────────────────────────────────────────────────
# INDEPENDENCE ENFORCEMENT -- do not remove
# ─────────────────────────────────────────────────────────────────────────────
# H1 claims that an *independent* perplexity score predicts detector FPR. That
# claim is only meaningful if the predictor model is absent from every detector.
# If it ever appears in one, the correlation becomes mechanical and H1 is void.
# This is enforced here, at import time, so no analysis can run in a violating
# configuration.


def _detector_model_ids() -> set[str]:
    """Every model id any detector touches, directly or via its reference."""
    ids: set[str] = set()
    for det in DETECTORS.values():
        if det.get("model_id"):
            ids.add(det["model_id"])
        for ref_key in det.get("references", []):
            ref = REFERENCE_MODELS[ref_key]
            ids.add(ref["model_id"])
            if ref.get("binoculars_pair"):
                ids.add(ref["binoculars_pair"])
    return ids


_INDEP = INDEPENDENT_PREDICTOR["model_id"]
_DETECTOR_IDS = _detector_model_ids()

if _INDEP in _DETECTOR_IDS:
    raise AssertionError(
        f"INDEPENDENCE VIOLATION: the H1 predictor {_INDEP!r} appears in a "
        f"detector registry. H1 would be mechanically true and therefore void. "
        f"Remove it from DETECTORS/REFERENCE_MODELS before running anything."
    )

# Binoculars must only be declared for references that actually have a pair.
for _ref_key in DETECTORS["D2"]["references"]:
    if not REFERENCE_MODELS[_ref_key].get("binoculars_pair"):
        raise AssertionError(
            f"Binoculars is declared for {_ref_key} but that reference has no "
            f"instruction-tuned sibling. Either add the pair or drop the "
            f"reference from D2."
        )

# The calibration stratum must exist and must not be one of the held-out strata,
# since thresholds have to be set before the forecast is revealed.
assert CALIBRATION_STRATUM in STRATA, "calibration stratum is not a declared stratum"
assert CALIBRATION_STRATUM not in FORECAST_STRATA, (
    "calibration stratum cannot be held out for the forecast: thresholds must "
    "be fixed before forecast strata are scored"
)


# ─────────────────────────────────────────────────────────────────────────────
# Provenance
# ─────────────────────────────────────────────────────────────────────────────
def config_hash() -> str:
    """
    Short SHA-256 over the frozen registries, stamped into every artifact.

    Results carrying different hashes were produced under different
    configurations and must not be pooled.
    """
    payload = json.dumps(
        {
            "independent_predictor": INDEPENDENT_PREDICTOR,
            "reference_models": REFERENCE_MODELS,
            "detectors": DETECTORS,
            "strata": STRATA,
            "corpus": CORPUS,
            "sources": SOURCES,
            "transforms": TRANSFORMS,
            "transform_intensities": TRANSFORM_INTENSITIES,
            "control": CONTROL,
            "analysis": ANALYSIS,
            "target_fpr": TARGET_FPR,
            "calibration_stratum": CALIBRATION_STRATUM,
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def binoculars_references() -> list[str]:
    """References with a usable base+instruct pair."""
    return [k for k, v in REFERENCE_MODELS.items() if v.get("binoculars_pair")]


def all_model_ids() -> dict[str, str]:
    """
    Every model the study downloads, mapped to its pinned revision.

    Used by the model-caching notebook so a Kaggle session never discovers a
    missing download mid-run.
    """
    models = {INDEPENDENT_PREDICTOR["model_id"]: INDEPENDENT_PREDICTOR["revision"]}
    for ref in REFERENCE_MODELS.values():
        models[ref["model_id"]] = ref["revision"]
        if ref.get("binoculars_pair"):
            models[ref["binoculars_pair"]] = ref.get("binoculars_pair_revision", "main")
    for det in DETECTORS.values():
        if det.get("model_id"):
            models[det["model_id"]] = det["revision"]
    models[CONTROL["model_id"]] = CONTROL["revision"]
    return models
