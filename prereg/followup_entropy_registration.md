# REFDIST follow-up E: Is the reference-swap effect carried by the reference's entropy?

**Registration of a secondary analysis of existing data**

Author: Independent researcher (fawjiajeba@gmail.com)
Parent study: REFDIST, https://osf.io/3a7hs (registered October 4, 2026)
Code: https://github.com/jeba-tech/refdist, file `refdist/analysis/entropy_followup.py`, at the commit named at the top of this registration
Date registered: (filled in by OSF)

---

## Where things stand at the moment of registering

This is not a new experiment. It is one more analysis of data the parent study has already collected. No new text is scored and no model is run. Because the data already exist, I want to be exact about what I have and have not seen.

**What I have already seen**, because the parent study and its paper report it:

- every false-positive rate for all ten detector instances on all twelve populations, including the four instances the positive control excluded;
- each reference model's mean log perplexity on each population;
- every document's detector score, which is a known function of the quantities in Section 1;
- the registered H2 result. Swapping Fast-DetectGPT's reference from Pythia-1.4B (R3) to OLMo-2-0425-1B (R4) changes which populations get flagged (permutation p = 0.0005), but the reference's own log perplexity doesn't explain the change (p = 0.85). On arXiv abstracts, mean log perplexity is almost the same under the two references, yet the false-positive rate doubles.

So the parent study already tells me that something other than mean log perplexity carries the shift. What it doesn't tell me is *what*.

**What I have not computed**, for any population or any document: the reference's entropy term, the variance term and the token-count term described below, alone or in any combination, and any breakdown of the shift into them. So far the analysis code has only run on simulated data (Section 6). It won't run on the study data until the link to this registration is saved in `prereg/followup_entropy_osf.txt`.

---

## 1. What I want to find out

Fast-DetectGPT (the analytic version, where the sampling model and the scoring model are the same) scores a document as

    score = (Σ log p − Σ μ) / √(Σ σ²) = √n · (H − NLL) / √V

For a document under reference model r, the four parts are:

- **NLL**, minus the mean log p: the reference's log perplexity on the text. This was the H2 predictor in the parent study.
- **H**, minus the mean μ: the reference's **entropy**, the loss it expects on text it generated itself.
- **V**, the mean σ²: how much that expectation varies, per token.
- **n**, the number of tokens under the reference's tokenizer.

In the parent paper I suggested that the reference's entropy, not its perplexity, carries the H2 effect. The idea is that two references can find a population equally predictable on average (the same NLL) and still score it very differently, if one is much more confident than the other on that kind of text (a different H). This follow-up asks **which of the four parts carries the change in which populations get flagged when the reference is swapped.**

The score depends on H by definition, so that alone proves nothing, and it isn't what I'm testing. The question is empirical: which part *differs between the two references in a way that depends on the population*, and so changes who gets flagged? Any of the four could turn out to matter most, or none could.

---

## 2. Hypotheses

E1 and E2 are primary. E3 is secondary. I will report all three, whatever they show.

### E1 (primary): Entropy carries Fast-DetectGPT's R3 → R4 shift

This is the same swap as the registered H2 test, and both instances passed the parent study's positive control.

**Confirmed** if H's share of the shift (Section 3) is above 0.5, *and* for each of the other three parts, the 95% bootstrap interval of (H's share − that part's share) lies entirely above 0.
**Rejected** if H's share is 0.25 or less, which is no more than an equal split between four parts.
Anything in between I will report as partial.

### E2 (primary): Binoculars' normaliser carries its R4 → R5 shift

Binoculars scores a document as −NLL_perf / X. NLL_perf is the performer model's log perplexity on the text. X is the cross-perplexity, meaning the observer model's predicted distribution scored by the performer, which is how the official Binoculars code computes it. If the mechanism is "a score is measured against what the reference itself expects", then X is the Binoculars equivalent of H. Both Binoculars instances (R4 and R5) passed the positive control.

**Confirmed** if X's share is above 0.5 and the 95% bootstrap interval of (X's share − NLL_perf's share) lies entirely above 0.
**Rejected** if that interval lies entirely below 0.
Anything else I will report as partial.

### E3 (secondary): The same pattern across all five references

I run the same breakdown for all ten pairs of Fast-DetectGPT references, always with the lower-numbered reference as the starting point (R1 → R2, R1 → R3, and so on). Three of those five instances failed the parent study's positive control, so, like the exploratory section of the parent paper, this is secondary.

**Supported** if H has the largest share in at least 6 of the 10 pairs.

### How I'll describe the result, decided now

- If E1 is confirmed, the paper will say that Fast-DetectGPT's reference effect is carried by the reference's entropy.
- If E2 is also confirmed, it will say the same holds for the Binoculars normaliser.
- If E1 is partial or rejected, the paper will say so and name the part with the largest share.

I won't change this.

---

## 3. Method

**Swapping one part at a time.** Take a starting reference *b* and a target reference *t*. For every subset T of the four parts, I score each of the 12,000 human documents with the parts in T taken from *t* and the rest from *b*. I then set the threshold on S6 exactly as the parent study did (the 95th percentile of the S6 scores, numpy method "higher", with a document flagged if its score is strictly above it) and compute the false-positive rate on each of the other eleven populations.

**How far a subset closes the gap.** D(T) is the sum, over the eleven populations, of |FPR(T) − FPR(target)|. D(∅), with nothing swapped, is the full distance between the two references' profiles. The value of a subset is v(T) = D(∅) − D(T), meaning how much of that distance taking T from the target closes.

**Shares.** Each part's Shapley value over v, divided by D(∅), is its **share** of the shift. The four shares add up to 1. A share can be negative, if swapping that part moves the profile away from the target rather than towards it. I also report how much each part closes when it is swapped on its own (the single-swap share).

**Binoculars.** Same procedure, with two parts, NLL_perf and X.

**Uncertainty.** I draw 1,000 bootstrap resamples of documents within each population (seed 42). Both references always use the same resampled documents, and the threshold is set again on each resampled S6. Intervals run from the 2.5th to the 97.5th percentile. The bootstrap is used for E1 and E2. E3 only counts which part has the largest share, so it needs no interval.

**Is there a shift to explain?** A shift has to exist before it can be broken down. So E1 and E2 are only read if the parent study's registered profile-shift permutation test (2,000 shuffles, seed 0, on the frozen registered flags) between the two instances gives p < 0.05. If it doesn't, the verdict is NOT ESTIMABLE. For R3 → R4 this p-value is already reported in the parent paper (0.0005). For Binoculars R4 → R5 it is computed when the analysis runs.

**A check before anything else.** The code first rebuilds every document's score from its parts. If any rebuilt score differs from the score the registered analysis used by more than 10⁻⁴, it stops.

---

## 4. Also reported (descriptive, no hypothesis)

- E1 in the reverse direction (R4 → R3).
- The mean NLL, H, V and n per population under every reference.
- The single-swap shares.

---

## 5. Limitations I already know about

1. **This is an accounting of the score, not an experiment on text.** It can say which part moves the profile, but not why a reference has the entropy it has.
2. **The parts are correlated**, H and NLL especially. Shapley values split what they share evenly between them, which is why I also report the single-swap shares.
3. **Only two references each in E1 and E2.** As in the parent study, R3 and R4 differ in training data, architecture, tokenizer, size and date. The token-count part captures only the tokenizer's direct effect on the score, not everything a tokenizer changes.
4. **What I already know.** I've seen the false-positive rates and perplexities listed above. The criteria are fixed here, and the code has only run on simulated data, so that knowledge can't steer the analysis. But it does mean "something other than NLL carries the shift" isn't a new finding. The open question is which part.

---

## 6. Checks on simulated data (already run)

I simulated twelve populations of 1,000 documents and planted a population-specific shift in one part at a time (seed 42, 200 bootstrap resamples):

| Shift planted in | Distance | Shares (NLL / H / V / n) | E1 verdict |
|---|---|---|---|
| H (entropy) | 0.369 | 0.04 / 0.96 / 0.00 / 0.00 | CONFIRMED |
| NLL | 0.369 | 0.97 / 0.03 / 0.00 / 0.00 | REJECTED |
| V | 0.109 | 0.16 / 0.11 / 0.73 / 0.00 | REJECTED |
| nothing (null) | 0.017 | 0.97 / 0.03 / 0.00 / 0.00 | REJECTED |

| Binoculars, planted in | Distance | Shares (NLL_perf / X) | E2 verdict |
|---|---|---|---|
| X (normaliser) | 1.159 | 0.01 / 0.99 | CONFIRMED |
| NLL_perf | 1.706 | 1.00 / 0.00 | REJECTED |

Each time, the method found the part I had planted the shift in. In the null run the shares mean nothing, because the profile hardly moves at all. That is exactly why the permutation check in Section 3 comes before any verdict. I also checked that the formulas give the same scores as the registered Fast-DetectGPT and Binoculars code on simulated tokens. The largest difference was 5 × 10⁻¹⁵.

---

## 7. Changes after registration

If I have to change anything, I'll post a dated public amendment explaining what changed and why. I won't amend the hypotheses, the confirmation criteria or the description rule in Section 2.
