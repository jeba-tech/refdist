"""
Writes the paper's numeric tables (LaTeX) straight from the registered results,
so no number in the paper is typed by hand.

  paper/tables/fpr.tex        FPR (%) of each included instance on each population
  paper/tables/verdicts.tex   one row per hypothesis: verdict and key numbers
  paper/tables/numbers.tex    \newcommand macros for every number quoted in the text

Usage:  REFDIST_DATA=data/kaggle/phase2/data python scripts/paper_tables.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from statsmodels.stats.proportion import proportion_confint  # noqa: E402

from refdist import config, paths  # noqa: E402
from refdist.analysis import tables  # noqa: E402
from refdist.analysis.forecast import reveal  # noqa: E402
from refdist.metrics import flags as fl  # noqa: E402

OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
R = json.loads((paths.results() / "results.json").read_text())
EXP = json.loads((ROOT / "data" / "results" / "exploratory_all_instances.json").read_text())

scores, docs = tables.load()
inc = R["included_instances"]
cells = tables.cell_table(scores[scores["instance"].isin(fl.load_thresholds())
                                 | ~scores["detector"].str.match(r"^D\d$")], docs)
ppl = cells.groupby("stratum")["log_ppl_indep"].mean()
cz = cells[cells["instance"].isin(inc)]
fpr = cz.pivot_table(index="stratum", columns="instance", values="fpr").loc[ppl.sort_values().index, inc]
n = cz.pivot_table(index="stratum", columns="instance", values="n").loc[fpr.index, inc]

# ---- Table: FPR by population ------------------------------------------------
NAMES = {"S1": "Literary prose 1850--99", "S2": "Literary prose 1900--21", "S3": "arXiv 1995--2005",
         "S4": "arXiv 2015--21", "S5": "Stack Exch.\\ 2011--15", "S6": "Stack Exch.\\ 2018--21$^\\dagger$",
         "S7": "PELIC (L2)", "S8": "ICNALE (L2)", "S9": "ASAP (US school)", "S10": "TwitterAAE",
         "S11": "Legal opinions", "S12": "AMI meetings"}
HEAD = {"D1@R3": "FD@R3", "D1@R4": "FD@R4", "D2@R4": "Bino@R4", "D2@R5": "Bino@R5",
        "D4": "HC3-RoB", "D5": "RADAR"}
lines = [r"\begin{tabular}{@{}l r " + "r" * len(inc) + r"@{}}", r"\toprule",
         r" & & \multicolumn{4}{c}{zero-shot} & \multicolumn{2}{c}{trained} \\",
         r"\cmidrule(lr){3-6}\cmidrule(l){7-8}",
         "Population & $\\log\\mathrm{PPL}$ & " + " & ".join(HEAD[i] for i in inc) + r" \\", r"\midrule"]
for s in fpr.index:
    row = []
    for i in inc:
        v = 100 * fpr.loc[s, i]
        cell = f"{v:.1f}"
        lo, _ = proportion_confint(round(fpr.loc[s, i] * n.loc[s, i]), n.loc[s, i], method="wilson")
        if s != "S6" and lo > 0.05:  # Wilson interval lies wholly above the 5% calibration rate
            cell = f"\\textbf{{{cell}}}"
        row.append(cell)
    lines.append(f"{s} {NAMES[s]} & {ppl[s]:.2f} & " + " & ".join(row) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "fpr.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

# ---- Forecast split by family ---------------------------------------------------
fc = reveal()
fam = fc["instance"].map(lambda i: "trained" if i in ("D3", "D4", "D5") else "zero_shot")
fc_mae = {"all": fc["abs_error"].mean(), "zero": fc.loc[fam == "zero_shot", "abs_error"].mean(),
          "trained": fc.loc[fam == "trained", "abs_error"].mean()}
fc_strat = fc.groupby("stratum")["abs_error"].mean()

# ---- Macros for every number quoted in the text ------------------------------------
H1, H2, H3, H4, H5, H6, B = R["H1"], R["H2"], R["H3"], R["H4"], R["H5"], R["H6"], R["bootstrap"]
pc = {d["instance"]: d["auroc"] for d in R["validity"]["positive_control"]}
lc = {d["instance"]: d["fpr_spread"] for d in R["validity"]["length_invariance"]}
same = [p for p in H6["pairs"] if p["same_family"]]
zs = [p["kappa"] for p in H6["pairs"] if p["a"].startswith("D1") or p["a"].startswith("D2")
      if p["b"].startswith("D1") or p["b"].startswith("D2")]
mixed = [p["kappa"] for p in H6["pairs"] if (p["a"][:2] in ("D1", "D2")) != (p["b"][:2] in ("D1", "D2"))]
raw_ratio = {k: np.std(list(v["per_transform_slope"].values()), ddof=1) / abs(v["slope"])
             for k, v in H5["per_instance"].items()}


def p_fmt(p: float) -> str:
    if p < 1e-4:
        return "<10^{-4}"
    if p < 1e-3:
        return f"={p:.4f}"
    return f"={p:.3f}" if p < 0.1 else f"={p:.2f}"


M = {
    "nDocs": f"{len(docs):,}".replace(",", "{,}"),
    # validity
    **{f"auc{k.replace('@', '').replace('D', 'D')}": f"{v:.3f}" for k, v in pc.items()},
    **{f"len{k.replace('@', '')}": f"{100 * v:.1f}" for k, v in lc.items()},
    "negFPR": f"{100 * R['validity']['negative_control'][0]['fpr']:.1f}",
    # H1
    "hOneCoef": f"{H1['coef_log_ppl']:.4f}", "hOneP": p_fmt(H1["p"]), "hOneDR": f"{H1['delta_r2']:.3f}",
    "hOneAtt": f"{H1['attenuation']:.2f}", "hOneDocWithout": f"{H1['doc_coef_without_stratum']:.2f}",
    "hOneDocWith": f"{H1['doc_coef_with_stratum']:.2f}", "hOneCells": str(H1["n_cells"]),
    "hOneCI": f"[{B['h1_coef_95ci'][0]:.4f}, {B['h1_coef_95ci'][1]:.4f}]",
    "hOnePts": f"{-100 * H1['coef_log_ppl']:.1f}",
    # H2
    "hTwoCoef": f"{H2['coef_log_ppl_ref']:.4f}", "hTwoP": p_fmt(H2["p"]), "hTwoShiftP": p_fmt(H2["profile_shift_p"]),
    "hTwoCI": f"[{B['h2_coef_95ci'][0]:.3f}, {B['h2_coef_95ci'][1]:.3f}]",
    "hTwoTau": f"{H2['kendall_tau']['R3~R4']:.2f}", "hTwoCells": str(H2["n_cells"]),
    "hTwoSipP": p_fmt(H2["robustness_source_in_pretraining"]["p_sip"]),
    # H3
    "hThreeCoef": f"{H3['coef_log_ppl']:.4f}", "hThreeP": p_fmt(H3["p"]), "hThreeDR": f"{H3['delta_r2']:.3f}",
    "hThreeDocWith": f"{H3['doc_coef_with_stratum']:.2f}", "hThreeShiftP": p_fmt(H3["trained_profile_shift_p"]),
    # H4 + forecast
    "hFourMAE": f"{100 * H4['mae']:.1f}", "hFourBase": f"{100 * H4['mae_baseline']:.1f}", "hFourSkill": f"{H4['skill']:.2f}",
    "fcMAE": f"{100 * fc_mae['all']:.1f}", "fcZero": f"{100 * fc_mae['zero']:.1f}", "fcTrained": f"{100 * fc_mae['trained']:.1f}",
    **{f"fc{s}": f"{100 * v:.1f}" for s, v in fc_strat.items()},
    # H5
    "hFiveN": str(H5["per_instance"]["D1@R3"]["n"]),
    "hFiveConf": str(len(H5["overall"]["confirmed_instances"])), "hFiveOf": str(H5["overall"]["n_instances"]),
    **{f"hFiveSlope{k.replace('@', '')}": f"{v['slope']:.2f}" for k, v in H5["per_instance"].items()},
    **{f"hFiveRatio{k.replace('@', '')}": f"{v['sd_ratio']:.2f}" for k, v in H5["per_instance"].items()},
    **{f"hFiveP{k.replace('@', '')}": p_fmt(v["p_holm"]) for k, v in H5["per_instance"].items()},
    # sensitivity: spread from the six separate least-squares slopes instead of the mixed model
    **{f"hFiveRaw{k.replace('@', '')}": f"{raw_ratio[k]:.4f}" for k in raw_ratio},
    "hFiveRawConf": str(sum(1 for k, v in H5["per_instance"].items()
                            if v["slope"] < 0 and v["p_holm"] < 0.05 and raw_ratio[k] <= 0.25)),
    # H6
    "hSixGap": f"{H6['kappa_same_minus_diff']:.2f}", "hSixP": p_fmt(H6["p"]), "hSixSame": str(len(same)),
    "hSixPairs": str(len(H6["pairs"])), "kZeroLo": f"{min(zs):.2f}", "kZeroHi": f"{max(zs):.2f}",
    "kMixLo": f"{min(mixed):.2f}", "kMixHi": f"{max(mixed):.2f}",
    # exploratory
    "xHTwoCoef": f"{EXP['H2']['coef_log_ppl_ref']:.3f}", "xHTwoP": p_fmt(EXP["H2"]["p"]),
    "xMatchedShiftP": p_fmt(EXP["H2_matched_R2_R3"]["profile_shift_p"]),
    "xTauPile": f"{EXP['H2']['kendall_tau']['R2~R3']:.2f}", "xTauMin": f"{min(EXP['H2']['kendall_tau'].values()):.2f}",
    "xHSixGap": f"{EXP['H6']['kappa_same_minus_diff']:.2f}", "xHSixP": p_fmt(EXP["H6"]["p"]),
    "xHFiveConf": str(len(EXP["H5"]["overall"]["confirmed_instances"])), "xHFiveOf": str(EXP["H5"]["overall"]["n_instances"]),
    "xHOneDR": f"{EXP['H1']['delta_r2']:.3f}",
}
# a few FPRs quoted in the text
for s in fpr.index:
    for i in inc:
        M[f"fpr{s}{i.replace('@', '')}"] = f"{100 * fpr.loc[s, i]:.1f}"
for s in ("S3", "S4", "S7", "S8"):
    for r in ("R3", "R4"):
        M[f"ppl{s}{r}"] = f"{cells[(cells.detector == 'D1') & (cells.reference == r) & (cells.stratum == s)]['log_ppl_ref'].iloc[0]:.2f}"


# how closely the independent predictor tracks each reference's log-PPL:
# across documents, and for the changes the H5 edits cause
pp = scores[scores["detector"].isin(["PPL", "PPL_indep"])].assign(
    key=lambda x: x["reference"].fillna("indep"))
pp = pp.pivot_table(index=["item_id", "group"], columns="key", values="score").reset_index()
dpp = pp[~pp["group"].isin(["control", "variants"])].set_index("item_id")
vpp = pp[pp["group"] == "variants"].merge(
    pd.read_parquet(paths.variants_file(), columns=["variant_id", "doc_id"]),
    left_on="item_id", right_on="variant_id")
refs = ["R1", "R2", "R3", "R4", "R5"]
r_doc = [dpp["indep"].corr(dpp[r]) for r in refs]
delta = {k: vpp[k].to_numpy() - dpp.loc[vpp["doc_id"], k].to_numpy() for k in ["indep"] + refs}
r_edit = [np.corrcoef(delta["indep"], delta[r])[0, 1] for r in refs]
M["corrDocLo"], M["corrDocHi"] = f"{min(r_doc):.2f}", f"{max(r_doc):.2f}"
M["corrEditLo"], M["corrEditHi"] = f"{min(r_edit):.2f}", f"{max(r_edit):.2f}"

# exploratory FPRs quoted in Section 5 (all ten instances, gate not applied)
XF = EXP["fpr_table"]  # instance -> stratum -> fpr
for s in ("S3", "S4", "S11"):
    for r in ("R1", "R2", "R3", "R4", "R5"):
        M[f"xfpr{s}{r}"] = f"{100 * XF[f'D1@{r}'][s]:.1f}"
M["xfprSTenDThree"] = f"{100 * XF['D3']['S10']:.1f}"
M["xHFiveSlopeDThree"] = f"{EXP['H5']['per_instance']['D3']['slope']:.2f}"

# audit tool: width of the conformal interval around a 5% point estimate
AUD = json.loads((paths.results() / "audit_model.json").read_text())
q = AUD["conformal_q_logit"]
lg = np.log(0.05 / 0.95)
M["audQ"] = f"{q:.2f}"
M["audLo"] = f"{100 / (1 + np.exp(-(lg - q))):.1f}"
M["audHi"] = f"{100 / (1 + np.exp(-(lg + q))):.0f}"

# registered follow-up E (https://osf.io/rd6jg): which score component carries the reference swap
FU = json.loads((paths.results() / "followup_entropy.json").read_text())
E1, E2, E3 = FU["E1"], FU["E2"], FU["E3_secondary"]
ci2 = lambda c: f"[{c[0]:.2f}, {c[1]:.2f}]"
M["eOneVerdict"] = E1["verdict"].lower()
M["eOneH"] = f"{E1['share']['entropy']:.2f}"
M["eOneHCI"] = ci2(E1["entropy_share_ci"])
M["eOneNLL"] = f"{E1['share']['nll']:.2f}"
M["eOneRevH"] = f"{E1['reverse_R4_to_R3']['share']['entropy']:.2f}"
M["eOneDist"] = f"{100 * E1['total_distance']:.1f}"
M["eTwoDist"] = f"{100 * E2['total_distance']:.1f}"
M["eTwoVerdict"] = E2["verdict"].lower()
M["eTwoX"] = f"{E2['share']['xent']:.2f}"
M["eTwoXCI"] = ci2(E2["xent_share_ci"])
M["eTwoDiffCI"] = ci2(E2["xent_minus_nll_ci"])
M["eThreeVerdict"] = E3["verdict"].lower()
M["eThreeCount"] = str(E3["entropy_largest_in"])
_m, _e = f"{max(FU['score_reproduction_max_abs_diff'].values()):.0e}".split("e")
M["eReproMax"] = f"${_m}\\times 10^{{{int(_e)}}}$"
DM = FU["descriptive_means"]
POPS12 = [f"S{i}" for i in range(1, 13)]
dN = np.array([DM["R4"]["nll"][s] - DM["R3"]["nll"][s] for s in POPS12])
dH = np.array([DM["R4"]["entropy"][s] - DM["R3"]["entropy"][s] for s in POPS12])
M["eCorrNH"] = f"{np.corrcoef(dN, dH)[0, 1]:.2f}"
for s, a, b in zip(POPS12, dN, dH):
    M[f"eGap{s}"] = f"{b - a:+.3f}"

# negative numbers and intervals need a true minus sign in running text too
M = {k: (f"\\ensuremath{{{v}}}" if v.startswith(("-", "[")) else v) for k, v in M.items()}


def name(k: str) -> str:  # LaTeX macro names may contain letters only
    digits = "Zero One Two Three Four Five Six Seven Eight Nine".split()
    return "".join(digits[int(c)] if c.isdigit() else c for c in k)


(OUT / "numbers.tex").write_text(
    "% generated by scripts/paper_tables.py -- do not edit by hand\n"
    + "\n".join(f"\\newcommand{{\\{name(k)}}}{{{v}}}" for k, v in M.items()) + "\n",
    encoding="utf-8")

# ---- Verdict table ---------------------------------------------------------------
V = [("H1", "One perplexity explains FPRs", H1["verdict"],
      f"$\\beta={H1['coef_log_ppl']:.3f}$, $\\Delta R^2={H1['delta_r2']:.3f}$ (bar 0.10)"),
     ("H2", "Reference swap changes who is flagged", H2["verdict"],
      f"shift $p{p_fmt(H2['profile_shift_p'])}$; $\\beta={H2['coef_log_ppl_ref']:.3f}$, $p{p_fmt(H2['p'])}$"),
     ("H2$_{\\text{Pile}}$", "Same-data pair R2/R3", "NOT ESTIMABLE", "R2 failed the positive control"),
     ("H3", "Holds for trained classifiers", H3["verdict"],
      f"$\\beta={H3['coef_log_ppl']:.3f}$, $p{p_fmt(H3['p'])}$; shift $p{p_fmt(H3['trained_profile_shift_p'])}$"),
     ("H4", "Perplexity forecasts FPR", H4["verdict"],
      f"MAE {100 * H4['mae']:.1f} pts, skill {H4['skill']:.2f} (bar 0.25)"),
     ("H5", "Edits move scores with perplexity", H5["overall"]["verdict"],
      f"{len(H5['overall']['confirmed_instances'])}/{H5['overall']['n_instances']} instances"),
     ("H6", "Same reference $\\Rightarrow$ agreement", H6["verdict"],
      f"$\\Delta\\kappa={H6['kappa_same_minus_diff']:.2f}$, $p{p_fmt(H6['p'])}$")]
lines = [r"\begin{tabular}{@{}l l l l@{}}", r"\toprule", r" & Claim & Verdict & Key result \\", r"\midrule"]
lines += [f"{a} & {b} & {c.capitalize()} & {d} \\\\" for a, b, c, d in V]
lines += [r"\midrule", f"\\multicolumn{{4}}{{@{{}}l}}{{Registered decision (H1, H2, H5): \\textbf{{{R['decision']['scorer_distance_hypothesis'].lower()}}}}} \\\\",
          r"\bottomrule", r"\end{tabular}"]
(OUT / "verdicts.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

# ---- Appendix tables -----------------------------------------------------------------
# same abbreviations as the FPR table (FD = Fast-DetectGPT, Bino = Binoculars)
LONG = {"D1@R1": "FD@R1", "D1@R2": "FD@R2", "D1@R3": "FD@R3", "D1@R4": "FD@R4", "D1@R5": "FD@R5",
        "D2@R4": "Bino@R4", "D2@R5": "Bino@R5", "D3": "GPT2-RoB", "D4": "HC3-RoB", "D5": "RADAR"}
neg = {d["instance"]: d["fpr"] for d in R["validity"]["negative_control"]}
lines = [r"\begin{tabular}{@{}l r r r l@{}}", r"\toprule",
         r"Instance & AUROC & S6 FPR & Length spread & Status \\", r"\midrule"]
for d in R["validity"]["positive_control"]:
    i = d["instance"]
    lines.append(f"{LONG[i]} & {d['auroc']:.3f} & "
                 + (f"{100 * neg[i]:.1f} & {100 * lc[i]:.1f}" if i in inc else "-- & --")
                 + f" & {'included' if i in inc else 'excluded'} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "validity.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

lines = [r"\begin{tabular}{@{}l r r " + "r" * 6 + r" l@{}}", r"\toprule",
         r"Instance & Slope & SD ratio & T1 & T2 & T3 & T4 & T5 & T6 & Verdict \\", r"\midrule"]
for i, v in H5["per_instance"].items():
    t = v["per_transform_slope"]
    lines.append(f"{LONG[i]} & ${v['slope']:.2f}$ & {v['sd_ratio']:.2f} & "
                 + " & ".join(f"${t[k]:.2f}$" for k in ("T1", "T2", "T3", "T4", "T5", "T6"))
                 + f" & {v['verdict'].capitalize()} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "h5.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

lines = [r"\begin{tabular}{@{}l l r r@{}}", r"\toprule",
         r"Population & Instance & Forecast & Observed \\", r"\midrule"]
for _, row in fc.iterrows():
    lines.append(f"{row['stratum']} {NAMES[row['stratum']]} & {LONG[row['instance']]} & "
                 f"{100 * row['fpr_pred']:.1f} & {100 * row['fpr']:.1f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "forecast.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

# documents per length band, from the corpus build manifest
MAN = json.loads((ROOT / "data" / "processed" / "manifest.json").read_text())["strata"]
edges = MAN["S1"]["length_matching"]["bin_edges"]
lines = [r"\begin{tabular}{@{}l " + "r" * (len(edges) - 1) + r"@{}}", r"\toprule",
         # bands are [lo, hi) in tokens, so integer bounds are ceil(lo) .. ceil(hi) - 1
         "Population & " + " & ".join(f"{int(np.ceil(edges[b]))}--{int(np.ceil(edges[b + 1])) - 1}"
                                      for b in range(len(edges) - 1))
         + r" \\", r"\midrule"]
for s in sorted(MAN, key=lambda k: int(k[1:])):
    per = MAN[s]["length_matching"]["per_bin"]
    lines.append(f"{s} & " + " & ".join(str(per[str(b)]) for b in range(len(edges) - 1)) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "bands.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

# follow-up E: Shapley shares of each score component (registered verdict rules in rd6jg)
def share_row(label, e, verdict):
    sh, single = e["share"], e["single_swap_share"]
    cell = lambda d, c: f"${d[c]:.2f}$" if c in d else "--"
    return (f"{label} & " + " & ".join(cell(sh, c) for c in C4) + " & "
            + " & ".join(cell(single, c) for c in C4) + f" & {verdict} \\\\")


C4 = ("nll", "entropy", "var", "n")
E2_AS_4 = {k: {"nll": E2[k]["nll"], "entropy": E2[k]["xent"]} for k in ("share", "single_swap_share")}
lines = [r"\begin{tabular}{@{}l rrrr rrrr l@{}}", r"\toprule",
         r" & \multicolumn{4}{c}{Shapley share} & \multicolumn{4}{c}{Single-swap share} & \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r"Swap & NLL & $H$ & $V$ & $n$ & NLL & $H$ & $V$ & $n$ & Verdict \\", r"\midrule",
         share_row(r"E1: FD R3$\to$R4", E1, E1["verdict"].capitalize()),
         share_row(r"\quad reverse R4$\to$R3", E1["reverse_R4_to_R3"], "descriptive"),
         share_row(r"E2: Bino R4$\to$R5$^\ast$", E2_AS_4, E2["verdict"].capitalize()),
         r"\midrule"]
for pair, v in E3["pairs"].items():
    a, b = pair.split("->")
    lines.append(share_row(f"E3: FD {a}$\\to${b}", v, ""))
lines += [r"\midrule",
          f"\\multicolumn{{10}}{{@{{}}l}}{{E3: $H$ largest in {E3['entropy_largest_in']} of {E3['of']} "
          f"pairs (bar 6): {E3['verdict'].capitalize()}}} \\\\",
          r"\bottomrule", r"\end{tabular}"]
(OUT / "followup.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

# per-population change in mean NLL, H and their gap, R3 -> R4 (registered descriptive output)
lines = [r"\begin{tabular}{@{}l rrr@{}}", r"\toprule",
         r"Population & $\Delta$NLL & $\Delta H$ & $\Delta(H-\mathrm{NLL})$ \\", r"\midrule"]
lines += [f"{s} {NAMES[s]} & ${a:+.3f}$ & ${b:+.3f}$ & ${b - a:+.3f}$ \\\\" for s, a, b in zip(POPS12, dN, dH)]
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "followup_gap.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

print("wrote", sorted(p.name for p in OUT.iterdir()))
print("forecast MAE all/zero-shot/trained:", {k: round(100 * v, 1) for k, v in fc_mae.items()})
