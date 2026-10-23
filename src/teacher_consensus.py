"""
teacher_consensus.py -- combine the two report labellers into one better teacher.

WHY THIS FILE EXISTS
Vocabulary first, because the rest of the file is written in it. The TEACHER is the table of
soft labels read out of the radiology reports. The STUDENT is the vision model trained to
reproduce those labels from pixels alone -- there are no reports at submission time -- and
DISTILLATION is that copying. GOLD is the 58 of 4,407 studies whose twelve labels were read
off the IMAGES instead of the text; they are the only trustworthy instrument in the project,
and at n=58 they are also a noisy one.

The student has caught the teacher: 0.8612 gold vs the teacher's 0.861. Every distilled run
from here is capped by the label quality, not the vision model, so the cheapest remaining
gain is in label arithmetic -- no GPU, no new data, no new prompt.

We hold two labellings of the same 4,407 reports:
    report_labels_llm.csv    Qwen3-32B reading each report      gold macro 0.861
    report_labels_ours.csv   a regex/rule extractor             gold macro 0.788
(report_labels_calibrated.csv is NOT a third source -- see the note in --report table 1.)

THE ONE THING WORTH COMBINING, AND WHY IT IS NOT "ENSEMBLING"
The LLM's probability column takes exactly SEVEN distinct values -- it is the severity ladder
{0.03, 0.15, 0.35, 0.45, 0.62, 0.75, 0.95} and nothing else. So its score is really a
7-bucket ordinal, and every pair of studies landing in the same bucket is a TIE. AUC scores a
tie as a coin flip. Measured on the 58 gold studies, 12.9% of all (positive, negative) pairs
are ties the LLM cannot order at all -- 28.1% on Effusion, where "mild" swallows a third of
the corpus.

That tie mass is free real estate. Any tie-breaker better than a coin flip must raise AUC in
expectation, and it cannot lower the LLM's own ordering, because breaking a tie in the right
direction gains 0.5 and in the wrong direction loses 0.5 from a baseline of exactly 0.5. The
rule extractor, restricted to the pairs the LLM ties, scores 0.600 -- weak, but decisively
not a coin flip.

    predicted gain = tie_mass x (tie_break_auc - 0.5) = 0.129 x 0.100 = +0.0129
    measured  gain =                                                   +0.0112

So the rule is: KEEP THE LLM'S ORDERING EXACTLY, AND LET THE RULE EXTRACTOR ORDER ONLY
WITHIN ITS BUCKETS.

    p_consensus = p_llm + EPS * (p_ours - 0.5),   EPS = 0.05

EPS is not a knob to tune. Two constraints bound it and nothing else selects it:

  (a) NO REORDERING. The perturbation must be too small to move any study across a rung of
      the LLM's ladder. The smallest gap in that ladder is 0.10 (0.35 -> 0.45) and the
      largest perturbation is EPS x max|p_ours - 0.5| = EPS x 0.47, applied from both sides,
      so the bound is EPS < 0.10 / (2 x 0.47) = 0.106.
  (b) STILL A PROBABILITY. distill_loss feeds these to binary_cross_entropy_with_logits as
      soft targets. The lowest rung sits at 0.03, so the bound is EPS <= 0.03 / 0.47 = 0.064.
      (The self-test found this the hard way: EPS = 0.08 produced -0.0076.)

EPS = 0.05 is a round number comfortably inside both. Any EPS in (0, 0.064] yields the SAME
ordering and therefore the IDENTICAL AUC -- verified in the self-test across three decades of
EPS. Fitted parameters on the 58 gold: ZERO. That matters: fitting on 58 gold studies is what
collapsed the per-target calibration experiment (0.777 -> 0.614 leave-one-out).

(p_ours - 0.5) is centred rather than raw so the perturbation adds no systematic drift to the
soft targets -- distill_loss consumes these values directly as soft-target BCE, and the
corpus mean of each rung is left where the LLM put it.

WHAT IS DELIBERATELY ABSENT
  * NO symmetric blending. mean(prob) scores 0.8691 and rank-average 0.8596, both worse than
    the tie-break's 0.8718, and both with a paired delta CI straddling zero. Averaging a
    0.861 labeller with a 0.788 one drags the good one down; it is the wrong prior when the
    sources are known to be unequal. Measured, printed by --report, and rejected.
  * NO coverage union. Taking the rule extractor's assertion where the LLM is not_mentioned
    raises coverage by 0.2 pp -- 84.1% to 84.3% -- because inside the LLM's not_mentioned
    bucket the rule extractor is ALSO silent 98.6% of the time. Two extra branches for two
    tenths of a percent, and it moved gold AUC by -0.0001. Rejected.
  * NO masking on disagreement. Where the two labellers point in opposite directions (3.4% of
    cells) the LLM's direction is still right 69% of the time on gold, against 80% where they
    agree. Lower quality, yes -- but 69% is signal, and a masked cell contributes no gradient
    at all. Masking would delete 1,806 cells to remove a 31% error rate. Rejected.
  * NO per-target choice of rule. Twelve targets x "which variant wins here" is twelve
    parameters fitted on 58 studies. One rule, applied identically everywhere.
  * NO re-derivation of the state columns. The mask is the LLM's, verbatim. The LLM decides
    whether a report ASSERTED anything (it reads nine languages and handles negation; a regex
    does not), and the rule extractor only ranks within what the LLM already asserted. This
    also makes the change auditable: the consensus differs from report_labels_llm.csv in the
    probability columns and NOWHERE ELSE.

USAGE
    python teacher_consensus.py                # write report_labels_consensus.csv
    python teacher_consensus.py --report       # the full comparison tables, no file written
    python teacher_consensus.py --self-test    # invariants; CPU; ~30 s

    train.py --labels report_labels_consensus.csv     # no edit to any locked file needed
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# Copied, not imported, from model.KneeStudies. Importing model.py drags torch and timm into
# what is pure label arithmetic, and re-imports a file that a live training run's spawned
# dataloader workers are also importing. The self-test greps model.py's SOURCE to prove this
# set still matches, which catches drift without loading it.
NOT_ASSERTED = {"silent", "not_mentioned"}

# Qwen's severity ladder, from llm_label_reports.SEVERITY_SCORE. The tie-break is provably
# safe only because this is a small closed set with a known minimum gap.
LLM_LADDER = (0.03, 0.15, 0.35, 0.45, 0.62, 0.75, 0.95)
EPS = 0.05


# ============================================================================= the rule
def consensus_prob(p_llm: np.ndarray, p_ours: np.ndarray, eps: float = EPS) -> np.ndarray:
    """The whole method. Everything else in this file measures it."""
    return p_llm + eps * (p_ours - 0.5)


def max_safe_eps(p_ours: np.ndarray, ladder=LLM_LADDER) -> float:
    """The binding upper bound on eps -- the tighter of the two constraints above.

    (a) order:       two adjacent rungs close by at most eps*max|p_ours-0.5| from each side,
                     so 2*eps*max_dev < min_gap.
    (b) probability: the lowest rung must not go below 0 nor the highest above 1, so
                     eps*max_dev <= min(min_rung, 1 - max_rung).

    Both are computed from the OBSERVED p_ours range and the ladder, never assumed. If either
    labeller's score map ever widens, this shrinks and the self-test fires -- instead of the
    tie-break silently reordering the LLM, or emitting a negative BCE target.
    """
    dev = float(np.max(np.abs(p_ours - 0.5)))
    lad = np.asarray(ladder)
    order = float(np.min(np.diff(lad))) / (2.0 * dev)
    prob = min(float(lad.min()), 1.0 - float(lad.max())) / dev
    return min(order, prob)


# ============================================================================= data
def load_sources(ours_csv: str = "report_labels_ours.csv"):
    """The two labellings, row-aligned on StudyInstanceUID (llm reindexed onto ours).

    `ours_csv` exists so a REBUILT rule extractor can be scored through the same consensus
    without overwriting the shipped one -- the tie-break is only as good as the ordering it
    is handed, so a change to the extractor has to be measured here, not just on its own.
    """
    llm = pd.read_csv(ROOT / "report_labels_llm.csv").set_index("StudyInstanceUID")
    ours = pd.read_csv(ROOT / ours_csv).set_index("StudyInstanceUID")
    assert set(llm.index) == set(ours.index), "the two labellers cover different studies"
    return llm.loc[ours.index], ours


def gold_frame():
    """The 58 studies with real labels. The only trustworthy instrument we have."""
    tr = pd.read_csv(ROOT / "train.csv")
    g = tr[tr[TARGETS].notna().all(axis=1)]
    return g[["StudyInstanceUID"] + TARGETS].reset_index(drop=True)


def build_consensus(llm: pd.DataFrame, ours: pd.DataFrame, eps: float = EPS) -> pd.DataFrame:
    """-> StudyInstanceUID + 12 probabilities + 12 '<target>__state', the schema train.py eats.

    The state columns are the LLM's, unchanged: this file changes the ORDER within the
    teacher's buckets, never which cells the teacher speaks about.
    """
    out = pd.DataFrame({"StudyInstanceUID": ours.index.values})
    for t in TARGETS:
        out[t] = consensus_prob(llm[t].values.astype(float),
                               ours[t].values.astype(float), eps)
    for t in TARGETS:
        out[t + "__state"] = llm[t + "__state"].values
    return out


# ============================================================================= measurement
def _cmp(s: np.ndarray) -> np.ndarray:
    """n x n pairwise 'i beats j' matrix, ties counted as 0.5 -- exactly how AUC sees them."""
    d = s[:, None] - s[None, :]
    return (d > 0).astype(float) + 0.5 * (d == 0)


def boot_macro(scores: np.ndarray, Y: np.ndarray, counts: np.ndarray):
    """(per-target AUC, bootstrap macro distribution). Resamples STUDIES, not cells.

    A BOOTSTRAP here means: draw 58 studies WITH replacement from the 58 we have, 2,000
    times, recomputing the macro on each draw. The spread of those 2,000 values is how far
    the number would move had a different 58 studies been labelled -- the only honest way to
    ask whether a +0.011 is real at this sample size.

    Written as the quadratic form a'Sb / (sum a)(sum b) over the comparison matrix so that
    2,000 resamples x 12 targets x a dozen variants runs in seconds. Calling roc_auc_score
    inside that loop overran the 3-minute self-test budget on its own.

    WHY THAT FORM *IS* THE AUC, not an approximation of it: AUC is by definition the chance
    that a random positive outranks a random negative, a tie counting half. S[i,k] holds
    exactly that outcome for one pair, so summing S over the (positive, negative) pairs and
    dividing by how many there are gives the AUC. A bootstrap draw changes only how many
    times each study appears, so multiplying the two indicator vectors by `counts` reweights
    the same S rather than rebuilding it -- one n x n matrix per target serves all 2,000
    draws, which is the whole speed-up.

    `ok` discards draws in which a target happened to come out all-positive or all-negative:
    AUC is undefined there, and such a draw contributes to neither the running sum nor its
    divisor `cnt`. That is why the macro divides by cnt and not by 12.
    """
    nb = len(counts)
    acc, cnt, point = np.zeros(nb), np.zeros(nb), []
    for j in range(Y.shape[1]):
        y, S = Y[:, j], _cmp(scores[:, j])
        point.append(float(y @ S @ (1 - y) / (y.sum() * (1 - y).sum())))
        a, b = counts * y, counts * (1 - y)
        na, nb_ = a.sum(1), b.sum(1)
        ok = (na > 0) & (nb_ > 0)
        num = np.einsum("bi,ij,bj->b", a, S, b)
        acc[ok] += num[ok] / (na[ok] * nb_[ok])
        cnt[ok] += 1
    return np.array(point), acc / np.maximum(cnt, 1)


def _boot_counts(n, nb=2000, seed=0):
    """(nb, n) multiplicities: row b says how many times each study appears in draw b.

    Counts rather than index lists, because that is the form boot_macro can reuse a single
    comparison matrix for. The seed is fixed and the result is computed ONCE and handed to
    every variant, so table 3's paired deltas compare variants on the identical 2,000
    resamples. A paired delta on shared draws is far tighter than two marginal CIs, which at
    n=58 are ~0.07 wide and overlap everything.
    """
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(nb)])


# Direction of an assertion, per each labeller's OWN documented vocabulary. hedged /
# uncertain / degen_only are genuinely equivocal and are given NO direction, so they can
# never count as a disagreement -- calling them positive or negative would be inventing an
# opinion neither labeller expressed.
POS_LLM = {"mild", "moderate", "severe", "present_ungraded"}
NEG_LLM = {"absent"}
POS_OURS = {"asserted", "mild", "mild(subchondral)", "moderate", "severe"}
NEG_OURS = {"negated"}


def _direction(states, pos, neg):
    return np.where(np.isin(states, list(pos)), 1,
                    np.where(np.isin(states, list(neg)), -1, 0))


def report(ours_csv: str = "report_labels_ours.csv"):
    """Everything the decision rests on. Nothing here is fitted; it is all confirmation."""
    from scipy.stats import rankdata, spearmanr

    llm, ours = load_sources(ours_csv)
    gold = gold_frame()
    # gi maps the 58 gold studies onto row positions in the full 4,407-row arrays, so the
    # corpus-wide quantities (coverage, direction clashes) and the gold-only ones (AUC) can
    # be read off the same arrays. Get this wrong and gold labels are scored against the
    # wrong studies while still printing an entirely plausible number, so it is derived once.
    gi = pd.Index(ours.index).get_indexer(gold.StudyInstanceUID.values)
    Y = np.stack([gold[t].values.astype(float) for t in TARGETS], 1)
    PL = llm[TARGETS].values.astype(float)
    PO = ours[TARGETS].values.astype(float)
    SL = llm[[t + "__state" for t in TARGETS]].values.astype(str)
    SO = ours[[t + "__state" for t in TARGETS]].values.astype(str)
    na_l, na_o = SL == "not_mentioned", SO == "silent"
    dl, do = _direction(SL, POS_LLM, NEG_LLM), _direction(SO, POS_OURS, NEG_OURS)
    conflict = (~na_l) & (~na_o) & (dl != 0) & (do != 0) & (dl != do)

    # ---- 1. the three label sets against gold and against each other
    cal = pd.read_csv(ROOT / "report_labels_calibrated.csv").set_index("StudyInstanceUID")
    cal = cal.loc[ours.index]
    print("=" * 104)
    print("1. THE LABEL SETS, PER TARGET     (AUC on the 58 gold; coverage over all 4,407)")
    print("=" * 104)
    print(f"{'target':<18}{'n_pos':>6}{'AUC llm':>9}{'AUC ours':>10}{'AUC cal+':>10}"
          f"{'cov llm':>9}{'cov ours':>10}{'spearman':>10}{'dir clash':>12}")
    acc = np.zeros(7)
    for j, t in enumerate(TARGETS):
        y = Y[:, j].astype(int)
        both = (~na_l[:, j]) & (~na_o[:, j])
        row = (roc_auc_score(y, PL[gi, j]), roc_auc_score(y, PO[gi, j]),
               roc_auc_score(y, cal[t].values[gi]), (~na_l[:, j]).mean(),
               (~na_o[:, j]).mean(), spearmanr(PL[both, j], PO[both, j]).statistic,
               conflict[:, j].mean())
        acc += row
        print(f"{t:<18}{y.sum():>6}{row[0]:>9.3f}{row[1]:>10.3f}{row[2]:>10.3f}"
              f"{row[3]:>9.3f}{row[4]:>10.3f}{row[5]:>10.3f}{100 * row[6]:>11.1f}%")
    a = acc / len(TARGETS)
    print(f"{'MACRO':<18}{'':>6}{a[0]:>9.3f}{a[1]:>10.3f}{a[2]:>10.3f}"
          f"{a[3]:>9.3f}{a[4]:>10.3f}{a[5]:>10.3f}{100 * a[6]:>11.1f}%")
    print("\n  'dir clash' = share of cells where both labellers assert and they point in")
    print("  OPPOSITE directions. 'spearman' is over cells where both assert.")
    print("\n  + report_labels_calibrated.csv is NOT an independent third labeller, and the")
    print("    0.789 above is NOT comparable to the other two columns, for two reasons:")
    print("      (a) it is ours' state column pushed through a map FITTED ON ALL 58 GOLD, so")
    print("          that number is in-sample. calibrate_report_labels.py's own leave-one-out")
    print("          check is the honest read and it does not beat the raw extractor.")
    print("      (b) it was written 2026-08-12 14:39, six hours BEFORE the")
    print("          report_labels_ours.csv it is derived from (20:58). It is stale: its")
    print("          scores no longer agree with ours' current states on 5 of 12 targets.")
    print("    It is excluded from the consensus. Folding a gold-fitted file into the teacher")
    print("    and then judging the teacher on gold would corrupt the only instrument we have.")

    # ---- 2. where the LLM ties, and whether the rule extractor can break those ties
    print("\n" + "=" * 104)
    print("2. TIE MASS -- (pos,neg) gold pairs the LLM's 7-value ladder cannot order at all")
    print("=" * 104)
    print(f"{'target':<18}{'pairs':>8}{'tied':>7}{'tie %':>8}{'ours AUC on the tied pairs':>28}")
    tie_f, tb_auc = [], []
    for j, t in enumerate(TARGETS):
        y = Y[:, j]
        P, N = np.where(y == 1)[0], np.where(y == 0)[0]
        npair = len(P) * len(N)
        tie = win = 0
        pl, po = PL[gi, j], PO[gi, j]
        # For each gold positive, find the gold negatives the LLM scored EXACTLY equal --
        # those are the pairs AUC is forced to call 0.5. `win` then asks how the rule
        # extractor orders that same set. Anything above 0.5 is information the seven-rung
        # ladder threw away and this file can put back.
        for i in P:
            k = N[pl[i] == pl[N]]
            tie += len(k)
            z = po[i] - po[k]
            win += (z > 0).sum() + 0.5 * (z == 0).sum()
        tie_f.append(tie / npair)
        tb_auc.append(win / tie if tie else np.nan)
        print(f"{t:<18}{npair:>8}{tie:>7}{100 * tie / npair:>7.1f}%"
              f"{(win / tie if tie else np.nan):>28.3f}")
    print(f"{'MACRO':<18}{'':>8}{'':>7}{100 * np.mean(tie_f):>7.1f}%"
          f"{np.nanmean(tb_auc):>28.3f}")
    print(f"\n  predicted gain = tie_mass x (tie-break AUC - 0.5) = {np.mean(tie_f):.3f} x "
          f"{np.nanmean(tb_auc) - 0.5:.3f} = {np.mean(tie_f) * (np.nanmean(tb_auc) - 0.5):+.4f}")
    print("  This prediction follows from the ladder's shape, not from trying combinations.")
    print("  Table 3 confirms it; table 3 is not the search that produced it.")

    # ---- 3. every variant, including the ones that lost
    print("\n" + "=" * 104)
    print("3. VARIANTS ON GOLD   (2,000-resample bootstrap over the 58 studies. Read the")
    print("   PAIRED delta CI -- the marginal CIs are ~0.07 wide and every one of them")
    print("   overlaps every other, so marginal CIs cannot separate anything here.)")
    print("=" * 104)
    rk = lambda x: np.apply_along_axis(lambda c: rankdata(c) / len(c), 0, x)
    V = {"llm (Qwen3-32B)": PL, "ours (rule extractor)": PO,
         "mean of probabilities": 0.5 * (PL + PO),
         "rank-average": 0.5 * (rk(PL) + rk(PO)),
         "rank-average 3:1 -> llm": 0.75 * rk(PL) + 0.25 * rk(PO)}
    for e in (0.001, 0.01, 0.05, 0.063):
        V[f"tie-break EPS={e:g}"] = consensus_prob(PL, PO, e)
    tb = consensus_prob(PL, PO)
    un = tb.copy()
    sw = na_l & (~na_o) & (do != 0)
    un[sw] = consensus_prob(PO, PL)[sw]
    V["tie-break + coverage union"] = un
    sh = tb.copy()
    sh[conflict] = 0.5 + EPS * (PO[conflict] - 0.5)
    V["tie-break + clash -> 0.5"] = sh

    counts = _boot_counts(len(gold))
    p_llm, b_llm = boot_macro(PL[gi], Y, counts)
    print(f"{'variant':<28}{'macro':>8}{'95% CI':>19}{'d vs llm':>10}"
          f"{'delta 95% CI':>20}{'P(d>0)':>8}")
    for k, P in V.items():
        p, b = boot_macro(P[gi], Y, counts)
        d = b - b_llm
        print(f"{k:<28}{p.mean():>8.4f}  [{np.percentile(b, 2.5):.4f},"
              f"{np.percentile(b, 97.5):.4f}]{p.mean() - p_llm.mean():>+10.4f}  "
              f"[{np.percentile(d, 2.5):+.4f},{np.percentile(d, 97.5):+.4f}]"
              f"{(d > 0).mean():>8.3f}")
    print("\n  The four EPS rows are identical to every printed digit because AUC reads only")
    print("  ORDER, and every EPS in (0, 0.064] produces the same order. EPS is a constraint,")
    print("  not a knob, and there is nothing here to tune on 58 studies.")

    # ---- 4. per target, chosen rule vs the LLM
    p_tb, b_tb = boot_macro(tb[gi], Y, counts)
    print("\n" + "=" * 104)
    print("4. CHOSEN RULE, PER TARGET")
    print("=" * 104)
    print(f"{'target':<18}{'llm':>8}{'consensus':>11}{'delta':>9}")
    for j, t in enumerate(TARGETS):
        print(f"{t:<18}{p_llm[j]:>8.3f}{p_tb[j]:>11.3f}{p_tb[j] - p_llm[j]:>+9.3f}")
    print(f"{'MACRO':<18}{p_llm.mean():>8.3f}{p_tb.mean():>11.3f}"
          f"{p_tb.mean() - p_llm.mean():>+9.3f}")

    # ---- 5. does the gain reach the student, or only the scoreboard
    print("\n" + "=" * 104)
    print("5. DOES THE GAIN REACH THE STUDENT?   A tie broken between two cells the teacher")
    print("   masks out is worth AUC on paper and exactly nothing in training.")
    print("=" * 104)
    g_all, g_train = [], []
    for j, t in enumerate(TARGETS):
        y = Y[:, j]
        P, N = np.where(y == 1)[0], np.where(y == 0)[0]
        pl, po, nl = PL[gi, j], PO[gi, j], na_l[gi, j]
        wa = wm = 0.0
        # w is the AUC credit one tied pair gains from the tie-break: a tie already scores
        # 0.5, so ordering it correctly is +0.5, incorrectly -0.5, and leaving it tied 0.
        # The sum is split by whether either cell is masked out of the loss, which separates
        # gain the scoreboard sees from gain the student can actually be trained on.
        for i in P:
            for k in N[pl[i] == pl[N]]:
                z = np.sign(po[i] - po[k])
                w = (0.5 * (z + 1) if z != 0 else 0.5) - 0.5
                if nl[i] or nl[k]:
                    wm += w
                else:
                    wa += w
        g_all.append((wa + wm) / (len(P) * len(N)))
        g_train.append(wa / (len(P) * len(N)))
    print(f"  total gold gain                        {np.mean(g_all):+.4f}")
    print(f"  ...on pairs where the LLM asserts BOTH {np.mean(g_train):+.4f}  <- reaches the student")
    print(f"  ...on pairs involving a masked cell    {np.mean(g_all) - np.mean(g_train):+.4f}")
    print("  Essentially all of it is trainable, because inside the LLM's not_mentioned")
    print("  bucket the rule extractor is silent too (98.6% of those cells), so there is no")
    print("  tie there to break in the first place.")

    # ---- 6. coverage, and the two mask edits that were rejected
    print("\n" + "=" * 104)
    print("6. COVERAGE (share of the 4,407 x 12 cells that carry gradient)")
    print("=" * 104)
    print(f"  llm mask policy  (SHIPPED)   {100 * (~na_l).mean():>6.1f}%")
    print(f"  ours mask policy             {100 * (~na_o).mean():>6.1f}%")
    print(f"  union, either asserts        {100 * (~(na_l & na_o)).mean():>6.1f}%   "
          f"(+{100 * (na_l & ~na_o).mean():.1f} pp -- rejected, see table 3)")
    print(f"  llm minus direction clashes  {100 * ((~na_l) & (~conflict)).mean():>6.1f}%   "
          f"(-{100 * ((~na_l) & conflict).mean():.1f} pp -- rejected, see below)")
    print("\n  COVERAGE CHANGE OF THE SHIPPED RULE: 0.0 pp. It is deliberately a pure")
    print("  re-ordering. Every cell the LLM spoke about is still trained; no cell the LLM")
    print("  was silent about was invented.")

    cg, yy = conflict[gi], Y.astype(int)
    ok = (~na_l[gi]) & (~cg) & (dl[gi] != 0)
    print(f"\n  direction clashes: {conflict.sum()} cells ({100 * conflict.mean():.1f}%), "
          f"{cg.sum()} of them inside gold")
    print(f"    llm  direction correct where they CLASH   "
          f"{((dl[gi] == 1) == (yy == 1))[cg].mean():.3f}")
    print(f"    ours direction correct where they CLASH   "
          f"{((do[gi] == 1) == (yy == 1))[cg].mean():.3f}")
    print(f"    llm  direction correct where they AGREE   "
          f"{((dl[gi] == 1) == (yy == 1))[ok].mean():.3f}  (n={ok.sum()})")
    print("    -> a clash marks a WEAKER cell, not a wrong one. 69% is still signal and a")
    print("       masked cell delivers none, so clashes are kept. (n=26 in gold: this is a")
    print("       weak measurement, which is another reason not to build a rule on it.)")

    # ---- verdict
    d = b_tb - b_llm
    print("\n" + "=" * 104)
    print("VERDICT")
    print("=" * 104)
    print(f"  rule        p = p_llm + {EPS} * (p_ours - 0.5)")
    print("  parameters  1 constant, 0 fitted on gold (AUC is invariant to it -- table 3)")
    print(f"  gold macro  {p_llm.mean():.4f} -> {p_tb.mean():.4f}  "
          f"({p_tb.mean() - p_llm.mean():+.4f})")
    print(f"  paired 95%  [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}], "
          f"P(delta > 0) = {(d > 0).mean():.3f}")
    print("  The project's one gold->LB calibration point (gold +0.0034 -> LB +0.002) would")
    print("  extrapolate to roughly +0.007 LB, but that is a 3x extrapolation from a single")
    print("  point. Treat this as 'a real gain of unknown size', not as +0.007.")


# ============================================================================= self-test
def self_test():
    """Invariants, not smoke. Every check below is a way this file could silently do nothing."""
    print("SELF-TEST  teacher_consensus.py\n" + "=" * 76)
    llm, ours = load_sources()
    out = build_consensus(llm, ours)
    tr = pd.read_csv(ROOT / "train.csv")

    # -- 1. the mask vocabulary this file assumes is still the one model.py implements.
    src = (ROOT / "model.py").read_text(encoding="utf-8", errors="replace")
    m = re.search(r"NOT_ASSERTED\s*=\s*\{([^}]*)\}", src)
    assert m, "model.py no longer defines NOT_ASSERTED -- the mask contract moved"
    got = set(re.findall(r'"([^"]+)"', m.group(1)))
    assert got == NOT_ASSERTED, f"model.py has {got}, this file assumes {NOT_ASSERTED}"
    print(f"  [ok] model.py NOT_ASSERTED still {sorted(got)}")

    # -- 2. schema is exactly the LLM file's, column for column.
    exp = ["StudyInstanceUID"] + TARGETS + [t + "__state" for t in TARGETS]
    assert list(out.columns) == exp, f"columns {list(out.columns)}"
    src_cols = list(pd.read_csv(ROOT / "report_labels_llm.csv", nrows=1).columns)
    assert set(out.columns) == set(src_cols), (
        f"schema differs from report_labels_llm.csv: {set(out.columns) ^ set(src_cols)}")
    print(f"  [ok] schema matches report_labels_llm.csv ({len(exp)} columns)")

    # -- 3. rows and IDs match train.csv exactly (no dropped or invented study).
    assert len(out) == len(tr) == 4407, f"{len(out)} rows vs train.csv {len(tr)}"
    assert set(out.StudyInstanceUID) == set(tr.StudyInstanceUID), "StudyInstanceUID set differs"
    assert out.StudyInstanceUID.is_unique, "duplicate StudyInstanceUID"
    print(f"  [ok] {len(out)} rows, ID set identical to train.csv")

    # -- 4. probabilities usable as soft BCE targets by distill_loss.
    P = out[TARGETS].values
    assert np.isfinite(P).all(), "non-finite probability"
    assert P.min() >= 0.0 and P.max() <= 1.0, f"prob range [{P.min()}, {P.max()}]"
    print(f"  [ok] probabilities finite and inside [{P.min():.4f}, {P.max():.4f}]")

    # -- 5. every state string is one KneeStudies can act on. It either masks the cell or
    #       trains it and there is no third behaviour, so the real risks are a NaN, a
    #       masking token the LLM never uses, or a column that is masked end to end.
    scol = [t + "__state" for t in TARGETS]
    assert not out[scol].isna().any().any(), "NaN in a state column"
    S = out[scol].values.astype(str)
    seen = set(np.unique(S))
    assert NOT_ASSERTED & seen == {"not_mentioned"}, (
        f"expected exactly the LLM's masking token, saw {NOT_ASSERTED & seen}")
    assert seen - NOT_ASSERTED, "every state is a masking token -- the teacher says nothing"
    for j, t in enumerate(TARGETS):
        assert (~np.isin(S[:, j], list(NOT_ASSERTED))).any(), f"{t} is masked end to end"
    print(f"  [ok] states {sorted(seen)}; masking token 'not_mentioned'; no target fully masked")

    # -- 6. THE aggregator failure mode: silently returning one of its inputs. Checked
    #       against both inputs, in both directions.
    PL = llm[TARGETS].values.astype(float)
    PO = ours[TARGETS].values.astype(float)
    dl_, do_ = (P != PL).mean(), (P != PO).mean()
    assert dl_ > 0.9, f"consensus equals the llm input on {100 * (1 - dl_):.1f}% of cells"
    assert do_ > 0.9, f"consensus equals the ours input on {100 * (1 - do_):.1f}% of cells"
    print(f"  [ok] differs from llm on {100 * dl_:.1f}% of cells, from ours on {100 * do_:.1f}%")

    # -- 7. it actually breaks ties. This is the failure the whole file exists to avoid: an
    #       EPS that underflows, or a p_ours that is constant inside every bucket, leaves the
    #       tie mass untouched and every check above still passes on a no-op.
    n_before = sum(len(np.unique(PL[:, j])) for j in range(len(TARGETS)))
    n_after = sum(len(np.unique(P[:, j])) for j in range(len(TARGETS)))
    assert n_after > 3 * n_before, (
        f"distinct levels barely moved ({n_before} -> {n_after}): no ties were broken")
    print(f"  [ok] distinct score levels across 12 targets: {n_before} -> {n_after}")

    # -- 8. and it breaks ONLY ties. If the perturbation ever reorders two cells the LLM
    #       ranked differently, the safety argument for EPS is void and this stops being a
    #       tie-break and becomes an unvalidated blend. Sampled pairs, not all 9.7M/target.
    e_max = max_safe_eps(PO)
    assert EPS < e_max, f"EPS={EPS} exceeds the provably safe bound {e_max:.3f}"
    rng = np.random.default_rng(0)
    for j in range(len(TARGETS)):
        i, k = rng.integers(0, len(out), 200_000), rng.integers(0, len(out), 200_000)
        a, b = np.sign(PL[i, j] - PL[k, j]), np.sign(P[i, j] - P[k, j])
        bad = int(((a != 0) & (a != b)).sum())
        assert bad == 0, f"{TARGETS[j]}: {bad} sampled pairs reordered against the LLM"
    print(f"  [ok] 2.4M sampled pairs, LLM strict order never inverted "
          f"(EPS {EPS} < safe bound {e_max:.3f})")

    # -- 9. the mask is untouched. A future edit that "improves coverage" here would change
    #       what the student trains on without moving any number this file prints.
    assert (S == llm[scol].values.astype(str)).all(), "the mask drifted from the llm input"
    print(f"  [ok] mask identical to the llm input; coverage "
          f"{100 * (~np.isin(S, list(NOT_ASSERTED))).mean():.1f}%")

    # -- 10. EPS is not a knob: gold macro must be identical across three decades of it.
    gold = gold_frame()
    gi = pd.Index(ours.index).get_indexer(gold.StudyInstanceUID.values)
    Y = np.stack([gold[t].values.astype(float) for t in TARGETS], 1)
    macro = lambda p: float(np.mean([roc_auc_score(Y[:, j].astype(int), p[:, j])
                                     for j in range(len(TARGETS))]))
    aucs = [macro(consensus_prob(PL, PO, e)[gi]) for e in (0.001, 0.01, 0.05, 0.063)]
    assert max(aucs) - min(aucs) < 1e-12, f"gold AUC depends on EPS: {aucs}"
    base = macro(PL[gi])
    print(f"  [ok] gold macro identical for EPS in [0.001, 0.063]: {aucs[0]:.4f}")

    # -- 11. and the point of the exercise. If this ever fails, do not ship the file.
    assert aucs[0] > base, f"consensus {aucs[0]:.4f} did not beat the llm {base:.4f} on gold"
    print(f"  [ok] gold macro llm {base:.4f} -> consensus {aucs[0]:.4f} ({aucs[0] - base:+.4f})")

    print("=" * 76 + "\nALL CHECKS PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--report", action="store_true", help="full comparison, no file written")
    ap.add_argument("--out", default="report_labels_consensus.csv")
    ap.add_argument("--ours", default="report_labels_ours.csv",
                    help="rule-extractor labels to tie-break with (default: the shipped one)")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.report:
        return report(a.ours)
    llm, ours = load_sources(a.ours)
    out = build_consensus(llm, ours)
    p = ROOT / a.out
    assert not p.exists(), f"{p} already exists -- nothing in this repo gets overwritten"
    out.to_csv(p, index=False, encoding="utf-8")
    print(f"-> {p}  ({len(out)} studies)")
    print(f"   p = p_llm + {EPS} * (p_ours - 0.5); state columns carried from the LLM unchanged")
    print(f"   train.py --labels {a.out}")


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()
