"""
validation.py -- steps C1-C3: folds, scoring, and submission hygiene. Import this everywhere.

Every experiment must be measured the same way or the numbers cannot be compared, so the
harness lives in one module rather than being rewritten per script.

C1  FOLDS. Grouped by `lang x manufacturer|model`, 62 groups. Site identity was stripped
    from the headers, so it is reconstructed from two independent signals: the scanner
    fingerprint, and the language the report was written in -- several languages map to a
    single manufacturer (bg -> Philips 220/220; de, el, fr, nl -> Siemens). Crossing them
    halves the largest group, 16.8% -> 5.9%. Random K-fold inflates AUC by memorising the
    scanner; this is the competition's actual thesis, not a hygiene detail.

    The 58 gold studies are balanced across folds explicitly. Left to chance, a fold can
    end up with two of them and its gold AUC becomes meaningless.

C2  SCORING. Three numbers, always reported together, because no one of them can carry a
    decision at n=58:
      * gold macro AUC with a bootstrap CI  (real labels, tiny, noisy)
      * grouped OOF against report labels   (large, but measures agreement with the
                                             teacher rather than correctness)
      * the grouped-vs-random gap           (how much a model is memorising the scanner)

C3  HYGIENE. A constant column scores AUC 0.5 by definition and silently forfeits a full
    twelfth of the score. Assert before writing, never after.

Usage:
    python validation.py                 # build folds.csv and print the layout
    from validation import evaluate_gold, grouped_oof_auc, check_submission
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
N_FOLDS = 5
SEED = 2026


# --------------------------------------------------------------------------- C1
def build_folds(n_folds=N_FOLDS, seed=SEED, out="folds.csv"):
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0

    tr = pd.read_csv(ROOT / "train.csv")
    meta = pd.read_csv(ROOT / "study_metadata.csv")

    lang = []
    for t in tr.Report.fillna(""):
        t = t.strip()
        try:
            lang.append(detect(t) if len(t) >= 10 else "unk")
        except Exception:
            lang.append("unk")
    df = pd.DataFrame({"StudyInstanceUID": tr.StudyInstanceUID, "lang": lang})
    df = df.merge(meta[["StudyInstanceUID", "manufacturer", "model", "g_mfr_model"]],
                  on="StudyInstanceUID")
    df["group"] = df.lang + "|" + df.g_mfr_model

    gold_ids = set(tr[tr[TARGETS].notna().all(axis=1)].StudyInstanceUID)
    df["is_gold"] = df.StudyInstanceUID.isin(gold_ids)

    # Assign whole groups to folds, largest first, always into the currently smallest
    # fold. Greedy, deterministic, and it keeps folds balanced without splitting a site.
    sizes = df.group.value_counts()
    gold_per_group = df[df.is_gold].group.value_counts()

    # Groups holding gold are placed first, round-robin, so the 58 spread across folds
    # instead of landing wherever the size-greedy pass happens to put them.
    fold_of = {}
    load = np.zeros(n_folds)
    gold_load = np.zeros(n_folds)
    for g in gold_per_group.index:
        f = int(np.argmin(gold_load + load / max(len(df), 1)))
        fold_of[g] = f
        gold_load[f] += gold_per_group[g]
        load[f] += sizes[g]
    for g in sizes.index:
        if g in fold_of:
            continue
        f = int(np.argmin(load))
        fold_of[g] = f
        load[f] += sizes[g]

    df["fold"] = df.group.map(fold_of)
    p = ROOT / out
    df.to_csv(p, index=False)

    print(f"folds -> {p}")
    print(f"  {df.group.nunique()} groups over {n_folds} folds\n")
    print(f"  {'fold':>4} {'studies':>8} {'gold':>6} {'groups':>7} {'languages':>10} {'vendors':>8}")
    for f in range(n_folds):
        s = df[df.fold == f]
        print(f"  {f:>4} {len(s):>8} {int(s.is_gold.sum()):>6} {s.group.nunique():>7} "
              f"{s.lang.nunique():>10} {s.manufacturer.nunique():>8}")
    print(f"\n  no group spans two folds: {bool((df.groupby('group').fold.nunique() == 1).all())}")
    return df


# --------------------------------------------------------------------------- C2
def evaluate_gold(pred: pd.DataFrame, n_boot=2000, seed=0, verbose=True):
    """pred: StudyInstanceUID + the 12 target columns. -> dict of metrics."""
    tr = pd.read_csv(ROOT / "train.csv")
    gold = tr[tr[TARGETS].notna().all(axis=1)]
    m = gold[["StudyInstanceUID"] + TARGETS].merge(pred, on="StudyInstanceUID",
                                                   suffixes=("_g", "_p"))
    if len(m) < len(gold):
        print(f"  WARNING: only {len(m)}/{len(gold)} gold studies present in predictions")

    # A target with a single class present has no defined AUC. On the full 58 that never
    # happens, but on one fold (~11 studies) it happens often -- and averaging NaN in
    # destroys the macro, which is the number being read to make the decision. Skip the
    # undefined targets and say how many were skipped, rather than reporting nothing.
    def auc(y, p):
        return roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan

    per = {t: auc(m[t + "_g"].values, m[t + "_p"].values) for t in TARGETS}
    ok = [t for t in TARGETS if not np.isnan(per[t])]
    macro = float(np.nanmean([per[t] for t in ok])) if ok else float("nan")

    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(m), len(m))
        vals = [auc(m[t + "_g"].values[i], m[t + "_p"].values[i]) for t in ok]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            bs.append(float(np.mean(vals)))
    lo, hi = ((float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
              if bs else (float("nan"), float("nan")))

    if verbose:
        print(f"  {'target':<18} {'AUC':>7} {'n_pos':>6}")
        for t in TARGETS:
            v = f"{per[t]:>7.3f}" if not np.isnan(per[t]) else f"{'n/a':>7}"
            print(f"  {t:<18} {v} {int(m[t + '_g'].sum()):>6}")
        print(f"  {'MACRO':<18} {macro:>7.3f}   95% CI [{lo:.3f}, {hi:.3f}]  (n={len(m)})")
        if len(ok) < len(TARGETS):
            print(f"  NOTE: {len(TARGETS) - len(ok)} target(s) had one class only and were "
                  f"excluded — this macro is NOT comparable to a full-12 macro")
    return dict(macro=macro, ci=(lo, hi), per_target=per, n=len(m), n_scored=len(ok))


def coverage(labels: pd.DataFrame, verbose=True):
    """Non-silent rate per target over all 4,407 studies.

    Must be reported alongside gold AUC, never instead of it, and never the other way
    round. Gold has 58 studies; a change that blinds the labeler on a thousand studies
    moves gold AUC by less than its own noise and is invisible there. That is not
    hypothetical -- it happened here, and coverage is the metric that caught it.
    """
    out = {}
    for t in TARGETS:
        col = t + "__state"
        out[t] = float((labels[col] != "silent").mean()) if col in labels.columns else np.nan
    if verbose:
        print(f"  {'target':<18} {'non-silent':>11} {'n studies':>10}")
        for t in TARGETS:
            print(f"  {t:<18} {100 * out[t]:>10.1f}% {int(out[t] * len(labels)):>10}")
        print(f"  {'MEAN':<18} {100 * np.nanmean(list(out.values())):>10.1f}%")
    return out


def report(pred_labels: pd.DataFrame, name="labeler"):
    """The standard pair. Every labeler change gets scored through this, both halves."""
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    print("GOLD AUC (58 studies — measures correctness, tiny and noisy)")
    g = evaluate_gold(pred_labels[["StudyInstanceUID"] + TARGETS])
    print("\nCOVERAGE (4,407 studies — measures recall, what the vision model trains on)")
    c = coverage(pred_labels)
    return dict(gold=g, coverage=c)


def grouped_oof_auc(pred: pd.DataFrame, labels: pd.DataFrame, folds: pd.DataFrame = None):
    """Agreement with the teacher on held-out folds. Large-n, but not correctness."""
    if folds is None:
        folds = pd.read_csv(ROOT / "folds.csv")
    m = (folds[["StudyInstanceUID", "fold"]]
         .merge(labels, on="StudyInstanceUID")
         .merge(pred, on="StudyInstanceUID", suffixes=("_y", "_p")))
    out = {}
    for t in TARGETS:
        y = (m[t + "_y"] >= 0.5).astype(int)
        if y.nunique() < 2:
            out[t] = np.nan
            continue
        out[t] = roc_auc_score(y, m[t + "_p"])
    return dict(macro=float(np.nanmean(list(out.values()))), per_target=out, n=len(m))


def compare(pred_a, pred_b, name_a="A", name_b="B", n_boot=400, seed=0, labels=None):
    """Is B better than A? PAIRED bootstrap, on both instruments.

    WHY PAIRED, AND WHY TEACHER-AGREEMENT
    Our remaining headroom to the teacher is ~0.011 macro, and essentially every idea left
    is worth less than that. Absolute gold macro AUC on 58 studies has a 95% CI of about
    +/-0.035, so it cannot see any of them: it would call a real +0.01 improvement and a
    real +0.00 improvement the same thing.

    Two changes fix the instrument without touching the data.

    PAIRING. Comparing two models on the SAME studies cancels the between-study variance,
    which is most of the noise. The quantity bootstrapped is the DIFFERENCE, not two
    independent scores.

    TEACHER-AGREEMENT ON ALL 4,407. Scoring against the report-derived labels instead of
    the 58 gold trades a little validity for a lot of n. Measured on the v2-vs-ensemble
    comparison: paired TA gives +0.0087 +/- 0.0009 (9.7 sigma) where paired gold gives
    +0.0102 +/- 0.0067 (1.5 sigma) -- same effect, 7x less noise.

    THE CATCH, which is why gold is still reported alongside. TA measures IMITATION of the
    teacher, not correctness. We already beat the teacher on Effusion (+0.094), Synovitis
    (+0.070) and Contusion (+0.073), so a change that improves real accuracy on those can
    LOWER TA. Use TA to screen -- it is the only instrument that can resolve a 0.01 effect --
    and gold as a no-regression check, never the other way round.
    """
    if labels is None:
        labels = pd.read_csv(ROOT / "report_labels_llm.csv")
    lab = labels.set_index("StudyInstanceUID")
    A = pred_a.set_index("StudyInstanceUID") if "StudyInstanceUID" in pred_a else pred_a
    B = pred_b.set_index("StudyInstanceUID") if "StudyInstanceUID" in pred_b else pred_b
    common = A.index.intersection(B.index).intersection(lab.index)

    def paired(Y, Pa, Pb, rng):
        d = []
        for _ in range(n_boot):
            i = rng.integers(0, len(Y), len(Y))
            ok = [j for j in range(len(TARGETS)) if len(np.unique(Y[i, j])) > 1]
            if not ok:
                continue
            d.append(np.mean([roc_auc_score(Y[i, j], Pb[i, j]) for j in ok])
                     - np.mean([roc_auc_score(Y[i, j], Pa[i, j]) for j in ok]))
        return np.asarray(d)

    out = {}
    rng = np.random.default_rng(seed)
    Y = (lab.loc[common, TARGETS].values >= 0.5).astype(int)
    d = paired(Y, A.loc[common, TARGETS].values, B.loc[common, TARGETS].values, rng)
    out["ta"] = dict(delta=float(d.mean()), sd=float(d.std()), n=len(common),
                     sigma=float(abs(d.mean()) / max(d.std(), 1e-12)))

    tr = pd.read_csv(ROOT / "train.csv")
    gold = tr[tr[TARGETS].notna().all(axis=1)].set_index("StudyInstanceUID")
    gi = [s for s in gold.index if s in common]
    dg = paired(gold.loc[gi, TARGETS].values.astype(int),
                A.loc[gi, TARGETS].values, B.loc[gi, TARGETS].values,
                np.random.default_rng(seed))
    out["gold"] = dict(delta=float(dg.mean()), sd=float(dg.std()), n=len(gi),
                       sigma=float(abs(dg.mean()) / max(dg.std(), 1e-12)))

    print(f"  {name_b} vs {name_a}")
    for k, lbl in [("ta", f"teacher-agreement (n={out['ta']['n']})"),
                   ("gold", f"gold           (n={out['gold']['n']})")]:
        r = out[k]
        verdict = ("clear" if r["sigma"] >= 3 else
                   "suggestive" if r["sigma"] >= 2 else "indistinguishable")
        print(f"    {lbl:<32} {r['delta']:+.4f} +/- {r['sd']:.4f}   "
              f"{r['sigma']:>4.1f} sigma  {verdict}")
    if out["ta"]["delta"] > 0 and out["ta"]["sigma"] >= 3 and out["gold"]["delta"] < -0.01:
        print("    WARNING: TA improved but gold fell — may be imitating the teacher's")
        print("             errors rather than getting more correct.")
    return out


def group_gap(pred, labels, _allow=False):
    """DO NOT USE on a single OOF file. Kept only to fail loudly instead of misleading.

    This originally re-scored one set of predictions under two different fold assignments
    and reported the difference. That is not the grouped-vs-random gap and cannot be: the
    predictions were produced by models trained under ONE fold assignment, and relabelling
    which fold a study belongs to afterwards changes nothing about them. Both calls scored
    the same numbers over the same studies, so it returned exactly +0.000 -- and +0.000 was
    read as "the model is not using the site shortcut", which the experiment never tested.

    Measuring the gap honestly costs a second training run:
        python train.py --all-folds ... --out oof_grouped.csv       (folds.csv)
        python train.py --all-folds ... --random-folds --out oof_random.csv
    then compare the two gold AUCs. The difference is the shortcut's contribution.
    """
    if not _allow:
        raise NotImplementedError(group_gap.__doc__)
    raise NotImplementedError


# --------------------------------------------------------------------------- C3
def check_submission(sub: pd.DataFrame, expect_rows=None, raise_on_fail=True):
    """Refuse to write a submission that silently forfeits score."""
    problems = []
    if list(sub.columns) != ["StudyInstanceUID"] + TARGETS:
        problems.append(f"column layout wrong: {list(sub.columns)}")
    if expect_rows is not None and len(sub) != expect_rows:
        problems.append(f"expected {expect_rows} rows, got {len(sub)}")
    if sub.StudyInstanceUID.duplicated().any():
        problems.append("duplicate StudyInstanceUID")
    for t in TARGETS:
        if t not in sub.columns:
            continue
        v = sub[t]
        if v.isna().any():
            problems.append(f"{t}: {int(v.isna().sum())} NaN")
        if not ((v >= 0) & (v <= 1)).all():
            problems.append(f"{t}: outside [0,1] -- range [{v.min()}, {v.max()}]")
        if v.nunique() <= 1:
            problems.append(f"{t}: CONSTANT -- forfeits 1/12 of the score (AUC 0.5)")
        elif v.nunique() < 10 and len(sub) > 100:
            problems.append(f"{t}: only {v.nunique()} distinct values -- coarse ranking")
    if problems:
        msg = "submission check failed:\n  - " + "\n  - ".join(problems)
        if raise_on_fail:
            raise ValueError(msg)
        print(msg)
        return False
    print(f"  submission OK: {len(sub)} rows, 12 targets, all in [0,1], none constant")
    return True


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    df = build_folds()

    print("\n" + "=" * 70)
    print("SANITY: score our own labeler through the harness")
    print("=" * 70)
    ours = pd.read_csv(ROOT / "report_labels_ours.csv")[["StudyInstanceUID"] + TARGETS]
    evaluate_gold(ours)
