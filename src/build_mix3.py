"""build_mix3.py -- report_labels_mix3.csv: the equal mean of three report-label readers.

  ours    report_labels_consensus.csv            asserts a cell iff __state not in {silent, not_mentioned}
  steven  stevenleehans llm_labels_v4_blend.csv  asserts iff value != 0.25 (0.25 = its silent code;
                                                  the convention nartaa's blend uses, verified exact).
          On 11 targets v4 == 0.25 <=> steven v1 (llm_labels_full) == 0.5 'not addressed' (<= 30 cells differ).
          NOT on Synovitis: where v1 Synovitis == 0.5 (3,689 studies) steven v2 imputes Synovitis =
          0.25 + 0.5 x v1 Effusion (exact), and v4 blends that, so v4 != 0.25 there. With --steven-syn-v1-silent
          those cells count as steven-silent (Synovitis only); write that variant to a NEW filename.
  gemlow  nartaa train_labels_v1.parquet y_<t>   asserts iff m_<t> == 1. (m_ is nartaa's OOF-scoring mask,
          --oof_mask; their training loss is unmasked BCE. Used here as a 'mentioned' proxy.)
          (= labels_v1_gemlow.parquet on all 4,349 non-gold rows, bit-exact; the train_labels_v1 file
          also carries the 58 gold rows, which is the table the gold-58 scorer read as 'gemini gemlow')

Values are used as-is (all three are already P(present) in [0,1]; this is what score_tables.py did).
Per cell: mean over the sources that assert it. state = ours' state if ours asserts, else
'public_asserted' if a public table asserts, else 'not_mentioned' (value = mean of the three raw
values, inert in training because the cell is masked). Nothing here is tuned on gold.

Usage: python build_mix3.py <root_with_inputs> <out_csv> --dl <public_tables_dir> [<copy_csv>]
                            [--steven-syn-v1-silent] [--gold-report]
  <root_with_inputs>  folder holding report_labels_consensus.csv (and train.csv for --gold-report)
  --dl                folder holding the two downloaded public tables:
                        stevenleehans_rsna-knee-llm-report-labels/{llm_labels_v4_blend.csv,llm_labels_full.csv}
                        nartaa_rsna-knee-hpo-assets/{train_labels_v1.parquet,labels_v1_gemlow.parquet}
  --gold-report       also print the gold-58 record and write <tag>_gold58_pertarget.csv and
                      <tag>_masking_spearman.csv next to <out_csv> (record only; changes nothing)
"""
import os
import shutil
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

SYN_V1_SILENT = "--steven-syn-v1-silent" in sys.argv
GOLD_REPORT = "--gold-report" in sys.argv
ARGS = [a for a in sys.argv[1:] if a not in ("--steven-syn-v1-silent", "--gold-report")]
if "--dl" not in ARGS:
    sys.exit("build_mix3.py: --dl <public_tables_dir> is required (see the docstring)")
_i = ARGS.index("--dl")
DL = ARGS[_i + 1]
ARGS = ARGS[:_i] + ARGS[_i + 2:]
ROOT, OUT = ARGS[0], ARGS[1]
COPY = ARGS[2] if len(ARGS) > 2 else None
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
     "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
U = {t: t.replace(" ", "_").replace("'", "") for t in T}
S = [t + "__state" for t in T]
NOT_ASSERTED = {"silent", "not_mentioned"}

ours = pd.read_csv(os.path.join(ROOT, "report_labels_consensus.csv"))
ids = ours.StudyInstanceUID
assert ids.is_unique and len(ours) == 4407
st = pd.read_csv(os.path.join(DL, "stevenleehans_rsna-knee-llm-report-labels", "llm_labels_v4_blend.csv"))
st = st.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID").reindex(ids)
gm = pd.read_parquet(os.path.join(DL, "nartaa_rsna-knee-hpo-assets", "train_labels_v1.parquet"))
gm = gm.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID").reindex(ids)
gl = pd.read_parquet(os.path.join(DL, "nartaa_rsna-knee-hpo-assets", "labels_v1_gemlow.parquet"))
gl = gl.set_index("StudyInstanceUID")

V_o = ours[T].to_numpy(float)
A_o = ~ours[S].isin(NOT_ASSERTED).to_numpy()
V_s = st[T].to_numpy(float)
A_s = ~np.isclose(V_s, 0.25)
v1 = pd.read_csv(os.path.join(DL, "stevenleehans_rsna-knee-llm-report-labels", "llm_labels_full.csv"))
v1 = v1.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID").reindex(ids)
SYN = T.index("Synovitis")
SYN_IMP = np.isclose(v1["Synovitis"].to_numpy(float), 0.5)   # steven v1 'not addressed' -> v2/v4 Effusion imputation
if SYN_V1_SILENT:
    A_s[:, SYN] &= ~SYN_IMP
V_g = gm[["y_" + U[t] for t in T]].to_numpy(float)
A_g = gm[["m_" + U[t] for t in T]].to_numpy(float) == 1.0
for nm, v in (("ours", V_o), ("steven", V_s), ("gemlow", V_g)):
    assert not np.isnan(v).any() and v.min() >= 0 and v.max() <= 1, nm
# gemlow table identity: train_labels_v1 y_ == labels_v1_gemlow on every row the latter has
assert np.array_equal(gm.loc[gl.index, ["y_" + U[t] for t in T]].to_numpy(float), gl[T].to_numpy(float))

V = np.stack([V_o, V_s, V_g])            # (3, N, 12)
A = np.stack([A_o, A_s, A_g])
n = A.sum(0)
mix = np.where(n > 0, (V * A).sum(0) / np.maximum(n, 1), V.mean(0))
state = np.where(A_o, ours[S].to_numpy(object),
                 np.where(n > 0, "public_asserted", "not_mentioned"))

out = pd.DataFrame(mix, columns=T)
out.insert(0, "StudyInstanceUID", ids.values)
for j, c in enumerate(S):
    out[c] = state[:, j]
out = out[list(ours.columns)]
# ---- verify
assert len(out) == 4407 and (out.StudyInstanceUID.values == ids.values).all()
assert list(out.columns) == list(ours.columns)
v = out[T].to_numpy(float)
assert not np.isnan(v).any() and v.min() >= 0 and v.max() <= 1
assert not out[S].isna().any().any()
out.to_csv(OUT, index=False)
if COPY:
    shutil.copy2(OUT, COPY)
print(f"wrote {OUT} {out.shape}" + (f" (+ copy {COPY})" if COPY else "")
      + f"  steven Synovitis rule: {'v1==0.5 -> silent' if SYN_V1_SILENT else 'v4!=0.25 -> asserted'}")
if not GOLD_REPORT:
    sys.exit(0)
mix = pd.read_csv(OUT)[T].to_numpy(float)    # report from the CSV as written (avoids in-memory float ties)

# ---- report
tr = pd.read_csv(os.path.join(ROOT, "train.csv"))
gold = tr[tr[T].notna().all(axis=1)][["StudyInstanceUID"] + T]
assert len(gold) == 58
gi = ids.reset_index().set_index("StudyInstanceUID").loc[gold.StudyInstanceUID, "index"].values
YG = gold[T].to_numpy(int)
ref = {"ours": V_o, "steven_v4": V_s, "gemlow": V_g, "mix3": mix,
       "mean3_unmasked": V.mean(0),
       "rankmean3": np.mean([pd.DataFrame(x).rank(pct=True).to_numpy() for x in (V_o, V_s, V_g)], 0)}


def aucs(P):
    return np.array([roc_auc_score(YG[:, j], P[gi, j]) for j in range(12)])


B = 2000
rng = np.random.default_rng(0)
IDX = rng.integers(0, 58, size=(B, 58))


def boot_macro(P):
    from scipy.stats import rankdata
    out = np.full((B, 12), np.nan)
    for j in range(12):
        y = YG[IDX, j]; s = P[gi, j][IDX]
        r = rankdata(s, axis=1); npos = y.sum(1); nneg = 58 - npos
        ok = (npos > 0) & (nneg > 0)
        a = ((r * y).sum(1) - npos * (npos + 1) / 2) / np.where(ok, npos * nneg, 1)
        out[ok, j] = a[ok]
    return np.nanmean(out, 1)


per = pd.DataFrame({k: aucs(P) for k, P in ref.items()}, index=T)
b0 = boot_macro(V_o)
summ = pd.DataFrame({k: dict(macro=per[k].mean(), d_vs_ours=per[k].mean() - per["ours"].mean(),
                             se_d=(boot_macro(P) - b0).std()) for k, P in ref.items()}).T
pd.set_option("display.width", 200)
print("\nGOLD-58 (record only; nothing tuned on it)")
print(summ.round(4).to_string())
print("\nper-target gold-58 AUC")
print(per.round(3).to_string())
print("\nper-target delta mix3 - ours")
print((per["mix3"] - per["ours"]).round(3).to_string())

ng = ~ids.isin(gold.StudyInstanceUID).values
rows = []
for j, t in enumerate(T):
    rows.append(dict(target=t,
                     masked_ours=1 - A_o[:, j].mean(), masked_mix3=(n[:, j] == 0).mean(),
                     assert_steven=A_s[:, j].mean(), assert_gemlow=A_g[:, j].mean(),
                     n_src_mean=n[:, j].mean(),
                     unmasked_by_public=(~A_o[:, j] & (n[:, j] > 0)).mean(),
                     spearman_nongold=spearmanr(V_o[ng, j], mix[ng, j]).correlation,
                     spearman_ours_asserted=spearmanr(V_o[ng & A_o[:, j], j],
                                                      mix[ng & A_o[:, j], j]).correlation,
                     prev_asserted_ours=V_o[A_o[:, j], j].mean(),
                     prev_asserted_mix3=mix[n[:, j] > 0, j].mean()))
R = pd.DataFrame(rows).set_index("target")
R.loc["MEAN"] = R.mean()
print("\nmasking / agreement (all 4,407 rows; Spearman over the 4,349 non-gold)")
print(R.round(3).to_string())
print("\nstate counts:", pd.Series(state.ravel()).value_counts().to_dict())
tr_syn = n[:, SYN] > 0
st_only = A_s[:, SYN] & ~A_o[:, SYN] & ~A_g[:, SYN]
print(f"Synovitis: trained cells {tr_syn.sum()}; public_asserted {(tr_syn & ~A_o[:, SYN]).sum()}; "
      f"steven asserts on an Effusion-imputed cell {(A_s[:, SYN] & SYN_IMP).sum()}; "
      f"cells whose ONLY input is steven's Effusion imputation {(st_only & SYN_IMP).sum()} "
      f"({(st_only & SYN_IMP).sum() / tr_syn.sum():.3f} of trained Synovitis cells)")
# gold-row leak check: does either public table hold the true gold labels on the gold rows?
for nm, P in (("steven_v4", V_s), ("gemlow", V_g)):
    print(f"gold rows: {nm} exact match with gold {np.isclose(P[gi], YG).mean():.3f}; "
          f"0/1 cells {np.isin(P[gi], [0, 1]).mean():.3f} gold vs {np.isin(P[ng], [0, 1]).mean():.3f} non-gold")
TAG = os.path.splitext(os.path.basename(OUT))[0].replace("report_labels_", "")
per.to_csv(os.path.join(os.path.dirname(os.path.abspath(OUT)), f"{TAG}_gold58_pertarget.csv"))
R.to_csv(os.path.join(os.path.dirname(os.path.abspath(OUT)), f"{TAG}_masking_spearman.csv"))
