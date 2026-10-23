"""Independent recompute of the C02R7 blend (copy of check_blend_c02.py with S6 -> R7) from the files a run leaves in /kaggle/working.

Reads ours_scores.csv (our family's blend input + ok flag), raptor_raw.csv (nartaa's raw sigmoids
+ empty_volume flag; ok rule re-derived here), r7_raw.csv (heliosli R7 per-study probabilities +
ok flag) and submission.csv; rebuilds the expected submission with pandas rank (not the notebook's
scipy rankdata path) and the pre-registered weights 0.40 / 0.30 / 0.30, renormalised per study over
the families that scored it, and reports the max abs difference. Any family file may be absent
(fallback runs).

Extra checks:
  * r7_raw.csv vs heliosli's own r7_sub.csv (his controller output): every row r7_raw marks ok must
    equal his row (both %.9g from the same numbers, so |diff| <= 1e-12), and every row he scored
    (not in his fallback list) must be ok in r7_raw  [only when r7_sub.csv exists]
  * --r7-pred DIR: rebuild r7 ok/values straight from his workers' predictions.csv files
    (replays keep them; on Kaggle they live in /tmp and are not in the output)
  * --instrument: ours_scores.csv is the frozen 4f96cbd computation (rankdata of the member mean
    / N) and nartaa's raptor_sub.csv ranks agree in order with raptor_raw.csv

usage: python tools/check_blend.py RUN_DIR [--instrument] [--r7-pred DIR]
  RUN_DIR: the notebook output folder (/kaggle/working, or kaggle kernels output -p RUN_DIR)
exit 0 iff every check passes.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
W = {"ours": 0.40, "raptor": 0.30, "r7": 0.30}


def fam_rank(df, ok):
    """per-column average rank / n over the ok rows; NaN elsewhere"""
    r = df[T].where(ok, np.nan)
    return r.rank(method="average") / ok.sum()


def families(run, ids, r7_pred=None):
    fams = {}
    p = run / "ours_scores.csv"
    if p.is_file():
        o = pd.read_csv(p, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").reindex(ids)
        fams["ours"] = (o, o["ok"].fillna(0).astype(bool))
    p = run / "raptor_raw.csv"
    if p.is_file():
        r = pd.read_csv(p, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").reindex(ids)
        ok = r[T].notna().all(1) & ~(r[T] == 0.5).all(1) & (r["empty_volume"].fillna(0) == 0) \
            & np.isfinite(r[T]).all(1)
        fams["raptor"] = (r, ok)
    if r7_pred is not None:
        parts = [pd.read_csv(f, dtype={"StudyInstanceUID": str})
                 for f in sorted(Path(r7_pred).glob("worker*/predictions.csv"))]
        if parts:
            s = pd.concat(parts).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID").reindex(ids)
            ok = s[T].notna().all(1) & ((s[T] >= 0) & (s[T] <= 1)).all(1)
            if ok.any():
                fams["r7"] = (s, ok)
    else:
        p = run / "r7_raw.csv"
        if p.is_file():
            s = pd.read_csv(p, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").reindex(ids)
            fams["r7"] = (s, s["ok"].fillna(0).astype(bool))
    return fams


def expected(run, r7_pred=None):
    sub = pd.read_csv(run / "submission.csv", dtype={"StudyInstanceUID": str})
    ids = sub.StudyInstanceUID
    fams = families(run, ids, r7_pred)
    num = pd.DataFrame(0.0, index=ids, columns=T)
    den = pd.Series(0.0, index=ids)
    for k, (df, ok) in fams.items():
        if ok.sum():
            num += W[k] * fam_rank(df, ok).fillna(0.0)
            den += W[k] * ok
    exp = num.div(den.where(den > 0), axis=0).fillna(0.5)
    counts = {k: int(v[1].sum()) for k, v in fams.items()}
    return sub.set_index("StudyInstanceUID")[T], exp, counts, fams


def r7_vs_his(run):
    """problems between r7_raw.csv and heliosli's r7_sub.csv + receipt (empty list = consistent)"""
    probs = []
    if not (run / "r7_raw.csv").is_file() or not (run / "r7_sub.csv").is_file():
        return probs
    raw = pd.read_csv(run / "r7_raw.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    his = pd.read_csv(run / "r7_sub.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    if sorted(raw.index) != sorted(his.index):
        probs.append("r7_raw / r7_sub study ids differ")
        return probs
    his = his.reindex(raw.index)
    ok = raw["ok"].astype(bool)
    d = float(np.abs(raw.loc[ok, T].values - his.loc[ok, T].values).max()) if ok.any() else 0.0
    if d > 1e-12:   # both written %.9g from the same parsed numbers
        probs.append(f"r7_raw ok rows differ from his r7_sub.csv by {d:.2e}")
    rc = run / "r7_run" / "submission_receipt.json"
    if rc.is_file():
        rec = json.loads(rc.read_text(encoding="utf-8"))
        n_fb = int(rec.get("fallback_studies", -1))
        if int(rec.get("scored", -1)) != int(ok.sum()) or n_fb != int((~ok).sum()):
            probs.append(f"his receipt scored {rec.get('scored')} / fallback {n_fb} vs r7_raw ok "
                         f"{int(ok.sum())} / {int((~ok).sum())}")
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--instrument", action="store_true")
    ap.add_argument("--r7-pred", default=None)
    a = ap.parse_args()
    run = Path(a.run)
    probs = []
    got, exp, counts, fams = expected(run)
    d = float(np.abs(got.values - exp.values).max())
    print(f"families present {counts}; submission vs independent recompute: max |diff| {d:.2e}")
    if d > 1e-9:
        probs.append(f"blend mismatch {d}")
    if a.r7_pred:
        got2, exp2, counts2, _ = expected(run, a.r7_pred)
        d2 = float(np.abs(got2.values - exp2.values).max())
        print(f"  with r7 rebuilt from his workers' predictions.csv: families {counts2}; max |diff| {d2:.2e}")
        if d2 > 1e-9 or counts2 != counts:
            probs.append(f"r7 from predictions.csv: families {counts2}, diff {d2}")
    p6 = r7_vs_his(run)
    print("r7_raw.csv vs his r7_sub.csv / receipt: " + ("consistent" if not p6 else "; ".join(p6))
          + ("" if (run / "r7_sub.csv").is_file() else " (no r7_sub.csv in this run)"))
    probs += p6
    if a.instrument:
        ins = json.loads((run / "instrument.json").read_text())
        if ins["ours"] is not None:
            fp = np.array(ins["fam_preds"]["v1"])
            ok = np.array(ins["fam_ok"]["v1"])
            from scipy.stats import rankdata
            frozen = np.apply_along_axis(rankdata, 0, fp) / len(fp)   # 4f96cbd family_scores
            o = pd.read_csv(run / "ours_scores.csv")
            d2 = float(np.abs(o[T].values - frozen).max())
            print(f"ours_scores.csv vs frozen family_scores(member mean): max |diff| {d2:.2e}; "
                  f"ok {int(ok.sum())}/{len(ok)}")
            if d2 > 1e-9:
                probs.append(f"ours_scores != frozen computation ({d2})")
        if (run / "raptor_sub.csv").is_file() and (run / "raptor_raw.csv").is_file():
            rs = pd.read_csv(run / "raptor_sub.csv")
            rr = pd.read_csv(run / "raptor_raw.csv")
            assert rs.StudyInstanceUID.tolist() == rr.StudyInstanceUID.tolist()

            def consistent(t):
                k = np.round(rs[t].values * (len(rs) - 1))
                r = pd.Series(rr[t].values)
                return bool(((r.rank(method="min") - 1 <= k) & (k <= r.rank(method="max") - 1)).all())
            bad = [t for t in T if not consistent(t)]
            print(f"nartaa raptor_sub.csv ranks == ranks of raptor_raw.csv on {12 - len(bad)}/12 findings"
                  + (f" (differ: {bad})" if bad else ""))
            if bad:
                probs.append(f"raptor_sub vs raw order differs on {bad}")
    print("CHECK OK" if not probs else "CHECK FAILED: " + "; ".join(probs))
    return 0 if not probs else 1


if __name__ == "__main__":
    sys.exit(main())
