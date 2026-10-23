"""
build_oai_labels.py -- join the OAI-ZIB knee volumes we already hold to real MOAKS readings.

WHY THIS FILE EXISTS
Our twelve labels come from an LLM reading radiology reports. D30b established that the model
is limited by that supervision rather than by its size: a bigger backbone fit better and
generalised worse. So the useful thing to buy is not more images, it is labels a radiologist
read OFF THE IMAGES.

Those exist, already on this machine, and nobody had connected them:

  data/OAI-ZIB/Images_Pretrain      507 OAI knee volumes (sagittal 3D DESS, NIfTI)
  data/OAI-ZIB/Zip_Files/info.zip   subInfo_*.xlsx -- the de-anonymisation table mapping
                                   oaizib_NNN -> real OAI SubjectID, knee side, KL grade
  data/3dreasonknee   MOAKS semi-quantitative readings (CC BY 4.0), keyed by
                                   OAI ID + SIDE

The subInfo table is the bridge. Without it the oaizib_NNN filenames are anonymous and the
readings cannot be attached to anything.

THE TRAP THIS FILE EXISTS TO AVOID
**Join on ID *and* SIDE.** MOAKS scores each knee separately and many OAI participants have
both knees read: in the baseline file, 2,030 unique IDs produce 3,017 rows. OAI-ZIB holds ONE
knee per subject. Joining on subject alone would silently attach the wrong knee's findings to
roughly a third of the data -- no error, no warning, just corrupted supervision. The self-test
asserts that same-ID/different-SIDE rows really do carry different scores, so the check has
teeth rather than being a comment.

WHAT MOAKS GIVES US -- 10 of our 12
  ACL              V00MACLTR                       tear
  Medial Meniscus  V00MMTMA / MMTMB / MMTMP        anterior horn / body / posterior horn
  Lateral Meniscus V00MMTLA / MMTLB / MMTLP        same, lateral
  Medial OA        V00MCM*FM* / V00MCM*TM*         medial femoral + tibial cartilage
  Lateral OA       V00MCM*FL* / V00MCM*TL*         lateral femoral + tibial cartilage
  PF OA            V00MCMPM / MCMPL / MCMFMA / MCMFLA   patella + anterior femur (trochlea)
  Effusion         V00MEFFWK                       whole-knee effusion
  Synovitis        V00MSYIC                        Hoffa's fat-pad signal, scored SEPARATELY
                                                   from effusion -- rare, and our worst target
  Baker's          V00MPOPCYS                      popliteal cyst
  Contusion        V00MBM* (45 subregions)         bone marrow lesions

  NOT AVAILABLE: MCL (MOAKS scores cruciates only, not collaterals) and Fracture (this is an
  osteoarthritis cohort). Those two stay report-supervised.

GRADE THRESHOLDS, AND WHY THEY ARE NOT OBVIOUS
MOAKS grades severity, our targets are binary, so a cut has to be chosen. The one that matters
is the meniscus: **grade 1 is intrasubstance signal, which is NOT a tear** -- it is degeneration
that does not reach the surface. A survey pass quoted 756 abnormal lateral posterior horns by
counting grade >= 1; at grade >= 2 the real figure is 359. Cutting in the wrong place would
have doubled the apparent positives and taught the model that degeneration is a tear.

Default cuts, all overridable:
  meniscal tear   >= 2   grade 1 is intrasubstance signal, not a tear
  cartilage       >= 2   grade 1 is <10% surface area; >= 2 is 10-75%, i.e. real damage
  BML             >= 1   any marrow lesion counts as a contusion signal
  effusion        >= 2   grade 1 is physiological; >= 2 is real distention
  synovitis       >= 1   Hoffa signal is abnormal at any grade
  ACL / Baker's   >= 1   presence items

WHAT THIS DOES NOT SOLVE
OAI-ZIB is **sagittal 3D DESS**, a research sequence. Our competition data is ordinary clinical
multi-sequence T1/PD/fat-sat across three planes. These labels are auxiliary supervision or
pretraining material, NOT a second training set, and ~300 knees is a pilot. Whether DESS
supervision transfers at all is the question the pilot exists to answer -- settle that before
anyone applies for the full OAI archive.

CPU-only: `python build_oai_labels.py --self-test`
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ZIB_INFO = Path(r"data/OAI-ZIB/Zip_Files/info.zip")
ZIB_IMAGES = Path(r"data/OAI-ZIB/Images_Pretrain")
MOAKS_DIR = Path(r"data/3dreasonknee/data/labels")

# MOAKS column families -> our targets. Suffix letters encode the subregion, and the
# medial/lateral letter is what makes this worth having: our lateral targets are the weak ones.
CUTS = dict(meniscus=2, cartilage=2, bml=1, effusion=2, synovitis=1, presence=1)


def moaks_columns(cols):
    """Group the V00* MOAKS columns by which of our targets they inform."""
    v = [c for c in cols if c.startswith("V00")]
    base = lambda c: re.sub(r"\.\d+$", "", c)          # V00MCMPM.1 -> V00MCMPM (size/depth pair)

    def cart(pred):
        return [c for c in v if base(c).startswith("V00MCM") and pred(base(c)[6:])]

    return {
        # Cartilage subregion codes: [P]atella, [F]emur, [T]ibia, then M/L for medial/lateral.
        "Medial OA":  cart(lambda s: s.startswith(("FM", "TM"))),
        "Lateral OA": cart(lambda s: s.startswith(("FL", "TL"))),
        # Patellofemoral = patella (PM/PL) plus the ANTERIOR femur, which is the trochlea.
        "PF OA":      cart(lambda s: s.startswith(("PM", "PL")) or s in ("FMA", "FLA")),
        "Medial Meniscus":  [c for c in v if base(c) in ("V00MMTMA", "V00MMTMB", "V00MMTMP")],
        "Lateral Meniscus": [c for c in v if base(c) in ("V00MMTLA", "V00MMTLB", "V00MMTLP")],
        "Contusion": [c for c in v if base(c).startswith("V00MBM")],
        "Effusion":  [c for c in v if base(c).startswith("V00MEF")],
        "Synovitis": [c for c in v if base(c).startswith("V00MSY")],
        "Baker's":   [c for c in v if "POPCYS" in base(c)],
        "ACL":       [c for c in v if "ACLTR" in base(c)],
    }


TARGET_CUT = {
    "Medial OA": "cartilage", "Lateral OA": "cartilage", "PF OA": "cartilage",
    "Medial Meniscus": "meniscus", "Lateral Meniscus": "meniscus",
    "Contusion": "bml", "Effusion": "effusion", "Synovitis": "synovitis",
    "Baker's": "presence", "ACL": "presence",
}


def load_zib_index():
    """oaizib_NNN -> (OAI SubjectID, side, KL grade). The bridge; without it, nothing joins."""
    with zipfile.ZipFile(ZIB_INFO) as z:
        sub = pd.concat([pd.read_excel(io.BytesIO(z.read(f"info/subInfo_{s}.xlsx")))
                         for s in ("train", "test")], ignore_index=True)
    sub["oaizib"] = sub["CMT-ID"].apply(lambda i: f"oaizib_{int(i):03d}")
    return sub


def load_moaks():
    """All baseline MOAKS reading files, stacked. A knee can be read by several projects."""
    frames = []
    for f in sorted(MOAKS_DIR.glob("taxonomymatch_*MOAKS*.csv")):
        d = pd.read_csv(f, low_memory=False)
        if {"ID", "SIDE"} <= set(d.columns):
            d["_src"] = f.name
            frames.append(d)
    if not frames:
        raise FileNotFoundError(f"no MOAKS csv under {MOAKS_DIR}")
    return pd.concat(frames, ignore_index=True)


def binarise(moaks, cuts=CUTS):
    """One row per (ID, SIDE) with our targets as 0/1.

    A knee read by several projects gives several rows. We take the MAX grade across them:
    the readings are of the same knee, so a finding seen by any reader is a finding, and
    averaging severity grades across differently-scoped projects would be meaningless.
    """
    fam = moaks_columns(moaks.columns)
    out = moaks[["ID", "SIDE"]].copy()
    for tgt, cols in fam.items():
        if not cols:
            out[tgt] = np.nan
            continue
        v = moaks[cols].apply(pd.to_numeric, errors="coerce")
        thr = cuts[TARGET_CUT[tgt]]
        # A knee is positive if ANY subregion reaches the threshold. all-NaN stays NaN, so an
        # unread knee is masked rather than silently called negative -- the same distinction
        # between "absent" and "not assessed" that the report labeller has to make.
        hit = (v >= thr)
        out[tgt] = np.where(v.notna().any(axis=1), hit.any(axis=1).astype(float), np.nan)
    return out.groupby(["ID", "SIDE"], as_index=False).max()


def build():
    zib = load_zib_index()
    lab = binarise(load_moaks())
    m = zib.merge(lab, left_on=["SubjectID", "KneeSide"], right_on=["ID", "SIDE"], how="left")
    m["image"] = m.oaizib.apply(
        lambda s: str(next(ZIB_IMAGES.glob(f"{s}_*.nii.gz"), "")) if ZIB_IMAGES.is_dir() else "")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="oai_zib_labels.csv")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return

    m = build()
    tg = [t for t in TARGET_CUT if t in m.columns]
    got = m[tg].notna().any(axis=1)
    print(f"  OAI-ZIB knees on disk          {len(m)}")
    print(f"  with MOAKS readings joined     {int(got.sum())}")
    print(f"  with an image file found       {int((m.image != '').sum())}\n")
    print(f"  {'target':20s} {'labelled':>9} {'positive':>9} {'prev':>7}")
    for t in tg:
        v = m[t]
        n, p = int(v.notna().sum()), int((v == 1).sum())
        print(f"  {t:20s} {n:>9} {p:>9} {100*p/max(n,1):>6.1f}%")
    print(f"  {'MCL':20s} {0:>9} {'--':>9}   not scored by MOAKS")
    print(f"  {'Fracture':20s} {0:>9} {'--':>9}   not scored by MOAKS")
    m.to_csv(a.out, index=False)
    print(f"\n  wrote {a.out}  ({len(m)} rows)")


def self_test():
    print("build_oai_labels self-test\n")
    zib = load_zib_index()
    assert len(zib) == 507, f"expected 507 OAI-ZIB knees, got {len(zib)}"
    assert zib.oaizib.nunique() == 507 and zib.SubjectID.nunique() == 507
    print(f"  bridge table complete                ok  507 knees, 507 distinct subjects")

    moaks = load_moaks()
    # THE TRAP: same subject, both knees, genuinely different findings. If this fails, joining
    # on ID alone would be harmless -- it does not fail, which is why the side is load-bearing.
    both = moaks.groupby("ID").SIDE.nunique()
    pairs = both[both > 1].index
    assert len(pairs) > 0, "no subject has both knees read; side handling is untested"
    fam = moaks_columns(moaks.columns)
    cols = [c for c in sum(fam.values(), []) if c in moaks.columns][:40]
    diff = 0
    for pid in pairs[:200]:
        g = moaks[moaks.ID == pid]
        a = g[g.SIDE == g.SIDE.min()][cols].apply(pd.to_numeric, errors="coerce").max()
        b = g[g.SIDE == g.SIDE.max()][cols].apply(pd.to_numeric, errors="coerce").max()
        if not a.equals(b):
            diff += 1
    frac = diff / min(len(pairs), 200)
    assert frac > 0.5, f"only {frac:.0%} of both-knee pairs differ — side may not matter"
    print(f"  left/right really differ             ok  {frac:.0%} of both-knee pairs differ")

    lab = binarise(moaks)
    assert lab.duplicated(["ID", "SIDE"]).sum() == 0, "binarise left duplicate knees"
    print(f"  one row per (ID, SIDE)               ok  {len(lab)} knees")

    # the meniscus cut is the one that was got wrong once; assert the two cuts really differ
    strict = binarise(moaks, {**CUTS, "meniscus": 2})["Lateral Meniscus"].sum()
    loose = binarise(moaks, {**CUTS, "meniscus": 1})["Lateral Meniscus"].sum()
    assert loose > strict * 1.3, f"grade cut has no effect: >=1 {loose} vs >=2 {strict}"
    print(f"  meniscus grade cut is load-bearing   ok  >=1 gives {int(loose)}, >=2 gives {int(strict)}")

    m = build()
    got = m[[t for t in TARGET_CUT if t in m.columns]].notna().any(axis=1).sum()
    assert got > 150, f"only {got} knees joined; the bridge is not working"
    print(f"  join produces supervision            ok  {int(got)} of 507 knees labelled")
    print("\n  ALL SELF-TESTS PASSED")


if __name__ == "__main__":
    main()
