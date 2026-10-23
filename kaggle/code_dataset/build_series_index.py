"""
build_series_index.py -- steps B1-B3: what each series IS, which knee it is, and which slot it fills.

WHY THIS FILE EXISTS
The model eats a fixed rack of six SLOTS per study (see SLOTS below), because a finding is
only visible on the sequences that show it. A study, however, arrives as an unordered pile
of 3-14 series (median 5). Something has to decide, per series, "this one is the sagittal
fat-suppressed fluid-sensitive scan", and neither of the two obvious shortcuts works: the
metadata RSNA ships carries one bit between its two flags (B1), and the free-text
SeriesDescription is missing or anonymised to "DummySeriesDesc!" on 17.7% of series, with
777 distinct spellings across the rest. So the answer has to come from the acquisition
PHYSICS -- TR, TE, scanning sequence -- which every series carries whatever the vendor
called it.

Two of the three answers below are silent-failure territory: a mis-assigned slot feeds the
wrong anatomy to a diagnosis head, and a wrong laterality mirrors a knee and swaps medial
for lateral on five of the twelve targets. Both train happily and score badly.

Measured outcome of this pass: laterality is known for 50% of studies from the tag alone
and 99.1% after adding geometry, and the geometric rule agrees with the tag on 98.1% of the
studies where both exist. That last number is the validation -- the rule is checked against
the half of the corpus that carries ground truth for it.

B1  SEQUENCE RECOVERY. `train_series.csv` ships `Fluid_Sensitive` and `Fat_Suppression`,
    which name two physically independent properties -- but they are identical on all
    24,371 rows, so as delivered they carry one bit, not two. The two axes are recoverable
    from the acquisition parameters:

        weighting   T1  : TR <~ 800 ms
                    T2  : TR >~ 800 ms and TE >~ 60 ms
                    PD  : TR >~ 800 ms and TE <~ 60 ms
        (gradient echo breaks the TR rule by design, so ScanningSequence is read first,
         and inversion recovery -- STIR/TIRM -- is its own case)

        fat sat     from ScanOptions / SequenceName / SeriesDescription where the protocol
                    names it

B2  SLOT ASSIGNMENT. Six slots span plane x acquisition, chosen so every one of the twelve
    findings has at least one sequence that shows it well. One series per slot per study;
    1,936 studies have more than one candidate somewhere, broken toward the stack with the
    most slices since a denser stack samples the joint better.

B3  LATERALITY. The `Laterality` tag is absent on half the studies -- by whole vendors, not
    scattered series -- so treating those as left silently mirrors every right knee among
    them, reversing the axis for the five side-defined targets. The patient coordinate
    system supplies it: +x points to the patient's left, so the sign of the image CENTRE's
    x says which knee this is. The centre, not ImagePositionPatient, which is a corner half
    a field of view away -- enough to flip the sign on a knee scanned near the midline.

Outputs series_index.parquet and study_index.parquet. Read-only w.r.t. the DICOM tree.

Usage:
    python build_series_index.py
    python build_series_index.py --limit 500      # smoke test
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

ROOT = Path(__file__).resolve().parent

# Naming the tags, because a bare hex list is unreadable and pydicom needs the numbers.
# Restricting the read to these makes pydicom stop parsing early, which over 24,371 files
# is the difference between minutes and an hour.
#   0018,0080 RepetitionTime      0018,0081 EchoTime          0018,0020 ScanningSequence
#   0018,0021 SequenceVariant     0018,0022 ScanOptions       0018,0024 SequenceName
#   0008,103E SeriesDescription   0020,0032 ImagePositionPatient
#   0020,0037 ImageOrientationPatient                         0028,0030 PixelSpacing
#   0028,0010 Rows                0028,0011 Columns           0018,0050 SliceThickness
#   0020,0060 Laterality          0018,5100 PatientPosition (fetched, not used downstream)
TAGS = [0x00180080, 0x00180081, 0x00180020, 0x00180021, 0x00180022, 0x00180024,
        0x0008103E, 0x00200032, 0x00200037, 0x00280030, 0x00280010, 0x00280011,
        0x00180050, 0x00200060, 0x00185100]

# The six slots: plane x contrast. "Fluid-sensitive" (T2/PD/STIR) makes water bright, so
# fluid, swelling and tears light up; "FS" means fat is additionally suppressed, which stops
# bright fat drowning that signal. Why each slot is here, one line each:
#   SAG_FLUID_FS    the cruciates run front-to-back, so the ACL is read sagittally; also
#                   bone bruise (Contusion) and joint fluid (Effusion), both bright here.
#   COR_FLUID_FS    the MCL runs down the inner side of the joint, and the medial vs lateral
#                   compartments are compared side by side -- both coronal reads.
#   AX_FLUID_FS     the patella faces the trochlea only in the axial plane, so this is the
#                   PF OA slot; Baker cyst (a fluid pouch behind the knee) shows here too.
#   SAG_FLUID_NOFS  the classic meniscus sequence. Fat suppression costs signal-to-noise,
#                   and the dark meniscal wedge reads better against bright fat.
#   COR_T1, SAG_T1  T1 makes fat bright and fluid dark: poor for swelling, good for bone
#                   and marrow -- fracture lines and chronic degenerative change.
# The order is fixed: everything downstream (cache axis 0, model slot indices, SLOT_PRIOR)
# refers to a slot by its POSITION in this list.
SLOTS = ["SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS", "SAG_FLUID_NOFS", "COR_T1", "SAG_T1"]

# Fat suppression carries a different trade name on every scanner: SPAIR/SPIR (Philips),
# TIRM (Siemens), FRFSE-FS (GE), STIR (generic inversion recovery). They all mean the same
# physical thing, so they all have to appear here or one vendor's whole fleet of series
# lands in the wrong slot.
_FS_RX = re.compile(r"\bfs\b|fatsat|fat[_ -]?sat|spair|spir|\bstir\b|tirm|\bfs[_ -]|suppress|"
                    r"\bsat\b|frfse.*fs|\bnex.*fs", re.I)
_STIR_RX = re.compile(r"stir|tirm", re.I)


def read_series(args):
    """One header per series -> acquisition parameters and geometry.

    Reads ONE arbitrary file per series. `files` comes from an unsorted os.scandir and the
    filenames are SOP Instance UIDs, whose order correlates -0.000 with anatomical position
    (that is the whole reason build_slice_order.py exists), so files[len//2] is the middle of
    an arbitrary list, NOT the middle of the stack. Measured on 40 real series, the file this
    picks lands anywhere from one end to the other, median 0.50 but only ~32% near the middle.

    That is fine for what this function needs, and the reason is worth stating so nobody
    "fixes" it into a sort. Acquisition parameters (TR, TE, sequence) are constant across a
    series, so any slice answers those. Geometry is used only for the SIGN of centre_x, which
    decides left versus right -- and the knee does not cross the midline, so every slice in a
    series agrees on that sign. Reading one header instead of sorting the whole series is the
    difference between a fast pass and a slow one over 24,371 series.
    """
    study, series, sdir = args
    try:
        files = [f.name for f in os.scandir(sdir) if f.is_file()]
    except OSError:
        return None
    if not files:
        return None
    try:
        ds = pydicom.dcmread(os.path.join(sdir, files[len(files) // 2]),
                             stop_before_pixels=True, force=True,
                             specific_tags=[pydicom.tag.Tag(t) for t in TAGS])
    except Exception:
        return None

    def g(name, default=None):
        v = getattr(ds, name, None)
        return default if v is None else v

    def flt(name):
        v = g(name)
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def seq_str(name):
        v = g(name)
        if v is None:
            return ""
        return " ".join(str(x) for x in v) if isinstance(v, (list, pydicom.multival.MultiValue)) else str(v)

    rec = dict(
        StudyInstanceUID=study, SeriesInstanceUID=series, n_slices=len(files),
        TR=flt("RepetitionTime"), TE=flt("EchoTime"),
        scanning_sequence=seq_str("ScanningSequence"),
        sequence_variant=seq_str("SequenceVariant"),
        scan_options=seq_str("ScanOptions"),
        sequence_name=seq_str("SequenceName"),
        series_description=seq_str("SeriesDescription"),
        slice_thickness=flt("SliceThickness"),
        rows=int(g("Rows", 0) or 0), cols=int(g("Columns", 0) or 0),
        laterality_tag=(str(g("Laterality", "") or "").strip().upper() or None),
    )
    # DICOM PixelSpacing is [between rows, between columns] -- ps[0] is the vertical step,
    # ps[1] the horizontal one. Swapping them is invisible on the square images that are the
    # common case here, and quietly wrong on the rest.
    ps = g("PixelSpacing")
    rec["pix_row"] = float(ps[0]) if ps is not None and len(ps) >= 2 else None
    rec["pix_col"] = float(ps[1]) if ps is not None and len(ps) >= 2 else None

    # Centre of the image in patient coordinates -- the laterality signal.
    ipp, iop = g("ImagePositionPatient"), g("ImageOrientationPatient")
    rec["centre_x"] = None
    if ipp is not None and iop is not None and len(iop) == 6 and len(ipp) == 3 and rec["pix_row"]:
        try:
            p = np.array([float(x) for x in ipp])
            r_hat = np.array([float(x) for x in iop[:3]])   # along a row (increasing column)
            c_hat = np.array([float(x) for x in iop[3:]])   # along a column (increasing row)
            centre = (p
                      + r_hat * rec["pix_col"] * (rec["cols"] / 2.0)
                      + c_hat * rec["pix_row"] * (rec["rows"] / 2.0))
            rec["centre_x"] = float(centre[0])
        except (TypeError, ValueError):
            pass
    return rec


def classify(row):
    """-> (weighting, fatsat). Sequence type first, then the TR/TE rule (both in ms).

    Order matters. Inversion recovery is tested first because STIR/TIRM suppress fat by
    physics rather than through a named option, and gradient echo second because it uses a
    short TR by design -- put it through the TR rule and every GRE series comes out T1.
    """
    blob = " ".join(str(row.get(k) or "") for k in
                    ("scan_options", "sequence_name", "series_description", "scanning_sequence"))
    fatsat = bool(_FS_RX.search(blob))

    ss = str(row.get("scanning_sequence") or "").upper()
    tr, te = row.get("TR"), row.get("TE")

    if _STIR_RX.search(blob) or ("IR" in ss):
        return "STIR", True          # inversion recovery suppresses fat by construction
    if "GR" in ss and "SE" not in ss:
        return "GRE", fatsat         # gradient echo: TR is short by design, rule cannot apply
    if tr is None or te is None:
        return "UNK", fatsat
    if tr <= 800:
        return "T1", fatsat
    return ("T2" if te >= 60 else "PD"), fatsat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=32)
    # A guard band, not a measurement. A knee sits well lateral of the patient midline, so
    # a study whose median image centre lands within a centimetre of x=0 is not confidently
    # either side. Leaving it unresolved is cheap (build_cache leaves unknown-side studies
    # unmirrored); guessing wrong mirrors the anatomy of the five side-defined targets.
    ap.add_argument("--deadband", type=float, default=8.0,
                    help="mm around the midline where laterality is left unresolved")
    a = ap.parse_args()

    base = ROOT / f"{a.split}_series"
    tser = pd.read_csv(ROOT / f"{a.split}_series.csv")
    jobs = [(r.StudyInstanceUID, r.SeriesInstanceUID,
             str(base / r.StudyInstanceUID / r.SeriesInstanceUID)) for r in tser.itertuples()]
    if a.limit:
        jobs = jobs[:a.limit]

    print(f"B1-B3: reading {len(jobs)} series headers ({a.threads} threads)", flush=True)
    t0 = time.time()
    recs = []
    with ThreadPoolExecutor(max_workers=a.threads) as ex:
        for i, r in enumerate(ex.map(read_series, jobs), 1):
            if r:
                recs.append(r)
            if i % 5000 == 0:
                print(f"  {i}/{len(jobs)}  ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(recs)
    print(f"  read {len(df)} series in {time.time() - t0:.0f}s\n")

    df = df.merge(tser, on=["StudyInstanceUID", "SeriesInstanceUID"], how="left")

    # ---------------------------------------------------------------- B1
    print("=" * 78)
    print("B1  SEQUENCE RECOVERY")
    print("=" * 78)
    cls = df.apply(classify, axis=1, result_type="expand")
    df["weighting"], df["fatsat"] = cls[0], cls[1]
    print("  recovered weighting:")
    for k, v in df.weighting.value_counts().items():
        print(f"    {k:<6} {v:>7}  ({100 * v / len(df):.1f}%)")
    print(f"\n  recovered fat-sat: {int(df.fatsat.sum())} / {len(df)} "
          f"({100 * df.fatsat.mean():.1f}%)")
    print(f"  provided Fat_Suppression: {int(df.Fat_Suppression.sum())} "
          f"({100 * df.Fat_Suppression.mean():.1f}%)")
    agree = (df.fatsat.astype(int) == df.Fat_Suppression).mean()
    print(f"  agreement with provided flag: {100 * agree:.1f}%")

    print("\n  provided Fluid_Sensitive x recovered weighting:")
    print(pd.crosstab(df.Fluid_Sensitive, df.weighting).to_string().replace("\n", "\n    "))
    print("\n  -> the provided flag collapses T1 vs PD/T2 AND fat-sat into one bit.")
    print("     Recovering weighting separately is what makes COR_T1 / SAG_T1 distinguishable")
    print("     from the fluid-sensitive slots at all.")

    # ---------------------------------------------------------------- B3
    print("\n" + "=" * 78)
    print("B3  LATERALITY")
    print("=" * 78)
    # One decision per STUDY, from the median over its series, so a localiser or an oddly
    # positioned series cannot flip the sign for the whole knee on its own.
    st = df.groupby("StudyInstanceUID").agg(
        centre_x=("centre_x", "median"),
        tag=("laterality_tag", lambda s: next((x for x in s if x in ("L", "R")), None)),
    ).reset_index()
    st["side_geom"] = np.where(st.centre_x.isna(), None,
                               np.where(st.centre_x < -a.deadband, "R",
                                        np.where(st.centre_x > a.deadband, "L", None)))
    resolved = st.side_geom.notna()
    print(f"  studies with a Laterality tag      : {int(st.tag.notna().sum())} / {len(st)}")
    print(f"  studies resolved from geometry     : {int(resolved.sum())} / {len(st)} "
          f"({100 * resolved.mean():.1f}%)")
    both = st[st.tag.notna() & resolved]
    if len(both):
        acc = (both.tag == both.side_geom).mean()
        print(f"  geometry vs tag, where both exist  : {100 * acc:.1f}% agree (n={len(both)})")
        print("  -> this is the validation that matters: the geometric rule is checked")
        print("     against the half of the corpus that carries ground truth for it.")
    # The tag wins where it exists; geometry only fills the gap. The agreement printed above
    # validates the geometric rule -- it does not license overriding a recorded tag with it.
    st["side"] = st.tag.where(st.tag.notna(), st.side_geom)
    print(f"  final laterality known             : {int(st.side.notna().sum())} / {len(st)} "
          f"({100 * st.side.notna().mean():.1f}%)")
    print(f"  distribution: {dict(st.side.value_counts(dropna=False))}")

    # ---------------------------------------------------------------- B2
    print("\n" + "=" * 78)
    print("B2  SLOT ASSIGNMENT")
    print("=" * 78)
    fluid = df.weighting.isin(["T2", "PD", "STIR"])
    t1 = df.weighting.eq("T1")
    plane = df.Anatomical_Plane
    df["slot"] = None
    df.loc[(plane == "Sagittal") & fluid & df.fatsat, "slot"] = "SAG_FLUID_FS"
    df.loc[(plane == "Coronal") & fluid & df.fatsat, "slot"] = "COR_FLUID_FS"
    df.loc[(plane == "Axial") & fluid & df.fatsat, "slot"] = "AX_FLUID_FS"
    df.loc[(plane == "Sagittal") & fluid & ~df.fatsat, "slot"] = "SAG_FLUID_NOFS"
    df.loc[(plane == "Coronal") & t1, "slot"] = "COR_T1"
    df.loc[(plane == "Sagittal") & t1, "slot"] = "SAG_T1"

    print(f"  series assigned to a slot: {int(df.slot.notna().sum())} / {len(df)} "
          f"({100 * df.slot.notna().mean():.1f}%)")
    print(f"  unassigned: {int(df.slot.isna().sum())} "
          f"(GRE/UNK weighting, or axial non-fatsat which has no slot)")

    # one series per slot per study; ties -> most slices
    df["_rank"] = df.groupby(["StudyInstanceUID", "slot"]).n_slices.rank(
        ascending=False, method="first")
    df["chosen"] = df.slot.notna() & (df._rank == 1)
    df = df.drop(columns="_rank")

    piv = (df[df.chosen].pivot_table(index="StudyInstanceUID", columns="slot",
                                     values="n_slices", aggfunc="size").fillna(0))
    for s in SLOTS:
        if s not in piv.columns:
            piv[s] = 0
    piv = piv[SLOTS]
    print(f"\n  {'slot':<18} {'studies with it':>16}")
    for s in SLOTS:
        print(f"  {s:<18} {int((piv[s] > 0).sum()):>10} / {len(piv)}  "
              f"({100 * (piv[s] > 0).mean():>5.1f}%)")
    nslot = (piv > 0).sum(axis=1)
    print(f"\n  slots filled per study: {dict(sorted(nslot.value_counts().items()))}")
    print(f"  studies with 0 slots: {int((nslot == 0).sum())}  <- these would be unusable")

    st = st.merge(nslot.rename("n_slots"), left_on="StudyInstanceUID",
                  right_index=True, how="left")
    st["n_slots"] = st.n_slots.fillna(0).astype(int)

    suffix = "" if a.split == "train" else "_test"
    df.to_parquet(ROOT / f"series_index{suffix}.parquet", index=False)
    st.to_parquet(ROOT / f"study_index{suffix}.parquet", index=False)
    print(f"\n-> series_index{suffix}.parquet ({len(df)} series)")
    print(f"-> study_index{suffix}.parquet  ({len(st)} studies)")


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()
