"""
build_series_index_v2.py -- S1: recover the 12.3% of series that v1 assigns to no slot.

WHAT V1 THROWS AWAY (measured on series_index.parquet, 24,371 series / 4,407 studies)
3,003 series (12.3%) get slot=None and are never decoded. Mean slots filled per study is
4.451 of 6. Every study on disk holds all three planes, yet v1 reads all three on only
89.6% of them: 290 studies get no axial read, 157 no coronal, 3 no sagittal.

This file is a POST-PROCESSOR over the header parquet that build_series_index.py already
writes. It re-derives weighting / fatsat / slot / chosen from the RAW DICOM columns in that
parquet (TR, TE, ScanningSequence, ...) and does not re-read the DICOM tree. That is the
reason it is a separate file rather than an edit: build_series_index.py owns the expensive
header pass and the laterality logic, both of which are correct and are reused unchanged.

FOUR CHANGES, each justified by a measurement printed by --report.

1. FAT SATURATION is the UNION of the shipped flag and a de-negated regex.
   train_series.csv / test_series.csv ship Fluid_Sensitive and Fat_Suppression -- present
   in BOTH CSVs, so this is legal at test time. v1's docstring dismisses them because they
   are identical on all 24,371 rows and therefore carry one bit, not two -- true, but that
   one bit is exactly "this is a fat-suppressed fluid-sensitive series", which is the bit
   three of the six slots are keyed on. Flag and regex disagree on 445 series (1.8%), and
   each source is right where the other is wrong, so neither can be taken alone:
     - 379 the regex MISSED and the flag caught: 265 t2_de3d_we_tra_Patella_fit_T (water
       excitation -- a spectral fat-suppression method the regex has no word for), 75
       "SMART FAT" / "Water: SMART FAT" (GE Dixon water image), 29 "Sag PD frFSE" (GE
       fat-sat naming without the letters "fs").
     - 12 the regex INVENTED by matching the letters inside a NEGATION: "SAG PD NON
       FATSAT", "SAG PD NO FS LT", "SAG PD NON FS", "SAG PD NO FS RT". _NEG_RX kills these.
     - 11 fluid-weighted series the FLAG misses and the regex catches: "SAG DP SPIR",
       "LT.SAG PDW_TSE_SPAIR", "pd_tse_fs_cor_320", "AXI T2 SPIR P/ ROTULA" and others
       whose ScanOptions literally reads "FS". Trusting the flag alone demotes these out of
       the *_FLUID_FS slots, and one study (...44321626...) then loses SAG_FLUID_FS
       outright because the demoted series collides with its real non-fat-sat twin.
     - the remaining 43 regex-only hits are T1, where no slot tests fat-sat, so they are
       free either way.
   The union recovers all 379 and loses none of the 11. Its only failure mode is a series
   that is fat-suppressed according to neither the curator nor its own protocol name.

2. FLUID-SENSITIVE admits gradient echo when the flag says so. v1's classify() returns
   "GRE" for gradient echo and the slot rules test weighting in (T2, PD, STIR), so all 765
   GRE series are unreachable by every slot rule -- verified, 765 of 765 unassigned. 270 of
   them are flagged Fluid_Sensitive=1 AND Fat_Suppression=1 by the curators; 269 are axial
   water-excited 3D double-echo (t2_de3d_we_tra_Patella_fit_T, a DESS-family cartilage
   scan). They are fluid-bright, fat-dark, axial, and they land in 253 studies that
   currently get NO axial read at all. Those fold into AX_FLUID_FS.
   The other 495 GRE are flagged 0 and are a genuinely different contrast (t2_me2d_sag
   T2* MEDIC, t1_vibe_we_tra_cartilage, 3D_WATSc) -- see change 3.

3. THREE NEW SLOTS, not a silent fold. Folding a non-fat-sat series into a *_FLUID_FS slot
   would make one slot mean two contrasts, and the rack already refuses to do that once:
   SAG_FLUID_NOFS exists precisely because non-fat-sat PD is a different image from PD-FS.
   The same argument gives the coronal and axial planes their own NOFS slots, and gives the
   495 non-fat-sat GRE their own:
     COR_FLUID_NOFS  1,127 series / 829 studies -- 155 of which read NO coronal today
     AX_FLUID_NOFS     538 series / 530 studies -- the axial non-fat-sat PD/T2 stacks
     GRE_NOFS          495 series / 494 studies -- T2* and 3D cartilage GRE. This is the
                       weakest of the three: 380 of the 383 sagittal-GRE studies already
                       read a sagittal series, so it buys contrast, not coverage.
   REJECTED: AX_T1 (573 series / 546 studies). 545 of those 546 studies already read an
   axial series, so it restores the axial plane for exactly ONE study, and axial T1 shows
   neither fluid nor the medial/lateral compartments better than the axial slots we now
   fill. Coverage without new information is not worth a slot.

4. THE TIE-BREAK IS MADE DETERMINISTIC. v1 ranks candidates with
   n_slices.rank(ascending=False, method="first"), so most-slices wins (verified: the
   chosen series is the group maximum in 1,703 of 1,703 multi-candidate groups) -- but
   method="first" falls back to DataFrame row order, which is train_series.csv order,
   which is sorted by UID. On the 1,134 groups where the maximum is TIED (926 studies,
   21.0%) the winner is therefore decided by UID lexicographic order, i.e. by a hash.
   v2 sorts on (most slices, finest in-plane pixel spacing, thinnest slice, UID) so the
   choice is a stated quality order and only the last key is arbitrary.

   NOT COMBINED, and here is the measurement that decided it. Of the 1,134 tied groups:
     443 (39%) have the two echo times differing by more than 20 ms -- a PD echo and a T2
                echo of one dual-echo acquisition. Averaging them destroys both contrasts.
     475 (42%) have different in-plane pixel spacing -- they are not even on the same
                sampling grid, so there is no array-index sense in which they can be added.
     362 (32%) match on TE, pixel spacing and slice thickness: true repeats, the only case
                where averaging would do anything but blur.
   Averaging is wrong for 68% of the groups it would touch. The runner-up is instead kept
   addressable: chosen_rank is written for every candidate, and build_cache_v2.py --alt
   can decode rank-2 into a sidecar volume for an inference-time ensemble, which needs no
   model change because the array shape is unchanged. Off by default.

Outputs series_index_v2.parquet (+ _test), reusing study_index.parquet unchanged --
laterality is not touched here.

Usage:
    python build_series_index_v2.py --report
    python build_series_index_v2.py --self-test
    python build_series_index_v2.py --self-test --neutralise classifier   # must FAIL
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

SLOTS_V1 = ["SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS", "SAG_FLUID_NOFS", "COR_T1", "SAG_T1"]
NEW_SLOTS = ["COR_FLUID_NOFS", "AX_FLUID_NOFS", "GRE_NOFS"]
SLOTS_V2 = SLOTS_V1 + NEW_SLOTS

# Same vocabulary as v1 -- kept verbatim so the fallback path behaves identically where the
# shipped flag is absent.
_FS_RX = re.compile(r"\bfs\b|fatsat|fat[_ -]?sat|spair|spir|\bstir\b|tirm|\bfs[_ -]|suppress|"
                    r"\bsat\b|frfse.*fs|\bnex.*fs", re.I)
# Deliberately UNBOUNDED, matching v1's _STIR_RX. _FS_RX above spells the stem "\bstir\b",
# which cannot match "STIR_aTSE_TR_16 SENSE" because underscore is a word character and the
# trailing \b therefore fails. Inversion recovery suppresses fat by construction, so this is
# a physics rule and must not depend on how the protocol name is punctuated.
_STIR_RX = re.compile(r"stir|tirm", re.I)
# ...plus the negation the v1 regex walks straight into. "SAG PD NON FATSAT" contains
# "fatsat" and v1 calls it fat-suppressed, which is the opposite of what the radiographer
# typed. 12 series in train.
_NEG_RX = re.compile(r"non[\s_.-]*fat|non[\s_.-]*fs\b|\bno[\s_.-]+f\.?s\.?\b|\bnofs\b|"
                     r"without[\s_.-]*fat", re.I)


def _blob(df):
    cols = ["scan_options", "sequence_name", "series_description", "scanning_sequence"]
    out = df[cols[0]].fillna("").astype(str)
    for c in cols[1:]:
        out = out + " " + df[c].fillna("").astype(str)
    return out


def derive(df, neutralise=None):
    """Add fatsat_v2 / fluid_v2 columns. `weighting` is taken from v1 unchanged."""
    blob = _blob(df)
    rx = blob.str.contains(_FS_RX) & ~blob.str.contains(_NEG_RX)

    if neutralise == "classifier":
        # v1 behaviour: regex only (negation bug included), gradient echo never fluid.
        df["fatsat_v2"] = blob.str.contains(_FS_RX)
        df["fluid_v2"] = df.weighting.isin(["T2", "PD", "STIR"])
        return df

    # df.get(col) returns None when the column is absent, and pd.to_numeric(None) yields a
    # bare numpy scalar whose .fillna does not exist -- so a test split that simply omits
    # these columns crashed here rather than degrading. Fall back to an all-zero Series so a
    # missing column means "the curator told us nothing", which the regex rules then decide.
    def _flag(col):
        if col not in df.columns:
            return pd.Series(0.0, index=df.index)
        return pd.to_numeric(df[col], errors="coerce")

    flag_fs = _flag("Fat_Suppression")
    flag_fl = _flag("Fluid_Sensitive")
    # UNION, not the flag alone. The flag is wrong the other way on 11 fluid-weighted
    # series -- `SAG DP SPIR`, `LT.SAG PDW_TSE_SPAIR`, `pd_tse_fs_cor_320`, and others
    # whose ScanOptions literally reads "FS" -- and flag-only demotes them out of the
    # *_FLUID_FS slots. One study (…44321626…) then loses SAG_FLUID_FS entirely, because
    # its demoted `SAG DP SPIR` collides with a real non-fat-sat `SAG DP` in
    # SAG_FLUID_NOFS and one of the two is dropped. Taking either source as authoritative
    # alone loses data; the union loses none, and the only way the union can be wrong is a
    # series that is fat-suppressed by neither the curator nor its own protocol name.
    # Inversion recovery suppresses fat BY CONSTRUCTION, and v1 encodes that as a rule that
    # fires independently of _FS_RX (build_series_index.py:210). Dropping it here looked safe
    # only because all 133 affected series happen to carry Fat_Suppression=1 in train -- so the
    # flag rescued them and the regression was invisible. That is a property of the train
    # metadata, not a guarantee: if the hidden test split curates the flag differently, 133
    # STIR series silently become "not fat-suppressed" and leave the *_FLUID_FS slots.
    #
    # The reason _FS_RX misses them at all is that it spells the stem "\bstir\b" while the
    # descriptions read "STIR_aTSE_TR_16 SENSE" -- underscore is a word character, so the
    # trailing \b never matches. v1's _STIR_RX has no boundaries and does match.
    ir = blob.str.contains(_STIR_RX) | df.get(
        "scanning_sequence", pd.Series("", index=df.index)).fillna("").str.contains("IR")
    df["fatsat_v2"] = (flag_fs.fillna(0) > 0) | rx | ir
    # Physics decides the weighting; the flag is allowed to promote GRE into the fluid
    # family and nothing else, so a real T1 series can never be pulled out of a T1 slot.
    df["fluid_v2"] = (df.weighting.isin(["T2", "PD", "STIR"])
                      | (df.weighting.eq("GRE") & (flag_fl.fillna(0) > 0)))
    return df


def assign(df, slots, neutralise=None):
    """Add `slot_v2`. `slots` is the rack; a slot absent from it is simply never filled."""
    fl, t1 = df.fluid_v2, df.weighting.eq("T1")
    fs = df.fatsat_v2
    pl = df.Anatomical_Plane
    rules = [
        ("SAG_FLUID_FS",   (pl == "Sagittal") & fl & fs),
        ("COR_FLUID_FS",   (pl == "Coronal") & fl & fs),
        ("AX_FLUID_FS",    (pl == "Axial") & fl & fs),
        ("SAG_FLUID_NOFS", (pl == "Sagittal") & fl & ~fs),
        ("COR_T1",         (pl == "Coronal") & t1),
        ("SAG_T1",         (pl == "Sagittal") & t1),
        ("COR_FLUID_NOFS", (pl == "Coronal") & fl & ~fs),
        ("AX_FLUID_NOFS",  (pl == "Axial") & fl & ~fs),
        # Non-fat-sat gradient echo, any plane. Sagittal T2* MEDIC and the 3D cartilage
        # volumes both live here; one slot rather than three because 494 studies split
        # three ways would leave each one filled on ~8% of the corpus.
        ("GRE_NOFS",       df.weighting.eq("GRE") & ~fl),
    ]
    df["slot_v2"] = None
    for name, m in rules:
        if name in slots:
            df.loc[m & df.slot_v2.isna(), "slot_v2"] = name

    if neutralise == "dupslot":
        # Declare the new slots but never fill them: the rack looks bigger and nothing new
        # is decoded. This is the "aggregator returns one input" failure mode.
        df.loc[df.slot_v2.isin(NEW_SLOTS), "slot_v2"] = None
    return df


def choose(df, neutralise=None):
    """Add `chosen_rank` (1 = winner) and `chosen_v2` per (study, slot).

    Quality order, most decisive first:
      n_slices desc   -- a denser stack samples the joint better (v1's only criterion)
      pix_row asc     -- the cache resamples 130 mm onto 336 px = 0.387 mm/px, so a series
                         acquired at 0.14 mm/px carries real detail up to that pitch and one
                         at 0.47 mm/px does not
      slice_thickness asc
      SeriesInstanceUID asc  -- determinism backstop only
    """
    d = df[df.slot_v2.notna()].copy()
    if neutralise == "tiebreak":
        # v1: rank on slices alone, ties resolved by DataFrame row order.
        d["chosen_rank"] = d.groupby(["StudyInstanceUID", "slot_v2"]).n_slices.rank(
            ascending=False, method="first").astype(int)
    else:
        d["_px"] = pd.to_numeric(d.pix_row, errors="coerce").fillna(np.inf)
        d["_th"] = pd.to_numeric(d.slice_thickness, errors="coerce").fillna(np.inf)
        d = d.sort_values(["StudyInstanceUID", "slot_v2", "n_slices", "_px", "_th",
                           "SeriesInstanceUID"],
                          ascending=[True, True, False, True, True, True], kind="mergesort")
        d["chosen_rank"] = d.groupby(["StudyInstanceUID", "slot_v2"]).cumcount() + 1
        d = d.drop(columns=["_px", "_th"])
    df = df.merge(d[["SeriesInstanceUID", "chosen_rank"]], on="SeriesInstanceUID", how="left")
    df["chosen_v2"] = df.chosen_rank.eq(1).fillna(False)
    return df


def build(df, slots=SLOTS_V2, neutralise=None):
    df = derive(df.copy(), neutralise)
    df = assign(df, slots, neutralise)
    return choose(df, neutralise)


# --------------------------------------------------------------------------- reporting
def _fill(df, slot_col, chosen_col, slots, studies):
    ch = df[df[chosen_col].astype(bool)]
    t = ch.pivot_table(index="StudyInstanceUID", columns=slot_col, values="n_slices",
                       aggfunc="size")
    t = t.reindex(index=studies, columns=slots).fillna(0)
    return t > 0


def _planes(df, chosen_col, studies):
    ch = df[df[chosen_col].astype(bool)]
    s = ch.groupby("StudyInstanceUID").Anatomical_Plane.apply(set).reindex(studies)
    return s.apply(lambda x: x if isinstance(x, set) else set())


def report(df2):
    studies = sorted(df2.StudyInstanceUID.unique())
    n = len(studies)
    f1 = _fill(df2, "slot", "chosen", SLOTS_V1, studies)
    f2 = _fill(df2, "slot_v2", "chosen_v2", SLOTS_V2, studies)
    n1, n2 = f1.sum(axis=1), f2.sum(axis=1)

    print("=" * 78)
    print("S1  RECOVERY REPORT   (v1 = build_series_index.py, v2 = this file)")
    print("=" * 78)
    print(f"  series with a slot   v1 {int(df2.slot.notna().sum()):>6}  "
          f"v2 {int(df2.slot_v2.notna().sum()):>6}   "
          f"(+{int(df2.slot_v2.notna().sum() - df2.slot.notna().sum())})")
    print(f"  series still orphan  v1 {int(df2.slot.isna().sum()):>6}  "
          f"v2 {int(df2.slot_v2.isna().sum()):>6}")
    print(f"  series DECODED       v1 {int(df2.chosen.sum()):>6}  "
          f"v2 {int(df2.chosen_v2.sum()):>6}   "
          f"(+{int(df2.chosen_v2.sum() - df2.chosen.sum())})")
    print()
    print(f"  {'slot':<16} {'v1 studies':>13} {'v2 studies':>13}   delta")
    for s in SLOTS_V2:
        a = int(f1[s].sum()) if s in f1 else 0
        b = int(f2[s].sum())
        tag = "  <- NEW" if s in NEW_SLOTS else ""
        print(f"  {s:<16} {a:>7} {100*a/n:>5.1f}% {b:>7} {100*b/n:>5.1f}%  "
              f"{b-a:>+6}{tag}")
    print()
    print(f"  mean slots per study        v1 {n1.mean():.3f} / 6      "
          f"v2 {n2.mean():.3f} / {len(SLOTS_V2)}")
    print(f"  mean slots/study on the v1 RACK ONLY (no model change): "
          f"{f2[SLOTS_V1].sum(axis=1).mean():.3f} / 6")
    print(f"  studies gaining >=1 slot    {int((n2 > n1).sum())} / {n} "
          f"({100*(n2 > n1).mean():.1f}%)")
    print(f"  studies gaining >=1 slot WITHIN the v1 rack (no model change needed): "
          f"{int((f2[SLOTS_V1].sum(axis=1) > n1).sum())}")
    print(f"  studies losing a slot       {int((n2 < n1).sum())}")
    print()
    p1, p2 = _planes(df2, "chosen", studies), _planes(df2, "chosen_v2", studies)
    for p in ["Sagittal", "Coronal", "Axial"]:
        a = sum(p in s for s in p1)
        b = sum(p in s for s in p2)
        print(f"  studies reading {p:<9} v1 {a:>5} ({100*a/n:>5.1f}%)  "
              f"v2 {b:>5} ({100*b/n:>5.1f}%)   {b-a:>+5}")
    gained = sum(1 for x, y in zip(p1, p2) if len(y) > len(x))
    print(f"  mean planes read per study  v1 {np.mean([len(s) for s in p1]):.3f}      "
          f"v2 {np.mean([len(s) for s in p2]):.3f}")
    print(f"  studies that gain a PLANE they could not read at all: {gained} "
          f"({100*gained/n:.1f}%)  <- the number that matters")
    print()
    both = df2[df2.chosen & df2.chosen_v2]
    moved = int((both.slot != both.slot_v2).sum())
    dropped = int((df2.chosen & ~df2.chosen_v2).sum())
    print(f"  CHURN on the existing rack: {moved} decoded series change slot, "
          f"{dropped} decoded in v1 are no longer rank 1 in v2")
    print(f"  runner-up series available for an --alt ensemble: "
          f"{int((df2.chosen_rank > 1).sum())} over "
          f"{df2[df2.chosen_rank > 1].StudyInstanceUID.nunique()} studies")
    return f1, f2, n1, n2


# --------------------------------------------------------------------------- self-test
def self_test(neutralise=None):
    """Fails if the recovery mechanism is neutralised. Pure CPU, ~20 s.

    Each check names the no-op it is built to catch, because this project has repeatedly
    shipped tests that a do-nothing implementation passes.
    """
    src = ROOT / "series_index.parquet"
    df = pd.read_parquet(src)
    df2 = build(df, SLOTS_V2, neutralise)
    studies = sorted(df2.StudyInstanceUID.unique())
    n = len(studies)
    f1 = _fill(df2, "slot", "chosen", SLOTS_V1, studies)
    f2 = _fill(df2, "slot_v2", "chosen_v2", SLOTS_V2, studies)
    n1, n2 = f1.sum(axis=1), f2.sum(axis=1)
    fails = []

    def check(ok, label, got, want):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:<57} {got}  (need {want})")
        if not ok:
            fails.append(label)

    print(f"build_series_index_v2 self-test  (neutralise={neutralise})")
    print(f"  source {src.name}: {len(df)} series / {n} studies\n")

    # 1. The orphan pile actually shrinks. A no-op classifier leaves it at 3,003.
    orph = int(df2.slot_v2.isna().sum())
    check(orph <= 700, "orphan series after v2 (v1 = 3003)", orph, "<= 700")

    # 2. The fat-sat correction alone, INSIDE the six original slots. This is the check the
    #    'classifier' no-op fails: without the flag, AX_FLUID_FS stays at 4117 studies.
    ax = int(f2["AX_FLUID_FS"].sum())
    check(ax >= 4350, "studies with AX_FLUID_FS (v1 = 4117)", ax, ">= 4350")

    # 3. The three new slots are really filled -- and by series v1 assigned to NO slot, not
    #    by re-pointing at one already in the rack. 'dupslot' fails here.
    for s in NEW_SLOTS:
        got = int(f2[s].sum())
        check(got >= 400, f"studies with {s}", got, ">= 400")
    newly = df2[df2.chosen_v2 & df2.slot_v2.isin(NEW_SLOTS)]
    fresh = int(newly.slot.isna().sum())
    check(fresh >= 1700 and fresh / max(len(newly), 1) > 0.95,
          "new-slot winners that v1 assigned to no slot", f"{fresh}/{len(newly)}",
          ">= 1700 and > 95%")

    # 4. Plane coverage: the point of the exercise.
    p1, p2 = _planes(df2, "chosen", studies), _planes(df2, "chosen_v2", studies)
    gained = sum(1 for x, y in zip(p1, p2) if len(y) > len(x))
    check(gained >= 380, "studies gaining a plane they could not read", gained, ">= 380")

    # 5. Nothing is lost. A 'recovery' that also drops existing content is not a recovery.
    check(int((n2 < n1).sum()) == 0, "studies losing a slot", int((n2 < n1).sum()), "== 0")

    # 6. THE TIE-BREAK. Shuffle the input rows and rebuild: a deterministic quality order
    #    returns the identical winner set, v1's method="first" does not. This is the check
    #    the 'tiebreak' no-op fails, and it fails it for the right reason -- the winner
    #    depends on row order, which is UID order, which is a hash.
    sh = df.sample(frac=1.0, random_state=12345).reset_index(drop=True)
    df2s = build(sh, SLOTS_V2, neutralise)
    a = set(df2[df2.chosen_v2].SeriesInstanceUID)
    b = set(df2s[df2s.chosen_v2].SeriesInstanceUID)
    check(a == b, "winner set is invariant to input row order", f"{len(a ^ b)} differ", "== 0")

    # 7. T1 series must not be pulled out of the T1 slots by the flag.
    t1 = df2[df2.weighting.eq("T1") & df2.chosen]
    kept = int(t1.slot_v2.isin(["COR_T1", "SAG_T1"]).sum())
    tot = int(t1.slot.isin(["COR_T1", "SAG_T1"]).sum())
    check(kept == tot, "decoded T1 winners still in a T1 slot", f"{kept}/{tot}", "equal")

    print(f"\n  {'ALL CHECKS PASSED' if not fails else 'FAILED: ' + '; '.join(fails)}")
    return 0 if not fails else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--neutralise", default=None,
                    choices=["classifier", "tiebreak", "dupslot"],
                    help="deliberately break one mechanism; --self-test must then FAIL")
    ap.add_argument("--rack", type=int, default=9, choices=[6, 9],
                    help="6 = classifier fixes only, drop-in for the current 6-slot model")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test(a.neutralise)

    suffix = "" if a.split == "train" else "_test"
    df = pd.read_parquet(ROOT / f"series_index{suffix}.parquet")
    slots = SLOTS_V1 if a.rack == 6 else SLOTS_V2
    df2 = build(df, slots, a.neutralise)
    if a.report:
        report(df2)
    if a.write:
        out = ROOT / f"series_index_v2{suffix}.parquet"
        df2.to_parquet(out, index=False)
        print(f"\n-> {out.name}  ({len(df2)} series, rack of {len(slots)})")
    return 0


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    sys.exit(main())
