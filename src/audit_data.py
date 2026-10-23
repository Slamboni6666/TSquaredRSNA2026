"""
audit_data.py -- Non-destructive audit of the RSNA 2026 Knee Abnormality Detection download.

WHY THIS FILE EXISTS
It runs before any modelling, because the shape of this dataset decides the architecture
rather than the other way round. Four questions, in priority order:

  1. How many studies carry gold labels?
  2. What are the exact 12 target names, and what is the submission layout?
  3. Do TEST studies have reports?   (decides whether the architecture inverts)
  4. What does the DICOM tree actually look like, and what groups CV folds?

WHAT IT FOUND, and what each answer forced. These are now assumptions baked into every
other file in the project; this is where they came from.

  58 of 4,407 studies are fully labelled (1.3%). Supervised training on the labels alone is
      hopeless, so the radiology reports have to become the training signal.
  TEST STUDIES HAVE NO REPORTS -- no Report column anywhere, no sidecars. The report reader
      is therefore a TRAINING-TIME teacher only and the deployed model is vision-only. This
      one answer is why the project is a distillation rather than a text model.
  The 58 are ENRICHED, not a random draw (mean positive rate 0.345, far above clinical base
      rates). AUC is prevalence-invariant, so they still rank models honestly -- but any
      probability calibration fitted on them inherits the enriched prior. Rank with them;
      never fit on them.
  Site identity is anonymised away and scanner fingerprints are too granular to group folds
      by (120 fingerprints, median 1 study each, in a 200-study sample). manufacturer|model
      gives 46 groups with the largest at 16.8%; validation.py crosses that with report
      language to get the folds actually used.

DO NOT COMMIT THIS SCRIPT'S OUTPUT. It prints radiology reports verbatim (one sample per
detected language) and real study/series UIDs, and audit_results.json contains both. That
is exactly how audit_report.txt reached a PUBLIC repo once already -- it looks like a log
file -- which is why check_repo_clean.py now exists. audit_out/, audit_results.json and
audit_report.txt are all in .gitignore; leave them there.

Writes a human report to stdout and audit_results.json alongside this file.
Reads only; never modifies or moves downloaded data.

Usage:
    python audit_data.py                    # default: 200-study DICOM sample, full file walk
    python audit_data.py --sample 400
    python audit_data.py --no-walk          # skip the 819k-file walk (fast re-run)
"""

import argparse
import collections
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "audit_out"
RESULTS_PATH = ROOT / "audit_results.json"

R = {}  # machine-readable results


# ----------------------------------------------------------------------------- helpers
def hdr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def human(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def jsonable(o):
    """numpy/pandas -> plain python, so json.dump doesn't choke."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, Path):
        return str(o)
    if o is None or isinstance(o, (str, int, float, bool)):
        return o
    return str(o)


def describe(vals, name):
    """Distribution, not just the mean -- the brief asks for this explicitly."""
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return {}
    q = np.percentile(a, [0, 5, 25, 50, 75, 95, 100])
    d = dict(
        n=int(a.size), mean=round(float(a.mean()), 3),
        min=float(q[0]), p5=float(q[1]), p25=float(q[2]), p50=float(q[3]),
        p75=float(q[4]), p95=float(q[5]), max=float(q[6]),
    )
    print(f"  {name:<28} n={d['n']:<7} mean={d['mean']:<9} "
          f"min/p25/med/p75/max = {d['min']:g} / {d['p25']:g} / {d['p50']:g} / {d['p75']:g} / {d['max']:g}")
    return d


# ----------------------------------------------------------------------------- 2.1 inventory
def section_inventory(do_walk):
    hdr("2.1  INVENTORY")

    print("\n-- top-level entries --")
    top = []
    for p in sorted(ROOT.iterdir()):
        if p.is_file():
            sz = p.stat().st_size
            top.append(dict(name=p.name, kind="file", bytes=sz))
            print(f"  {p.name:<45} {human(sz):>12}")
        else:
            n = sum(1 for _ in p.iterdir()) if p.name != "audit_out" else 0
            top.append(dict(name=p.name, kind="dir", immediate_children=n))
            print(f"  {p.name + '/':<45} {n:>7} immediate children")
    R["top_level"] = top

    if not do_walk:
        print("\n(--no-walk given: skipping recursive file walk)")
        return

    print("\n-- recursive walk (this is the ~819k-file question) --")
    ext_count = collections.Counter()
    ext_bytes = collections.Counter()
    per_root = {}

    for sub in ["train_series", "test_series"]:
        d = ROOT / sub
        if not d.exists():
            continue
        t0 = time.time()
        nfiles = 0
        nbytes = 0
        nstudies = 0
        nseries = 0
        for study in os.scandir(d):
            if not study.is_dir():
                continue
            nstudies += 1
            for series in os.scandir(study.path):
                if not series.is_dir():
                    continue
                nseries += 1
                for f in os.scandir(series.path):
                    if f.is_file():
                        nfiles += 1
                        st = f.stat()
                        nbytes += st.st_size
                        ext = os.path.splitext(f.name)[1].lower() or "<none>"
                        ext_count[ext] += 1
                        ext_bytes[ext] += st.st_size
        dt = time.time() - t0
        per_root[sub] = dict(studies=nstudies, series=nseries, files=nfiles,
                             bytes=nbytes, walk_seconds=round(dt, 1))
        print(f"  {sub:<14} studies={nstudies:<6} series={nseries:<7} "
              f"files={nfiles:<8} size={human(nbytes):>10}  ({dt:.0f}s walk)")

    print("\n-- file extensions --")
    for ext, c in ext_count.most_common():
        print(f"  {ext:<10} {c:>9} files   {human(ext_bytes[ext]):>12}")

    total = sum(e["bytes"] for e in per_root.values()) + sum(
        t.get("bytes", 0) for t in R["top_level"] if t["kind"] == "file")
    print(f"\n  TOTAL ON DISK (incl. the zip): {human(total)}")
    R["walk"] = dict(per_root=per_root,
                     extensions={k: dict(count=v, bytes=ext_bytes[k]) for k, v in ext_count.items()},
                     total_bytes_incl_zip=total)


# ----------------------------------------------------------------------------- 2.2/2.3 labels & targets
def section_labels_and_targets():
    hdr("2.2 / 2.3  GOLD LABELS AND TARGET DEFINITION  [HIGHEST PRIORITY]")

    tab = {}
    for name in ["train.csv", "test.csv", "train_series.csv", "test_series.csv", "sample_submission.csv"]:
        p = ROOT / name
        if not p.exists():
            print(f"  !! MISSING: {name}")
            tab[name] = None
            continue
        df = pd.read_csv(p)
        tab[name] = df
        print(f"\n-- {name}  shape={df.shape} --")
        print("   columns:", list(df.columns))
        print("   dtypes :", {c: str(t) for c, t in df.dtypes.items()})
        for idcol in ["StudyInstanceUID", "SeriesInstanceUID"]:
            if idcol in df.columns:
                print(f"   unique {idcol}: {df[idcol].nunique()}")

    sub = tab["sample_submission.csv"]
    train = tab["train.csv"]

    # ---- target names, straight from sample_submission (not guesses)
    targets = [c for c in sub.columns if c != "StudyInstanceUID"]
    wide = "StudyInstanceUID" in sub.columns and len(targets) == 12
    print("\n" + "-" * 78)
    print("THE 12 EXACT TARGET NAMES (verbatim from sample_submission.csv):")
    for i, t in enumerate(targets, 1):
        print(f"   {i:>2}. {t!r}")
    print(f"\n   LAYOUT: {'WIDE - one row per study, 12 label columns' if wide else 'NOT WIDE - inspect manually'}")
    print(f"   sample_submission rows = {len(sub)}  (== n studies in the public test stub)")
    print(f"   row_id style: {'none; StudyInstanceUID is the key' if wide else 'check'}")
    # NB: match on standalone laterality words only -- "Lateral Meniscus"/"Lateral OA"
    # are compartment names, not a laterality (left/right) column.
    lat = [c for c in sub.columns
           if c.lower() in ("laterality", "side", "knee", "left_right", "lr", "imagelaterality")]
    print(f"   laterality column present: {bool(lat)} -> "
          f"{'NO -> one knee per study, implied; L/R must come from DICOM' if not lat else lat}")
    print("-" * 78)

    R["targets"] = targets
    R["submission"] = dict(layout="wide" if wide else "unknown", n_target_cols=len(targets),
                           rows=len(sub), columns=list(sub.columns),
                           laterality_column=lat or None,
                           dtype_of_predictions=str(sub[targets[0]].dtype),
                           placeholder_value=float(sub[targets[0]].iloc[0]))

    # ---- the gold label count
    LAB = [c for c in train.columns if c in targets]
    nn = train[LAB].notna().sum(axis=1)
    n_any = int((nn > 0).sum())
    n_all = int((nn == len(LAB)).sum())
    n_none = int((nn == 0).sum())
    n_studies = int(train.StudyInstanceUID.nunique())

    print("\n" + "#" * 78)
    print(f"#  GOLD LABELS: {n_all} / {n_studies} studies fully labeled "
          f"({100 * n_all / n_studies:.2f}%)")
    print(f"#  studies with >=1 non-null label : {n_any}")
    print(f"#  studies with ZERO labels        : {n_none}   (report-only)")
    print(f"#  partially labeled studies       : {n_any - n_all}")
    print("#" * 78)

    print("\n-- per-target gold detail (the 58) --")
    print(f"  {'target':<18} {'n':>4} {'pos':>4} {'neg':>4} {'pos_rate':>9}  {'AUC SE @n_pos':>13}")
    per_t = {}
    for c in LAB:
        s = train[c].dropna()
        pos = int((s == 1).sum())
        neg = int((s == 0).sum())
        rate = pos / len(s) if len(s) else float("nan")
        # How far can this target's AUC wander on 58 studies from luck alone? This is the
        # Hanley-McNeil standard error, assuming a true AUC of a = 0.8; that assumption only
        # sets the scale, nothing is fitted here. It is governed by the SMALLER class -- MCL
        # with 9 positives comes out at 0.093, against 0.057 for Effusion with 35. Read the
        # column as "per-target differences smaller than this are noise". It is the reason
        # gold is used to RANK models and never to separate two that are close.
        se = 0.0
        if pos > 0 and neg > 0:
            a = 0.8
            q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
            se = float(np.sqrt((a * (1 - a) + (pos - 1) * (q1 - a * a) + (neg - 1) * (q2 - a * a)) / (pos * neg)))
        per_t[c] = dict(n=len(s), pos=pos, neg=neg, pos_rate=round(rate, 4),
                        approx_auc_se=round(se, 4),
                        unique_values=sorted(pd.unique(s).tolist()))
        print(f"  {c:<18} {len(s):>4} {pos:>4} {neg:>4} {rate:>9.3f}  {se:>13.3f}")

    R["gold"] = dict(n_studies_total=n_studies, n_fully_labeled=n_all,
                     n_any_label=n_any, n_unlabeled=n_none,
                     fraction_labeled=round(n_all / n_studies, 5),
                     label_columns=LAB, per_target=per_t,
                     gold_study_uids=train.loc[nn == len(LAB), "StudyInstanceUID"].tolist())

    # ---- is the gold subset a random sample, or enriched?
    mean_rate = float(np.mean([v["pos_rate"] for v in per_t.values()]))
    print(f"\n  mean positive rate across targets on the gold subset: {mean_rate:.3f}")
    print("  -> Positive rates 0.16-0.60 are FAR above plausible clinical base rates for")
    print("     things like Fracture/Contusion. The 58 are an ENRICHED/stratified sample,")
    print("     not a random draw. Consequences:")
    print("       * AUC is prevalence-invariant, so gold AUC stays an unbiased ranking")
    print("         estimate -- and enrichment REDUCES its variance. Good for validation.")
    print("       * Any probability CALIBRATION fit on these 58 inherits the enriched")
    print("         prior and will be badly miscalibrated on the test set. Since the metric")
    print("         is AUC, do not bother calibrating; calibrate only if you need it for")
    print("         loss weighting, and correct the prior shift explicitly if you do.")
    R["gold"]["mean_pos_rate"] = round(mean_rate, 4)
    R["gold"]["enriched_not_random"] = True

    # ---- look for labels hiding somewhere non-obvious
    print("\n-- searching for labels in non-obvious places --")
    flags = [c for c in train.columns
             if c.lower() in ("fold", "split", "is_labeled", "has_gold", "labeled", "subset", "site", "source")]
    print(f"   fold/split/is_labeled-style flag column in train.csv: {flags or 'NONE'}")
    extra = sorted(set(p.name for p in ROOT.rglob("*")
                       if p.is_file() and p.suffix.lower() in (".csv", ".json", ".parquet", ".txt", ".xlsx")
                       and "audit" not in p.name.lower() and "train_series" not in str(p) and "test_series" not in str(p)))
    print(f"   tabular/text files anywhere under root: {extra}")
    sidecars = [p for p in (ROOT / "train_series").rglob("*")
                if p.is_file() and p.suffix.lower() not in (".dcm", "")][:10] if (ROOT / "train_series").exists() else []
    print(f"   non-DICOM sidecars inside train_series: {[p.name for p in sidecars] or 'NONE'}")
    missing = [c for c in sub.columns if c not in train.columns and c != "StudyInstanceUID"]
    print(f"   sample_submission columns absent from train.csv: {missing or 'NONE (schema matches)'}")
    R["label_search"] = dict(flag_columns=flags, tabular_files=extra,
                             sidecars=[p.name for p in sidecars], schema_gap=missing)

    return tab


# ----------------------------------------------------------------------------- 2.4 reports
def section_reports(tab):
    hdr("2.4  REPORTS")

    train, test = tab["train.csv"], tab["test.csv"]
    rep = train["Report"]

    print(f"  reports live in: train.csv column 'Report' (not per-study text files)")
    print(f"  studies WITH a report   : {int(rep.notna().sum())} / {len(train)}")
    print(f"  studies WITHOUT a report: {int(rep.isna().sum())}")
    print(f"  empty/whitespace reports: {int((rep.fillna('').str.strip() == '').sum())}")

    # ---- THE SECOND MOST IMPORTANT QUESTION IN THE AUDIT
    test_has_report = "Report" in test.columns
    sub_has_report = "Report" in tab["sample_submission.csv"].columns
    test_series_has = "Report" in tab["test_series.csv"].columns
    test_sidecars = []
    td = ROOT / "test_series"
    if td.exists():
        test_sidecars = [p.name for p in td.rglob("*") if p.is_file() and p.suffix.lower() not in (".dcm", "")]

    print("\n" + "#" * 78)
    print("#  DO TEST STUDIES HAVE REPORTS?")
    print(f"#    'Report' column in test.csv          : {test_has_report}")
    print(f"#    'Report' column in test_series.csv   : {test_series_has}")
    print(f"#    'Report' column in sample_submission : {sub_has_report}")
    print(f"#    non-DICOM sidecars under test_series : {test_sidecars or 'NONE'}")
    verdict = any([test_has_report, test_series_has, sub_has_report]) or bool(test_sidecars)
    print(f"#  ==> ANSWER: {'YES - architecture INVERTS (see brief 4)' if verdict else 'NO. Test is IMAGES ONLY.'}")
    if not verdict:
        print("#      Reports are a TRAINING-TIME asset only. The deployed model is")
        print("#      vision-only; text is used to manufacture targets, then discarded.")
    print("#" * 78)
    R["test_reports"] = dict(present=verdict, test_csv_columns=list(test.columns),
                             test_series_columns=list(tab["test_series.csv"].columns),
                             sidecars=test_sidecars)

    # ---- length distribution
    print("\n-- report character length --")
    lens = rep.fillna("").str.len().tolist()
    R["report_chars"] = describe(lens, "characters")
    words = rep.fillna("").str.split().str.len().tolist()
    R["report_words"] = describe(words, "whitespace tokens")

    # ---- language
    print("\n-- language detection --")
    langs = {}
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        t0 = time.time()
        counts = collections.Counter()
        failures = 0
        per_study = []
        for txt in rep.fillna(""):
            t = txt.strip()
            if len(t) < 10:
                counts["<too-short>"] += 1
                per_study.append(None)
                continue
            try:
                lg = detect(t)
                counts[lg] += 1
                per_study.append(lg)
            except Exception:
                failures += 1
                counts["<undetected>"] += 1
                per_study.append(None)
        print(f"   detected in {time.time() - t0:.0f}s; {failures} failures")
        print(f"   {'lang':<14} {'n':>6} {'share':>8}")
        for lg, c in counts.most_common():
            print(f"   {lg:<14} {c:>6} {100 * c / len(rep):>7.2f}%")
        real = [k for k in counts if not k.startswith("<")]
        print(f"\n   DISTINCT LANGUAGES DETECTED: {len(real)}  -> {sorted(real)}")
        print(f"   RSNA claims 12 languages. Detected {len(real)}. "
              f"{'CONSISTENT' if len(real) >= 10 else 'CHECK - langdetect confuses close pairs (es/pt, nb/da, hr/sl)'}")
        langs = dict(counts=dict(counts), n_distinct=len(real), languages=sorted(real))
        R["_lang_per_study"] = per_study
    except ImportError:
        print("   langdetect not installed -> pip install langdetect")
        langs = dict(error="langdetect missing")
    R["languages"] = langs

    # ---- sample reports, one per language, so the labeler design is grounded in reality
    # From here on, verbatim patient text goes to stdout AND into audit_results.json. Both
    # destinations are gitignored -- see the warning at the top of this file.
    print("\n-- sample report per detected language (first 320 chars) --")
    seen = set()
    samples = {}
    per_study = R.get("_lang_per_study") or []
    for i, lg in enumerate(per_study):
        if lg and lg not in seen:
            seen.add(lg)
            txt = str(rep.iloc[i])
            samples[lg] = txt[:800]
            print(f"\n   [{lg}] study {train.StudyInstanceUID.iloc[i][-12:]}")
            print("   " + txt[:320].replace("\n", "\n   "))
    R["report_samples"] = samples
    R.pop("_lang_per_study", None)

    # ---- do the gold 58 have reports, and in which languages?
    gold_uids = set(R["gold"]["gold_study_uids"])
    gmask = train.StudyInstanceUID.isin(gold_uids)
    print(f"\n-- the {int(gmask.sum())} gold studies --")
    print(f"   all have reports: {bool(train.loc[gmask, 'Report'].notna().all())}")
    if "languages" in R and "error" not in langs:
        gl = collections.Counter()
        for i in np.where(gmask.values)[0]:
            t = str(rep.iloc[i]).strip()
            if len(t) >= 10:
                try:
                    from langdetect import detect
                    gl[detect(t)] += 1
                except Exception:
                    gl["<undetected>"] += 1
        print(f"   gold report languages: {dict(gl)}")
        print("   -> If the gold 58 are concentrated in one language, your calibration and")
        print("      your validation both inherit that language's reporting conventions.")
        R["gold"]["report_languages"] = dict(gl)


# ----------------------------------------------------------------------------- 2.5 / 2.6 DICOM
DICOM_TAGS = [
    "Manufacturer", "ManufacturerModelName", "MagneticFieldStrength", "InstitutionName",
    "StationName", "SoftwareVersions", "SeriesDescription", "ProtocolName", "BodyPartExamined",
    "ImageOrientationPatient", "PixelSpacing", "SliceThickness", "Rows", "Columns",
    "ScanningSequence", "SequenceVariant", "ScanOptions", "MRAcquisitionType",
    "RepetitionTime", "EchoTime", "FlipAngle", "Laterality", "ImageLaterality",
    "PatientSex", "PatientAge", "PhotometricInterpretation", "BitsStored",
]


def plane_from_iop(iop):
    """Derive acquisition plane from ImageOrientationPatient via the slice normal.

    Patient axes are x = left-right, y = front-back, z = head-foot. The cross product of the
    two in-plane direction cosines points THROUGH the slice, so whichever axis it lies
    closest to names the plane: mostly-x means slices are stacked left to right (sagittal),
    mostly-y front to back (coronal), mostly-z head to foot (axial) -- which is the order of
    the list below. Taking the LARGEST component rather than demanding an exact match is
    what keeps this correct for obliqued acquisitions, which are the norm here (674 distinct
    orientations in a 200-study sample).
    """
    if iop is None or len(iop) != 6:
        return "unknown"
    a = np.array(iop[:3], dtype=float)
    b = np.array(iop[3:], dtype=float)
    n = np.cross(a, b)
    ax = int(np.argmax(np.abs(n)))
    return ["Sagittal", "Coronal", "Axial"][ax]


def section_dicom(tab, n_sample, seed=0):
    hdr(f"2.5 / 2.6  DICOM STRUCTURE  (sample of {n_sample} studies)")
    import pydicom

    d = ROOT / "train_series"
    studies = sorted(p.name for p in d.iterdir() if p.is_dir())
    rng = random.Random(seed)
    sample = rng.sample(studies, min(n_sample, len(studies)))
    print(f"  sampling {len(sample)} of {len(studies)} studies (seed={seed})")

    tags = collections.defaultdict(collections.Counter)
    series_per_study, slices_per_series, series_shapes = [], [], []
    plane_counter = collections.Counter()
    plane_vs_csv = collections.Counter()
    fingerprints = collections.Counter()
    study_fingerprint = {}
    anomalies = []
    tsuid = collections.Counter()
    n_hdr = 0
    corrupt = 0

    tser = tab["train_series.csv"]
    csv_plane = dict(zip(tser.SeriesInstanceUID, tser.Anatomical_Plane))

    t0 = time.time()
    for suid in sample:
        sdir = d / suid
        series = [p for p in sdir.iterdir() if p.is_dir()]
        series_per_study.append(len(series))
        if not series:
            anomalies.append(dict(study=suid, issue="zero series"))
            continue
        fp_bits = None
        for ser in series:
            files = [f for f in ser.iterdir() if f.is_file()]
            slices_per_series.append(len(files))
            if not files:
                anomalies.append(dict(study=suid, series=ser.name, issue="zero slices"))
                continue
            try:
                ds = pydicom.dcmread(str(files[len(files) // 2]), stop_before_pixels=True, force=True)
                n_hdr += 1
            except Exception as e:
                corrupt += 1
                anomalies.append(dict(study=suid, series=ser.name, issue=f"unreadable header: {e}"))
                continue

            for t in DICOM_TAGS:
                v = getattr(ds, t, None)
                if v is None:
                    tags[t]["<absent>"] += 1
                else:
                    if t in ("ImageOrientationPatient", "PixelSpacing"):
                        v = tuple(round(float(x), 4) for x in v)
                    tags[t][str(v)] += 1

            try:
                tsuid[str(ds.file_meta.TransferSyntaxUID)] += 1
            except Exception:
                tsuid["<no file_meta>"] += 1

            pl = plane_from_iop(getattr(ds, "ImageOrientationPatient", None))
            plane_counter[pl] += 1
            if ser.name in csv_plane:
                plane_vs_csv[f"{csv_plane[ser.name]}|{pl}"] += 1

            rows, cols = getattr(ds, "Rows", None), getattr(ds, "Columns", None)
            if rows and cols:
                series_shapes.append((int(rows), int(cols)))
                if rows < 128 or cols < 128 or rows > 1024 or cols > 1024:
                    anomalies.append(dict(study=suid, series=ser.name,
                                          issue=f"outlying dims {rows}x{cols}"))

            # SITE FINGERPRINT. Anonymisation stripped InstitutionName, but a site is still
            # identifiable through the machine it scans on, so scanner identity stands in for
            # it. This matters because a model can score well by recognising a scanner rather
            # than a knee: folds split at random inflate CV, folds grouped by site do not.
            # Only the first readable series of a study sets fp_bits, so one study -> one
            # fingerprint.
            if fp_bits is None:
                ps = getattr(ds, "PixelSpacing", None)
                fp_bits = "|".join([
                    str(getattr(ds, "InstitutionName", "") or "-"),
                    str(getattr(ds, "Manufacturer", "") or "-"),
                    str(getattr(ds, "ManufacturerModelName", "") or "-"),
                    str(getattr(ds, "MagneticFieldStrength", "") or "-"),
                    str(getattr(ds, "StationName", "") or "-"),
                    f"{float(ps[0]):.3f}" if ps else "-",
                ])
        if fp_bits:
            fingerprints[fp_bits] += 1
            study_fingerprint[suid] = fp_bits
    dt = time.time() - t0

    print(f"\n  read {n_hdr} headers in {dt:.1f}s  ({n_hdr / max(dt, 1e-9):.0f} hdr/s), {corrupt} unreadable")

    print("\n-- structure distributions --")
    R["series_per_study"] = describe(series_per_study, "series per study")
    R["slices_per_series"] = describe(slices_per_series, "slices per series")
    tot = [a * b for a, b in zip(series_per_study, [1] * len(series_per_study))]
    print(f"  series-count histogram: {dict(sorted(collections.Counter(series_per_study).items()))}")
    sl = collections.Counter(slices_per_series)
    print(f"  slice-count: {len(sl)} distinct values; most common "
          f"{sl.most_common(8)}")
    R["series_count_hist"] = dict(collections.Counter(series_per_study))

    print("\n-- image dimensions --")
    shape_c = collections.Counter(series_shapes)
    for s, c in shape_c.most_common(10):
        print(f"   {s[0]:>4} x {s[1]:<4}  {c:>6} series")
    print(f"   distinct shapes: {len(shape_c)}")
    R["shapes"] = {f"{k[0]}x{k[1]}": v for k, v in shape_c.most_common(30)}

    print("\n-- transfer syntax (decode cost) --")
    NAMES = {
        "1.2.840.10008.1.2": "Implicit VR LE (uncompressed)",
        "1.2.840.10008.1.2.1": "Explicit VR LE (uncompressed)",
        "1.2.840.10008.1.2.4.50": "JPEG Baseline (lossy)",
        "1.2.840.10008.1.2.4.51": "JPEG Extended (lossy)",
        "1.2.840.10008.1.2.4.57": "JPEG Lossless",
        "1.2.840.10008.1.2.4.70": "JPEG Lossless SV1",
        "1.2.840.10008.1.2.4.90": "JPEG 2000 Lossless",
        "1.2.840.10008.1.2.4.91": "JPEG 2000",
        "1.2.840.10008.1.2.5": "RLE Lossless",
        "1.2.840.10008.1.2.1.99": "Deflated Explicit VR LE",
    }
    compressed = 0
    for k, c in tsuid.most_common():
        nm = NAMES.get(k, "UNKNOWN")
        if "uncompressed" not in nm and k in NAMES:
            compressed += c
        print(f"   {k:<28} {c:>6}  {nm}")
    print(f"   compressed series: {compressed}/{sum(tsuid.values())} "
          f"-> {'need pylibjpeg/gdcm' if compressed else 'plain numpy decode, no extra codecs needed'}")
    R["transfer_syntax"] = dict(counts=dict(tsuid), n_compressed=compressed)

    print("\n-- header tag coverage (top values) --")
    for t in DICOM_TAGS:
        c = tags[t]
        if not c:
            continue
        absent = c.get("<absent>", 0)
        total = sum(c.values())
        top = [x for x in c.most_common(6) if x[0] != "<absent>"][:5]
        print(f"   {t:<26} absent={100 * absent / total:>5.1f}%  distinct={len(c):<5} "
              f"top={[(v[:38], n) for v, n in top]}")
    R["dicom_tags"] = {t: dict(distinct=len(c), absent_pct=round(100 * c.get('<absent>', 0) / max(sum(c.values()), 1), 2),
                               top=dict(c.most_common(15))) for t, c in tags.items()}

    print("\n-- plane derived from ImageOrientationPatient --")
    print(f"   {dict(plane_counter)}")
    print("\n-- provided Anatomical_Plane  vs  IOP-derived plane --")
    agree = sum(v for k, v in plane_vs_csv.items() if k.split("|")[0] == k.split("|")[1])
    tot_p = sum(plane_vs_csv.values())
    for k, v in plane_vs_csv.most_common():
        a, b = k.split("|")
        print(f"   csv={a:<10} iop={b:<10} {v:>6} {'' if a == b else '   <-- MISMATCH'}")
    print(f"   AGREEMENT: {agree}/{tot_p} = {100 * agree / max(tot_p, 1):.2f}%")
    R["plane_check"] = dict(iop_counts=dict(plane_counter), confusion=dict(plane_vs_csv),
                            agreement=round(agree / max(tot_p, 1), 4))

    print("\n-- 2.6  SITE / SCANNER FINGERPRINT (the CV grouping variable) --")
    print(f"   distinct fingerprints in sample: {len(fingerprints)}")
    print(f"   {'studies':>8}  fingerprint (institution|mfr|model|field|station|pixspacing)")
    for fp, c in fingerprints.most_common(25):
        print(f"   {c:>8}  {fp[:110]}")
    counts = np.array(list(fingerprints.values()), dtype=float)
    if counts.size:
        print(f"\n   study-count distribution across fingerprints: "
              f"min={counts.min():.0f} med={np.median(counts):.0f} max={counts.max():.0f}")
        print(f"   largest fingerprint holds {100 * counts.max() / counts.sum():.1f}% of sampled studies")
    inst_absent = tags["InstitutionName"].get("<absent>", 0)
    print(f"   InstitutionName absent in {100 * inst_absent / max(sum(tags['InstitutionName'].values()), 1):.1f}% "
          f"of sampled series -> {'de-identified; must use scanner fingerprint' if inst_absent else 'usable directly'}")
    R["fingerprints"] = dict(n_distinct=len(fingerprints),
                             top=dict(fingerprints.most_common(40)),
                             study_fingerprint_sample=study_fingerprint)

    print(f"\n-- anomalies ({len(anomalies)} flagged) --")
    for a in anomalies[:20]:
        print(f"   {a}")
    if len(anomalies) > 20:
        print(f"   ... and {len(anomalies) - 20} more")
    R["anomalies"] = anomalies[:200]
    R["anomaly_count"] = len(anomalies)

    return sample


# ----------------------------------------------------------------------------- 2.7 sanity
def section_sanity(sample, n_png=6):
    hdr("2.7  DECODE SANITY + TIMING")
    import pydicom
    from PIL import Image

    OUT_DIR.mkdir(exist_ok=True)
    d = ROOT / "train_series"
    saved, times, made = [], [], 0

    for suid in sample:
        if made >= n_png:
            break
        sdir = d / suid
        for ser in sorted(p for p in sdir.iterdir() if p.is_dir()):
            files = sorted(f for f in ser.iterdir() if f.is_file())
            if not files:
                continue
            mid = files[len(files) // 2]
            try:
                t0 = time.time()
                ds = pydicom.dcmread(str(mid))
                arr = ds.pixel_array.astype(np.float32)
                times.append(time.time() - t0)
            except Exception as e:
                print(f"   DECODE FAIL {mid.name}: {e}")
                continue
            if made < n_png:
                lo, hi = np.percentile(arr, [1, 99])
                img = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
                if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                    img = 1 - img
                name = (f"{made:02d}_{plane_from_iop(getattr(ds, 'ImageOrientationPatient', None))}"
                        f"_{str(getattr(ds, 'SeriesDescription', 'na'))[:24].replace('/', '-').replace(' ', '_')}.png")
                name = "".join(ch for ch in name if ch.isalnum() or ch in "._-")
                Image.fromarray((img * 255).astype(np.uint8)).save(OUT_DIR / name)
                saved.append(name)
                print(f"   saved {name}  shape={arr.shape} range=[{arr.min():.0f},{arr.max():.0f}] "
                      f"dtype={ds.pixel_array.dtype}")
                made += 1
            break

    # timing extrapolation over the whole corpus
    per = float(np.median(times)) if times else float("nan")
    total_files = R.get("walk", {}).get("per_root", {}).get("train_series", {}).get("files")
    print(f"\n   median single-slice full decode: {per * 1000:.1f} ms")
    if total_files:
        secs = per * total_files
        print(f"   total files: {total_files}")
        print(f"   ESTIMATED FULL-CORPUS DECODE (1 core): {secs / 3600:.2f} h")
        print(f"   at 8 workers: {secs / 3600 / 8:.2f} h    at 16 workers: {secs / 3600 / 16:.2f} h")
        R["decode"] = dict(median_seconds_per_slice=per, total_files=total_files,
                           est_single_core_hours=round(secs / 3600, 2),
                           est_8_worker_hours=round(secs / 3600 / 8, 2))
    print(f"   PNGs written to {OUT_DIR}")
    R["png_samples"] = saved


# ----------------------------------------------------------------------------- series metadata
def section_series_metadata(tab):
    hdr("BONUS  PROVIDED SERIES METADATA  (train_series.csv / test_series.csv)")
    tr, te = tab["train_series.csv"], tab["test_series.csv"]
    print("  RSNA ships Fluid_Sensitive / Fat_Suppression / Anatomical_Plane per series.")
    print("  This is a labeled series classifier, free. Brief 6 assumed we'd have to build one.\n")
    for name, df in [("train_series.csv", tr), ("test_series.csv", te)]:
        print(f"-- {name}: {len(df)} series over {df.StudyInstanceUID.nunique()} studies --")
        for c in ["Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"]:
            print(f"   {c:<18} {dict(df[c].value_counts(dropna=False))}")
        combo = df.groupby(["Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"]).size()
        print(f"   plane x fluid x fatsat combos: {len(combo)}")
        print(combo.to_string().replace("\n", "\n     "))
        print()

    n_ser = tr.groupby("StudyInstanceUID").size()
    print("-- series per study (from CSV, all 4407 studies) --")
    describe(n_ser.tolist(), "series per study")
    print(f"   histogram: {dict(sorted(n_ser.value_counts().items()))}")

    print("\n-- plane coverage per study --")
    piv = tr.pivot_table(index="StudyInstanceUID", columns="Anatomical_Plane",
                         values="SeriesInstanceUID", aggfunc="count").fillna(0)
    for p in piv.columns:
        print(f"   studies with >=1 {p:<10}: {int((piv[p] > 0).sum())} / {len(piv)} "
              f"({100 * (piv[p] > 0).mean():.1f}%)")
    missing_any = int((piv == 0).any(axis=1).sum())
    print(f"   studies MISSING at least one plane: {missing_any} ({100 * missing_any / len(piv):.1f}%)")
    print(f"   studies with all three planes     : {len(piv) - missing_any}")

    print("\n-- fluid-sensitive coverage (Effusion/Synovitis/Contusion depend on it) --")
    fs = tr.groupby("StudyInstanceUID").Fluid_Sensitive.max()
    print(f"   studies with >=1 fluid-sensitive series: {int((fs > 0).sum())} / {len(fs)}")
    print(f"   studies with NONE                      : {int((fs == 0).sum())}")

    # cross-check studies in csv vs on disk
    disk = set(p.name for p in (ROOT / "train_series").iterdir() if p.is_dir())
    csv_studies = set(tr.StudyInstanceUID)
    lab_studies = set(tab["train.csv"].StudyInstanceUID)
    print("\n-- consistency: train.csv vs train_series.csv vs disk --")
    print(f"   train.csv studies        : {len(lab_studies)}")
    print(f"   train_series.csv studies : {len(csv_studies)}")
    print(f"   on-disk study dirs       : {len(disk)}")
    print(f"   in csv but NOT on disk   : {len(csv_studies - disk)}")
    print(f"   on disk but NOT in csv   : {len(disk - csv_studies)}")
    print(f"   labeled but NOT on disk  : {len(lab_studies - disk)}")

    R["series_metadata"] = dict(
        train_series_rows=len(tr), test_series_rows=len(te),
        plane_counts=jsonable(dict(tr.Anatomical_Plane.value_counts())),
        fluid_counts=jsonable(dict(tr.Fluid_Sensitive.value_counts())),
        fatsat_counts=jsonable(dict(tr.Fat_Suppression.value_counts())),
        series_per_study=describe(n_ser.tolist(), "  (recomputed)"),
        studies_missing_a_plane=missing_any,
        studies_no_fluid_sensitive=int((fs == 0).sum()),
        consistency=dict(train_csv=len(lab_studies), series_csv=len(csv_studies),
                         on_disk=len(disk), csv_not_on_disk=len(csv_studies - disk),
                         disk_not_in_csv=len(disk - csv_studies)),
    )


# ----------------------------------------------------------------------------- full header pass
def section_full_headers(tab):
    """Scan one header per series across ALL studies.

    The 200-study sample produced 120 distinct scanner fingerprints (median 1 study
    each) -- far too granular to group CV folds with. This pass settles the real
    grouping variable and writes study_metadata.csv for downstream use.
    """
    hdr("2.6b  FULL HEADER PASS -- DEFINITIVE CV GROUPING VARIABLE")
    import pydicom

    d = ROOT / "train_series"
    tser = tab["train_series.csv"]
    by_study = tser.groupby("StudyInstanceUID").SeriesInstanceUID.apply(list).to_dict()

    rows = []
    t0 = time.time()
    for i, (suid, sers) in enumerate(by_study.items()):
        if i % 500 == 0:
            print(f"   ... {i}/{len(by_study)} studies  ({time.time() - t0:.0f}s)", flush=True)
        rec = dict(StudyInstanceUID=suid, n_series=len(sers))
        lats, bodyparts, mfrs, models, fields, bits, npx = [], [], [], [], [], [], 0
        for ser in sers:
            sdir = d / suid / ser
            if not sdir.is_dir():
                continue
            files = [f for f in os.scandir(sdir.path if hasattr(sdir, "path") else str(sdir)) if f.is_file()]
            if not files:
                continue
            try:
                ds = pydicom.dcmread(files[len(files) // 2].path, stop_before_pixels=True, force=True)
            except Exception:
                continue
            mfrs.append(str(getattr(ds, "Manufacturer", "") or ""))
            models.append(str(getattr(ds, "ManufacturerModelName", "") or ""))
            f_ = getattr(ds, "MagneticFieldStrength", None)
            fields.append(round(float(f_), 1) if f_ else None)
            lv = str(getattr(ds, "Laterality", "") or "").strip().upper()
            if lv:
                lats.append(lv[0])
            bodyparts.append(str(getattr(ds, "BodyPartExamined", "") or ""))
            bits.append(getattr(ds, "BitsStored", None))
            r, c = getattr(ds, "Rows", 0), getattr(ds, "Columns", 0)
            npx += int(r) * int(c) * len(files)

        def mode(xs):
            xs = [x for x in xs if x not in (None, "")]
            return collections.Counter(xs).most_common(1)[0][0] if xs else None

        # normalise vendor spelling: Siemens ships as SIEMENS / Siemens / Siemens Healthineers
        mfr_raw = mode(mfrs) or "UNKNOWN"
        m = mfr_raw.upper()
        mfr = ("SIEMENS" if "SIEMENS" in m else "PHILIPS" if "PHILIPS" in m
               else "GE" if m.startswith("GE") else "CANON" if ("CANON" in m or "TOSHIBA" in m)
               else "HITACHI" if "HITACHI" in m else "UIH" if ("UIH" in m or "UNITED IMAGING" in m)
               else m.split()[0] if m else "UNKNOWN")
        rec.update(
            manufacturer_raw=mfr_raw, manufacturer=mfr,
            model=mode(models) or "UNKNOWN", field=mode(fields),
            laterality=(collections.Counter(lats).most_common(1)[0][0] if lats else None),
            laterality_mixed=len(set(lats)) > 1,
            bodypart=mode(bodyparts), bits_stored=mode(bits), n_pixels=npx,
        )
        rows.append(rec)
    meta = pd.DataFrame(rows)
    print(f"   scanned {len(meta)} studies in {time.time() - t0:.0f}s")

    # ---- candidate grouping variables, coarse to fine
    print("\n-- candidate CV grouping variables --")
    meta["g_mfr"] = meta.manufacturer
    meta["g_mfr_model"] = meta.manufacturer + "|" + meta.model.astype(str)
    meta["g_mfr_model_field"] = meta.g_mfr_model + "|" + meta.field.astype(str)
    print(f"  {'variable':<22} {'groups':>7} {'median':>7} {'max':>6} {'largest%':>9} {'singletons':>11}  usable for 5-fold?")
    grp_stats = {}
    for g in ["g_mfr", "g_mfr_model", "g_mfr_model_field"]:
        vc = meta[g].value_counts()
        largest = 100 * vc.max() / len(meta)
        singl = int((vc == 1).sum())
        ok = "YES" if len(vc) >= 10 and largest < 40 else ("NO - too few groups" if len(vc) < 10 else "NO - one group dominates")
        print(f"  {g:<22} {len(vc):>7} {vc.median():>7.0f} {vc.max():>6} {largest:>8.1f}% {singl:>11}  {ok}")
        grp_stats[g] = dict(n_groups=int(len(vc)), median=float(vc.median()), max=int(vc.max()),
                            largest_pct=round(largest, 2), singletons=singl,
                            counts=jsonable(dict(vc.head(40))))
    R["grouping"] = grp_stats

    print("\n-- manufacturer breakdown (all 4407 studies) --")
    for k, v in meta.manufacturer.value_counts().items():
        print(f"   {k:<12} {v:>6}  ({100 * v / len(meta):.1f}%)")
    print("\n-- top 25 manufacturer|model --")
    for k, v in meta.g_mfr_model.value_counts().head(25).items():
        print(f"   {v:>6}  {k}")
    print("\n-- field strength --")
    print(f"   {jsonable(dict(meta.field.value_counts(dropna=False)))}")

    # ---- how do the 58 gold studies scatter across groups?
    gold = set(R["gold"]["gold_study_uids"])
    gm = meta[meta.StudyInstanceUID.isin(gold)]
    print(f"\n-- the {len(gm)} gold studies across scanner groups --")
    print(f"   distinct manufacturers among gold : {gm.manufacturer.nunique()}  {jsonable(dict(gm.manufacturer.value_counts()))}")
    print(f"   distinct mfr|model among gold     : {gm.g_mfr_model.nunique()}")
    print("   -> With 58 gold studies over "
          f"{gm.g_mfr_model.nunique()} scanner groups, leave-one-site-out on GOLD is not")
    print("      viable (a few studies per held-out group). Group the UNLABELED set by")
    print("      scanner for distillation folds; evaluate gold with bootstrap CIs instead.")
    R["gold"]["manufacturers"] = jsonable(dict(gm.manufacturer.value_counts()))
    R["gold"]["n_scanner_groups"] = int(gm.g_mfr_model.nunique())

    # ---- laterality
    print("\n-- laterality (needed to mirror left knees to a canonical side) --")
    vc = meta.laterality.value_counts(dropna=False)
    print(f"   {jsonable(dict(vc))}")
    n_missing = int(meta.laterality.isna().sum())
    print(f"   studies with NO laterality tag anywhere: {n_missing} ({100 * n_missing / len(meta):.1f}%)")
    print(f"   studies with CONFLICTING laterality across series: {int(meta.laterality_mixed.sum())}")
    print("   -> Missing laterality must be inferred from pixels (or skip mirroring).")
    R["laterality"] = dict(counts=jsonable(dict(vc)), missing=n_missing,
                           mixed=int(meta.laterality_mixed.sum()))

    # ---- body part anomalies
    print("\n-- BodyPartExamined anomalies --")
    bp = meta.bodypart.value_counts(dropna=False)
    print(f"   {jsonable(dict(bp))}")
    odd = meta[~meta.bodypart.isin(["KNEE", "EXTREMITY", "LEG", None]) & meta.bodypart.notna()]
    print(f"   studies with a non-knee BodyPartExamined: {len(odd)} "
          f"-> {sorted(set(odd.bodypart))[:10] if len(odd) else ''}")
    print("   (tag junk from anonymisation, almost certainly; images are still knees.)")
    R["bodypart"] = jsonable(dict(bp))

    # ---- cache size estimate
    total_px = int(meta.n_pixels.sum())
    print("\n-- preprocessed cache size estimate --")
    print(f"   total pixels across all train slices: {total_px / 1e9:.1f} G")
    print(f"   uint8 at native resolution          : {human(total_px)}")
    for side in (224, 256, 320, 384):
        nsl = R.get("walk", {}).get("per_root", {}).get("train_series", {}).get("files", 819078)
        b = nsl * side * side
        print(f"   uint8 resampled to {side}x{side:<5}        : {human(b)}")
    R["cache_estimate"] = dict(total_pixels=total_px, uint8_native_bytes=total_px,
                               uint8_224=819078 * 224 * 224, uint8_256=819078 * 256 * 256)

    out = ROOT / "study_metadata.csv"
    meta.to_csv(out, index=False)
    print(f"\n   per-study metadata written -> {out}")
    print("   (this file IS the CV grouping table; join on StudyInstanceUID)")
    R["study_metadata_csv"] = str(out)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=200, help="studies to sample for DICOM header scan")
    ap.add_argument("--no-walk", action="store_true", help="skip the full recursive file walk")
    ap.add_argument("--full-headers", action="store_true",
                    help="scan one header per series across ALL 4407 studies (~3 min); "
                         "settles the CV grouping variable and writes study_metadata.csv")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    # reports are multilingual (Cyrillic, CJK, accented Latin); Windows stdout
    # defaults to cp1252 and would raise UnicodeEncodeError on the first sample.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("RSNA 2026 Knee Abnormality Detection -- DATA AUDIT")
    print(f"root: {ROOT}")
    print(f"python: {sys.version.split()[0]}   pandas: {pd.__version__}   numpy: {np.__version__}")

    section_inventory(not a.no_walk)
    tab = section_labels_and_targets()
    section_reports(tab)
    section_series_metadata(tab)
    sample = section_dicom(tab, a.sample, a.seed)
    section_sanity(sample)
    if a.full_headers:
        section_full_headers(tab)

    RESULTS_PATH.write_text(json.dumps(jsonable(R), indent=2, ensure_ascii=False), encoding="utf-8")
    hdr("DONE")
    print(f"  machine-readable results -> {RESULTS_PATH}")
    print(f"  sample PNGs              -> {OUT_DIR}")
    print(f"\n  HEADLINE: GOLD LABELS: {R['gold']['n_fully_labeled']} / "
          f"{R['gold']['n_studies_total']} studies fully labeled "
          f"({100 * R['gold']['fraction_labeled']:.2f}%)")
    print(f"  TARGETS : {R['targets']}")
    print(f"  TEST REPORTS: {'PRESENT' if R['test_reports']['present'] else 'ABSENT'}")


if __name__ == "__main__":
    main()
