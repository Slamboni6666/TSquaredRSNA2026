"""
build_reasonknee_labels.py -- the FULL 3DReasonKnee annotation set onto our soft-label ladder.

WHAT THIS SCALES UP
The 246-knee pilot (build_oai_labels.py + oai_ladder.py, consumed by D42) joins the 507
OAI-ZIB volumes on disk to MOAKS readings. The full 3DReasonKnee release
(data/3dreasonknee, CC BY 4.0) carries 11,501 reading rows over 3,156 knees /
2,669 participants, and a volume manifest (split_files/all_studies.csv) of 8,020 sagittal
3D DESS volumes across visits 00M/12M/24M/48M(/6x18M). This script produces the full-scale
label + fold tables in the EXACT schema the pilot's oai_cache_labels_ladder.csv uses, so
the D42 consumption machinery (oai_aux.attach_oai) can read either file unchanged.

IT DOES NOT TOUCH THE PILOT. Outputs are new files (oai_full_labels_ladder.csv,
folds_oai_full.csv, oai_full_manifest.csv) on D:, never in the public repo (rows carry OAI
participant ids and the fold table carries competition UIDs). oai_ladder.py,
build_oai_labels.py and everything D42 trains on are imported, not edited.

THE ONE STRUCTURAL DIFFERENCE FROM THE PILOT: the join is PER VISIT.
The pilot pooled every reading of a knee (max grade across projects AND visits) because
OAI-ZIB holds one volume per knee. At full scale the readings are per-visit -- the
taxonomymatch files are MOAKS timepoint releases (BICL00=00M, BICL01=12M, BICL03=24M,
BICL06=48M; every grading column is taxonomy-renamed to a V00 prefix, so the FILE, not the
column name, carries the visit) -- and the volumes are per-visit too. Attaching a 48M
finding to a 00M image would invent supervision, so each volume gets only its own visit's
reading. Verified on load: 8,014 of 8,020 volumes have a same-visit reading; the 6 without
are the 18M studies covered only by manual_annotated_kMRI_18M.csv, which is EXCLUDED --
those six annotations are the 3DReasonKnee authors' own, not OAI central readings, and six
knees buy nothing. WITHIN a knee-visit, max across reading projects is inherited from the
pilot: same knee, same date, worst finding any reader saw stands.

EVERYTHING ELSE IS INHERITED, not re-derived -- these encode a week of verified bug-fixes:
  * SCALES / LADDER / TARGET_SCALE / UNCOVERED / NEGATIVE_RUNGS import from oai_ladder.py.
    Grade 1 -> uncertain 0.35 (never a confident negative); meniscus 2-5 are tear-TYPE
    codes collapsing to present_ungraded 0.62; grades above a scale's top (cartilage 5
    exists in this corpus) -> present_ungraded; hard 0.0/1.0 never appear on a trained cell.
  * A cell MOAKS did not read is state="silent", value 0.0, masked downstream. Silent ==
    unread, cell for cell, enforced against a rebuild from source.
  * Contusion is FORCED SILENT on every row (UNCOVERED): MOAKS V00MBM* is the bone-marrow-
    lesion family, a feature OF osteoarthritis; a report "contusion" is acute trauma, and
    the conditional relationship reverses sign. MCL / Fracture: not scored by MOAKS.
  * moaks_columns() from build_oai_labels.py groups the V00 grading columns per target.

KEYS AND FOLDS
  StudyInstanceUID = oaifull_{OAI id}_{visit}_{R|L}   (SIDE 1=right, 2=left -- OAI coding;
  the pilot's all-KneeSide-1 bridge and OAI-ZIB's right-knees-only documentation agree).
  The prefix is deliberately NOT "oaizib": the D42 tripwires key on that prefix and the two
  caches must never collide.

  folds_oai_full.csv = folds.csv + one row per full-scale knee at fold = -1 (train-only in
  every fold-k split, structurally incapable of reaching an OOF), with group = "OAI|{id}"
  -- GROUPED BY PARTICIPANT, because full OAI has both knees and up to 5 visits per person
  (owner-recorded constraint; the pilot had one knee per participant and did not need it).
  folds.csv itself stays OAI-free: train.py hardcodes it and --random-folds permutes it.

USAGE (CPU only, never the GPU)
  python build_reasonknee_labels.py              # writes the three tables + full report
  python build_reasonknee_labels.py --self-test  # validation + 3 corruption demos that FAIL
"""

from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from build_oai_labels import moaks_columns
from oai_ladder import (LADDER, NEGATIVE_RUNGS, SCALES, SILENT, TARGETS, TARGET_SCALE,
                        UNCOVERED)

ROOT = Path(__file__).resolve().parent
DATA = Path(r"data/3dreasonknee/data")
OAI_CACHE = Path(r"data/oai_cache")

# taxonomymatch file -> the visit its readings describe. BICL is the MOAKS reading-release
# code; 3DReasonKnee renamed every grading column to a V00 prefix, so the visit lives ONLY
# in the filename. manual_annotated_kMRI_18M.csv is excluded on purpose (author-annotated,
# not an OAI central reading; 6 rows).
VISIT = {"BICL00": "00M", "BICL01": "12M", "BICL03": "24M", "BICL06": "48M"}
SIDE_LETTER = {1: "R", 2: "L"}
UID_RE = re.compile(r"^oaifull_(\d+)_(00M|12M|24M|48M)_([RL])$")

# The semantic-mismatch gate. Any target whose full-scale positive rate exceeds the
# teacher's by more than this factor is REFUSED unless a decision is documented below.
# This is the scan that caught Contusion (82.9% BML vs 16.5% report-contusion, conditional
# sign reversed); at full scale nothing may cross it silently.
MISMATCH_RATIO = 2.5

# Documented decisions for targets the >2.5x scan flags. build() RAISES if a flagged
# target has no entry here, and if an entry goes stale (target no longer flagged) so the
# documentation cannot drift from the data. Contusion never reaches this gate: it is
# forced silent upstream (UNCOVERED), which is the stronger exclusion.
#
# The shared rationale for the OA-family entries: OAI is an osteoarthritis cohort, so the
# SAME finding is simply more common -- a prevalence shift, not a changed definition (the
# Contusion failure mode). The evidence distinguishing the two is the conditional
# structure: for each included target the OA-conditional lift has the SAME sign in our
# teacher labels and in MOAKS (printed by the report as `lift`); Contusion's reversed.
# These labels are auxiliary supervision at fold -1, never validation and never
# calibration, so an enriched prevalence costs nothing (TRACKER 2026-08-26 caveat).
DECISIONS = {
    "Synovitis": "INCLUDE -- flagged at 3.6x (59.3% vs the teacher's 16.6%), but the "
                 "evidence separates this from Contusion: (a) the OA-conditional lift has "
                 "the SAME sign in both corpora (+0.20 OAI, +0.36 teacher; Contusion's "
                 "reversed), so MOAKS Hoffa-synovitis and a report 'synovitis' move with "
                 "disease the same way; (b) 80% of OAI positives are grade-1 'mild', which "
                 "lands on the 0.45 rung, not a confident positive -- MOAKS scores Hoffa "
                 "signal abnormal at ANY grade while reports mention only florid synovitis, "
                 "so the excess is reading sensitivity plus OA-cohort enrichment, not a "
                 "changed definition; (c) Synovitis is our worst target (0.757) and the one "
                 "the OAI lane exists to fix. Auxiliary supervision at fold -1 only; these "
                 "prevalences calibrate nothing.",
}

# ----------------------------------------------------------------------------- sources --

def load_readings(labels_dir=DATA / "labels"):
    """All taxonomymatch MOAKS files, stacked, each row tagged with its visit."""
    frames = []
    for f in sorted(Path(labels_dir).glob("taxonomymatch_*MOAKS*.csv")):
        d = pd.read_csv(f, low_memory=False)
        if not {"ID", "SIDE"} <= set(d.columns):
            raise ValueError(f"{f.name}: no ID/SIDE columns")
        code = re.search(r"BICL\d\d", f.name)
        if code is None or code.group(0) not in VISIT:
            raise ValueError(f"{f.name}: cannot read a visit code from the filename")
        d["_time"] = VISIT[code.group(0)]
        frames.append(d)
    if not frames:
        raise FileNotFoundError(f"no taxonomymatch MOAKS csv under {labels_dir}")
    return pd.concat(frames, ignore_index=True)


def max_grades(moaks):
    """-> one row per (ID, SIDE, _time), the max grade per target across subregions and
    reading projects WITHIN that visit. NaN iff nothing was read. The pilot's max rule,
    scoped to the visit the volume was acquired at."""
    fam = {t: cols for t, cols in moaks_columns(moaks.columns).items() if t in TARGET_SCALE}
    g = moaks.groupby(["ID", "SIDE", "_time"])
    out = {}
    for t, cols in fam.items():
        v = moaks[cols].apply(pd.to_numeric, errors="coerce")
        out[t] = v.groupby([moaks.ID, moaks.SIDE, moaks._time]).max().max(axis=1)
    out = pd.DataFrame(out)
    out.index.names = ["ID", "SIDE", "_time"]
    return out


def load_manifest(studies_csv=DATA / "split_files" / "all_studies.csv"):
    """all_studies.csv -> one row per (patient, side, visit) with a deterministic uid and
    the image paths the acquire/cache lane needs. 22 knee-visits list two series; the
    lowest series_folder wins, deterministically, and n_volumes records the fact."""
    s = pd.read_csv(studies_csv)
    need = {"patient_id", "time", "series_folder", "knee_side", "path", "image_filepath"}
    if not need <= set(s.columns):
        raise ValueError(f"{studies_csv}: missing columns {sorted(need - set(s.columns))}")
    bad = set(s.knee_side.unique()) - set(SIDE_LETTER)
    if bad:
        raise ValueError(f"knee_side values {bad} are neither 1 (right) nor 2 (left)")
    s = s.sort_values(["patient_id", "knee_side", "time", "series_folder"])
    g = s.groupby(["patient_id", "knee_side", "time"], as_index=False)
    man = g.first()
    man["n_volumes"] = g.size()["size"].values
    man["uid"] = [f"oaifull_{p}_{t}_{SIDE_LETTER[k]}"
                  for p, k, t in zip(man.patient_id, man.knee_side, man.time)]
    return man


# ------------------------------------------------------------------------------ ladder --

def ladder_from_grades(uids, grades):
    """(uid vector, per-target max-grade frame) -> the exact frame KneeStudies indexes:
    StudyInstanceUID then target/target__state interleaved, model.py order. The mapping
    loop mirrors oai_ladder.ladder_table cell for cell; the rung numbers, scales and the
    UNCOVERED exclusion import from oai_ladder so they cannot drift from the pilot."""
    out = pd.DataFrame({"StudyInstanceUID": np.asarray(uids, dtype=object)})
    for t in TARGETS:
        if t in UNCOVERED:
            out[t] = np.zeros(len(out), np.float32)
            out[t + "__state"] = SILENT
            continue
        g = pd.to_numeric(grades[t], errors="coerce").values
        state = np.full(len(out), SILENT, dtype=object)
        val = np.zeros(len(out), np.float32)
        tab = SCALES[TARGET_SCALE[t]]
        ok = ~np.isnan(g)
        if (g[ok] != np.floor(g[ok])).any() or (g[ok] < 0).any():
            raise ValueError(f"{t}: fractional or negative MOAKS grade in a covered family")
        for k in np.unique(g[ok]).astype(int):
            if k in tab:
                rung = tab[k]
            elif k > max(tab):
                rung = "present_ungraded"    # real finding, unreadable severity
            else:
                raise ValueError(f"{t}: MOAKS grade {k} is outside the {TARGET_SCALE[t]} scale")
            sel = ok & (g == k)
            state[sel] = rung
            val[sel] = LADDER[rung]
        out[t] = val
        out[t + "__state"] = state
    keep = ~(out[[t + "__state" for t in TARGETS]] == SILENT).all(axis=1)
    return out[keep].reset_index(drop=True)


def build(labels_dir=DATA / "labels", studies_csv=DATA / "split_files" / "all_studies.csv"):
    """-> (label table, manifest of emitted rows, rebuilt grade frame indexed by uid)."""
    grades = max_grades(load_readings(labels_dir))
    man = load_manifest(studies_csv)

    key = list(zip(man.patient_id, man.knee_side, man.time))
    have = grades.index
    hit = [k in have for k in key]
    n_unread = len(man) - sum(hit)
    n_orphan = len(have) - sum(hit)
    man = man[hit].reset_index(drop=True)
    g = grades.loc[list(zip(man.patient_id, man.knee_side, man.time))].reset_index(drop=True)
    g.index = man.uid.values

    tbl = ladder_from_grades(man.uid.values, g)
    man = man[man.uid.isin(set(tbl.StudyInstanceUID))].reset_index(drop=True)
    g = g.loc[man.uid.values]
    print(f"  volumes without a same-visit reading   {n_unread}  (18M manual-only studies)")
    print(f"  graded knee-visits without a volume    {n_orphan}  (dropped; nothing to train on)")
    return tbl, man, g


# ------------------------------------------------------------------------------- folds --

def merged_fold_table(tbl, man, folds_csv=ROOT / "folds.csv"):
    """folds.csv + one row per full-scale knee-visit at fold = -1, group = OAI|{patient}.
    Competition rows pass through byte-identical; fold -1 joins every fold's train side and
    no fold's val side. Grouping by participant is what the pilot did not need and full
    scale MUST have: both knees and up to 5 visits share one group."""
    folds = pd.read_csv(folds_csv)
    if folds.StudyInstanceUID.astype(str).str.startswith(("oaizib", "oaifull")).any():
        raise ValueError("folds.csv contains OAI rows; it must stay OAI-free -- train.py "
                         "hardcodes it and --random-folds permutes it.")
    pid = dict(zip(man.uid, man.patient_id))
    add = pd.DataFrame({"StudyInstanceUID": sorted(tbl.StudyInstanceUID.astype(str))})
    for c in folds.columns:
        if c == "StudyInstanceUID":
            continue
        if c == "group":
            add[c] = [f"OAI|{pid[u]}" for u in add.StudyInstanceUID]
        else:
            add[c] = {"fold": -1, "is_gold": False}.get(c, "OAI_EXTERNAL")
    return pd.concat([folds, add[folds.columns.tolist()]], ignore_index=True)


# ------------------------------------------------------------------------- validation --

def _fail(msg):
    raise AssertionError(msg)


def validate(tbl, grades, pilot_csv=OAI_CACHE / "oai_cache_labels_ladder.csv"):
    """Every invariant the ladder table must hold, against a REBUILD from source. Raises
    AssertionError naming the violated rule. This is what the corruption demos must fail."""
    states = tbl[[t + "__state" for t in TARGETS]].values
    vals = tbl[TARGETS].values.astype(np.float64)
    trained = states != SILENT

    # schema: exactly the pilot ladder file's columns, in its order
    if pilot_csv and Path(pilot_csv).exists():
        want = pd.read_csv(pilot_csv, nrows=0).columns.tolist()
        if tbl.columns.tolist() != want:
            _fail("SCHEMA: columns differ from oai_cache_labels_ladder.csv")

    # keys: unique, parseable, and pointing at a real graded knee-visit
    uids = tbl.StudyInstanceUID.astype(str)
    if uids.duplicated().any():
        _fail("KEYS: duplicate StudyInstanceUID")
    if uids.str.startswith("oaizib").any():
        _fail("KEYS: a uid collides with the pilot's oaizib namespace")
    for u in uids:
        if not UID_RE.match(u):
            _fail(f"KEYS: uid {u!r} does not parse as oaifull_id_visit_side")
    missing = set(uids) - set(grades.index.astype(str))
    if missing:
        _fail(f"KEYS: {len(missing)} uids have no graded source knee-visit")

    # values: ladder rungs only, no hard 0/1 on a trained cell, masked cells exactly 0.0
    used = sorted({round(float(v), 4) for v in vals[trained]})
    if not set(used) <= {round(v, 4) for v in LADDER.values()}:
        _fail(f"VALUES: off-ladder values {used}")
    if 0.0 in used or 1.0 in used:
        _fail("VALUES: hard 0/1 on a trained cell")
    if not (vals[~trained] == 0.0).all():
        _fail("VALUES: a masked cell is not 0.0 (0*nan poisons the batch)")

    # state and value written together
    for i, t in enumerate(TARGETS):
        s = states[:, i]
        for name in set(s[s != SILENT]):
            if not np.allclose(vals[s == name, i], LADDER[name]):
                _fail(f"STATE: {t}/{name} value disagrees with the ladder")

    # the uncovered three: silent on every row
    for t in UNCOVERED:
        if not (tbl[t + "__state"] == SILENT).all() or not (tbl[t] == 0.0).all():
            _fail(f"UNCOVERED: {t} asserted on some knee")

    # silent == unread AND every graded cell matches a rebuild from source, cell for cell.
    # This is the check that catches an invented confident negative (state where source is
    # NaN) and a row carrying another knee's findings (side/visit inconsistency: the grades
    # under this uid no longer reproduce its cells).
    g = grades.loc[list(uids)]
    for t in TARGET_SCALE:
        src = pd.to_numeric(g[t], errors="coerce").values
        st = tbl[t + "__state"].values
        if ((st == SILENT) != np.isnan(src)).any():
            _fail(f"SOURCE: {t}: silent/unread mismatch -- an invented or dropped cell")
        tab = SCALES[TARGET_SCALE[t]]
        ok = ~np.isnan(src)
        expect = np.array([tab.get(int(k), "present_ungraded") if k == k else SILENT
                           for k in src], dtype=object)
        if (st[ok] != expect[ok]).any():
            n = int((st[ok] != expect[ok]).sum())
            _fail(f"SOURCE: {t}: {n} cells disagree with a rebuild from the MOAKS source")


def validate_folds(merged, tbl, man, folds_csv=ROOT / "folds.csv",
                   patient_map_csv=ROOT / "patient_map.csv"):
    folds = pd.read_csv(folds_csv)
    oai = merged.StudyInstanceUID.astype(str).str.startswith("oaifull")
    if int(oai.sum()) != len(tbl):
        _fail("FOLDS: OAI row count differs from the label table")
    if not (merged.loc[oai, "fold"] == -1).all():
        _fail("FOLDS: an OAI row carries a fold other than -1")
    if merged.loc[oai, "is_gold"].any():
        _fail("FOLDS: an OAI row claims is_gold")
    if not merged.loc[~oai].reset_index(drop=True).equals(folds):
        _fail("FOLDS: competition rows changed on the way through the merge")
    # participant grouping: group encodes the participant in the uid, one group per person
    grp = merged.loc[oai, ["StudyInstanceUID", "group"]]
    for u, gname in zip(grp.StudyInstanceUID, grp.group):
        if gname != f"OAI|{UID_RE.match(u).group(1)}":
            _fail(f"FOLDS: group {gname!r} does not match participant in {u!r}")
    for k in range(5):
        va = merged.loc[merged.fold == k, "StudyInstanceUID"].astype(str)
        if va.str.startswith("oaifull").any():
            _fail(f"FOLDS: OAI knee in validation fold {k}")
    # different populations, but assert it: no OAI participant id is a competition patient
    pmap = pd.read_csv(patient_map_csv)
    comp = set(pmap.PatientID.astype(str))
    ours = set(man.patient_id.astype(str))
    if comp & ours:
        _fail(f"FOLDS: {len(comp & ours)} participants appear in BOTH competition and OAI")


# -------------------------------------------------------------------------- reporting --

def teacher_rates(labels_csv=ROOT / "report_labels_synfix.csv"):
    """Positive rate per target in the teacher's labels, over ALL studies -- a report that
    does not mention a finding is a clinical negative, so not_mentioned counts in the base.
    This is the arithmetic of the Contusion precedent ("82.9% here vs the teacher's
    report-contusion rate"), and the stricter of the two possible bases: counting only
    mentioned cells would hide exactly the sensitivity gaps this gate exists to surface.
    Positive iff the STATE asserts presence, matching oai_ladder's rung split."""
    d = pd.read_csv(labels_csv)
    masked = set(NEGATIVE_RUNGS) | {"not_mentioned"}
    out = {}
    for t in TARGETS:
        s = d[t + "__state"].astype(str)
        p = int((~s.isin(masked)).sum())
        out[t] = (len(d), p, p / len(d))
    return out


def _lift(pos_a, pos_t):
    """P(target+ | anchor+) - P(target+ | anchor-), on rows where both are readable."""
    m = pos_a.notna() & pos_t.notna()
    a, t = pos_a[m].astype(bool), pos_t[m].astype(bool)
    if a.sum() == 0 or (~a).sum() == 0:
        return np.nan
    return t[a].mean() - t[~a].mean()


def report(tbl, teacher_csv=ROOT / "report_labels_synfix.csv"):
    """Per-target supervision report + the semantic-mismatch gate. RAISES if a flagged
    target has no documented decision, or a documented decision went stale."""
    states = tbl[[t + "__state" for t in TARGETS]].values
    trained = states != SILENT
    teach = teacher_rates(teacher_csv)

    def posvec(frame, t, masked):
        s = frame[t + "__state"].astype(str)
        v = pd.Series(np.where(s.isin([SILENT, "not_mentioned"]), np.nan,
                               (~s.isin(masked)).astype(float)), index=frame.index)
        return v

    teacher_df = pd.read_csv(teacher_csv)
    masked = set(NEGATIVE_RUNGS) | {"not_mentioned"}
    print(f"\n  knees {len(tbl)}   supervised cells {int(trained.sum())} of {trained.size}")
    print(f"\n  {'target':18s} {'cells':>6} {'pos':>6} {'rate':>7} {'teacher':>8} "
          f"{'ratio':>6}   {'lift oai':>9} {'lift teach':>10}   verdict")
    flagged = []
    for i, t in enumerate(TARGETS):
        n = int(trained[:, i].sum())
        p = int((~np.isin(states[:, i], NEGATIVE_RUNGS)).sum())
        rate = p / max(n, 1)
        tn, tp, tr = teach[t]
        ratio = rate / tr if tr > 0 else np.inf
        if t in UNCOVERED:
            verdict = ("FORCED SILENT (semantic mismatch, oai_ladder rule 4)"
                       if t == "Contusion" else "not scored by MOAKS")
            lo = lt = np.nan
        elif n == 0:
            verdict, lo, lt = "no readings", np.nan, np.nan
        else:
            lo = _lift(posvec(tbl, "Medial OA", set(NEGATIVE_RUNGS)),
                       posvec(tbl, t, set(NEGATIVE_RUNGS)))
            lt = _lift(posvec(teacher_df, "Medial OA", masked), posvec(teacher_df, t, masked))
            if ratio > MISMATCH_RATIO:
                flagged.append(t)
                verdict = "FLAGGED >2.5x -- " + ("decided: include" if t in DECISIONS
                                                 else "NO DECISION")
            else:
                verdict = "ok"
        print(f"  {t:18s} {n:>6} {p:>6} {100*rate:>6.1f}% {100*tr:>7.1f}% "
              f"{ratio:>6.2f}   {lo:>+9.3f} {lt:>+10.3f}   {verdict}"
              if lo == lo else
              f"  {t:18s} {n:>6} {p:>6} {100*rate:>6.1f}% {100*tr:>7.1f}% "
              f"{ratio:>6.2f}   {'--':>9} {'--':>10}   {verdict}")

    undecided = [t for t in flagged if t not in DECISIONS]
    stale = [t for t in DECISIONS if t not in flagged]
    if undecided:
        raise ValueError(f"semantic-mismatch gate: {undecided} exceed {MISMATCH_RATIO}x the "
                         "teacher's positive rate with NO documented decision. Add an entry "
                         "to DECISIONS or exclude the target; silent inclusion is the "
                         "Contusion failure mode.")
    if stale:
        raise ValueError(f"DECISIONS documents {stale} but the scan no longer flags them; "
                         "remove the stale entries so the documentation tracks the data.")
    if flagged:
        print(f"\n  decisions on the {len(flagged)} flagged targets:")
        for t in flagged:
            print(f"    {t:18s} {DECISIONS[t]}")
    return flagged


# -------------------------------------------------------------------------- self-test --

def self_test():
    print("build_reasonknee_labels self-test\n")
    tbl, man, grades = build()
    validate(tbl, grades)
    merged = merged_fold_table(tbl, man)
    validate_folds(merged, tbl, man)
    print(f"  honest build passes validation       ok  {len(tbl)} knees")

    # participants really do have both knees and multiple visits at full scale -- the
    # reason grouping is mandatory here and was not in the pilot
    pids = man.patient_id
    per = man.groupby("patient_id").size()
    both = man.groupby("patient_id").knee_side.nunique()
    assert (per > 1).sum() > 1000 and (both > 1).sum() > 300, \
        "full scale does not look multi-knee/multi-visit; grouping claim is stale"
    print(f"  grouping is load-bearing             ok  {(per > 1).sum()} participants with "
          f">1 row, {(both > 1).sum()} with both knees")

    def expect_fail(name, corrupt):
        c = copy.deepcopy(tbl)
        corrupt(c)
        try:
            validate(c, grades)
        except AssertionError as e:
            print(f"  {name:37s}REJECTED  ({str(e)[:60]})")
            return True
        print(f"  {name:37s}ACCEPTED -- THE VALIDATOR IS BLIND")
        return False

    results = []

    # 1. a confident negative invented on a cell MOAKS never read
    def corrupt_neg(c):
        col = "Effusion__state"
        i = c.index[c[col] == SILENT][0]
        c.loc[i, col] = "absent"
        c.loc[i, "Effusion"] = np.float32(LADDER["absent"])
    results.append(expect_fail("A confident negative on unread cell   ", corrupt_neg))

    # 2. a hard 1.0 on an asserted cell
    def corrupt_hard(c):
        i = c.index[c["Medial OA__state"] == "severe"][0]
        c.loc[i, "Medial OA"] = np.float32(1.0)
    results.append(expect_fail("B hard 1.0 on an asserted cell        ", corrupt_hard))

    # 3. inconsistent SIDE: one participant's left-knee row silently carrying the right
    # knee's findings -- the join trap the pilot proved matters (both-knee readings differ)
    def corrupt_side(c):
        ids = c.StudyInstanceUID.astype(str)
        info = pd.DataFrame([UID_RE.match(u).groups() for u in ids],
                            columns=["pid", "time", "side"])
        info["uid"] = ids.values
        pair = None
        for (pid, tm), g in info.groupby(["pid", "time"]):
            if set(g.side) == {"R", "L"}:
                ur = g.uid[g.side == "R"].iloc[0]
                ul = g.uid[g.side == "L"].iloc[0]
                a = c[c.StudyInstanceUID == ur].iloc[0]
                b = c[c.StudyInstanceUID == ul].iloc[0]
                if not a[1:].equals(b[1:]):
                    pair = (ur, ul)
                    break
        assert pair, "no both-knee pair with differing findings; corruption cannot be staged"
        src = c.index[c.StudyInstanceUID == pair[0]][0]
        dst = c.index[c.StudyInstanceUID == pair[1]][0]
        cols = [x for x in c.columns if x != "StudyInstanceUID"]
        c.loc[dst, cols] = c.loc[src, cols].values
    results.append(expect_fail("C participant with inconsistent SIDE  ", corrupt_side))

    # and the fold table refuses a participant reaching a validation fold
    def corrupt_fold():
        m2 = merged.copy()
        i = m2.index[m2.StudyInstanceUID.astype(str).str.startswith("oaifull")][0]
        m2.loc[i, "fold"] = 2
        try:
            validate_folds(m2, tbl, man)
        except AssertionError as e:
            print(f"  {'D OAI knee handed fold 2              ':37s}REJECTED  ({str(e)[:60]})")
            return True
        print("  D OAI knee handed fold 2              ACCEPTED -- THE VALIDATOR IS BLIND")
        return False
    results.append(corrupt_fold())

    if not all(results):
        raise SystemExit("\n  A CORRUPTION WAS ACCEPTED -- do not ship this table")
    print(f"\n  ALL SELF-TESTS PASSED -- {sum(results)}/4 corruptions rejected")


# -------------------------------------------------------------------------------- main --

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", default=str(DATA / "labels"))
    ap.add_argument("--studies", default=str(DATA / "split_files" / "all_studies.csv"))
    ap.add_argument("--teacher", default=str(ROOT / "report_labels_consensus.csv"))
    ap.add_argument("--out", default=str(OAI_CACHE / "oai_full_labels_ladder.csv"))
    ap.add_argument("--manifest-out", default=str(OAI_CACHE / "oai_full_manifest.csv"))
    ap.add_argument("--folds-out", default=str(OAI_CACHE / "folds_oai_full.csv"))
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return

    tbl, man, grades = build(Path(a.labels_dir), Path(a.studies))
    validate(tbl, grades)
    merged = merged_fold_table(tbl, man)
    validate_folds(merged, tbl, man)
    report(tbl, Path(a.teacher))

    tbl.to_csv(a.out, index=False)
    print(f"\n  wrote {a.out}  ({len(tbl)} knees)")
    man = man.rename(columns={"path": "oai_dicom_path"})
    keep = ["uid", "patient_id", "knee_side", "time", "n_volumes",
            "oai_dicom_path", "image_filepath", "label_filepath"]
    man[[c for c in keep if c in man.columns]].to_csv(a.manifest_out, index=False)
    print(f"  wrote {a.manifest_out}  (uid -> OAI series for the acquire/cache lane)")
    merged.to_csv(a.folds_out, index=False)
    n_oai = int(merged.StudyInstanceUID.astype(str).str.startswith("oaifull").sum())
    print(f"  wrote {a.folds_out}  ({len(merged)} rows, {n_oai} OAI at fold=-1, "
          f"grouped over {man.patient_id.nunique()} participants)")


if __name__ == "__main__":
    main()
