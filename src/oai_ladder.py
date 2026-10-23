"""
oai_ladder.py -- MOAKS grades onto our soft-label ladder, plus the merged fold table.

ONE COMMAND produces everything the OAI auxiliary-supervision run consumes:

    python oai_ladder.py            # writes BOTH tables (paths below)
    python oai_ladder.py --self-test

Outputs (both on D:, never in the repo -- the fold table carries competition UIDs and the
repo is public):

    data/oai_cache/oai_cache_labels_ladder.csv   the OAI label table
    data/oai_cache/folds_oai.csv                 folds.csv + 246 OAI rows at fold = -1

WHY THIS FILE EXISTS

The original MOAKS -> label mapping, `build_oai_cache.label_table()`, wrote

    out[t] = np.where(np.isnan(v), 0.0, v)                       # v is 0/1 from binarise()
    out[t + "__state"] = np.where(np.isnan(v), "silent", "present_ungraded")

and the file it produced holds exactly {0.0, 1.0} in every target column. The adversarial
verification measured the consequence: 23.6% of the WEAK-4 supervision mass was a hard 0.0
on a knee a radiologist graded ABNORMAL (grade 1: cartilage loss under 10% of the surface,
meniscal intrasubstance signal, a physiological effusion) -- invented confident negatives,
the exact failure fix_synovitis_labels.py exists to prevent. This file replaces that table.

THE FOUR RULES, in the order they were violated

  1. A cell MOAKS did not read is state="silent". Masked. Never 0.0-with-an-asserted-state.
  2. A cell MOAKS graded lands on OUR ladder (0.03 .. 0.95), never on hard 0.0 / 1.0. Every
     other supervision source in this project is soft; a hard label's pull never saturates
     under BCE and out-pulls every teacher cell (at p = 0.90 a 1.0 target pulls 2.0x the
     most confident report cell we have).
  3. A MOAKS grade 0 IS a real asserted absence -- a radiologist looked and graded it. It
     earns the 0.03 rung, unlike the LLM's invented `absent`.
  4. Contusion is SILENT on every OAI row. Not a prevalence shift -- a changed definition.
     MOAKS "Contusion" is V00MBM*, the bone-marrow-lesion family (82.9% positive here vs
     16.5% in our reports); a BML is a feature OF osteoarthritis, a report "contusion" is
     acute trauma, and the conditional relationship reverses sign (OA anti-predicts
     contusion in our reports, lift -1.0 to -6.5 pp; it strongly predicts BML in OAI, +19.2
     to +36.8 pp). Training that head on MOAKS teaches it the wrong sign. Forced silent at
     the LABEL level so no selection flag downstream can re-admit it by accident.

THE LADDER USED HERE, and why each rung

  cartilage (Medial OA / Lateral OA / PF OA)   MOAKS 0-3, ordinal in area of loss
      0 absent 0.03 | 1 uncertain 0.35 | 2 moderate 0.75 | 3 severe 0.95
  meniscus (Medial / Lateral)                  MOAKS 0-8, a MORPHOLOGY CODE, not a scale
      0 absent 0.03 | 1 intrasubstance -> uncertain 0.35
      2-5 discrete tear (vertical / horizontal / complex / radial-root) -> present_ungraded 0.62
      6-8 maceration -> severe 0.95
      2-5 collapse to ONE rung on purpose: those codes name tear TYPE and cannot be ordered
      by severity; `present_ungraded` is exactly "asserted, no severity claim". 6-8 are the
      maceration codes, grouped for the same reason, at the rung a report would reach for a
      destroyed meniscus. The one target where the mapping is a judgement rather than a
      reading of an ordinal scale; isolated in SCALES["meniscus"] so it changes in one place.
  effusion                                     MOAKS 0-3, grade 1 is physiological
      0 absent 0.03 | 1 uncertain 0.35 | 2 moderate 0.75 | 3 severe 0.95
  synovitis                                    MOAKS 0-3, abnormal at any grade
      0 absent 0.03 | 1 mild 0.45 | 2 moderate 0.75 | 3 severe 0.95
  ACL tear                                     MOAKS 0-2
      0 absent 0.03 | 1 partial -> moderate 0.75 | 2 complete -> severe 0.95
  Baker's cyst                                 MOAKS 0-1, presence only
      0 absent 0.03 | 1 present_ungraded 0.62

  MCL and Fracture are not scored by MOAKS; Contusion is excluded (rule 4). All three are
  written 0.0 with state "silent", which KneeStudies masks out of the loss. 0.0 and not NaN
  because the mask MULTIPLIES and 0 * nan is nan, which would take the whole batch with it.

FOR THE RECORD -- two Contusion findings preserved from the earlier revision, in case a
wider OAI pull ever re-admits the target (that requires editing UNCOVERED here AND restoring
both guards):
  * build_oai_labels.moaks_columns() pools all 45 V00MBM* columns and cuts at >= 1, but they
    are three different items -- V00MBMN* a COUNT, V00MBMP* a PERCENT, V00MBMS* the SIZE.
    Corpus-wide, 29 of 3,154 knees were called positive only because a count/percent column
    reached 1 while the size was 0. Use the SIZE items alone.
  * Contusion is scored over 15 subregions; 33 of our 246 knees had exactly ONE read. A
    negative on 1/15 of the evidence is an unread cell, not an absence -- mask it.

THE MERGED FOLD TABLE, and the leak it is shaped against

folds_oai.csv is folds.csv with 246 OAI rows appended at fold = -1: in a `fold != k` /
`fold == k` split, -1 joins every fold's train side and no fold's val side, so an OAI knee
is structurally incapable of reaching an OOF. Two hard rules, both enforced by the
self-test and by pending_patches/verify_oai_aux.py:

  * OAI ids NEVER enter folds.csv itself -- the file train.py hardcodes. train.py's
    --random-folds does `folds['fold'] = rng.permutation(folds.fold.values)`, which
    redistributes whatever values the table holds; run on a table containing 246 rows at
    -1 it would put ~233 of them into a validation fold (measured by simulation below).
    folds.csv stays OAI-free, so that path cannot touch them.
  * Any tool that DOES consume folds_oai.csv must treat fold = -1 as train-only and must
    never permute it. The self-test demonstrates the hazard rather than describing it.

The current wiring (pending_patches/OAI-aux-supervision.md) attaches the OAI knees via
ConcatDataset and never hands train.py any fold table but folds.csv; folds_oai.csv is the
canonical statement of fold membership for every other consumer.

WHAT THIS FILE DOES NOT DO, DELIBERATELY

It writes label and fold tables. It does not build the cache, does not modify any locked
file, and does not overwrite the binarised oai_cache_labels.csv, so the two label files can
be compared. It also does not score anything: at n = 246 the per-target AUC SD is ~0.035,
noisier than any effect being chased, so nothing here computes an AUC or selects anything.

CPU only, never touches the GPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from build_oai_labels import load_moaks, load_zib_index, moaks_columns

ROOT = Path(__file__).resolve().parent
OAI_CACHE = Path(r"data/oai_cache")

# model.py's target order. Retyped rather than imported so this stays a numpy/pandas script
# and does not become a new import edge into a file the live run's dataloader workers
# re-import.
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# llm_label_reports.SEVERITY_SCORE. KneeStudies masks a cell iff its state is "silent" or
# "not_mentioned", so every name used below except SILENT names a TRAINED cell.
LADDER = {"severe": 0.95, "moderate": 0.75, "present_ungraded": 0.62, "mild": 0.45,
          "uncertain": 0.35, "not_mentioned": 0.15, "absent": 0.03}
SILENT = "silent"

# MOAKS grade -> ladder rung, per item family. Keyed by rung NAME so the ladder above stays
# the single source of the numbers. A grade missing from a table is out of range for that
# scale and ladder_table raises rather than guessing; the self-test exercises it on the
# real data.
SCALES = {
    "cartilage": {0: "absent", 1: "uncertain", 2: "moderate", 3: "severe"},
    "meniscus":  {0: "absent", 1: "uncertain", 2: "present_ungraded", 3: "present_ungraded",
                  4: "present_ungraded", 5: "present_ungraded",
                  6: "severe", 7: "severe", 8: "severe"},
    "effusion":  {0: "absent", 1: "uncertain", 2: "moderate", 3: "severe"},
    "synovitis": {0: "absent", 1: "mild", 2: "moderate", 3: "severe"},
    "acl":       {0: "absent", 1: "moderate", 2: "severe"},
    "presence":  {0: "absent", 1: "present_ungraded"},
}
TARGET_SCALE = {
    "Medial OA": "cartilage", "Lateral OA": "cartilage", "PF OA": "cartilage",
    "Medial Meniscus": "meniscus", "Lateral Meniscus": "meniscus",
    "Effusion": "effusion", "Synovitis": "synovitis",
    "ACL": "acl", "Baker's": "presence",
}
# MCL / Fracture: MOAKS scores neither. Contusion: semantic mismatch, forced silent -- see
# rule 4 in the docstring. Masked on every row, never a negative.
UNCOVERED = ("MCL", "Fracture", "Contusion")

# A cell counts as POSITIVE iff its rung asserts the finding is there. NOT `value > 0.5`:
# Synovitis grade 1 lands on `mild` = 0.45, an assertion of presence sitting below 0.5, and
# counting by value would report 37 Synovitis positives where MOAKS read 108. This is the
# split oai_aux.class_weights prices on, so it lives here, next to the ladder it reads.
NEGATIVE_RUNGS = (SILENT, "absent", "uncertain")


def max_grades(moaks):
    """-> one row per (ID, SIDE), MultiIndexed; the max grade per target across subregions
    and reading projects, matching build_oai_labels.binarise(): the readings are of the same
    knee, so the worst thing any reader saw in any subregion is what the knee has. NaN iff
    nothing was read."""
    fam = {t: cols for t, cols in moaks_columns(moaks.columns).items()
           if t in TARGET_SCALE}
    key = list(zip(moaks.ID, moaks.SIDE))
    grade = {}
    for t, cols in fam.items():
        v = moaks[cols].apply(pd.to_numeric, errors="coerce").assign(_k=key).groupby("_k")
        grade[t] = v[cols].max().max(axis=1)
    out = pd.DataFrame(grade)
    out.index = pd.MultiIndex.from_tuples(out.index, names=["ID", "SIDE"])
    return out


def ladder_table():
    """-> the frame KneeStudies indexes: StudyInstanceUID, the 12 targets, 12 __state cols.

    StudyInstanceUID is the oaizib_NNN id, which is what build_oai_cache wrote the .npy
    files under. Only knees with at least one readable target are emitted.
    """
    zib = load_zib_index()
    grade = max_grades(load_moaks())
    m = zib.merge(grade.reset_index(), left_on=["SubjectID", "KneeSide"],
                  right_on=["ID", "SIDE"], how="left")

    out = pd.DataFrame({"StudyInstanceUID": m.oaizib.values})
    for t in TARGETS:
        if t in UNCOVERED:
            out[t] = np.zeros(len(m), np.float32)
            out[t + "__state"] = SILENT
            continue
        g = pd.to_numeric(m[t], errors="coerce").values
        state = np.full(len(m), SILENT, dtype=object)
        val = np.zeros(len(m), np.float32)
        tab = SCALES[TARGET_SCALE[t]]
        ok = ~np.isnan(g)
        for k in np.unique(g[ok]).astype(int):
            if k in tab:
                rung = tab[k]
            elif k > max(tab):
                # Above the documented top of the scale. The finding is real, the severity
                # is not readable, and `present_ungraded` is exactly that claim. Anything
                # else (a negative code, a non-integer) still raises below.
                rung = "present_ungraded"
            else:
                raise ValueError(f"{t}: MOAKS grade {k} is outside the {TARGET_SCALE[t]} scale")
            sel = ok & (g == k)
            state[sel] = rung
            val[sel] = LADDER[rung]
        out[t] = val
        out[t + "__state"] = state

    keep = ~(out[[t + "__state" for t in TARGETS]] == SILENT).all(axis=1)
    return out[keep].reset_index(drop=True)


def merged_fold_table(tbl=None, folds_csv=ROOT / "folds.csv"):
    """folds.csv + one row per OAI knee at fold = -1. Competition rows pass through
    byte-identical; the OAI rows carry a group name no scanner fingerprint can produce and
    is_gold = False, so no reporting path mistakes them for competition studies."""
    folds = pd.read_csv(folds_csv)
    if folds.StudyInstanceUID.astype(str).str.startswith("oaizib").any():
        raise ValueError(
            "folds.csv already contains OAI rows. It must stay OAI-free: train.py "
            "hardcodes it, and its --random-folds path permutes fold values, which would "
            "move train-only rows into validation folds.")
    ids = sorted((tbl if tbl is not None else ladder_table()).StudyInstanceUID.astype(str))
    add = pd.DataFrame({"StudyInstanceUID": ids})
    for c in folds.columns:
        if c == "StudyInstanceUID":
            continue
        add[c] = {"fold": -1, "is_gold": False}.get(c, "OAI_EXTERNAL")
    return pd.concat([folds, add[folds.columns.tolist()]], ignore_index=True)


def summarise(tbl):
    states = tbl[[t + "__state" for t in TARGETS]].values
    trained = states != SILENT
    print(f"  knees {len(tbl)}   supervised cells {int(trained.sum())} of {trained.size}\n")
    print(f"  {'target':18s} {'cells':>6} {'pos':>6} {'rate':>7}   rungs used")
    for i, t in enumerate(TARGETS):
        n = int(trained[:, i].sum())
        p = int((~np.isin(states[:, i], NEGATIVE_RUNGS)).sum())
        rungs = tbl.loc[trained[:, i], t + "__state"].value_counts().to_dict()
        note = ("FORCED SILENT (semantic mismatch)" if t == "Contusion"
                else "not scored by MOAKS" if n == 0 else rungs)
        print(f"  {t:18s} {n:>6} {p:>6} {100 * p / max(n, 1):>6.1f}%   {note}")
    return int(trained.sum())


def self_test():
    print("oai_ladder self-test\n")
    tbl = ladder_table()
    states = tbl[[t + "__state" for t in TARGETS]].values
    vals = tbl[TARGETS].values
    trained = states != SILENT

    # 1. every emitted value is a ladder rung, and 0.0 / 1.0 never appear on a trained cell.
    # round through float64: the column is float32 downstream (build_oai_cache's contract,
    # what KneeStudies casts to), so 0.03 comes back 0.029999999329 and a bare == would fail.
    used = sorted({round(float(v), 4) for v in vals[trained]})
    assert set(used) <= {round(v, 4) for v in LADDER.values()}, f"off-ladder values {used}"
    assert 0.0 not in used and 1.0 not in used, f"hard 0/1 emitted on a trained cell: {used}"
    assert (vals[~trained] == 0.0).all(), "masked cell is not 0.0; 0*nan would poison the batch"
    print(f"  values are ladder rungs only         ok  {used}")

    # 2. state and value agree cell by cell -- they are written together, so assert it
    for i, t in enumerate(TARGETS):
        s = states[:, i]
        for name in set(s[s != SILENT]):
            v = vals[s == name, i]
            assert np.allclose(v, LADDER[name]), f"{t}/{name}: {set(v)} != {LADDER[name]}"
    print(f"  state matches value                  ok  {len(LADDER)} rungs checked")

    # 3. MCL, Fracture AND Contusion masked everywhere -- never a negative. Contusion is the
    # semantic-mismatch exclusion: this check is what makes it label-level rather than a
    # selection flag someone can override.
    for t in UNCOVERED:
        assert (tbl[t + "__state"] == SILENT).all(), f"{t} asserted on some knee"
        assert (tbl[t] == 0.0).all(), f"{t} carries a nonzero value"
    print(f"  MCL / Fracture / Contusion silent    ok  {len(tbl)} knees each")

    # 4. THE BUG THIS FILE EXISTS FOR: subthreshold grade 1 must not be a confident negative
    moaks = load_moaks()
    grade = max_grades(moaks)
    zib = load_zib_index()
    m = zib.merge(grade.reset_index(), left_on=["SubjectID", "KneeSide"],
                  right_on=["ID", "SIDE"], how="left").set_index("oaizib")
    ti = tbl.set_index("StudyInstanceUID")
    n_g1 = 0
    for t in ("Medial OA", "Lateral OA", "PF OA", "Medial Meniscus", "Lateral Meniscus",
              "Effusion"):
        ids = [i for i in m.index[pd.to_numeric(m[t], errors="coerce") == 1] if i in ti.index]
        assert (ti.loc[ids, t] == LADDER["uncertain"]).all(), \
            f"{t} grade 1 is not on the uncertain rung"
        n_g1 += len(ids)
    assert n_g1 > 200, f"only {n_g1} grade-1 cells found; the check is not exercising anything"
    print(f"  grade 1 -> uncertain 0.35            ok  {n_g1} cells rescued from a hard 0.0")

    # 5. silent <-> unread, exactly. For every covered target: state is silent iff MOAKS has
    # no reading for that knee. This is the invariant whose violation is an invented
    # confident negative, so it gets its own check rather than riding on check 1.
    for t in TARGET_SCALE:
        g = pd.to_numeric(m[t], errors="coerce")
        have = set(ti.index)
        unread = {i for i in m.index[g.isna()] if i in have}
        silent = set(ti.index[ti[t + "__state"] == SILENT])
        assert unread == silent, (f"{t}: silent/unread mismatch -- "
                                  f"{len(unread ^ silent)} cells differ")
    print(f"  silent == unread, cell for cell      ok  {len(TARGET_SCALE)} covered targets")

    # 6. the fold table: OAI at -1 only, competition rows untouched, and the split that
    # train.py's build_loaders performs can never validate an OAI knee.
    folds = pd.read_csv(ROOT / "folds.csv")
    assert not folds.StudyInstanceUID.astype(str).str.startswith("oaizib").any(), \
        "folds.csv contains OAI rows; train.py --random-folds would validate them"
    merged = merged_fold_table(tbl)
    oai = merged.StudyInstanceUID.astype(str).str.startswith("oaizib")
    assert int(oai.sum()) == len(tbl) and (merged.loc[oai, "fold"] == -1).all(), \
        "an OAI row carries a fold id other than -1"
    assert not merged.loc[oai, "is_gold"].any(), "an OAI row claims is_gold"
    comp = merged.loc[~oai].reset_index(drop=True)
    assert comp.equals(folds), "competition rows changed on the way through the merge"
    for k in range(5):
        va = merged.loc[merged.fold == k, "StudyInstanceUID"].astype(str)
        trn = merged.loc[merged.fold != k, "StudyInstanceUID"].astype(str)
        assert not va.str.startswith("oaizib").any(), f"fold {k}: OAI knee in validation"
        assert trn.str.startswith("oaizib").sum() == len(tbl), f"fold {k}: OAI knee missing"
    print(f"  fold -1 is train-only in every fold  ok  {len(tbl)} train / 0 val, folds 0-4")

    # 7. the hazard the two rules above exist for, demonstrated rather than described:
    # train.py's --random-folds line is `folds['fold'] = rng.permutation(folds.fold.values)`.
    # Run on the MERGED table it scatters the -1s and puts most OAI knees into a validation
    # fold. That is why folds.csv stays OAI-free and why no consumer may permute folds_oai.
    rng = np.random.default_rng(2026)
    perm = merged.copy()
    perm["fold"] = rng.permutation(perm.fold.values)
    leaked = int(perm.loc[perm.StudyInstanceUID.astype(str).str.startswith("oaizib"),
                          "fold"].ge(0).sum())
    assert leaked > 200, f"only {leaked} leak under permutation; the hazard claim is stale"
    print(f"  --random-folds hazard is real        ok  permuting the merged table would put "
          f"{leaked}/{len(tbl)} OAI knees into a val fold")
    print("\n  ALL SELF-TESTS PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OAI_CACHE / "oai_cache_labels_ladder.csv"))
    ap.add_argument("--folds-out", default=str(OAI_CACHE / "folds_oai.csv"))
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    tbl = ladder_table()
    n = summarise(tbl)
    tbl.to_csv(a.out, index=False)
    print(f"\n  wrote {a.out}  ({len(tbl)} knees, {n} supervised cells)")
    merged = merged_fold_table(tbl)
    merged.to_csv(a.folds_out, index=False)
    n_oai = int(merged.StudyInstanceUID.astype(str).str.startswith("oaizib").sum())
    print(f"  wrote {a.folds_out}  ({len(merged)} rows, {n_oai} OAI at fold=-1)")


if __name__ == "__main__":
    main()
