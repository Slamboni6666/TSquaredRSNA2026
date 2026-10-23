"""check_patient_leak.py -- does one patient contribute studies to more than one fold?

WHY THIS EXISTS
folds.csv groups on `lang x manufacturer x model` (62 site groups). That is a deliberate and
defensible choice for domain shift, but it does not constrain PATIENT identity at all. If a
single patient contributed several studies -- a follow-up scan, a re-image after a failed
series -- and those studies land in different folds, then the validation fold contains a knee
the model already trained on. Every OOF number we have would be optimistic, and the effect
would NOT show up on the leaderboard, so it would look like a real gain that fails to ship.

PatientID is present in the DICOM headers but appears in none of our indexes, so this has
never been checked. We already know 49 report texts are duplicated across 183 studies, which
is circumstantial evidence for repeat patients.

CPU only, reads one header per study with specific_tags so it is fast.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pydicom

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent


def patient_of(study: str) -> tuple[str, str | None]:
    """One header read per study. Returns (study, PatientID or None)."""
    sdir = ROOT / "train_series" / study
    try:
        f = next(sdir.rglob("*.dcm"), None)
        if f is None:
            return study, None
        ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True,
                             specific_tags=["PatientID"])
        pid = getattr(ds, "PatientID", None)
        return study, (str(pid) if pid else None)
    except Exception:
        return study, None


def main():
    folds = pd.read_csv(ROOT / "folds.csv")
    fcol = "fold" if "fold" in folds.columns else \
        [c for c in folds.columns if "fold" in c.lower()][0]
    studies = folds.StudyInstanceUID.tolist()
    print(f"  reading PatientID for {len(studies)} studies...", flush=True)

    with ThreadPoolExecutor(max_workers=16) as ex:
        got = dict(ex.map(patient_of, studies))

    folds["PatientID"] = folds.StudyInstanceUID.map(got)
    n_missing = int(folds.PatientID.isna().sum())
    d = folds[folds.PatientID.notna()]
    print(f"  resolved {len(d)}, missing {n_missing}\n")

    print(f"  distinct patients          {d.PatientID.nunique()}")
    print(f"  studies                    {len(d)}")

    vc = d.PatientID.value_counts()
    multi = vc[vc > 1]
    print(f"  patients with >1 study     {len(multi)}  "
          f"({int(multi.sum())} studies, max {int(vc.max())} per patient)")

    # THE question: do any of those patients straddle a fold boundary?
    g = d[d.PatientID.isin(multi.index)].groupby("PatientID")[fcol].nunique()
    span = g[g > 1]
    aff = d[d.PatientID.isin(span.index)]
    print(f"  ... spanning >1 FOLD       {len(span)}  ({len(aff)} studies, "
          f"{len(aff)/max(len(d),1):.2%} of the corpus)   <- LEAK")

    if len(span):
        print("\n  A patient in two folds means the model trained on one of their knees and is")
        print("  validated on another. That inflates OOF without touching the leaderboard.")
        ex_pid = span.index[0]
        rows = d[d.PatientID == ex_pid]
        print(f"\n  example patient: {len(rows)} studies in folds {sorted(rows[fcol].unique())}")

        # Does it touch the 58 gold studies -- our ranking instrument?
        T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
             "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
        tr = pd.read_csv(ROOT / "train.csv")
        gold = set(tr[tr[T].notna().all(axis=1)].StudyInstanceUID)
        hit = gold & set(aff.StudyInstanceUID)
        print(f"  gold studies affected: {len(hit)} of {len(gold)}")

    out = ROOT / "patient_map.csv"
    folds[["StudyInstanceUID", "PatientID"]].to_csv(out, index=False)
    print(f"\n  wrote {out.name} -- use it to regroup folds by patient if the leak is real")


if __name__ == "__main__":
    main()
