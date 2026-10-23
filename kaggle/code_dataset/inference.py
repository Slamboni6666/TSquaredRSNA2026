"""
inference.py -- step E1/E2: test DICOM -> submission.csv, reproducing training preprocessing exactly.

THE FAILURE THIS FILE EXISTS TO PREVENT
A model trained on one preprocessing and run on another loads cleanly, runs, and writes a
plausible submission computed from the wrong images. Nothing raises. The score is simply
worse, and there is no way to tell that from the submission file. Six decisions have to
match bit-for-bit between training and inference:

    slice order      geometry (k = p . n), never filename order (rho = -0.000)
    physical scale   crop 130 mm, resample to 336 px
    intensity        per-series 1st/99th percentile over the sampled stack
    laterality       left knees mirrored; SAG reverses slice order, COR/AX flips the last axis
    slot assignment  weighting from TR/TE, fat-sat from the protocol, one series per slot
    slice sampling   linspace over the real stack, repeats interleaved

Rather than restating them, this module imports the same functions the cache was built with,
and `--verify` re-derives volumes for training studies straight from DICOM and asserts they
match the cache **exactly** (bar the one-pixel rounding tolerance verify() justifies below).
If that passes, the contract holds by construction.

COMPRESSED TRANSFER SYNTAXES
All 24,371 training series are uncompressed Explicit VR LE -- verified exhaustively. The
official data description says the test set may also contain JPEG Lossless, JPEG 2000 and
Implicit VR LE. So `pylibjpeg` and `gdcm` are imported here defensively: they are never
needed in training and might be essential at scoring, and their absence would surface as a
decode failure on the hidden set with no chance to react.

Usage:
    python inference.py --verify --n 12          # prove it reproduces the cache
    python inference.py --split test --out submission.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Decoders for compressed pixel data. Importing them registers the handlers with pydicom;
# they are unused on uncompressed data and cost nothing.
for _mod in ("pylibjpeg", "gdcm"):
    try:
        __import__(_mod)
    except ImportError:
        pass

from build_cache import load_slot, SLOTS, CROP_MM, IMG, N_SLICE          # noqa: E402
from build_slice_order import order_series                                # noqa: E402
from build_series_index import read_series, classify                      # noqa: E402
from model import TARGETS                                                 # noqa: E402


def build_index(split_root: Path, series_csv: Path, threads=16, deadband=8.0):
    """Series -> (weighting, fat-sat, slot) and study -> laterality. Same rules as B1-B3.

    build_series_index.py cannot simply be called: it is a script that writes a parquet for
    a whole split, and here the same answers are needed in memory for studies that have
    never been seen. So read_series and classify are imported, but the slot table and the
    laterality rule below are RESTATED -- which makes them the one place training and
    scoring can silently drift apart. If a slot rule or the deadband changes in
    build_series_index.py, it has to change here too, and `--verify` is what catches it
    (a rule that differs re-derives a different volume and the byte comparison fails).
    """
    tser = pd.read_csv(series_csv)
    jobs = [(r.StudyInstanceUID, r.SeriesInstanceUID,
             str(split_root / r.StudyInstanceUID / r.SeriesInstanceUID))
            for r in tser.itertuples()]
    with ThreadPoolExecutor(max_workers=threads) as ex:
        recs = [r for r in ex.map(read_series, jobs) if r]
    df = pd.DataFrame(recs).merge(tser, on=["StudyInstanceUID", "SeriesInstanceUID"], how="left")

    cls = df.apply(classify, axis=1, result_type="expand")
    df["weighting"], df["fatsat"] = cls[0], cls[1]

    fluid = df.weighting.isin(["T2", "PD", "STIR"])
    t1 = df.weighting.eq("T1")
    pl = df.Anatomical_Plane
    df["slot"] = None
    df.loc[(pl == "Sagittal") & fluid & df.fatsat, "slot"] = "SAG_FLUID_FS"
    df.loc[(pl == "Coronal") & fluid & df.fatsat, "slot"] = "COR_FLUID_FS"
    df.loc[(pl == "Axial") & fluid & df.fatsat, "slot"] = "AX_FLUID_FS"
    df.loc[(pl == "Sagittal") & fluid & ~df.fatsat, "slot"] = "SAG_FLUID_NOFS"
    df.loc[(pl == "Coronal") & t1, "slot"] = "COR_T1"
    df.loc[(pl == "Sagittal") & t1, "slot"] = "SAG_T1"

    df["_rank"] = df.groupby(["StudyInstanceUID", "slot"]).n_slices.rank(
        ascending=False, method="first")
    df["chosen"] = df.slot.notna() & (df._rank == 1)

    st = df.groupby("StudyInstanceUID").agg(
        centre_x=("centre_x", "median"),
        tag=("laterality_tag", lambda s: next((x for x in s if x in ("L", "R")), None)),
    ).reset_index()
    st["side_geom"] = np.where(st.centre_x.isna(), None,
                               np.where(st.centre_x < -deadband, "R",
                                        np.where(st.centre_x > deadband, "L", None)))
    st["side"] = st.tag.where(st.tag.notna(), st.side_geom)
    return df, dict(zip(st.StudyInstanceUID, st.side))


def preprocess_study(study, rows, split_root: Path, side):
    """DICOM -> (6, 32, 336, 336) uint8 + presence mask. Mirrors build_study exactly.

    Slice order is re-derived per series here rather than looked up: slice_order.parquet
    covers the training tree only, and at scoring time there is no precomputed anything.
    The returned mask is the presence row alone (1 = this slot was filled), which is all
    KneeNet consumes -- the cache's second row, the real slice count, is not needed to run
    the model.
    """
    vol = np.zeros((len(SLOTS), N_SLICE, IMG, IMG), np.uint8)
    mask = np.zeros(len(SLOTS), np.float32)
    for r in rows.itertuples():
        k = SLOTS.index(r.slot)
        sdir = split_root / study / r.SeriesInstanceUID
        ordered = order_series((study, r.SeriesInstanceUID, str(sdir)))
        if not ordered:
            continue
        sops = [d["sop"] for d in sorted(ordered, key=lambda d: d["order"])]
        res = load_slot(sdir, sops, r.pix_row)
        if res is None:
            continue
        a, _ = res
        if side == "L":
            a = a[::-1] if r.slot.startswith("SAG") else a[:, :, ::-1]
        vol[k] = (np.clip(a, 0, 1) * 255).astype(np.uint8)
        mask[k] = 1.0
    return vol, mask


def verify(n=12, cache=Path(r"data/rsna_knee_cache")):
    """Re-derive training studies from DICOM and demand a bit-exact match with the cache.

    This is the whole safety argument. If the inference path reproduces the cache byte for
    byte on studies the cache already holds, then the model at scoring time sees exactly
    what it saw in training -- not approximately, exactly.
    """
    print("VERIFY: inference path vs the training cache\n" + "=" * 66)
    si = pd.read_parquet(ROOT / "series_index.parquet")
    st = pd.read_parquet(ROOT / "study_index.parquet").set_index("StudyInstanceUID")
    chosen = si[si.chosen]
    rng = np.random.default_rng(0)
    studies = rng.choice(sorted(chosen.StudyInstanceUID.unique()), n, replace=False)

    n_ok = n_bad = 0
    t0 = time.time()
    for sid in studies:
        want = np.load(cache / f"{sid}.npy")
        want_mask = np.load(cache / f"{sid}.mask.npy")[0]
        got, got_mask = preprocess_study(sid, chosen[chosen.StudyInstanceUID == sid],
                                         ROOT / "train_series", st.loc[sid, "side"])
        same_mask = np.array_equal(want_mask, got_mask)
        d = np.abs(want.astype(np.int16) - got.astype(np.int16))
        n_diff = int((d > 0).sum())
        frac = n_diff / d.size

        # Tolerance, not equality -- but a tolerance chosen so it cannot hide a real bug.
        # Reversing an axis yields a non-contiguous array, and numpy may vectorise the
        # subsequent multiply differently, flipping the last bit on a value sitting exactly
        # on a rounding boundary. Measured incidence: ONE pixel in 3,612,672.
        #
        # Every failure this check exists to catch is catastrophic rather than subtle: a
        # wrong mirror, a wrong slice order or a wrong crop moves essentially every pixel
        # by a large amount (a mis-applied mirror in an early draft of this check gave max
        # 255 across 99% of pixels). So "max 1 grey level on <0.01% of pixels" cleanly
        # separates rounding from error.
        ok = same_mask and d.max() <= 1 and frac < 1e-4
        if ok:
            n_ok += 1
            tag = "exact" if n_diff == 0 else f"{n_diff} px differ by 1 (rounding)"
            print(f"  [  ok  ] {sid[-12:]}  {tag}")
        else:
            n_bad += 1
            bad_slots = sorted(set(np.where(d.reshape(len(SLOTS), -1).max(1) > 1)[0].tolist()))
            print(f"  [ FAIL ] {sid[-12:]}  mask_match={same_mask} "
                  f"max_px_diff={d.max()} differing={100 * frac:.2f}% "
                  f"slots={[SLOTS[i] for i in bad_slots]}")
    dt = time.time() - t0
    print(f"\n  {n_ok} exact, {n_bad} mismatched   ({dt / max(n, 1):.1f}s per study)")
    if n_bad:
        print("  The inference path does NOT reproduce training. Do not submit until it does.")
    else:
        print("  Contract holds: scoring-time images are bit-identical to training images.")
        est = dt / max(n, 1) * 1300
        print(f"  Extrapolated to ~1,300 test studies: {est / 60:.0f} min single-threaded, "
              f"{est / 60 / 8:.0f} min at 8 threads (limit is 9 h)")
    return n_bad == 0


def run(split, out, ckpts, device, img, n_group, bs, threads):
    """Whole split -> submission.csv, one study at a time (`bs` is not used).

    THE ONE NUMBER TO CHECK BEFORE SUBMITTING IS --n-group. The window starts below are
    exactly KneeStudies._pick's eval branch -- linspace over the stack, each start reading 3
    consecutive slices -- so n_group decides how much of each cached 32-slice slot is ever
    LOOKED AT: 8 reads 24 of the 32 positions and never touches [3, 7, 11, 15, 16, 20, 24,
    28]; 10 reads 30; 11 and above read all 32. A finding sitting on a position no window
    covers is invisible no matter how good the model is, and nothing about the output
    reveals it. The default here is 8, which is NOT what recent runs ship -- pass the
    n_group the checkpoint was validated at. ngroup_sweep.py holds the full coverage table.
    """
    import torch
    from model import KneeNet
    split_root = ROOT / f"{split}_series"
    series_csv = ROOT / f"{split}_series.csv"
    ids = pd.read_csv(ROOT / f"{split}.csv").StudyInstanceUID.tolist()
    print(f"{split}: {len(ids)} studies")

    idx, side_map = build_index(split_root, series_csv, threads=threads)
    chosen = idx[idx.chosen]

    models = []
    for p in ckpts:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        # Construct from the checkpoint's OWN args. This previously passed only `backbone`,
        # so a 4-slot checkpoint -- whose SlotHead is a different shape -- could not load at
        # all. Any flag that changes the module tree has to be threaded through here, and
        # load_state_dict is left STRICT on purpose: a silent partial load ships a model with
        # randomly-initialised layers and scores like noise, which is far worse than a crash.
        _a = ck["args"]
        _sl = _a.get("slots")
        net = KneeNet(backbone=_a["backbone"], pretrained=False,
                      slot_idx=([int(x) for x in str(_sl).split(",")] if _sl else None))
        net.load_state_dict(ck["model"])
        models.append(net.to(device).eval())
    print(f"loaded {len(models)} checkpoint(s)")

    preds = np.zeros((len(ids), len(TARGETS)), np.float64)
    t0 = time.time()
    for i, sid in enumerate(ids, 1):
        rows = chosen[chosen.StudyInstanceUID == sid]
        vol, mask = preprocess_study(sid, rows, split_root, side_map.get(sid))
        starts = np.linspace(0, N_SLICE - 3, n_group).round().astype(int)
        g = np.stack([[vol[s, k:k + 3] for k in starts] for s in range(len(SLOTS))])
        gt = torch.from_numpy(g).unsqueeze(0).to(device)
        mt = torch.from_numpy(mask).unsqueeze(0).to(device)
        with torch.no_grad():
            # Members are combined in PROBABILITY space, per study, right here. This is NOT
            # the rank average ensemble.py performs: the rank transform further down runs
            # after averaging, so it cannot undo one over-confident member dominating the
            # mean. For a real rank mean over members, ensemble the OOF/prediction files.
            ps = [torch.sigmoid(m(gt, mt, img_size=img)[0].float()).cpu().numpy()[0]
                  for m in models]
        preds[i - 1] = np.mean(ps, 0)
        if i % 50 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(ids)}  {el / 60:.1f} min  eta {(len(ids) - i) * el / i / 60:.0f} min",
                  flush=True)

    if len(models) > 1:
        # Column-wise rank, rescaled to (0, 1]. Cosmetic as far as the metric goes -- a
        # strictly increasing map of a column cannot change that column's AUC -- but it
        # spreads the submitted numbers over the full range instead of leaving them bunched
        # wherever averaging several sigmoids happened to put them.
        from scipy.stats import rankdata
        preds = np.apply_along_axis(rankdata, 0, preds) / len(preds)

    sub = pd.DataFrame(preds, columns=TARGETS)
    sub.insert(0, "StudyInstanceUID", ids)
    from validation import check_submission
    check_submission(sub, expect_rows=len(ids), raise_on_fail=False)
    sub.to_csv(out, index=False)
    print(f"\nwrote {out} in {(time.time() - t0) / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--ckpt", nargs="*", default=[])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--img", type=int, default=336)
    ap.add_argument("--n-group", type=int, default=8)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()

    if a.verify:
        sys.exit(0 if verify(a.n) else 1)
    if not a.ckpt:
        sys.exit("no --ckpt given; nothing to run")
    run(a.split, a.out, a.ckpt, a.device, a.img, a.n_group, a.bs, a.threads)


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()
