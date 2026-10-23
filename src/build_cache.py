"""
build_cache.py -- step B4: decode every (study, slot) once into uint8 and never read DICOM again.

Fine-tuning revisits the same pixels every epoch. Reading them off the mount each time makes
the epoch count a function of I/O rather than of learning, so the pixels are decoded once and
held as bytes. Intensity is already normalised into [0,1] before quantising, and eight bits
cost nothing a bilinear resize has not already cost.

Four decisions are baked in here, and every one of them changes what a slice IS. A model
trained under one and run under another loads cleanly, produces a submission, and scores
nonsense -- so they travel with the cache in meta.json:

  ORDER      slices come from slice_order.parquet (geometry), never from filename order,
             which correlates -0.000 with anatomy.
  SCALE      crop a constant 130 mm of anatomy, then resample to 336 px. A fixed-pixel
             resize would hand the encoder images whose physical scale differs several-fold.
             130 mm sits below the field of view of 99.6% of series while still containing
             the joint; 336 px puts the pitch at 0.387 mm, under the d/2 bound for a 1 mm
             meniscal tear. At 224 px the pitch is 0.580 mm and the tear is destroyed before
             the first convolution.
  INTENSITY  per-series 1st/99th percentile over the sampled stack -- not per slice, so
             slices keep their relative contrast, and percentiles not extremes, so one
             bright vessel does not compress everything else. MR has no absolute scale.
  SIDE       left knees are mirrored to a canonical side. Coronally and axially the
             medial-lateral axis lies in the image plane, so the last axis is flipped;
             sagittally it is the SLICE axis, so the stack order is reversed instead.

Resumable: one .npy per study, skipped if already written. Safe to kill and restart.

Usage:
    python build_cache.py --limit 20 --out <dir>     # smoke test
    python build_cache.py                            # full build (hours)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image

ROOT = Path(__file__).resolve().parent
SLOTS = ["SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS", "SAG_FLUID_NOFS", "COR_T1", "SAG_T1"]

CROP_MM = 130.0
IMG = 336
# Every slot is stored at exactly N_SLICE frames, so one study is one rectangular array and
# a batch never has to deal with ragged depths. 32 is more than most series actually hold
# (62.6% of the chosen series are shorter), so most slots carry interleaved repeats -- see
# sample_indices -- and row 1 of the mask file records how many frames were really acquired.
N_SLICE = 32


def sample_indices(n_avail, n_want):
    """Evenly spaced positions through the stack, inclusive of both ends.

    The same linspace serves both directions, and that is deliberate. 62.6% of the chosen
    series hold fewer than N_SLICE slices, so the short case is the common one -- and the
    obvious handling of it, taking every slice then padding the tail, puts a solid block of
    identical frames at the end of the volume. A 16-slice series became 16 real slices
    followed by 16 copies of the last one. Attention pooling over that bag sees a sixteenth
    of the anatomy repeated sixteen times and can put most of its weight there.

    Spreading the repeats evenly instead means every output position still maps to a real
    acquired slice, duplicates are interleaved rather than clumped, and no single frame can
    dominate. `n_valid` records the true count so the model can down-weight the difference.

    Note how that bug was found: not by an assertion -- the volumes had the right shape, the
    right dtype and a plausible intensity range, and training ran fine -- but by rendering
    cached volumes to PNG and looking at them. That is why test_cache.py keeps a --render
    mode alongside its numeric checks.
    """
    if n_avail <= 0:
        return []
    return list(np.linspace(0, n_avail - 1, n_want).round().astype(int))


def load_slot(sdir: Path, sops, pix_mm, want=N_SLICE):
    """Ordered file list -> ((want, IMG, IMG) float32 in [0,1], n_real), or None.

    `sops` must already be in anatomical order (build_slice_order.py); `pix_mm` is the
    row spacing in mm, which also sizes the columns because 99.8% of the chosen series have
    square pixels.

    The order of the three steps below is load-bearing: crop to a constant physical extent
    FIRST, then take intensity percentiles over what survives, then resample. Percentiles
    taken before the crop would be set by whatever surrounds the joint -- background air, a
    coil, soft tissue outside the field of interest -- so the same knee would land at a
    different brightness depending only on how wide the scan was.
    """
    idx = sample_indices(len(sops), want)
    if not idx:
        return None
    arrs = []
    # 62.6% of the chosen series hold fewer than `want` slices, so sample_indices
    # deliberately emits REPEATED positions (see its docstring). Decoding the same file once
    # per repeat is 12.2% of every pixel decode across the 24,371 indexed series, for a
    # bit-for-bit identical result. Memoize by position instead: same file, same pixels.
    # Sharing the array object between repeats is safe because np.stack below copies and
    # nothing mutates a decoded frame in place -- `filler` is already shared this way.
    seen = {}
    for j in idx:
        if j in seen:
            arrs.append(seen[j])
            continue
        try:
            ds = pydicom.dcmread(str(sdir / sops[j]), force=True)
            a = ds.pixel_array.astype(np.float32)
            # MONOCHROME1 stores an inverted image: the HIGHEST stored value is black.
            # Left alone, such a series arrives as a photographic negative of every other
            # series -- fluid dark instead of bright -- which is exactly the contrast the
            # fluid-sensitive slots are chosen for.
            if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                a = a.max() - a
        except Exception:
            # A decode failure must not be silently indistinguishable from a black knee.
            seen[j] = None
            arrs.append(None)
            continue
        seen[j] = a
        arrs.append(a)

    # A frame that failed to decode, or that disagrees with the rest of the series about
    # its shape, becomes black. The shape reference comes from a frame that DID decode, so
    # one bad file cannot resize the slot. If nothing decoded, the slot is refused outright
    # rather than written as an all-black volume that the mask would still call present.
    good = [a for a in arrs if a is not None]
    if not good:
        return None
    h, w = good[0].shape[:2]
    filler = np.zeros((h, w), np.float32)
    arrs = [a if (a is not None and a.shape[:2] == (h, w)) else filler for a in arrs]
    vol = np.stack(arrs)                                     # (n, H, W)

    # Crop a constant physical extent about the image centre. THE EXCEPTION: when the
    # requested 130 mm is as wide as the image itself the crop is skipped, not padded, and
    # that series is resampled at its native extent -- so it alone sits at a different
    # mm/px. Measured on the chosen series: 342 of 19,616 (1.7%), all with a field of view
    # already at or below 130 mm (smallest 96 mm), so the mismatch is bounded.
    if pix_mm and pix_mm > 0:
        n_pix = int(round(CROP_MM / pix_mm))
        if 16 < n_pix < min(h, w):
            r0, c0 = (h - n_pix) // 2, (w - n_pix) // 2
            vol = vol[:, r0:r0 + n_pix, c0:c0 + n_pix]

    lo, hi = np.percentile(vol, [1, 99])
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)

    out = np.empty((len(vol), IMG, IMG), np.float32)
    for i, s in enumerate(vol):
        out[i] = np.asarray(Image.fromarray(s).resize((IMG, IMG), Image.BILINEAR))

    return out, len(sops)


def build_study(job):
    study, rows, side, outdir, order_map = job
    dst = outdir / f"{study}.npy"
    if dst.exists():
        return "skip", study, 0.0
    t0 = time.time()
    vol = np.zeros((len(SLOTS), N_SLICE, IMG, IMG), np.uint8)
    mask = np.zeros(len(SLOTS), np.float32)
    n_valid = np.zeros(len(SLOTS), np.int16)   # real slices behind each slot

    for r in rows.itertuples():
        k = SLOTS.index(r.slot)
        sdir = ROOT / "train_series" / study / r.SeriesInstanceUID
        sops = order_map.get(r.SeriesInstanceUID)
        if not sops:
            continue
        res = load_slot(sdir, sops, r.pix_row)
        if res is None:
            continue
        a, n_real = res
        # Canonical side: mirror left knees. `side` is None where laterality could not be
        # established (build_series_index leaves the midline deadband unresolved); those
        # studies are left as acquired, because mirroring the wrong way is worse than not
        # mirroring -- it swaps medial for lateral on five of the twelve targets.
        if side == "L":
            if r.slot.startswith("SAG"):
                a = a[::-1]              # sagittal: medial-lateral is the slice axis
            else:
                a = a[:, :, ::-1]        # coronal / axial: it lies in the image plane
        vol[k] = (np.clip(a, 0, 1) * 255).astype(np.uint8)
        mask[k] = 1.0
        n_valid[k] = min(n_real, N_SLICE)

    if mask.sum() == 0:
        return "empty", study, time.time() - t0
    np.save(dst, vol)
    np.save(outdir / f"{study}.mask.npy", np.stack([mask, n_valid.astype(np.float32)]))
    return "ok", study, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"data/rsna_knee_cache")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)

    si = pd.read_parquet(ROOT / "series_index.parquet")
    st = pd.read_parquet(ROOT / "study_index.parquet")
    so = pd.read_parquet(ROOT / "slice_order.parquet")

    chosen = si[si.chosen].copy()
    order_map = (so.sort_values("order").groupby("SeriesInstanceUID").sop
                 .apply(list).to_dict())
    side_map = dict(zip(st.StudyInstanceUID, st.side))

    studies = sorted(chosen.StudyInstanceUID.unique())
    if a.limit:
        studies = studies[:a.limit]
    by_study = {s: g for s, g in chosen.groupby("StudyInstanceUID")}

    (outdir / "meta.json").write_text(json.dumps(dict(
        slots=SLOTS, crop_mm=CROP_MM, img=IMG, n_slice=N_SLICE,
        order="geometry (slice_order.parquet)", intensity="per-series 1-99 percentile",
        side="left knees mirrored; SAG reverses slice order, COR/AX flips last axis",
        sampling="linspace over the real stack in both directions; repeats interleaved, never a tail block",
        mask_file="row 0 = slot present (0/1); row 1 = real slice count behind that slot",
        dtype="uint8", shape=[len(SLOTS), N_SLICE, IMG, IMG],
    ), indent=2), encoding="utf-8")

    jobs = [(s, by_study[s], side_map.get(s), outdir, order_map) for s in studies]
    gb = len(jobs) * len(SLOTS) * N_SLICE * IMG * IMG / 1e9
    print(f"building cache for {len(jobs)} studies -> {outdir}")
    print(f"  {len(SLOTS)} slots x {N_SLICE} slices x {IMG}px  =  ~{gb:.1f} GB")
    print(f"  {a.threads} threads\n", flush=True)

    t0 = time.time()
    counts = {"ok": 0, "skip": 0, "empty": 0}
    with ThreadPoolExecutor(max_workers=a.threads) as ex:
        for i, (status, study, dt) in enumerate(ex.map(build_study, jobs), 1):
            counts[status] += 1
            if i % 25 == 0 or i == len(jobs):
                el = time.time() - t0
                rate = i / el
                print(f"  {i}/{len(jobs)}  {counts}  {el / 60:.1f} min elapsed, "
                      f"eta {(len(jobs) - i) / max(rate, 1e-9) / 60:.0f} min", flush=True)

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min: {counts}")
    if counts["empty"]:
        print(f"  WARNING: {counts['empty']} studies produced no usable slot")


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()
