"""
build_cache_f2.py -- the SECOND-FAMILY input representation. A new cache, not a new model.

WHY A SECOND CACHE AND NOT A SECOND BACKBONE ON THE OLD ONE
We have run that experiment. XCiT was pointed at data/rsna_knee_cache -- the
same 6 slots, the same 32 slices, the same 130 mm crop, the same 336 px grid -- and it
added nothing to the ensemble, then reversed on the leaderboard. Two backbones that read
the SAME pixels through the SAME sampling grid make the same mistakes about the same
studies, because most of what a model gets wrong here is decided before the first
convolution: which series was chosen, which slices of it were kept, how much anatomy the
crop left in, and how the intensities were mapped. A second family has to differ THERE.

So this file is deliberately a different answer to every one of those four questions, and
`--self-test` fails if any one of them silently reverts to the v1 answer. The differences,
and what each is expected to decorrelate:

  RACK       5 slots, not 6, and each is a PLANE+ROLE with a FALLBACK LADDER rather than an
             exact contrast. v1's SAG_FLUID_FS is empty unless the study holds a sagittal
             fluid-sensitive FAT-SUPPRESSED series; f2's SAG_FLUID takes that, else the
             non-fat-sat sagittal fluid series, else sagittal T1. Measured on
             series_index_v2.parquet, per-slot study fill goes
                 SAG  94.3% -> 99.9%    COR  96.5% -> 100.0%    AX  93.4% -> 100.0%
                 2nd sagittal 62.6% -> 96.7%   2nd coronal 63.6% -> 74.5%
             4.71 of 5 slots filled against v1's 4.451 of 6. The two models therefore
             disagree most on exactly the studies v1 reads thinnest, which is where the
             ensemble has something to gain.

  BUDGET     UNEVEN slices per slot -- 18/14/12/8/12 = 64 -- not 32 everywhere. v1 spends
             the same 32 frames on the sagittal fluid series (median 30 real slices, so
             repeats) and on the second coronal (median 30, same). The uneven budget spends
             them where the findings are: 18 on the sagittal fluid stack that carries the
             cruciates and both meniscal bodies, 8 on the second coronal which is a
             corroborating view. Same total pixels per study as 2 of v1's slots, spread
             differently.
             --budget dense (26/24/26/16/28 = 120) is the same rack at about native pitch:
             64 frames keep only 38.9% of the chosen slices, so COR_FLUID frames sit 8.1 mm
             apart against a native 3.5 mm. Dense brings the median frame pitch to
             3.5/3.8/3.6/5.9/4.0 mm per slot. f2 remains the default.

  BAND       slices are drawn from 6%-94% of the stack, not 0%-100%. The first and last few
             slices of a knee series are at or outside the joint margin -- capsule, skin,
             coil -- and v1's linspace is pinned to both ENDPOINTS, so two of its 32 frames
             are guaranteed to be the two least informative slices in the series. Trimming
             the band is the single cheapest change here and it changes every slot.

  SCALE      140 mm across 384 px = 0.3646 mm/px, against v1's 130 mm across 336 px =
             0.3869 mm/px. The extra 10 mm is not a resolution claim -- see the measurements
             in build_cache_hires.py, which establish that this source is already
             oversampled by ~2x and that more pixels buy interpolation, not detail. It is an
             ANATOMY claim: 140 mm keeps the posterior capsule and the popliteal fossa
             inside the frame on series where 130 mm clips them, and Baker's cyst is by
             definition posterior. 384 px is not a choice at all -- see IMG below.

  SCALE IS   v1 SKIPS the crop when the requested extent is as wide as the image, so those
  CONSTANT   series alone sit at a different mm/px. At 130 mm that is 362 of 21,751 chosen
             series (1.66%) and the mismatch is bounded. At 140 mm it is 1,478 (6.80%), and
             1-in-15 is no longer a rounding error. f2 PADS instead: the crop window is
             taken as requested, whatever falls outside the image is zero, and the
             percentiles are computed over the VALID region only so the pad cannot darken
             the knee. Every series in this cache is at 0.3646 mm/px, with no exceptions
             clause for the loader to not know about.

  INTENSITY  per-series 0.5/99.5 percentile over the DISTINCT decoded slices. Two changes
             from v1's 1/99 over the sampled stack, both deliberate:
               * DISTINCT. When a series is shorter than its budget the sampler emits
                 repeated positions, and v1 lets a repeated frame vote once per repeat. A
                 16-slice series in a 32-slice slot has every frame counted twice -- which
                 is harmless -- but a 9-slice series has some frames counted four times and
                 others three, so the percentile is set by an accident of the rounding.
               * 0.5/99.5, not 1/99. The top percentile of a fluid-sensitive knee IS the
                 effusion, the cyst and the marrow oedema. Clipping it flat is clipping the
                 finding. A wider window keeps more of that tail; it costs contrast
                 elsewhere, which is the trade a second family should be making differently.

  IMG = 384  This one is NOT free. `coatnet_rmlp_2_rw_384` carries
             fixed_input_size=True: its relative-position-bias tables are built for a 12x12
             final grid and nothing else. Verified on CPU -- 64 px and 336 px both raise
             `RuntimeError: The size of tensor a (441) must match the size of tensor b
             (576)`, and only 384 px returns (1, 1024, 12, 12). A cache at any other width
             cannot feed this backbone at all, so 384 is the backbone's requirement, not our
             preference.

DELIBERATELY IDENTICAL TO v1, and each for a reason:
  ORDER      slice_order.parquet. Geometry is geometry; filename order correlates -0.000
             with anatomy in BOTH representations. Changing this would be a bug, not a
             difference.
  SIDE       left knees mirrored to a canonical side; sagittal reverses the slice axis,
             coronal/axial flips the last axis; studies whose laterality is unresolved (40
             of 4,407) are left as acquired. Un-mirroring would make the second model learn
             both chiralities from half the data -- worse, not decorrelated.
  RESIZE     PIL BILINEAR. build_cache_hires.py measured this specific resampler's stopband
             on this specific source; swapping in cv2 or torch would change the frequency
             content of every slice in a way nothing in this repo has measured.
  DTYPE      uint8 after normalisation into [0,1].

ON-DISK CONTRACT -- read this before writing a loader
  <study>.npy       (N_TOTAL, 384, 384) uint8: 64 under --budget f2, 120 under dense; the
                    cache's meta.json records which. The f2 reasoning below holds for both.
                    FLAT, not (5, 18, 384, 384). The budgets are
                    uneven, so a rectangular slot axis would have to pad every slot out to
                    18 and cost 90 frames per study instead of 64: 58 GB rather than 42 GB,
                    which does not fit on this disk. The flat layout is forced by the
                    uneven budget, and it is the one thing here a v1-shaped loader cannot
                    consume -- see pending_patches/F2-coatnet-384-input.md.
  <study>.meta.npy  (4, 5) float32. NOT called .mask.npy: a v1 loader globbing *.mask.npy
                    must not be able to pick up an f2 sidecar and read a (2,6) contract out
                    of a (4,5) file.
                      row 0  present   1.0 if this slot found a series and decoded it
                      row 1  start     first row of this slot inside the N_TOTAL
                      row 2  budget    how many rows it owns
                      row 3  n_distinct  how many DISTINCT real slices are behind those
                                       rows. Equals budget when the series was long enough;
                                       below it when the band had to repeat frames, which
                                       is the number a loader needs to down-weight a slot
                                       that is mostly duplicates.
                    An absent slot's rows in the volume are all zero AND row 0 is 0.0. Both,
                    so a loader that checks either one is right.

Never writes into data/rsna_knee_cache. --out must be a different path.

Usage:
    python build_cache_f2.py --self-test
    python build_cache_f2.py --self-test --neutralise band        # must FAIL
    python build_cache_f2.py --plan                               # fill rates + disk budget
    python build_cache_f2.py --limit 50 --out data/rsna_knee_cache_f2_smoke
    python build_cache_f2.py --out data/rsna_knee_cache_f2
    python build_cache_f2.py --self-test --budget dense
    python build_cache_f2.py --budget dense --threads 8 --out data/rsna_knee_cache_f2_dense
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- the contract, in one place
SLOTS = ["SAG_FLUID", "SAG_ALT", "COR_FLUID", "COR_ALT", "AX_FLUID"]
# Named per-slot frame budgets, in SLOTS order; --budget picks one. "f2" is the layout the
# original cache was built with and stays the default. "dense" brings the frame pitch back to
# about native (median in-band native count is 26/26/26/26/28): f2 keeps 38.9% of the
# chosen slices and dense 68.1%. COR_ALT stays at roughly half density on purpose, as the
# corroborating view. See TRACKER 2026-10-06.
BUDGETS = {
    "f2": (18, 14, 12, 8, 12),        # 64
    "dense": (26, 24, 26, 16, 28),    # 120
}
BUDGET = BUDGETS["f2"]
OFFSET = tuple(int(x) for x in np.concatenate([[0], np.cumsum(BUDGET)[:-1]]))
N_TOTAL = int(sum(BUDGET))            # 64
CROP_MM = 140.0
IMG = 384
BAND = (0.06, 0.94)
PCTL = (0.5, 99.5)
V1_CACHE = Path(r"data/rsna_knee_cache")


def use_budget(name):
    """Rebind BUDGET / OFFSET / N_TOTAL to a named layout. build_study, plan and the
    self-test read these module globals at call time, so call this before any of them."""
    global BUDGET, OFFSET, N_TOTAL
    BUDGET = BUDGETS[name]
    OFFSET = tuple(int(x) for x in np.concatenate([[0], np.cumsum(BUDGET)[:-1]]))
    N_TOTAL = int(sum(BUDGET))


# The fallback ladder. Each entry is (plane, kind); the first tier that has a series this
# study has not already spent wins. Tier order within a slot is a statement about which
# contrast answers that slot's question BEST, not about which is most common.
#
# SAG_ALT's job is to be a DIFFERENT sagittal image from SAG_FLUID, so it prefers the
# non-fat-sat fluid series (the sequence radiologists read menisci on) and falls back to
# T1, then to non-fat-sat gradient echo. COR_ALT prefers T1 because the coronal T1 is the
# marrow/osteophyte view that COR_FLUID's fat-suppressed image is worst at.
LADDER = {
    "SAG_FLUID": [("Sagittal", "fluid_fs"), ("Sagittal", "fluid_nofs"), ("Sagittal", "t1")],
    "SAG_ALT":   [("Sagittal", "fluid_nofs"), ("Sagittal", "t1"), ("Sagittal", "gre")],
    "COR_FLUID": [("Coronal", "fluid_fs"), ("Coronal", "fluid_nofs"), ("Coronal", "t1")],
    "COR_ALT":   [("Coronal", "t1"), ("Coronal", "fluid_nofs"), ("Coronal", "fluid_fs")],
    "AX_FLUID":  [("Axial", "fluid_fs"), ("Axial", "fluid_nofs"), ("Axial", "t1")],
}


def _kind_mask(df, kind):
    """Which rows of a series-index table are of this contrast kind.

    Reads fluid_v2 / fatsat_v2, the columns build_series_index_v2.py derives -- the union of
    the shipped Fluid_Sensitive / Fat_Suppression flags with the de-negated protocol-name
    regex. f2 does NOT re-derive them: that classifier was measured against 3,003 orphan
    series and is correct, and re-deriving it here would mean two different answers to
    "is this fat-suppressed" in one repo.
    """
    fl = df.fluid_v2.astype(bool)
    fs = df.fatsat_v2.astype(bool)
    if kind == "fluid_fs":
        return fl & fs
    if kind == "fluid_nofs":
        return fl & ~fs
    if kind == "t1":
        return df.weighting.eq("T1")
    if kind == "gre":
        return df.weighting.eq("GRE")
    raise ValueError(kind)


def select_slots(df, ladder=None, neutralise=None):
    """Series index -> one row per (study, f2 slot) that found a series.

    EXCLUSIVE. A series spent on SAG_FLUID is not available to SAG_ALT, so a study holding
    exactly one sagittal series fills one sagittal slot and leaves the other honestly empty
    rather than showing the same pixels twice under two names and letting L3 attend to its
    own echo.

    DETERMINISTIC. Candidates inside a tier are ranked (most slices, finest in-plane
    spacing, thinnest slice, UID) -- the same quality order build_series_index_v2.choose
    settled on, for the same reason: pandas' method="first" resolves ties by row order,
    which is UID order, which is a hash.
    """
    ladder = ladder or LADDER
    if neutralise == "ladder":
        # No fallback: each slot takes its first-choice contrast or stays empty. This is the
        # "new rack that is the old rack with different labels" failure mode.
        ladder = {k: v[:1] for k, v in ladder.items()}

    d = df.copy()
    d["_px"] = pd.to_numeric(d.pix_row, errors="coerce").fillna(np.inf)
    d["_th"] = pd.to_numeric(d.slice_thickness, errors="coerce").fillna(np.inf)
    d["_n"] = pd.to_numeric(d.n_slices, errors="coerce").fillna(0)
    if neutralise == "nondet":
        # Rank on slice count alone and let row order break the tie -- v1's bug, reproduced
        # exactly so the determinism check has something real to catch.
        d = d.sort_values(["_n"], ascending=False, kind="mergesort")
    else:
        d = d.sort_values(["_n", "_px", "_th", "SeriesInstanceUID"],
                          ascending=[False, True, True, True], kind="mergesort")

    kinds = {k: _kind_mask(d, k) for k in ("fluid_fs", "fluid_nofs", "t1", "gre")}
    plane = d.Anatomical_Plane

    rows = []
    for sid, g in d.groupby("StudyInstanceUID", sort=True):
        spent = set()
        for slot in SLOTS:
            for tier, (pl, kind) in enumerate(ladder[slot]):
                m = (plane.loc[g.index] == pl) & kinds[kind].loc[g.index]
                cand = g[m.values & ~g.SeriesInstanceUID.isin(spent).values]
                if len(cand):
                    r = cand.iloc[0]
                    spent.add(r.SeriesInstanceUID)
                    rows.append(dict(StudyInstanceUID=sid, slot=slot,
                                     SeriesInstanceUID=r.SeriesInstanceUID,
                                     tier=tier, pix_row=r.pix_row, n_slices=int(r._n)))
                    break
    return pd.DataFrame(rows, columns=["StudyInstanceUID", "slot", "SeriesInstanceUID",
                                       "tier", "pix_row", "n_slices"])


def band_indices(n_avail, n_want, band=BAND, neutralise=None):
    """Positions in [0, n_avail) spanning `band` of the stack, evenly spaced.

    v1 uses linspace(0, n_avail-1, n_want): pinned to BOTH endpoints, so slice 0 and slice
    n-1 are in every volume it writes. In a knee series those are the two slices least
    likely to contain the joint -- lateral to the lateral condyle, or medial to the medial
    one, on a sagittal stack. Spanning 6%-94% instead costs two frames of coverage at the
    margins and spends them inside the joint.

    Short series behave exactly as v1's do: the same linspace emits repeated positions, and
    they come out INTERLEAVED rather than as a block at the tail, so no single frame can
    dominate an attention pool. n_distinct records how many real slices are actually behind
    the budget -- see build_study.
    """
    if n_avail <= 0:
        return []
    lo, hi = (0.0, 1.0) if neutralise == "band" else band
    return list((np.linspace(lo, hi, n_want) * (n_avail - 1)).round().astype(int))


def _norm(x, lo, hi):
    """(x - lo) / (hi - lo), clipped to [0, 1], in FLOAT32 under any NumPy.

    `lo`/`hi` come from np.percentile as np.float64. NumPy 1.x (which built every cache on
    this machine, 1.23.5) casts such a scalar to the array's float32 and stays there; NumPy 2
    (NEP 50) promotes float32 - np.float64 to float64. Pillow then rounds the float64 slice
    back to float32 before resizing, and the truncating uint8 cast can land a pixel one grey
    level off. Measured under NumPy 2.4.6 on the 60-study `inference_f2.py --verify`: 59/60
    byte-identical, one study with 4 pixels off by 1; with this cast, 60/60. Casting the
    scalars explicitly is the float32 path NumPy 1.x took, so the bytes no longer depend on
    which NumPy the Kaggle image ships.
    """
    return np.clip((x - np.float32(lo)) / np.float32(max(hi - lo, 1e-6)), 0, 1)


def _crop_pad(vol, pix_mm, crop_mm, uniq_rows, neutralise=None):
    """Take a constant physical extent about the image centre; pad where the image runs out.

    Returns (normalised volume, n_pix, padded?). The ORDER is load-bearing and is v1's, with
    one addition: restrict to the crop window, take the percentiles over the VALID part of
    that window ONLY, normalise, and only then place the result on a zero canvas. Taking
    them before the crop lets whatever surrounds the joint set the brightness; taking them
    after the pad lets the pad do it.
    """
    n, h, w = vol.shape
    if not (pix_mm and pix_mm > 0):
        lo, hi = np.percentile(vol[uniq_rows], list(PCTL))
        return _norm(vol, lo, hi), min(h, w), False
    n_pix = int(round(crop_mm / pix_mm))
    if n_pix <= 16:
        lo, hi = np.percentile(vol[uniq_rows], list(PCTL))
        return _norm(vol, lo, hi), min(h, w), False

    r0, c0 = (h - n_pix) // 2, (w - n_pix) // 2
    rs, re = max(r0, 0), min(r0 + n_pix, h)
    cs, ce = max(c0, 0), min(c0 + n_pix, w)
    padded = (rs != r0) or (cs != c0) or (re != r0 + n_pix) or (ce != c0 + n_pix)

    if neutralise == "nopad" and padded:
        # v1 behaviour: when the requested extent does not fit, skip the crop entirely and
        # let this series sit at its own mm/px. 6.80% of chosen series at 140 mm.
        lo, hi = np.percentile(vol[uniq_rows], list(PCTL))
        return _norm(vol, lo, hi), min(h, w), False

    sub = vol[:, rs:re, cs:ce]
    lo, hi = np.percentile(sub[uniq_rows], list(PCTL))
    sub = _norm(sub, lo, hi)
    if not padded:
        return sub, n_pix, False
    out = np.zeros((n, n_pix, n_pix), np.float32)
    out[:, rs - r0:rs - r0 + (re - rs), cs - c0:cs - c0 + (ce - cs)] = sub
    return out, n_pix, True


def load_slot(sdir: Path, sops, pix_mm, want, crop_mm=CROP_MM, img=IMG,
              neutralise=None):
    """Ordered file list -> ((want, img, img) float32 in [0,1], n_distinct), or None.

    `sops` must already be in anatomical order (build_slice_order.py).
    """
    idx = band_indices(len(sops), want, neutralise=neutralise)
    if not idx:
        return None

    # Decode each distinct position once. The band emits repeats on short series and
    # re-reading the same file for a bit-identical result is pure I/O.
    seen, arrs = {}, []
    for j in idx:
        if j in seen:
            arrs.append(seen[j])
            continue
        try:
            ds = pydicom.dcmread(str(sdir / sops[j]), force=True)
            a = ds.pixel_array.astype(np.float32)
            # MONOCHROME1 stores an inverted image: the highest stored value is black. Left
            # alone, such a series arrives as a negative -- fluid dark instead of bright --
            # which is exactly the contrast three of these five slots are chosen for.
            if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                a = a.max() - a
        except Exception:
            a = None
        seen[j] = a
        arrs.append(a)

    good = [a for a in arrs if a is not None]
    if not good:
        return None
    h, w = good[0].shape[:2]
    filler = np.zeros((h, w), np.float32)
    arrs = [a if (a is not None and a.shape[:2] == (h, w)) else filler for a in arrs]
    vol = np.stack(arrs)

    # Rows of `vol` holding the FIRST occurrence of each distinct source slice. These are
    # the only rows the percentiles may see: a frame the band repeated four times must not
    # outvote one it took once.
    first = {}
    for row, j in enumerate(idx):
        first.setdefault(j, row)
    uniq_rows = sorted(first.values())

    vol, _, _ = _crop_pad(vol, pix_mm, crop_mm, uniq_rows, neutralise=neutralise)

    out = np.empty((len(vol), img, img), np.float32)
    for i, s in enumerate(vol):
        out[i] = np.asarray(Image.fromarray(s).resize((img, img), Image.BILINEAR))
    return out, len(first)


def assemble_study(study, rows, side, order_map, series_root, budget=None, neutralise=None):
    """One study -> ((n_total, IMG, IMG) uint8 volume, (4, 5) float32 sidecar), in memory.

    THE builder. build_study below writes its result into the cache, and the Kaggle notebook
    (inference_f2.preprocess_study_f2) calls it on test DICOMs, so the two cannot differ.
    `rows` are select_slots rows for this study, `order_map` maps SeriesInstanceUID to its
    files in anatomical order, `side` is "L"/"R"/None. `budget` is the per-slot frame layout,
    default the module's (set by use_budget); passing it keeps the notebook off the module
    globals.
    """
    budget = tuple(BUDGET if budget is None else budget)
    n_total = int(sum(budget))
    vol = np.zeros((n_total, IMG, IMG), np.uint8)
    meta = np.zeros((4, len(SLOTS)), np.float32)
    meta[1] = tuple(int(x) for x in np.concatenate([[0], np.cumsum(budget)[:-1]]))
    meta[2] = budget
    crop_mm = 130.0 if neutralise == "crop" else CROP_MM
    if neutralise == "budget":
        # Even budgets summing to the same N_TOTAL (13/13/13/13/12 for f2, 24 x 5 for
        # dense). The rack looks identical on disk and every shape assertion still passes
        # -- which is the point: only a test that reads how many DISTINCT slices a long
        # series contributed can tell these apart.
        q, r = divmod(n_total, len(SLOTS))
        budget = tuple(q + (i < r) for i in range(len(SLOTS)))
        meta[1] = tuple(int(x) for x in np.concatenate([[0], np.cumsum(budget)[:-1]]))
        meta[2] = budget

    for r in rows.itertuples():
        k = SLOTS.index(r.slot)
        sops = order_map.get(r.SeriesInstanceUID)
        if not sops:
            continue
        res = load_slot(Path(series_root) / study / r.SeriesInstanceUID, sops,
                        r.pix_row, budget[k], crop_mm=crop_mm, neutralise=neutralise)
        if res is None:
            continue
        a, n_distinct = res
        if side == "L" and neutralise != "mirror":
            # Sagittally the medial-lateral axis IS the slice axis, so the stack is
            # reversed; coronally and axially it lies in the image plane, so the last axis
            # is flipped. The band is symmetric about the stack centre, so reversing after
            # selection picks the same physical slices as reversing before it.
            a = a[::-1] if r.slot.startswith("SAG") else a[:, :, ::-1]
        off, bud = int(meta[1, k]), int(meta[2, k])
        vol[off:off + bud] = (np.clip(a, 0, 1) * 255).astype(np.uint8)
        meta[0, k] = 1.0
        meta[3, k] = n_distinct

    if neutralise == "mask":
        # Declare every slot present regardless of what decoded. An L3 masked softmax then
        # attends over all-zero volumes and nothing anywhere warns.
        meta[0] = 1.0
    return vol, meta


def build_study(job):
    study, rows, side, outdir, order_map, series_root, neutralise = job
    dst = outdir / f"{study}.npy"
    if dst.exists():
        return "skip", study, 0.0
    t0 = time.time()
    vol, meta = assemble_study(study, rows, side, order_map, series_root,
                               neutralise=neutralise)
    if meta[0].sum() == 0:
        return "empty", study, time.time() - t0
    np.save(dst, vol)
    np.save(outdir / f"{study}.meta.npy", meta)
    return "ok", study, time.time() - t0


# =============================================================== synthetic data for the tests
def _write_synth_series(sdir: Path, n, side_px, pix_mm, square_mm=70.0,
                        in_band_only=True, square_left=False, first_half_only=False):
    """Write `n` single-frame DICOMs: background 40, plus a centred bright square.

    The square is the probe. It is specified in MILLIMETRES, so a test can assert that its
    apparent width in the finished 384 px volume is the same whatever pix_mm was -- which is
    the only way to check "a constant physical extent" without trusting the code that
    implements it.

    `in_band_only` leaves the outer 5% of slices square-free, so a volume whose slices all
    carry a square proves the 6-94% band was honoured and one containing a flat slice proves
    it was not. `square_left` puts the square off-centre for the mirroring check;
    `first_half_only` restricts it to the low half of the stack for the sagittal reversal
    check.
    """
    sdir.mkdir(parents=True, exist_ok=True)
    sq = int(round(square_mm / pix_mm))
    names = []
    for j in range(n):
        a = np.full((side_px, side_px), 40, np.uint16)
        frac = j / max(n - 1, 1)
        draw = True
        if in_band_only and (frac < 0.05 or frac > 0.95):
            draw = False
        if first_half_only and frac > 0.5:
            draw = False
        if draw:
            r0 = (side_px - sq) // 2
            c0 = (side_px - sq) // 2 - (sq if square_left else 0)
            c0 = max(c0, 0)
            # The square's BRIGHTNESS carries the slice index, its geometry does not. That
            # makes every in-band slice byte-distinct after the uint8 quantisation -- so
            # "18 distinct slices" is a real measurement rather than a tautology -- while
            # leaving the width probe, which reads a mid-row threshold, untouched.
            a[r0:r0 + sq, c0:c0 + sq] = 400 + 8 * j
        ds = pydicom.Dataset()
        ds.file_meta = pydicom.dataset.FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
        ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        ds.SOPClassUID = pydicom.uid.MRImageStorage
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
        ds.Rows, ds.Columns = side_px, side_px
        ds.PixelSpacing = [pix_mm, pix_mm]
        ds.ImagePositionPatient = [0.0, 0.0, float(j) * 3.0]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.InstanceNumber = j + 1
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
        ds.PixelRepresentation = 0
        ds.PixelData = a.tobytes()
        fn = f"{j:04d}.dcm"
        pydicom.dcmwrite(sdir / fn, ds, enforce_file_format=True)
        names.append(fn)
    return names


def _square_width(img2d):
    """Apparent width in pixels of the bright square in one finished 384 px slice."""
    row = img2d[img2d.shape[0] // 2].astype(np.float32)
    thr = (row.max() + row.min()) / 2.0
    return int((row > thr).sum())


def _synth_index(rows):
    """rows: list of dicts -> a DataFrame shaped like series_index_v2.parquet."""
    df = pd.DataFrame(rows)
    for c, v in [("fluid_v2", False), ("fatsat_v2", False), ("weighting", "PD"),
                 ("pix_row", 0.5), ("slice_thickness", 3.0), ("n_slices", 30)]:
        if c not in df:
            df[c] = v
    return df


# ================================================================================ self-test
def self_test(neutralise=None):
    """Every rule in the contract, each with a mutation that makes it fail. CPU, ~40 s."""
    fails = []

    def check(ok, label, got, want):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:<52} {got}  (need {want})")
        if not ok:
            fails.append(label)

    print(f"build_cache_f2 self-test  (budget={BUDGET}, neutralise={neutralise})\n")

    # ---- 1. the budget itself -------------------------------------------------------
    check(BUDGETS == {"f2": (18, 14, 12, 8, 12), "dense": (26, 24, 26, 16, 28)}
          and BUDGET in BUDGETS.values() and N_TOTAL == sum(BUDGET) and len(set(BUDGET)) > 1,
          "budget is a named uneven layout, sums to N_TOTAL", f"{BUDGET} = {N_TOTAL}",
          "(18,14,12,8,12)=64 or (26,24,26,16,28)=120")
    check(OFFSET in ((0, 18, 32, 44, 52), (0, 26, 50, 76, 92))
          and OFFSET == tuple(int(x) for x in np.cumsum((0,) + BUDGET[:-1])),
          "offsets are the cumulative budget", str(OFFSET), "(0,18,32,44,52) or (0,26,50,76,92)")

    # ---- 2. the band, as a pure function --------------------------------------------
    bi = band_indices(100, 18, neutralise=neutralise)
    check(bi[0] == 6 and bi[-1] == 93, "band_indices(100,18) spans 6..93",
          f"{bi[0]}..{bi[-1]}", "6..93")
    check(len(band_indices(9, 18, neutralise=neutralise)) == 18,
          "a 9-slice series still yields 18 positions",
          len(band_indices(9, 18, neutralise=neutralise)), "18")
    short = band_indices(9, 18, neutralise=neutralise)
    runs = max(len(list(g)) for g in __import__("itertools").groupby(short))
    check(runs <= 3, "repeats are interleaved, not a tail block", f"longest run {runs}", "<= 3")

    # ---- 3. the ladder, as a pure function ------------------------------------------
    sag_fs = dict(StudyInstanceUID="S1", SeriesInstanceUID="a", Anatomical_Plane="Sagittal",
                  fluid_v2=True, fatsat_v2=True, n_slices=40)
    sag_no = dict(StudyInstanceUID="S1", SeriesInstanceUID="b", Anatomical_Plane="Sagittal",
                  fluid_v2=True, fatsat_v2=False, n_slices=30)
    cor_fs = dict(StudyInstanceUID="S1", SeriesInstanceUID="c", Anatomical_Plane="Coronal",
                  fluid_v2=True, fatsat_v2=True, n_slices=30)
    cor_t1 = dict(StudyInstanceUID="S1", SeriesInstanceUID="d", Anatomical_Plane="Coronal",
                  fluid_v2=False, fatsat_v2=False, weighting="T1", n_slices=30)
    ax_fs = dict(StudyInstanceUID="S1", SeriesInstanceUID="e", Anatomical_Plane="Axial",
                 fluid_v2=True, fatsat_v2=True, n_slices=30)
    full = select_slots(_synth_index([sag_fs, sag_no, cor_fs, cor_t1, ax_fs]),
                        neutralise=neutralise)
    got = dict(zip(full.slot, full.SeriesInstanceUID))
    check(got == {"SAG_FLUID": "a", "SAG_ALT": "b", "COR_FLUID": "c",
                  "COR_ALT": "d", "AX_FLUID": "e"},
          "a complete study fills all five slots, tier 0", str(got), "a b c d e")
    check(len(set(full.SeriesInstanceUID)) == len(full),
          "no series is spent on two slots", f"{len(set(full.SeriesInstanceUID))}/{len(full)}",
          "equal")

    # FALLBACK: a study with no fat-suppressed sagittal must still fill SAG_FLUID. This is
    # what the 'ladder' mutation removes.
    fb = select_slots(_synth_index([
        dict(sag_no, StudyInstanceUID="S2"),
        dict(StudyInstanceUID="S2", SeriesInstanceUID="f", Anatomical_Plane="Sagittal",
             fluid_v2=False, fatsat_v2=False, weighting="T1", n_slices=25)]),
        neutralise=neutralise)
    m = dict(zip(fb.slot, fb.SeriesInstanceUID))
    check(m.get("SAG_FLUID") == "b" and m.get("SAG_ALT") == "f",
          "no fat-sat sagittal: SAG_FLUID falls back, SAG_ALT takes T1",
          str(m), "SAG_FLUID=b SAG_ALT=f")

    # EXCLUSIVITY: one sagittal series fills one sagittal slot, never both.
    one = select_slots(_synth_index([dict(sag_fs, StudyInstanceUID="S3")]),
                       neutralise=neutralise)
    check(set(one.slot) == {"SAG_FLUID"}, "a lone sagittal series does not fill both slots",
          str(sorted(one.slot)), "['SAG_FLUID']")

    # DETERMINISM of the choice: two equally long candidates, different pixel spacing.
    tie = _synth_index([
        dict(sag_fs, SeriesInstanceUID="x", n_slices=40, pix_row=0.40),
        dict(sag_fs, SeriesInstanceUID="y", n_slices=40, pix_row=0.20)])
    a1 = select_slots(tie, neutralise=neutralise)
    a2 = select_slots(tie.sample(frac=1.0, random_state=7).reset_index(drop=True),
                      neutralise=neutralise)
    same = (a1.SeriesInstanceUID.tolist() == a2.SeriesInstanceUID.tolist())
    check(same and a1.iloc[0].SeriesInstanceUID == "y",
          "winner is the finer series and survives a row shuffle",
          f"{a1.iloc[0].SeriesInstanceUID} / shuffle-stable={same}", "y / True")

    # ---- 4. end to end, on synthetic DICOMs -----------------------------------------
    tmp = Path(tempfile.mkdtemp(prefix="f2_selftest_"))
    try:
        sroot = tmp / "series"
        out = tmp / "cache"
        out.mkdir(parents=True)
        # One synthetic study: a 50-slice sagittal fluid-FS series at 0.5 mm/px on a 400 px
        # grid (FOV 200 mm, so the 140 mm crop fits), plus a 50-slice CORONAL series at
        # 0.25 mm/px on an 800 px grid -- a different grid AND a different spacing, so the
        # constant-mm check has two points to compare.
        sops = {}
        sops["sag"] = _write_synth_series(sroot / "ST" / "sag", 50, 400, 0.5)
        sops["cor"] = _write_synth_series(sroot / "ST" / "cor", 50, 800, 0.25)
        # A third series whose field of view (120 mm) is SMALLER than the 140 mm crop, so
        # the pad path has to run.
        sops["ax"] = _write_synth_series(sroot / "ST" / "ax", 50, 240, 0.5)

        rows = pd.DataFrame([
            dict(slot="SAG_FLUID", SeriesInstanceUID="sag", pix_row=0.5),
            dict(slot="COR_FLUID", SeriesInstanceUID="cor", pix_row=0.25),
            dict(slot="AX_FLUID", SeriesInstanceUID="ax", pix_row=0.5)])
        job = ("ST", rows, None, out, sops, sroot, neutralise)
        status, _, _ = build_study(job)
        check(status == "ok", "synthetic study builds", status, "ok")
        vol = np.load(out / "ST.npy")
        meta = np.load(out / "ST.meta.npy")

        check(vol.shape == (N_TOTAL, 384, 384) and vol.dtype == np.uint8,
              f"volume is ({N_TOTAL},384,384) uint8", f"{vol.shape} {vol.dtype}",
              f"({N_TOTAL},384,384) uint8")
        check(meta.shape == (4, 5) and meta.dtype == np.float32,
              "sidecar is (4,5) float32", f"{meta.shape} {meta.dtype}", "(4,5) float32")

        # MASK. Two slots were never offered a series; they must read absent AND be zero.
        absent = [i for i, s in enumerate(SLOTS) if s in ("SAG_ALT", "COR_ALT")]
        check(all(meta[0, i] == 0.0 for i in absent),
              "unfilled slots are marked absent", meta[0].tolist(), "[1,0,1,0,1]")
        zero_ok = all(vol[int(meta[1, i]):int(meta[1, i] + meta[2, i])].max() == 0
                      for i in absent)
        check(zero_ok, "unfilled slots hold no pixels", zero_ok, "True")

        # BUDGET, the only check an even-budget rack fails. The sagittal source has 50
        # slices, so an 18-frame (26 dense) budget over the 6-94% band takes that many
        # DISTINCT slices.
        k = SLOTS.index("SAG_FLUID")
        b = BUDGET[k]
        sag = vol[int(meta[1, k]):int(meta[1, k] + meta[2, k])]
        n_uniq = len({s.tobytes() for s in sag})
        check(meta[2, k] == b and n_uniq == b and meta[3, k] == b,
              f"SAG_FLUID holds {b} distinct real slices",
              f"budget {int(meta[2, k])} distinct {n_uniq} n_dist {int(meta[3, k])}",
              f"{b} / {b} / {b}")

        # BAND. Every in-band source slice carries a square; the outer 5% do not. A flat
        # slice anywhere in the volume means an endpoint was sampled.
        stds = [s.std() for s in sag]
        check(min(stds) > 1.0, "no flat end-slice was sampled (6-94% band honoured)",
              f"min per-slice std {min(stds):.2f}", "> 1.0")

        # CROP IN MM. The square is 70 mm on both series, i.e. exactly half of the 140 mm
        # crop, so it must be ~192 px wide in BOTH -- despite 0.5 vs 0.25 mm/px and 400 vs
        # 800 px grids. That equality is the whole claim.
        kc = SLOTS.index("COR_FLUID")
        cor = vol[int(meta[1, kc]):int(meta[1, kc] + meta[2, kc])]
        w_sag = _square_width(sag[len(sag) // 2])
        w_cor = _square_width(cor[len(cor) // 2])
        check(abs(w_sag - 192) <= 4 and abs(w_cor - 192) <= 4,
              "70 mm reads as 192 px at both pixel spacings",
              f"sag {w_sag}px cor {w_cor}px", "192 +-4")

        # PAD. The axial series has a 120 mm field of view, narrower than the crop. Its
        # 70 mm square must still be 192 px: v1 would skip the crop and give 224 px.
        ka = SLOTS.index("AX_FLUID")
        ax = vol[int(meta[1, ka]):int(meta[1, ka] + meta[2, ka])]
        w_ax = _square_width(ax[len(ax) // 2])
        check(abs(w_ax - 192) <= 4, "a 120 mm FOV is padded, not rescaled",
              f"{w_ax}px", "192 +-4")

        # DETERMINISM, end to end and byte for byte.
        out2 = tmp / "cache2"
        out2.mkdir()
        build_study(("ST", rows, None, out2, sops, sroot, neutralise))
        check(np.array_equal(vol, np.load(out2 / "ST.npy")),
              "two builds of one study are byte-identical", "compared", "equal")

        # MIRRORING, coronal: an off-centre square on a left knee must come back on the
        # other side of the image.
        sops["corL"] = _write_synth_series(sroot / "STL" / "corL", 50, 800, 0.25,
                                           square_left=True)
        rl = pd.DataFrame([dict(slot="COR_FLUID", SeriesInstanceUID="corL", pix_row=0.25)])
        outL = tmp / "cacheL"
        outL.mkdir()
        build_study(("STL", rl, "L", outL, sops, sroot, neutralise))
        vL = np.load(outL / "STL.npy")
        mL = np.load(outL / "STL.meta.npy")
        sl = vL[int(mL[1, kc]):int(mL[1, kc] + mL[2, kc])][6]
        mid = sl.shape[1] // 2
        right_heavier = sl[:, mid:].astype(float).mean() > sl[:, :mid].astype(float).mean()
        check(right_heavier, "left knee: coronal square moves to the right half",
              f"L {sl[:, :mid].mean():.1f} R {sl[:, mid:].mean():.1f}", "R > L")

        # MIRRORING, sagittal: the slice AXIS is reversed, not the image.
        sops["sagL"] = _write_synth_series(sroot / "STL2" / "sagL", 50, 400, 0.5,
                                           first_half_only=True)
        rl2 = pd.DataFrame([dict(slot="SAG_FLUID", SeriesInstanceUID="sagL", pix_row=0.5)])
        outL2 = tmp / "cacheL2"
        outL2.mkdir()
        build_study(("STL2", rl2, "L", outL2, sops, sroot, neutralise))
        v2 = np.load(outL2 / "STL2.npy")
        sg = v2[OFFSET[0]:OFFSET[0] + BUDGET[0]]
        early = float(np.mean([s.std() for s in sg[:6]]))
        late = float(np.mean([s.std() for s in sg[-6:]]))
        check(late > early * 3, "left knee: sagittal stack order is reversed",
              f"first-6 std {early:.1f} last-6 std {late:.1f}", "last >> first")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n  {'ALL CHECKS PASSED' if not fails else 'FAILED: ' + '; '.join(fails)}")
    return 0 if not fails else 1


# ===================================================================================== main
def _load_inputs():
    si = pd.read_parquet(ROOT / "series_index_v2.parquet")
    st = pd.read_parquet(ROOT / "study_index.parquet")
    so = pd.read_parquet(ROOT / "slice_order.parquet")
    order_map = (so.sort_values("order").groupby("SeriesInstanceUID").sop
                 .apply(list).to_dict())
    side_map = dict(zip(st.StudyInstanceUID, st.side))
    return si, order_map, side_map


def plan(sel, n_studies, outdir):
    print(f"\n  slot        tier0   tier1   tier2    filled   fill%")
    for s in SLOTS:
        g = sel[sel.slot == s]
        t = [int((g.tier == i).sum()) for i in range(3)]
        print(f"  {s:<11} {t[0]:>5}  {t[1]:>6}  {t[2]:>6}   {len(g):>7}  "
              f"{100 * len(g) / n_studies:>5.1f}%")
    print(f"  mean slots filled per study: "
          f"{len(sel) / n_studies:.3f} / {len(SLOTS)}")
    need = n_studies * N_TOTAL * IMG * IMG
    free = shutil.disk_usage(Path(outdir).anchor).free
    print(f"\n  size: {n_studies} studies x {N_TOTAL} x {IMG} x {IMG} uint8 "
          f"= {need / 1e9:.1f} GB ({need / 2**30:.1f} GiB)")
    print(f"  free on {Path(outdir).anchor}: {free / 2**30:.1f} GiB   "
          f"headroom after build: {(free - need) / 2**30:.1f} GiB")
    return need, free


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"data/rsna_knee_cache_f2")
    ap.add_argument("--series-root", default=str(ROOT / "train_series"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--studies", default=None, help="file with one StudyInstanceUID per line")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--budget", default="f2", choices=list(BUDGETS),
                    help="named per-slot frame layout: f2 = 18/14/12/8/12 (64), "
                         "dense = 26/24/26/16/28 (120)")
    ap.add_argument("--plan", action="store_true", help="fill rates and disk budget, no build")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--neutralise", default=None,
                    choices=["band", "budget", "crop", "ladder", "nopad", "mirror",
                             "nondet", "mask"],
                    help="break one rule; --self-test must then FAIL")
    ap.add_argument("--min-free-gib", type=float, default=8.0,
                    help="refuse to start if the build would leave less than this free")
    a = ap.parse_args()
    use_budget(a.budget)

    if a.self_test:
        return self_test(a.neutralise)

    outdir = Path(a.out)
    if outdir.resolve() == V1_CACHE.resolve():
        print(f"REFUSED: --out is the v1 cache ({V1_CACHE}). Pick a new path.")
        return 2
    # build_study skips any study whose .npy already exists, so a second layout aimed at an
    # existing cache would write nothing and then overwrite its meta.json with the wrong
    # budget. One layout per directory.
    old_meta = outdir / "meta.json"
    if old_meta.exists():
        old_budget = json.loads(old_meta.read_text(encoding="utf-8")).get("budget")
        if old_budget != list(BUDGET):
            print(f"REFUSED: {outdir} holds a cache with budget {old_budget}, not "
                  f"{list(BUDGET)} (--budget {a.budget}). Pick a new --out.")
            return 2

    si, order_map, side_map = _load_inputs()
    sel = select_slots(si, neutralise=a.neutralise)
    studies = sorted(si.StudyInstanceUID.unique())
    if a.studies:
        want = [l.strip() for l in Path(a.studies).read_text().splitlines() if l.strip()]
        studies = [s for s in studies if s in set(want)]
    if a.limit:
        studies = studies[:a.limit]

    need, free = plan(sel[sel.StudyInstanceUID.isin(studies)], len(studies), outdir)
    if a.plan:
        return 0
    if free - need < a.min_free_gib * 2 ** 30:
        print(f"\nREFUSED: build needs {need / 2**30:.1f} GiB, only {free / 2**30:.1f} GiB "
              f"free, and --min-free-gib is {a.min_free_gib}.")
        return 2

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "meta.json").write_text(json.dumps(dict(
        family="f2", slots=SLOTS, budget_name=a.budget, budget=list(BUDGET), offset=list(OFFSET),
        n_total=N_TOTAL, crop_mm=CROP_MM, img=IMG, band=list(BAND), pctl=list(PCTL),
        layout="FLAT (n_total, img, img); slot k occupies rows offset[k]:offset[k]+budget[k]",
        order="geometry (slice_order.parquet)",
        intensity=f"per-series {PCTL[0]}/{PCTL[1]} percentile over DISTINCT decoded slices, "
                  f"taken inside the crop window and before any pad",
        scale=f"constant {CROP_MM} mm centre crop, PADDED (not skipped) where the field of "
              f"view is narrower; {CROP_MM / IMG:.4f} mm/px for every series",
        sampling=f"linspace over {BAND[0]:.0%}-{BAND[1]:.0%} of the real stack; repeats "
                 f"interleaved, never a tail block",
        side="left knees mirrored; SAG reverses slice order, COR/AX flips last axis",
        rack="plane+role with a fallback ladder; a series is spent on at most one slot",
        meta_file="<study>.meta.npy (4,5) float32: row0 present, row1 start, row2 budget, "
                  "row3 distinct real slices behind the slot",
        dtype="uint8", shape=[N_TOTAL, IMG, IMG],
    ), indent=2), encoding="utf-8")

    by_study = {s: g for s, g in sel.groupby("StudyInstanceUID")}
    jobs = [(s, by_study[s], side_map.get(s), outdir, order_map, a.series_root, a.neutralise)
            for s in studies if s in by_study]
    print(f"\nbuilding f2 cache for {len(jobs)} studies -> {outdir}  ({a.threads} threads)\n",
          flush=True)

    t0 = time.time()
    counts = {"ok": 0, "skip": 0, "empty": 0}
    with ThreadPoolExecutor(max_workers=a.threads) as ex:
        for i, (status, study, dt) in enumerate(ex.map(build_study, jobs), 1):
            counts[status] += 1
            if i % 25 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  {i}/{len(jobs)}  {counts}  {el / 60:.1f} min, "
                      f"{i / el:.2f} study/s, eta "
                      f"{(len(jobs) - i) / max(i / el, 1e-9) / 60:.0f} min", flush=True)

    el = time.time() - t0
    print(f"\ndone in {el / 60:.1f} min: {counts}   "
          f"{counts['ok'] / max(el, 1e-9):.2f} study/s built")
    if counts["empty"]:
        print(f"  WARNING: {counts['empty']} studies produced no usable slot")
    return 0


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    sys.exit(main())
