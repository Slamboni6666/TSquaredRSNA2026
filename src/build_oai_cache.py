"""
build_oai_cache.py -- turn the 246 label-bearing OAI-ZIB DESS volumes into build_cache format.

WHY THIS FILE EXISTS
build_oai_labels.py joined 246 of the 507 OAI-ZIB knees to real MOAKS readings -- findings a
radiologist read OFF THE IMAGES, not out of an LLM's paraphrase of a report. Two of our four
remaining headroom targets are label-limited rather than model-limited:

    Synovitis     teacher 0.648, ours 0.751 -- we already beat the labels by +0.103
    Lateral OA    teacher 0.813, ours 0.814 -- we have caught the labels exactly

No architecture fixes a 0.648 teacher. MOAKS gives Synovitis 108 positives of 170 read and
Lateral OA 130 of 246, graded by a human looking at the pixels. That is the only thing on this
machine that can move those two targets. This file is the pixel half of making them usable;
build_oai_labels.py is the label half.

It writes to data/oai_cache. It NEVER writes into the competition cache, which a
live 5-fold run is reading.

------------------------------------------------------------------------------------------
DECISION 1 -- WHICH SLOT THE DESS OCCUPIES.  Answer: slot 0, SAG_FLUID_FS. Not slot 3.
------------------------------------------------------------------------------------------
Everything downstream depends on this, so it was measured rather than assumed.

The brief proposed SAG_FLUID_NOFS (index 3) on the grounds that DESS is "a fluid-sensitive
gradient echo without fat suppression". The first half is right and the second half is wrong
for THIS data. The OAI protocol series is SAG_3D_DESS_**WE** -- water excitation -- and water
excitation IS a fat-suppression technique: it selectively excites water and leaves fat
unexcited. The files on disk match that protocol exactly and unanimously: all 507 are
160 x 384 x 384 at 0.70 / 0.3646 / 0.3646 mm, which is the published OAI DESS-WE geometry.

Measured, not argued. Every cached competition slot and 40 OAI volumes were pushed through the
identical normalisation (130 mm crop, 1-99 percentile, 336 px) and the fraction of pixels above
0.5 -- "how much bright tissue is there", which is exactly what fat suppression removes --
compared:

    SAG_FLUID_FS      0.2403          <-- OAI sits on top of this one
    OAI DESS-WE       0.2444
    AX_FLUID_FS       0.1671
    COR_FLUID_FS      0.1570
    SAG_T1            0.3813
    SAG_FLUID_NOFS    0.4283          <-- 1.75x the OAI value
    COR_T1            0.4760

And visually, which is how this project has caught every preprocessing bug it has caught: in
the OAI volumes the femoral and tibial MARROW is the darkest tissue in the frame. In a cached
SAG_FLUID_NOFS or COR_T1 volume the marrow is the brightest. Marrow is fat. The DESS is
suppressed; slot 3 is not the sequence it resembles.

So the DESS matches slot 0 on all three axes that define a slot -- sagittal plane,
fluid-sensitive contrast, fat suppressed. Two practical consequences follow, both favourable:
slot 0 is present in 93.4% of competition studies against slot 3's 62.4%, and SLOT_PRIOR in
model.py already tilts six of the twelve targets towards slot 0. Supervision landing there
reaches the slot the model actually reads.

WHAT THIS STILL DOES NOT FIX: DESS is a steady-state gradient echo, not a clinical PD/T2
fat-sat. Cartilage is bright and sharply delineated on DESS -- that is why the cartilage
segmentation community uses it -- where on clinical PD-FS it is intermediate. The domain gap is
real; slot 0 is the closest of six, not a match. `--slot 3` is one flag away so the alternative
stays measurable.

------------------------------------------------------------------------------------------
DECISION 2 -- WHICH END OF THE SLICE AXIS IS MEDIAL.  Answer: index 0 is MEDIAL, so OAI axis 0
is kept as acquired for a right knee. This one nearly went in backwards.
------------------------------------------------------------------------------------------
The trap is that DICOM patient coordinates are **LPS** (+x = patient LEFT) and NIfTI is **RAS**
(+x = patient RIGHT). The sign flips between the two halves of the argument, and getting it
wrong swaps medial for lateral on five of the twelve targets -- including Lateral Meniscus and
Lateral OA, two of the four this dataset exists to fix. Silent, too: the volumes still have the
right shape, the right dtype and a plausible histogram.

Three independent lines of evidence, all agreeing:

  GEOMETRY.  build_slice_order.py sorts by k = p . (r x d). For the standard sagittal
    IOP (0,1,0, 0,0,-1) that normal is (-1,0,0) in LPS, i.e. it points to the patient's RIGHT.
    Measured over 60 randomly chosen cached sagittal series: ImagePositionPatient x decreases
    along `order` in 60 of 60. So the cached stack advances toward the patient's right. For an
    unmirrored right knee that is medial -> lateral; build_cache reverses left knees, which
    lands them the same way round. Cached index 0 = MEDIAL.

  SEGMENTATION GROUND TRUTH.  data/OAI-ZIB/Labels_Pretrain carries the OAIZIB-CM masks, in which
    label 4 is MEDIAL and label 5 is LATERAL tibial cartilage. Over 25 volumes the medial
    centroid sits at axis-0 index 54.4 and the lateral at 113.5, and medial < lateral in 25 of
    25. So OAI axis 0 already runs medial -> lateral. This is anatomy, not inference, and the
    self-test re-runs it.

  DIRECT CORRELATION.  Mean OAI frames against mean competition SAG_FLUID_FS frames at matched
    positions, summed over frames 2/6/25/29: as-acquired 2.863, reversed 2.650.

Hence: right knee -> keep, left knee -> reverse. All 507 OAI-ZIB knees are RIGHT (subInfo
KneeSide == 1 for all 507, and info.zip's kneeSideInfo.csv says "right" 507 times; the
self-test asserts the two sources agree). The left branch is therefore dead code on this data,
which is exactly why the self-test exercises it with a synthetic left knee built by mirroring a
real right one -- that case has an exact expected answer, so the test asserts equality rather
than a correlation.

------------------------------------------------------------------------------------------
DECISION 3 -- SLAB AVERAGING, because a 0.7 mm frame is not a 3 mm frame.
------------------------------------------------------------------------------------------
Every frame in the competition cache is one acquired clinical slice: over the chosen sagittal
series, median thickness 3.0 mm (p10 2.5, p90 4.0) and median 30 slices. The DESS is 160 slices
at 0.7 mm. Feeding thin slices into a cache built of thick ones is a second domain shift on top
of the sequence one: a 0.7 mm frame carries ~sqrt(5) more noise and none of the through-plane
partial-volume averaging the model has learned to expect.

So consecutive slices are averaged into slabs before sampling. The default 3.5 mm is chosen
because 3.5 / 0.7 = 5 exactly and 160 / 5 = 32 = N_SLICE: every cached frame is then a distinct
real slab, nothing is duplicated or interpolated along the slice axis, n_valid is honestly 32,
and the stack spans 112 mm against the ~105 mm a typical 30-slice competition sagittal covers.
In-plane resolution is untouched -- 0.3646 mm native is finer than the 0.387 mm/px the cache
resamples to, so nothing is lost where the 1 mm meniscal tear bound applies.

`--slab-mm 0` disables it. The self-test asserts it is not a no-op, because a preprocessing
step that silently does nothing is this project's characteristic failure mode.

------------------------------------------------------------------------------------------
EVERYTHING ELSE IS build_cache's CONTRACT, IMPORTED RATHER THAN RETYPED
------------------------------------------------------------------------------------------
CROP_MM, IMG, N_SLICE, SLOTS and sample_indices are imported from build_cache so the two caches
cannot drift apart; the self-test asserts the imported values are still the ones this file was
written against, so a change over there fails loudly here instead of quietly producing an
incompatible cache.

  shape       (6, 32, 336, 336) uint8 -- five slots are all zeros. That is 4.4 GB of zeros
              over 246 studies, and it is the price of being drop-in: KneeStudies indexes
              vol[s] for s in range(6), so a 1-slot array would read out of bounds. mmap means
              the zeros are never actually paged in.
  mask        (2, 6) float32 -- row 0 slot-present, row 1 n_valid. One-hot at the DESS slot.
  crop        130 mm of anatomy about the image centre, then 336 px => 0.387 mm/px, identical
              to the competition cache. At 0.3646 mm/px native that is a 357 px crop of 384.
  intensity   1st/99th percentile over the sampled stack, after the crop, in exactly the order
              build_cache.load_slot does it.
  in-plane    row index increases INFERIORLY, column index increases POSTERIORLY, matching the
              standard sagittal DICOM (IOP row-dir +y, col-dir -z). From the RAS array that is
              slice.T[::-1, ::-1]; picked by correlating mean images against the competition
              cache -- 0.732 for that against 0.475 / -0.135 / -0.394 for the alternatives.

Also writes oai_cache_labels.csv in the exact schema KneeStudies indexes (StudyInstanceUID, the
12 targets, and 12 __state columns). MCL and Fracture are not scored by MOAKS, and neither are
the unread cells of Effusion / Synovitis / Baker's; every one of those is written 0.0 with state
"silent" so KneeStudies masks it out of the loss. 0.0 and not NaN deliberately -- the mask
multiplies, and 0 * nan is nan, which would take the whole batch's loss with it.

CPU-only. Never touches the GPU, never writes outside --out.

Usage:
    python build_oai_cache.py --self-test
    python build_oai_cache.py --limit 4 --out data/oai_cache_smoke
    python build_oai_cache.py                     # the 246 labelled knees, ~5.3 GB
    python build_oai_cache.py --all               # all 507, for self-supervised use
"""

from __future__ import annotations

import argparse
import io
import itertools
import json
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Imported, not retyped, so the two caches cannot drift apart. self_test() asserts these are
# still the values this file was written against -- see the note in the docstring.
from build_cache import CROP_MM, IMG, N_SLICE, SLOTS, sample_indices

ROOT = Path(__file__).resolve().parent
ZIB_IMAGES = Path(r"data/OAI-ZIB/Images_Pretrain")
ZIB_LABELS = Path(r"data/OAI-ZIB/Labels_Pretrain")
ZIB_INFO = Path(r"data/OAI-ZIB/Zip_Files/info.zip")
OUT_DEFAULT = Path(r"data/oai_cache")

# See DECISION 1. Measured, not assumed: bright-pixel fraction 0.2444 for OAI against 0.2403
# for SAG_FLUID_FS and 0.4283 for SAG_FLUID_NOFS.
DESS_SLOT = 0

# See DECISION 3. 3.5 / 0.7 = 5 and 160 / 5 = 32 = N_SLICE exactly.
SLAB_MM = 3.5

# model.py's target order. Retyped here on purpose: importing model.py would pull torch and timm
# into a CPU preprocessing script, and worse, would make this module a new import edge into a
# file the live run's dataloader workers re-import every epoch.
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
# KneeStudies masks a cell out of the loss iff its state is in {"silent", "not_mentioned"}.
SILENT = "silent"
ASSERTED = "present_ungraded"


def load_nifti(path):
    """-> (arr, mm) with arr (n_sag, n_ant, n_sup) float32 in RAS index order.

    as_closest_canonical reorders and flips the axes so that increasing index means +x (patient
    RIGHT), +y (anterior), +z (superior) whatever the source orientation was. Measured over all
    507 OAI-ZIB files this is already a no-op -- every one is ('R','A','S') with affine[0,0] > 0
    -- but the medial/lateral rule below is written in terms of "+x is the patient's right", so
    it is guaranteed rather than assumed. A volume that arrived in some other orientation would
    otherwise be silently mirrored, which is the one error with no visible symptom.
    """
    import nibabel as nib
    img = nib.as_closest_canonical(nib.load(str(path)))
    ax = nib.aff2axcodes(img.affine)
    if ax != ("R", "A", "S"):
        raise ValueError(f"{Path(path).name}: cannot canonicalise orientation, got {ax}")
    return (np.asanyarray(img.dataobj).astype(np.float32),
            tuple(float(z) for z in img.header.get_zooms()))


def slab_average(arr, mm_slice, slab_mm=SLAB_MM):
    """Average consecutive slices into ~slab_mm slabs. -> (arr, n_avg).

    The remainder is trimmed from BOTH ENDS, not from the tail. Trimming the tail alone would
    always eat the lateral edge of the knee, and Lateral Meniscus and Lateral OA are two of the
    four targets this dataset was brought in to fix.
    """
    n_avg = 1 if slab_mm <= 0 else max(1, int(round(slab_mm / mm_slice)))
    if n_avg <= 1:
        return arr, 1
    n = arr.shape[0]
    rem = n % n_avg
    lo = rem // 2
    arr = arr[lo:lo + (n - rem)]
    return arr.reshape(-1, n_avg, *arr.shape[1:]).mean(1), n_avg


def prepare(arr, mm, side, slab_mm=SLAB_MM):
    """The whole pixel pipeline as a pure function -> ((32, IMG, IMG) float32 in [0,1], n_slab).

    `arr` is (n_sag, n_ant, n_sup) in RAS index order, `mm` its zooms in the same order, `side`
    is "R" or "L". Kept separate from file IO so the self-test can feed it a synthetic left knee
    and a truncated volume without having to write NIfTI files.

    Step order follows build_cache.load_slot exactly, and the order is load-bearing: sample,
    then crop to a constant physical extent, THEN take percentiles over what survived, then
    resample. Percentiles taken before the crop are set by whatever surrounds the joint, so the
    same knee would land at a different brightness depending only on how wide the scan was.
    """
    if arr.ndim != 3:
        raise ValueError(f"expected a 3D volume, got {arr.shape}")
    mm_slice, mm_ant, mm_sup = mm

    # --- medial/lateral canonicalisation. See DECISION 2. Canonical: index 0 = MEDIAL.
    # After as_closest_canonical, increasing index 0 means +x = patient RIGHT. Lateral is +x for
    # a right knee and -x for a left one, so a right knee already runs medial -> lateral and a
    # left knee runs the other way. Written as the derivation rather than as a bare
    # `if side == "L"` so the reasoning is auditable at the point of use.
    lateral_at_high_index = (side == "R")
    if not lateral_at_high_index:
        arr = arr[::-1]

    # --- slabs, so that a cached frame is a ~3 mm slice like every other frame in the cache
    arr, _ = slab_average(arr, mm_slice, slab_mm)
    n_slab = arr.shape[0]

    # --- build_cache's own sampler, imported so that the interleave-the-repeats behaviour bug
    # B4a forced on it applies here identically for short volumes.
    idx = sample_indices(n_slab, N_SLICE)
    if not idx:
        raise ValueError("empty volume")

    # --- (n_ant, n_sup) plane -> (row = inferior, col = posterior), the standard sagittal
    # display the competition DICOMs are stored in (IOP row-dir +y posterior, col-dir -z
    # inferior). Chosen by correlating mean images against the cache: 0.732 for this against
    # 0.475 / -0.135 / -0.394 for the three alternatives.
    vol = np.stack([arr[j].T[::-1, ::-1] for j in idx])
    n_row, n_col = vol.shape[1], vol.shape[2]
    mm_row, mm_col = mm_sup, mm_ant          # rows run along S, columns along A

    # --- a constant 130 mm of anatomy about the image centre. Each axis gets its own pixel
    # count: build_cache uses pix_row for both because 99.8% of its series have square pixels,
    # and doing it per-axis here is identical when they are square (OAI always is, 0.3646 both
    # ways) and correct when they are not.
    def crop_n(extent_px, mm_px):
        n = int(round(CROP_MM / mm_px)) if mm_px and mm_px > 0 else 0
        # build_cache's exception: when the requested extent is as wide as the image, resample
        # at the native extent rather than pad. Never triggers on OAI (357 of 384), but a
        # truncated volume must not be padded into a fake field of view.
        return n if 16 < n < extent_px else extent_px
    nr, nc = crop_n(n_row, mm_row), crop_n(n_col, mm_col)
    r0, c0 = (n_row - nr) // 2, (n_col - nc) // 2
    vol = vol[:, r0:r0 + nr, c0:c0 + nc]

    lo, hi = np.percentile(vol, [1, 99])
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)

    out = np.empty((len(vol), IMG, IMG), np.float32)
    for i, s in enumerate(vol):
        out[i] = np.asarray(Image.fromarray(s).resize((IMG, IMG), Image.BILINEAR))
    return out, n_slab


def build_one(job):
    """(sid, path, side, outdir, slot, slab_mm) -> status. Resumable: skips what already exists.

    "Already exists" means already exists AT THE REQUESTED SLOT. An unconditional skip made
    `--slot` a no-op against a directory an earlier run had filled: every knee came back "skip",
    the DESS stayed where it was, and main() rewrote meta.json to claim the new slot. Nothing
    downstream could notice -- shape, dtype and the sidecar are all still valid -- so the cache
    would have been read as SAG_FLUID_NOFS while holding SAG_FLUID_FS. The sidecar records
    which slot was written, so ask it.

    NOTE the same hazard is still open for --slab-mm: the sidecar cannot express slab size, so a
    re-run with a different --slab-mm against a filled directory skips everything and meta.json
    is rewritten to claim the new value. Build a different --out for that one.
    """
    sid, path, side, outdir, slot, slab_mm = job
    dst = outdir / f"{sid}.npy"
    msk = outdir / f"{sid}.mask.npy"
    if dst.exists() and msk.exists():
        try:
            if float(np.load(msk)[0][slot]) == 1.0:
                return "skip", sid, 0.0
        except Exception:
            pass                              # unreadable sidecar -> rebuild, do not trust it
    t0 = time.time()
    try:
        arr, mm = load_nifti(path)
        a, n_slab = prepare(arr, mm, side, slab_mm)
    except Exception as e:                    # a decode failure must not look like a black knee
        return f"fail:{type(e).__name__}", sid, time.time() - t0

    vol = np.zeros((len(SLOTS), N_SLICE, IMG, IMG), np.uint8)
    vol[slot] = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    mask = np.zeros(len(SLOTS), np.float32)
    n_valid = np.zeros(len(SLOTS), np.float32)
    mask[slot] = 1.0
    n_valid[slot] = min(n_slab, N_SLICE)
    np.save(dst, vol)
    np.save(outdir / f"{sid}.mask.npy", np.stack([mask, n_valid]))
    return "ok", sid, time.time() - t0


# ================================================================================== labels
def side_letter(v):
    """OAI SIDE coding: 1 = RIGHT, 2 = LEFT."""
    return {1: "R", 2: "L"}.get(int(v)) if pd.notna(v) else None


def label_table(lab):
    """oai_zib_labels.csv rows -> the frame KneeStudies indexes.

    Every cell MOAKS did not read -- MCL and Fracture entirely, plus the unread rows of
    Effusion / Synovitis / Baker's -- is written 0.0 with state "silent". 0.0 rather than NaN
    because KneeStudies multiplies by the mask and 0 * nan is nan, which would propagate out of
    the masked cell and take the whole batch's loss with it.
    """
    out = pd.DataFrame({"StudyInstanceUID": lab.oaizib.values})
    for t in TARGETS:
        v = (pd.to_numeric(lab[t], errors="coerce").values if t in lab.columns
             else np.full(len(lab), np.nan))
        out[t] = np.where(np.isnan(v), 0.0, v).astype(np.float32)
        out[t + "__state"] = np.where(np.isnan(v), SILENT, ASSERTED)
    return out


def read_side_map():
    """oaizib id -> "R"/"L", cross-checked against the second source shipped in info.zip.

    Two independent sources ride in the same archive, and this is the field that, if wrong,
    mirrors the anatomy with no other symptom. Disagreement is fatal rather than warned about:
    there would be no way to tell which of the two to believe.
    """
    lab = pd.read_csv(ROOT / "oai_zib_labels.csv")
    a = {r.oaizib: side_letter(r.KneeSide) for r in lab.itertuples()}
    with zipfile.ZipFile(ZIB_INFO) as z:
        # header=None: the first row of kneeSideInfo.csv is data, not a header row.
        ks = pd.read_csv(io.BytesIO(z.read("info/kneeSideInfo.csv")), header=None,
                         names=["file", "side"])
    b = {Path(str(f)).name.replace(".nii.gz", ""): str(s).strip()[:1].upper()
         for f, s in zip(ks.file, ks.side)}
    both = set(a) & set(b)
    bad = sorted(k for k in both if a[k] != b[k])
    if bad:
        raise ValueError(f"subInfo and kneeSideInfo disagree on knee side for {len(bad)} "
                         f"volumes, e.g. {bad[:5]} -- refusing to guess")
    return a, b, len(both)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--slot", type=int, default=DESS_SLOT, help="cache slot the DESS occupies")
    ap.add_argument("--slab-mm", type=float, default=SLAB_MM, help="0 disables slab averaging")
    ap.add_argument("--all", action="store_true", help="all 507, not just the 246 labelled")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    lab = pd.read_csv(ROOT / "oai_zib_labels.csv")
    sides, _, _ = read_side_map()

    have = lab[[t for t in TARGETS if t in lab.columns]].notna().any(axis=1)
    keep = lab if a.all else lab[have]
    keep = keep[keep.image.astype(str).str.len() > 0]
    if a.limit:
        keep = keep.head(a.limit)

    tbl = label_table(keep)
    assert not tbl[TARGETS].isna().any().any(), "NaN survived into the label table"
    tbl.to_csv(outdir / "oai_cache_labels.csv", index=False)

    (outdir / "meta.json").write_text(json.dumps(dict(
        source="OAI-ZIB sagittal 3D DESS-WE (NIfTI), 160 x 384 x 384 @ 0.70/0.3646/0.3646 mm",
        slots=SLOTS, dess_slot=a.slot, dess_slot_name=SLOTS[a.slot],
        dess_slot_reason="DESS-WE is water-excited, i.e. fat-suppressed: bright-pixel fraction "
                         "0.2444 vs SAG_FLUID_FS 0.2403 and SAG_FLUID_NOFS 0.4283",
        crop_mm=CROP_MM, img=IMG, n_slice=N_SLICE,
        order="medial (index 0) -> lateral, matching the competition cache; OAI RAS axis 0 "
              "already runs that way for a right knee",
        intensity="per-volume 1-99 percentile over the sampled stack, after the crop",
        side="all 507 OAI-ZIB knees are RIGHT; a left knee would reverse the slice axis",
        slab_mm=a.slab_mm,
        slab_reason="0.7 mm DESS slices averaged into ~3.5 mm slabs to match the competition "
                    "median sagittal slice thickness of 3.0 mm (p10 2.5, p90 4.0)",
        in_plane="row index increases inferiorly, column index increases posteriorly",
        mask_file="row 0 = slot present (0/1); row 1 = real slab count behind that slot",
        dtype="uint8", shape=[len(SLOTS), N_SLICE, IMG, IMG],
        labels="oai_cache_labels.csv; MCL and Fracture are state=silent (MOAKS does not score "
               "them), as is every cell MOAKS did not read",
    ), indent=2), encoding="utf-8")

    jobs = [(r.oaizib, r.image, sides.get(r.oaizib), outdir, a.slot, a.slab_mm)
            for r in keep.itertuples()]
    missing = [j[0] for j in jobs if j[2] is None]
    if missing:
        raise ValueError(f"no knee side for {len(missing)} volumes, e.g. {missing[:5]}")

    gb = len(jobs) * len(SLOTS) * N_SLICE * IMG * IMG / 1e9
    print(f"building OAI cache for {len(jobs)} knees -> {outdir}")
    print(f"  slot {a.slot} ({SLOTS[a.slot]}), slab {a.slab_mm} mm, "
          f"{len(SLOTS)} x {N_SLICE} x {IMG}px = ~{gb:.1f} GB\n", flush=True)

    t0 = time.time()
    counts = {}
    with ThreadPoolExecutor(max_workers=a.threads) as ex:
        for i, (status, sid, dt) in enumerate(ex.map(build_one, jobs), 1):
            counts[status] = counts.get(status, 0) + 1
            if i % 25 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  {i}/{len(jobs)}  {counts}  {el / 60:.1f} min, "
                      f"eta {(len(jobs) - i) / max(i / el, 1e-9) / 60:.0f} min", flush=True)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min: {counts}")
    bad = {k: v for k, v in counts.items() if k.startswith("fail")}
    if bad:
        print(f"  WARNING: {bad}")


# =============================================================================== self-test
def _check(name, cond, detail=""):
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}" + (f"\n         {detail}" if detail else ""))
    assert cond, f"{name}: {detail}"


def _frame_centroid(frames):
    """(n, H, W) stack -> its intensity-weighted (row, col) centroid as fractions of the frame."""
    w = float(frames.sum())
    rr = np.arange(frames.shape[1], dtype=np.float64)
    cc = np.arange(frames.shape[2], dtype=np.float64)
    return (float((frames.sum(2) * rr).sum() / w) / frames.shape[1],
            float((frames.sum(1) * cc).sum() / w) / frames.shape[2])


def _corr(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def self_test():
    print("build_oai_cache self-test\n")
    t0 = time.time()

    # --- the imported contract is still the one this file was written against --------------
    _check("build_cache contract unchanged",
           (CROP_MM, IMG, N_SLICE, len(SLOTS)) == (130.0, 336, 32, 6)
           and SLOTS[0] == "SAG_FLUID_FS" and SLOTS[3] == "SAG_FLUID_NOFS",
           f"CROP_MM={CROP_MM} IMG={IMG} N_SLICE={N_SLICE} SLOTS[0]={SLOTS[0]}")

    # --- knee side: two independent sources, and they must agree ---------------------------
    sides, ks, nboth = read_side_map()
    _check("knee side agrees across subInfo and kneeSideInfo",
           nboth >= 500 and set(sides.values()) == {"R"},
           f"{nboth} ids in both sources, sides present: {sorted(set(sides.values()))}")

    files = sorted(ZIB_IMAGES.glob("*.nii.gz"))
    _check("OAI volumes on disk", len(files) == 507, f"found {len(files)}")

    arr, mm = load_nifti(files[0])
    _check("source geometry is as documented",
           arr.shape == (160, 384, 384) and np.allclose(mm, (0.7, 0.3646, 0.3646), atol=1e-3),
           f"{arr.shape} {tuple(round(x, 4) for x in mm)}")

    # --- shape / dtype / range contract ----------------------------------------------------
    out, n_slab = prepare(arr, mm, "R")
    _check("prepare returns build_cache's frame contract",
           out.shape == (N_SLICE, IMG, IMG) and out.dtype == np.float32
           and out.min() >= 0.0 and out.max() <= 1.0,
           f"{out.shape} {out.dtype} range [{out.min():.3f}, {out.max():.3f}]")
    _check("3.5 mm slabs land exactly on N_SLICE", n_slab == 32, f"n_slab={n_slab}")
    _check("frames are neither blank nor constant",
           float((out[16] > 8 / 255).mean()) > 0.25 and out[16].std() > 0.05,
           f"fill {float((out[16] > 8 / 255).mean()):.3f}, std {out[16].std():.3f}")

    # --- the mm crop, which is what a fixed-pixel resize would silently destroy -------------
    n_pix = int(round(CROP_MM / mm[1]))
    _check("130 mm crop is 357 px of 384 at 0.3646 mm/px, then resampled to 336",
           n_pix == 357 and 16 < n_pix < 384, f"n_pix={n_pix}")
    # A model trained at 0.387 mm/px and run at anything else scores nonsense, so assert the
    # two caches agree rather than trusting they were built the same way.
    _check("output pitch matches the competition cache",
           abs(CROP_MM / IMG - 0.3869) < 1e-4, f"{CROP_MM / IMG:.6f} mm/px")
    # ...and the spacing is actually READ and USED, not ignored in favour of a fixed pixel
    # count. Doubling the header spacing must halve the crop and change every pixel; a
    # fixed-pixel resize would return the identical array and nothing else would notice.
    scaled, _ = prepare(arr, (mm[0], mm[1] * 2, mm[2] * 2), "R")
    _check("the crop is driven by the NIfTI spacing, not a fixed pixel count",
           not np.allclose(scaled, out, atol=1e-3),
           f"at 2x spacing the crop is {int(round(CROP_MM / (mm[1] * 2)))} px not {n_pix}; "
           f"mean abs diff {float(np.abs(scaled - out).mean()):.4f}")

    # --- MIRRORING. All 507 are right knees, so the left branch is dead on real data. Build a
    # left knee by mirroring a real right one about the sagittal midplane -- which for a
    # sagittal stack reverses the slice order and leaves each in-plane frame alone -- and
    # require the pipeline to put it back byte-for-byte. Exact equality, not a correlation:
    # here the expected answer is known.
    left = arr[::-1].copy()
    out_L, _ = prepare(left, mm, "L")
    out_R, _ = prepare(arr, mm, "R")
    _check("a mirrored right knee processed as LEFT lands identical to the right knee",
           np.array_equal(out_L, out_R),
           f"max abs diff {float(np.abs(out_L - out_R).max()):.6f}")
    # ...and the check has teeth: mislabelling the side must NOT give the same answer.
    wrong, _ = prepare(left, mm, "R")
    _check("the side flag actually does something",
           not np.array_equal(wrong, out_R),
           f"max abs diff to the correct output {float(np.abs(wrong - out_R).max()):.4f}")

    # --- MEDIAL/LATERAL DIRECTION against real anatomy. OAIZIB-CM label 4 is medial and 5 is
    # lateral tibial cartilage. Canonical is index 0 = MEDIAL, so the medial centroid must sit
    # at a LOWER slice index than the lateral one. This is the assertion that would have caught
    # the LPS/RAS sign error, and it is ground truth rather than inference.
    lab_files = sorted(ZIB_LABELS.glob("*.nii.gz"))[:8]
    _check("segmentation masks available for the direction check", len(lab_files) == 8,
           f"found {len(lab_files)} in {ZIB_LABELS}")
    med, lat = [], []
    d_row, d_col = [], []
    for f in lab_files:
        v, _vmm = load_nifti(f)
        sid = f.name.replace(".nii.gz", "")
        # --- IN-PLANE ORIENTATION against the same ground truth, on the first three volumes.
        # Everything above pins the SLICE axis; nothing pinned the two axes inside the frame,
        # so `slice.T[::-1, ::-1]` could be replaced by a bare `.T` -- every knee rotated 180
        # degrees, anterior swapped with posterior AND superior with inferior -- and every
        # check still passed. The transform is exercised, not read: label 1 (femoral bone) and
        # label 3 (tibial bone) go through prepare() itself as indicator volumes, and the
        # centroids of what comes out must land where the docstring says the axes point --
        # row index increasing INFERIORLY and column index increasing POSTERIORLY. The femur is
        # above the tibia and the femoral shaft sits in front of the tibial shaft; both hold in
        # 24 of 24 volumes measured, by 0.398-0.454 and 0.032-0.117 of the frame respectively.
        if len(d_row) < 3:
            side = sides.get(sid, "R")
            fem = _frame_centroid(prepare((np.rint(v) == 1).astype(np.float32), _vmm, side)[0])
            tib = _frame_centroid(prepare((np.rint(v) == 3).astype(np.float32), _vmm, side)[0])
            d_row.append(tib[0] - fem[0])    # > 0 iff the femur is SUPERIOR to the tibia
            d_col.append(tib[1] - fem[1])    # > 0 iff the femur is ANTERIOR to the tibia
        if sides.get(sid, "R") != "R":       # same canonicalisation prepare() applies
            v = v[::-1]
        i4, i5 = np.where(np.rint(v) == 4)[0], np.where(np.rint(v) == 5)[0]
        if len(i4) and len(i5):
            med.append(i4.mean() / v.shape[0])
            lat.append(i5.mean() / v.shape[0])
    med, lat = np.array(med), np.array(lat)
    _check("index 0 is MEDIAL: medial tibial cartilage precedes lateral in every volume",
           len(med) >= 6 and bool((med < lat).all()),
           f"n={len(med)}, medial at {med.mean():.3f} vs lateral at {lat.mean():.3f} of the "
           f"stack, {int((med < lat).sum())}/{len(med)} volumes")
    # 0.20 against a measured 0.398-0.454: a vertical flip sends this to about -0.43, and
    # dropping the .T sends it to the A/P separation, which is far under the threshold.
    _check("rows run INFERIORLY: the femur lands above the tibia in the cached frame",
           len(d_row) == 3 and all(d > 0.20 for d in d_row),
           f"tibia centroid minus femur centroid, as a fraction of frame height: "
           f"{[round(d, 3) for d in d_row]} (must exceed +0.20; a flipped row axis gives ~-0.43)")
    # 0.01 against a measured 0.032-0.117. Smaller than the row prior, but it is the only axis
    # a row flip leaves alone, so without it a horizontal mirror passes everything.
    _check("columns run POSTERIORLY: the femoral shaft lands in front of the tibial shaft",
           len(d_col) == 3 and all(d > 0.01 for d in d_col),
           f"tibia centroid minus femur centroid, as a fraction of frame width: "
           f"{[round(d, 3) for d in d_col]} (must exceed +0.01; a flipped column axis "
           f"gives ~-0.06)")

    # --- SLAB AVERAGING IS NOT A NO-OP. The characteristic failure here is a step that runs,
    # changes nothing, and gets reported as working.
    flat, n_flat = prepare(arr, mm, "R", slab_mm=0.0)
    _check("slab averaging changes the pixels",
           n_flat == 160 and not np.allclose(flat, out, atol=1e-3),
           f"n_slab {n_flat} unslabbed vs {n_slab} slabbed, "
           f"mean abs diff {float(np.abs(flat - out).mean()):.4f}")
    # ...but "changed the pixels" is also true of DECIMATION -- keeping one slice out of every
    # five instead of averaging them. That is the live failure mode here, it changes n_slab and
    # the pixels exactly as averaging does, and it throws away the whole point of the step.
    # So assert the thing only averaging can do: an output frame must be the MEAN of the five
    # acquired slices behind it, and must NOT be any one of them. Run one slab group through
    # both ways -- slabbed gives a single frame, slab_mm=0 gives the five slices it spans --
    # and compare with correlation, which is blind to the two calls normalising differently.
    # Decimation puts a 1.000000 in `best_single` and loses to the mean; averaging measures
    # 0.999+ against the mean and 0.94-0.97 against the closest single slice.
    for k in (12, 20):
        five = arr[5 * k:5 * k + 5]
        one, n_one = prepare(five, mm, "R")
        raw5, _ = prepare(five, mm, "R", slab_mm=0.0)
        singles = np.unique(raw5.reshape(len(raw5), -1), axis=0)
        best_single = max(_corr(one[0], s) for s in singles)
        to_mean = _corr(one[0], singles.mean(0))
        _check(f"a slab frame is the MEAN of its five slices, not one of them (group {k})",
               n_one == 1 and len(singles) == 5 and to_mean > 0.99
               and to_mean > best_single + 0.01,
               f"corr(frame, mean of the 5) {to_mean:.6f} vs corr(frame, closest single "
               f"slice) {best_single:.6f}; decimation puts 1.000000 on the right")
    # and it changes them in the right direction: averaging must REDUCE through-plane noise.
    # `flat` is the SAME 32 positions at the SAME 3.5 mm spacing, one raw slice each instead of
    # a 5-slice mean, so anatomy cancels between the two and the gap is noise. The old form of
    # this check asked only for that gap to be positive and called `flat` "raw 0.7 mm frames",
    # which it is not -- it is the decimated stack, i.e. the sabotage itself, which cleared the
    # bar by +0.0004 on chance alone. The 0.7 mm stack the message always claimed belongs here
    # as the SCALE: 32 CONSECUTIVE acquired slices, whose adjacent-frame delta is dominated by
    # single-slice noise. Averaging five of them should remove 1 - 1/sqrt(5) = 55% of that;
    # measured 0.344-0.370 over the first five volumes, so 0.20 is a conservative floor and
    # decimation returns about 0.005.
    c0 = arr.shape[0] // 2
    raw07, _ = prepare(arr[c0 - N_SLICE // 2:c0 + N_SLICE // 2], mm, "R", slab_mm=0.0)
    d_slab = float(np.abs(np.diff(out, axis=0)).mean())
    d_flat = float(np.abs(np.diff(flat, axis=0)).mean())
    d_raw07 = float(np.abs(np.diff(raw07, axis=0)).mean())
    _check("slab averaging removes a real share of the 0.7 mm single-slice noise",
           (d_flat - d_slab) > 0.20 * d_raw07,
           f"mean |delta| between adjacent frames: slabbed {d_slab:.4f}, same positions "
           f"unaveraged {d_flat:.4f}, gap {d_flat - d_slab:.4f} = "
           f"{(d_flat - d_slab) / d_raw07:.3f} x the raw 0.7 mm noise scale {d_raw07:.4f} "
           f"(floor 0.200; decimation gives ~0.005)")

    # --- n_valid HONESTY on a short volume. Truncate to 60 slices -> 12 slabs -> 32 frames
    # holding exactly 12 distinct images, and -- bug B4a -- the repeats must be INTERLEAVED,
    # never a solid block at the tail.
    short = arr[:60]
    s_out, s_nslab = prepare(short, mm, "R")
    keys = [s_out[i].tobytes() for i in range(N_SLICE)]
    uniq = set(keys)
    _check("a 60-slice volume gives 12 slabs and an honest n_valid",
           s_nslab == 12 and min(s_nslab, N_SLICE) == 12 and len(uniq) == 12,
           f"n_slab={s_nslab}, {len(uniq)} distinct frames of {N_SLICE}, "
           f"every real slab survives")
    # The tail-block form of bug B4a is "12 real frames then 20 copies of the last one". It is
    # invisible to shape, dtype and intensity checks, so assert the shape of the repetition
    # itself: the longest run of identical consecutive frames. Interleaved gives 3 here, the
    # tail block would give 21.
    runs = [len(list(g)) for _, g in itertools.groupby(keys)]
    _check("repeats are interleaved, not a tail block", max(runs) <= 4,
           f"longest run of identical consecutive frames {max(runs)} "
           f"(interleaved gives 3; a tail block would give 21)")

    # --- FAT SUPPRESSION, i.e. the slot decision, as a regression guard. Marrow fat is the
    # brightest tissue when it is not suppressed, and the bright-pixel fraction separates the
    # two families cleanly (SAG_FLUID_FS 0.2403 / OAI 0.2444 / SAG_FLUID_NOFS 0.4283).
    bright = float((out > 0.5).mean())
    _check("the DESS looks fat-suppressed, i.e. slot 0 and not slot 3", bright < 0.35,
           f"bright-pixel fraction {bright:.4f} "
           f"(SAG_FLUID_FS 0.240, SAG_FLUID_NOFS 0.428)")

    # --- END TO END through build_one, into a scratch directory -----------------------------
    tmp = ROOT / "audit_out" / "oai_selftest"
    tmp.mkdir(parents=True, exist_ok=True)
    for f in files[:2]:
        sid = f.name.split("_0000")[0]
        for p in (tmp / f"{sid}.npy", tmp / f"{sid}.mask.npy"):
            if p.exists():
                p.unlink()                    # our own scratch output, rewritten every run
        st, _, _ = build_one((sid, str(f), sides[sid], tmp, DESS_SLOT, SLAB_MM))
        assert st == "ok", f"build_one returned {st}"
    sid = files[0].name.split("_0000")[0]
    v = np.load(tmp / f"{sid}.npy")
    mk = np.load(tmp / f"{sid}.mask.npy")
    _check("the on-disk volume matches build_cache's shape and dtype",
           v.shape == (6, N_SLICE, IMG, IMG) and v.dtype == np.uint8, f"{v.shape} {v.dtype}")
    _check("mask is (2,6) float32, one-hot at the DESS slot with n_valid 32",
           mk.shape == (2, 6) and mk.dtype == np.float32
           and mk[0].tolist() == [1., 0., 0., 0., 0., 0.] and mk[1][DESS_SLOT] == 32,
           f"present {mk[0].tolist()}, n_valid {mk[1].tolist()}")
    _check("absent slots hold no data and the present slot does",
           bool(v[DESS_SLOT].any())
           and not v[[k for k in range(6) if k != DESS_SLOT]].any(),
           f"present slot nonzero: {bool(v[DESS_SLOT].any())}")

    # --- --slot MUST APPLY AGAINST A DIRECTORY THAT IS ALREADY FULL. Which of the six slots
    # the DESS occupies is the most consequential choice in this file -- it is what the model
    # believes it is looking at -- and an unconditional resume made the flag a no-op on any
    # second run: every knee returned "skip", the pixels stayed at the old slot, and meta.json
    # was rewritten to claim the new one. Own scratch directory so the volumes above and the
    # render below are left alone, and nothing is deleted: the run ends with the directory at
    # `alt`, so BOTH slot calls below assert the slot that is actually on disk afterwards. That
    # is what makes the check independent of the state the previous run left behind -- an
    # unconditional skip fails the first assertion from one starting state and the second from
    # the other, so neither ordering lets it through.
    tmp_slot = ROOT / "audit_out" / "oai_selftest_slot"
    tmp_slot.mkdir(parents=True, exist_ok=True)
    f0, sid0 = files[0], files[0].name.split("_0000")[0]

    def _built_slots():
        vv = np.load(tmp_slot / f"{sid0}.npy", mmap_mode="r")
        return ([k for k in range(len(SLOTS)) if bool(np.asarray(vv[k]).any())],
                np.load(tmp_slot / f"{sid0}.mask.npy")[0])

    job0 = (sid0, str(f0), sides[sid0], tmp_slot, DESS_SLOT, SLAB_MM)
    st, _, _ = build_one(job0)
    held0, row0 = _built_slots()
    _check(f"a build at slot {DESS_SLOT} leaves the DESS at slot {DESS_SLOT} on disk",
           st in ("ok", "skip") and held0 == [DESS_SLOT] and row0[DESS_SLOT] == 1.0
           and row0.sum() == 1.0,
           f"status {st}, mask row0 {row0.tolist()}, slots holding data {held0}")
    st_again, _, _ = build_one(job0)
    _check("a re-run at the SAME slot still resumes instead of rebuilding",
           st_again == "skip", f"second call at slot {DESS_SLOT} returned {st_again}; if that "
           f"is not 'skip' the resume is gone and a 246-knee build restarts from scratch")
    alt = 3 if DESS_SLOT != 3 else 0
    st_alt, _, _ = build_one((sid0, str(f0), sides[sid0], tmp_slot, alt, SLAB_MM))
    held, row_alt = _built_slots()
    _check(f"--slot {alt} against a directory already built at slot {DESS_SLOT} moves the pixels",
           st_alt == "ok" and held == [alt] and row_alt[alt] == 1.0 and row_alt.sum() == 1.0,
           f"status {st_alt}, mask row0 {row_alt.tolist()}, slots holding data {held} "
           f"(a silent skip leaves [{DESS_SLOT}] here while meta.json claims {alt})")

    # --- LABEL SIDECAR ----------------------------------------------------------------------
    lab = pd.read_csv(ROOT / "oai_zib_labels.csv")
    have = lab[[t for t in TARGETS if t in lab.columns]].notna().any(axis=1)
    tbl = label_table(lab[have])
    _check("label table carries all 12 targets and 12 states for the 246 joined knees",
           len(tbl) == 246
           and all(t in tbl.columns and t + "__state" in tbl.columns for t in TARGETS),
           f"{len(tbl)} rows, {len(tbl.columns)} columns")
    _check("no NaN reaches KneeStudies", not bool(tbl[TARGETS].isna().any().any()))
    _check("MOAKS-unscored targets are masked and scored ones are not",
           (tbl["MCL__state"] == SILENT).all() and (tbl["Fracture__state"] == SILENT).all()
           and int((tbl["Synovitis__state"] == ASSERTED).sum()) == 170
           and int((tbl["Lateral OA__state"] == ASSERTED).sum()) == 246,
           f"asserted cells: Synovitis {int((tbl['Synovitis__state'] == ASSERTED).sum())}, "
           f"Lateral OA {int((tbl['Lateral OA__state'] == ASSERTED).sum())}, "
           f"MCL {int((tbl['MCL__state'] == ASSERTED).sum())}")

    # --- RENDER. A padding bug in this project was caught ONLY by looking at pixels.
    png = ROOT / "audit_out" / "oai_cachecheck.png"
    rows = []
    for f in files[:2]:
        s = f.name.split("_0000")[0]
        vv = np.load(tmp / f"{s}.npy", mmap_mode="r")
        rows.append(np.concatenate([np.asarray(vv[DESS_SLOT, i]) for i in (4, 12, 20, 28)], 1))
    Image.fromarray(np.concatenate(rows, 0)).save(png)
    print(f"\n  rendered {png}")
    print("  LOOK AT IT: frames 4/12/20/28, medial to lateral, marrow dark.")
    print(f"\n  ALL SELF-TESTS PASSED in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()
