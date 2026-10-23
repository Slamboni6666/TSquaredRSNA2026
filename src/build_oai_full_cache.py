"""
build_oai_full_cache.py -- raw OAI SAG_3D_DESS DICOM series -> the pilot cache contract,
for the full-scale 3DReasonKnee set (7,992 knee-visit volumes, 3,157 knees).

WHY THIS FILE EXISTS
build_oai_cache.py proved the contract on 246 OAI-ZIB knees that arrived as NIfTI. The
full-scale set arrives as raw OAI DICOM, extracted per visit by the download lane into

    <dicom-root>/<anything>/<release>/<participant>/<date>/<series>/*.dcm

so this file re-derives the same volume from DICOM headers instead of a NIfTI affine, and
otherwise changes NOTHING: the output is byte-compatible with the 246-knee pilot cache.

  shape       (6, 32, 336, 336) uint8, DESS at slot 0 (SAG_FLUID_FS -- DECISION 1 in
              build_oai_cache.py, measured not assumed), five slots all zeros.
  mask        (2, 6) float32 -- row 0 slot-present one-hot, row 1 n_valid (real slab count).
  order       slices sorted by k = IPP . n with n = cross(IOP row-dir, IOP col-dir), the
              build_slice_order.py rule; InstanceNumber fallback when geometry is missing.
  direction   canonical index 0 = MEDIAL. The pilot measured (60/60 competition series,
              25/25 OAI segmentations) that ascending k advances toward the patient's
              RIGHT under the standard sagittal IOP. Here that measured property becomes
              an ENFORCED invariant: n is sign-normalised to point patient-right before
              sorting, so a scanner that stored a flipped IOP cannot silently mirror
              medial for lateral. Then build_cache's side rule: LEFT knees reverse the
              slice order. 3,100 of the 7,992 volumes are left knees -- the pilot had
              zero, which is why the left path is a first-class self-test here.
  in-plane    the standard sagittal DICOM frame is already row=INFERIOR, col=POSTERIOR
              (IOP col-dir -z, row-dir +y), which is the house orientation; the IOP signs
              are checked per series and a flipped axis is flipped back, counted, never
              silent.
  slabs       0.7 mm DESS slices averaged into 3.5 mm slabs (5 slices), 160 -> 32 = N_SLICE
              exactly -- build_oai_cache.slab_average, imported.
  crop        130 mm about the image centre from PixelSpacing, per-axis; then 336 px.
  intensity   per-volume 1-99 percentile over the sampled stack, AFTER the crop.
  MONOCHROME1 inverted per-slice, exactly as build_cache.load_slot does it.

  uids        oaifull_{participant}_{visit}_{R|L}, matching oai_full_labels_ladder.csv.
              A built volume without a label row and a label row without a volume are both
              LOGGED in build_report.json; neither is fatal.

HONEST UNKNOWNS THIS FILE DEFENDS AGAINST (validated on the first real participant)
  * laterality at scale: the manifest side (OAI SIDE, 1=R 2=L) is cross-checked against
    every header hint present (SeriesDescription LEFT/RIGHT, Laterality, ImageLaterality).
    Disagreement is a per-volume failure, logged -- mirroring the wrong way swaps medial
    for lateral on five of the twelve targets with no other symptom.
  * multi-volume series: 22 of 7,992 manifest rows say n_volumes=2 (two interleaved
    echoes/acquisitions in one series folder). Slices are grouped by spatial position and
    split by (EchoTime, EchoNumbers, AcquisitionNumber, InstanceNumber); the _N suffix of
    the manifest's image_filepath selects the same sub-volume 3DReasonKnee converted.
  * slice spacing: computed from the sorted k values (median gap), not trusted from
    SliceThickness. The first built volume is strictly validated (~160 slices, ~0.7 mm,
    sagittal normal, ~0.36 mm in-plane) and the run ABORTS on surprise unless --lax.

The two OAI SCREENSHOTS zips on data/oai_screenshots contain PNG renders, not DICOM, so the only
real-data format check they support is the frame count (160) and the LEFT/RIGHT naming;
the self-test reads them in place, read-only, and skips cleanly when that folder is absent.

THIS FILE DELETES NOTHING. It only writes {uid}.npy / {uid}.mask.npy / meta.json /
build_report.json / build_log.csv / oai_full_cache_labels.csv under --out. Archive
deletion belongs to the download lane and its manifest, not here.

CPU-only. Run with CUDA_VISIBLE_DEVICES="" like every preprocessing script.

Usage:
    python build_oai_full_cache.py --self-test
    python build_oai_full_cache.py --bench 4
    python build_oai_full_cache.py --visit 00M --limit 8     # first real participant check
    python build_oai_full_cache.py --visit 00M               # one visit (~2,400 knees)
    python build_oai_full_cache.py                           # everything extracted so far
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image

# Imported, not retyped, so the three caches cannot drift apart. self_test() asserts the
# values are still the ones this file was written against.
from build_cache import CROP_MM, IMG, N_SLICE, SLOTS, sample_indices
from build_oai_cache import DESS_SLOT, SLAB_MM, TARGETS, slab_average

ROOT = Path(__file__).resolve().parent
MANIFEST_DEFAULT = Path(r"data/oai_cache/oai_full_manifest.csv")
LABELS_DEFAULT = Path(r"data/oai_cache/oai_full_labels_ladder.csv")
DICOM_ROOT_DEFAULT = Path(r"data/oai_dicom")
OUT_DEFAULT = Path(r"data/oai_full_cache")
# Discovered by pattern rather than named: the literal filenames embed OAI participant IDs
# (quasi-identifiers), and this file will eventually be tracked in a PUBLIC repo.
SCREENSHOT_ZIPS = sorted(Path(r"data/oai_screenshots").glob("*_SAG_3D_DESS_*_SCREENSHOTS.zip"))

# Header pass tags: geometry + ordering + splitting + laterality. Restricting the read makes
# pydicom stop early on every file; at 160 files/knee x 7,992 knees that is the difference
# between minutes and hours (same trick as build_slice_order.py).
HDR_TAGS = ["ImagePositionPatient", "ImageOrientationPatient", "InstanceNumber",
            "EchoTime", "EchoNumbers", "AcquisitionNumber", "PixelSpacing",
            "SliceThickness", "SpacingBetweenSlices", "Rows", "Columns",
            "SeriesDescription", "Laterality", "ImageLaterality", "NumberOfFrames"]

SIDE_OF = {1: "R", 2: "L", "1": "R", "2": "L"}


class SeriesError(Exception):
    """A per-series refusal with a short status tag, e.g. 'blank' or 'side_mismatch'."""
    def __init__(self, tag, detail=""):
        super().__init__(f"{tag}: {detail}" if detail else tag)
        self.tag = tag


# ================================================================= header pass / ordering
def read_header(path):
    """One DICOM file -> the header dict the ordering and splitting need, pixels untouched."""
    ds = pydicom.dcmread(path, stop_before_pixels=True, force=True, specific_tags=HDR_TAGS)
    h = {"file": os.path.basename(path)}
    ipp = getattr(ds, "ImagePositionPatient", None)
    iop = getattr(ds, "ImageOrientationPatient", None)
    try:
        h["ipp"] = np.array([float(x) for x in ipp]) if ipp is not None and len(ipp) == 3 else None
        h["iop"] = np.array([float(x) for x in iop]) if iop is not None and len(iop) == 6 else None
    except (TypeError, ValueError):
        h["ipp"] = h["iop"] = None
    for k, tag in [("inst", "InstanceNumber"), ("echo_t", "EchoTime"),
                   ("echo_n", "EchoNumbers"), ("acq", "AcquisitionNumber")]:
        v = getattr(ds, tag, None)
        try:
            h[k] = float(v) if v is not None else None
        except (TypeError, ValueError):
            h[k] = None
    ps = getattr(ds, "PixelSpacing", None)
    try:
        h["pix"] = (float(ps[0]), float(ps[1])) if ps is not None and len(ps) == 2 else None
    except (TypeError, ValueError):
        h["pix"] = None
    h["frames"] = int(getattr(ds, "NumberOfFrames", 1) or 1)
    try:
        h["thick"] = float(getattr(ds, "SliceThickness", None))
    except (TypeError, ValueError):
        h["thick"] = None
    # Every laterality hint the header offers. SeriesDescription is the one OAI reliably
    # fills ("SAG_3D_DESS_LEFT"); the two formal tags are collected in case it stops being.
    # NOT \b: underscore is a word character, so \bLEFT\b can never match OAI's
    # "SAG_3D_DESS_LEFT" -- that regex silently disabled the laterality cross-check for
    # every real OAI series (found by the 00M audit; the formal tags are empty there).
    sides = set()
    m = re.search(r"(?<![A-Za-z])(LEFT|RIGHT)(?![A-Za-z])",
                  str(getattr(ds, "SeriesDescription", "")), re.I)
    if m:
        sides.add(m.group(1)[0].upper())
    for tag in ("ImageLaterality", "Laterality"):
        v = str(getattr(ds, tag, "")).strip().upper()
        if v in ("L", "R"):
            sides.add(v)
    h["sides"] = sides
    return h


def canonical_normal(iop):
    """IOP -> (unit normal pointing toward the patient's RIGHT, flipped?, |n_x|).

    build_slice_order sorts by k = p . (row-dir x col-dir) and, measured on the competition
    data, that normal always pointed patient-right for sagittal series (LPS: n_x < 0).
    Enforce the sign instead of trusting it: a scanner that stored flipped direction
    cosines would otherwise reverse the stack and silently swap medial for lateral.
    """
    n = np.cross(iop[:3], iop[3:])
    nn = np.linalg.norm(n)
    if nn < 1e-6:
        raise SeriesError("bad_geometry", "IOP direction cosines are parallel")
    n = n / nn
    flipped = n[0] > 0            # LPS: +x is the patient's LEFT, so n_x>0 points left
    return (-n if flipped else n), flipped, abs(n[0])


def order_series(headers):
    """Header dicts -> (list of ordered index lists, one per sub-volume; info dict).

    Ordering follows build_slice_order.order_series: geometry for the whole series or not
    at all, then InstanceNumber, then filename -- never mixed within a series. On top of
    that, two things the competition path never needed:

      * the sort key is k = IPP . n with n sign-normalised toward patient-right, so
        ascending order ALWAYS runs medial -> lateral for a right knee;
      * slices sharing a spatial position (two echoes -- 22 of 7,992 manifest rows) are
        split into sub-volumes by (EchoTime, EchoNumbers, AcquisitionNumber, Instance),
        each sub-volume keeping the same spatial order.
    """
    info = {"in_flip_rows": False, "in_flip_cols": False, "slice_dir_flipped": False,
            "n_files": len(headers), "n_subvols": 1, "mm_slice": None}
    if any(h["frames"] > 1 for h in headers):
        raise SeriesError("multiframe", "multi-frame DICOM is not a DESS series layout we know")

    if all(h["ipp"] is not None and h["iop"] is not None for h in headers):
        iop = headers[0]["iop"]
        if not all(np.allclose(h["iop"], iop, atol=0.01) for h in headers):
            raise SeriesError("bad_geometry", "IOP varies within the series")
        n_hat, flipped, nx = canonical_normal(iop)
        if nx < 0.8:
            raise SeriesError("not_sagittal", f"|normal_x|={nx:.2f}")
        info["slice_dir_flipped"] = bool(flipped)
        # House in-plane orientation: columns advance POSTERIOR (+y), rows INFERIOR (-z).
        info["in_flip_cols"] = bool(iop[1] < 0)
        info["in_flip_rows"] = bool(iop[5] > 0)
        info["source"] = "geometry"
        ks = np.array([float(h["ipp"] @ n_hat) for h in headers])
    elif all(h["inst"] is not None for h in headers):
        info["source"] = "instance_number"
        ks = np.array([h["inst"] for h in headers], dtype=float)
    else:
        info["source"] = "filename"
        ks = np.arange(len(headers), dtype=float)

    order = np.argsort(ks, kind="stable")

    # Sub-volume split by spatial position. Only meaningful under geometry: identical k
    # (within 0.01 mm) means the same slice location acquired more than once.
    if info["source"] == "geometry":
        pos_key = np.round(ks, 2)
        uniq, counts = np.unique(pos_key, return_counts=True)
        v = int(counts.max())
        if v > 1:
            if not (counts == v).all():
                raise SeriesError("ragged_subvols",
                                  f"positions repeat {counts.min()}..{counts.max()} times")
            info["n_subvols"] = v
            groups = {}
            for j in order:                       # spatial order preserved group by group
                groups.setdefault(pos_key[j], []).append(j)
            for g in groups.values():             # echo order within a position
                g.sort(key=lambda j: (headers[j]["echo_t"] if headers[j]["echo_t"] is not None else 0,
                                      headers[j]["echo_n"] if headers[j]["echo_n"] is not None else 0,
                                      headers[j]["acq"] if headers[j]["acq"] is not None else 0,
                                      headers[j]["inst"] if headers[j]["inst"] is not None else 0))
            ordered_pos = sorted(groups)          # ascending rounded k = canonical order
            subvols = [[groups[p][i] for p in ordered_pos] for i in range(v)]
            gaps = np.diff(sorted(uniq))
        else:
            subvols = [list(order)]
            gaps = np.diff(ks[order])
        if len(gaps):
            info["mm_slice"] = float(np.median(np.abs(gaps)))
    else:
        subvols = [list(order)]
    return subvols, info


# ======================================================================== pixel pipeline
def assemble(sdir, side, subvol_index=1):
    """Series dir -> ((n, H, W) float32 canonical stack, spacing (mm_slice, mm_row, mm_col), info).

    Canonical means: slice index advances toward the patient's RIGHT (medial -> lateral
    for a right knee), rows run inferior, columns run posterior -- whatever the scanner
    stored. `side` is the MANIFEST side; every header laterality hint must agree with it
    or the series is refused, because a silent wrong-way mirror has no other symptom.
    """
    files = sorted(f.name for f in os.scandir(sdir) if f.is_file())
    if len(files) < 8:
        raise SeriesError("too_few_slices", f"{len(files)} files")
    headers = [read_header(os.path.join(sdir, f)) for f in files]

    seen_sides = set().union(*[h["sides"] for h in headers])
    if seen_sides - {side}:
        raise SeriesError("side_mismatch",
                          f"manifest says {side}, headers say {sorted(seen_sides)}")
    subvols, info = order_series(headers)
    info["side_verified"] = bool(seen_sides)

    if subvol_index > info["n_subvols"]:
        if info["n_subvols"] == 1:
            # Measured on the real 00M extraction: ~47% of manifest rows carry a _2 (or
            # higher) suffix in image_filepath while the DICOM series holds exactly ONE
            # 160-slice volume. The suffix in 3DReasonKnee's nii naming is therefore NOT an
            # echo/sub-volume index for these rows -- it is their converter's own output
            # counter. When the physical series is unambiguous (one sub-volume), the suffix
            # carries no routing information and the single volume IS the volume.
            info["suffix_ignored"] = subvol_index
            picked = subvols[0]
        else:
            raise SeriesError("subvol_missing",
                              f"wanted sub-volume {subvol_index} of {info['n_subvols']}")
    else:
        picked = subvols[subvol_index - 1]

    arrs, n_bad = [], 0
    shape_ref, last_err, tsuid = None, None, None
    for j in picked:
        try:
            ds = pydicom.dcmread(os.path.join(sdir, headers[j]["file"]), force=True)
            if tsuid is None:
                tsuid = str(getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", "?"))
            a = ds.pixel_array.astype(np.float32)
            # MONOCHROME1 stores an inverted image -- build_cache.load_slot's rule verbatim.
            if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                a = a.max() - a
        except Exception as e:
            a, last_err = None, f"{type(e).__name__}: {str(e)[:80]}"
        if a is not None and shape_ref is None:
            shape_ref = a.shape[:2]
        arrs.append(a)
    if shape_ref is None:
        # Name the transfer syntax: "no handler" here usually means a compressed syntax
        # this env cannot decode (pylibjpeg/gdcm are not installed), and that is a very
        # different fix from a truncated extraction.
        raise SeriesError("decode", f"no frame decoded (transfer syntax {tsuid}; {last_err})")
    filler = np.zeros(shape_ref, np.float32)
    for i, a in enumerate(arrs):
        if a is None or a.shape[:2] != shape_ref:
            arrs[i] = filler
            n_bad += 1
    # build_cache tolerates a bad frame as a black filler; at OAI scale a series that is
    # >10% filler is a broken extraction, not a knee, and must not enter training quietly.
    if n_bad > 0.1 * len(arrs):
        raise SeriesError("decode", f"{n_bad}/{len(arrs)} frames unreadable")
    info["n_bad_frames"] = n_bad
    vol = np.stack(arrs)

    if info["in_flip_rows"]:
        vol = vol[:, ::-1, :]
    if info["in_flip_cols"]:
        vol = vol[:, :, ::-1]

    if float(vol.max()) <= 0:
        raise SeriesError("blank", "all-zero pixel data")

    pix = next((h["pix"] for h in headers if h["pix"]), None)
    if pix is None:
        raise SeriesError("bad_geometry", "no PixelSpacing anywhere in the series")
    mm_slice = info.get("mm_slice")
    if not mm_slice or mm_slice <= 0:
        # geometry absent (instance/filename ordering): SliceThickness is all there is
        mm_slice = headers[0]["thick"] or 0.7
        info["mm_slice"] = mm_slice
        info["mm_slice_assumed"] = True
    return vol, (float(mm_slice), float(pix[0]), float(pix[1])), info


def prepare_frames(vol, mm, side, slab_mm=SLAB_MM):
    """Canonical stack -> ((N_SLICE, IMG, IMG) float32 in [0,1], n_slab).

    Pure function, separated from IO so the self-test can feed it phantoms. Step order is
    build_oai_cache.prepare's, which is build_cache.load_slot's: mirror, slab, sample,
    crop to a constant physical extent, THEN percentiles over what survived, then resample.
    """
    if vol.ndim != 3:
        raise SeriesError("bad_geometry", f"expected 3D, got {vol.shape}")
    mm_slice, mm_row, mm_col = mm

    # build_cache's side rule for sagittal stacks: medial-lateral is the SLICE axis, so a
    # left knee reverses the stack. The stack already advances patient-right (canonical),
    # which is medial->lateral for a right knee and lateral->medial for a left one.
    if side == "L":
        vol = vol[::-1]
    elif side != "R":
        raise SeriesError("bad_side", repr(side))

    vol, _ = slab_average(vol, mm_slice, slab_mm)
    n_slab = vol.shape[0]

    idx = sample_indices(n_slab, N_SLICE)
    if not idx:
        raise SeriesError("blank", "empty stack")
    vol = vol[idx]

    n_row, n_col = vol.shape[1], vol.shape[2]

    def crop_n(extent_px, mm_px):
        n = int(round(CROP_MM / mm_px)) if mm_px and mm_px > 0 else 0
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
    """(uid, sdir, side, subvol, outdir, slot, slab_mm) -> (status, uid, info, secs).

    Resumable exactly as the pilot is: skip only when the mask sidecar says the volume is
    already present AT THE REQUESTED SLOT (the --slot-is-a-no-op incident, immortalised in
    build_oai_cache.build_one's docstring, applies here unchanged).
    """
    uid, sdir, side, subvol, outdir, slot, slab_mm = job
    dst = outdir / f"{uid}.npy"
    msk = outdir / f"{uid}.mask.npy"
    if dst.exists() and msk.exists():
        try:
            if float(np.load(msk)[0][slot]) == 1.0:
                return "skip", uid, {}, 0.0
        except Exception:
            pass                                  # unreadable sidecar -> rebuild
    t0 = time.time()
    try:
        vol, mm, info = assemble(sdir, side, subvol)
        a, n_slab = prepare_frames(vol, mm, side, slab_mm)
    except SeriesError as e:
        return f"fail:{e.tag}", uid, {"detail": str(e)}, time.time() - t0
    except Exception as e:                        # a crash must not look like a black knee
        return f"fail:{type(e).__name__}", uid, {"detail": str(e)[:120]}, time.time() - t0

    out = np.zeros((len(SLOTS), N_SLICE, IMG, IMG), np.uint8)
    out[slot] = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    mask = np.zeros(len(SLOTS), np.float32)
    n_valid = np.zeros(len(SLOTS), np.float32)
    mask[slot] = 1.0
    n_valid[slot] = min(n_slab, N_SLICE)
    np.save(dst, out)
    np.save(msk, np.stack([mask, n_valid]))
    info = dict(info, n_slab=n_slab, mm_slice=mm[0], mm_row=mm[1], mm_col=mm[2])
    return "ok", uid, info, time.time() - t0


# ==================================================================== manifest / lookup
def index_dicom_root(root):
    """Walk --dicom-root once -> {(participant, date, series): absolute dir}.

    The lane lays series out as <visit>/<release>/<participant>/<date>/<series>/, but only
    the last three components are trusted: participant is a 7-digit 9xxxxxxx id, date is
    8 digits, series is 6+ digits. Keying on that triplet makes the layout above them --
    including whatever the NDA package actually turns out to look like -- irrelevant.
    Series directories are pruned from the walk so their ~160 files are never listed.
    """
    idx, dupes = {}, 0
    root = Path(root)
    if not root.exists():
        return idx, dupes
    for dirpath, dirnames, _ in os.walk(root):
        base = os.path.basename(dirpath)
        parent = os.path.basename(os.path.dirname(dirpath))
        gp = os.path.basename(os.path.dirname(os.path.dirname(dirpath)))
        if (base.isdigit() and len(base) >= 6 and parent.isdigit() and len(parent) == 8
                and gp.isdigit() and len(gp) == 7):
            key = (gp, parent, base)
            if key in idx:
                dupes += 1
            else:
                idx[key] = dirpath
            dirnames[:] = []                      # never descend into the slice files
    return idx, dupes


def parse_manifest_row(r):
    """Manifest row -> (uid, side, (pid, date, series), subvol). Refuses inconsistent rows."""
    uid = r["uid"]
    side = SIDE_OF.get(r["knee_side"])
    if side is None or not uid.endswith("_" + side):
        raise ValueError(f"{uid}: knee_side={r['knee_side']} disagrees with the uid suffix")
    parts = r["oai_dicom_path"].strip("/").split("/")
    pid, date, series = parts[-3], parts[-2], parts[-1]
    m = re.search(r"_(\d+)\.nii\.gz$", r.get("image_filepath", ""))
    subvol = int(m.group(1)) if m else 1
    return uid, side, (pid, date, series), subvol


def scan_built_uids(outdir):
    """Volume files only -- the .mask.npy sidecars match 'oaifull_*.npy' too, and letting
    them through once inflated the join report with phantom '..._R.mask' uids."""
    return {p.name[:-4] for p in Path(outdir).glob("oaifull_*.npy")
            if not p.name.endswith(".mask.npy")}


def merge_build_log(log_rows, path):
    """Write build_log.csv without letting a resume run destroy the build run's evidence.

    A skip row carries no information (no n_files, side_verified, n_subvols, mm_slice).
    The 00M audit found the log reduced to 2,416 bare 'skip' lines: a resume rewrote the
    file after the DICOM was already deleted, so the only per-volume record of how each
    knee was actually built was gone. A skip row therefore never replaces an existing
    informative row; everything else (rebuilds, new failures) overwrites as before."""
    new = pd.DataFrame(log_rows)
    path = Path(path)
    if path.exists() and "status" in new.columns:
        try:
            old = pd.read_csv(path)
            keep = old[(old.status != "skip")
                       & old.uid.isin(set(new.loc[new.status == "skip", "uid"]))]
            if len(keep):
                new = pd.concat([new[~new.uid.isin(set(keep.uid))], keep],
                                ignore_index=True).sort_values("uid", ignore_index=True)
        except Exception:
            pass                                  # an unreadable old log must not stop a build
    new.to_csv(path, index=False)
    return new


def join_report(manifest_uids, label_uids, built_uids):
    """The uid join, as data. Nothing here is fatal; everything here is visible."""
    return {
        "volumes_without_labels": sorted(built_uids - label_uids),
        "labels_without_volume": sorted(label_uids - built_uids),
        "manifest_not_built": len(manifest_uids - built_uids),
        "n_built": len(built_uids), "n_labels": len(label_uids),
    }


# =============================================================================== main
def strict_first_check(info, mm, n_slices, uid, warn_only=False):
    """The first real volume is where the OAI unknowns stop being unknown. Bounds are the
    published DESS-WE geometry; a surprise here means the package structure or protocol is
    not what we believed, and 7,991 more volumes should not be built on that belief.

    warn_only=True (the --lax path, audit 2026-09-07): --lax exists because OAI headers
    carry no laterality tag, but it used to skip ALL of these checks silently -- a 12M/24M
    geometry surprise would never have been seen. Now the list is still computed and
    PRINTED under --lax; only the abort is waived. Returns the problem list."""
    problems = []
    if not (100 <= n_slices <= 200):
        problems.append(f"{n_slices} slices (expected ~160)")
    if not (0.5 <= mm[0] <= 1.0):
        problems.append(f"slice gap {mm[0]:.3f} mm (expected ~0.7)")
    if not (0.25 <= mm[1] <= 0.55 and 0.25 <= mm[2] <= 0.55):
        problems.append(f"in-plane {mm[1]:.4f}/{mm[2]:.4f} mm (expected ~0.3646)")
    if info.get("source") != "geometry":
        problems.append(f"ordered by {info.get('source')} not geometry")
    if not info.get("side_verified"):
        problems.append("no laterality hint in any header -- side rests on the manifest alone")
    if problems:
        msg = (f"STRICT FIRST-VOLUME CHECK FAILED on {uid}:\n  - "
               + "\n  - ".join(problems)
               + "\n(--lax overrides, but read the list first)")
        if warn_only:
            print(f"  LAX MODE, continuing anyway -- {msg}", flush=True)
        else:
            raise SystemExit(msg)
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom-root", default=str(DICOM_ROOT_DEFAULT))
    ap.add_argument("--manifest", default=str(MANIFEST_DEFAULT))
    ap.add_argument("--labels", default=str(LABELS_DEFAULT))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--visit", default="", help="comma list, e.g. 00M,12M; default all")
    ap.add_argument("--slot", type=int, default=DESS_SLOT)
    ap.add_argument("--slab-mm", type=float, default=SLAB_MM)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--lax", action="store_true", help="skip the strict first-volume check")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--bench", type=int, default=0, help="N synthetic full-size knees, report knees/s")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.bench:
        return bench(a.bench, a.threads)

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(a.manifest, dtype=str)
    lab = pd.read_csv(a.labels)
    label_uids = set(lab.StudyInstanceUID)

    visits = [v.strip() for v in a.visit.split(",") if v.strip()]
    if visits:
        man = man[man.time.isin(visits)]

    print(f"indexing {a.dicom_root} ...", flush=True)
    idx, dupes = index_dicom_root(a.dicom_root)
    print(f"  {len(idx)} series directories found" + (f", {dupes} DUPLICATE keys" if dupes else ""))
    if not idx:
        raise SystemExit(f"no series directories under {a.dicom_root} -- nothing extracted yet?")

    jobs, not_found = [], []
    for r in man.to_dict("records"):
        uid, side, key, subvol = parse_manifest_row(r)
        sdir = idx.get(key)
        if sdir is None:
            not_found.append(uid)
        else:
            jobs.append((uid, sdir, side, subvol, outdir, a.slot, a.slab_mm))
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"  {len(jobs)} of {len(man)} manifest rows have an extracted series"
          f" ({len(not_found)} not found -- other visits, usually)")
    if not jobs:
        raise SystemExit("nothing to build")

    (outdir / "meta.json").write_text(json.dumps(dict(
        source="OAI SAG_3D_DESS DICOM (NDA packages), full-scale 3DReasonKnee set",
        slots=SLOTS, dess_slot=a.slot, dess_slot_name=SLOTS[a.slot],
        crop_mm=CROP_MM, img=IMG, n_slice=N_SLICE, slab_mm=a.slab_mm,
        order="k = IPP . n, n sign-normalised toward patient right; index 0 = MEDIAL",
        side="manifest OAI SIDE (1=R,2=L), cross-checked against header laterality; "
             "LEFT knees reverse the slice axis (build_cache's sagittal rule)",
        intensity="per-volume 1-99 percentile over the sampled stack, after the 130 mm crop",
        in_plane="row index increases inferiorly, column index increases posteriorly",
        mask_file="row 0 = slot present (0/1); row 1 = real slab count behind that slot",
        dtype="uint8", shape=[len(SLOTS), N_SLICE, IMG, IMG],
        labels=os.path.basename(a.labels),
    ), indent=2), encoding="utf-8")

    # The first volume runs alone and is strictly validated before the pool opens.
    counts, log_rows = {}, []
    t0 = time.time()
    first_status, first_uid, first_info, first_dt = build_one(jobs[0])
    if first_status == "ok" and not a.lax:
        mm = (first_info["mm_slice"], first_info["mm_row"], first_info["mm_col"])
        strict_first_check(first_info, mm, first_info["n_files"], first_uid)
        print(f"  first volume {first_uid}: {first_info['n_files']} slices @ "
              f"{mm[0]:.3f}/{mm[1]:.4f}/{mm[2]:.4f} mm, source={first_info['source']}, "
              f"side_verified={first_info['side_verified']} -- strict check PASSED", flush=True)
    elif first_status == "ok" and a.lax:
        mm = (first_info["mm_slice"], first_info["mm_row"], first_info["mm_col"])
        probs = strict_first_check(first_info, mm, first_info["n_files"], first_uid,
                                   warn_only=True)
        if not probs:
            print(f"  first volume {first_uid}: strict check passed (running --lax)", flush=True)
    elif first_status.startswith("fail") and not a.lax:
        raise SystemExit(f"first volume {first_uid} failed: {first_status} "
                         f"{first_info.get('detail', '')} -- fix before bulk-building (--lax overrides)")

    results = [(first_status, first_uid, first_info, first_dt)]
    with ThreadPoolExecutor(max_workers=a.threads) as ex:
        for i, res in enumerate(ex.map(build_one, jobs[1:]), 2):
            results.append(res)
            counts[res[0]] = counts.get(res[0], 0) + 1
            if i % 25 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  {i}/{len(jobs)}  {counts}  {el / 60:.1f} min, "
                      f"{i / el:.2f} knees/s, eta {(len(jobs) - i) / max(i / el, 1e-9) / 60:.0f} min",
                      flush=True)
    counts[first_status] = counts.get(first_status, 0) + 1

    for status, uid, info, dt in results:
        log_rows.append(dict(uid=uid, status=status, secs=round(dt, 2),
                             **{k: info.get(k) for k in
                                ("n_files", "n_slab", "source", "mm_slice", "n_subvols",
                                 "side_verified", "slice_dir_flipped", "in_flip_rows",
                                 "in_flip_cols", "n_bad_frames", "detail")}))
    merge_build_log(log_rows, outdir / "build_log.csv")

    built_uids = scan_built_uids(outdir)
    lab[lab.StudyInstanceUID.isin(built_uids)].to_csv(outdir / "oai_full_cache_labels.csv",
                                                      index=False)
    rep = join_report(set(man.uid), label_uids, built_uids)
    rep.update(counts=counts, series_not_found=len(not_found), duplicate_series_keys=dupes,
               elapsed_min=round((time.time() - t0) / 60, 1))
    (outdir / "build_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")

    # Render the first two volumes this run actually built. A padding bug in this project
    # was caught ONLY by looking at pixels, so the real run always leaves some to look at.
    fresh = [uid for status, uid, _, _ in results if status == "ok"][:2]
    if fresh:
        panel = [np.concatenate([np.asarray(np.load(outdir / f"{u}.npy", mmap_mode="r")
                                            [a.slot, i]) for i in (4, 12, 20, 28)], 1)
                 for u in fresh]
        png = outdir / "cachecheck.png"
        Image.fromarray(np.concatenate(panel, 0)).save(png)
        print(f"\n  rendered {png} -- LOOK AT IT: frames 4/12/20/28, medial to lateral, "
              f"marrow dark, femur above tibia")

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min: {counts}")
    print(f"  cache now holds {len(built_uids)} volumes; labels sidecar covers "
          f"{int(lab.StudyInstanceUID.isin(built_uids).sum())} of them")
    if rep["volumes_without_labels"]:
        print(f"  LOGGED: {len(rep['volumes_without_labels'])} volumes without labels")
    if rep["labels_without_volume"]:
        print(f"  LOGGED: {len(rep['labels_without_volume'])} labels without a volume "
              f"(not yet extracted, or failed)")
    bad = {k: v for k, v in counts.items() if k.startswith("fail")}
    if bad:
        print(f"  WARNING: {bad} -- see build_log.csv")


# ===================================================================== phantom factory
# Synthetic DICOM series in the exact OAI layout. Deterministic: same config -> same
# filenames and bytes, so re-running the self-test overwrites rather than accumulates and
# nothing ever needs deleting.
def _phantom_frames(n, rows, cols):
    """Asymmetric on every axis, so any flip or shuffle changes the answer.

    Per canonical slice i: a DC ramp (shuffles change adjacent-frame roughness), a medial
    marker block on slices i < n//5 (side flips move it to the far end), an in-plane block
    in the superior-anterior quadrant (in-plane flips move it), and a diagonal gradient.
    Pixel [0,0] is forced to 0 so MONOCHROME1's per-slice max-inversion is exact.
    """
    rr = np.linspace(0, 1, rows)[:, None]
    cc = np.linspace(0, 1, cols)[None, :]
    frames = []
    for i in range(n):
        # Values are balanced against the 1-99 percentile normalisation: the ramps stay well
        # below the markers so the markers saturate to 1.0 and the ramp background does NOT,
        # keeping the medial-marker ROI separable after clipping.
        f = 200.0 + 10.0 * i + 100.0 * rr + 150.0 * cc          # ramps: slice DC + in-plane
        if i < max(2, n // 5):
            f[rows // 3: rows // 2, cols // 3: cols // 2] += 2000.0   # MEDIAL marker
        f[rows // 8: rows // 4, cols // 8: cols // 4] += 1000.0       # sup-ant in-plane marker
        f[0, 0] = 0.0
        frames.append(f.astype(np.float32))
    return frames


def _write_dcm(path, pixels, ipp, iop, inst, pix=(0.3646, 0.3646), photometric="MONOCHROME2",
               desc=None, laterality=None, echo=None):
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage
    meta = Dataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid(entropy_srcs=[str(path)])
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "MR"
    ds.Rows, ds.Columns = pixels.shape
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.PixelSpacing = [f"{pix[0]:.4f}", f"{pix[1]:.4f}"]
    ds.SliceThickness = "0.7"
    if ipp is not None:
        ds.ImagePositionPatient = [f"{x:.4f}" for x in ipp]
    if iop is not None:
        ds.ImageOrientationPatient = [f"{x:.4f}" for x in iop]
    if inst is not None:
        ds.InstanceNumber = int(inst)
    if desc:
        ds.SeriesDescription = desc
    if laterality:
        ds.ImageLaterality = laterality
    if echo is not None:
        ds.EchoNumbers = int(echo)
        ds.EchoTime = str(5.0 + 4.0 * (echo - 1))
    ds.PixelData = np.clip(pixels, 0, 65535).astype(np.uint16).tobytes()
    ds.save_as(str(path), enforce_file_format=True)


def make_phantom_series(sdir, n=40, rows=96, cols=96, side="R", tag_side=True,
                        desc_side_only=False, shuffle_names=False, strip_geometry=False,
                        iop_variant="standard", photometric="MONOCHROME2", echoes=1,
                        blank=False, seed=0):
    """Fabricate one OAI-shaped DICOM series; returns the dir. Deterministic per config.

    The phantom is DEFINED in canonical coordinates (index 0 = medial, row inferior, col
    posterior) and then STORED the way the requested scanner variant would store it, so
    the pipeline's job -- undoing storage order -- has an exact expected answer.
      side="L":      the true mirror of the right phantom: same in-plane frames, IPP x
                     ascending instead of descending (the knee sits at x<0, mirrored).
      iop_variant="flipped": columns stored advancing ANTERIOR, which also flips the
                     IOP normal to point patient-LEFT; the in-plane and slice-direction
                     canonicalisers must each undo their half exactly.
      strip_geometry: no IPP/IOP at all -> InstanceNumber fallback.
    """
    sdir = Path(sdir)
    sdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + n + rows)
    frames = _phantom_frames(n, rows, cols)
    if blank:
        frames = [np.zeros_like(f) for f in frames]

    names = [f"{rng.integers(0, 16**12):012x}.dcm" for _ in range(n)]
    file_order = list(range(n))                   # file j holds canonical slice file_order[j]
    if shuffle_names:
        rng.shuffle(file_order)

    # desc_side_only is the REAL OAI header shape: the side lives only in the underscore-
    # joined SeriesDescription, Laterality/ImageLaterality both empty.
    desc = (f"SAG_3D_DESS_{'RIGHT' if side == 'R' else 'LEFT'}"
            if (tag_side or desc_side_only) else "SAG_3D_DESS")
    lat = side if (tag_side and not desc_side_only) else None
    x0 = 50.0 if side == "R" else -50.0
    step = -0.7 if side == "R" else 0.7           # canonical index advances patient-right
    written = 0
    for j, ci in enumerate(file_order):
        for e in range(1, echoes + 1):
            f = frames[ci] if e == 1 else frames[ci] * 0.5 + 100.0
            ipp, iop = (x0 + step * ci, -10.0, 20.0), (0, 1, 0, 0, 0, -1)
            if iop_variant == "flipped":
                f = f[:, ::-1]                    # stored with columns advancing anterior
                iop = (0, -1, 0, 0, 0, -1)
            if strip_geometry:
                ipp = iop = None
            name = names[j] if echoes == 1 else f"e{e}_{names[j]}"
            _write_dcm(sdir / name, f, ipp, iop, inst=j + 1, photometric=photometric,
                       desc=desc, laterality=lat, echo=e if echoes > 1 else None)
            written += 1
    n_files = sum(1 for f in os.scandir(sdir) if f.is_file())
    if n_files != written:
        raise RuntimeError(f"{sdir} holds {n_files} files but this run wrote {written}: "
                           f"stale files from an older self-test version -- clean it up "
                           f"manually (this script deletes nothing)")
    return sdir


# =========================================================================== self-test
def _check(name, cond, detail=""):
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}" + (f"\n         {detail}" if detail else ""))
    assert cond, f"{name}: {detail}"


def _order_rho(out):
    """Spearman correlation of the phantom's ramp ROI against frame index.

    The phantom carries a monotone DC ramp along the canonical slice axis, read from the
    bottom-right quadrant where neither marker interferes. A correctly ordered stack gives
    ~+0.99; a scrambled one gives noise around 0. (Slab averaging dilutes an adjacent-frame
    roughness detector below a safe margin -- measured 0.0097 vs 0.0159 -- so the monotone
    ramp is the property the shuffle test leans on instead.)
    """
    from scipy.stats import spearmanr
    tr = out[:, 200:, 200:].mean((1, 2))
    return float(spearmanr(tr, np.arange(len(tr))).statistic)


def _marker_end(out):
    """Which end of the stack the medial marker landed on: 'low' is correct.

    Reads the marker's own ROI. Valid for the 96x96 correctness phantoms, where the 130 mm
    crop never triggers (96 px < 357 px) and the frame maps to 336 px whole, so the marker
    block at rows/cols 32..48 lands at 112..168.
    """
    roi = out[:, 112:168, 112:168].mean((1, 2))
    q = max(1, len(out) // 4)
    lo, hi = roi[:q].mean(), roi[-q:].mean()
    if abs(lo - hi) < 0.05:
        return "ambiguous"
    return "low" if lo > hi else "high"


def self_test():
    print("build_oai_full_cache self-test\n")
    t0 = time.time()
    base = ROOT / "audit_out" / "oai_full_selftest"
    base.mkdir(parents=True, exist_ok=True)

    # --- imported contract unchanged --------------------------------------------------------
    _check("build_cache/build_oai_cache contract unchanged",
           (CROP_MM, IMG, N_SLICE, len(SLOTS), DESS_SLOT, SLAB_MM)
           == (130.0, 336, 32, 6, 0, 3.5) and SLOTS[0] == "SAG_FLUID_FS",
           f"CROP_MM={CROP_MM} IMG={IMG} N_SLICE={N_SLICE} slot={DESS_SLOT} slab={SLAB_MM}")

    # --- ground truth: an ordered right-knee phantom ----------------------------------------
    d_truth = make_phantom_series(base / "truth_R", side="R")
    vol, mm, info = assemble(d_truth, "R")
    _check("phantom assembles by geometry at the written spacing",
           info["source"] == "geometry" and abs(mm[0] - 0.7) < 1e-3
           and abs(mm[1] - 0.3646) < 1e-3 and not info["slice_dir_flipped"],
           f"source={info['source']} mm={tuple(round(x, 4) for x in mm)}")
    truth, n_slab = prepare_frames(vol, mm, "R")
    _check("output contract: (32, 336, 336) float32 in [0,1], 40 slices -> 8 slabs",
           truth.shape == (N_SLICE, IMG, IMG) and truth.dtype == np.float32
           and 0 <= truth.min() and truth.max() <= 1 and n_slab == 8,
           f"{truth.shape} {truth.dtype} n_slab={n_slab}")
    keys = [truth[i].tobytes() for i in range(N_SLICE)]
    import itertools
    runs = [len(list(g)) for _, g in itertools.groupby(keys)]
    # 8 slabs into 32 frames: linspace interleaving gives runs of 4-5; the tail-block form
    # of bug B4a ("8 real frames then 24 copies of the last one") gives a run of 25.
    _check("short-volume repeats are interleaved, never a tail block (bug B4a)",
           len(set(keys)) == 8 and max(runs) <= 6,
           f"{len(set(keys))} distinct frames, longest identical run {max(runs)} "
           f"(interleaved gives 5; a tail block would give 25)")
    _check("medial marker lands at the LOW end of the stack", _marker_end(truth) == "low")

    # --- FAILS ON WRONG SLICE ORDER ---------------------------------------------------------
    # Shuffled filenames + instance numbers but intact geometry: order must be recovered
    # bit-for-bit. Then the same shuffle with geometry stripped: the fallback follows the
    # (shuffled) instance numbers and the roughness detector must catch the scramble.
    d_shuf = make_phantom_series(base / "shuffled_geom", side="R", shuffle_names=True)
    v2, mm2, i2 = assemble(d_shuf, "R")
    out2, _ = prepare_frames(v2, mm2, "R")
    _check("shuffled files with intact geometry reorder to the identical volume",
           np.array_equal(out2, truth), f"max abs diff {np.abs(out2 - truth).max():.6f}")
    d_bad = make_phantom_series(base / "shuffled_blind", side="R", shuffle_names=True,
                                strip_geometry=True)
    v3, mm3, i3 = assemble(d_bad, "R")
    out3, _ = prepare_frames(v3, mm3, "R")
    rho_ok, rho_bad = _order_rho(truth), _order_rho(out3)
    _check("a scrambled stack is DETECTED: the ramp monotonicity detector fires only on it",
           i3["source"] == "instance_number" and rho_ok > 0.9 and abs(rho_bad) < 0.5
           and not np.array_equal(out3, truth),
           f"spearman(ramp, index) ordered {rho_ok:+.3f} vs scrambled {rho_bad:+.3f} "
           f"(measured -0.205 for this seed; correct ordering gives +0.992)")

    # --- FAILS ON A MISLABELLED SIDE --------------------------------------------------------
    # The left phantom is the exact mirror of the right one, so processed as "L" it must
    # come back byte-identical to the right knee -- the untested left path from the pilot
    # verification, now with an exact expected answer.
    d_left = make_phantom_series(base / "truth_L", side="L")
    vL, mmL, iL = assemble(d_left, "L")
    outL, _ = prepare_frames(vL, mmL, "L")
    _check("a LEFT knee mirrors back byte-identical to its right twin",
           np.array_equal(outL, truth), f"max abs diff {np.abs(outL - truth).max():.6f}")
    # Mislabel it with the header hints stripped (so only the phantom's asymmetry can
    # object): the medial marker lands at the WRONG end and the check fails.
    d_left_bare = make_phantom_series(base / "left_untagged", side="L", tag_side=False)
    vB, mmB, iB = assemble(d_left_bare, "R")      # manifest side WRONG, no header to argue
    outB, _ = prepare_frames(vB, mmB, "R")
    _check("a mislabelled side is DETECTED: the medial marker lands at the wrong end",
           _marker_end(outB) == "high" and not np.array_equal(outB, truth),
           f"marker end {_marker_end(outB)} (correct pipeline gives 'low')")
    # With the header hints PRESENT the mislabel never reaches the pixels at all:
    try:
        assemble(d_left, "R")
        caught = None
    except SeriesError as e:
        caught = e.tag
    _check("manifest/header laterality disagreement is refused outright",
           caught == "side_mismatch", f"got {caught}")
    # The real OAI header shape: side ONLY in the underscore-joined SeriesDescription
    # ("SAG_3D_DESS_LEFT"), formal laterality tags empty. \b cannot fire next to an
    # underscore, so the original regex read NO side from every real OAI series and the
    # cross-check above was running on the phantom's ImageLaterality alone.
    d_desc = make_phantom_series(base / "left_desc_only", side="L", desc_side_only=True)
    try:
        assemble(d_desc, "R")
        caught_d = None
    except SeriesError as e:
        caught_d = e.tag
    _, _, i_desc = assemble(d_desc, "L")
    _check("underscore-joined SeriesDescription laterality is read (real OAI shape)",
           caught_d == "side_mismatch" and i_desc["side_verified"],
           f"mislabel got {caught_d}; correct side_verified={i_desc['side_verified']}")

    # --- scanner variants: flipped IOP and MONOCHROME1 must be undone exactly ---------------
    d_flip = make_phantom_series(base / "flipped_iop", side="R", iop_variant="flipped")
    vF, mmF, iF = assemble(d_flip, "R")
    outF, _ = prepare_frames(vF, mmF, "R")
    _check("a flipped-IOP scanner variant canonicalises back byte-identical",
           np.array_equal(outF, truth) and iF["slice_dir_flipped"] and iF["in_flip_cols"],
           f"max abs diff {np.abs(outF - truth).max():.6f}, "
           f"flips slice={iF['slice_dir_flipped']} cols={iF['in_flip_cols']}")
    d_m1 = make_phantom_series(base / "mono1", side="R", photometric="MONOCHROME1")
    # MONOCHROME1 phantoms store inverted pixels; invert at write time by storing M - v.
    for f in os.scandir(d_m1):
        ds = pydicom.dcmread(f.path, force=True)
        px = ds.pixel_array.astype(np.float32)
        ds.PixelData = (4000.0 - px).clip(0, 65535).astype(np.uint16).tobytes()
        ds.save_as(f.path, enforce_file_format=True)
    vM, mmM, _ = assemble(d_m1, "R")
    outM, _ = prepare_frames(vM, mmM, "R")
    _check("MONOCHROME1 inverts back byte-identical",
           np.array_equal(outM, truth), f"max abs diff {np.abs(outM - truth).max():.6f}")

    # --- the crop is driven by PixelSpacing, not a fixed pixel count. At the phantom's 96 px
    # the true 0.3646 mm spacing never triggers the 130 mm crop (needs 357 px), so the check
    # runs at spacings where it does: 1.5 mm crops 87 of 96 px, 3.0 mm crops 43. A pipeline
    # that ignored PixelSpacing would return identical arrays for the two.
    crop_a, _ = prepare_frames(vol, (mm[0], 1.5, 1.5), "R")
    crop_b, _ = prepare_frames(vol, (mm[0], 3.0, 3.0), "R")
    _check("the 130 mm crop is driven by the header spacing, not a fixed pixel count",
           not np.allclose(crop_a, crop_b, atol=1e-3)
           and not np.allclose(crop_a, truth, atol=1e-3),
           f"mean abs diff 1.5mm-vs-3.0mm {np.abs(crop_a - crop_b).mean():.4f}")

    # --- FAILS ON AN ALL-ZERO VOLUME --------------------------------------------------------
    d_blank = make_phantom_series(base / "blank", side="R", blank=True)
    for p in (base / "oaifull_9999001_00M_R.npy", base / "oaifull_9999001_00M_R.mask.npy"):
        if p.exists():
            p.unlink()                            # our own scratch output, rewritten every run
    st, _, inf, _ = build_one(("oaifull_9999001_00M_R", d_blank, "R", 1, base, DESS_SLOT, SLAB_MM))
    _check("an all-zero volume is refused, not written as a black knee",
           st == "fail:blank" and not (base / "oaifull_9999001_00M_R.npy").exists(), st)

    # --- two-echo split: the manifest's _N suffix picks the sub-volume ----------------------
    d_echo = make_phantom_series(base / "two_echo", side="R", echoes=2)
    vE1, mmE1, iE1 = assemble(d_echo, "R", subvol_index=1)
    outE1, _ = prepare_frames(vE1, mmE1, "R")
    vE2, mmE2, _ = assemble(d_echo, "R", subvol_index=2)
    outE2, _ = prepare_frames(vE2, mmE2, "R")
    try:
        assemble(d_echo, "R", subvol_index=3)
        st3 = None
    except SeriesError as e:
        st3 = e.tag
    _check("a two-echo series splits: echo 1 equals truth, echo 2 differs, echo 3 refuses",
           iE1["n_subvols"] == 2 and np.array_equal(outE1, truth)
           and not np.array_equal(outE2, truth) and st3 == "subvol_missing",
           f"n_subvols={iE1['n_subvols']}, echo1 diff {np.abs(outE1 - truth).max():.6f}")

    # --- end to end through build_one: on-disk contract -------------------------------------
    uid = "oaifull_9999002_00M_L"
    for p in (base / f"{uid}.npy", base / f"{uid}.mask.npy"):
        if p.exists():
            p.unlink()                            # our own scratch output, rewritten every run
    st, _, inf, _ = build_one((uid, d_left, "L", 1, base, DESS_SLOT, SLAB_MM))
    v = np.load(base / f"{uid}.npy")
    mk = np.load(base / f"{uid}.mask.npy")
    _check("on-disk volume matches the pilot cache contract",
           st == "ok" and v.shape == (6, N_SLICE, IMG, IMG) and v.dtype == np.uint8
           and mk.shape == (2, 6) and mk.dtype == np.float32
           and mk[0].tolist() == [1., 0., 0., 0., 0., 0.] and mk[1][DESS_SLOT] == 8.0
           and bool(v[DESS_SLOT].any())
           and not v[[k for k in range(6) if k != DESS_SLOT]].any(),
           f"status={st} {v.shape} {v.dtype} mask row0 {mk[0].tolist()} n_valid {mk[1].tolist()}")
    st2, _, _, _ = build_one((uid, d_left, "L", 1, base, DESS_SLOT, SLAB_MM))
    _check("a re-run resumes instead of rebuilding", st2 == "skip", st2)
    scanned = scan_built_uids(base)
    _check("the built-uid scan sees volumes only, never the .mask sidecars",
           uid in scanned and not any(u.endswith(".mask") for u in scanned),
           f"{sorted(scanned)}")
    _check("the cached left knee equals the quantised right truth",
           np.array_equal(v[DESS_SLOT], (np.clip(truth, 0, 1) * 255).astype(np.uint8)))

    # --- a skip resume must not clobber the build run's per-volume log evidence -------------
    lp = base / "selftest_build_log.csv"
    pd.DataFrame([dict(uid="oaifull_1_00M_R", status="ok", n_files=160, n_subvols=1,
                       side_verified=True),
                  dict(uid="oaifull_2_00M_L", status="fail:decode", n_files=12)]
                 ).to_csv(lp, index=False)
    merge_build_log([dict(uid="oaifull_1_00M_R", status="skip"),
                     dict(uid="oaifull_2_00M_L", status="ok", n_files=160)], lp)
    back = pd.read_csv(lp).set_index("uid")
    _check("a skip resume preserves the informative build_log row; a rebuild replaces it",
           back.loc["oaifull_1_00M_R", "status"] == "ok"
           and int(back.loc["oaifull_1_00M_R", "n_files"]) == 160
           and back.loc["oaifull_2_00M_L", "status"] == "ok",
           back.to_string())

    # --- FAILS ON A UID MISMATCH vs the labels ----------------------------------------------
    rep = join_report(manifest_uids={"oaifull_1_00M_R", "oaifull_2_00M_L"},
                      label_uids={"oaifull_1_00M_R", "oaifull_3_00M_R"},
                      built_uids={"oaifull_1_00M_R", "oaifull_2_00M_L"})
    _check("uid mismatches are logged both ways, fatal neither way",
           rep["volumes_without_labels"] == ["oaifull_2_00M_L"]
           and rep["labels_without_volume"] == ["oaifull_3_00M_R"], json.dumps(rep))

    # --- the real label table joins the real manifest 1:1 -----------------------------------
    if MANIFEST_DEFAULT.exists() and LABELS_DEFAULT.exists():
        man = pd.read_csv(MANIFEST_DEFAULT, dtype=str)
        lab = pd.read_csv(LABELS_DEFAULT)
        parsed = [parse_manifest_row(r) for r in man.to_dict("records")]
        _check("manifest parses: 7,992 rows, uid/side/visit all self-consistent",
               len(parsed) == 7992 and len({p[0] for p in parsed}) == 7992)
        _check("manifest and label ladder share exactly the same 7,992 uids",
               set(man.uid) == set(lab.StudyInstanceUID))
        _check("both knee sides are present at scale (3,100 LEFT)",
               sum(1 for p in parsed if p[1] == "L") == 3100
               and sum(1 for p in parsed if p[1] == "R") == 4892)
        _check("the 22 two-volume rows resolve to a sub-volume index",
               sum(1 for p in parsed if p[3] == 2) >= 1
               and all(p[3] in (1, 2) for p in parsed))

    # --- directory index: the resolver finds phantoms laid out the OAI way ------------------
    lane = base / "lane" / "00M" / "0.C.2" / "9999003" / "20040101" / "10111213"
    make_phantom_series(lane, n=10, side="R")
    idx, dupes = index_dicom_root(base / "lane")
    _check("index_dicom_root keys on (participant, date, series) regardless of depth",
           idx.get(("9999003", "20040101", "10111213")) == str(lane) and dupes == 0,
           f"{list(idx)[:2]}")

    # --- real-data hint: the SCREENSHOTS zips (PNG renders, read-only, skipped if absent) ---
    present = [z for z in SCREENSHOT_ZIPS if z.exists()]
    if present:
        import zipfile
        for z in present:
            with zipfile.ZipFile(z) as zf:
                pngs = [n for n in zf.namelist() if n.lower().endswith(".png")]
            side = "L" if "_LEFT_" in z.name else "R"
            _check(f"OAI screenshots {z.name[:7]}... : 160 frames, side {side} in the name",
                   len(pngs) == 160, f"{len(pngs)} PNG frames")
        print("         (PNG renders only -- no DICOM in the zips; real-header validation "
              "happens on the first extracted participant via the strict check)")
    else:
        print("  [skip] SCREENSHOTS zips not reachable (folder absent) -- read-only hint only")

    print(f"\n  ALL SELF-TESTS PASSED in {time.time() - t0:.0f}s")


def bench(n_knees, threads):
    """Throughput on synthetic full-geometry series: 160 x 384 x 384 uint16, the real DESS
    shape. Prints measured knees/s and the projected wall time for a 2,500-volume visit."""
    print(f"bench: {n_knees} synthetic knees at real DESS geometry, {threads} threads")
    base = ROOT / "audit_out" / "oai_full_bench"
    jobs = []
    t0 = time.time()
    for i in range(n_knees):
        sdir = base / "series" / f"999900{i}" / "20040101" / f"1010101{i}"
        if not any(sdir.glob("*.dcm")):
            make_phantom_series(sdir, n=160, rows=384, cols=384, side="R" if i % 2 else "L",
                                seed=i)
        uid = f"oaifull_999900{i}_00M_{'R' if i % 2 else 'L'}"
        out = base / "out"
        out.mkdir(parents=True, exist_ok=True)
        for p in (out / f"{uid}.npy", out / f"{uid}.mask.npy"):
            if p.exists():
                p.unlink()                        # our own bench output, rewritten every run
        jobs.append((uid, sdir, "R" if i % 2 else "L", 1, out, DESS_SLOT, SLAB_MM))
    print(f"  fabricated in {time.time() - t0:.0f}s; building...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=threads) as ex:
        res = list(ex.map(build_one, jobs))
    el = time.time() - t0
    ok = sum(1 for r in res if r[0] == "ok")
    rate = ok / el
    print(f"  {ok}/{n_knees} ok in {el:.1f}s -> {rate:.2f} knees/s")
    print(f"  a 2,500-volume visit at this rate: {2500 / max(rate, 1e-9) / 60:.0f} min "
          f"(target: hours, not days)")
    for r in res:
        if r[0] != "ok":
            print(f"  {r[1]}: {r[0]} {r[2].get('detail', '')}")


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()
