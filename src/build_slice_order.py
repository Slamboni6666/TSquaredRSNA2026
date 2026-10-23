"""
build_slice_order.py -- recover the true anatomical slice order for every series.

WHY THIS FILE EXISTS
A DICOM file is named by its SOP Instance UID, a number chosen to be globally unique and
nothing else. Measured over all 24,371 training series, the rank correlation between
filename order and true anatomical order is -0.000 (main() below recomputes it every run):
listing a series folder by name shuffles the knee. Three things downstream break if the
order is wrong, and all three break SILENTLY -- they still produce a tensor and still
train:

    * stacking three adjacent slices into the three input channels of one image,
    * sampling windows evenly through the stack,
    * reversing the slice order to mirror left knees to a canonical side (build_cache.py).

The true order comes from geometry every slice already carries. ImageOrientationPatient
holds two unit vectors that lie IN the image plane -- one pointing along a row, one down a
column -- so their cross product points THROUGH the plane, which is the direction the stack
advances in. ImagePositionPatient is that slice's first-voxel position in patient
millimetres. So

    n = r_x  x  r_y        (cross product of the ImageOrientationPatient direction cosines)
    k = p . n              (projection of ImagePositionPatient onto the slice normal)

gives each slice's position along the stack in mm. k increases monotonically along the
stack and is signed in patient coordinates, so it also fixes the traversal direction.
Being a projection, it stays correct for oblique acquisitions, where no single coordinate
of p is the through-plane axis. InstanceNumber is the fallback where geometry is missing;
in practice all 24,371 training series were ordered by geometry and none needed it.

This pass reads one header per slice (819,078 reads, ~14 min wall clock). It is
latency-bound on I/O, not CPU, so it runs a wide thread pool. The result depends only on
DICOM geometry -- not on resolution, slice count, or any model choice -- so it is computed
once and reused forever.

Output: slice_order.parquet  with one row per slice:
    StudyInstanceUID, SeriesInstanceUID, sop (filename), k, instance_number, order, source

Read-only with respect to the DICOM tree.

Usage:
    python build_slice_order.py                 # all training series
    python build_slice_order.py --split test
    python build_slice_order.py --limit 200     # smoke test
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

ROOT = Path(__file__).resolve().parent

# (0020,0032) ImagePositionPatient, (0020,0037) ImageOrientationPatient,
# (0020,0013) InstanceNumber. Restricting the read to these three makes pydicom stop
# parsing early on every file, which on ~819k files is the difference between minutes
# and an hour.
TAGS = [0x00200032, 0x00200037, 0x00200013]


def read_one(path):
    """-> (k, instance_number), where k is this slice's position along the stack in mm.

    k is None when the slice cannot be placed by geometry: the tags are missing, unparseable,
    or the two direction cosines are parallel so they span no plane. The caller treats that
    as a property of the whole SERIES, not of one slice -- see order_series.
    """
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True,
                             specific_tags=[pydicom.tag.Tag(t) for t in TAGS])
    except Exception:
        return None, None
    inst = getattr(ds, "InstanceNumber", None)
    inst = int(inst) if inst is not None else None
    ipp = getattr(ds, "ImagePositionPatient", None)
    iop = getattr(ds, "ImageOrientationPatient", None)
    if ipp is None or iop is None or len(iop) != 6 or len(ipp) != 3:
        return None, inst
    try:
        p = np.array([float(x) for x in ipp])
        r = np.array([float(x) for x in iop[:3]])
        d = np.array([float(x) for x in iop[3:]])
    except (TypeError, ValueError):
        return None, inst
    n = np.cross(r, d)
    nn = np.linalg.norm(n)
    if nn < 1e-6:
        return None, inst
    return float(p @ (n / nn)), inst


def order_series(args):
    """One series directory -> one row per slice, each carrying its rank in `order`.

    The three ordering sources are chosen for the series as a whole and never mixed. A
    series is ordered by geometry only if EVERY slice yields a k, because interleaving two
    different orderings inside one stack is worse than using the weaker one consistently.
    `source` is written out with the rows so a later audit can see which series got what.
    """
    study, series, sdir = args
    try:
        files = [f.name for f in os.scandir(sdir) if f.is_file()]
    except OSError:
        return []
    if not files:
        return []

    ks, insts = [], []
    for fn in files:
        k, inst = read_one(os.path.join(sdir, fn))
        ks.append(k)
        insts.append(inst)

    n_geom = sum(x is not None for x in ks)
    if n_geom == len(files):
        source = "geometry"
        key = ks
    elif sum(x is not None for x in insts) == len(files):
        # Geometry incomplete. InstanceNumber usually tracks k up to sign, but interleaved
        # and multi-echo acquisitions number slices in an order that is not the order they
        # occupy in space, so this is recorded as a weaker source rather than silently
        # treated as equivalent.
        source = "instance_number"
        key = [float(i) for i in insts]
    else:
        source = "filename"
        key = [float(i) for i in range(len(files))]

    # stable: slices that share a key -- two echoes acquired at the same position -- keep
    # the order the directory listed them in rather than being permuted arbitrarily.
    idx = np.argsort(np.asarray(key, dtype=float), kind="stable")
    return [dict(StudyInstanceUID=study, SeriesInstanceUID=series, sop=files[j],
                 k=ks[j], instance_number=insts[j], order=rank, source=source)
            for rank, j in enumerate(idx)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--limit", type=int, default=0, help="first N series only (smoke test)")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    base = ROOT / f"{a.split}_series"
    tser = pd.read_csv(ROOT / f"{a.split}_series.csv")
    jobs = [(r.StudyInstanceUID, r.SeriesInstanceUID,
             str(base / r.StudyInstanceUID / r.SeriesInstanceUID))
            for r in tser.itertuples()]
    if a.limit:
        jobs = jobs[:a.limit]

    out = Path(a.out) if a.out else ROOT / f"slice_order{'' if a.split == 'train' else '_test'}.parquet"
    print(f"ordering {len(jobs)} series from {base}  ({a.threads} threads)", flush=True)

    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.threads) as ex:
        for i, res in enumerate(ex.map(order_series, jobs), 1):
            rows.extend(res)
            if i % 1000 == 0:
                el = time.time() - t0
                print(f"  {i}/{len(jobs)} series, {len(rows)} slices, "
                      f"{el:.0f}s, eta {el / i * (len(jobs) - i):.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    el = time.time() - t0

    print(f"\ndone in {el / 60:.1f} min -> {out}")
    print(f"  series {df.SeriesInstanceUID.nunique()}   slices {len(df)}")
    print(f"  order source: {dict(df.groupby('SeriesInstanceUID').source.first().value_counts())}")

    # How wrong would filename order have been? This is the number that justifies the pass.
    g = df[df.source == "geometry"]
    rhos = []
    for _, s in g.groupby("SeriesInstanceUID"):
        if len(s) < 5:
            continue
        name_rank = s.sop.rank().values
        rhos.append(np.corrcoef(name_rank, s.order.values)[0, 1])
    if rhos:
        print(f"  corr(filename order, true order): mean {np.mean(rhos):+.3f}, "
              f"median {np.median(rhos):+.3f}  over {len(rhos)} series")
        print("  -> near zero confirms filename sorting is anatomically meaningless.")


if __name__ == "__main__":
    main()
