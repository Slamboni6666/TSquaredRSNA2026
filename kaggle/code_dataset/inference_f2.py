"""
inference_f2.py -- the f2 family at scoring time: test DICOM -> the bytes the f2 cache holds.

inference.py does this for v1; this file does it for build_cache_f2.py's cache, and changes
nothing in either. It holds no image rule of its own. Every decision that sets a pixel --
the slot ladder and its tie-breaks, geometric slice order, the 6-94% band with numpy's
round-half-even, MONOCHROME1 inversion, the 140 mm crop with zero padding, the 0.5/99.5
percentiles over distinct slices inside the valid crop, PIL BILINEAR on float32, the
truncating uint8 cast, mirroring after selection, the (4,5) sidecar -- is the function the
cache was built with, imported:

    series table   inference.build_index          the v1 header pass, run once per split
    S1 flags       build_series_index_v2.derive   fluid_v2 / fatsat_v2
    slot choice    build_cache_f2.select_slots    ladder, exclusivity, deterministic ranking
    slice order    build_slice_order.order_series geometry, as slice_order.parquet was built
    pixels         build_cache_f2.assemble_study  the same function build_study saves

What is restated here is only glue: series_index_v2.parquet is derive() over
build_series_index.py's header table, and build_index rebuilds that table for an unseen
split; slice_order.parquet is order_series() per series, called here per chosen series.

`--verify` re-derives training studies from raw DICOM through exactly this path -- header
pass included, nothing read from the parquets -- and demands the cache's bytes back.

Usage:
    python inference_f2.py --verify --n 60
    python inference_f2.py --verify --n 30 --threads 1      # also times one study at a time
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from build_cache_f2 import SLOTS, assemble_study, select_slots          # noqa: E402
from build_series_index_v2 import derive                                # noqa: E402
from build_slice_order import order_series                              # noqa: E402

DENSE_CACHE = Path(r"data/rsna_knee_cache_f2_dense")


def build_index_f2(idx):
    """inference.build_index's series table -> one select_slots row per (study, f2 slot)."""
    return select_slots(derive(idx.copy()))


def preprocess_study_f2(study, rows, split_root, side, budget):
    """One study's select_slots rows -> ((sum(budget), 384, 384) uint8, (4, 5) float32).

    `budget` is the checkpoint's own per-slot layout (train.py saves it as args f2_budget),
    so a dense member gets 26/24/26/16/28 = 120 frames and a 64-frame member 18/14/12/8/12.
    """
    order_map = {}
    for r in rows.itertuples():
        sdir = Path(split_root) / study / r.SeriesInstanceUID
        ordered = order_series((study, r.SeriesInstanceUID, str(sdir)))
        if ordered:
            order_map[r.SeriesInstanceUID] = [d["sop"] for d in
                                              sorted(ordered, key=lambda d: d["order"])]
    return assemble_study(study, rows, side, order_map, split_root, budget=tuple(budget))


# ================================================================================ verify
def _pick_studies(n, cache, seed=0):
    """A sample weighted towards the cases that break a builder: left knees, unresolved
    laterality, missing slots, short series (repeated frames), ladder fallbacks, padded
    crops -- then topped up at random. Returns (ids, {category: [ids]})."""
    si = pd.read_parquet(ROOT / "series_index_v2.parquet")
    st = pd.read_parquet(ROOT / "study_index.parquet")
    side = dict(zip(st.StudyInstanceUID, st.side))
    sel = select_slots(si)
    sel = sel.merge(si[["SeriesInstanceUID", "rows", "cols"]], on="SeriesInstanceUID")
    n_pix = (140.0 / pd.to_numeric(sel.pix_row, errors="coerce")).round()
    sel["padded"] = (n_pix > sel.rows) | (n_pix > sel.cols)
    metas = {p.name[:-len(".meta.npy")]: np.load(p) for p in cache.glob("*.meta.npy")}
    ids = sorted(metas)
    cats = {
        "left knee": [s for s in ids if side.get(s) == "L"],
        "laterality unresolved": [s for s in ids if side.get(s) not in ("L", "R")],
        "missing slot": [s for s in ids if metas[s][0].min() == 0],
        "short series": [s for s in ids
                         if ((metas[s][0] == 1) & (metas[s][3] < metas[s][2])).any()],
        "ladder fallback": sorted(set(sel[sel.tier > 0].StudyInstanceUID) & set(ids)),
        "padded crop": sorted(set(sel[sel.padded].StudyInstanceUID) & set(ids)),
    }
    rng = np.random.default_rng(seed)
    want = max(n // 8, 1)
    pick = []
    for c, pool in cats.items():
        pool = [s for s in pool if s not in pick]
        pick += list(rng.choice(pool, min(want, len(pool)), replace=False))
    rest = [s for s in ids if s not in pick]
    pick += list(rng.choice(rest, max(n - len(pick), 0), replace=False))
    return pick, {c: [s for s in pick if s in set(v)] for c, v in cats.items()}


def verify(n=60, cache=DENSE_CACHE, threads=8, seed=0):
    from inference import build_index
    cache = Path(cache)
    budget = tuple(json.loads((cache / "meta.json").read_text(encoding="utf-8"))["budget"])
    print(f"VERIFY f2 test-time builder vs {cache}  (budget {budget}, numpy "
          f"{np.__version__})\n" + "=" * 78)
    ids, cats = _pick_studies(n, cache, seed)
    for c, v in cats.items():
        print(f"  {c:<22} {len(v):>3} of the {len(ids)} studies")

    # The header pass over exactly these studies' series, as the notebook runs it over
    # test_series.csv. Every rule downstream is per study, so a subset changes nothing.
    tser = pd.read_csv(ROOT / "train_series.csv")
    tmp = Path(tempfile.mkdtemp(prefix="verify_f2_"))
    sub_csv = tmp / "series.csv"
    tser[tser.StudyInstanceUID.isin(ids)].to_csv(sub_csv, index=False)
    t0 = time.time()
    idx, side_map = build_index(ROOT / "train_series", sub_csv, threads=threads)
    sel = build_index_f2(idx)
    t_index = time.time() - t0

    # The choice the cache was built from, for the same studies.
    ref = select_slots(pd.read_parquet(ROOT / "series_index_v2.parquet"))
    ref = ref[ref.StudyInstanceUID.isin(ids)]
    key = ["StudyInstanceUID", "slot", "SeriesInstanceUID", "tier"]
    same_sel = (sel[key].sort_values(key).reset_index(drop=True)
                .equals(ref[key].sort_values(key).reset_index(drop=True)))
    st = pd.read_parquet(ROOT / "study_index.parquet").set_index("StudyInstanceUID").side
    side_ok = sum((side_map.get(s) if side_map.get(s) in ("L", "R") else None)
                  == (st[s] if st[s] in ("L", "R") else None) for s in ids)
    print(f"\n  header pass + select_slots over {len(idx)} series: {t_index:.1f}s")
    print(f"  slot choice identical to series_index_v2.parquet: {same_sel}  "
          f"({len(sel)} slot rows)")
    print(f"  laterality identical to study_index.parquet:      {side_ok}/{len(ids)}\n")

    def one(sid):
        t = time.time()
        vol, meta = preprocess_study_f2(sid, sel[sel.StudyInstanceUID == sid],
                                        ROOT / "train_series", side_map.get(sid), budget)
        dt = time.time() - t
        want = np.load(cache / f"{sid}.npy")
        want_meta = np.load(cache / f"{sid}.meta.npy")
        d = np.abs(want.astype(np.int16) - vol.astype(np.int16))
        bad = [SLOTS[k] for k in range(len(SLOTS))
               if d[int(want_meta[1, k]):int(want_meta[1, k] + want_meta[2, k])].max() > 0]
        return sid, dt, (vol.shape == want.shape and d.max() == 0
                         and np.array_equal(meta, want_meta)), int((d > 0).sum()), \
            int(d.max()), bad, bool(np.array_equal(meta, want_meta))

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=threads) as ex:
        res = list(ex.map(one, ids))
    wall = time.time() - t0
    n_ok = 0
    for sid, dt, ok, n_px, mx, bad, meta_ok in res:
        n_ok += ok
        tags = [c for c, v in cats.items() if sid in v]
        if not ok:
            print(f"  [ FAIL ] {sid[-14:]}  {n_px} px differ (max {mx}), sidecar "
                  f"{'ok' if meta_ok else 'DIFFERS'}, slots {bad}  {tags}")
    per = [r[1] for r in res]
    print(f"  {n_ok}/{len(ids)} studies byte-identical (volume and sidecar)   "
          f"{wall:.0f}s wall on {threads} thread(s), median {np.median(per):.2f}s "
          f"per study build (order + decode + resize)")
    ok = same_sel and side_ok == len(ids) and n_ok == len(ids)
    print("  CONTRACT HOLDS: scoring-time f2 volumes are the cache's bytes." if ok else
          "  MISMATCH: the f2 test-time path does NOT reproduce the cache. Do not ship it.")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--cache", default=str(DENSE_CACHE))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.verify:
        return 0 if verify(a.n, a.cache, a.threads, a.seed) else 1
    ap.print_help()
    return 0


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    sys.exit(main())
