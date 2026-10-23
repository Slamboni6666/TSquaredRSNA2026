"""test_full_data.py -- --exclude-gold and the full-data fold (--fold 5..7). CPU, seconds.

Synthetic ids and labels only (build_loaders reads no pixels until iterated), so this needs no
competition data and writes nothing.

  (a) --exclude-gold drops exactly the gold studies from TRAINING and empties the gold label
      override in both datasets; a held-out fold keeps its own validation set, gold included.
  (b) flag off: training ids and the gold override are exactly what they always were.
  (c) a fold no study carries (folds.csv uses 0-4) trains on every study with an empty
      validation set, and predict() on that empty loader returns a (0, 12) array instead of
      raising on torch.cat([]). Under --exclude-gold it trains on every NON-gold study and
      its validation set is the gold studies, in folds.csv order.

Usage: python test_full_data.py      exit 0 = all PASS, 1 = any FAIL
"""
import os
import sys
from argparse import Namespace
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import train as T  # noqa: E402

IDS = [f"s{i:02d}" for i in range(20)]
FOLDS = pd.DataFrame({"StudyInstanceUID": IDS, "fold": [i % 5 for i in range(20)]})
GOLD = {s: np.ones(len(T.TARGETS), np.float32) for s in ("s00", "s03", "s07", "s12")}
LABELS = pd.DataFrame({"StudyInstanceUID": IDS, **{t: 0.5 for t in T.TARGETS}})


def loaders(fold, exclude):
    a = Namespace(slots=None, cache="unused", n_group=2, img=96, slot_drop=0.0, bs=2,
                  workers=0, cache_format="v1", oai_weight=0.0, gold_weight=8.0,
                  exclude_gold=exclude)
    tr, va, va_ids = T.build_loaders(fold, LABELS, FOLDS, GOLD, a)
    return tr.dataset, va, va_ids


def main():
    res = []

    def check(label, ok, why=""):
        res.append(ok)
        print(f"  {'PASS' if ok else 'FAIL'}  {label}{(': ' + why) if why else ''}", flush=True)

    # (a) fold 0 with --exclude-gold
    tr, va, va_ids = loaders(0, True)
    want_tr = [s for s in IDS if FOLDS.set_index("StudyInstanceUID").fold[s] != 0
               and s not in GOLD]
    check("(a) --exclude-gold fold 0: training = non-held-out minus gold",
          tr.ids == want_tr, f"{len(tr.ids)} train ids, gold among them "
                             f"{sum(s in GOLD for s in tr.ids)}")
    check("(a) --exclude-gold: override emptied in train and val datasets",
          tr.gold == {} and tr.gold_ids == set() and va.dataset.gold == {}
          and va.dataset.gold_ids == set())
    check("(a) --exclude-gold: held-out fold keeps its gold",
          va_ids == [s for s in IDS if s in ("s00", "s05", "s10", "s15")] and "s00" in va_ids)

    # (b) flag off, fold 0: unchanged
    tr, va, va_ids = loaders(0, False)
    check("(b) flag off: training = every non-held-out study, gold included",
          tr.ids == [s for s in IDS if FOLDS.set_index("StudyInstanceUID").fold[s] != 0])
    check("(b) flag off: gold override intact",
          set(tr.gold) == set(GOLD) and tr.gold_ids == set(GOLD))

    # (c) the full-data fold
    tr, va_g, va_ids_g = loaders(5, True)
    check("(c) fold 5 --exclude-gold: trains on the 16 non-gold, validates on the 4 gold",
          tr.ids == [s for s in IDS if s not in GOLD] and va_ids_g == ["s00", "s03", "s07", "s12"]
          and va_g.dataset.ids == va_ids_g and tr.gold == {} and va_g.dataset.gold == {},
          f"train {len(tr.ids)}, validation {va_ids_g}")
    tr, va, va_ids = loaders(5, False)
    check("(c) fold 5: trains on all 20, holds none out",
          tr.ids == IDS and len(va.dataset) == 0 and va_ids == [])

    class Never(torch.nn.Module):
        def forward(self, *a, **k):
            raise AssertionError("forward must not run on an empty loader")

    p = T.predict(Never(), va, torch.device("cpu"), None)
    check("(c) predict() on the empty loader", isinstance(p, np.ndarray)
          and p.shape == (0, len(T.TARGETS)) and p.dtype == np.float32, f"shape {p.shape}")
    out = pd.DataFrame(p, columns=T.TARGETS)
    out.insert(0, "StudyInstanceUID", va_ids)
    check("(c) run_fold's OOF frame builds from it", out.shape == (0, 13))

    n_fail = res.count(False)
    print(f"\n{'ALL PASS' if not n_fail else f'{n_fail} FAILED'} ({len(res) - n_fail}/{len(res)})")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
