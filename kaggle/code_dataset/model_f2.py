"""
model_f2.py -- the dataset for the f2 cache, and the anatomy prior that goes with its rack.

`KneeStudies` cannot read an f2 volume, and the reason is not a missing flag. v1's cache is
`(6, 32, 336, 336)` -- a slot AXIS -- and `__getitem__` indexes it as `vol[s, st:st+group]`.
f2's is `(64, 384, 384)`: FLAT, because the budgets are uneven (18/14/12/8/12) and a
rectangular slot axis would have to pad every slot out to 18, which is 90 frames per study
instead of 64 -- 58 GB instead of 42 GB, and 58 GB does not fit on this disk. The dense
rebuild (`build_cache_f2.py --budget dense`, 26/24/26/16/28) is `(120, 384, 384)`, flat for
the same reason. This loader reads both: the frame layout comes from each study's sidecar,
never from a constant here. So `vol[s]` on an f2 volume is one 384x384 slice, not a slot,
and the loader fails with

    ValueError: could not broadcast input array from shape (3,384) into shape (3,384,384)

which is the best available outcome: a v1 loader cannot quietly read nonsense out of an f2
cache. This file is the loader that can, and it changes nothing else.

WHAT IS CARRIED OVER FROM `KneeStudies`, DELIBERATELY UNCHANGED
  * the emitted 5-tuple `(groups, slot_mask, y, label_mask, weight)` with
    groups `(n_slot, n_group, 3, H, W)` uint8 -- so `train.py`'s loop, `KneeNet.forward`,
    `PerDiagnosisMIL` and `SlotHead` are all untouched;
  * label handling in full: the `silent` / `not_mentioned` mask, the gold override with its
    `gold_weight` multiplier and its all-ones label mask, and the per-target teacher weights;
  * slot dropout with the same guards -- independent per slot, at most `SLOT_DROP_MAX`, never
    the last remaining slot, training only, drawn from the same `rng`;
  * the rng discipline that makes `--seed` mean something: on the TRAIN path the per-study
    Generator is seeded out of numpy's GLOBAL state (not OS entropy), which is what carries
    the seed across the Windows DataLoader worker boundary; on EVAL it is seeded from the
    study id and never read, because `_pick` returns a linspace there;
  * emitting at the CACHE resolution and leaving any resize to `KneeNet.forward(img_size=)`;
  * the `getattr` seatbelts on `slot_idx` / `slot_drop`, for the Windows worker that
    re-imports a changed module against an object the parent pickled.

WHAT IS DELIBERATELY NOT CARRIED OVER, AND WHY
  * `vol.shape[1]` as a single slice count. There is no such number here: each slot owns
    `budget[k]` rows starting at `start[k]`, both read from the sidecar rather than from this
    file's constants, so a cache rebuilt with different budgets stays readable.
  * The v1 sidecar. f2 writes `<study>.meta.npy` `(4,5)` -- present / start / budget /
    distinct-real-slices -- NOT `<study>.mask.npy` `(2,6)`. Different name on purpose: a v1
    loader globbing `*.mask.npy` must not be able to find an f2 sidecar and read a (2,6)
    contract out of a (4,5) file.
  * Photometric augmentation. It never lived in the dataset -- `KneeNet.forward` applies it
    GPU-side on [0,1] pixels before normalisation -- so there is nothing here to carry.

THE ONE HONEST COMPROMISE: `n_group` IS STILL A SCALAR
The budgets are uneven but the emitted tensor must be rectangular, so every slot gets the
same number of windows. With `--n-group 10`, the 18-frame slot draws 10 windows from 16
distinct 3-slice positions while the 8-frame slot draws 10 from 6, so the small slot repeats.
(On the dense layout the 28-frame slot has 26 positions and the 16-frame slot 14, so no slot
needs to repeat a window until `n_group` exceeds 14.)
That is the same behaviour v1 already has on a series shorter than its budget, and it is
what keeps `train.py`'s change to one flag. A per-slot window vector is the better answer
and belongs in its own patch, with its own measurement -- it is not smuggled in here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset

from model import TARGETS, SLOT_DROP_MAX

F2_CACHE = Path(r"data/rsna_knee_cache_f2")              # 64 frames/study
F2_DENSE_CACHE = Path(r"data/rsna_knee_cache_f2_dense")  # 120 frames/study
SLOTS_F2 = ["SAG_FLUID", "SAG_ALT", "COR_FLUID", "COR_ALT", "AX_FLUID"]

# The SAME anatomy as model.SLOT_PRIOR, re-expressed in f2's rack. It is not a re-tuning:
# every target keeps the planes and contrasts it preferred in v1, mapped through
#     v1 0 SAG_FLUID_FS  -> f2 0 SAG_FLUID      v1 3 SAG_FLUID_NOFS -> f2 1 SAG_ALT
#     v1 1 COR_FLUID_FS  -> f2 2 COR_FLUID      v1 4 COR_T1         -> f2 3 COR_ALT
#     v1 2 AX_FLUID_FS   -> f2 4 AX_FLUID       v1 5 SAG_T1         -> f2 1 SAG_ALT
#
# This mapping is why the prior cannot simply be reused. `SlotHead` remaps SLOT_PRIOR through
# `pos = {orig: j for j, orig in enumerate(slot_idx)}`, so running f2 with `--slots 0,1,2,3,4`
# would apply v1's MEANINGS to f2's POSITIONS: f2 slot 1 is SAG_ALT but would inherit
# COR_FLUID_FS's preferences, and f2 slot 3 is COR_ALT but would inherit SAG_FLUID_NOFS's.
# Nothing crashes -- the prior is a 1.7x tilt, not a mask -- so the run would simply be worse
# for a reason no log would show.
SLOT_PRIOR_F2 = {
    "ACL": (0, 1),                      # cruciates are read sagittally, both sagittal slots
    "MCL": (2, 3),                      # collaterals coronally
    "Medial Meniscus": (0, 2, 1),
    "Lateral Meniscus": (0, 2, 1),
    "Medial OA": (2, 3),
    "Lateral OA": (2, 3),
    "PF OA": (4, 0, 1),                 # patellofemoral joint is axial first, then sagittal
    "Effusion": (0, 2, 4),
    "Synovitis": (0, 2, 4),
    "Baker's": (0, 2, 4),
    "Contusion": (0, 2, 4),
    "Fracture": (0, 2, 3),
}


def eval_groups_f2(vol, meta, slot_idx, n_group, group=3):
    """KneeStudiesF2's EVAL windows, for a volume already in memory (the Kaggle notebook).

    (n_slot, n_group, group, H, W) uint8 and the (n_slot,) presence flags -- what
    KneeStudiesF2(train=False)[i] returns as its first two items for the same volume and
    sidecar. A copy of that branch, not a refactor of it: the training __getitem__ stays
    untouched (its rng draws are what --resume replays). test_f2_deploy.py holds the two
    equal on real cache studies.
    """
    present = meta[0].astype(np.float32).copy()
    starts, budgets = meta[1].astype(int), meta[2].astype(int)
    out = np.zeros((len(slot_idx), n_group, group) + tuple(vol.shape[-2:]), np.uint8)
    for j, s in enumerate(slot_idx):
        if present[s] < 0.5:
            continue
        o, b = int(starts[s]), int(budgets[s])
        hi = o + b - group
        for g, st in enumerate(np.linspace(0, max(b - group, 0), n_group).round().astype(int)):
            st = min(o + int(st), max(hi, o))
            out[j, g] = vol[st:st + group]
    return out, present[list(slot_idx)]


class KneeStudiesF2(Dataset):
    """One study -> (groups, slot_mask, labels, label_mask, weight).

    groups: (n_slot, n_group, 3, H, W) uint8 -- identical in shape and meaning to what
            `KneeStudies` emits, so nothing downstream can tell which cache it came from.
    """

    def __init__(self, study_ids, labels: pd.DataFrame, cache=F2_CACHE, n_group=8, group=3,
                 img=384, train=True, gold_ids=None, gold_labels=None,
                 target_weights=None, gold_weight=8.0, slot_idx=None, slot_drop=0.0):
        self.ids = list(study_ids)
        self.cache = Path(cache)
        self.n_group, self.group, self.img, self.train = n_group, group, img, train
        self.slot_drop = float(slot_drop)
        self.slot_idx = list(range(len(SLOTS_F2))) if slot_idx is None else list(slot_idx)
        lab = labels.set_index("StudyInstanceUID")
        self.y = lab[TARGETS].astype(np.float32)
        # Both labellers' spellings, for the same reason model.py accepts both: checking only
        # one silently trains every unmentioned finding as a confident negative.
        NOT_ASSERTED = {"silent", "not_mentioned"}
        state_cols = [t + "__state" for t in TARGETS]
        if all(c in lab.columns for c in state_cols):
            self.silent = lab[state_cols].isin(NOT_ASSERTED).values
        else:
            self.silent = np.zeros((len(lab), len(TARGETS)), bool)
        self.silent = pd.DataFrame(self.silent, index=lab.index, columns=TARGETS)
        self.gold = dict(gold_labels or {})
        self.gold_ids = set(gold_ids or self.gold.keys())
        self.gold_weight = gold_weight
        self.tw = (np.ones(len(TARGETS), np.float32) if target_weights is None
                   else np.asarray(target_weights, np.float32))

    def __len__(self):
        return len(self.ids)

    def _pick(self, n_slice, rng):
        """Start indices for n_group windows of `group` consecutive slices WITHIN one slot.

        Byte-for-byte the policy in `KneeStudies._pick`, with `n_slice` now the slot's own
        budget rather than a per-study constant: stratified on the train path so the whole
        slot is represented every step, a deterministic linspace on the eval path.
        """
        hi = max(n_slice - self.group, 0)
        if not self.train:
            return np.linspace(0, hi, self.n_group).round().astype(int)
        edges = np.linspace(0, hi + 1, self.n_group + 1)
        out = np.empty(self.n_group, dtype=int)
        for i in range(self.n_group):
            lo_i, hi_i = int(np.floor(edges[i])), int(np.ceil(edges[i + 1])) - 1
            hi_i = min(max(hi_i, lo_i), hi)
            out[i] = rng.integers(lo_i, hi_i + 1)
        return out

    def __getitem__(self, i):
        sid = self.ids[i]
        vol = np.load(self.cache / f"{sid}.npy", mmap_mode="r")     # (64 or 120,H,W) uint8
        meta = np.load(self.cache / f"{sid}.meta.npy")              # (4,5) float32
        present = meta[0].astype(np.float32).copy()
        starts, budgets = meta[1].astype(int), meta[2].astype(int)

        # Same seeding contract as KneeStudies: out of numpy's GLOBAL state on the train
        # path, so --seed reaches the window jitter inside every DataLoader worker; from the
        # study id on the eval path, where _pick never reads it.
        rng = np.random.default_rng(np.random.randint(0, 2 ** 31) if self.train
                                    else abs(hash(sid)) % (2 ** 31))

        slot_idx = getattr(self, "slot_idx", None) or list(range(len(starts)))
        cache_px = vol.shape[-1]
        n_slot = len(slot_idx)

        drop_p = getattr(self, "slot_drop", 0.0)
        if self.train and drop_p > 0:
            cand = [s for s in slot_idx if present[s] >= 0.5]
            keep = min(SLOT_DROP_MAX, len(cand) - 1)      # never the last remaining slot
            if keep > 0:
                hit = [s for s in cand if rng.random() < drop_p]
                if len(hit) > keep:
                    hit = rng.choice(hit, size=keep, replace=False).tolist()
                for s in hit:
                    present[s] = 0.0

        out = np.zeros((n_slot, self.n_group, self.group, cache_px, cache_px), np.uint8)
        for j, s in enumerate(slot_idx):
            if present[s] < 0.5:
                continue
            o, b = int(starts[s]), int(budgets[s])
            # Windows are taken INSIDE the slot's own rows. A window must never straddle a
            # slot boundary: rows o+b-1 and o+b are different sequences, often different
            # planes, and stacking them as three channels of one image would hand the
            # backbone a chimera. (The public recipe does let its windows straddle, because
            # it pools globally over the flat volume; our L2 pools per slot, so we cannot.)
            hi = o + b - self.group
            for g, st in enumerate(self._pick(b, rng)):
                st = min(o + int(st), max(hi, o))
                out[j, g] = vol[st:st + self.group]
        present = present[slot_idx]

        y = self.y.loc[sid].values.copy()
        lmask = (~self.silent.loc[sid].values).astype(np.float32)
        w = self.tw.copy()
        if sid in self.gold_ids and sid in self.gold:
            y = self.gold[sid]
            lmask = np.ones_like(lmask)
            w = w * self.gold_weight

        return (torch.from_numpy(out), torch.from_numpy(present),
                torch.from_numpy(y), torch.from_numpy(lmask), torch.from_numpy(w))
