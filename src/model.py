"""
model.py -- the vision model: dataset, architecture, loss. Steps D1-D3.

A study is a BAG. Twelve labels are attached to the study, nothing says which slice or which
sequence carries a finding, and a meniscal tear occupies two or three slices out of ~150. So
the whole design is about not diluting a small focal signal, at three levels:

  L1  within a slice   patch/spatial tokens -> one vector.  Mean over 576 tokens divides a
                       lesion's response by two orders of magnitude, so the upper tail of
                       each channel is kept alongside the mean.
  L2  within a slot    ~10 three-slice groups -> one vector, by GATED ATTENTION (Ilse et
                       al. attention-MIL). This level does not exist in the public
                       baseline, which feeds 3 slices per slot and no aggregation at all --
                       18 slices of the ~150 acquired. It is the largest structural gap we
                       found.
  L3  across slots     6 slot vectors -> 12 logits, each diagnosis attending over slots
                       with absent ones masked out of the softmax. Findings are
                       sequence-specific (cruciates sagittally, collaterals coronally,
                       patellar cartilage axially); pooling slots identically would dilute
                       the one carrying the evidence with five that do not.

The head is deliberately small. With a study-level label there is no signal telling the
model which part of a study matters, so parameters below the slot level would have nothing
to learn from and would spend capacity fitting noise.

CPU-testable: `python model.py --smoke` builds tiny tensors, runs a forward and backward
pass, and overfits a handful of synthetic studies. Shape, mask and loss bugs surface there,
not after an hour of GPU time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# L2 lives in its own module, the way aug.py does, because it carries a self-test that takes
# a few hundred training steps -- far too much to run inside model.py --smoke, and far too
# important to skip. See its docstring for the measurement that motivated the change.
from mil_perdiag import PerDiagnosisMIL, build_qmap

ROOT = Path(__file__).resolve().parent
CACHE = Path(r"data/rsna_knee_cache")
SLOTS = ["SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS", "SAG_FLUID_NOFS", "COR_T1", "SAG_T1"]
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Which slots plausibly show which finding. Used only as a small additive tilt on the L3
# attention logits, never as a hard mask -- exp(0.55) makes a preferred slot ~1.7x the
# weight of an unpreferred one, which biases the softmax without ever excluding a slot.
SLOT_PRIOR = {
    "ACL": (0, 3), "MCL": (1, 4), "Medial Meniscus": (0, 1, 3),
    "Lateral Meniscus": (0, 1, 3), "Medial OA": (1, 4), "Lateral OA": (1, 4),
    # PF OA previously pointed at COR_T1 (index 4), which is anatomically wrong: a coronal
    # sequence shows the tibiofemoral compartments, not the patellofemoral joint. The
    # patella and trochlea are read AXIALLY, and sagittally for the patellar cartilage.
    # PF OA was also the largest remaining gap to the teacher (-0.146).
    "PF OA": (2, 0, 5), "Effusion": (0, 1, 2), "Synovitis": (0, 1, 2),
    "Baker's": (0, 1, 2), "Contusion": (0, 1, 2), "Fracture": (0, 1, 4),
}
SLOT_PRIOR_STRENGTH = 0.55

# Slot dropout never removes more than this many slots from one study, and never the last
# one. Measured over all 4,407 cached studies: 5.1% hold only 2 slots at all, and under
# `--slots 0,1,2` 274 studies hold exactly 1. An uncapped Bernoulli would eventually hand
# the L3 head a study with nothing to attend over -- masked_fill sets every slot to
# finfo.min, the softmax over a constant vector comes back UNIFORM, and the head averages
# six all-zero slot vectors. Nothing crashes and nothing warns; the study just becomes
# label noise carrying a full-weight gradient. Hence the cap and the "never the last" guard.
SLOT_DROP_MAX = 2


# ============================================================================= dataset
class KneeStudies(Dataset):
    """One study -> (groups, slot_mask, labels, label_mask, weight).

    groups: (n_slot, n_group, 3, H, W) uint8 -- each group is 3 adjacent cached slices
            stacked as the three channels a 2D backbone expects.
    """

    def __init__(self, study_ids, labels: pd.DataFrame, cache=CACHE, n_group=8, group=3,
                 img=336, train=True, gold_ids=None, gold_labels=None,
                 target_weights=None, gold_weight=8.0, slot_idx=None, slot_drop=0.0,
                 fill=None, fill_weight=1.0):
        self.ids = list(study_ids)
        self.cache = Path(cache)
        self.n_group, self.group, self.img, self.train = n_group, group, img, train
        # Probability that any ONE present slot is hidden from this epoch's view of this
        # study. 0.0 = off, and off is the default: the gain has not been measured against
        # the 58 gold yet, and nothing unmeasured gets switched on by default here.
        self.slot_drop = float(slot_drop)
        # Which cached slots to actually load. Ablation showed COR_T1 contributes -0.014
        # and SAG_T1 is neutral, so dropping them is both a likely small gain and a third
        # off the compute -- which the efficiency prize prices at 72 s per 0.001 AUC.
        # Slots are dropped at LOAD time, not masked afterwards, so the saving is real.
        self.slot_idx = list(range(len(SLOTS))) if slot_idx is None else list(slot_idx)
        lab = labels.set_index("StudyInstanceUID")
        self.y = lab[TARGETS].astype(np.float32)
        # "silent" cells carry no assertion either way. Forcing them to a number teaches the
        # model to predict the prior; masking them out costs nothing and says the truth.
        # The two labellers name this state differently -- the rule extractor writes
        # "silent", the LLM writes "not_mentioned". Checking only one silently disables
        # masking for the other, which would train every unmentioned finding as a confident
        # negative. Accept both.
        NOT_ASSERTED = {"silent", "not_mentioned"}
        state_cols = [t + "__state" for t in TARGETS]
        if all(c in lab.columns for c in state_cols):
            self.silent = lab[state_cols].isin(NOT_ASSERTED).values
        else:
            self.silent = np.zeros((len(lab), len(TARGETS)), bool)
        self.silent = pd.DataFrame(self.silent, index=lab.index, columns=TARGETS)
        # A dict, not a DataFrame: target names carry spaces and an apostrophe, so
        # attribute access over itertuples silently mangles them.
        self.gold = dict(gold_labels or {})
        self.gold_ids = set(gold_ids or self.gold.keys())
        self.gold_weight = gold_weight
        self.tw = (np.ones(len(TARGETS), np.float32) if target_weights is None
                   else np.asarray(target_weights, np.float32))
        # MASKED-CELL FILL (train.py --fill-masked). `fill` is a frame of cross-fit ensemble
        # probabilities (StudyInstanceUID + TARGETS). For each of THIS dataset's studies that is
        # not gold-overridden, its silent cells take the fill value as target and are unmasked
        # at (per-target weight x fill_weight). Asserted cells are never touched. Only the
        # TRAIN dataset is ever given a fill (build_loaders). sid -> (silent row, fill row);
        # a study without a silent cell has no entry and is served exactly as before.
        self.fill, self.fill_weight = {}, float(fill_weight)
        if fill is not None:
            fv = fill.set_index("StudyInstanceUID")
            for sid in self.ids:
                if sid in self.gold_ids and sid in self.gold:
                    continue
                sil = self.silent.loc[sid].values
                if not sil.any():
                    continue
                if sid not in fv.index:
                    raise ValueError(f"fill table has no row for training study {sid}, "
                                     f"which has masked cells")
                self.fill[sid] = (sil.copy(), fv.loc[sid, TARGETS].values.astype(np.float32))

    def __len__(self):
        return len(self.ids)

    def _pick(self, n_slice, rng):
        """Start indices for n_group windows of `group` consecutive slices.

        STRATIFIED, not uniform-random. Drawing n_group starts independently lets windows
        collide, so coverage saturates well below the stack: at 10 groups of 3 over 32
        slices, independent draws touch only ~20 distinct slices in expectation, and even
        20 groups reach ~27.5. Since v1->v2 showed that the focal findings -- ligaments,
        meniscal horns -- are exactly the ones that improve when coverage rises, those
        collisions are expensive.

        Splitting the stack into n_group equal segments and drawing one window inside each
        guarantees the whole volume is represented every step, while keeping the jitter
        within a segment that makes this an augmentation rather than a fixed crop.
        """
        hi = max(n_slice - self.group, 0)
        if not self.train:
            # deterministic and evenly spread at inference
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
        vol = np.load(self.cache / f"{sid}.npy", mmap_mode="r")     # (6,32,H,W) uint8
        mk = np.load(self.cache / f"{sid}.mask.npy")                # (2,6)
        present = mk[0].astype(np.float32)
        # TRAINING: seed this study's Generator OUT OF numpy's GLOBAL state instead of from
        # OS entropy. `default_rng(None)` ignores np.random.seed() entirely, so a --seed
        # flag without this line would fix the model init and the shuffle order and control
        # NOTHING about which slices get read -- reproducibility that looks real and is not,
        # which is worse than none at all. Measured: with train.py seeded and this line left
        # alone, two --seed 7 runs still ended epoch 3 3.2e-3 apart, against a 4.0e-3 gap
        # between seed 7 and seed 8. The noise was 80% the size of the signal. With this
        # line the two runs agree to all 12 printed decimals.
        #
        # Why the global state reaches the workers: PyTorch draws one base_seed off the
        # parent's global torch generator and each worker then runs
        # np.random.seed(_generate_state(base_seed, worker_id)) before its first __getitem__
        # (torch/utils/data/_utils/worker.py). Windows spawn does not change this -- the
        # seed travels as a constructor argument, not as inherited state -- so seeding torch
        # in the parent is what carries across the process boundary. With --workers 0 there
        # are no workers and the parent's own numpy global state is used, which set_seed()
        # also sets. Either way, this line is the only place the two ever meet.
        #
        # No diversity is lost: the global state advances on every draw, so the windows
        # still differ per study, per slot and per epoch, exactly as before.
        #
        # EVAL: byte-identical. _pick returns a linspace and never touches rng in that
        # branch, so the process-randomised hash() is inert. Left alone deliberately --
        # "fixing" it would change nothing except this file's md5.
        rng = np.random.default_rng(np.random.randint(0, 2 ** 31) if self.train
                                    else abs(hash(sid)) % (2 ** 31))

        # Always emit at the CACHE resolution. Downsampling belongs to the model, which
        # takes img_size and interpolates -- that way every configuration sees the same
        # pixels through a different sampling grid rather than a different crop, and a
        # resolution sweep never needs the cache rebuilt.
        # getattr with a fallback, not self.slot_idx directly: on Windows the DataLoader
        # pickles this object to worker processes that RE-IMPORT the module. If the source
        # changes between the parent creating the dataset and a worker starting, the worker
        # runs new code against an old object and dies with AttributeError -- which is
        # exactly how a 63-minute run was lost on 2026-08-15. The real fix is not to edit
        # source a job is using; this is the seatbelt.
        slot_idx = getattr(self, "slot_idx", None) or list(range(vol.shape[0]))
        n_slice, cache_px = vol.shape[1], vol.shape[-1]
        n_slot = len(slot_idx)

        # SLOT DROPOUT. Hide present slots so the L3 attention cannot settle on
        # SAG_FLUID_FS and stop reading anything else. That slot is present in 93.4% of
        # studies and SLOT_PRIOR already tilts six of the twelve targets towards it, so it
        # is the one the head can most cheaply over-fit to.
        #
        # This is not only a regulariser. Missing sequences are the NORMAL case here:
        # measured over the cache, 6.6% of studies have no SAG_FLUID_FS, 3.6% no
        # COR_FLUID_FS, 6.6% no AX_FLUID_FS, 37.6% no SAG_FLUID_NOFS, 36.4% no COR_T1 and
        # 64.1% no SAG_T1 -- 4.451 of 6 slots on average. At test time the model routinely
        # faces exactly the configuration this simulates, so this trains the distribution
        # it is actually scored on rather than a cleaner one.
        #
        # INDEPENDENT PER SLOT, not "with probability P drop one of them". Under the latter
        # a 2-slot study loses half its evidence at exactly the rate a 6-slot study loses a
        # sixth -- the pressure would be hardest where there is least to spare. Per-slot
        # Bernoulli puts it the right way round: at p=0.15, P(any drop) is 0.28 for a
        # 2-slot study and 0.62 for a 6-slot one.
        #
        # IN THE DATASET, NOT THE MODEL. forward() builds its gather index from slot_mask,
        # so a slot zeroed in either place costs zero backbone passes -- the compute is
        # identical. What only the dataset can skip is the mmap read: 3.4 MB per dropped
        # slot at n_group=10 / 336 px, off the dataloader's critical path. Doing it here
        # also keeps every stochastic choice about a study in one object, under one rng,
        # behind one `self.train` -- rather than adding a second training-only branch inside
        # forward(), where aug.py already had to reach for torch.is_grad_enabled() because
        # net.eval() and no_grad() are not the same question. A slot silently dropped at
        # inference would corrupt every OOF and every submission without warning.
        #
        # getattr, not self.slot_drop: Windows DataLoader workers re-import this module and
        # unpickle a dataset built by the parent. A worker running new code against an
        # object pickled before this attribute existed dies with AttributeError -- the same
        # failure that cost a 63-minute run on 2026-08-15. Same seatbelt as slot_idx above.
        #
        # Drawn from `rng`, so --seed covers it and it adds no new source of randomness.
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
            for g, st in enumerate(self._pick(n_slice, rng)):
                out[j, g] = vol[s, st:st + self.group]
        present = present[slot_idx]

        y = self.y.loc[sid].values.copy()
        lmask = (~self.silent.loc[sid].values).astype(np.float32)
        w = self.tw.copy()
        if sid in self.gold_ids and sid in self.gold:
            # gold is image-derived and severity-thresholded: the only labels read from the
            # images rather than from someone's prose. Trust them, and trust them everywhere.
            y = self.gold[sid]
            lmask = np.ones_like(lmask)
            w = w * self.gold_weight
        elif sid in getattr(self, "fill", {}):
            # the silent cells were the only masked ones, so every cell is now supervised
            sil, fv = self.fill[sid]
            y = np.where(sil, fv, y).astype(np.float32)
            lmask = np.ones_like(lmask)
            w = np.where(sil, w * np.float32(self.fill_weight), w).astype(np.float32)

        return (torch.from_numpy(out), torch.from_numpy(present),
                torch.from_numpy(y), torch.from_numpy(lmask), torch.from_numpy(w))


# ============================================================================= model
# L2 is now mil_perdiag.PerDiagnosisMIL. It is the same Ilse et al. gated attention -- mean
# pooling washes out a lesion that appears in two groups out of ten, max pooling keeps one
# group and throws away corroboration -- with two additions, both default OFF:
#
#   one distribution PER QUERY, so an ACL tear and patellofemoral OA can read different
#     slices of the same sagittal series instead of sharing one compromise;
#   a standardised logit scale, because the shipped version was measured UNIFORM
#     (entropy 0.998 of 1.000 on fold0_aug.pt) -- ||w|| never grew past its init, so the
#     softmax had nothing to sharpen with and L2 was silently doing mean pooling.
#
# At n_query=1, sharpen=False it is this class exactly, parameter names included, so every
# checkpoint written before this change still loads.


class SlotHead(nn.Module):
    """L3. Each diagnosis attends over the slots it can actually be read on."""

    def __init__(self, dim, n_slot, n_out, hidden=256, p=0.2, prior=True, slot_idx=None,
                 qmap=None, prior_map=None):
        super().__init__()
        # Which of L2's Q attention outputs feeds each of the n_out targets. Default: all of
        # them read query 0, which is the single shared attention that has always shipped.
        # persistent=False is load-bearing -- this is a fixed routing table, not learned
        # state, and putting it in the state_dict would make every existing checkpoint fail
        # with "Missing key(s) in state_dict: head.qmap".
        self.register_buffer("qmap",
                             torch.zeros(n_out, dtype=torch.long) if qmap is None
                             else torch.as_tensor(qmap, dtype=torch.long),
                             persistent=False)
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden
        # Remap the anatomy prior onto whichever slots are actually present. Dropping a
        # slot simply removes it from every target's preference list -- which is also how
        # the OA heads stop being pointed at COR_T1, the slot the ablation showed hurts.
        idx = list(range(n_slot)) if slot_idx is None else list(slot_idx)
        pos = {orig: j for j, orig in enumerate(idx)}
        p_ = torch.zeros(n_out, n_slot)
        # `prior_map` names WHICH rack these slot indices refer to. None means the v1 rack
        # and SLOT_PRIOR, which is what every checkpoint on disk was trained under, so the
        # default path is unchanged. The f2 rack passes model_f2.SLOT_PRIOR_F2: its indices
        # mean [SAG_FLUID, SAG_ALT, COR_FLUID, COR_ALT, AX_FLUID], and reusing SLOT_PRIOR
        # there would tilt each target towards the wrong sequence -- silently, because the
        # prior is a 1.7x weight on a softmax and not a mask.
        pm = SLOT_PRIOR if prior_map is None else prior_map
        if prior:
            for t, slots in pm.items():
                for o in slots:
                    if o in pos:
                        p_[TARGETS.index(t), pos[o]] = SLOT_PRIOR_STRENGTH
        self.register_buffer("slot_prior", p_)

    def forward(self, x, mask):                 # (B,S,Q,D) or (B,S,D); mask (B,S)
        # (B,S,D) is still accepted so anything holding one vector per slot -- e.g.
        # slot_ablation.py's cached V -- keeps working unchanged.
        if x.ndim == 3:
            x = x.unsqueeze(2)
        # proj BEFORE routing: at Q=5 this projects 5 vectors per slot instead of 12. It
        # cannot move to the other side of the L2 pooling instead -- proj opens with a
        # LayerNorm, which is per-vector and nonlinear, so LN(sum_g a_g x_g) is not
        # sum_g a_g LN(x_g).
        h = self.proj(x) + self.slot_emb.unsqueeze(1)          # (B,S,Q,H)
        h = h.index_select(2, self.qmap)                       # (B,S,O,H)
        att = torch.einsum("bsoh,oh->bos", h, self.query) / self.hidden ** 0.5
        att = att + self.slot_prior.unsqueeze(0)
        # Absent slots leave the softmax entirely, so a study missing an axial series
        # renormalises onto what it does have rather than attending to a zero vector.
        att = att.masked_fill(mask.unsqueeze(1) < 0.5, torch.finfo(att.dtype).min).softmax(-1)
        ctx = self.drop(torch.einsum("bos,bsoh->boh", att, h))
        return (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias, att


class KneeNet(nn.Module):
    def __init__(self, backbone="convnext_tiny", pretrained=True, n_slot=6,
                 focal_k=8, prior=True, drop=0.2, slot_idx=None, aug=None,
                 mil_queries="shared", mil_sharpen=False, drop_path=0.0,
                 prior_map=None, img_size=None, geo_aug=None):
        super().__init__()
        import timm
        # Stochastic depth. 0.0 (the default) must reproduce today's behaviour exactly, so
        # the kwarg is only passed when set -- an empty dict makes the call byte-identical.
        # DropPath holds no parameters, so checkpoints are unaffected either way.
        bb_kw = {"drop_path_rate": float(drop_path)} if drop_path else {}
        # TRAINING RESOLUTION. `img_size` is the size forward() resizes every slice to. A
        # backbone that declares fixed_input_size (coatnet_rmlp_*, maxvit_*, xcit_*) builds
        # its position tables for one grid, so it is built AT that size when it differs from
        # the native one: coatnet_rmlp_2_rw_224 built natively raises at 288 px ("324 must
        # match 196"); built with img_size=288 it returns a 9x9 grid, and its pretrained
        # weights still load, because rmlp's relative-position MLP does not depend on the
        # grid. Backbones without the flag never receive the kwarg (ConvNeXt's constructor
        # would reject it), and img_size=None keeps this call byte-identical.
        _fixed = timm.get_pretrained_cfg(backbone) if img_size is not None else None
        if (_fixed is not None and _fixed.fixed_input_size
                and int(img_size) != _fixed.input_size[-1]):
            bb_kw["img_size"] = int(img_size)
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0,
                                          global_pool="", **bb_kw)
        self.focal_k = focal_k
        # SHAPE PROBE. A 64 px probe is free for a CNN and FATAL for a backbone that builds
        # position-bias tables for one grid: coatnet_rmlp_2_rw_384 raises
        # "The size of tensor a (16) must match the size of tensor b (576)" here, at
        # CONSTRUCTION time, long before 384 px is ever attempted. timm already states the
        # requirement, so ask it instead of hardcoding 64.
        #
        # What this changes, measured, not assumed:
        #   convnext_small / convnext_tiny / resnet18  fixed_input_size=False -> probe stays
        #     64, token_dim 768 / 768 / 512, byte-identical to before.
        #   xcit_small_12_p16_224  fixed_input_size=True -> probe moves 64 -> 224. Only the
        #     TOKEN COUNT changes (17 -> 197); token_dim is f.shape[-1] = 384 either way, so
        #     the constructed module and every prediction are unchanged. Verified against the
        #     saved fold-0 baseline at max|dp| == 0.
        # The probe runs under no_grad and its output is discarded apart from the width.
        # A backbone built at img_size above is probed at that size: its pretrained_cfg still
        # reports the native size, and probing there is the crash this replaces.
        _cfg = getattr(self.backbone, "pretrained_cfg", {}) or {}
        _probe = bb_kw.get("img_size") or (
            _cfg["input_size"][-1] if _cfg.get("fixed_input_size") else 64)
        with torch.no_grad():
            f = self.backbone(torch.zeros(1, 3, _probe, _probe))
        self.token_dim = f.shape[-1] if f.ndim == 3 else f.shape[1]
        dim = self.token_dim * 3                      # mean + max + focal tail
        self.slot_idx = list(range(n_slot)) if slot_idx is None else list(slot_idx)
        # qmap is built ONCE and handed to both levels, so L2's query count and L3's routing
        # can never disagree -- a mismatch there would index a query that does not exist, or
        # worse, silently feed every target query 0 while L2 trained twelve.
        qmap = build_qmap(mil_queries, TARGETS)
        self.mil = PerDiagnosisMIL(dim, n_query=int(qmap.max()) + 1, sharpen=mil_sharpen)
        self.head = SlotHead(dim, len(self.slot_idx), len(TARGETS), p=drop, prior=prior,
                             slot_idx=self.slot_idx, qmap=qmap, prior_map=prior_map)
        self.aug = aug
        # aug.GeoAug or None (train.py --geo-aug). Not a module: no parameters, no state_dict
        # entry, and knee_net_from_args never passes it, so inference cannot see it.
        self.geo_aug = geo_aug
        # NORMALISATION. The backbone knows what it was pretrained under; asking it is not a
        # new policy, it is the same policy stated once instead of assumed. Every backbone we
        # hold a checkpoint for -- convnext_small, convnext_tiny, resnet18, xcit_small_12 --
        # reports exactly IMAGENET_MEAN/STD, so these buffers are bit-identical and the saved
        # fold-0 baseline reproduces at max|dp| == 0. It matters for the f2 family:
        # coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k was trained at mean=std=(0.5,0.5,0.5), and
        # feeding it ImageNet statistics is a distribution shift at step 0 that nothing warns
        # about. IMAGENET_* stay the fallback for any backbone whose cfg omits them.
        self.register_buffer("mean", torch.tensor(
            tuple(_cfg.get("mean") or IMAGENET_MEAN)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(
            tuple(_cfg.get("std") or IMAGENET_STD)).view(1, 3, 1, 1))

    def _tokens(self, x):
        f = self.backbone(x)
        if f.ndim == 4:                               # CNN: (B,C,H,W) -> (B,HW,C)
            f = f.flatten(2).transpose(1, 2)
        return f

    def _pool(self, f):
        """L1. Mean, max, and the top-k tail per channel.

        A finding occupies a small part of the field, so a plain mean over the tokens
        dilutes it by two orders of magnitude. The tail is taken per channel rather than by
        selecting whole tokens, which keeps the strongest responses wherever they occur.
        """
        k = max(1, f.shape[1] // self.focal_k)
        return torch.cat([f.mean(1), f.amax(1), f.topk(k, dim=1).values.mean(1)], dim=-1)

    def forward(self, groups, slot_mask, img_size=None):
        """Encode only the slots that are actually present.

        A study holds on average 4.45 of the 6 slots, so encoding all six spends **26% of
        every backbone pass on all-zero tensors** whose output the L3 masked softmax then
        discards. Gathering the present (study, slot) pairs, encoding those, and scattering
        the results back is mathematically identical -- absent slots contribute nothing to
        the softmax either way -- and removes that quarter of the work on GPU and CPU alike.

        It matters most exactly where we are weakest: on a session without a usable GPU,
        26% off the dominant cost is the difference between a comfortable run and a tight
        one.
        """
        B, S, G, C, H, W = groups.shape
        present = (slot_mask.reshape(B * S) > 0.5)
        idx = present.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:                       # nothing to read; keep shapes valid
            idx = torch.zeros(1, dtype=torch.long, device=groups.device)

        sel = groups.reshape(B * S, G, C, H, W).index_select(0, idx)
        x = sel.reshape(-1, C, H, W).float().div_(255.0)
        if img_size is not None and img_size != H:
            # antialias only when DOWNscaling (e.g. the 384 px f2 cache read at 224/288):
            # plain bilinear samples 4 pixels per output and aliases at 384 -> 224. Upscaling
            # keeps antialias=False, so every checkpoint on disk (336 -> 336 or 336 -> 384)
            # takes exactly the call it was trained with.
            x = F.interpolate(x, size=(img_size, img_size), mode="bilinear",
                              align_corners=False, antialias=img_size < H)
        # Geometric augmentation, training only, BEFORE the photometric one (positioning comes
        # before the scanner's intensity profile, and padded corners then take that profile
        # like any other background). x holds each instance's G windows as G contiguous rows,
        # so viewing it as (n, G*C, h, w) puts one (study, slot) per row and GeoAug's single
        # grid per row moves every window and slice channel of it identically. The plane comes
        # from the cache slot, self.slot_idx[idx % S]. getattr: the same seatbelt as slot_idx.
        geo = getattr(self, "geo_aug", None)
        if geo is not None and self.training:
            h, w = x.shape[-2:]
            slot = torch.as_tensor(self.slot_idx, device=x.device)[idx % S]
            x = geo(x.reshape(idx.numel(), G * C, h, w), slot).reshape(-1, C, h, w)
        # Photometric augmentation: on [0,1] pixels, BEFORE the ImageNet normalisation, and
        # only while training. One intensity profile per STUDY, since that is what a scanner
        # setting is. `idx` indexes flattened (study, slot) pairs and each contributes G
        # rows, so idx // S recovers the study and repeat_interleave(G) expands it to rows.
        if self.aug is not None and self.training:
            x = self.aug(x, torch.div(idx, S, rounding_mode="floor").repeat_interleave(G))
        x = (x - self.mean) / self.std
        emb = self._pool(self._tokens(x)).reshape(idx.numel(), G, -1)
        vec_present, att_present = self.mil(emb)   # (n,Q,D), (n,G,Q)

        # (B,S,Q,D) is 8*6*12*2304 = 1.33M elements = 2.65 MB in bf16 -- a 12x widening of
        # something small, since D is 2304 but B*S is only 48. The backbone's activations for
        # 8*4.45*10 = 356 images at 336 px dominate it by three orders of magnitude, so there
        # is nothing to gain from deferring this to L3's consumption point.
        Q, D = vec_present.shape[-2], vec_present.shape[-1]
        slot_vec = torch.zeros(B * S, Q, D, device=vec_present.device,
                               dtype=vec_present.dtype)
        slot_vec.index_copy_(0, idx, vec_present)
        group_att = torch.zeros(B * S, G, Q, device=att_present.device,
                                dtype=att_present.dtype)
        group_att.index_copy_(0, idx, att_present)

        # group_att gains a trailing query axis: (B,S,G) -> (B,S,G,Q). Nothing in the repo
        # consumes it today, but the entropy probe that found L2 was uniform does, and it
        # now has to be read per diagnosis to mean anything.
        logits, slot_att = self.head(slot_vec.reshape(B, S, Q, D), slot_mask)
        return logits, dict(group_att=group_att.reshape(B, S, G, Q), slot_att=slot_att)


def knee_net_from_args(a, pretrained=False):
    """The KneeNet a checkpoint was trained as, rebuilt from its saved args (vars(args)).

    One constructor for everything that loads a checkpoint for inference -- the Kaggle
    notebook and package_for_kaggle.verify_loadable -- so the module tree cannot drift from
    training. Each argument is one train.py passes to KneeNet that changes either the module
    tree (slots, mil_queries, mil_sharpen, a fixed-size backbone's img) or the anatomy prior
    (cache_format). For a v1 checkpoint trained with the defaults this is exactly the old
    KneeNet(backbone, pretrained=False, slot_idx) call: ConvNeXt ignores img, "shared"/False
    are the L2 defaults and prior_map stays None. dropout and drop_path are left at their
    defaults: neither holds parameters and both are identity in eval.
    """
    sl = a.get("slots")
    sidx = [int(x) for x in str(sl).split(",")] if sl else None
    prior_map = None
    if (a.get("cache_format") or "v1") == "f2":
        from model_f2 import SLOT_PRIOR_F2
        prior_map = SLOT_PRIOR_F2
    return KneeNet(backbone=a["backbone"], pretrained=pretrained, slot_idx=sidx,
                   mil_queries=a.get("mil_queries") or "shared",
                   mil_sharpen=bool(a.get("mil_sharpen")), prior_map=prior_map,
                   img_size=a.get("img"))


# ============================================================================= loss
def distill_loss(logits, y, label_mask, weight):
    """Soft-target cross-entropy, masked and per-target weighted.

    The targets are GRADED, not binary -- a severity ladder from the report. Binarising at a
    midpoint before the loss throws away the teacher's uncertainty, which is the most
    valuable thing it produces, so the soft value is used directly.

    No pos_weight anywhere. Class rebalancing moves a decision threshold, and the metric is
    AUC, which has none. It would cost compute and buy nothing.
    """
    per = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    m = label_mask * weight
    return (per * m).sum() / m.sum().clamp_min(1e-6)


# ============================================================================= any-of loss
# "AT LEAST ONE OF THESE HEADS". MRNet's meniscus label says a meniscal tear is present but
# not on which side, so it cannot supervise Medial Meniscus or Lateral Meniscus alone: writing
# it into both heads invents a tear on the intact side about half the time. What it does say
# is P(any) = 1 - (1 - p_MM)(1 - p_LM), a noisy-OR, and that is the quantity it trains. A
# NEGATIVE exam says neither meniscus is torn, which is two ordinary per-head negatives, so
# those rows stay ordinary masked cells (ext_aux.py, "THE ANY-OF TARGET").
#
# The gradient on head j is p_j (P - y) / P: a positive row pushes up whichever head the
# model already finds more likely (its own lateralisation, learned from the lateralised
# competition labels, assigns the credit), and with one head at p = 0 the term is plain BCE
# on the other.
ANY_OF = {"Any Meniscus": ("Medial Meniscus", "Lateral Meniscus")}


def any_of_terms(logits, y, weight, idx):
    """-> (numerator, denominator) of the noisy-OR BCE over the rows that carry an any-of
    target for the heads `idx`.

    ENCODING (ext_aux.ExtStudies is its only producer): on such a row the heads' own cells
    are masked (label_mask 0, so distill_loss never sees them), y[:, idx[0]] holds the any-of
    target and weight[:, idx[0]] holds MINUS its weight. Every other source emits weights
    > 0, so the sign cannot collide, and label_mask * weight = 0 keeps distill_loss blind to it.

    log(1 - P) = -sum_j softplus(z_j) exactly, so log P = log(-expm1(-s)) is stable at any
    logit a model produces. The clamp at float32 `tiny` only guards s == 0 (every logit
    below about -87), where the backward pass would otherwise compute inf * 0.
    """
    k = idx[0]
    on = weight[:, k] < 0
    s = F.softplus(logits[:, list(idx)]).sum(1)
    log_p = torch.log(-torch.expm1(-s.clamp_min(torch.finfo(s.dtype).tiny)))
    per = -(y[:, k] * log_p - (1.0 - y[:, k]) * s)
    wa = torch.where(on, -weight[:, k], torch.zeros_like(weight[:, k]))
    return (torch.where(on, per, torch.zeros_like(per)) * wa).sum(), wa.sum()


def distill_loss_any(logits, y, label_mask, weight, idx):
    """distill_loss plus the any-of term under ONE normaliser: an any-of row counts as one
    cell carrying its own weight. A batch without any-of rows gives distill_loss bit for bit
    (x + 0.0 == x), so switching this in only changes batches that hold such rows."""
    per = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    m = label_mask * weight
    num, den = any_of_terms(logits, y, weight, idx)
    return ((per * m).sum() + num) / (m.sum() + den).clamp_min(1e-6)


# ============================================================================= smoke test
def smoke():
    """CPU-only. Catches shape, mask and loss bugs before any GPU time is spent."""
    print("SMOKE TEST (CPU)\n" + "=" * 60)
    torch.manual_seed(0)
    B, S, G, C, H = 2, 6, 3, 3, 96

    net = KneeNet(backbone="resnet18", pretrained=False, focal_k=8)
    print(f"  backbone token dim {net.token_dim}, params "
          f"{sum(p.numel() for p in net.parameters()) / 1e6:.1f}M")

    groups = torch.randint(0, 255, (B, S, G, C, H, H), dtype=torch.uint8)
    mask = torch.tensor([[1., 1., 1., 1., 1., 0.], [1., 1., 1., 0., 0., 0.]])
    logits, aux = net(groups, mask)
    assert logits.shape == (B, len(TARGETS)), logits.shape
    print(f"  forward OK -> logits {tuple(logits.shape)}")

    # absent slots must receive exactly zero attention, or the mask is decorative
    sa = aux["slot_att"]
    leaked = (sa * (mask < 0.5).unsqueeze(1)).abs().max().item()
    assert leaked < 1e-6, f"absent slot got attention {leaked}"
    print(f"  masked softmax OK (attention on absent slots = {leaked:.2e})")
    rowsum = sa.sum(-1)
    assert torch.allclose(rowsum, torch.ones_like(rowsum), atol=1e-4)
    print("  attention rows sum to 1")

    y = torch.rand(B, len(TARGETS))
    lm = torch.ones(B, len(TARGETS))
    lm[0, 3] = 0
    w = torch.ones(B, len(TARGETS))
    loss = distill_loss(logits, y, lm, w)
    loss.backward()
    ng = sum(1 for p in net.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  backward OK, loss {loss.item():.4f}, {ng} tensors received gradient")

    # a masked cell must not influence the loss at all
    l2 = distill_loss(logits.detach(), y, lm, w)
    y2 = y.clone(); y2[0, 3] = 1 - y2[0, 3]
    l3 = distill_loss(logits.detach(), y2, lm, w)
    assert abs(l2 - l3) < 1e-6, "masked label changed the loss"
    print("  masking OK (flipping a masked target does not move the loss)")

    # it must be able to memorise a tiny set -- if it cannot overfit 4 studies, it will
    # never learn 4,407, and the bug is here rather than in the data
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4)
    gx = torch.randint(0, 255, (4, S, G, C, 64, 64), dtype=torch.uint8)
    gm = torch.ones(4, S)
    gy = (torch.rand(4, len(TARGETS)) > 0.5).float()
    gl = torch.ones(4, len(TARGETS))
    gw = torch.ones(4, len(TARGETS))
    first = None
    for step in range(40):
        opt.zero_grad()
        lo, _ = net(gx, gm)
        l = distill_loss(lo, gy, gl, gw)
        l.backward()
        opt.step()
        if step == 0:
            first = l.item()
    print(f"  overfit 4 studies: loss {first:.4f} -> {l.item():.4f}")
    assert l.item() < first * 0.5, "model cannot memorise 4 studies -- something is wrong"
    print("\n  ALL SMOKE CHECKS PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--check-data", action="store_true", help="load 2 real studies")
    a = ap.parse_args()
    if a.smoke:
        smoke()
    if a.check_data:
        print("\nREAL DATA CHECK\n" + "=" * 60)
        folds = pd.read_csv(ROOT / "folds.csv")
        labels = pd.read_csv(ROOT / "report_labels_ours.csv")
        ds = KneeStudies(folds.StudyInstanceUID[:2], labels, n_group=4, img=336, train=True)
        g, m, y, lm, w = ds[0]
        print(f"  groups {tuple(g.shape)} {g.dtype}   slot_mask {m.tolist()}")
        print(f"  labels {y.numpy().round(2).tolist()}")
        print(f"  label_mask (0 = silent, masked out) {lm.numpy().astype(int).tolist()}")
        print(f"  bytes per study in a batch: {g.numel() / 1e6:.1f} MB")


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()
