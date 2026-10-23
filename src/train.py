"""
train.py -- step D4: train the vision model on report-derived labels, validate honestly.

Three things here are not defaults, and each follows from something measured earlier:

  PER-TARGET TEACHER WEIGHTING. The labels are not equally trustworthy. Measured against
  the 58 gold studies our labeller scores 0.930 on ACL and 0.599 on Synovitis -- the latter
  is barely above chance, because radiologists see synovitis and do not write it down. A
  head trained equally hard on both is being taught noise on one of them. Each target's
  distillation loss is scaled by (auc - 0.5), so a teacher that knows nothing pulls almost
  nothing, and that head leans on gold instead.

  DISCRIMINATIVE LEARNING RATES. The head is random at initialisation and has everything to
  learn; the backbone starts from a good solution and needs only to be moved off it. One
  learning rate either leaves the head untrained or destroys the backbone in the first few
  hundred steps.

  GROUPED VALIDATION, REPORTED THREE WAYS. Gold AUC on the held-out fold (real labels, ~11
  studies, uselessly noisy alone), agreement with the teacher across the whole fold (large
  n, but measures imitation rather than correctness), and the grouped-vs-random gap (how
  much of the score is scanner memorisation). No one of them can carry a decision.

Usage:
    python train.py --smoke                       # CPU, 6 studies, verifies the loop
    python train.py --fold 0 --epochs 8
    python train.py --all-folds                   # full OOF
    python train.py --all-folds --resume          # same command after a kill: continue mid-fold
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, get_worker_info

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from model import (KneeStudies, KneeNet, distill_loss, distill_loss_any, TARGETS, CACHE,  # noqa: E402
                   SLOTS)
from rank_loss import pairwise_rank_loss  # noqa: E402
from noise_loss import gce_loss  # noqa: E402
from oai_aux import attach_oai  # noqa: E402
from ext_aux import attach_ext  # noqa: E402

# Qwen3-32B's per-target gold AUC, measured on all 58. (The rule extractor's
# numbers were here before; it now scores 0.788 macro against Qwen's 0.861.)
# Our labeller's per-target gold AUC. This IS the teacher's reliability, measured, not
# assumed. Recomputed whenever the labeller changes -- see build_report_labels.py output.
TEACHER_AUC = {
    "ACL": 0.982, "MCL": 0.968, "Medial Meniscus": 0.919, "Lateral Meniscus": 0.856,
    "Medial OA": 0.929, "Lateral OA": 0.801, "PF OA": 0.875, "Effusion": 0.810,
    "Synovitis": 0.652, "Baker's": 0.927, "Contusion": 0.783, "Fracture": 0.826,
}


def teacher_weights(floor=0.15):
    """(auc - 0.5), normalised so the best target weighs 1.

    A floor keeps the weakest head learning something rather than being switched off; at
    0.599 Synovitis lands near it, which is the intent.
    """
    w = np.array([max(TEACHER_AUC[t] - 0.5, 0.0) for t in TARGETS], np.float32)
    w = w / w.max()
    return np.maximum(w, floor)


def target_weights(args):
    """The per-target weight vector of the COMPETITION rows (--target-weights).

    Omitted or 'teacher': teacher_weights(), exactly as before. 'uniform': 1.0 for every
    target -- for a label table that is not the one TEACHER_AUC was measured on (e.g. a mix
    of readers), where those reliabilities no longer describe the labels. Gold rows still
    multiply this by --gold-weight; OAI / ext rows never see it (attach_oai takes no tw).
    """
    if getattr(args, "target_weights", None) == "uniform":
        return np.ones(len(TARGETS), np.float32)
    return teacher_weights()


def set_seed(seed):
    """Every source of randomness in a run, in one call. Missing any one makes --seed a lie.

      torch global CPU generator -- backbone and head init, Dropout, the shuffle order
        (RandomSampler with generator=None draws its own seed off it), and the DataLoader's
        base_seed for the workers.
      torch CUDA generators -- Dropout and PhotometricAug on device. torch.manual_seed seeds
        every device, so the one call covers them.
      every DataLoader worker -- python-random, torch and numpy-GLOBAL, all derived from
        base_seed above (torch/utils/data/_utils/worker.py). Windows spawn does not change
        that: base_seed travels as an argument, not as inherited state. This is only worth
        anything because KneeStudies now draws its window jitter out of the numpy global
        state -- see the comment in KneeStudies.__getitem__.
      numpy global state in the PARENT -- which is what --workers 0 uses instead of workers.

    DELIBERATELY ABSENT: torch.use_deterministic_algorithms(True) and cudnn.deterministic.
    Several conv backward kernels have no deterministic implementation and would simply
    raise, and the ones that do cost throughput. Bitwise GPU reproducibility is not what
    this flag is for -- being able to say "these two runs differ only in the seed" is. On
    --device cpu the run IS bitwise reproducible (measured: two runs agreed on step loss to
    all 12 printed decimals); on CUDA the residual cuDNN jitter is far below the
    seed-to-seed spread.

    ALSO ABSENT: PYTHONHASHSEED. KneeStudies hashes the study id only on the eval path,
    where _pick returns a linspace and never reads the rng, so process hash randomisation
    cannot reach any number this pipeline produces.
    """
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


# ============================================================================= mid-fold resume
# A 48-epoch fold is ~4 h on a GPU whose owner can reclaim it at any moment, and oof_parts/
# only saves a fold once it is FINISHED. So every epoch writes a sidecar, and --resume picks
# a killed fold up from it. The bar is that the split run IS the uninterrupted run -- on CPU,
# bit for bit -- and restoring weights, AdamW, OneCycle and the four RNG states is not enough
# to clear it, because of one thing set_seed's notes do not cover:
#
#   PERSISTENT WORKERS ARE NEVER RE-SEEDED. build_loaders runs persistent_workers=True, so
#   torch seeds each worker once, from the base_seed it draws when the fold's first epoch
#   starts, and from then on the worker's numpy global state -- the slice-window jitter --
#   is wherever the epochs before left it (torch 2.9 _MultiProcessingDataLoaderIter._reset
#   sends _ResumeIteration and nothing else). oai_aux's exposure rotation lives in the same
#   place for the same reason (its DECISION 5). A resumed process starts fresh workers, from
#   a base_seed drawn at the wrong moment, so both have to be put back explicitly:
#
#     base_seed      the train loader is given a generator holding the torch state the
#                    original drew it from, so torch seeds the new workers identically.
#     numpy global   every training __getitem__ makes EXACTLY ONE global draw, the
#                    np.random.randint(0, 2 ** 31) that seeds its local Generator (model.py
#                    KneeStudies, model_f2.py KneeStudiesF2; OAIStudies inherits it) -- so a
#                    worker's state is a function of base_seed and how many samples it has
#                    served, and replaying that many draws restores it.
#     OAI rotation   a pure function of (frac_seed, worker, OAI rows served), replayed the
#                    same way through OAIStudies._next_row.
#     ext2 rotation  ext_aux's second source, the same mechanism (engaged only at
#                    --ext2-frac < 1). The train set is [competition | OAI | ext2 | ext3], so
#                    each sample's source is its index range and the rotations are counted
#                    and replayed apart.
#     ext3 rotation  ext_aux's third source (MRNet), the same again (--ext3-frac < 1, which
#                    its default 0.3 is).
#
#   Both counts come from _Tap: batch j of every epoch goes to worker j % workers (in-order
#   round robin, restarted at worker 0 each epoch), so the sampler's own index stream says
#   exactly who served what. With --workers 0 there are no workers: the parent's numpy state
#   IS the jitter stream and is restored directly, and only the OAI rotation is replayed.
#
# ADD A SECOND GLOBAL DRAW TO A DATASET'S __getitem__ AND RESUME STOPS BEING EXACT -- quietly,
# because nothing crashes; the resumed run is merely a different run. What proved this path,
# and is what to redo after touching any __getitem__: on --device cpu, a 4-epoch fold killed
# after epoch 2 and resumed against the same fold uninterrupted -- final weights and OOF
# bitwise equal at --workers 0, at 2, and with the OAI rotation engaged -- while skipping the
# RNG restore, the worker replay or the OAI replay each made them differ.
#
# Arguments that may legitimately differ between the killed command and the resuming one.
# Anything else -- --workers included, since the replay is per worker -- refuses.
RESUME_FREE = {"resume", "force", "all_folds", "fold", "mem_frac", "save"}


class _Tap:
    """The train loader's RandomSampler, passed through, remembering what it hands out.

    Bookkeeping only. The indices, their order and the single torch draw that seeds the
    shuffle are still the sampler's own, so a run with the tap is the run without it.
    """

    def __init__(self, sampler):
        self.sampler, self.seen = sampler, []

    def __iter__(self):
        for i in self.sampler:
            self.seen.append(i)
            yield i

    def __len__(self):
        return len(self.sampler)


def _count_served(seen, bs, workers, n_comp, draws, oai, ext2=None, ext2_lo=None,
                  ext3=None, ext3_lo=None):
    """Add one epoch to the per-worker tallies of samples, OAI, ext2 and ext3 rows served.

    Only whole batches were ever dispatched: under drop_last the sampler also hands out a
    trailing partial batch that BatchSampler then discards. Under --max-steps the batches
    a worker had prefetched past the break count too -- the worker still serves them,
    before the next epoch's _ResumeIteration, and they still consume its state.

    The train set is [competition | OAI | ext2 | ext3]: OAI rows are [n_comp, ext2_lo), ext2
    rows [ext2_lo, ext3_lo), ext3 rows [ext3_lo, end); an absent source's range is empty (its
    lo is the next source's). Counting every index >= n_comp as OAI once another source is
    attached would replay OAI's rotation too far on --resume. ext2=None (train_vlm.py) is the
    original rule; ext3=None is the two-source rule.
    """
    hi2 = float("inf") if ext3_lo is None else ext3_lo
    for j in range(len(seen) // bs):
        b = seen[j * bs:(j + 1) * bs]
        draws[j % max(workers, 1)] += len(b)
        if ext2 is None:
            oai[j % max(workers, 1)] += sum(ix >= n_comp for ix in b)
        else:
            oai[j % max(workers, 1)] += sum(n_comp <= ix < ext2_lo for ix in b)
            ext2[j % max(workers, 1)] += sum(ext2_lo <= ix < hi2 for ix in b)
            if ext3 is not None:
                ext3[j % max(workers, 1)] += sum(ix >= ext3_lo for ix in b)


def _source_start(ds, src):
    """First index of external source `src`'s rows in the train set; len(ds) if not attached."""
    if isinstance(ds, ConcatDataset):
        for d, lo in zip(ds.datasets, [0] + ds.cumulative_sizes[:-1]):
            if getattr(d, "source", None) == src:
                return lo
    return len(ds)


def _advance_oai(ds, n, src="oai"):
    """Replay n rows of an external source's exposure rotation (oai_aux's by default,
    ext_aux's with src="ext2" / "ext3") in the process that serves them. Found by its source
    tag, not its position: ext2 and ext3 are appended after OAI. A no-op when that source is
    not attached or its scheduler is not engaged (the 246-knee pilot; ext2 at --ext2-frac 1.0)."""
    oai = next((d for d in getattr(ds, "datasets", ()) if getattr(d, "source", None) == src), None)
    if getattr(oai, "epoch_rows", None) is not None:
        for _ in range(n):
            oai._next_row()


def _fast_forward(draws, oai, wid, ext2=None, ext3=None):
    """worker_init_fn on --resume: replay what this worker consumed before the kill.

    torch has already seeded the worker from the restored base_seed when this runs, so
    replaying the draws lands it exactly where the uninterrupted worker stood. Module-level
    and bound with functools.partial so Windows spawn can pickle it.
    """
    for _ in range(draws[wid]):
        np.random.randint(0, 2 ** 31)          # the call KneeStudies.__getitem__ makes
    _advance_oai(get_worker_info().dataset, oai[wid])
    if ext2 is not None:
        _advance_oai(get_worker_info().dataset, ext2[wid], "ext2")
    if ext3 is not None:
        _advance_oai(get_worker_info().dataset, ext3[wid], "ext3")


def load_fill(path, labels):
    """The --fill-masked table: StudyInstanceUID + the 12 TARGETS as probabilities, one row
    per study of `labels`, no NaN, every value in [0, 1]. Anything else is refused."""
    f = pd.read_csv(ROOT / path)
    miss = [c for c in ["StudyInstanceUID", *TARGETS] if c not in f.columns]
    if miss:
        raise SystemExit(f"--fill-masked {path}: missing columns {miss}")
    v = f[TARGETS].to_numpy(np.float64)
    if (not f.StudyInstanceUID.is_unique or np.isnan(v).any() or (v < 0).any()
            or (v > 1).any()):
        raise SystemExit(f"--fill-masked {path}: ids must be unique and values in [0, 1], "
                         f"no NaN")
    if set(f.StudyInstanceUID) != set(labels.StudyInstanceUID):
        raise SystemExit(f"--fill-masked {path}: its study ids are not the label table's")
    return f[["StudyInstanceUID", *TARGETS]]


def build_loaders(fold, labels, folds, gold, args):
    tr_ids = folds.loc[folds.fold != fold, "StudyInstanceUID"].tolist()
    va_ids = folds.loc[folds.fold == fold, "StudyInstanceUID"].tolist()
    # --exclude-gold: the gold studies leave TRAINING and the gold label override is emptied,
    # so gold stays a clean held-out check for this model. A held-out fold (0-4) is left as it
    # is -- its gold studies are still predicted. A full-data fold (--fold 5, which folds.csv
    # never uses) holds nothing out, so there the gold studies just taken out of training
    # become its validation set: the run predicts them into its OOF and main() prints their
    # gold AUC, as for a fold run. Without the flag a full-data fold validates on nothing.
    if getattr(args, "exclude_gold", False):
        n_all = len(tr_ids)
        tr_ids = [s for s in tr_ids if s not in gold]
        full = not va_ids
        if full:
            va_ids = [s for s in folds.StudyInstanceUID if s in gold]
        gold = {}
        print(f"  --exclude-gold: {n_all - len(tr_ids)} gold studies out of training, "
              f"{len(tr_ids)} competition studies train; no gold label override"
              + (f"; full-data fold, so those {len(va_ids)} gold studies are its validation "
                 f"set" if full else ""))
    tw = target_weights(args)
    sidx = ([int(x) for x in args.slots.split(',')] if args.slots else None)
    common = dict(labels=labels, cache=args.cache, n_group=args.n_group,
                  img=args.img, target_weights=tw, gold_labels=gold,
                  gold_weight=args.gold_weight, slot_idx=sidx)
    # Window length. train.py has no --group flag, so its datasets keep the default of 3
    # slices; train_vlm.py sets args.group = 2 (one window = one Qwen temporal patch).
    if getattr(args, "group", None):
        common["group"] = int(args.group)
    # WHICH CACHE LAYOUT. v1 is (6,32,336,336) with a slot axis and a (2,6) .mask.npy; f2 is
    # a FLAT (64,384,384) with a (4,5) .meta.npy, because its per-slot budgets are uneven.
    # The two datasets emit the identical 5-tuple, so nothing below this line changes.
    # getattr with a default: a checkpoint's stored args predate this flag, and
    # ngroup_sweep / inference rebuild datasets from those args.
    DS = KneeStudies
    if getattr(args, "cache_format", "v1") == "f2":
        from model_f2 import KneeStudiesF2
        DS = KneeStudiesF2
    # slot_drop on the TRAIN loader only. KneeStudies also guards on self.train, so this is
    # belt and braces -- but validation has to be produced from every slot the study
    # actually has, or the OOF measures a differently-handicapped model than the one that
    # gets deployed and gold AUC stops meaning what we think it means.
    # --fill-masked: the TRAIN dataset only, like slot_drop. `va` never sees the fill, so
    # OOF, gold AUC and teacher agreement are scored against the labels as they were.
    fill_kw = {}
    if getattr(args, "fill_masked", None):
        if DS is not KneeStudies:
            raise SystemExit("--fill-masked is implemented for the v1 cache (KneeStudies) only")
        fill_kw = dict(fill=load_fill(args.fill_masked, labels),
                       fill_weight=args.fill_weight)
    tr = DS(tr_ids, train=True, slot_drop=args.slot_drop, **common, **fill_kw)
    va = DS(va_ids, train=False, **common)
    if fill_kw:
        n = sum(int(s.sum()) for s, _ in tr.fill.values())
        print(f"  --fill-masked: {n} masked cells in {len(tr.fill)} training studies filled "
              f"from {Path(args.fill_masked).name} at x{args.fill_weight} their target weight")
    # OAI knees join the TRAIN dataset and nothing else. `va` is already built and is
    # never passed to attach_oai, so there is no code path -- not a convention someone
    # has to remember -- by which an OAI knee reaches an OOF, a gold AUC, or a model
    # selection decision. Note what is NOT passed: `tw`. OAI cells carry their own
    # weight (see oai_aux.py, defect 4); the teacher's reliability vector cannot reach
    # them because the function has no parameter for it.
    if getattr(args, "oai_weight", 0.0) > 0:
        # The OAI cache is v1-shaped -- (6,32,336,336), verified in its meta.json -- so an
        # f2 run cannot mix it in: the two datasets would hand the collate (5,G,3,384,384)
        # and (6,G,3,336,336) tensors. That does fail on its own, but it fails as a collate
        # stack trace some minutes into epoch 0 rather than as a sentence at startup, and
        # the champion command this one is derived from carries --oai-weight 1.0.
        if getattr(args, "cache_format", "v1") == "f2":
            raise SystemExit(
                "--oai-weight > 0 with --cache-format f2: the OAI cache is v1-shaped "
                "(6,32,336,336). Pass --oai-weight 0.0, or build an f2 OAI cache first.")
        tr = attach_oai(tr, args, labels)
    # A SECOND and a THIRD external source (ext_aux.py; SKM-TEA and MRNet today), under the
    # same structural guarantee: attached to `tr` only, after `va` exists. Appended AFTER OAI,
    # in this order, so the train set is [competition | OAI | ext2 | ext3] -- the per-source
    # resume counts in run_fold rely on it.
    for src in ("ext2", "ext3"):
        if getattr(args, f"{src}_weight", 0.0) > 0:
            if getattr(args, "cache_format", "v1") == "f2":
                raise SystemExit(
                    f"--{src}-weight > 0 with --cache-format f2: the {src} cache is v1-shaped "
                    f"(6,32,336,336). Pass --{src}-weight 0.0, or build an f2 {src} cache first.")
            tr = attach_ext(tr, args, labels, src)
    return (DataLoader(tr, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                       pin_memory=True, drop_last=True, persistent_workers=args.workers > 0),
            DataLoader(va, batch_size=args.bs, shuffle=False, num_workers=args.workers,
                       pin_memory=True, persistent_workers=args.workers > 0),
            va_ids)


def make_optimizer(net, args):
    head = [p for n, p in net.named_parameters() if not n.startswith("backbone.")]
    back = [p for n, p in net.named_parameters() if n.startswith("backbone.")]
    return torch.optim.AdamW(
        [{"params": back, "lr": args.lr_backbone}, {"params": head, "lr": args.lr_head}],
        weight_decay=args.wd)


@torch.no_grad()
def predict(net, loader, dev, amp_dtype, img_size=None):
    net.eval()
    out = []
    for g, m, y, lm, w in loader:
        g, m = g.to(dev, non_blocking=True), m.to(dev, non_blocking=True)
        with torch.autocast(dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits, _ = net(g, m, img_size=img_size)
        out.append(torch.sigmoid(logits.float()).cpu())
    if not out:                    # an empty loader: the full-data run has no held-out fold
        return np.zeros((0, len(TARGETS)), np.float32)
    return torch.cat(out).numpy()


def build_augs(args):
    """(PhotometricAug or None, GeoAug or None) for this run's flags.

    --geo-aug: one affine per (study, slot), see aug.GeoAug. It REPLACES PhotometricAug's
    shift (one draw per batch that wraps via torch.roll), hence p_shift=0 under it. Without
    the flag this builds exactly what run_fold always built: PhotometricAug() or nothing.
    """
    geo = None
    if getattr(args, "geo_aug", False):
        from aug import GeoAug
        if getattr(args, "cache_format", "v1") == "f2":
            from model_f2 import SLOTS_F2
            geo = GeoAug(SLOTS_F2)
        else:
            geo = GeoAug(SLOTS)
    aug = None
    if getattr(args, "aug", False):
        from aug import PhotometricAug
        aug = PhotometricAug(p_shift=0.0) if geo is not None else PhotometricAug()
    return aug, geo


def run_fold(fold, args, labels, folds, gold):
    # Per FOLD, not once per process. Otherwise `--fold 3` and fold 3 of `--all-folds` train
    # different models from the same seed, and a run resumed out of oof_parts/ is not the
    # run it claims to be continuing -- which matters here, because resuming is the normal
    # case. seed*1000 + fold is injective for any fold count this project will ever use.
    set_seed(args.seed * 1000 + fold)
    dev = torch.device(args.device)
    amp = torch.bfloat16 if (dev.type == "cuda" and args.amp) else None
    tr_ld, va_ld, va_ids = build_loaders(fold, labels, folds, gold, args)
    sidx = ([int(x) for x in args.slots.split(',')] if args.slots else None)
    aug, geo = build_augs(args)
    # The anatomy prior is indexed by SLOT, so it has to name which rack those indices mean.
    # None keeps model.SLOT_PRIOR, which is what every checkpoint on disk was trained under.
    prior_map = None
    if getattr(args, "cache_format", "v1") == "f2":
        from model_f2 import SLOT_PRIOR_F2
        prior_map = SLOT_PRIOR_F2
    # img_size: a fixed-input-size backbone (coatnet_rmlp_2_rw_224 at --img 288) is built at
    # the resolution forward() resizes to. Every other backbone ignores it.
    net = KneeNet(backbone=args.backbone, pretrained=not args.no_pretrained,
                  drop=args.dropout, slot_idx=sidx, aug=aug,
                  mil_queries=args.mil_queries, mil_sharpen=args.mil_sharpen,
                  drop_path=args.drop_path, prior_map=prior_map, img_size=args.img,
                  geo_aug=geo).to(dev)
    if args.grad_ckpt:
        try:
            net.backbone.set_grad_checkpointing(True)
        except Exception as e:
            print(f'  grad checkpointing unavailable for {args.backbone}: {e}')
    opt = make_optimizer(net, args)
    steps = max(len(tr_ld) * args.epochs, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr_backbone, args.lr_head], total_steps=steps, pct_start=0.15)

    print(f"\nfold {fold}: train {len(tr_ld.dataset)}  val {len(va_ld.dataset)}  "
          f"steps {steps}  device {dev} amp={amp}"
          + (f"  slot-drop {args.slot_drop}" if args.slot_drop else ""))
    if not va_ids:
        print(f"  FULL-DATA: fold {fold} holds no studies -- nothing held out, OOF and "
              f"validation are skipped; the model is checkpoints/fold{fold}_<tag>.pt")

    # MID-FOLD RESUME -- see the block above _Tap. The sidecar sits beside the final
    # checkpoint and is named the same way, so it is unique per --out and fold. Everything
    # above this line ran exactly as in the killed process (same seed, same init draws);
    # the restore below then overwrites its results.
    tap = _Tap(tr_ld.batch_sampler.sampler)
    tr_ld.batch_sampler.sampler = tap
    ds = tr_ld.dataset
    n_comp = len(ds.datasets[0]) if isinstance(ds, ConcatDataset) else len(ds)
    tag = Path(args.out).stem.replace("oof_", "") or args.backbone
    side = ROOT / "checkpoints" / f"fold{fold}_{tag}.resume.pt"
    # seed_state: the torch state the first epoch's iterator draws base_seed from. Taken
    # here, before any epoch, and carried through every sidecar of the fold.
    start, seed_state = 0, torch.get_rng_state()
    draws, oai = [0] * max(args.workers, 1), [0] * max(args.workers, 1)
    # An absent source's range is empty: its lo is the next source's (len(ds) at the end).
    ext3, ext3_lo = [0] * max(args.workers, 1), _source_start(ds, "ext3")
    ext2, ext2_lo = [0] * max(args.workers, 1), min(_source_start(ds, "ext2"), ext3_lo)
    # The any-of target (ext_aux, "THE ANY-OF TARGET"): on when an attached source carries it,
    # and then the loss is distill_loss_any. Read off the dataset, not the flags, so a run
    # without such a source takes distill_loss exactly as before.
    any_idx = next((d.any_idx for d in getattr(ds, "datasets", ())
                    if getattr(d, "any_idx", None)), None)
    if any_idx is not None and (args.noise_q > 0 or args.rank_weight > 0):
        raise SystemExit(
            "--noise-q / --rank-weight with an any-of target attached: neither loss has a "
            "noisy-OR form (GCE would silently drop those rows, and the rank loss averages the "
            "per-row weights an any-of row carries negated). Drop one of the two.")
    if getattr(args, "resume", False) and side.exists():
        st = torch.load(side, map_location="cpu", weights_only=False)
        now = vars(args)
        bad = sorted(k for k in set(st["args"]) | set(now)
                     if k not in RESUME_FREE and st["args"].get(k) != now.get(k))
        if bad:
            raise SystemExit(
                f"--resume: {side.name} was written under different "
                f"{', '.join('--' + k.replace('_', '-') for k in bad)}. Continuing it would "
                f"splice two different runs into one model. Rerun the original command, "
                f"or drop --resume to start the fold over.")
        net.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start, seed_state, draws, oai = st["epoch"], st["seed_state"], st["draws"], st["oai"]
        ext2, ext3 = st.get("ext2", ext2), st.get("ext3", ext3)
        random.setstate(st["rng"]["python"])
        np.random.set_state(st["rng"]["numpy"])
        torch.set_rng_state(st["rng"]["torch"])
        if st["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(st["rng"]["cuda"])
        if args.workers > 0:
            tr_ld.generator = torch.Generator()
            tr_ld.generator.set_state(seed_state)
            tr_ld.worker_init_fn = partial(_fast_forward, list(draws), list(oai),
                                           ext2=list(ext2), ext3=list(ext3))
        else:
            _advance_oai(ds, oai[0])
            _advance_oai(ds, ext2[0], "ext2")
            _advance_oai(ds, ext3[0], "ext3")
        print(f"  resumed from {side.name}: {start}/{args.epochs} epochs done", flush=True)
    t0 = time.time()
    for ep in range(start, args.epochs):
        net.train()
        tot = n = 0
        tap.seen.clear()
        for i, (g, m, y, lm, w) in enumerate(tr_ld, 1):
            g, m = g.to(dev, non_blocking=True), m.to(dev, non_blocking=True)
            y, lm, w = y.to(dev), lm.to(dev), w.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, dtype=amp, enabled=amp is not None):
                logits, _ = net(g, m, img_size=args.img)
                loss = (gce_loss(logits.float(), y, lm, w, q=args.noise_q)
                        if args.noise_q > 0 else distill_loss(logits.float(), y, lm, w)
                        if any_idx is None else
                        distill_loss_any(logits.float(), y, lm, w, any_idx))
                if args.rank_weight > 0:
                    # Added, not substituted. A pure ranking loss leaves the output scale
                    # unconstrained, and with only ~28 pairs per target per step the estimate
                    # is noisy; BCE supplies the stable signal and this supplies the one the
                    # metric actually reads.
                    loss = loss + args.rank_weight * pairwise_rank_loss(
                        logits.float(), y, lm, w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            opt.step()
            sched.step()
            tot += loss.item() * len(g)
            n += len(g)
            if args.max_steps and i >= args.max_steps:
                break
        # The sidecar, rewritten every epoch -- temp file then os.replace, so a kill
        # mid-write leaves the previous epoch's intact -- and BEFORE the epoch line, so an
        # epoch the log reports is an epoch --resume can continue from. Not here because
        # there is nothing to hold: a GradScaler (bf16 autocast needs none, and none
        # exists) and best-epoch / early-stop state (the fold's model is its last epoch).
        _count_served(tap.seen, args.bs, args.workers, n_comp, draws, oai, ext2, ext2_lo,
                      ext3, ext3_lo)
        side.parent.mkdir(exist_ok=True)
        tmp = side.with_name(side.name + ".tmp")
        torch.save({"epoch": ep + 1, "model": net.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                            "torch": torch.get_rng_state(),
                            "cuda": (torch.cuda.get_rng_state_all() if dev.type == "cuda"
                                     else None)},
                    "seed_state": seed_state, "draws": draws, "oai": oai, "ext2": ext2,
                    "ext3": ext3, "args": vars(args)}, tmp)
        os.replace(tmp, side)
        print(f"  epoch {ep + 1}/{args.epochs}  loss {tot / max(n, 1):.4f}  "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)

    pred = predict(net, va_ld, dev, amp, img_size=args.img)
    out = pd.DataFrame(pred, columns=TARGETS)
    out.insert(0, "StudyInstanceUID", va_ids)

    if args.save:
        ck = ROOT / "checkpoints"
        ck.mkdir(exist_ok=True)
        # Name by the RUN, not just the backbone. Every convnext_tiny experiment previously
        # wrote fold{N}_convnext_tiny.pt, so each run silently destroyed the last one's
        # weights: on 2026-08-16 three trained models (v2, v3, 4-slot) were lost this way
        # and the reported ensemble turned out to be undeployable. The OOF predictions
        # survived, so the science did not -- but the models did.
        tag = Path(args.out).stem.replace("oof_", "") or args.backbone
        torch.save({"model": net.state_dict(), "args": vars(args), "fold": fold},
                   ck / f"fold{fold}_{tag}.pt")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="convnext_tiny")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--all-folds", action="store_true")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--n-group", type=int, default=12)
    ap.add_argument("--img", type=int, default=336)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--drop-path", type=float, default=0.0,
                    help="stochastic depth rate for the backbone (timm drop_path_rate, "
                         "linearly ramped over blocks). We have always run 0.0; the "
                         "ConvNeXt-S recipe uses 0.4 from scratch, fine-tunes use 0.1-0.3. "
                         "Training-only (DropPath is identity in eval), holds no "
                         "parameters, so checkpoints stay loadable everywhere. 0.0 "
                         "reproduces the current create_model call byte-identically.")
    ap.add_argument("--mil-queries", default="shared",
                    choices=["shared", "band", "per_target"],
                    help="how many L2 attention distributions exist. 'shared' is what has "
                         "always shipped: ONE distribution over the ~10 slice groups of a "
                         "slot, used by all 12 findings. Measured on fold0_aug.pt that "
                         "distribution is UNIFORM (normalised entropy 0.998), i.e. L2 is "
                         "mean pooling. 'per_target' gives each finding its own query; "
                         "'band' groups them into 5 by slice band. See mil_perdiag.py.")
    ap.add_argument("--mil-sharpen", action="store_true",
                    help="standardise the L2 attention logits across the group axis and "
                         "scale by a learned per-query gain. Without it the softmax spread "
                         "is set by ||w||, which weight decay holds at its init value "
                         "(rms 0.0478 after 14 epochs) and the attention stays uniform. "
                         "Per-diagnosis queries without this are 12 copies of one uniform "
                         "distribution, so run this flag on its own first.")
    ap.add_argument("--gold-weight", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds torch (init, dropout, shuffle order), numpy, python-random, "
                         "and -- via PyTorch's per-worker base_seed -- the slice-window "
                         "jitter inside every DataLoader worker. Each fold is seeded "
                         "seed*1000+fold, so --fold N reproduces fold N of --all-folds. "
                         "For multi-seed averaging: same command, different --seed.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--cache-format", default="v1", choices=["v1", "f2"],
                    help="which on-disk cache contract --cache points at. 'v1' is the "
                         "(6,32,336,336) slot-axis cache with a (2,6) .mask.npy sidecar and "
                         "is the default, reproducing current behaviour exactly. 'f2' is "
                         "build_cache_f2.py's FLAT (64,384,384) cache with a (4,5) "
                         ".meta.npy: 5 slots, uneven budgets 18/14/12/8/12, 140 mm crop, "
                         "6-94%% slice band. It also switches the L3 anatomy prior to "
                         "model_f2.SLOT_PRIOR_F2, because slot indices mean different "
                         "sequences in the two racks. Requires --slots 0,1,2,3,4. --img "
                         "below 384 (e.g. 224/288 with coatnet_rmlp_2_rw_224) is read with "
                         "an antialiased downscale; the cache's frame layout is saved in "
                         "the checkpoint args as f2_budget for the Kaggle builder.")
    ap.add_argument("--labels", default="report_labels_llm.csv")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--save", action="store_true", default=True)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="CPU, 6 studies, verifies the loop")
    ap.add_argument("--out", default="oof_predictions.csv")
    ap.add_argument("--mem-frac", type=float, default=0.0,
                    help="hard cap on this process's share of VRAM (0 = uncapped). "
                         "Set when sharing the GPU: PyTorch's caching allocator "
                         "otherwise grows to fill the card and never gives memory "
                         "back, so a co-tenant OOMs even while we sit idle.")
    ap.add_argument("--aug", action="store_true",
                    help="photometric augmentation (gamma / brightness / contrast / "
                         "noise / small shift), per study, GPU-side, training only. "
                         "The pipeline has had NO photometric augmentation until now: "
                         "the only stochasticity was slice-window jitter.")
    ap.add_argument("--geo-aug", action="store_true",
                    help="geometric augmentation (aug.GeoAug), training only: one affine per "
                         "(study, slot), shared by all its windows and slices, p 0.5 -- "
                         "rotation +-10 deg sag/cor and +-15 deg axial, zoom-in 1.00-1.06, "
                         "shift +-5%% of the side, zero padding, never wraps. It also turns "
                         "OFF --aug's shift (one draw per batch that wraps). Off reproduces "
                         "the current run exactly.")
    ap.add_argument("--exclude-gold", action="store_true",
                    help="drop the 58 gold studies from TRAINING and empty the gold label "
                         "override, so gold stays a clean held-out check. Gold in a held-out "
                         "fold is still predicted. With --fold 5 (a fold folds.csv never "
                         "uses) the run trains on every non-gold study and validates on the "
                         "58 gold: their predictions are its OOF and their gold AUC prints at "
                         "the end. More seeds: --seed N, each with its own --out.")
    ap.add_argument("--slot-drop", type=float, default=0.0,
                    help="probability of hiding each PRESENT slot from one epoch's view of "
                         "a study. Training only, at most 2 slots, never the last one. "
                         "Recommended 0.15. Off by default because the gain is unmeasured: "
                         "the 58 gold studies are the only instrument and no run has used "
                         "this yet.")
    ap.add_argument("--noise-q", type=float, default=0.0,
                    help="generalised cross-entropy exponent (D35). BCE is unbounded on a "
                         "confidently-wrong label, so with report-derived labels a minority of "
                         "bad cells dominates the gradient -- which is why convnext_base fit "
                         "better and generalised worse (D30b). GCE bounds each cell's "
                         "contribution at 1/q. 0.7 is the paper default; 0.0 reproduces "
                         "distill_loss EXACTLY and is verified to in noise_loss.py.")
    ap.add_argument("--rank-weight", type=float, default=0.0,
                    help="weight on the pairwise ranking term (D28). The metric is macro "
                         "ROC-AUC, which reads only the ORDER of scores, while BCE reads their "
                         "VALUE -- D15 measured a teacher ranking gain of +0.0112 delivering "
                         "only +0.0043 to the student. Start at 0.2: the term is estimated "
                         "from at most B*(B-1)/2 = 28 pairs per target at bs 8, so it is "
                         "noisier per step than BCE and is a corrective, not a replacement. "
                         "0.0 reproduces the current behaviour exactly.")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="gradient checkpointing in the backbone: recompute activations "
                         "during backward instead of storing them. Cuts activation memory "
                         "several-fold for ~30%% more time. This is how a bigger backbone "
                         "or higher resolution fits under a shared-GPU memory cap without "
                         "taking VRAM away from the other tenant.")
    ap.add_argument("--slots", default=None,
                    help="comma-separated cached-slot indices to use, e.g. 0,1,2,3. "
                         "Default all 6. Slots are skipped at load time, so dropping "
                         "them saves real compute as well as testing the score effect.")
    ap.add_argument("--random-folds", action="store_true",
                    help="reassign folds at random, ignoring site groups. The ONLY "
                         "honest way to measure the scanner shortcut: train both ways "
                         "and compare gold AUC. Metadata alone yields +0.139, so the "
                         "difference here is how much of our score is site memory.")
    ap.add_argument("--force", action="store_true",
                    help="redo folds whose predictions are already saved in oof_parts/")
    ap.add_argument("--resume", action="store_true",
                    help="continue a fold killed MID-training from its per-epoch sidecar "
                         "(checkpoints/fold{N}_{tag}.resume.pt, written by every run) "
                         "instead of restarting it: weights, AdamW, OneCycle, the RNG "
                         "states, and the DataLoader workers' jitter / OAI-rotation "
                         "position. Finished folds are still skipped via oof_parts/. A "
                         "sidecar written under different arguments is refused. Without "
                         "this flag every fold starts from scratch exactly as before.")
    ap.add_argument("--oai-weight", type=float, default=0.0,
                    help="mix radiologist-read OAI/MOAKS knees into TRAINING at this "
                         "weight (0 = off, and 0 reproduces current behaviour exactly -- "
                         "attach_oai returns the dataset object unchanged). Real "
                         "radiologist readings off the images, the only supervision "
                         "here that does not come from the LLM teacher. Never scored: "
                         "see oai_aux.py. Start at 1.0.")
    ap.add_argument("--oai-targets", default="weak",
                    help="'weak' (Lateral Meniscus, Lateral OA, PF OA, Synovitis -- "
                         "the four targets holding the remaining headroom, all covered "
                         "by MOAKS), 'safe' (those plus Medial OA/Meniscus, ACL, "
                         "Effusion, Baker's), or a comma-separated list. MCL/Fracture "
                         "(unscored) and Contusion (semantic mismatch, forced silent) "
                         "are refused.")
    ap.add_argument("--oai-cache", default=None,
                    help="override the OAI cache directory (default data/"
                         "oai_cache, the 246-knee pilot). Full scale: data/"
                         "oai_full_cache. Never the competition cache.")
    ap.add_argument("--oai-labels", default=None,
                    help="explicit path to the ladder label table (default <oai-cache>/"
                         "oai_cache_labels_ladder.csv, the pilot table). The full-scale "
                         "run MUST name data/oai_cache/"
                         "oai_full_labels_ladder.csv -- the per-visit build; the pilot's "
                         "visit-mixed table cannot describe the 7,992-row cache.")
    ap.add_argument("--oai-frac", type=float, default=None,
                    help="cap OAI EXPOSURE: the sampler sees round(frac x competition "
                         "train rows) OAI rows per epoch, drawn at PARTICIPANT level "
                         "(both knees and all visits travel together), rotating through "
                         "the pool across epochs, deterministic under --seed/--workers. "
                         "None -> oai_aux.py's measured default 0.125 (the low sweep "
                         "arm, ~5%% of weighted mass). Values outside (0, 0.5] are "
                         "REFUSED -- above 0.5 the competition would stop outnumbering "
                         "OAI 2:1 -- and a second unconditional guard refuses an epoch "
                         "with more OAI rows than competition rows. See oai_aux.py "
                         "DECISION 5. When the cap covers the whole pool (the 246-knee "
                         "pilot) the scheduler never engages.")
    ap.add_argument("--ext2-weight", type=float, default=0.0,
                    help="mix a SECOND external source into TRAINING, beside OAI (ext_aux.py; "
                         "SKM-TEA by default: 145 knees whose report findings were localised "
                         "and graded on the images as 3-D boxes). 0 = off, and 0 reproduces "
                         "current behaviour exactly. Train loader only, never scored, "
                         "repriced to our base rate per target like OAI. Start at 1.0.")
    ap.add_argument("--ext2-targets", default="skm",
                    help="'skm' (ACL, Medial/Lateral Meniscus, Medial/Lateral OA, PF OA, "
                         "Effusion -- what SKM-TEA reaches), 'asserted' (every target the "
                         "table asserts), or a comma-separated list. A target the table never "
                         "asserts is refused.")
    ap.add_argument("--ext2-cache", default=None,
                    help="ext2 cache directory (default data/skmtea_cache, "
                         "build_skmtea_cache.py's output). Never the competition cache.")
    ap.add_argument("--ext2-labels", default=None,
                    help="ext2 ladder label table (default <ext2-cache>/labels.csv).")
    ap.add_argument("--ext2-frac", type=float, default=None,
                    help="fraction of the ext2 POOL exposed per epoch, in (0, 1]. None -> "
                         "1.0: every row every epoch (145 rows are ~4%% of a fold's "
                         "competition rows, so there is no majority to cap; see ext_aux.py). "
                         "Below 1 engages OAI's participant-level rotation, exact under "
                         "--resume. OAI + ext2 rows together may never outnumber the "
                         "competition rows in an epoch.")
    ap.add_argument("--ext3-weight", type=float, default=0.0,
                    help="mix a THIRD external source into TRAINING, after OAI and ext2 "
                         "(ext_aux.py; MRNet by default: 1,250 Stanford clinical knee MRIs in "
                         "slots 0/2/4, labels hand-extracted from their reports -- ACL tear, and "
                         "a meniscal tear without its side, trained as a noisy-OR over the "
                         "Medial/Lateral Meniscus heads). 0 = off, and 0 reproduces current "
                         "behaviour exactly. Train loader only, never scored, repriced to our "
                         "base rate. Start at 1.0.")
    ap.add_argument("--ext3-targets", default="mrnet",
                    help="'mrnet' (ACL + 'Any Meniscus'), 'asserted', or a comma-separated "
                         "list. 'Any Meniscus' is the any-of group: a positive exam trains "
                         "P(any) = 1-(1-p_MM)(1-p_LM), a negative one both heads as absent. "
                         "Medial/Lateral Meniscus without the group are refused (negatives "
                         "only). With the group on, --noise-q and --rank-weight are refused.")
    ap.add_argument("--ext3-cache", default=None,
                    help="ext3 cache directory (default data/mrnet_cache, "
                         "build_mrnet_cache.py's output). Never the competition cache.")
    ap.add_argument("--ext3-labels", default=None,
                    help="ext3 ladder label table (default <ext3-cache>/labels.csv).")
    ap.add_argument("--ext3-frac", type=float, default=None,
                    help="fraction of the ext3 POOL exposed per epoch, in (0, 1]. None -> "
                         "0.3: 375 of MRNet's 1,250 rows, so with OAI at 0.5 and SKM at 1.0 "
                         "the competition stays ~61%% of samples (1.0 would leave it ~53%%; "
                         "see ext_aux.py). Participant-level rotation, exact under --resume. "
                         "OAI + ext2 + ext3 rows may never outnumber the competition rows.")
    ap.add_argument("--fill-masked", default=None,
                    help="CSV of cross-fit ensemble probabilities (StudyInstanceUID + the 12 "
                         "targets, e.g. oof_fill_bl48.csv). On COMPETITION TRAINING rows only, "
                         "cells whose state is silent/not_mentioned take the fill value as "
                         "target and are unmasked at (per-target weight x --fill-weight). "
                         "Validation, gold studies and external rows are untouched. None = off "
                         "(current behaviour exactly).")
    ap.add_argument("--fill-weight", type=float, default=1.0,
                    help="multiplier on a filled cell's usual per-target weight (only with "
                         "--fill-masked)")
    ap.add_argument("--target-weights", default=None, choices=["teacher", "uniform"],
                    help="per-target loss weight of the COMPETITION rows. Omitted = 'teacher' "
                         "= the TEACHER_AUC weights (current behaviour exactly; omitted rather "
                         "than defaulted so sidecars written before this flag still --resume). "
                         "'uniform' = 1.0 for every target: for a label table TEACHER_AUC was "
                         "not measured on. Gold rows keep the x--gold-weight multiplier on top; "
                         "OAI / ext rows keep their own weighting.")
    a = ap.parse_args()

    # An f2 checkpoint must say which frame layout it was trained on (64 = 18/14/12/8/12 or
    # dense 120 = 26/24/26/16/28): the Kaggle notebook rebuilds exactly that layout from
    # DICOM, and the layout is recorded only in the cache's meta.json. Saved in vars(args),
    # so a run killed before this existed is refused by --resume rather than spliced.
    if a.cache_format == "f2":
        _meta = json.loads((Path(a.cache) / "meta.json").read_text(encoding="utf-8"))
        a.f2_budget = [int(x) for x in _meta["budget"]]

    if a.mem_frac and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(a.mem_frac, 0)
        tot = torch.cuda.get_device_properties(0).total_memory
        print(f"VRAM capped at {a.mem_frac:.0%} = {a.mem_frac*tot/1e9:.1f} GB "
              f"(sharing the GPU)")

    labels = pd.read_csv(ROOT / a.labels)
    folds = pd.read_csv(ROOT / "folds.csv")
    if a.random_folds:
        rng = np.random.default_rng(2026)
        folds = folds.copy()
        folds['fold'] = rng.permutation(folds.fold.values)
        spans = folds.groupby('group').fold.nunique()
        print(f"RANDOM FOLDS: sites now span {int((spans > 1).sum())}/{len(spans)} "
              f"groups across folds (grouped run: 0). Any gain over the grouped run "
              f"is scanner memorisation, not skill.")

    tr = pd.read_csv(ROOT / "train.csv")
    g = tr[tr[TARGETS].notna().all(axis=1)]
    gold = {r["StudyInstanceUID"]: np.asarray([r[t] for t in TARGETS], np.float32)
            for _, r in g.iterrows()}
    print(f"labels {a.labels}   gold studies {len(gold)}   seed {a.seed}")

    if a.target_weights == "uniform":
        print("\nper-target weight: UNIFORM 1.0 on competition rows (--target-weights uniform)")
    else:
        tw = teacher_weights()
        print("\nper-target teacher weight (from measured gold AUC):")
        for t, w in zip(TARGETS, tw):
            print(f"  {t:<18} auc {TEACHER_AUC[t]:.3f} -> weight {w:.3f}"
                  + ("   <- near floor, head leans on gold" if w <= 0.16 else ""))

    if a.smoke:
        a.device, a.epochs, a.bs, a.workers = "cpu", 1, 2, 0
        a.n_group, a.img, a.max_steps, a.amp = 2, 96, 0, False
        a.epochs = 3
        a.no_pretrained, a.save = True, False
        # distinct stem: smoke parts must never be resumed into a real run
        a.out = "oof_smoke.csv"
        a.backbone = "resnet18"
        # Include gold studies deliberately. A smoke test that never touches the gold
        # path leaves the label-override and the 8x weight completely unexercised, and
        # both fail silently -- training would simply learn from the weaker labels.
        gsel = folds[folds.is_gold].groupby("fold").head(2).StudyInstanceUID
        keep = pd.concat([gsel, folds.groupby("fold").head(2).StudyInstanceUID]).unique()
        folds = folds[folds.StudyInstanceUID.isin(keep)].reset_index(drop=True)
        print(f"\nSMOKE: cpu, {len(folds)} studies, resnet18, 96px, 2 groups")

    from validation import evaluate_gold, grouped_oof_auc

    # Write each fold's predictions as soon as it finishes, and skip folds already on disk.
    # An eight-hour five-fold run that dies on the last fold otherwise loses everything the
    # first four produced -- which is exactly what happened on 2026-08-13.
    folds_to_run = range(5) if a.all_folds else [a.fold]
    part_dir = ROOT / "oof_parts"
    part_dir.mkdir(exist_ok=True)
    stem = Path(a.out).stem
    parts = []
    for f in folds_to_run:
        pf = part_dir / f"{stem}_fold{f}.csv"
        if pf.exists() and not a.force:
            print(f"\nfold {f}: already done -> {pf.name} (use --force to redo)")
            parts.append(pd.read_csv(pf))
            continue
        df = run_fold(f, a, labels, folds, gold)
        df.to_csv(pf, index=False)
        print(f"  saved {pf.name}")
        parts.append(df)
    oof = pd.concat(parts, ignore_index=True)
    if oof.empty:
        # A full-data run without --exclude-gold (--fold 5) predicts nothing: an empty OOF
        # file would only be a trap for scripts that glob oof_*.csv. The empty part in
        # oof_parts/ stays, as the marker that lets a rerun skip the finished fold.
        print(f"\nno held-out studies: {a.out} not written, validation skipped")
        return
    oof.to_csv(ROOT / a.out, index=False)
    print(f"\nwrote {a.out}  ({len(oof)} studies)")

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    have_gold = oof.StudyInstanceUID.isin(gold).sum()
    if have_gold >= 8:
        print(f"GOLD AUC  ({have_gold} gold studies in this OOF)")
        evaluate_gold(oof)
    else:
        print(f"only {have_gold} gold studies predicted — run --all-folds for a gold read")
    if len(oof) > 200:
        agree = grouped_oof_auc(oof, labels)
        print(f"\nteacher agreement (macro over held-out folds): {agree['macro']:.3f}")
        print("scanner-shortcut gap: NOT MEASURED — needs a second run with random folds")
        print("  (metadata alone yields +0.139, so the question is live)")


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()
