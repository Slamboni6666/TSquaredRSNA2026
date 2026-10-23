"""oai_aux.py -- mix OAI-ZIB / MOAKS knees into training as AUXILIARY supervision.

STAGED, NOT APPLIED. Copy to the repo root when pending_patches/OAI-aux-supervision.md is
applied. Nothing in the repo imports it until train.py gains the one import line in that
patch, so its presence here cannot affect a running job.

WHY
Our twelve labels come from an LLM reading reports, and the ensemble plateau (five
structurally different models all at LB 0.933-0.936, pairwise correlation ~0.76) is the
shared-error structure of that one teacher plus one input pipeline. MOAKS is the only
supervision on this machine that a radiologist read OFF THE IMAGES, so it is the only thing
here that can make a model wrong in DIFFERENT places rather than reshuffling the same errors.

WHAT THIS IS NOT
OAI is never an evaluation set. 246 knees give a per-target AUC SD of ~0.035, noisier than
the effects we chase, and the cohort is OA-enriched, so a score on held-out OAI knees would
measure neither our data nor our metric. These knees enter the TRAIN loader only; `attach_oai`
has no path that can put one in a validation loader, and they never appear in folds.csv, so
no fold logic -- including --random-folds, which permutes that file's fold column -- can ever
route one into an OOF.

THE FOUR DEFECTS THE ADVERSARIAL VERIFICATION FOUND, and where each fix now lives
  1 FOLD LEAK      OAI ids stay OUT of folds.csv (asserted at attach time and in the verify
                   script); fold membership is published as fold = -1 in folds_oai.csv,
                   which train.py never reads. See oai_ladder.py's fold-table notes.
  2 INVENTED       This module now consumes oai_cache_labels_ladder.csv, where a cell MOAKS
    NEGATIVES      did not read is state="silent" and a graded cell sits on the soft ladder
                   (0.03..0.95). It REFUSES a table with hard 0/1 on any asserted cell, so
                   the old binarised oai_cache_labels.csv cannot be wired back in.
  3 SELF-TEST      verify_oai_aux.py now scans pixel content and rebuilds the label table
                   from the MOAKS source; corrupt_oai_demo.py proves each check can fail.
  4 TEACHER-WEIGHT `attach_oai` no longer accepts the teacher's per-target reliability
    CONTAMINATION  vector at all. OAI cells carry their OWN weight: base 1.0 (a radiologist
                   read is not discounted by the LLM's unreliability on that target -- the
                   old wiring would have priced Synovitis, the target OAI exists to fix, at
                   0.15 on the strength of the noise it exists to beat), times the
                   class-reprice below, times --oai-weight.

------------------------------------------------------------------------------------------
DECISION 1 -- WHICH TARGETS. The four weak ones by default. Contusion cannot be selected.
------------------------------------------------------------------------------------------
MOAKS does not score MCL or the collaterals, and this is an osteoarthritis cohort with no
Fracture column: both arrive as state="silent" and are masked by KneeStudies for free.

Contusion is excluded at the LABEL level (silent on every OAI row, enforced by oai_ladder.py
and re-checked here) because it is a semantic mismatch, not a prevalence shift: MOAKS
"Contusion" is the V00MBM* bone-marrow-lesion family, a feature OF osteoarthritis, while a
report "contusion" is acute trauma. The conditional relationship reverses sign (OA
anti-predicts contusion in our reports, lift -1.0 to -6.5 pp; it predicts BML in OAI, +19.2
to +36.8 pp), so MOAKS supervision would teach that head the wrong sign. Selecting it here
raises rather than silently training nothing.

The default set is the four targets target_adapter.py independently named as the whole
remaining headroom -- Lateral Meniscus, Lateral OA, PF OA, Synovitis -- and MOAKS covers all
four. The targets where our teacher is weakest are exactly the ones a cartilage/meniscus
reading protocol scores best.

------------------------------------------------------------------------------------------
DECISION 2 -- PREVALENCE REPRICING, which is what stops the supervision being wasted.
------------------------------------------------------------------------------------------
The cohort is OA-enriched, and measured over all 4,407 cached competition studies EXACTLY
ZERO present the slot-0-only mask every OAI knee presents. "One slot, and it is
SAG_FLUID_FS" is a free domain flag on top of DESS contrast, so the cheapest way to reduce
an unweighted OAI loss is a domain-conditional INTERCEPT -- "this is a DESS knee, raise the
OA logits" -- which transfers nothing and reads out as a null we would misread as "DESS does
not transfer".

The fix is to price that shortcut out. Cells are classed by RUNG, not by value: a rung that
asserts presence (mild 0.45 upward) is a positive even when it sits below 0.5 -- Synovitis
grade 1 lands on mild, and classing by value would call 71 of MOAKS's 108 Synovitis
positives negative. With f_pos/f_neg the class fractions among asserted OAI cells, m_pos/
m_neg the mean label MASS each class carries, and q our own soft prevalence, solve exactly:

    w_pos * f_pos + w_neg * f_neg = 1      MEAN-PRESERVING: --oai-weight stays the only
                                           control on total mass
    w_pos * m_pos + w_neg * m_neg = q      the OAI subset carries OUR base rate, so a
                                           domain intercept buys ~nothing and the only way
                                           to reduce the OAI loss is to discriminate WITHIN
                                           the 246 -- the rank information AUC reads

(For hard 0/1 labels this reduces to the familiar w_pos = q/p, w_neg = (1-q)/(1-p); the
two-moment form is what keeps both properties EXACT now that the labels are soft.)
Degenerate targets (one-sided, or numerically singular) fall back to 1/1 -- the label mask
has already bounded what they can do. A weight outside (0, WEIGHT_CAP] raises: clamping
would silently break both properties, and nothing in the default set comes near the cap.

------------------------------------------------------------------------------------------
DECISION 3 -- THE LOSS WEIGHT.
------------------------------------------------------------------------------------------
Base weight 1.0 per cell -- never the teacher's reliability vector (defect 4 above). The
ladder labels are soft, so the old "every OAI cell is a hard label whose pull never
saturates" doubling no longer applies; the honest budget is the nominal weighted mass, which
the verify script measures against the competition total on the real tables (printed there,
quoted in the patch doc). Default --oai-weight 1.0 on the weak four. Deliberately NOT lower:
a sub-1% mass would produce an uninterpretable null at the same GPU cost as an informative
run.

DELIBERATELY NOT UPSAMPLED. 246 knees repeated R times carry no more information and give
the domain shortcut R times the opportunity. The mixing ratio is whatever 246 against the
fold's ~3,525 produces; the only knob is the weight.

------------------------------------------------------------------------------------------
DECISION 4 -- FOLDS. All 246 knees are in the train set of EVERY fold.
------------------------------------------------------------------------------------------
They are external, so they cannot leak into our validation labels; they are never scored
(`attach_oai` touches the train dataset only, and the OOF is written from va_ids); and no
OAI id is a competition StudyInstanceUID -- attach_oai additionally REFUSES to run if any
OAI id already appears in the training fold table, which is the tripwire against someone
later merging folds_oai.csv into folds.csv. The ids are sorted so composition is
deterministic under --seed. One consequence stated out loud: the five fold-models now share
246 training knees, which makes them marginally less independent as an ensemble.

------------------------------------------------------------------------------------------
DECISION 5 -- EXPOSURE (--oai-frac), the knob that only exists at full scale.
------------------------------------------------------------------------------------------
At 246 rows against ~3,525 the mixing ratio took care of itself; at 7,992 rows natural
ConcatDataset mixing would make OAI 69.4% of samples and ~48.9% of weighted mass -- a
majority-OAI model wearing a competition model's name. The missing knob is EXPOSURE, and it
is a hard structural cap, not a preference: OAIStudies reports len() = N_epoch, so the
sampler CANNOT draw more OAI rows per epoch than the cap, whatever else changes.

    N_epoch = round(oai_frac x len(competition train rows)),  capped at the pool size

THE DEFAULT, 0.125, IS THE LOW SWEEP ARM (~5% of weighted mass), derived so (all constants
measured 2026-09-07 on the real tables -- verify_oai_aux.py re-measures them):

    per-fold competition train rows (mean over 5 folds)     N_c = 3,525.6
    per-fold competition weighted cell mass                 M_c = 30,196
        (consensus labels; per-cell teacher weight tw, gold-58 at their 8x override;
         identical to 3 significant figures on the synfix table)
    OAI weak-4 weighted mass per row                        m   = 3.6097
        (the reprice is MEAN-PRESERVING -- mean class weight over asserted cells is
         exactly 1 per target -- so a row's expected mass is just its asserted weak-4
         cell count: (7,992+7,992+7,989+4,876)/7,992 = 3.61. Full-table reprice weights,
         consensus q, all comfortably inside the (0, 4] refusal cap:
         Lateral Meniscus 0.639/1.075, Lateral OA 0.310/1.559, PF OA 0.444/2.394,
         Synovitis 0.576/1.618)

    5% of mass:  N x m = (0.05/0.95) x M_c  =>  N = 440.3  =>  F* = 440.3/3,525.6 = 0.1249

    F = 0.125  ->  441 rows/epoch, 11.1% of samples, 5.01% of weighted mass (per-fold
                   arithmetic; the addendum's "F=0.15 -> 4.9%" divided by the FULL-table
                   mass -- per-fold, 0.15 is 5.95%, so 0.125 is the honest 5% arm)
    F = 0.50   ->  1,763 rows/epoch, 33.3% of samples, 17.4% of mass (the high arm, D43b)

REFUSE-ABOVE-HALF TRIPWIRE. A frac outside (0, 0.5] RAISES -- 0.5 is the pre-registered
high arm and the largest value at which competition rows still outnumber OAI 2:1. A
second, unconditional guard raises if the attached rows would exceed the competition rows
in an epoch (OAI the majority of samples); with the frac guard intact it is unreachable,
which is the point -- it survives someone editing OAI_FRAC_MAX.

PER-EPOCH REDRAW WITHOUT SEEING EPOCHS. The subsample must rotate (a static 441-row subset
would waste 94% of the only radiologist supervision we own), but train.py builds its
loaders with persistent_workers=True and PyTorch neither re-pickles the dataset nor
re-seeds workers between epochs (torch 2.9 _BaseDataLoaderIter._reset), so a map-style
dataset cannot observe an epoch boundary from inside a worker. The scheduler is therefore
epoch-FREE with the same guarantees: at first access each worker takes a disjoint,
participant-whole shard of the pool (participants shuffled once under the attach seed,
dealt round-robin by worker id -- both knees and all visits of a participant travel
together, the owner-recorded patient-level grouping); each worker then walks its shard in
a seeded order, reshuffling AT PARTICIPANT LEVEL each time the shard is exhausted. Every
epoch the sampler pulls exactly N_epoch OAI items (that is len()), so per epoch the model
sees N_epoch rows, no row repeats inside one shard pass, and after pool/N_epoch ~= 18.1
epochs every one of the 7,992 rows has trained at least once -- inside a 28-epoch run each
row is seen ~1.55x. Deterministic: the stream is a pure function of (--seed, the fold's
train ids, worker id, pass number); like the slice-window jitter it also depends on
--workers, and on nothing else. The pilot path is untouched: when the cap covers the whole
pool (246 < 441) the scheduler never engages and behaviour is bit-identical to D42.

THE SLOT-DISTRIBUTION SHIFT, QUANTIFIED (masks of both caches, r1's slot_drop 0.25).
Competition present-slot histogram {2: 224, 3: 235, 4: 1,615, 5: 1,995, 6: 338}; all
7,992 OAI masks are slot-0-only (scanned, not assumed). Exact enumeration of model.py's
drop mechanics (Bernoulli 0.25 per present slot, at most min(2, n-1) dropped):

                              control (r1)     F = 0.125      F = 0.50
    single-slot sample share      3.06%          13.83%         35.37%
    competition share of samples 100%            88.9%          66.7%
    slot-0-only purity (OAI)       --            98.35%         99.58%

The addendum's design premise -- competition >= 2/3 of samples at every arm -- holds at
both arms, and its hazard statement ("one slot, and it is SAG_FLUID_FS" stays a ~99%-pure
OAI flag at ANY mixing ratio) was already priced in: the reprice bounds what the flag can
buy and the exposure cap bounds the drift, with the slot-stratified TA readout as the
run-time detector. The default stays 0.125. What DID move: ~9.2 slot-0-only competition
presentations per full-data epoch under slot_drop 0.25 (vs ~3.6 at G1's 0.15), so the flag
is 98.4% rather than 99.9% pure -- direction favourable, magnitude irrelevant.

VERIFY: pending_patches/verify_oai_aux.py (CPU, no GPU) -- and pending_patches/
corrupt_oai_demo.py, which proves the verifier fails on a zeroed volume, slot-permuted
pixels, an invented confident negative, and a leaked fold id.
"""

from __future__ import annotations

import os
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, get_worker_info

from model import KneeStudies, TARGETS
from oai_ladder import NEGATIVE_RUNGS  # single source of the rung split; oai_ladder.py is
                                       # at the repo root, beside where this file lands.

OAI_CACHE = Path(r"data/oai_cache")
LADDER_LABELS = "oai_cache_labels_ladder.csv"

# DECISION 5's knob. The default is the LOW sweep arm, derived in the docstring above
# (0.125 x per-fold train rows ~= 441 rows/epoch ~= 5.01% of weighted mass); anything
# outside (0, OAI_FRAC_MAX] is refused, never clamped.
DEFAULT_OAI_FRAC = 0.125
OAI_FRAC_MAX = 0.5

# The four targets target_adapter.py measured as the whole remaining headroom. MOAKS covers
# all four.
WEAK = ("Lateral Meniscus", "Lateral OA", "PF OA", "Synovitis")
# Everything the ladder table asserts. MCL/Fracture are unscored by MOAKS; Contusion is
# forced silent (semantic mismatch) -- selecting any of the three raises in build_target_set.
EXCLUDED = ("MCL", "Fracture", "Contusion")
SAFE = tuple(t for t in TARGETS if t not in EXCLUDED)

# Any repriced class weight outside (0, WEIGHT_CAP] raises -- see DECISION 2.
WEIGHT_CAP = 4.0

NOT_ASSERTED = {"silent", "not_mentioned"}


def build_target_set(spec):
    """'weak' | 'safe' | 'Name,Other Name' -> list of target names."""
    if spec == "weak":
        names = list(WEAK)
    elif spec == "safe":
        names = list(SAFE)
    else:
        names = [s.strip() for s in str(spec).split(",") if s.strip()]
    bad = [n for n in names if n not in TARGETS]
    if bad:
        raise ValueError(f"unknown OAI target(s) {bad}; known: {list(TARGETS)}")
    banned = [n for n in names if n in EXCLUDED]
    if banned:
        raise ValueError(
            f"OAI target(s) {banned} cannot be selected: MCL and Fracture are not scored "
            f"by MOAKS, and Contusion is forced silent because MOAKS BML is a different "
            f"disease than a report contusion (see oai_ladder.py rule 4). Selecting them "
            f"would train nothing while looking like it did.")
    return names


def validate_ladder(oai):
    """Refuse a label table that could reintroduce the invented-confident-negative defect."""
    need = ["StudyInstanceUID"] + TARGETS + [t + "__state" for t in TARGETS]
    missing = [c for c in need if c not in oai.columns]
    if missing:
        raise ValueError(f"OAI label table is missing columns {missing[:4]}...")
    for t in TARGETS:
        asserted = ~oai[t + "__state"].isin(NOT_ASSERTED)
        v = pd.to_numeric(oai.loc[asserted, t], errors="coerce")
        if ((v <= 0.0) | (v >= 1.0)).any() or v.isna().any():
            raise ValueError(
                f"{t}: asserted cell at hard 0/1 (or NaN). This is the binarised "
                f"oai_cache_labels.csv defect -- 23.6% of its WEAK-4 mass was a confident "
                f"0.0 on a knee a radiologist graded abnormal. Run `python oai_ladder.py` "
                f"and use {LADDER_LABELS}.")
        if t in EXCLUDED and asserted.any():
            raise ValueError(
                f"{t} is asserted on {int(asserted.sum())} OAI rows; it must be silent on "
                f"all of them (MCL/Fracture unscored; Contusion semantic mismatch).")


def _soft_prevalence(df, t):
    """Mean asserted soft label -- the label mass per cell."""
    m = ~df[t + "__state"].isin(NOT_ASSERTED)
    v = pd.to_numeric(df.loc[m, t], errors="coerce").dropna().values
    return float(v.mean()) if len(v) else np.nan


def class_weights(comp_labels, oai_labels, targets, cap=WEIGHT_CAP):
    """Per-target (w_pos, w_neg) that make the OAI subset carry OUR base rate.

    Exact by construction (see DECISION 2): mean weight over asserted OAI cells is 1, and
    the weighted mean label equals the competition prevalence q. Classes are split by RUNG
    (NEGATIVE_RUNGS from oai_ladder), not by value.
    """
    wp = np.ones(len(TARGETS), np.float32)
    wn = np.ones(len(TARGETS), np.float32)
    for t in targets:
        i = TARGETS.index(t)
        q = _soft_prevalence(comp_labels, t)
        st = oai_labels[t + "__state"]
        asserted = ~st.isin(NOT_ASSERTED)
        if not asserted.any() or not np.isfinite(q):
            continue
        y = pd.to_numeric(oai_labels.loc[asserted, t], errors="coerce").values
        pos = (~st[asserted].isin(NEGATIVE_RUNGS)).values
        f_pos, f_neg = float(pos.mean()), float((~pos).mean())
        m_pos, m_neg = float((y * pos).mean()), float((y * ~pos).mean())
        det = m_pos * f_neg - m_neg * f_pos
        if f_pos in (0.0, 1.0) or abs(det) < 1e-9:
            continue                       # one-sided target: nothing to rebalance
        w_pos = (q * f_neg - m_neg) / det
        w_neg = (m_pos - q * f_pos) / det
        if not (0.0 < w_pos <= cap and 0.0 < w_neg <= cap):
            raise ValueError(
                f"{t}: repriced class weights w_pos {w_pos:.3f} / w_neg {w_neg:.3f} fall "
                f"outside (0, {cap}]. The OAI/competition prior gap is too extreme for a "
                f"two-class reprice; clamping would silently break the mean-preserving and "
                f"reprice properties, so this refuses instead.")
        wp[i], wn[i] = w_pos, w_neg
    return wp, wn


def cell_weight_matrix(oai_labels, ids, targets, wp, wn):
    """(N, 12) per-cell weight, rows aligned to `ids`. Positive-rung cells get w_pos,
    everything else w_neg; masked cells get 1.0 (their label mask is 0, so the value is
    inert -- 1.0 rather than 0.0 so a masking bug upstream cannot hide here)."""
    lab = oai_labels.set_index("StudyInstanceUID")
    keep = np.zeros(len(TARGETS), np.float32)
    for t in targets:
        keep[TARGETS.index(t)] = 1.0
    w = np.ones((len(ids), len(TARGETS)), np.float32)
    for j, t in enumerate(TARGETS):
        if t not in targets:
            continue
        st = lab[t + "__state"].reindex(ids)
        asserted = ~st.isin(NOT_ASSERTED)
        pos = ~st.isin(NEGATIVE_RUNGS)
        w[:, j] = np.where(asserted, np.where(pos, wp[j], wn[j]), 1.0)
    return w, keep


def epoch_rows_for(frac, n_comp, n_pool):
    """DECISION 5's arithmetic in one place: how many OAI rows one epoch may hold.

    `frac` None means the flag was omitted -> DEFAULT_OAI_FRAC (the low arm). The
    REFUSE-ABOVE-HALF tripwire lives here: a frac outside (0, OAI_FRAC_MAX] raises --
    0.9 is refused, 0.5 (the pre-registered high arm) is the largest expressible value.
    Never clamped: a clamp would run a different experiment than the one on the command
    line and label the OOF with the requested number.
    """
    f = DEFAULT_OAI_FRAC if frac is None else float(frac)
    if not np.isfinite(f) or not (0.0 < f <= OAI_FRAC_MAX):
        raise ValueError(
            f"--oai-frac {f} is outside (0, {OAI_FRAC_MAX}]. Above {OAI_FRAC_MAX} the "
            f"competition rows would stop outnumbering OAI 2:1 and the training "
            f"distribution would no longer be competition-dominated -- the design premise "
            f"of the whole OAI lane (oai_aux.py DECISION 5). Refused, not clamped. "
            f"Sweep arms: {DEFAULT_OAI_FRAC} (low, ~5% of weighted mass) and 0.5 (high).")
    cap = int(round(f * n_comp))
    if cap < 1:
        raise ValueError(
            f"--oai-frac {f} with {n_comp} competition train rows admits ZERO OAI rows "
            f"per epoch; the run would silently be a control wearing the OAI flags. "
            f"Drop --oai-weight instead if that is what you want.")
    return min(cap, int(n_pool))


def refuse_majority(n_epoch, n_comp):
    """The unconditional second face of the tripwire: OAI rows may never exceed 50% of an
    epoch's samples, i.e. never outnumber the competition rows. With epoch_rows_for's
    (0, 0.5] refusal intact this is unreachable (n_epoch <= round(0.5 x n_comp) <=
    n_comp); it exists so that editing OAI_FRAC_MAX, or a future 'keep every row' mode,
    still cannot produce a majority-OAI epoch."""
    if int(n_epoch) > int(n_comp):
        raise ValueError(
            f"{n_epoch} OAI rows against {n_comp} competition rows would make OAI the "
            f"MAJORITY of an epoch's samples. The competition distribution must dominate "
            f"training at any --oai-frac (oai_aux.py DECISION 5); refusing.")


def _participant(sid):
    """oaifull_9NNNNNN_00M_R -> '9NNNNNN' (participant); an id without visit structure
    (oaizib_002, the pilot namespace) groups alone -- one knee, one participant."""
    p = str(sid).split("_")
    return p[1] if len(p) >= 3 else str(sid)


class OAIStudies(KneeStudies):
    """KneeStudies over the OAI cache, with per-target keep-mask and per-cell class weights.

    target_weights is passed as None ON PURPOSE: KneeStudies then uses ones, so an OAI cell's
    base weight is 1.0 -- its own, never the LLM teacher's reliability vector (defect 4).
    Everything else -- the slot-0-only mask, silent-cell masking, window jitter, collate
    shape -- is inherited unchanged, because build_oai_cache.py wrote the OAI cache to
    build_cache.py's contract and the ladder table uses the same "__state" convention.
    """

    # train.py's resume bookkeeping finds each external source's rotation by this tag, not by
    # its position in the ConcatDataset (ext_aux.ExtStudies, a second source, is "ext2").
    source = "oai"

    def __init__(self, ids, keep, cell_w, oai_weight=1.0, epoch_rows=None, frac_seed=0,
                 **kw):
        kw.pop("target_weights", None)     # structurally impossible to smuggle tw in
        super().__init__(ids, target_weights=None, **kw)
        self.keep = torch.as_tensor(keep, dtype=torch.float32)
        self.cell_w = torch.as_tensor(cell_w, dtype=torch.float32)
        self.oai_weight = float(oai_weight)
        # DECISION 5. epoch_rows < pool engages the exposure scheduler: __len__ reports
        # epoch_rows (the HARD cap -- the sampler cannot ask for more), and __getitem__
        # ignores its index and pops the next row off a per-worker rotation instead.
        # epoch_rows >= pool (the pilot: 246 < any sane cap) stores None and every path
        # below is bypassed -- behaviour bit-identical to the pre-frac module.
        self.epoch_rows = (int(epoch_rows)
                           if epoch_rows is not None and int(epoch_rows) < len(self.ids)
                           else None)
        self.frac_seed = int(frac_seed) & 0xFFFFFFFF

    def __len__(self):
        n = getattr(self, "epoch_rows", None)
        return int(n) if n else len(self.ids)

    # ---- the exposure scheduler (DECISION 5) -------------------------------------------
    # Epoch-FREE by necessity: train.py's loaders run persistent_workers=True, and PyTorch
    # neither re-pickles the dataset nor re-seeds a persistent worker between epochs, so
    # nothing in here can observe an epoch boundary. Instead each worker owns a disjoint,
    # participant-whole shard and walks it in seeded passes; the per-epoch guarantee is
    # enforced from OUTSIDE by __len__ (the sampler draws exactly len() OAI indices per
    # epoch). State is process-local and lazily built AFTER unpickling (the pid check), so
    # the parent's copy never leaks half-advanced state into a worker.

    def _init_stream(self):
        info = get_worker_info()
        wid, W = (info.id, info.num_workers) if info is not None else (0, 1)
        force = getattr(self, "_force_worker", None)   # verify_oai_aux.py only
        if force is not None:
            wid, W = force
        by_part = {}
        for row, sid in enumerate(self.ids):
            by_part.setdefault(_participant(sid), []).append(row)
        parts = sorted(by_part)                        # deterministic base order
        deal = np.random.default_rng(self.frac_seed).permutation(len(parts))
        mine = [parts[k] for k in deal[wid::W]]
        if not mine:                                   # more workers than participants --
            mine = [parts[k] for k in deal]            # degenerate; overlap beats starving
        self._stream = {"pid": os.getpid(), "wid": wid, "rows": by_part, "parts": mine,
                        "pass": -1, "order": np.empty(0, np.int64), "k": 0}
        return self._stream

    def _next_row(self):
        st = getattr(self, "_stream", None)
        if st is None or st.get("pid") != os.getpid():
            st = self._init_stream()
        if st["k"] >= len(st["order"]):
            st["pass"] += 1
            rng = np.random.default_rng((self.frac_seed, st["wid"], st["pass"]))
            st["order"] = np.concatenate(
                [np.asarray(st["rows"][st["parts"][k]], np.int64)
                 for k in rng.permutation(len(st["parts"]))])
            st["k"] = 0
        row = int(st["order"][st["k"]])
        st["k"] += 1
        return row

    def __getitem__(self, i):
        # getattr, not self.X: Windows DataLoader workers re-import the module and unpickle
        # a dataset built by the parent. Same seatbelt model.py uses for slot_idx/slot_drop.
        if getattr(self, "epoch_rows", None) is not None:
            i = self._next_row()           # DECISION 5: index -> next row of the rotation
        g, m, y, lm, w = super().__getitem__(i)
        keep = getattr(self, "keep", None)
        if keep is None:
            return g, m, y, lm, w
        # lm * keep -- an OAI knee contributes gradient ONLY where MOAKS read the target AND
        # we selected it. w is ones (see __init__), times the per-cell class weight, times
        # --oai-weight. Cells with lm = 0 are multiplied out inside distill_loss regardless.
        return g, m, y, lm * keep, w * self.cell_w[i] * self.oai_weight


def attach_oai(train_ds, args, comp_labels, cache=OAI_CACHE):
    """Return ConcatDataset([competition train, OAI]) -- or train_ds unchanged if disabled.

    Called from build_loaders on the TRAIN dataset only. There is deliberately no argument
    that could route these knees to a validation loader, and no argument that could hand
    them the teacher's weight vector.

    `comp_labels` is the DataFrame train.py has already read; it is passed in rather than
    re-read from args.labels so the prevalence used for repricing is by construction the
    SAME table the competition half of the batch is supervised from.
    """
    if getattr(args, "oai_weight", 0.0) <= 0:
        return train_ds
    cache = Path(getattr(args, "oai_cache", None) or cache)
    # --oai-labels names the ladder table EXPLICITLY. The full-scale run must pass it
    # (data/oai_cache/oai_full_labels_ladder.csv -- the per-visit build):
    # the 7,992-row oai_full_cache is not described by any table inside the cache dir,
    # and the pilot's visit-mixed table must never be attached to it. Default (None)
    # keeps the pilot rule, <cache>/oai_cache_labels_ladder.csv, unchanged.
    lab_path = Path(getattr(args, "oai_labels", None) or (cache / LADDER_LABELS))
    if not lab_path.exists():
        raise FileNotFoundError(
            f"{lab_path} missing -- run `python oai_ladder.py` (it also writes "
            f"folds_oai.csv) before --oai-weight")

    oai = pd.read_csv(lab_path)
    validate_ladder(oai)
    targets = build_target_set(getattr(args, "oai_targets", "weak"))

    # Tripwire for the fold leak: if an OAI id ever appears in the fold table train.py
    # splits on, it can also appear in a VALIDATION split we cannot see from here. Refuse.
    overlap = set(map(str, getattr(train_ds, "ids", []))) & set(
        oai.StudyInstanceUID.astype(str))
    if overlap:
        raise ValueError(
            f"{len(overlap)} OAI ids (e.g. {sorted(overlap)[0]}) are already in the "
            f"training fold table. OAI knees must never enter folds.csv -- under "
            f"--random-folds its fold column is permuted and train-only rows would land in "
            f"validation folds. Remove them and use attach_oai as the only OAI path.")

    # The DESS occupies slot 0 only. Restricting --slots to a set excluding 0 would hand
    # every OAI knee an all-absent mask: the masked softmax comes back UNIFORM over six zero
    # vectors, nothing crashes, and 246 knees become pure label noise at full weight. Refuse.
    sidx = ([int(x) for x in args.slots.split(",")] if getattr(args, "slots", None) else None)
    if sidx is not None and 0 not in sidx:
        raise ValueError(
            f"--slots {args.slots} excludes slot 0, which is the only slot the OAI cache "
            f"fills; every OAI knee would be an empty study. Drop --oai-weight or keep slot 0.")

    ids = sorted(oai.StudyInstanceUID.astype(str))          # deterministic composition

    # The label table and the cache dir must describe the SAME build. With --oai-labels
    # naming the table explicitly, a mismatched --oai-cache (the pilot's visit-mixed table
    # against the 7,992-volume full cache, or the reverse) would otherwise survive attach
    # and die at the first worker fetch, an hour into a run. Probe one volume now.
    probe = cache / f"{ids[0]}.npy"
    if not probe.exists():
        raise FileNotFoundError(
            f"{probe} missing: the label table ({lab_path.name}, ids like {ids[0]}) does "
            f"not describe the cache at {cache}. Pair the pilot table with the pilot "
            f"cache and oai_full_labels_ladder.csv with oai_full_cache.")

    # DECISION 5: exposure. The frac (default = the low arm) caps how many OAI rows one
    # epoch may hold; epoch_rows_for refuses a frac outside (0, 0.5], refuse_majority is
    # the unconditional belt behind it. The rotation seed folds --seed together with the
    # fold's own train ids, so each fold walks the pool in its own order while any single
    # fold is exactly reproducible under --seed.
    n_comp = len(train_ds)
    n_epoch = epoch_rows_for(getattr(args, "oai_frac", None), n_comp, len(ids))
    refuse_majority(n_epoch, n_comp)
    frac_seed = zlib.crc32(
        f"{int(getattr(args, 'seed', 0))}|".encode()
        + ",".join(sorted(map(str, getattr(train_ds, "ids", [])))).encode())

    wp, wn = class_weights(comp_labels, oai, targets)
    cell_w, keep = cell_weight_matrix(oai, ids, targets, wp, wn)

    ds = OAIStudies(
        ids, keep, cell_w, oai_weight=args.oai_weight,
        epoch_rows=n_epoch, frac_seed=frac_seed,
        labels=oai, cache=cache, n_group=args.n_group, img=args.img, train=True,
        # window length must match the competition half of the batch or the collate fails;
        # 3 (KneeStudies' default) for train.py, 2 for train_vlm.py.
        group=int(getattr(args, "group", None) or 3),
        # No gold override can apply: an oaizib_NNN id is never a competition study id, and
        # an empty dict makes that structural rather than incidental.
        gold_labels={}, gold_ids=set(), gold_weight=1.0,
        slot_idx=sidx,
        # slot_drop OFF for OAI. These knees hold ONE slot; model.py's guard (never the last
        # remaining slot) already makes dropping a no-op, so this only says so out loud.
        slot_drop=0.0)

    print(f"  OAI aux: {len(ids)} knees, weight {args.oai_weight}, targets {targets}")
    frac_shown = getattr(args, "oai_frac", None)
    print(f"  exposure: {len(ds)} of {len(ids)} rows per epoch "
          f"(--oai-frac {DEFAULT_OAI_FRAC if frac_shown is None else frac_shown}"
          f"{' = default low arm' if frac_shown is None else ''}), "
          f"{len(ds) / (n_comp + len(ds)):.1%} of samples"
          + ("" if len(ds) == len(ids) else
             f"; rotation covers the pool every ~{len(ids) / len(ds):.1f} epochs"))
    print(f"  base weight 1.0 (own, NOT the teacher's reliability vector), then repriced:")
    for t in targets:
        i = TARGETS.index(t)
        print(f"    {t:<18} w_pos {wp[i]:.3f}  w_neg {wn[i]:.3f}")
    return ConcatDataset([train_ds, ds])
