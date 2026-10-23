"""ext_aux.py -- EXTERNAL auxiliary-supervision sources beside OAI, behind --ext2-* and --ext3-*.

WHAT THEY CARRY TODAY
ext2  SKM-TEA (build_skmtea_cache.py): 145 labelled knees in oai_full_cache's exact contract
      (qDESS E1 in slot 0) and the ladder label schema, reaching seven targets: ACL, Medial /
      Lateral Meniscus, Medial / Lateral OA, PF OA, Effusion. Provenance, stated precisely
      because it is the axis this project ranks label sources on: each box is a CLINICAL-REPORT
      finding that annotators localised and graded on the images under two MSK radiologists'
      supervision (SKM-TEA paper, section 4.1) -- image-checked report labels, not an
      independent image read like MOAKS.
ext3  MRNet (build_mrnet_cache.py): 1,250 Stanford clinical knee MRIs, three planes in the
      v1 contract (sagittal T2 FS -> slot 0, coronal T1 -> slot 4, axial PD FS -> slot 2).
      Labels HAND-EXTRACTED FROM THE CLINICAL REPORTS (the same kind of label as our own
      teacher, from a different population): ACL tear, and "a meniscal tear" WITHOUT its side,
      which trains through the any-of target below. 'abnormal' is not used.
Nothing here is source-specific except SOURCES (default cache, default exposure) and PRESETS.

THE MECHANISM IS oai_aux's, IMPORTED, NOT RETYPED
  ExtStudies IS OAIStudies (KneeStudies + per-target keep mask + per-cell class weight + the
  participant-level exposure rotation), tagged with its source ("ext2" / "ext3") so train.py's
  resume bookkeeping can tell the rotations apart. target_weights is structurally None: an
  external cell's base weight is 1.0, never the LLM teacher's reliability vector (oai_aux
  defect 4).
  class_weights / cell_weight_matrix: the two-moment reprice to the competition's soft
  prevalence, mean-preserving, refusing weights outside (0, 4] (oai_aux DECISION 2). It
  matters for every source: an SKM knee presents the slot-0-only DESS mask an OAI knee does,
  an MRNet knee the exact {0, 2, 4} mask (1 competition study holds it; slot dropout shows it
  on ~94 competition rows per fold-epoch against ~369 MRNet rows) -- each a free "this is an
  external knee" flag. Repriced per source, the flag buys an intercept in none of them.

WHAT DIFFERS FROM attach_oai, AND WHY
  1 EXPOSURE IS A FRACTION OF ITS OWN POOL (--extN-frac).
    --oai-frac is a fraction of the COMPETITION rows because 7,992 OAI rows would otherwise be
    69% of an epoch. ext2 defaults to 1.0: 145 SKM rows are 4.1% of a fold's ~3,525
    competition rows, a rotation would only throw away part of a tiny pool, and oai_aux's own
    rule stands -- never upsampled. ext3 defaults to 0.3: 1,250 MRNet rows at 1.0 would make
    the epoch [3,525 competition | 1,763 OAI | 145 SKM | 1,250 MRNet], competition 52.7% of
    samples; at 0.3 (375 rows, the pool covered every ~3.3 epochs) competition stays at 60.7%
    and MRNet carries ~10% of the ACL and meniscus heads' weighted mass -- a dose that can
    show, not one that can take the heads over. Below 1 engages OAIStudies' participant-level
    rotation (exact under --resume); refused outside (0, 1].
  2 TARGETS: a preset ('skm', 'mrnet'), 'asserted' (whatever the table asserts), or a comma
    list. A selected target the table never asserts is REFUSED -- it would train nothing while
    looking like it did (oai_aux's Contusion rule, made data-driven).
  3 TRIPWIRES, stronger than attach_oai's: an external id may appear in neither the
    competition label table (every study folds.csv can ever hold, so no fold -- not just this
    fold's train side -- can route one into validation), nor the fold's train ids, nor any
    source already attached. And the external rows of an epoch, OAI + ext2 + ext3 together,
    may never outnumber the competition rows (oai_aux.refuse_majority, extended to every
    source; checked again at each attach).
  4 STRUCTURE: each source APPENDS, flattening the ConcatDataset it is handed, so datasets[0]
    stays the competition dataset (run_fold's n_comp) and each source holds one contiguous
    index range -- [comp | oai | ext2 | ext3] -- which train.py's per-source resume counts rely
    on. Like attach_oai it is handed the TRAIN dataset only, after build_loaders has built
    `va`: there is no argument that could route an external knee to a validation loader, an
    OOF, a gold AUC or a model-selection decision.
  5 --extN-weight 0 (the default) returns the dataset object unchanged: bit-identical runs.

THE ANY-OF TARGET ("Any Meniscus"; model.ANY_OF names its heads)
  A label table may carry a column named after a model.ANY_OF group, plus its __state. It is
  asserted (a POSITIVE rung) exactly on the rows whose finding is known to be in SOME head of
  the group but not which -- MRNet's meniscal tears -- and those rows keep the member heads'
  own cells SILENT. A row known NEGATIVE for the whole group asserts the member cells directly
  (absent: "neither meniscus is torn" is two ordinary negatives, and for a negative the
  noisy-OR loss IS the sum of the per-head losses -- log(1 - P) = log(1 - p_MM) + log(1 - p_LM)
  -- so the per-cell form loses nothing and keeps the ladder's absent rung per head instead of
  the noisy-OR's ~half of it). validate_table enforces that layout.
  Selecting the group selects its member heads too (their negative cells) and adds the
  noisy-OR term for the positive rows; ExtStudies emits those rows in model.any_of_terms'
  encoding and train.py switches to model.distill_loss_any. Selecting a member head WITHOUT
  its group is refused: in such a table the member is one-sided (negatives only), so it could
  teach nothing but an intercept.
  REPRICE (any_of_reprice): the per-target two-moment reprice cannot price a group whose
  positives live in one term and negatives in two others, so the same two properties are
  solved for the group: the member cells and the OR terms together are MEAN-PRESERVING (weight
  1 per term on average), and the expected loss of a constant predictor is STATIONARY at the
  competition's soft prevalence of BOTH heads (q_MM, q_LM) -- for a single BCE target that is
  exactly class_weights' "weighted mean label = q", so this is its generalisation, not a new
  policy. Closed form in any_of_reprice; refused outside (0, 4] like class_weights.

ID CONVENTION (the rotation groups by oai_aux._participant = the SECOND '_' field):
  skmtea_{subject}_{scan}_{side}; mrnet_{exam}_{split}. A future source must keep its
  participant there too.
"""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset

from model import ANY_OF, TARGETS, KneeStudies
from oai_aux import (NOT_ASSERTED, WEIGHT_CAP, OAIStudies, _participant, _soft_prevalence,
                     cell_weight_matrix, class_weights)
from oai_ladder import NEGATIVE_RUNGS

EXT2_CACHE = Path(r"data/skmtea_cache")
EXT3_CACHE = Path(r"data/mrnet_cache")
EXT2_LABELS = "labels.csv"
DEFAULT_EXT2_FRAC = 1.0
DEFAULT_EXT3_FRAC = 0.3
# Per source: default cache, default --extN-frac, default --extN-targets, and the builder that
# writes the cache (named in the error a missing table raises).
SOURCES = {
    "ext2": dict(cache=EXT2_CACHE, frac=DEFAULT_EXT2_FRAC, targets="skm",
                 builder="build_skmtea_cache.py"),
    "ext3": dict(cache=EXT3_CACHE, frac=DEFAULT_EXT3_FRAC, targets="mrnet",
                 builder="build_mrnet_cache.py"),
}
PRESETS = {"skm": ("ACL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
                   "PF OA", "Effusion"),
           "mrnet": ("ACL", "Any Meniscus")}


class ExtStudies(OAIStudies):
    """OAIStudies over an external cache, tagged with its source, plus the any-of rows.

    any_y / any_w: per-row any-of target and weight (weight 0 = the row carries none), any_idx
    the group's TARGETS indices. Without them (every ext2 table) __getitem__ IS OAIStudies'."""
    source = "ext2"

    def __init__(self, ids, keep, cell_w, source="ext2", any_y=None, any_w=None, any_idx=None,
                 **kw):
        super().__init__(ids, keep, cell_w, **kw)
        self.source = source
        self.any_idx = tuple(any_idx) if any_idx is not None else None
        self.any_y = None if any_y is None else torch.as_tensor(any_y, dtype=torch.float32)
        self.any_w = None if any_w is None else torch.as_tensor(any_w, dtype=torch.float32)

    def __getitem__(self, i):
        # getattr, not self.X: Windows DataLoader workers unpickle a dataset the parent built.
        if getattr(self, "any_w", None) is None:
            return super().__getitem__(i)
        if getattr(self, "epoch_rows", None) is not None:
            i = self._next_row()           # the exposure rotation, exactly as OAIStudies
        g, m, y, lm, w = KneeStudies.__getitem__(self, i)
        lm, w = lm * self.keep, w * self.cell_w[i] * self.oai_weight
        if self.any_w[i] > 0:
            # model.any_of_terms' encoding: the group's own cells masked, the any-of target in
            # y[k], MINUS its weight in w[k]. y, lm, w are this item's own tensors.
            k = self.any_idx[0]
            lm[list(self.any_idx)] = 0.0
            y[k] = self.any_y[i]
            w[k] = -self.any_w[i]
        return g, m, y, lm, w


def _asserted(tab, t):
    return ~tab[t + "__state"].isin(NOT_ASSERTED)


def _positive(tab, t):
    return _asserted(tab, t) & ~tab[t + "__state"].isin(NEGATIVE_RUNGS)


def validate_table(tab):
    """The ladder invariants, for any external table: the schema KneeStudies reads, unique ids,
    asserted cells strictly inside (0, 1) (oai_aux's invented-confident-negative rule), finite
    silent cells (the mask MULTIPLIES; 0 * nan is nan and takes the batch with it) -- and, for
    an any-of column, the layout the docstring states: asserted only as a positive, only where
    every member cell is silent, and members asserted only as negatives."""
    need = ["StudyInstanceUID"] + TARGETS + [t + "__state" for t in TARGETS]
    missing = [c for c in need if c not in tab.columns]
    if missing:
        raise ValueError(f"external label table is missing columns {missing[:4]}...")
    if tab.StudyInstanceUID.duplicated().any():
        raise ValueError("external label table repeats a StudyInstanceUID")
    for t in TARGETS + [g for g in ANY_OF if g in tab.columns]:
        if t + "__state" not in tab.columns:
            raise ValueError(f"any-of column {t!r} has no {t}__state column")
        a = _asserted(tab, t)
        v = pd.to_numeric(tab[t], errors="coerce")
        if ((v[a] <= 0.0) | (v[a] >= 1.0)).any() or v[a].isna().any():
            raise ValueError(f"{t}: asserted cell at hard 0/1 (or NaN) -- ladder values "
                             f"only (see oai_ladder.py rule 2)")
        if not np.isfinite(v[~a]).all():
            raise ValueError(f"{t}: a silent cell holds NaN/inf")
    for g, members in ANY_OF.items():
        if g not in tab.columns:
            continue
        a = _asserted(tab, g)
        if (a & ~_positive(tab, g)).any():
            raise ValueError(f"{g}: asserted with a negative rung -- an any-of row is a "
                             f"positive; a negative is written into the member cells")
        for t in members:
            if (a & _asserted(tab, t)).any():
                raise ValueError(f"{g}: a row asserts both the group and its member {t!r}; "
                                 f"the member cells must be silent where the group is asserted")
            if _positive(tab, t).any():
                raise ValueError(f"{t}: asserted POSITIVE in a table that carries {g!r} -- a "
                                 f"lateralised positive belongs in an ordinary table")


def build_ext_targets(spec, tab):
    """preset | 'asserted' | 'Name,Other Name' -> list of names (TARGETS and/or model.ANY_OF
    groups), refusing names the table never asserts and a group member without its group."""
    if spec in PRESETS:
        names = list(PRESETS[spec])
    elif spec == "asserted":
        names = [t for t in TARGETS if _asserted(tab, t).any()]
        names += [g for g in ANY_OF if g in tab.columns and _asserted(tab, g).any()]
        names = [t for t in names if not any(t in ANY_OF[g] for g in names if g in ANY_OF)]
    else:
        names = [s.strip() for s in str(spec).split(",") if s.strip()]
    bad = [n for n in names if n not in TARGETS and n not in ANY_OF]
    if bad:
        raise ValueError(f"unknown external target(s) {bad}; known: {list(TARGETS)} and the "
                         f"any-of groups {list(ANY_OF)}")
    dead = [n for n in names if n not in tab.columns or not _asserted(tab, n).any()]
    if dead:
        raise ValueError(f"external target(s) {dead} are silent on every row of the table: "
                         f"selecting them would train nothing while looking like it did.")
    for g, members in ANY_OF.items():
        if g in tab.columns and g not in names and any(t in names for t in members):
            raise ValueError(f"{[t for t in members if t in names]} selected without {g!r}: "
                             f"in this table their positives live in {g!r}, so alone they "
                             f"are negatives only and could teach nothing but an intercept.")
    if sum(n in ANY_OF for n in names) > 1:
        raise ValueError("more than one any-of group selected; the loss encoding carries one")
    if not names:
        raise ValueError("no external targets selected")
    return names


def any_of_reprice(comp_labels, tab, group, cap=WEIGHT_CAP):
    """-> ({member: weight of its asserted (negative) cells}, weight of the any-of terms).

    Solved exactly (docstring, THE ANY-OF TARGET). With n_k asserted member-k cells of mean
    label u_k, N any-of rows of mean target v, q_k the competition's soft prevalence of head k
    and P = 1 - prod(1 - q_k), the expected loss of a constant predictor has gradient on head k

        n_k w_k (q_k - u_k)  +  N w_or q_k (P - v) / P

    so stationarity at (q_k) gives w_k = N w_or q_k (v - P) / (P n_k (q_k - u_k)), and
    mean-preservation (sum_k n_k w_k + N w_or = sum_k n_k + N) fixes w_or."""
    members = ANY_OF[group]
    on = _asserted(tab, group)
    N = int(on.sum())
    v = float(pd.to_numeric(tab.loc[on, group]).mean()) if N else np.nan
    n, u, q = [], [], []
    for t in members:
        a = _asserted(tab, t)
        n.append(int(a.sum()))
        u.append(float(pd.to_numeric(tab.loc[a, t]).mean()) if a.any() else np.nan)
        q.append(_soft_prevalence(comp_labels, t))
    if N == 0 or min(n) == 0:
        raise ValueError(f"{group}: needs both its positive rows ({N}) and member negatives "
                         f"({dict(zip(members, n))}) to be repriced; one-sided, refused")
    P = 1.0 - float(np.prod([1.0 - x for x in q]))
    a = [N * qk * (v - P) / (P * nk * (qk - uk)) for qk, nk, uk in zip(q, n, u)]
    w_or = (sum(n) + N) / (sum(nk * ak for nk, ak in zip(n, a)) + N)
    w = {t: ak * w_or for t, ak in zip(members, a)}
    if not all(0.0 < x <= cap for x in list(w.values()) + [w_or]):
        raise ValueError(f"{group}: any-of reprice weights {w} / {w_or:.3f} fall outside "
                         f"(0, {cap}] (competition P(any) {P:.3f} vs the table's positive "
                         f"label {v:.3f}); clamping would break both properties, refused.")
    return w, float(w_or)


def ext_epoch_rows(frac, n_pool, default=DEFAULT_EXT2_FRAC, flag="--ext2-frac"):
    """--extN-frac -> rows per epoch. A fraction of the source's OWN pool (docstring point 1);
    None = the source's default. Outside (0, 1] is refused, never clamped."""
    f = default if frac is None else float(frac)
    if not np.isfinite(f) or not (0.0 < f <= 1.0):
        raise ValueError(f"{flag} {f} is outside (0, 1]: it is the fraction of the source's "
                         f"pool exposed per epoch; refused, not clamped.")
    n = int(round(f * n_pool))
    if n < 1:
        raise ValueError(f"{flag} {f} of {n_pool} rows admits ZERO rows per epoch.")
    return min(n, int(n_pool))


def attach_ext(train_ds, args, comp_labels, src="ext2", cache=None):
    """Return ConcatDataset([competition, (OAI,) (ext2,) src]) -- or train_ds unchanged if
    --{src}-weight is 0.

    Called from build_loaders on the TRAIN dataset only, AFTER attach_oai (and ext2 before
    ext3). `comp_labels` is the table the competition half is supervised from, so the reprice
    prices against exactly the base rate the rest of the batch carries.
    """
    weight = float(getattr(args, f"{src}_weight", 0.0) or 0.0)
    if weight <= 0:
        return train_ds
    cfg = SOURCES[src]
    cache = Path(getattr(args, f"{src}_cache", None) or cache or cfg["cache"])
    lab_path = Path(getattr(args, f"{src}_labels", None) or (cache / EXT2_LABELS))
    if not lab_path.exists():
        raise FileNotFoundError(f"{lab_path} missing -- build it ({cfg['builder']}) or pass "
                                f"--{src}-labels")
    tab = pd.read_csv(lab_path)
    validate_table(tab)
    names = build_ext_targets(getattr(args, f"{src}_targets", None) or cfg["targets"], tab)
    groups = [n for n in names if n in ANY_OF]
    plain = list(dict.fromkeys([n for n in names if n in TARGETS]
                               + [t for g in groups for t in ANY_OF[g]]))

    parts = list(train_ds.datasets) if isinstance(train_ds, ConcatDataset) else [train_ds]
    comp = parts[0]
    ids = sorted(tab.StudyInstanceUID.astype(str))           # deterministic composition

    # Tripwires (docstring point 3). The competition label table holds every study id the
    # fold table can hold, so an overlap there is a potential VALIDATION leak in some fold.
    taken = (set(map(str, comp_labels.StudyInstanceUID)) | set(map(str, getattr(comp, "ids", []))))
    for d in parts[1:]:
        taken |= set(map(str, getattr(d, "ids", [])))
    overlap = set(ids) & taken
    if overlap:
        raise ValueError(f"{len(overlap)} {src} ids (e.g. {sorted(overlap)[0]}) already belong "
                         f"to the competition or another external source; refusing.")

    sidx = ([int(x) for x in args.slots.split(",")] if getattr(args, "slots", None) else None)
    probe = cache / f"{ids[0]}.npy"
    if not probe.exists() or not (cache / f"{ids[0]}.mask.npy").exists():
        raise FileNotFoundError(f"{probe} (or its .mask.npy) missing: the label table "
                                f"{lab_path.name} does not describe the cache at {cache}")
    present = set(np.nonzero(np.load(cache / f"{ids[0]}.mask.npy")[0] >= 0.5)[0].tolist())
    if sidx is not None and not present & set(sidx):
        raise ValueError(f"--slots {args.slots} excludes every slot the {src} cache fills "
                         f"({sorted(present)}): each knee would be an empty study.")

    n_comp = len(comp)
    n_epoch = ext_epoch_rows(getattr(args, f"{src}_frac", None), len(ids), cfg["frac"],
                             f"--{src}-frac")
    n_ext = sum(len(d) for d in parts[1:]) + n_epoch
    if n_ext > n_comp:
        raise ValueError(f"{n_ext} external rows (OAI + every ext source) against {n_comp} "
                         f"competition rows would make external knees the MAJORITY of an "
                         f"epoch; refusing.")
    if n_epoch < len(ids) and len({_participant(u) for u in ids}) < 2:
        raise ValueError(f"{src} ids do not parse into participants (prefix_PARTICIPANT_...): "
                         f"the exposure rotation would hand one worker everything.")
    frac_seed = zlib.crc32(f"{int(getattr(args, 'seed', 0))}|{src}|".encode()
                           + ",".join(sorted(map(str, getattr(comp, "ids", [])))).encode())

    wp, wn = class_weights(comp_labels, tab, plain)
    any_kw, w_or = {}, None
    for g in groups:                                   # at most one (build_ext_targets)
        w_mem, w_or = any_of_reprice(comp_labels, tab, g)
        for t, x in w_mem.items():
            wn[TARGETS.index(t)] = x                   # member cells are negatives only
        lab = tab.set_index("StudyInstanceUID")
        on = (~lab[g + "__state"].isin(NOT_ASSERTED)).reindex(ids).values
        any_kw = dict(any_idx=[TARGETS.index(t) for t in ANY_OF[g]],
                      any_y=np.where(on, pd.to_numeric(lab[g]).reindex(ids).values, 0.0),
                      any_w=np.where(on, w_or * weight, 0.0))
    cell_w, keep = cell_weight_matrix(tab, ids, plain, wp, wn)
    ds = ExtStudies(
        ids, keep, cell_w, source=src, oai_weight=weight, epoch_rows=n_epoch,
        frac_seed=frac_seed, labels=tab, cache=cache, n_group=args.n_group, img=args.img,
        train=True, group=int(getattr(args, "group", None) or 3),
        gold_labels={}, gold_ids=set(), gold_weight=1.0, slot_idx=sidx, slot_drop=0.0,
        **any_kw)

    frac_shown = getattr(args, f"{src}_frac", None)
    print(f"  {src} aux: {len(ids)} knees from {cache.name}, weight {weight}, targets {names}")
    print(f"  {src} exposure: {len(ds)} of {len(ids)} rows per epoch (--{src}-frac "
          f"{cfg['frac'] if frac_shown is None else frac_shown}), "
          f"{len(ds) / (n_comp + n_ext):.1%} of samples"
          + ("" if len(ds) == len(ids) else
             f"; rotation covers the pool every ~{len(ids) / len(ds):.1f} epochs"))
    for t in plain:
        i = TARGETS.index(t)
        print(f"    {t:<18} w_pos {wp[i]:.3f}  w_neg {wn[i]:.3f}")
    for g in groups:
        print(f"    {g:<18} noisy-OR over {list(ANY_OF[g])}: {int((any_kw['any_w'] > 0).sum())} "
              f"rows, w {w_or:.3f} (members' negatives repriced jointly, above)")
    return ConcatDataset(parts + [ds])


def attach_ext2(train_ds, args, comp_labels, cache=EXT2_CACHE):
    """The ext2 (SKM-TEA) source: attach_ext(..., "ext2"), unchanged in behaviour."""
    return attach_ext(train_ds, args, comp_labels, "ext2", cache)
