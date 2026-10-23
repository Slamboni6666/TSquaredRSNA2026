"""
mil_perdiag.py -- L2 attention that is (a) alive and (b) per diagnosis.

THE WORDS, ONCE, BECAUSE THE REST OF THIS FILE LEANS ON THEM
  MIL, "multiple-instance learning"  The label belongs to the whole STUDY, but the evidence
      sits on two or three slices out of ~150 and nothing tells us which. So the model scores
      every piece and then has to decide how to boil a bag of piece-scores down to one number.
      Mean pooling divides a 2-slice finding by 150. MIL is any pooling that does better.
  ATTENTION-MIL  The pooling learns a weight per piece and takes a weighted mean, so it can
      concentrate on the pieces that look informative instead of averaging them away.
  GATED attention  How the weight is computed here: g = tanh(Vx) * sigmoid(Ux). tanh gives a
      signed description of the window, sigmoid gives a 0..1 relevance switch, and their
      product lets the module suppress a window outright rather than only down-weight it. The
      single score fed to the softmax is then w . g -- w is the "query" this file multiplies.
  SLOT  One canonical sequence type (sagittal fat-suppressed, coronal T1, ...). A study fills
      4.45 of six on average.
  GROUP  One of the ~10 windows of 3 adjacent slices a slot is read through. L2 pools over
      these. L1 pools tokens within a slice; L3 pools the six slots together.

WHY THIS FILE EXISTS

The brief was "give each of the 12 findings its own attention over the ~10 slice groups of a
slot, because an ACL tear and patellofemoral OA are on different slices". Before building it
I measured what the SHIPPED shared attention actually does, on the best checkpoint we have
(`checkpoints/fold0_aug.pt`, convnext_small + photometric aug, gold 0.8764), over 24 present
(study, slot) pairs of real fold-0 studies:

    normalised entropy of group_att   0.998      (1.000 = uniform)
    peak weight                       0.107      (uniform = 0.100)
    max/min weight ratio              1.2

**Our L2 attention is mean pooling.** It is not compromising between twelve diagnoses; it is
not choosing anything at all. Bolting twelve queries onto it as specified would have shipped
twelve uniform distributions -- trains fine, passes every shape test, buys nothing. That is
precisely the failure mode this was supposed to guard against, so the diagnosis had to come
first.

WHY IT IS UNIFORM -- and it is not the reason you would guess

Two candidates, and they demand opposite responses:

  (b) the ~10 group embeddings of a slot are near-identical, so no scoring vector could
      separate them. Then nothing at L2 helps and this idea is dead.
  (a) the pre-softmax logits are simply too small.

Measured on the same checkpoint, same studies:

    pairwise cosine between the 10 group embeddings of a slot   mean 0.637   min 0.236
    ||e_g - mean|| / ||mean||                                   mean 0.936
    mean |gate| = |tanh(Vx) * sigmoid(Ux)|                      0.586   (not saturated)
    std of the pre-softmax logits ACROSS the 10 groups          0.086
      -> softmax spread exp(2 * 0.086)                          1.19x

So (b) is refuted outright: the windows are strongly distinguishable, and the gate sits in a
healthy unsaturated range. The failure is pure SCALE. `mil.w.weight` has rms **0.0478** after
14 epochs against a PyTorch default init of 0.051 -- after a full training run the scoring
vector has not grown at all. ||w|| ~ 0.54 against a per-dimension gate variation of ~0.16
gives logits of O(0.1), and a softmax over logits of O(0.1) is uniform by arithmetic.
Nothing pushes ||w|| up: sharpening the attention is a second-order effect on a study-level
loss, while weight decay 0.01 pushes down every step.

This also re-reads D16 (2026-08-19). Raising n_group at inference improved TA monotonically
out to 16, long after volume coverage pegged at 100% from n_group=11, and the entry concluded
"more groups for the L2 attention to pool over". With the attention uniform, the honest
reading is that more groups is a better MEAN -- averaging more windows -- which is why the
curve keeps climbing smoothly instead of saturating when coverage does.

WHAT THIS MODULE DOES

  1. SHARPEN. Standardise the logits across the group axis and multiply by a learned gain:
     z <- (z - mean_g z) / std_g z, then z * gain. The spread of the softmax is now set by
     `gain`, a single learned number per query, and is completely independent of ||w||. w
     decides WHICH group wins; gain decides HOW MUCH it wins by. Weight decay can shrink w
     to nothing and the attention stays exactly as sharp. This is the part that makes any
     L2 attention -- shared or per-diagnosis -- a live mechanism rather than a decoration.

  2. PER-DIAGNOSIS QUERIES. V and U are SHARED; only the scoring vector w is per query.
     The gate g = tanh(Vx) * sigmoid(Ux) is a 128-dim description of a slice window; each
     diagnosis needs its own readout direction over that description, not its own encoder.
     This is exactly SlotHead's factorisation one level up -- one shared `proj`, twelve
     learned `query` vectors -- applied at L2. Cost: 11 x 128 = 1,408 extra weights.

Neither half is worth shipping alone. Per-diagnosis queries without (1) give twelve uniform
distributions. (1) without per-diagnosis queries gives one sharp distribution that all twelve
findings must share, which is the compromise the brief set out to remove. They are staged as
two flags so the ablation can still separate them -- see pending_patches/per-diagnosis-mil.md.

WHAT WAS ACTUALLY APPROVED TO RUN (TRACKER D22, 2026-08-19): `RUN, plan corrected`.
The mechanism above was verified independently and stands. The RUN ORDER was changed: the
first ~9 GPU-h were to go on `--mil-sharpen`, and sharpening is never shown to help. Check 7
of this file's own self-test has sharpen=False separating the twelve targets BETTER (mean
pairwise TV 0.515 against 0.359) at identical AUC, so the first run is plain per-diagnosis
queries with SHARPEN OFF. That is not a refutation of the paragraph above -- the toy has a
strong signal and no backbone, so its w is free to grow and sharpen itself, which is exactly
the freedom the shipped model's weak-signal-plus-weight-decay regime does not have (check 4
tests that directly). It does mean the sharpen claim is untested where it matters, and an
untested claim does not get to spend the first nine hours.

WHY 12 QUERIES AND NOT A HANDFUL OF ANATOMICAL GROUPS

The brief's argument for grouping is that 4,407 studies with noisy report labels cannot
support 12 independent attentions, and that the head is 1.19M by deliberate choice. That
argument is decisive against 12 independent GatedAttentionMIL modules (7.1M extra, below) and
it is what forced the shared-V/U factorisation. Once factorised, it stops biting: 12 queries
cost 1,548 weights against 516 for a 4-way grouping. A difference of 1,032 parameters out of
50.7M cannot be an overfitting argument, and pretending otherwise would be reasoning by
analogy instead of from our numbers.

What the label noise does argue about is gradient mass per query, and that is unequal.
Asserted (non-silent) cells x teacher weight, counted over all 4,407 rows of
report_labels_llm.csv:

    ACL 4,167   MCL 4,033   Medial Meniscus 3,681   Medial OA 3,541   PF OA 3,135
    Lateral Meniscus 3,096   Effusion 2,669   Lateral OA 2,459   Baker's 2,415
    Fracture 2,209   Contusion 1,837   Synovitis 787

Synovitis is 5.3x thinner than ACL. But even Synovitis is 787 studies x 14 epochs = ~11,000
gradient contributions onto a 128-dim vector, ~86 per parameter, with dropout and weight decay
on top. Thin, not hopeless -- and Synovitis is a target we already BEAT the teacher on, so its
attention going lazy costs little.

The decisive argument is anatomical, and it points the other way from the brief. The obvious
tissue grouping -- ligaments / menisci / compartmental OA / diffuse -- ties `Medial Meniscus`
to `Lateral Meniscus` and `Medial OA` to `Lateral OA`. Sagittally those pairs are separated by
SLICE INDEX; they are the pairs L2 most needs to separate. A tissue grouping would merge
exactly the distributions this change exists to split. If you group, group by slice band
(`SLICE_BAND` below) -- and that is a second hand-drawn anatomy prior, of the kind that has
already been caught wrong once here (PF OA's slot prior pointed at COR_T1, a coronal sequence,
for a patellofemoral finding). 12 queries cost 1,032 more parameters than 5 and require no
such guess. Default 12; `band` is kept as the ablation to run if 12 overfits.

WHAT IS DELIBERATELY ABSENT

  TWELVE INDEPENDENT GatedAttentionMIL MODULES. The obvious reading of "per-diagnosis
  attention", and it costs 12 x 0.59M = 7.1M parameters -- a 6x blow-up of a head that is
  1.19M by deliberate choice, on 4,407 studies with report-derived labels. Rejected on those
  grounds, and that rejection is what makes the 12-vs-5 question above cheap.

  A PER-STUDY TEMPERATURE. `gain` is per query, not per (study, slot, query). Standardising
  fixes the logit spread for every study, which does cost the model the ability to say "this
  particular slot has nothing distinctive, read it uniformly". A per-study gain would restore
  that, and would also restore the exact degree of freedom that collapsed to zero in the
  shipped model. Not worth re-opening the failure mode to buy back expressiveness we have no
  evidence was ever used.

  ANY DEFAULT-ON BEHAVIOUR. `n_query=1, sharpen=False` reproduces the shipped module exactly,
  parameter names and all, so every existing checkpoint still loads. Nothing here has been
  measured against the 58 gold; per house rule, nothing unmeasured is switched on by default.

CPU-testable: `python mil_perdiag.py --self-test` plants each target's evidence in a different
slice group of a synthetic bag, trains a few hundred CPU steps, and asserts the learned
attentions separate onto the right groups. `--self-test --neutralise` ties the 12 queries
into 1, changes nothing else, and must FAIL at check 5. Read `_ToyHead` before trusting any of
it: the first version of that control passed at AUC 1.000, and the reason is instructive.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
import torch.nn as nn

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# Fallback grouping if 12 independent queries turn out to overfit. NOT the anatomical tissue
# grouping that first suggests itself -- see the module docstring: that one merges the medial
# and lateral pairs, which are separated by slice index on the sagittal slots and are exactly
# what L2 is meant to split. If you group at all, group by where in the stack a finding sits.
SLICE_BAND = {
    "MCL": 0, "Medial Meniscus": 0, "Medial OA": 0,           # medial band
    "ACL": 1,                                                  # intercondylar notch
    "Lateral Meniscus": 2, "Lateral OA": 2,                    # lateral band
    "PF OA": 3,                                                # patellofemoral / anterior
    "Effusion": 4, "Synovitis": 4, "Baker's": 4,               # whole-volume fluid
    "Contusion": 4, "Fracture": 4,
}


def build_qmap(mode="per_target", targets=TARGETS):
    """target index -> query index. The one knob that decides how many attentions exist.

    "shared"     all 12 -> query 0.  Reproduces the shipped module exactly.
    "per_target" 12 queries.  The default when this is enabled at all.
    "band"       5 queries by slice band (SLICE_BAND above).
    """
    if mode == "shared":
        return torch.zeros(len(targets), dtype=torch.long)
    if mode == "per_target":
        return torch.arange(len(targets), dtype=torch.long)
    if mode == "band":
        return torch.tensor([SLICE_BAND[t] for t in targets], dtype=torch.long)
    raise ValueError(f"unknown qmap mode {mode!r}")


class PerDiagnosisMIL(nn.Module):
    """L2. Gated attention over the groups of one slot, one distribution per query.

    Drop-in for GatedAttentionMIL. At n_query=1 and sharpen=False the parameter names, the
    shapes and the arithmetic are identical to it, so the `mil.` keys in every existing
    checkpoint still load.

    forward(x): x (N, G, D) -> ctx (N, Q, D), att (N, G, Q)
    """

    def __init__(self, dim, n_query=1, hidden=128, p=0.1, sharpen=False, gain=0.5):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        # The bias is kept only so an existing state_dict still matches key-for-key. It is
        # DEAD weight: softmax over the group axis is invariant to a constant added to every
        # group, and this bias is constant over that axis by construction. Its trained value
        # in fold0_aug.pt is 0.0117 and it has never affected a single prediction we made.
        self.w = nn.Linear(hidden, n_query)
        self.drop = nn.Dropout(p)
        self.n_query = n_query
        self.sharpen = sharpen
        if sharpen:
            # One number per query. Positive = attend to high-scoring groups, and the model
            # is free to drive it to 0 for a genuinely diffuse finding (Effusion, Synovitis)
            # and recover mean pooling -- which is where the shipped model sits for ALL
            # twelve. Init 0.5: with standardised logits over G=10 the top group then holds
            # ~0.19 against a uniform 0.10 -- clearly alive, without committing hard to a
            # randomly chosen window while the backbone is still warming up.
            self.gain = nn.Parameter(torch.full((n_query,), float(gain)))
        # TIED INIT. Every query starts as an exact copy of query 0, so the module BEGINS as
        # a single shared attention and differentiates only as it learns. They do not stay
        # tied despite being numerically identical: the gradient reaching w[q] arrives
        # through target q's own L3 query, its own SLOT_PRIOR row and its own row of
        # out.weight, so L3 breaks the symmetry from the first step. The separation test in
        # self_test() starts from exactly this tied state and is the evidence for that claim.
        if n_query > 1:
            with torch.no_grad():
                self.w.weight.copy_(self.w.weight[:1].expand_as(self.w.weight))
                self.w.bias.copy_(self.w.bias[:1].expand_as(self.w.bias))

    def forward(self, x):                                     # (N, G, D)
        g = torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))  # (N, G, H)
        z = self.w(g)                                         # (N, G, Q)
        if self.sharpen and z.shape[1] > 1:
            # unbiased=False so a one-group slot degrades to 0/eps = 0 rather than nan.
            mu = z.mean(1, keepdim=True)
            sd = z.std(1, unbiased=False, keepdim=True)
            z = (z - mu) / (sd + 1e-5) * self.gain.view(1, 1, -1)
        a = z.softmax(dim=1)                                  # (N, G, Q)
        ctx = torch.einsum("ngq,ngd->nqd", a, x)              # (N, Q, D)
        return self.drop(ctx), a


# ============================================================================= self-test
#
# THE FIRST VERSION OF THIS TEST WAS BLIND, and the reason is worth more than the test.
#
# It gave each target its own readout direction, exactly as SlotHead does. The neutralised
# control -- ONE shared attention for both targets -- then solved the task at AUC 1.000 and
# the assertion had to be deleted rather than trusted. The leak, measured directly:
#
#     corr(shared att on group 0, y_A) = -0.896
#     corr(shared att on group 7, y_B) = -0.880
#
# A shared attention is still INPUT-DEPENDENT. It lowered its weight on group 0 when A's
# marker was there and its weight on group 7 when B's was, so the context carried y_A scaled
# onto the group-0 positional signature and y_B onto the group-7 one -- two separate channels
# -- and two different readout directions picked them apart. The shared attention smuggled
# both labels through its MAGNITUDE without ever needing to be two distributions.
#
# The fix is one shared readout direction for every target, with only a per-target bias. Then
# every target sees literally the same scalar unless the ATTENTION differs, and the ceiling
# for the neutralised control is arithmetic: the best a single score can do is rank on the
# sum of the markers it pooled, which for 4 independent targets is AUC 0.773.

class _ToyHead(nn.Module):
    """MIL + the smallest readout that makes the separation task well posed.

    Deliberately NOT SlotHead: SlotHead already has per-diagnosis queries at L3, and a test
    that let those absorb the signal would pass with L2 doing nothing at all.

    Deliberately ONE readout direction shared by every target, unlike SlotHead's per-target
    `out.weight` rows -- see the note above. Per-target rows make this control blind.
    """

    def __init__(self, dim, n_out, qmap, **kw):
        super().__init__()
        self.mil = PerDiagnosisMIL(dim, n_query=int(qmap.max()) + 1, **kw)
        self.register_buffer("qmap", qmap)
        self.r = nn.Parameter(torch.randn(dim) / dim ** 0.5)   # ONE direction, shared
        self.b = nn.Parameter(torch.zeros(n_out))              # per-target offset only

    def forward(self, x):
        ctx, a = self.mil(x)                                   # (N,Q,D), (N,G,Q)
        ctx = ctx.index_select(1, self.qmap)                   # (N,O,D)
        return ctx @ self.r + self.b, a


def _toy_bag(n=512, G=10, D=32, ev=(0, 3, 6, 9), seed=0):
    """Each target's evidence is the SAME marker vector, planted in a DIFFERENT group.

    Using one marker for all targets is the whole point: a pooling that does not choose a
    group sees only how many markers are in the bag, never which. With 4 independent
    Bernoulli(0.5) targets, ranking on that count gives AUC 0.773 per target -- the ceiling
    any single shared distribution is stuck at, and the number check 6 is measured against.

    A per-group positional signature p[g] is added so a query CAN recognise a group. In the
    real data that signature is free: anterior and posterior slices look different, which is
    why the measured pairwise cosine between a slot's group embeddings is only 0.637.
    """
    r = np.random.default_rng(seed)
    pos = torch.tensor(r.normal(0, 1, (G, D)), dtype=torch.float32)
    m = torch.tensor(r.normal(0, 1, (D,)), dtype=torch.float32)
    m = m / m.norm() * D ** 0.5
    x = torch.tensor(r.normal(0, 0.5, (n, G, D)), dtype=torch.float32) + pos
    y = torch.tensor(r.integers(0, 2, (n, len(ev))), dtype=torch.float32)
    for k, g in enumerate(ev):
        x[:, g] += y[:, k:k + 1] * m
    return x, y


def _auc(y, s):
    y = np.asarray(y)
    s = np.asarray(s)
    o = np.argsort(s)
    rank = np.empty(len(s), float)
    rank[o] = np.arange(1, len(s) + 1)
    npos, nneg = y.sum(), (1 - y).sum()
    return float((rank[y > 0].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def _train_toy(qmap, sharpen, steps=700, seed=0, wd=0.01, G=10, D=32, ev=(0, 3, 6, 9)):
    torch.manual_seed(seed)
    x, y = _toy_bag(G=G, D=D, ev=ev, seed=seed)
    net = _ToyHead(D, len(ev), qmap, hidden=64, p=0.0, sharpen=sharpen)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad()
        lo, _ = net(x)
        nn.functional.binary_cross_entropy_with_logits(lo, y).backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        lo, a = net(x)
    aucs = [_auc(y[:, k].numpy(), lo[:, k].numpy()) for k in range(y.shape[1])]
    att = a.mean(0).index_select(1, net.qmap).numpy()          # (G, O) per TARGET
    return aucs, att, net


def _mean_tv(att):
    """Mean pairwise total-variation distance between the per-target attentions. 0 = tied."""
    o = att.shape[1]
    d = [0.5 * np.abs(att[:, i] - att[:, j]).sum() for i in range(o) for j in range(i + 1, o)]
    return float(np.mean(d))


def self_test(neutralise=False):
    """neutralise=True ties the 12 queries into 1 and changes NOTHING else.

    That is the whole mechanism switched off, so check 5 -- the separation test -- must then
    fail. `--self-test --neutralise` exists so that failure is reproducible on demand rather
    than asserted in prose. If it ever passes, check 5 is measuring something other than the
    per-diagnosis attention and has to be rewritten.
    """
    tag = "   [NEUTRALISED: queries tied]" if neutralise else ""
    print(f"mil_perdiag self-test (CPU){tag}\n" + "=" * 74)
    torch.manual_seed(0)
    EV = (0, 3, 6, 9)
    qm5 = torch.tensor([0, 0, 0, 0] if neutralise else [0, 1, 2, 3])

    # ---------------------------------------------------------------- 1. drop-in
    x = torch.randn(7, 10, 64)
    base = PerDiagnosisMIL(64, n_query=1, p=0.0)
    ctx, a = base(x)
    assert ctx.shape == (7, 1, 64) and a.shape == (7, 10, 1), (ctx.shape, a.shape)
    assert torch.allclose(a.sum(1), torch.ones(7, 1), atol=1e-5)
    g = torch.tanh(base.V(x)) * torch.sigmoid(base.U(x))       # shipped GatedAttentionMIL
    ref = (base.w(g).softmax(dim=1) * x).sum(1)
    d = (ctx[:, 0] - ref).abs().max().item()
    assert d < 1e-6, d
    print(f"  1. drop-in for GatedAttentionMIL at n_query=1        ok  (max |delta| {d:.1e})")

    # the kept bias really is inert -- softmax over the group axis is shift invariant
    b2 = PerDiagnosisMIL(64, n_query=1, p=0.0)
    b2.load_state_dict(base.state_dict())
    with torch.no_grad():
        b2.w.bias += 3.7
    assert torch.allclose(b2(x)[1], a, atol=1e-6), "the w bias is NOT inert"
    print("  2. w bias is inert (kept only for checkpoint compat)  ok")

    # ---------------------------------------------------------------- 3. tied at init
    assert build_qmap("shared").tolist() == [0] * 12
    assert build_qmap("per_target").tolist() == list(range(12))
    assert build_qmap("band").tolist() == [1, 0, 0, 2, 0, 2, 3, 4, 4, 4, 4, 4]
    per = PerDiagnosisMIL(64, n_query=12, p=0.0, sharpen=True)
    aq = per(x)[1]
    spread = (aq - aq[:, :, :1]).abs().max().item()
    assert spread < 1e-6, f"queries are not tied at init: {spread}"
    print(f"  3. 12 queries start EXACTLY tied                     ok  (spread {spread:.1e})")

    # ---------------------------------------------------------------- 4. sharpen is alive
    # The guarantee: with standardised logits the spread is set by `gain`, so the attention
    # cannot go uniform however small ||w|| becomes. Test it the way the shipped model
    # failed -- shrink w to the measured trained scale (rms 0.048) and below.
    print("\n  4. does the attention survive a small ||w||?  "
          "(shipped fold0_aug.pt sits at rms 0.048)")
    print(f"     {'w rms':>10} {'sharpen=False':>16} {'sharpen=True':>16}    peak weight, "
          f"uniform = {1 / 10:.3f}")
    off_p, on_p = [], []
    for scale in (1.0, 0.1, 0.01):
        torch.manual_seed(1)
        off = PerDiagnosisMIL(64, n_query=1, p=0.0, sharpen=False)
        torch.manual_seed(1)
        on = PerDiagnosisMIL(64, n_query=1, p=0.0, sharpen=True, gain=1.0)
        with torch.no_grad():
            off.w.weight *= scale
            on.w.weight *= scale
        rms = off.w.weight.pow(2).mean().sqrt().item()
        po = off(x)[1].max(1).values.mean().item()
        pn = on(x)[1].max(1).values.mean().item()
        off_p.append(po)
        on_p.append(pn)
        print(f"     {rms:>10.4f} {po:>16.3f} {pn:>16.3f}")
    assert off_p[-1] < 0.11, "the unsharpened control did not collapse -- test is blind"
    assert min(on_p) > 0.25, f"sharpened attention collapsed too: {on_p}"
    assert max(on_p) - min(on_p) < 0.02, "sharpened spread still depends on ||w||"
    print("     -> unsharpened decays to uniform; sharpened is independent of ||w||    ok")

    # ---------------------------------------------------------------- 5. THE REAL TEST
    print(f"\n  5. separation task: 4 targets, evidence planted in groups {EV} of 10,")
    print("     same marker for all four, ONE shared readout direction. Any pooling that")
    print("     does not choose a group sees only the marker COUNT: ceiling AUC 0.773.")
    aucs, att, net = _train_toy(qm5, sharpen=True)
    tv = _mean_tv(att)
    print("     PER-DIAGNOSIS (4 queries, tied at init, sharpen=True)")
    print(f"       train AUC        {[round(v, 3) for v in aucs]}")
    print(f"       attention each TARGET (row) puts on each EVIDENCE group (column):")
    print(f"       {'':>10}" + "".join(f"{'g' + str(g):>8}" for g in EV) + "     elsewhere")
    for k, gk in enumerate(EV):
        rest = att[:, k].sum() - att[list(EV), k].sum()
        print(f"       target {k}  " + "".join(f"{att[g, k]:>8.3f}" for g in EV)
              + f"     {rest:>6.3f}")
    print(f"       mean pairwise TV distance between the 4 attentions   {tv:.3f}")
    print(f"       learned gain     {net.mil.gain.detach().numpy().round(2).tolist()} "
          f"(init 0.5)")
    # Assert on the evidence submatrix, not on a raw argmax over all 10 groups. The four
    # queries all park their leftover softmax mass on the SAME evidence-free group -- they
    # share V, U and the readout direction, so they agree on where the harmless dump is --
    # and that group can tie with the diagonal. It is not a failure: the claim being tested
    # is that each target reads ITS OWN evidence group and not another target's, which is
    # exactly diagonal dominance of the 4x4 below, both along rows and down columns.
    E = att[list(EV), :]                                       # E[j, k] = target k on g_j
    for k in range(len(EV)):
        assert E[:, k].argmax() == k, (f"target {k} prefers group {EV[E[:, k].argmax()]} "
                                       f"over its own {EV[k]}")
        row = [E[j, k] for j in range(len(EV)) if j != k]
        col = [E[k, j] for j in range(len(EV)) if j != k]
        assert E[k, k] > 3 * max(row), (f"target {k} puts {max(row):.3f} on another "
                                        f"target's group vs {E[k, k]:.3f} on its own")
        assert E[k, k] > 3 * max(col), (f"another target reads group {EV[k]} at "
                                        f"{max(col):.3f}, nearly as hard as its owner "
                                        f"{E[k, k]:.3f} -- these are not separate attentions")
    assert tv > 0.25, f"the attentions barely differ (mean TV {tv:.3f})"
    assert min(aucs) > 0.90, f"per-diagnosis MIL could not solve the task: {aucs}"
    print("     -> each diagnosis learned ITS OWN slice group                          ok")

    # ---------------------------------------------------------------- 6. NEUTRALISED
    # Tie the queries -- one shared attention, every other part of the model identical --
    # and the same test must FAIL. If it passes, the separation above was not coming from
    # the mechanism and this whole file proves nothing. (It did pass, once. See the note
    # above the _ToyHead class.)
    print("\n  6. NEUTRALISED CONTROL: qmap 'shared', one attention serving all four")
    print("     targets, everything else byte-identical.")
    n_aucs, n_att, _ = _train_toy(torch.tensor([0, 0, 0, 0]), sharpen=True)
    n_tv = _mean_tv(n_att)
    print(f"       train AUC        {[round(v, 3) for v in n_aucs]}")
    print(f"       attention        {np.round(n_att[:, 0], 2).tolist()}")
    print(f"       mean pairwise TV distance                            {n_tv:.3f}")
    assert n_tv < 1e-6, "the tied control produced different attentions -- qmap is not tying"
    assert min(n_aucs) < 0.85, (f"NEUTRALISED CONTROL STILL PASSED: one shared attention "
                                f"reached AUC {[round(v, 3) for v in n_aucs]} -- the "
                                f"separation test does not depend on the mechanism it "
                                f"claims to test")
    assert min(aucs) - min(n_aucs) > 0.10, (f"per-diagnosis {min(aucs):.3f} is not clearly "
                                            f"above shared {min(n_aucs):.3f}")
    print(f"     -> shared attention cannot separate them: worst AUC {min(n_aucs):.3f} "
          f"vs {min(aucs):.3f}    ok")

    # Reported, not asserted: in a toy with a strong signal and no backbone, w is free to
    # grow and can sharpen itself. The shipped model's failure is the weak-signal +
    # weight-decay regime, which check 4 tests directly.
    s_aucs, s_att, _ = _train_toy(torch.tensor([0, 1, 2, 3]), sharpen=False)
    print(f"\n  7. same task with sharpen=False (reported, not asserted): "
          f"AUC {[round(v, 3) for v in s_aucs]}, TV {_mean_tv(s_att):.3f}")

    # ---------------------------------------------------------------- 8. cost
    d2 = 2304
    p1 = sum(p.numel() for p in PerDiagnosisMIL(d2, 1).parameters())
    p5 = sum(p.numel() for p in PerDiagnosisMIL(d2, 5, sharpen=True).parameters())
    p12 = sum(p.numel() for p in PerDiagnosisMIL(d2, 12, sharpen=True).parameters())
    print(f"\n  8. parameters at dim {d2}: shared {p1:,}   band(5) {p5:,}   "
          f"per-target(12) {p12:,}")
    print(f"     12 queries cost {p12 - p1:,} extra weights = "
          f"{100 * (p12 - p1) / 50.7e6:.4f}% of the 50.7M model; 12 vs 5 is {p12 - p5:,}")
    assert p12 - p1 < 2000, "the per-diagnosis path is not supposed to cost real parameters"

    print("\n  ALL MIL-PERDIAG SELF-TESTS PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--neutralise", action="store_true",
                    help="tie the 12 queries into 1 and re-run: check 5 must FAIL")
    a = ap.parse_args()
    if a.self_test:
        self_test(neutralise=a.neutralise)


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()
