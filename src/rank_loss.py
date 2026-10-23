"""
rank_loss.py -- D28. Train on the metric instead of on a proxy for it.

WHY THIS FILE EXISTS
The competition scores macro ROC-AUC. AUC for one target is exactly

    P(score of a positive study > score of a negative study)

which depends ONLY on the ORDER of the scores. Add 0.3 to every prediction, or squash them all
toward 0.5, and the AUC does not move. `distill_loss` optimises
`binary_cross_entropy_with_logits` against soft targets, which measures how close each
prediction is to its target VALUE -- calibration. We are not scored on calibration.

That is not a theoretical objection. This project measured the gap twice:

  D15  The consensus teacher improved purely by BREAKING TIES -- reordering studies inside a
       severity rung. Its own gold AUC rose +0.0112. The student trained on those labels gained
       +0.0043, at 0.4 sigma. BCE could not read the improvement, because the labels moved by a
       mean of 0.0168 and never crossed 0.5.
  D24  The obvious fix -- scale the tie-break up until BCE notices -- is capped at 1.28x by the
       severity ladder's own geometry. There is no room.

So the ranking information in our labels is being discarded at the loss, and no amount of
tuning the labels recovers it. The fix belongs in the loss.

WHAT THIS COMPUTES
A margin-weighted pairwise logistic loss (RankNet), per target, over the studies in a batch:

    for each target t, for each pair (i, j) with y_i > y_j and both cells asserted:
        L += (y_i - y_j) * softplus(-(s_i - s_j))

`softplus(-d)` is the smooth surrogate for "is d > 0", i.e. for the indicator AUC actually
counts. Weighting each pair by `(y_i - y_j)` is what makes this fit OUR labels rather than a
generic ranking task: the teacher does not emit binary labels, it emits a 7-rung severity
ladder, and the size of the gap between two rungs is real information. A severe-vs-absent pair
should matter more than a mild-vs-uncertain pair, and the tie-break perturbations that D15
introduced -- tiny but consistently signed -- become usable gradient rather than rounding error.

WHAT IS DELIBERATELY ABSENT

  NO BINARISATION AT 0.5. Thresholding to make "positives" and "negatives" would throw away the
  ladder and reintroduce exactly the blindness this file exists to fix.

  NO REPLACEMENT OF distill_loss. This returns a term to be ADDED to BCE with a weight. Pure
  ranking losses leave the output scale unconstrained, and the submission writes probabilities;
  more importantly the pair count per batch is small (see below), so the ranking signal is
  noisier per step than BCE and is better used as a corrective than a substitute.

  NO CROSS-TARGET PAIRS. AUC is computed per target and then averaged. Comparing an ACL score
  against a Fracture score is meaningless, and pooling them would optimise a quantity nobody
  reports.

THE LIMITATION THAT MATTERS, STATED UP FRONT
Pairs come from within a batch. At bs=8 that is at most 8*7/2 = 28 pairs per target, and fewer
once masked cells are dropped -- so each step sees a small, noisy sample of the ranking. AUC is
a dataset-level statistic being estimated from 28 comparisons. Raising the batch size is the
direct fix and costs memory; the alternative is accumulating scores across steps, which drifts
because the model changes between them. This is why the weight starts small.

CPU-testable: `python rank_loss.py --self-test`.
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def pairwise_rank_loss(logits, y, label_mask, weight, min_gap=1e-4):
    """Margin-weighted pairwise ranking loss, per target, averaged over targets.

    logits      (B, T) raw scores -- NOT sigmoid'd. Only differences matter, and the sigmoid
                is monotone, so applying it first would only compress the useful range.
    y           (B, T) soft targets from the severity ladder.
    label_mask  (B, T) 1 where the report asserted something, 0 where it was silent. A pair is
                usable only if BOTH of its cells are asserted -- an unasserted cell carries no
                claim about order any more than it carries one about value.
    weight      (T,) or (B, T) per-target weights, matching distill_loss's convention.
    min_gap     pairs whose labels differ by less than this are skipped. Two studies on the same
                rung carry no ordering information, and including them would push their scores
                together for no reason.
    """
    B, T = logits.shape
    if B < 2:                                  # nothing to compare against
        return logits.sum() * 0.0

    # (B, B, T): d[i, j, t] = s_i - s_j and g[i, j, t] = y_i - y_j
    d = logits.unsqueeze(1) - logits.unsqueeze(0)
    g = y.unsqueeze(1) - y.unsqueeze(0)

    m = label_mask.unsqueeze(1) * label_mask.unsqueeze(0)      # both cells asserted
    valid = (g > min_gap).to(logits.dtype) * m                 # keep i>j once, not both ways

    # softplus(-d) is the smooth surrogate for the step function AUC counts. Weighted by the
    # label gap so a severe-vs-absent pair outweighs a mild-vs-uncertain one.
    per_pair = F.softplus(-d) * g.clamp_min(0.0) * valid

    w = weight if weight.dim() == 1 else weight.mean(0)        # (T,)
    num = (per_pair.sum(dim=(0, 1)) * w).sum()
    den = ((valid * g.clamp_min(0.0)).sum(dim=(0, 1)) * w).sum()
    return num / den.clamp_min(1e-6)


# ------------------------------------------------------------------ self-test
def self_test():
    torch.manual_seed(0)
    print("rank_loss self-test (CPU)\n")
    B, T = 16, 4

    # 1. direction. Scores that agree with the labels must cost less than scores that oppose.
    y = torch.rand(B, T)
    mask = torch.ones(B, T)
    w = torch.ones(T)
    good = (y - 0.5) * 8            # perfectly ordered
    bad = -(y - 0.5) * 8            # exactly reversed
    lg = pairwise_rank_loss(good, y, mask, w).item()
    lb = pairwise_rank_loss(bad, y, mask, w).item()
    assert lg < lb, f"ordered scores must cost less: {lg} vs {lb}"
    print(f"  correctly ordered costs less        ok   {lg:.4f} vs {lb:.4f} reversed")

    # 2. INVARIANT TO SHIFT AND SCALE -- the property that makes this a ranking loss at all.
    #    A loss that moved here would be measuring calibration again.
    l0 = pairwise_rank_loss(good, y, mask, w).item()
    l1 = pairwise_rank_loss(good + 3.0, y, mask, w).item()
    assert abs(l0 - l1) < 1e-5, f"shift changed the loss: {l0} vs {l1}"
    print(f"  invariant to a constant shift       ok   {l0:.6f} vs {l1:.6f}")

    # 3. THE MONEY TEST. This is the D15 scenario: a pure ranking improvement, too small in
    #    VALUE for BCE to notice. The whole file is pointless if this does not separate.
    torch.manual_seed(3)
    s = torch.randn(B, T)
    base = torch.full((B, T), 0.75)                     # everything on one rung -> all tied
    tie = base + 0.017 * torch.sign(s)                  # D15-sized nudge, correctly ordered
    anti = base - 0.017 * torch.sign(s)                 # same size, wrongly ordered
    r_t = pairwise_rank_loss(s, tie, mask, w).item()
    r_a = pairwise_rank_loss(s, anti, mask, w).item()
    b_t = F.binary_cross_entropy_with_logits(s, tie, reduction="mean").item()
    b_a = F.binary_cross_entropy_with_logits(s, anti, reduction="mean").item()
    print(f"\n  D15 scenario -- labels differ by 0.017, ordering flipped:")
    print(f"    ranking loss  correct {r_t:.4f}  vs wrong {r_a:.4f}   separation {abs(r_t-r_a):.4f}")
    print(f"    BCE           correct {b_t:.4f}  vs wrong {b_a:.4f}   separation {abs(b_t-b_a):.4f}")
    assert abs(r_t - r_a) > 10 * abs(b_t - b_a), "ranking loss must see what BCE cannot"
    print(f"    -> ranking loss is {abs(r_t-r_a)/max(abs(b_t-b_a),1e-9):.0f}x more sensitive   ok")

    # 4. masked cells contribute nothing
    m2 = mask.clone(); m2[:, 0] = 0
    full = pairwise_rank_loss(good, y, mask, w)
    part = pairwise_rank_loss(good[:, 1:], y[:, 1:], mask[:, 1:], w[1:])
    got = pairwise_rank_loss(good, y, m2, w)
    assert abs(got.item() - part.item()) < 1e-5, f"masking wrong: {got.item()} vs {part.item()}"
    print(f"\n  masked target excluded exactly      ok   {got.item():.6f} == {part.item():.6f}")

    # 5. it actually trains: descending on this loss must raise real AUC
    from sklearn.metrics import roc_auc_score
    torch.manual_seed(1)
    yb = (torch.rand(64, 1) > 0.5).float()
    z = torch.randn(64, 1, requires_grad=True)
    opt = torch.optim.SGD([z], lr=5.0)
    a0 = roc_auc_score(yb[:, 0].numpy(), z.detach()[:, 0].numpy())
    for _ in range(300):
        opt.zero_grad()
        pairwise_rank_loss(z, yb, torch.ones_like(yb), torch.ones(1)).backward()
        opt.step()
    a1 = roc_auc_score(yb[:, 0].numpy(), z.detach()[:, 0].numpy())
    assert a1 > a0 and a1 > 0.99, f"optimising it did not raise AUC: {a0:.3f} -> {a1:.3f}"
    print(f"  descending it raises real AUC       ok   {a0:.3f} -> {a1:.3f}")

    # 6. degenerate inputs must not produce nan
    for name, args in [("batch of 1", (torch.randn(1, T), torch.rand(1, T), torch.ones(1, T))),
                       ("all masked", (torch.randn(B, T), torch.rand(B, T), torch.zeros(B, T))),
                       ("all tied", (torch.randn(B, T), torch.full((B, T), 0.5), torch.ones(B, T)))]:
        v = pairwise_rank_loss(*args, torch.ones(T))
        assert torch.isfinite(v), f"{name} produced {v}"
    print("  degenerate inputs stay finite       ok   (batch of 1, all masked, all tied)")

    print("\n  ALL RANK-LOSS SELF-TESTS PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    if ap.parse_args().self_test:
        self_test()
    else:
        ap.print_help()
