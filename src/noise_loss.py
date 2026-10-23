"""
noise_loss.py -- a loss that does not let a wrong label dominate training.

WHY THIS FILE EXISTS
D30b settled what limits this model. Doubling the schedule bought +0.0072 gold; doubling the
backbone (convnext_small 50M -> convnext_base 88M) LOST 0.0334, while training loss fell in
both. The bigger model fits harder and generalises worse. With 4,407 studies whose labels were
inferred by an LLM from radiology reports, extra capacity buys memorisation of label noise.

**We have never used a loss designed for noisy labels.** That is the gap this fills.

WHY PLAIN CROSS-ENTROPY IS THE WORST CHOICE HERE
BCE's gradient with respect to the logit is (p - y). For a confidently wrong label -- the model
says 0.01, the teacher insists 1.0 -- the loss is -log(0.01) = 4.6 and climbing without bound
as the model gets *more* right. Cross-entropy is unboundedly confident that the label is
correct, so a minority of bad labels can dominate the total gradient. That is exactly the
regime we are in: the teacher scores 0.8718 on gold, so roughly one cell in eight is wrong.

WHAT THIS COMPUTES -- generalised cross-entropy (Zhang & Sabuncu, 2018)
Replace log(p) with the bounded surrogate (1 - p^q) / q:

    L_q(p) = (1 - p^q) / q

  q -> 0   recovers cross-entropy exactly (by L'Hopital), unbounded, noise-sensitive
  q  = 1   gives mean absolute error, fully bounded, noise-robust but slow to optimise
  q  = 0.7 the paper's default, and what we use: most of MAE's robustness, most of CE's
           gradient signal near the decision boundary

The bound is the whole point: at q = 0.7 the worst a single mislabelled cell can contribute is
1/q = 1.43, against cross-entropy's unbounded 4.6 and rising.

ADAPTED FOR SOFT TARGETS, WHICH IS NOT THE PAPER'S SETTING
GCE is published for hard one-hot labels. Our targets are GRADED -- the teacher emits a
severity ladder (severe 0.95 ... absent 0.03) and D15 showed that the graded value carries real
information. So we apply the bounded surrogate to both arms of the binary term, weighted by the
soft target exactly as BCE does:

    L = y * L_q(p) + (1 - y) * L_q(1 - p)

At q -> 0 this reduces to standard soft-target BCE, so `--noise-q 0` is a genuine no-op and the
A/B against D30a is honest. The self-test asserts that rather than assuming it.

WHAT IS DELIBERATELY UNCHANGED
The mask and the per-target weights. "Not mentioned" cells still contribute nothing, and the
teacher-reliability weighting still applies. This changes how hard a KEPT label can pull, not
which labels are kept.

CPU-testable: `python noise_loss.py --self-test`
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EPS = 1e-6


def gce_loss(logits, y, label_mask, weight, q=0.7):
    """Masked, weighted, soft-target generalised cross-entropy.

    Same signature and same reduction as model.distill_loss, so it is a drop-in replacement.
    q = 0 falls back to that function's exact behaviour.
    """
    if q <= 0:
        per = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    else:
        p = torch.sigmoid(logits).clamp(EPS, 1 - EPS)
        # (1 - p^q)/q, applied to each arm. Bounded above by 1/q, which is the robustness.
        pos = (1.0 - p.pow(q)) / q
        neg = (1.0 - (1.0 - p).pow(q)) / q
        per = y * pos + (1.0 - y) * neg
    m = label_mask * weight
    return (per * m).sum() / m.sum().clamp_min(1e-6)


def self_test():
    print("noise_loss self-test (CPU)\n")
    torch.manual_seed(0)
    B, T = 32, 12
    logits = torch.randn(B, T)
    y = torch.rand(B, T)
    mask = (torch.rand(B, T) > 0.2).float()
    w = torch.ones(T)

    # 1. q -> 0 must reproduce distill_loss EXACTLY, or the A/B against D30a is invalid
    import model as M
    a = M.distill_loss(logits, y, mask, w).item()
    b = gce_loss(logits, y, mask, w, q=0.0).item()
    assert abs(a - b) < 1e-6, f"q=0 is not a no-op: {a} vs {b}"
    print(f"  q=0 reproduces distill_loss          ok  {a:.9f} == {b:.9f}")

    # 2. THE POINT: the loss on a confidently-wrong label must be BOUNDED.
    wrong = torch.tensor([[-8.0]])            # model says ~0.0003
    insist = torch.tensor([[1.0]])            # label insists 1.0
    one = torch.ones(1, 1)
    ce = M.distill_loss(wrong, insist, one, torch.ones(1)).item()
    print(f"\n  a single confidently-WRONG label:")
    print(f"    cross-entropy          {ce:8.3f}   (unbounded — grows without limit)")
    for q in (0.3, 0.7, 1.0):
        g = gce_loss(wrong, insist, one, torch.ones(1), q=q).item()
        print(f"    GCE q={q:<4}             {g:8.3f}   (bounded by 1/q = {1/q:.2f})")
        assert g <= 1.0 / q + 1e-3, f"q={q} exceeded its own bound"
    assert gce_loss(wrong, insist, one, torch.ones(1), q=0.7).item() < ce / 2
    print("  bounded, and far below CE            ok")

    # 3. it must still LEARN — a bounded loss that cannot optimise is useless
    from sklearn.metrics import roc_auc_score
    torch.manual_seed(1)
    x = torch.randn(256, 4)
    true = (x[:, :1] @ torch.tensor([[2.0]]) > 0).float().repeat(1, 1)
    noisy = true.clone()
    flip = torch.rand(256, 1) < 0.20                      # 20% label noise
    noisy[flip] = 1 - noisy[flip]
    lin = torch.nn.Linear(4, 1)
    opt = torch.optim.Adam(lin.parameters(), lr=0.05)
    for _ in range(400):
        opt.zero_grad()
        gce_loss(lin(x), noisy, torch.ones_like(noisy), torch.ones(1), q=0.7).backward()
        opt.step()
    auc = roc_auc_score(true[:, 0].numpy(), lin(x).detach()[:, 0].numpy())
    assert auc > 0.9, f"failed to learn through noise: AUC {auc:.3f}"
    print(f"\n  learns through 20% label noise       ok  AUC {auc:.3f} against the TRUE labels")

    # 4. and it should beat CE at it, which is the entire hypothesis
    torch.manual_seed(1)
    lin2 = torch.nn.Linear(4, 1)
    opt2 = torch.optim.Adam(lin2.parameters(), lr=0.05)
    for _ in range(400):
        opt2.zero_grad()
        M.distill_loss(lin2(x), noisy, torch.ones_like(noisy), torch.ones(1)).backward()
        opt2.step()
    auc_ce = roc_auc_score(true[:, 0].numpy(), lin2(x).detach()[:, 0].numpy())
    print(f"  cross-entropy on the same data           AUC {auc_ce:.3f}")
    print(f"  -> GCE {'beats' if auc > auc_ce else 'does NOT beat'} CE by {auc - auc_ce:+.4f}"
          f"  (a linear probe is a weak test; the real one is the 5-fold run)")

    # 5. masking and weighting must be untouched
    m0 = mask.clone(); m0[:, 0] = 0
    full = gce_loss(logits, y, m0, w, q=0.7)
    part = gce_loss(logits[:, 1:], y[:, 1:], m0[:, 1:], w[1:], q=0.7)
    assert abs(full.item() - part.item()) < 1e-6, "masked target still contributes"
    print(f"\n  masked target excluded exactly       ok  {full.item():.6f} == {part.item():.6f}")
    assert torch.isfinite(gce_loss(torch.randn(4, T) * 50, y[:4], mask[:4], w, q=0.7))
    print("  extreme logits stay finite           ok")
    print("\n  ALL NOISE-LOSS SELF-TESTS PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    if ap.parse_args().self_test:
        self_test()
    else:
        ap.print_help()
