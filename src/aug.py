"""
aug.py -- photometric augmentation, applied on the GPU inside the forward pass.

WHY THIS FILE EXISTS
The pipeline currently has NO photometric augmentation. The only stochasticity anywhere is
the slice-window jitter in `KneeStudies._pick`; pixels go div_(255) -> interpolate ->
ImageNet-normalise untouched. With 4,407 studies, noisy report-derived labels, and 14 epochs,
that is a live overfitting risk and the cheapest unclaimed axis left.

WHY ON THE GPU, NOT IN THE DATASET
Measured elsewhere in this project: CPU-side augmentation costs ~1.3 s/study, which would
several-fold the wall clock of a run that is already ~0.1 s/study. These ops are pure
elementwise arithmetic on a tensor already resident on the device, so doing them after the
H2D copy is effectively free.

WHAT IS DELIBERATELY ABSENT

  HORIZONTAL FLIP. build_cache.py mirrors every left knee to a canonical side. Flipping
  would hand the model a handedness it has never seen, and five of the twelve targets are
  defined by side (both menisci, both tibiofemoral compartments, the MCL) -- a flip silently
  swaps their anatomy. Canonicalising is what buys the consistency; flipping spends it.

  BLUR / LOW-RESOLUTION SIMULATION. Our 130 mm crop at 336 px is 0.387 mm/px against a
  Nyquist bound of 0.5 mm/px for a 1 mm meniscal tear. We are already close to the limit;
  deliberately blurring would push focal findings past it.

  ROTATION / LARGE AFFINE. Slice ordering and the physical-mm crop are the two invariants the
  whole cache is built on. Small shifts are safe, but anything that changes apparent scale
  fights the mm-normalisation that preprocessing exists to guarantee.
  (GeoAug below, default OFF behind train.py --geo-aug, is the small version of this: rotation
  plus zoom-IN of at most 6%, per (study, slot), never a mirror and never a wrap.)

So: intensity only, plus a small translation. Every transform is per-STUDY, not per-slice --
a scanner's intensity profile is a property of the acquisition, and varying it within one
study would teach the model that slices of the same knee disagree.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class PhotometricAug:
    """Intensity jitter for a (B, S, G, C, H, W) uint8-derived float batch in [0, 1].

    Each parameter is drawn once per study and broadcast across that study's slots, groups
    and slices, so the augmentation mimics a different acquisition rather than noise.
    """

    def __init__(self, gamma=(0.7, 1.5), p_gamma=0.3,
                 brightness=0.15, contrast=0.15, p_bc=0.3,
                 noise_std=0.03, p_noise=0.2, shift_frac=0.03, p_shift=0.3):
        self.gamma, self.p_gamma = gamma, p_gamma
        self.brightness, self.contrast, self.p_bc = brightness, contrast, p_bc
        self.noise_std, self.p_noise = noise_std, p_noise
        self.shift_frac, self.p_shift = shift_frac, p_shift

    def __call__(self, x, study):
        """x: (N, C, H, W) in [0,1], flattened from (B,S,G,...).

        `study` is either an int B -- rows split into B equal contiguous blocks -- or, and
        this is what the model actually passes, a LongTensor of length N giving each row's
        study. The int form is only correct when every study contributes the same number of
        rows; ours do not (a study holds 4.45 of 6 slots on average, and absent slots are
        dropped before this point), so equal blocks would straddle study boundaries and hand
        one knee two different intensity profiles -- the exact thing this is built to avoid.
        """
        if not torch.is_grad_enabled():
            return x
        N = x.shape[0]
        dev, dt = x.device, x.dtype
        if torch.is_tensor(study):
            row = study.to(dev).long()
            n_study = int(row.max().item()) + 1 if N else 1
        else:
            n_study = max(int(study), 1)
            per = max(N // n_study, 1)
            row = (torch.arange(N, device=dev) // per).clamp_max(n_study - 1)

        def study_param(lo, hi):
            v = torch.empty(n_study, device=dev, dtype=dt).uniform_(lo, hi)
            return v[row].view(N, 1, 1, 1)

        def study_mask(p):
            m = (torch.rand(n_study, device=dev) < p).to(dt)
            return m[row].view(N, 1, 1, 1)

        # gamma: mimics a different contrast curve. clamp first -- pow of a negative is nan
        m = study_mask(self.p_gamma)
        g = study_param(*self.gamma)
        x = torch.where(m > 0, x.clamp_min(1e-6).pow(g), x)

        # brightness / contrast about the batch mean
        m = study_mask(self.p_bc)
        b = study_param(-self.brightness, self.brightness)
        c = study_param(1 - self.contrast, 1 + self.contrast)
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = torch.where(m > 0, (x - mean) * c + mean + b, x)

        # acquisition noise
        m = study_mask(self.p_noise)
        if float(m.sum()) > 0:
            x = x + m * torch.randn_like(x) * self.noise_std

        # small translation: the joint is centred by the mm-crop, but not perfectly
        m = study_mask(self.p_shift)
        if float(m.sum()) > 0:
            H, W = x.shape[-2:]
            dx = int(self.shift_frac * W)
            if dx > 0:
                sx = int(torch.randint(-dx, dx + 1, (1,)).item())
                sy = int(torch.randint(-dx, dx + 1, (1,)).item())
                rolled = torch.roll(x, shifts=(sy, sx), dims=(-2, -1))
                x = torch.where(m > 0, rolled, x)

        return x.clamp(0, 1)


class GeoAug:
    """Small affine jitter -- rotation, zoom-in, translation -- ONE draw per (study, slot).

    Default OFF (train.py --geo-aug). What it simulates is positioning: the knee sits a few
    degrees off the scanner axes (in-plane obliquity SD 2.7 / 3.0 / 7.4 deg sagittal / coronal /
    axial over the chosen series, axial p99 21.9 deg) and a little off the crop centre, and
    build_cache.py does not de-rotate. That belongs to the ACQUISITION, so every window and every
    slice channel of one (study, slot) shares one transform, and slots draw independently --
    each series is planned on its own.

    Fixed config, no sweep. Each (study, slot) is transformed with probability p = 0.5:
      rotation     uniform +-10 deg sagittal / coronal, +-15 deg axial. The plane is the slot
                   name's prefix (SAG_ / COR_ / AX_), the same in the v1 and the f2 rack.
      zoom         IN only, uniform 1.00-1.06: the 130 mm crop is never asked for anatomy it
                   does not hold, and the mm scale moves by at most 6%.
      translation  uniform +-5% of the side on each axis, of the content in the output frame.
      padding      zeros, bilinear. It never wraps: what leaves the frame is gone.
    det > 0 always (a rotation times a positive zoom), so no draw can mirror the knee and swap
    medial for lateral -- the canonical side build_cache.py set up is kept.

    RNG: everything is drawn from torch's default generator on x's device -- the stream
    PhotometricAug draws from -- with a fixed count per call (5 values per instance). The
    train.py sidecar saves the CPU and every CUDA generator state, so --resume stays exact.
    """

    def __init__(self, rack, p=0.5, rot=10.0, rot_ax=15.0, zoom=(1.0, 1.06), shift=0.05):
        # rack: the cache's slot names by CACHE slot index -- model.SLOTS (v1) or
        # model_f2.SLOTS_F2 (f2). KneeNet.forward maps each instance through self.slot_idx.
        self.axial = [str(n).startswith("AX") for n in rack]
        self.p, self.rot, self.rot_ax, self.zoom, self.shift = p, rot, rot_ax, zoom, shift

    def params(self, slot):
        """slot: LongTensor(n), the cache slot index of each (study, slot) instance.
        -> on (n,) bool, theta (n, 2, 3) for F.affine_grid (output coords -> input coords)."""
        dev, n = slot.device, slot.numel()
        on = torch.rand(n, device=dev) < self.p
        u = torch.rand(n, 4, device=dev)
        lim = torch.where(torch.tensor(self.axial, device=dev)[slot], self.rot_ax, self.rot)
        th = (u[:, 0] * 2 - 1) * lim * (math.pi / 180)
        s = self.zoom[0] + u[:, 1] * (self.zoom[1] - self.zoom[0])
        d = (u[:, 2:] * 2 - 1) * (2 * self.shift)           # a side spans 2 normalised units
        a = torch.stack([torch.stack([th.cos(), -th.sin()], -1),
                         torch.stack([th.sin(), th.cos()], -1)], 1) / s.view(n, 1, 1)
        return on, torch.cat([a, -(a @ d.unsqueeze(-1))], -1)   # input = a @ (output - d)

    def __call__(self, x, slot):
        """x: (n, K, H, W) in [0,1], square -- ONE row per (study, slot) instance, with all of
        its windows and slices stacked on K, so a single sampling grid moves every one of them.
        slot: LongTensor(n). A no-op whenever grad is disabled (eval), as in PhotometricAug.
        Rows not drawn are returned untouched, bit for bit."""
        if not torch.is_grad_enabled() or self.p <= 0:
            return x
        on, theta = self.params(slot.to(x.device))
        if not bool(on.any()):
            return x
        sel = on.nonzero(as_tuple=True)[0]
        # fp32 whatever autocast the caller runs under: affine_grid is a matmul, and under bf16
        # autocast it comes back bf16, ~1 px off at 336 px (measured on CPU autocast).
        with torch.autocast(x.device.type, enabled=False):
            xi = x.index_select(0, sel).float()
            grid = F.affine_grid(theta.index_select(0, sel), list(xi.shape), align_corners=False)
            y = F.grid_sample(xi, grid, mode="bilinear", padding_mode="zeros",
                              align_corners=False)
        return x.index_copy(0, sel, y.to(x.dtype))


def self_test():
    """Cheap invariants. An augmentation that silently does nothing is the usual failure."""
    torch.manual_seed(0)
    aug = PhotometricAug()
    x = torch.rand(12, 3, 64, 64)          # 2 studies x 6 rows

    with torch.no_grad():
        same = aug(x, 2)
    assert torch.equal(same, x), "must be a no-op when grad is disabled (i.e. at eval)"
    print("  eval-time no-op                      ok")

    x.requires_grad_(False)
    with torch.enable_grad():
        y = aug(x.clone(), 2)
    assert not torch.equal(y, x), "augmentation did nothing while training"
    print(f"  changes the input when training      ok  (mean |delta| {(y - x).abs().mean():.4f})")
    assert y.min() >= 0 and y.max() <= 1, f"escaped [0,1]: [{y.min()}, {y.max()}]"
    print(f"  stays in [0,1]                       ok  [{y.min():.3f}, {y.max():.3f}]")
    assert torch.isfinite(y).all(), "produced nan/inf"
    print("  finite                               ok")

    # per-STUDY, not per-row: rows of one study must share their transform. Compare the
    # row-to-row spread within a study against the spread between studies.
    torch.manual_seed(3)
    xs = torch.rand(1, 3, 32, 32).repeat(12, 1, 1, 1)
    with torch.enable_grad():
        ys = aug(xs.clone(), 2)
    a, b = ys[:6], ys[6:]
    within = (a - a.mean(0, keepdim=True)).abs().mean()
    between = (a.mean(0) - b.mean(0)).abs().mean()
    print(f"  per-study not per-slice              within {within:.4f} vs between {between:.4f}")
    assert within <= between + 1e-6, "transform varies within a study — should be per-study"
    print("\n  ALL AUGMENTATION SELF-TESTS PASSED")


if __name__ == "__main__":
    self_test()
