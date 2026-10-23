"""
kaggle_notebook.py -- the body of the submitted Kaggle notebook. Paste into a code cell.

DESIGN: this file contains no preprocessing logic of its own. It imports the same modules
the cache was built with, uploaded as a Kaggle dataset. Restating the preprocessing here --
even carefully -- would break the one property `inference.py --verify` establishes: that
scoring-time images are bit-identical to training images. A copy diverges the first time
either side is edited, and the divergence is silent.

WHAT TO UPLOAD, as two Kaggle datasets:
  rsna-knee-code     build_cache.py  build_slice_order.py  build_series_index.py
                     model.py  mil_perdiag.py  inference.py  validation.py
                     build_cache_f2.py  build_series_index_v2.py  inference_f2.py  model_f2.py
                     (package_for_kaggle.py CODE_FILES is the authoritative list)
  rsna-knee-weights  checkpoints/*.pt

Then attach both to the notebook, plus the competition data, and disable internet.

TWO FAMILIES. A checkpoint's saved args name its cache_format. "v1" members read the
(6, 32, 336, 336) volume inference.preprocess_study builds; "f2" members read the flat
(sum(f2_budget), 384, 384) volume inference_f2.preprocess_study_f2 builds, which
`python inference_f2.py --verify` proves byte-identical to the f2 training cache. Each study
is preprocessed once per format. Members are averaged within their family exactly as the
whole ensemble used to be (probabilities, then a per-column rank when the family has more
than one member); families are then rank-averaged with FAMILY_WEIGHTS. With only v1
checkpoints attached, every number written is what it was before families existed.

ORDER OF OPERATIONS. A valid submission is written BEFORE anything that can fail, and
overwritten once real predictions exist. A notebook that dies after two hours of decoding
otherwise scores nothing at all; this way it scores the benchmark, which is strictly better
than an error.

VOCABULARY, because this file is read by people who did not build the model
  slot     one of six canonical MRI sequence types a study may hold: sagittal, coronal and
           axial fluid-sensitive fat-suppressed, sagittal fluid-sensitive without fat
           suppression, coronal T1, sagittal T1. Most studies have only some of them, so a
           presence mask records which are real and the model pools over those. The plane
           matters diagnostically: cruciate ligaments and meniscal horns are read on
           sagittals, the collaterals and the tibiofemoral compartments on coronals.
  n_group  how many windows of 3 consecutive slices are cut out of each slot's 32-slice
           stack and pushed through the backbone. Those 3 slices become the backbone's
           three input channels -- adjacent anatomy, not colour.
  gold     the 58 studies that carry real radiologist labels. Every other label came from
           an LLM reading the study's report, so gold is the only clean read we have, and
           58 studies put its noise floor at roughly +/-0.03 AUC.
  teacher-agreement
           AUC against those LLM labels over all 4,407 studies. About 7x less noisy than
           gold, but it scores IMITATION of the LLM rather than correctness, so it is used
           to screen a change for signal, never to decide which change is better.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

T0 = time.time()
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
OUT = Path("/kaggle/working/submission.csv")


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- mounts
def find_competition():
    """Locate the competition mount. Kaggle nests it inconsistently, so search rather
    than hardcode -- a wrong hardcoded path fails at the worst possible moment."""
    for c in [Path("/kaggle/input/rsna-knee-abnormality-detection"),
              Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")]:
        if (c / "test.csv").is_file():
            return c
    base = Path("/kaggle/input")
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        if (d / "test.csv").is_file():
            return d
        for sub in sorted(p for p in d.iterdir() if p.is_dir()):
            if (sub / "test.csv").is_file():
                return sub
    raise FileNotFoundError("competition mount not found")


def find_dir(marker, skip_big=True):
    """First mounted dataset containing `marker`. The competition mount holds hundreds of
    thousands of files, so it is skipped by inspection rather than by name -- at the top
    level and one level down, because Kaggle mounts it at /kaggle/input/competitions/<slug>.
    Inspecting only the top level walked the whole mount on every call: the 14 logged runs
    spent 307-816 s finding the code and weights, then more again for the wheels."""
    def big(p):
        return skip_big and any((p / s).is_dir() for s in ("train_series", "test_series"))

    base = Path("/kaggle/input")
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        if big(d):
            continue
        kids = [k for k in d.iterdir() if k.is_dir() and not k.is_symlink()]
        if any(big(k) for k in kids):  # search around the competition mount, never into it
            hit = next(d.glob(marker), None) or next(
                (h for k in kids if not big(k) for h in k.rglob(marker)), None)
        else:
            hit = next(d.rglob(marker), None)
        if hit is not None:
            return hit.parent
    return None


COMP = find_competition()
log(f"competition mount: {COMP}")

test_ids = pd.read_csv(COMP / "test.csv").StudyInstanceUID.tolist()
log(f"{len(test_ids)} test studies")

# The benchmark submission, written first. Everything after this point may fail.
pd.DataFrame({"StudyInstanceUID": test_ids,
              **{t: 0.5 for t in TARGETS}}).to_csv(OUT, index=False)
log(f"wrote benchmark {OUT} — the run now always leaves a scoreable file")

CODE = find_dir("inference.py")
WEIGHTS = find_dir("*.pt") or find_dir("fold0*.pt")
log(f"code: {CODE}   weights: {WEIGHTS}")
if CODE is None:
    raise FileNotFoundError("code dataset not attached")
sys.path.insert(0, str(CODE))

# Decoders for compressed pixel data. Training is 100% uncompressed Explicit VR LE, but the
# data description says the test set may hold JPEG Lossless / JPEG 2000 / Implicit VR LE.
#
# These are absent from the Kaggle image and the notebook runs with internet OFF, so they
# cannot be pip-installed at runtime -- they have to ride in as a dataset of wheels. Both
# cp310 and cp311 manylinux builds are shipped; pip picks the tag matching the interpreter,
# so this survives Kaggle changing its Python version under us. Failure here is deliberately
# NOT fatal: training data is entirely uncompressed, so if the test set is too, the decoders
# are never needed and a dead install must not take the submission down with it.
WHEELS = find_dir("pylibjpeg*.whl")
if WHEELS is not None:
    import subprocess
    rc = subprocess.call([sys.executable, "-m", "pip", "install", "--no-index",
                          f"--find-links={WHEELS}", "--quiet",
                          "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg",
                          "python-gdcm"])
    log(f"decoder wheels: install from {WHEELS} exited {rc}")
else:
    log("decoder wheels: dataset not attached")

for _m in ("pylibjpeg", "gdcm"):
    try:
        __import__(_m)
        log(f"{_m} available")
    except ImportError:
        log(f"{_m} MISSING — compressed series will fail to decode. Training data is "
            f"100% uncompressed, so this only bites if the TEST set is not.")

import torch  # noqa: E402
from inference import build_index, preprocess_study  # noqa: E402
from build_cache import N_SLICE  # the cache contract lives here, not in model  # noqa: E402
from model import knee_net_from_args  # noqa: E402

def pick_device():
    """Probe the GPU with a real convolution before trusting it.

    `torch.cuda.is_available()` is NOT sufficient. Kaggle assigns a Tesla P100 (sm_60) to
    some sessions, and the preinstalled PyTorch ships kernels only for sm_70 and newer. On
    such a session `is_available()` returns True, the model moves to the device without
    complaint, and then EVERY forward pass raises
    `CUDA error: no kernel image is available for execution on the device`.

    That is not hypothetical: it is exactly what happened on the 2026-08-17 run, where all
    studies fell back to 0.5 and the submission had to be discarded. So run an actual conv
    and a matmul, and believe the result rather than the capability flag.
    """
    if not torch.cuda.is_available():
        log("no CUDA device -> cpu")
        return "cpu"
    try:
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        supported = torch.cuda.get_arch_list()
        log(f"gpu {name} sm_{cap[0]}{cap[1]}; torch built for {supported}")
        c = torch.nn.Conv2d(3, 8, 3).cuda()
        y = c(torch.randn(1, 3, 32, 32, device="cuda"))
        _ = (y.float().sum() + (torch.randn(64, 64, device="cuda") @
                                torch.randn(64, 64, device="cuda")).sum()).item()
        torch.cuda.synchronize()
        log("gpu conv+matmul probe passed -> cuda")
        return "cuda"
    except Exception as e:
        log(f"GPU PRESENT BUT UNUSABLE ({type(e).__name__}: {str(e)[:90]})")
        log("falling back to CPU. Slower, but a real submission beats a benchmark one.")
        return "cpu"


DEV = pick_device()

def pick_dtype(dev):
    """Choose an autocast dtype by probing a CONVOLUTION, not a matmul.

    The first version of this probe tested a matmul and got it wrong. A T4 (sm_75) happily
    runs bfloat16 *matmuls*, so the probe reported "bfloat16 OK" -- and then every real
    forward pass died with `GET was unable to find an engine to execute this computation`,
    because cuDNN has no bfloat16 *convolution* engine below sm_80. The probe has to
    exercise the same operation the model does, or it certifies the wrong thing.

    Two guards now: a capability gate (bf16 convs need sm_80+) and an actual conv under
    autocast, run through .item() so any async CUDA error surfaces here rather than later.
    """
    if dev != "cuda":
        return None
    major = torch.cuda.get_device_capability(0)[0]
    order = [(torch.bfloat16, "bfloat16"), (torch.float16, "float16")]
    if major < 8:
        order = [(torch.float16, "float16")]     # bf16 conv unsupported below sm_80
        log(f"sm_{major}x: skipping bfloat16 (needs sm_80+ for conv)")
    for dt, name in order:
        try:
            c = torch.nn.Conv2d(3, 8, 3).cuda()
            with torch.autocast("cuda", dtype=dt):
                y = c(torch.randn(2, 3, 64, 64, device="cuda"))
            _ = y.float().sum().item()
            torch.cuda.synchronize()
            log(f"autocast dtype: {name} (conv probe passed)")
            return dt
        except Exception as e:
            log(f"autocast {name} unusable: {type(e).__name__}")
    log("no working autocast dtype -> float32")
    return None


AMP = pick_dtype(DEV)
log(f"device {DEV}")

# --------------------------------------------------------------------------- models
# Each checkpoint carries the preprocessing it was fitted under, and those differ: the
# 4-slot member has a 4-slot head and must be fed 4 slots, the 384 member wants a different
# resample. Loading a 4-slot state_dict into a 6-slot KneeNet does not merely score badly --
# it raises on shape mismatch. So a member is (network, slot_idx, img_size), and the loop
# below feeds each one exactly what it was trained on.
# INFERENCE n_group IS DELIBERATELY HIGHER THAN TRAINING n_group.
# Measured 2026-08-19 by re-predicting the five convnext_small checkpoints at several
# settings -- no retraining, the same weights throughout:
#
#     n_group   gold     vs 10      teacher-agreement vs 10
#        10    0.8612      --                --
#        11    0.8556    -0.0062          +0.0021  (2.4 sigma)
#        12    0.8690    +0.0077          +0.0035  (4.2 sigma)
#        14    0.8651    +0.0036          +0.0038  (4.4 sigma)
#        16    0.8710    +0.0096          +0.0046  (5.7 sigma)
#
# Note what this is NOT. At n_group=10 the deterministic inference path never reads cache
# positions 9 and 22, and the obvious story is that closing that hole is the gain. It is not:
# n_group=11 closes it completely and is WORSE on gold. The benefit instead rises
# monotonically well past the point where coverage is already 100%, so the mechanism is more
# groups for the L2 attention to pool over -- more chances for a focal finding to sit
# well-centred in some window rather than at a window edge.
#
# Cost is linear: 1.6x the backbone passes. The efficiency prize prices runtime at 72 s per
# 0.001 AUC, and this buys ~0.005-0.010 for ~1.6x of a ~200 s forward, which clears that bar
# comfortably. Set to None to fall back to whatever each checkpoint trained at.
# v1 members only: that measurement was made on v1 convnext_small checkpoints.
INFER_N_GROUP = 16
# f2 members: None = the n_group each checkpoint trained at. Nothing has been measured for
# f2 yet, and on the dense layout COR_ALT's 16 frames hold only 14 distinct windows.
INFER_N_GROUP_F2 = None
# Family blend: each family's (study x target) scores are ranked per column, then averaged
# with these weights over the families that predicted the study. Only consulted when more
# than one family is loaded. The f2 weight is to be set from its OOF/LB once it exists.
FAMILY_WEIGHTS = {"v1": 1.0, "f2": 1.0}


ckpts = sorted(Path(WEIGHTS).rglob("*.pt")) if WEIGHTS else []
members = []
for p in ckpts:
    ck = torch.load(p, map_location="cpu", weights_only=False)
    a = ck["args"]
    fam = a.get("cache_format") or "v1"
    if fam == "f2" and not a.get("f2_budget"):
        # train.py records the cache's frame layout as f2_budget; without it the volume
        # this member was trained on cannot be rebuilt, so it does not get a guessed one.
        log(f"SKIPPING {p.stem}: f2 checkpoint without f2_budget in its args")
        continue
    net = knee_net_from_args(a)  # pretrained=False: no internet, weights come from ck
    net.load_state_dict(ck["model"])
    members.append(dict(net=net.to(DEV).eval(), family=fam,
                        layout=tuple(a["f2_budget"]) if fam == "f2" else None,
                        slots=list(net.slot_idx), img=a["img"],
                        n_group=((INFER_N_GROUP_F2 if fam == "f2" else INFER_N_GROUP)
                                 or a["n_group"]), tag=p.stem))
if DEV == "cpu" and len(members) > 1:
    # Budget: ~1,300 studies x members x n_group backbone passes. On CPU the full set cannot
    # finish inside 9 h, and a timed-out notebook scores nothing at all -- strictly worse
    # than one member's honest predictions.
    members = members[:1]
    members[0]["n_group"] = max(4, members[0]["n_group"] // 2)
    log(f"CPU fallback: 1 member, n_group={members[0]['n_group']}")
log(f"loaded {len(members)} member(s):")
for m in members:
    log(f"    {m['tag']:<28} {m['family']} slots={len(m['slots'])} img={m['img']} "
        f"n_group={m['n_group']}" + (f" layout={m['layout']}" if m["layout"] else ""))
if not members:
    raise RuntimeError("no checkpoints found — benchmark submission stands")

# --------------------------------------------------------------------------- run
idx, side_map = build_index(COMP / "test_series", COMP / "test_series.csv", threads=8)
chosen = idx[idx.chosen]
log(f"indexed {len(idx)} series, {chosen.StudyInstanceUID.nunique()} studies with slots")

# One preprocessing per FORMAT, keyed by layout: None is the v1 volume, a budget tuple an
# f2 volume. Every member reads whichever key it was trained on.
need_v1 = any(m["family"] == "v1" for m in members)
f2_layouts = sorted({m["layout"] for m in members if m["family"] == "f2"})
if f2_layouts:
    from inference_f2 import build_index_f2, preprocess_study_f2
    from model_f2 import eval_groups_f2
    chosen_f2 = build_index_f2(idx)
    log(f"f2 slots: {len(chosen_f2)} series over {chosen_f2.StudyInstanceUID.nunique()} "
        f"studies, layouts {f2_layouts}")

families = sorted({m["family"] for m in members})
fam_preds = {f: np.full((len(test_ids), len(TARGETS)), 0.5, np.float64) for f in families}
fam_ok = {f: np.zeros(len(test_ids), bool) for f in families}
n_fail = 0
n_prep_fail = {}

for i, sid in enumerate(test_ids):
    try:
        inputs = {}
        for lay in ([None] if need_v1 else []) + f2_layouts:
            # A format that cannot be built for this study costs only its own members.
            try:
                if lay is None:
                    rows = chosen[chosen.StudyInstanceUID == sid]
                    vol, mask = preprocess_study(sid, rows, COMP / "test_series",
                                                 side_map.get(sid))
                    present = mask
                else:
                    rows = chosen_f2[chosen_f2.StudyInstanceUID == sid]
                    vol, mask = preprocess_study_f2(sid, rows, COMP / "test_series",
                                                    side_map.get(sid), lay)
                    present = mask[0]
                if present.sum() > 0:
                    inputs[lay] = (vol, mask)
            except Exception as e:
                n_prep_fail[lay] = n_prep_fail.get(lay, 0) + 1
                if n_prep_fail[lay] <= 5:
                    log(f"study {sid[-10:]} {lay or 'v1'} preprocessing failed: "
                        f"{type(e).__name__}: {e}")
        if not inputs:
            n_fail += 1
            continue
        ps = {f: [] for f in families}
        for m in members:
            if m["layout"] not in inputs:
                continue
            sl = m["slots"]
            vol, mask = inputs[m["layout"]]
            if m["family"] == "f2":
                # KneeStudiesF2's eval windows, from model_f2 rather than restated here:
                # n_group starts spread over each slot's OWN rows of the flat volume.
                g, mk = eval_groups_f2(vol, mask, sl, m["n_group"])
                if mk.sum() == 0:
                    continue
            else:
                # Cut the windows exactly as KneeStudies._pick does with train=False:
                # n_group start positions spread evenly over the 32-slice stack, each window
                # taking 3 consecutive slices. The result is (slot, group, 3, H, W) and
                # unsqueeze(0) below adds the batch axis. This is the ONLY thing the notebook
                # re-implements from the training path, so it has to track _pick's eval
                # branch exactly -- a drift here is a train/test mismatch that raises
                # nothing and just scores worse.
                starts = np.linspace(0, N_SLICE - 3, m["n_group"]).round().astype(int)
                g = np.stack([[vol[s, k:k + 3] for k in starts] for s in sl])
                # Presence flags for this member's slots only. All zero means the study
                # holds none of the sequences this member was trained on, so it cannot vote.
                mk = mask[sl].copy()
                if mk.sum() == 0:
                    continue
            gt = torch.from_numpy(g).unsqueeze(0).to(DEV)
            mt = torch.from_numpy(mk).unsqueeze(0).to(DEV)
            with torch.no_grad(), torch.autocast(DEV, dtype=AMP or torch.float32,
                                                 enabled=AMP is not None):
                # forward returns (logits, attention maps): [0] takes the logits, and
                # the [0] below takes the single study in this batch of one.
                out = m["net"](gt, mt, img_size=m["img"])[0]
            ps[m["family"]].append(torch.sigmoid(out.float()).cpu().numpy()[0])
        for f, fp in ps.items():
            if fp:
                fam_preds[f][i] = np.mean(fp, 0)
                fam_ok[f][i] = True
        if not any(ps.values()):
            n_fail += 1
    except Exception as e:
        # One unreadable study must not cost the whole submission. It keeps 0.5 and the
        # count is reported, because a silent failure looks identical to a healthy run.
        n_fail += 1
        if n_fail <= 5:
            log(f"study {sid[-10:]} failed: {type(e).__name__}: {e}")
    if (i + 1) % 100 == 0:
        el = time.time() - T0
        log(f"{i + 1}/{len(test_ids)}  eta {(len(test_ids) - i - 1) * el / (i + 1) / 60:.0f} min")

log(f"predicted {len(test_ids) - n_fail}/{len(test_ids)} studies ({n_fail} fell back to 0.5)")
for f in families:
    log(f"    family {f}: {int(fam_ok[f].sum())}/{len(test_ids)} studies")


def family_scores(f):
    """One family's matrix, combined the way the whole ensemble always was.

    Rank-normalise per column when the family has more than one member: AUC reads only
    order, and averaging raw probabilities lets whichever member is most confident dominate.
    """
    P = fam_preds[f]
    if sum(m["family"] == f for m in members) > 1:
        from scipy.stats import rankdata
        P = np.apply_along_axis(rankdata, 0, P) / len(P)
    return P


if len(families) == 1:
    # One family: exactly the pre-family computation, so a v1-only run is unchanged.
    preds = family_scores(families[0])
else:
    # Families score on different scales (two CNNs' sigmoids, or ranks against raw
    # probabilities when one family has a single member), so each is ranked per column
    # first, then weighted over the families that actually predicted the study.
    from scipy.stats import rankdata
    num = np.zeros((len(test_ids), len(TARGETS)))
    den = np.zeros(len(test_ids))
    for f in families:
        R = np.apply_along_axis(rankdata, 0, family_scores(f)) / len(test_ids)
        w = FAMILY_WEIGHTS.get(f, 1.0) * fam_ok[f]
        num += w[:, None] * R
        den += w
    preds = np.where(den[:, None] > 0, num / np.maximum(den, 1e-12)[:, None], 0.5)
    log(f"family blend weights {({f: FAMILY_WEIGHTS.get(f, 1.0) for f in families})}")

sub = pd.DataFrame(preds, columns=TARGETS)
sub.insert(0, "StudyInstanceUID", test_ids)

# Refuse to ship a submission that silently forfeits score.
problems = []
for t in TARGETS:
    v = sub[t]
    if v.isna().any():
        problems.append(f"{t}: NaN")
    if not ((v >= 0) & (v <= 1)).all():
        problems.append(f"{t}: outside [0,1]")
    if v.nunique() <= 1:
        problems.append(f"{t}: CONSTANT — forfeits 1/12 of the score")
if problems:
    log("SUBMISSION PROBLEMS: " + "; ".join(problems))
    log("keeping the benchmark file rather than shipping a broken one")
else:
    sub.to_csv(OUT, index=False)
    log(f"wrote {OUT}  {sub.shape}")

log(f"total runtime {(time.time() - T0) / 60:.1f} min "
    f"(limit 540 min; efficiency term = runtime/32400)")
