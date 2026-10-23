"""
C02R7 -- THREE FAMILIES, PER-FINDING RANK BLEND, FIXED WEIGHTS ours 0.40 / raptor 0.30 / r7 0.30
(pre-registered 2026-10-15; never tuned on the leaderboard). C02 with heliosli's R7 in place of S6.

Owner decision 10-15: other teams' PUBLIC weights may be blended (host ruling 10-14, topic
744056: publicly shared weights are prize-eligible when listed), including heliosli's R7 with
its fastMRI+ auxiliary training task. Listed resources:
 * nartaa/rsna-knee-publication-swa-weights-20261007 v1, raptor_ft_alldata_t16_blendjev_oai_d96_
   r384_swa.pt (sha256 7e5315da...), run by nartaa's own inference source (nartaa/rsna-knee-0949-
   anatomical-mirror cells 3+4, scored source sha256 bcd864cb..., public LB 0.949). Credits:
   nartaa; dreaddevelopment (Raptor recipe); timm CoAtNet; weak labels by stevenleehans, pilkwang,
   riadmohamed42 (see nartaa's dataset README).
 * heliosli/rsna-knee-compact-r7cnxfsss v1, R7 bundle 8c7b666e27dc0e8f: four ConvNeXt-small fold
   models E_lrprior_s0_f0..3 (trained with a fastMRI+ lesion-box auxiliary task, fm_aux 0.3),
   side_cls.pt, state_spec.json, Effusion fluid stacker and Baker's state stacker
   (manifest.json sha256 c4bb0f73...). Run by the R7 stage of heliosli's public scored notebook
   heliosli/0-957-latest-dataset-score-in-one-click (cells 4 + 5, byte-identical, sha256
   cb168cfe... / 68f70dd3...), with heliosli/rsna-orthofoundation-fold0-epoch7-20261004 v1 (his
   decode/render code, pinned pydicom 3.0.1 / timm 1.0.25 wheels, and anatomy.pt = the SKM-TEA
   model-zoo V-Net, upstream licence retained). Credits: heliosli; timm ConvNeXt; fastMRI+.
 * OAI: Data and/or research tools used in the preparation of this manuscript were obtained and
   analyzed from the controlled access datasets distributed from the Osteoarthritis Initiative
   (OAI), a data repository housed within the NIMH Data Archive (NDA). Dataset identifier(s):
   10.15154/0hcg-f676.

Order: (a) benchmark submission; (b) Raptor exactly as C01 (isolated subprocess, 90 min hard
timeout); (c) R7: heliosli's cells 4 + 5, unchanged except seven asserted orchestration patches
(no IPython display; its temp dir, run dir and output file are ours; its pinned wheels go to a
private --target dir on its own PYTHONPATH instead of the global site-packages; its deadline is
our budget), as an isolated subprocess with a hard stop; whatever its workers scored is read from
their predictions.csv (flushed per study, so a stopped run still counts); (d) our pipeline exactly
as 4f96cbd, weights PINNED to our dataset, plus a deadline guard (a timed-out notebook scores
nothing); (e) per finding, each family is ranked over the studies it scored, final = the
pre-registered weighted mean of the ranks of the families that scored the study (weights
renormalised over those families; 0.5 if none did); (f) /kaggle/working/submission.csv.

Budget (9 h limit = 540 min from our start): the R7 deadline leaves our stage
60 + 11 min per checkpoint (fit to this notebook's 7 scored runs) plus a 25 min margin.
"""

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


# --------------------------------------------------------------------------- C02 setup
WORK = Path("/kaggle/working")
C02_W = {"ours": 0.40, "raptor": 0.30, "r7": 0.30}   # pre-registered 2026-10-15; never LB-tuned
LIMIT_MIN = 540          # Kaggle's 9 h, counted from our T0 (kernel start is a minute or two earlier)
SAFETY_MIN = 25          # kernel start, blend, writing, margin
OURS_STOP_MIN = 520      # our stage starts no study after T0 + this
OURS_STOP = T0 + 60 * OURS_STOP_MIN


def ours_weights_dir():
    return next((p for p in (Path("/kaggle/input/rsna-knee-weights"),) if p.is_dir()),
                None) or find_dir("fold*.pt")


_wd = ours_weights_dir()
N_OURS = len(sorted(Path(_wd).rglob("*.pt"))) if _wd else 0     # what our loader will load
OURS_RESERVE_MIN = 60 + 11 * N_OURS     # 7 scored runs of this notebook: 60 + 11.1 min/checkpoint
R7_DEADLINE = T0 + 60 * (LIMIT_MIN - SAFETY_MIN - OURS_RESERVE_MIN)
R7_GRACE_MIN = 10        # hard stop this long after his deadline (his workers check it per study)
log(f"C02 budget: {N_OURS} of our checkpoints -> {OURS_RESERVE_MIN} min reserved for our stage; "
    f"r7 deadline T0+{(R7_DEADLINE - T0) / 60:.0f} min, hard stop +{R7_GRACE_MIN} min; "
    f"our stop T0+{OURS_STOP_MIN} min")

# --------------------------------------------------------------------------- Raptor family
# nartaa's public Raptor-384 inference, run FIRST as an isolated subprocess: its own run dir,
# its own interpreter process (nothing of ours is imported or on its path yet), its own log, a
# hard timeout that kills its whole process group (its fork pool included). Its output can only
# reach us through raptor_raw.csv; if anything goes wrong the blend proceeds without it.
RAPTOR_FILE = "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt"
RAPTOR_PT_SHA = "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef"
RAPTOR_SRC_SHA = {"cell3": "fd96541cbe2e5b37494636dd13b79871a7a6ed9ad94fe528dc6f7a253d4be8b8",
                  "cell4": "bcd864cbf349882e876959b49cb348cbb89b4c62fc204d4e4261049c5e3a79c5"}
RAPTOR_TIMEOUT_S = 90 * 60
RAPTOR_DIR = WORK / "raptor_run"
RAPTOR_RAW = WORK / "raptor_raw.csv"
# (old, new, expected count). Model, preprocessing, windows, TTA and his rank output: untouched.
RAPTOR_PATCHES = [
    # 1. his submission goes to raptor_sub.csv, never over ours
    ('    out = "/kaggle/working/submission.csv"\n',
     '    out = os.environ["RAPTOR_OUT"]\n', 1),
    # 2. remember studies whose volume came back empty (his _pre_one failure path predicts on zeros)
    ('    q = queue.Queue(maxsize=QUEUE_MAX); done_n = [0]; lock = threading.Lock()\n',
     '    q = queue.Queue(maxsize=QUEUE_MAX); done_n = [0]; lock = threading.Lock()\n'
     '    _C01_EMPTY = set()\n', 1),
    ('            sid, vol, mask = item; i = pos[sid]\n',
     '            sid, vol, mask = item; i = pos[sid]\n'
     '            if int(mask.sum()) == 0: _C01_EMPTY.add(sid)\n', 1),
    # 3. raw sigmoid probabilities (his CSV holds ranks, where a 0.5 FALLBACK row is invisible)
    ('    print(f"[gpu] done: {done_n[0]}/{N} studies on {len(devs)} device(s) | {time.time()-t0:.0f}s", flush=True)\n',
     '    print(f"[gpu] done: {done_n[0]}/{N} studies on {len(devs)} device(s) | {time.time()-t0:.0f}s", flush=True)\n'
     '    _raw = pd.DataFrame(arm_probs[0].astype(np.float64), columns=LAB)\n'
     '    _raw.insert(0, "StudyInstanceUID", test_ids)\n'
     '    _raw["empty_volume"] = [int(s in _C01_EMPTY) for s in test_ids]\n'
     '    _raw.to_csv(os.environ["RAPTOR_RAW"], index=False, float_format="%.9g")\n', 1),
    # 4. _fastread.py is written next to the script (his non-Kaggle branch), not into /kaggle/working
    ("    p = '/kaggle/working/_fastread.py' if os.path.isdir('/kaggle/working') else os.path.join(",
     "    p = os.path.join(", 1),
]
RAPTOR_PATCHES_REPLAY = []   # local replay harness only; empty on Kaggle


def run_raptor():
    """-> (P, ok): his raw probabilities in test_ids order and which studies he really scored,
    or None. Never raises."""
    import base64
    import hashlib
    import signal
    import subprocess
    srcs = {}
    for k, b in (("cell3", RAPTOR_CELL3_B64), ("cell4", RAPTOR_CELL4_B64)):
        srcs[k] = base64.b64decode(b)
        h = hashlib.sha256(srcs[k]).hexdigest()
        if h != RAPTOR_SRC_SHA[k]:
            log(f"raptor: {k} source sha256 {h[:16]} is not the published one -> no raptor family")
            return None
    pt_dir = find_dir(RAPTOR_FILE)
    if pt_dir is None:
        log("raptor: weights dataset not attached -> no raptor family")
        return None
    with open(pt_dir / RAPTOR_FILE, "rb") as fh:
        h = hashlib.file_digest(fh, "sha256").hexdigest()
    if h != RAPTOR_PT_SHA:
        log(f"raptor: {pt_dir / RAPTOR_FILE} sha256 {h[:16]} is not the published one -> no raptor family")
        return None
    log(f"raptor: weights {pt_dir / RAPTOR_FILE} sha256 {h[:16]} = published checkpoint")
    src = srcs["cell3"].decode("utf-8") + "\n\n" + srcs["cell4"].decode("utf-8")
    for old, new, n in RAPTOR_PATCHES + RAPTOR_PATCHES_REPLAY:
        if src.count(old) != n:
            log(f"raptor: patch anchor seen {src.count(old)}x, want {n}: {old.strip()[:60]!r} -> no raptor family")
            return None
        src = src.replace(old, new)
    RAPTOR_DIR.mkdir(parents=True, exist_ok=True)
    (RAPTOR_DIR / "raptor_run.py").write_text(src, encoding="utf-8")
    env = dict(os.environ, RAPTOR_OUT=str(WORK / "raptor_sub.csv"), RAPTOR_RAW=str(RAPTOR_RAW))
    t = time.time()
    with open(RAPTOR_DIR / "raptor.log", "w", encoding="utf-8") as fh:
        proc = subprocess.Popen([sys.executable, "-u", "raptor_run.py"], cwd=RAPTOR_DIR, env=env,
                                stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=RAPTOR_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            if hasattr(os, "killpg"):
                os.killpg(proc.pid, signal.SIGKILL)   # its process group: fork-pool workers too
            else:
                proc.kill()
            proc.wait()
            rc = f"killed at the {RAPTOR_TIMEOUT_S} s timeout"
    lines = (RAPTOR_DIR / "raptor.log").read_text(encoding="utf-8", errors="replace").splitlines()
    for ln in (lines if len(lines) <= 120 else lines[:60] + [f"... {len(lines) - 120} lines ..."] + lines[-60:]):
        print("  raptor| " + ln, flush=True)
    log(f"raptor: subprocess exit {rc} after {(time.time() - t) / 60:.1f} min")
    if rc != 0 or not RAPTOR_RAW.is_file():
        log("raptor: no raw probabilities -> no raptor family")
        return None
    raw = pd.read_csv(RAPTOR_RAW, dtype={"StudyInstanceUID": str}).drop_duplicates(
        "StudyInstanceUID").set_index("StudyInstanceUID").reindex(test_ids)
    P = raw[TARGETS].to_numpy(np.float64)
    missing = np.isnan(P).any(1)
    nonfinite = ~np.isfinite(P).all(1) & ~missing
    half = (P == 0.5).all(1)                       # his FALLBACK rows keep the 0.5 init
    empty = raw["empty_volume"].fillna(0).to_numpy() > 0      # missing rows counted above
    ok = ~(missing | nonfinite | half | empty)
    log(f"C02 family raptor: scored {int(ok.sum())}/{len(test_ids)} studies "
        f"(not scored: missing {int(missing.sum())}, all-0.5 {int(half.sum())}, "
        f"empty volume {int(empty.sum())}, non-finite {int(nonfinite.sum())})")
    return np.where(ok[:, None], P, 0.5), ok


T_RAPTOR = time.time()
try:
    RAPTOR = run_raptor()
except Exception as e:   # the Raptor stage must never cost the other families
    log(f"raptor: stage stopped ({type(e).__name__}: {e}) -> no raptor family")
    RAPTOR = None
log(f"C02 raptor stage {(time.time() - T_RAPTOR) / 60:.1f} min")

# --------------------------------------------------------------------------- R7 family
# heliosli's public R7 stage (cells 4 + 5 of his scored notebook 0-957-latest-dataset-score-in-
# one-click, byte-identical, sha-checked here and again by its own asserts: code dataset modules,
# his 7 embedded sources, every bundle file and both stackers against the bundle manifest, the
# trainer md5, anatomy.pt), run SECOND as an isolated subprocess: its own run dir, its own temp
# dir, its pinned wheels in a private --target dir that only its processes see (PYTHONPATH), a
# hard stop that kills its whole process group (controller, both GPU workers and their spawn
# loaders). Its scores reach us only through his workers' predictions.csv.
R7_SRC_SHA = {"cell4": "cb168cfe82f5a04114894f6c07bbd6daf0373ad2eb0dfcee34df3ecf625986af",
              "cell5": "68f70dd31cc3544cb5734313d06538355a34b122ff430a32820756175c9c9dd9"}
R7_SLUG = "rsna-knee-compact-r7cnxfsss"
R7_BUNDLE = "8c7b666e27dc0e8f"
R7_MANIFEST_SHA = "c4bb0f7399ada9a020e0c15cae6252033cf0c4335e999961793a176063074ec6"   # v1
R7_DIR = WORK / "r7_run"
R7_OUT = WORK / "r7_sub.csv"            # his controller's submission (failed studies = column mean)
R7_RAW = WORK / "r7_raw.csv"            # what his workers really scored + ok flag (blend input)
R7_TMP = (Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")) / "c02_r7"
# his cell 4 reads NB_T0 (set by his Raptor cell) and cell 5 calls IPython's display()
R7_PRELUDE = ('import os\n'
              'NB_T0 = float(os.environ["R7_NB_T0"])   # [C02R7] his Raptor cell set this\n'
              'display = print                         # [C02R7] no IPython in a subprocess\n\n')
# (old, new, expected count). Model, bundle, decode, views, windows, stackers, compile: untouched.
R7_PATCHES = [
    # 1. his temp dir is ours (fresh, known to the notebook: his workers' predictions.csv live there)
    ("TMP = TMP / 'probe'\n", "TMP = Path(os.environ['R7_TMP'])   # [C02R7]\n", 1),
    # 2. his receipt and worker logs go to our r7 run dir, not to /kaggle/working
    ("WORK = Path('/kaggle/working')\n", "WORK = Path(os.environ['R7_WORK'])   # [C02R7]\n", 1),
    # 3. his pinned wheels (pydicom 3.0.1, timm 1.0.25) into a private dir, not the global env
    ("subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', *wheels], check=True)\n",
     "subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--target', "
     "str(TMP / 'site'), *wheels], check=True)   # [C02R7] private, not the global env\n", 1),
    ("env = dict(os.environ, PYTHONPATH=str(CODE) + os.pathsep + os.environ.get('PYTHONPATH', ''), "
     "PYTHONDONTWRITEBYTECODE='1')\n",
     "env = dict(os.environ, PYTHONPATH=str(CODE) + os.pathsep + str(TMP / 'site') + os.pathsep + "
     "os.environ.get('PYTHONPATH', ''), PYTHONDONTWRITEBYTECODE='1')   # [C02R7] + his wheels\n", 1),
    # 4. his deadline is our budget (his own: notebook start + 8.3 h)
    ("deadline = T0 + DEADLINE_HOURS * 3600\n",
     "deadline = float(os.environ['R7_DEADLINE'])   # [C02R7] our budget\n", 1),
    # 5. his submission file goes to r7_sub.csv, never over anything of ours
    ("shutil.copyfile(pred / 'submission.csv', WORK / 'ours_sub.csv')\n",
     "shutil.copyfile(pred / 'submission.csv', os.environ['R7_OUT'])   # [C02R7]\n", 1),
    ("display(pd.read_csv(WORK / 'ours_sub.csv').head())",
     "display(pd.read_csv(os.environ['R7_OUT']).head())   # [C02R7]", 1),
]
R7_PATCHES_REPLAY = []   # local replay harness only; empty on Kaggle


def r7_mounts(slug, required):
    """his find_mount's candidate list (shallow globs only), every match"""
    inp = Path("/kaggle/input")
    c = [inp / slug, inp / "competitions" / slug]
    c += list(inp.glob("datasets/*/" + slug)) + list(inp.glob("datasets/*/" + slug + "/*"))
    c += list(inp.glob("*/" + slug)) + list(inp.glob(slug + "/*"))
    return list(dict.fromkeys(p for p in c if (p / required).is_file()))


def read_r7():
    """-> (P, ok) from his workers' predictions.csv (one row per scored study, written and
    flushed as each study finishes), or None if nothing was scored."""
    parts = []
    for f in sorted((R7_TMP / "prediction").glob("worker*/predictions.csv")):
        try:
            parts.append(pd.read_csv(f, dtype={"StudyInstanceUID": str}))
        except Exception as e:   # e.g. a row cut by the hard stop
            log(f"r7: {f.parent.name}/predictions.csv unreadable ({type(e).__name__}) -> its rows not scored")
    if not parts:
        return None
    df = pd.concat(parts).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    P = df.reindex(test_ids)[TARGETS].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    ok = np.isfinite(P).all(1) & ((P >= 0) & (P <= 1)).all(1)
    return np.where(ok[:, None], P, 0.5), ok


def run_r7():
    """-> (P, ok) or None. Never raises."""
    import base64
    import hashlib
    import signal
    import subprocess
    srcs = {}
    for k, b in (("cell4", R7_CELL4_B64), ("cell5", R7_CELL5_B64)):
        srcs[k] = base64.b64decode(b)
        h = hashlib.sha256(srcs[k]).hexdigest()
        if h != R7_SRC_SHA[k]:
            log(f"r7: {k} source sha256 {h[:16]} is not the published one -> no r7 family")
            return None
    found = r7_mounts(R7_SLUG, "manifest.json")
    if len(found) != 1:
        log(f"r7: {R7_SLUG} attached {len(found)} times (want 1) -> no r7 family")
        return None
    h = hashlib.sha256((found[0] / "manifest.json").read_bytes()).hexdigest()
    if h != R7_MANIFEST_SHA:
        log(f"r7: {found[0] / 'manifest.json'} sha256 {h[:16]} is not the published one -> no r7 family")
        return None
    log(f"r7: bundle manifest {found[0] / 'manifest.json'} sha256 {h[:16]} = published bundle {R7_BUNDLE}")
    left = R7_DEADLINE - time.time()
    if left < 30 * 60:
        log(f"r7: only {left / 60:.0f} min to its deadline -> no r7 family")
        return None
    src = R7_PRELUDE + srcs["cell4"].decode("utf-8") + "\n\n" + srcs["cell5"].decode("utf-8")
    for old, new, n in R7_PATCHES + R7_PATCHES_REPLAY:
        if src.count(old) != n:
            log(f"r7: patch anchor seen {src.count(old)}x, want {n}: {old.strip()[:60]!r} -> no r7 family")
            return None
        src = src.replace(old, new)
    if R7_TMP.exists():
        log(f"r7: {R7_TMP} already exists -> no r7 family")
        return None
    R7_DIR.mkdir(parents=True, exist_ok=True)
    (R7_DIR / "r7_run.py").write_text(src, encoding="utf-8")
    env = dict(os.environ, R7_NB_T0=repr(T0), R7_TMP=str(R7_TMP), R7_WORK=str(R7_DIR),
               R7_OUT=str(R7_OUT), R7_DEADLINE=repr(R7_DEADLINE))
    t = time.time()
    with open(R7_DIR / "r7.log", "w", encoding="utf-8") as fh:
        proc = subprocess.Popen([sys.executable, "-u", "r7_run.py"], cwd=R7_DIR, env=env,
                                stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=max(60.0, R7_DEADLINE + 60 * R7_GRACE_MIN - time.time()))
        except subprocess.TimeoutExpired:
            if hasattr(os, "killpg"):
                os.killpg(proc.pid, signal.SIGKILL)   # controller, GPU workers, spawn loaders
            else:
                proc.kill()
            proc.wait()
            rc = f"killed at the hard stop T0+{(R7_DEADLINE - T0) / 60 + R7_GRACE_MIN:.0f} min"
    lines = (R7_DIR / "r7.log").read_text(encoding="utf-8", errors="replace").splitlines()
    for ln in (lines if len(lines) <= 120 else lines[:60] + [f"... {len(lines) - 120} lines ..."] + lines[-60:]):
        print("  r7| " + ln, flush=True)
    log(f"r7: subprocess exit {rc} after {(time.time() - t) / 60:.1f} min")
    got = read_r7()
    if got is None or got[1].sum() == 0:
        log("r7: no scored studies -> no r7 family")
        return None
    P, ok = got
    _r = pd.DataFrame(P, columns=TARGETS)
    _r.insert(0, "StudyInstanceUID", test_ids)
    _r["ok"] = ok.astype(int)
    _r.to_csv(R7_RAW, index=False, float_format="%.9g")
    if R7_OUT.is_file():   # his own file: scored rows must be the same numbers (both %.9g)
        sub7 = pd.read_csv(R7_OUT, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
        d = np.abs(sub7.reindex(test_ids)[TARGETS].to_numpy(np.float64)[ok] - P[ok])
        log(f"r7: r7_sub.csv vs workers' rows: max |diff| {np.nanmax(d) if d.size else 0.0:.1e}")
    log(f"C02 family r7: scored {int(ok.sum())}/{len(test_ids)} studies "
        f"(not scored: {int((~ok).sum())})")
    return P, ok


T_R7 = time.time()
try:
    R7 = run_r7()
except Exception as e:   # the R7 stage must never cost the other families
    log(f"r7: stage stopped ({type(e).__name__}: {e}) -> no r7 family")
    R7 = None
log(f"C02 r7 stage {(time.time() - T_R7) / 60:.1f} min")

# --------------------------------------------------------------------------- our family
T_OURS = time.time()
OURS = None
try:
    CODE = find_dir("inference.py")
    # C01/C02: PINNED to our dataset. find_dir("*.pt") returns the first .pt under /kaggle/input
    # in unspecified rglob order, which can be nartaa's or heliosli's checkpoint (no ck["args"]).
    WEIGHTS = ours_weights_dir()
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
    n_late = 0   # C02: studies not started before OURS_STOP
    n_prep_fail = {}

    for i, sid in enumerate(test_ids):
        if time.time() > OURS_STOP:
            # C02: past the stop time a study keeps the other families' ranks; running on
            # into the 9 h limit would score nothing at all
            n_fail += 1
            n_late += 1
            continue
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
    if n_late:
        log(f"C02 our stage: stop time T0+{OURS_STOP_MIN} min reached, {n_late} studies not started")
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

    OURS = (np.asarray(preds, np.float64),
            np.logical_or.reduce([fam_ok[f] for f in families]))
    log(f"C02 family ours: scored {int(OURS[1].sum())}/{len(test_ids)} studies")
    _o = pd.DataFrame(OURS[0], columns=TARGETS)
    _o.insert(0, "StudyInstanceUID", test_ids)
    _o["ok"] = OURS[1].astype(int)
    _o.to_csv(WORK / "ours_scores.csv", index=False, float_format="%.12g")
except Exception as e:   # our pipeline stopping must not cost the other families
    log(f"our pipeline stopped ({type(e).__name__}: {e}) -> no ours family")
    OURS = None
log(f"C02 our stage {(time.time() - T_OURS) / 60:.1f} min")

# --------------------------------------------------------------------------- C02 blend
# Per finding, each family is ranked over the studies IT scored (rankdata / n_scored, ties
# averaged, so every family sits on the same (0, 1] scale); a study takes the C02_W-weighted mean
# of the ranks of the families that scored it (weights renormalised over those families), and
# 0.5 when none did.
from scipy.stats import rankdata  # noqa: E402

FAMS = {k: v for k, v in (("ours", OURS), ("raptor", RAPTOR), ("r7", R7)) if v is not None}
c02_num = np.zeros((len(test_ids), len(TARGETS)))
c02_den = np.zeros(len(test_ids))
for name, (P, ok) in FAMS.items():
    if ok.sum() == 0:
        continue
    R = np.zeros_like(c02_num)
    R[ok] = np.apply_along_axis(rankdata, 0, P[ok]) / ok.sum()
    c02_num += C02_W[name] * ok[:, None] * R
    c02_den += C02_W[name] * ok
final = np.where(c02_den[:, None] > 0, c02_num / np.maximum(c02_den, 1e-12)[:, None], 0.5)
_ok = {k: (FAMS[k][1] if k in FAMS else np.zeros(len(test_ids), bool)) for k in C02_W}
_pat = {}
for i in range(len(test_ids)):
    key = "+".join(k for k in C02_W if _ok[k][i]) or "none"
    _pat[key] = _pat.get(key, 0) + 1
log(f"C02 blend: per-finding rank average, weights ours {C02_W['ours']} / raptor "
    f"{C02_W['raptor']} / r7 {C02_W['r7']}; studies by families: "
    + ", ".join(f"{k} {v}" for k, v in sorted(_pat.items(), key=lambda kv: -kv[1])))

sub = pd.DataFrame(final, columns=TARGETS)
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
