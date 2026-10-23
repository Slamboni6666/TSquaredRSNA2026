"""Build the C02R7 push dir: C02 with heliosli's R7 in place of S6. Three families, per-finding
RANK blend with FIXED weights ours 0.40 / Raptor 0.30 / R7 0.30 (pre-registered 2026-10-15):
  ours    our 13 ConvNeXt-small checkpoints, the frozen 4f96cbd pipeline (weights pinned)
  raptor  nartaa's public Raptor-384 (nartaa/rsna-knee-publication-swa-weights-20261007 v1), his
          own inference source, run exactly as in C01/C02 (isolated subprocess, 90 min timeout)
  r7      heliosli's public R7 ConvNeXt-small, 4 folds (heliosli/rsna-knee-compact-r7cnxfsss v1,
          bundle 8c7b666e27dc0e8f, manifest sha256 c4bb0f73...), run by the R7 stage (cells 4 + 5)
          of heliosli's scored notebook heliosli/0-957-latest-dataset-score-in-one-click
          (byte-identical, ipynb md5 95920079...; the same cells as bliverrigpdoge's 0.957 notebook
          and, apart from the bundle slug / a FLIP_TTA=False switch, his 0.9573 notebook), with the
          SAME 7 asserted orchestration patches as C02's S6 stage, as a second isolated subprocess
          with its own deadline / hard stop / private package dir

This is build_c02.py with the S6 stage swapped for the R7 stage; everything else (frozen-cell
edits, Raptor stage, budget, our stage, blend) is C02's, character for character apart from the
s6 -> r7 names. Every change to the frozen cell is an exact-string edit asserted to occur once, or
the re-indentation of one contiguous region into a try block (checked to be whitespace-only).

Notebook layout
  cell 0  data: nartaa cells 3+4 (as C01/C02) and heliosli 0-957 cells 4+5, base64, byte-identical
  cell 1  the frozen 4f96cbd cell with the C02R7 edits

usage: python kaggle/build/build_c02r7.py [--third-party-dir DIR] [--out DIR]
  Rebuilds kaggle/rsna-knee-submission.ipynb from files shipped in this repo:
    kaggle/build/kaggle_notebook_4f96cbd.py  (= git show 4f96cbd:kaggle_notebook.py)
    kaggle/build/kpush_4f96cbd.ipynb         (the frozen 4f96cbd notebook, md5 7b32f8ac...)
    <third-party-dir>/nartaa/cell3.py, cell4.py and <third-party-dir>/heliosli/0957_cell4.txt,
    0957_cell5.txt (fetched by third_party/fetch_third_party.py; sha256 checked here)
  and asserts the rebuilt notebook's md5 is NB_MD5 below. The scored notebook (md5 c9b16569...)
  differed in one string only: ours_weights_dir() also tried a mount path under the team's
  Kaggle account, which this public copy drops (the fold*.pt search finds the weights).
  Writes <out>/rsna-knee-submission.ipynb, <out>/c02r7_cell.py and <out>/c02r7_data_cell.py.
"""
import argparse
import base64
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent                 # kaggle/build
REPO_ROOT = HERE.parents[1]
_ap = argparse.ArgumentParser(description="rebuild the C02R7 Kaggle notebook byte-identically")
_ap.add_argument("--third-party-dir", default=str(REPO_ROOT / "third_party" / "cache"),
                 help="folder with nartaa/ and heliosli/ cells (third_party/fetch_third_party.py)")
_ap.add_argument("--out", default=str(HERE / "out"), help="output folder")
_args = _ap.parse_args()
TP = Path(_args.third_party_dir)
OUT_DIR = Path(_args.out)
NB_MD5 = "c86262fa868ca7252c7c0ff629633796"           # scored notebook minus the account mount path
RAPTOR_CELLS = {"cell3": "fd96541cbe2e5b37494636dd13b79871a7a6ed9ad94fe528dc6f7a253d4be8b8",
                "cell4": "bcd864cbf349882e876959b49cb348cbb89b4c62fc204d4e4261049c5e3a79c5"}
# heliosli/0-957-latest-dataset-score-in-one-click (ipynb md5 95920079d1bb9b236821401907daa411,
# pulled 10-14 and again 10-15, identical): code cells 4 (R7 assets) and 5 (R7 controller run)
R7_CELLS = {"cell4": ("0957_cell4.txt",
                      "cb168cfe82f5a04114894f6c07bbd6daf0373ad2eb0dfcee34df3ecf625986af"),
            "cell5": ("0957_cell5.txt",
                      "68f70dd31cc3544cb5734313d06538355a34b122ff430a32820756175c9c9dd9")}
R7_NB_MD5 = "95920079d1bb9b236821401907daa411"
NEW_DS = ["nartaa/rsna-knee-publication-swa-weights-20261007",
          "heliosli/rsna-knee-compact-r7cnxfsss",
          "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"]

frozen = (HERE / "kaggle_notebook_4f96cbd.py").read_bytes().decode("utf-8").replace("\r\n", "\n")
nb0 = json.loads((HERE / "kpush_4f96cbd.ipynb").read_text(encoding="utf-8"))
assert hashlib.md5((HERE / "kpush_4f96cbd.ipynb").read_bytes()).hexdigest() \
    == "7b32f8acbefd5651251459d4afa67a8b"
_src0 = nb0["cells"][0]["source"]
assert ("".join(_src0) if isinstance(_src0, list) else _src0) == frozen, "frozen nb != 4f96cbd"

b64 = {}
for k, sha in RAPTOR_CELLS.items():
    s = (TP / "nartaa" / f"{k}.py").read_bytes()
    assert hashlib.sha256(s).hexdigest() == sha, k
    b64["RAPTOR_" + k.upper()] = base64.b64encode(s).decode("ascii")
for k, (fn, sha) in R7_CELLS.items():
    s = (TP / "heliosli" / fn).read_bytes()
    assert hashlib.sha256(s).hexdigest() == sha, k
    b64["R7_" + k.upper()] = base64.b64encode(s).decode("ascii")


def edit(src, old, new, n=1):
    assert src.count(old) == n, (src.count(old), old[:80])
    return src.replace(old, new)


# ------------------------------------------------------------------ 1. docstring
C02R7_DOC = '''"""
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
'''
src = C02R7_DOC + "\n" + frozen

# ------------------------------------------------------------------ 2. our weights pinned
src = edit(src,
           'WEIGHTS = find_dir("*.pt") or find_dir("fold0*.pt")\n',
           '# C01/C02: PINNED to our dataset. find_dir("*.pt") returns the first .pt under /kaggle/input\n'
           '# in unspecified rglob order, which can be nartaa\'s or heliosli\'s checkpoint (no ck["args"]).\n'
           'WEIGHTS = ours_weights_dir()\n')

# ------------------------------------------------------------------ 3. our deadline guard
src = edit(src, 'n_fail = 0\n', 'n_fail = 0\nn_late = 0   # C02: studies not started before OURS_STOP\n')
src = edit(src,
           'for i, sid in enumerate(test_ids):\n    try:\n',
           'for i, sid in enumerate(test_ids):\n'
           '    if time.time() > OURS_STOP:\n'
           '        # C02: past the stop time a study keeps the other families\' ranks; running on\n'
           '        # into the 9 h limit would score nothing at all\n'
           '        n_fail += 1\n'
           '        n_late += 1\n'
           '        continue\n'
           '    try:\n')
src = edit(src,
           'log(f"predicted {len(test_ids) - n_fail}/{len(test_ids)} studies ({n_fail} fell back to 0.5)")\n',
           'log(f"predicted {len(test_ids) - n_fail}/{len(test_ids)} studies ({n_fail} fell back to 0.5)")\n'
           'if n_late:\n'
           '    log(f"C02 our stage: stop time T0+{OURS_STOP_MIN} min reached, {n_late} studies not started")\n')

# ------------------------------------------------------------------ 4. Raptor + R7 stages
STAGES = r'''
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
'''
START = 'CODE = find_dir("inference.py")\n'
SPLIT = 'sub = pd.DataFrame(preds, columns=TARGETS)\n'
assert src.count(START) == 1 and src.count(SPLIT) == 1
head, rest = src.split(START)
body, tail = rest.split(SPLIT)
body = START + body
ind = "".join(("    " + ln if ln.strip() else ln) for ln in body.splitlines(True))
assert re.sub(r"\s+", "", ind) == re.sub(r"\s+", "", body)          # whitespace-only change
OURS_END = r'''    OURS = (np.asarray(preds, np.float64),
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
'''
src = head + STAGES + ind + OURS_END + tail
compile(src, "c02r7_cell", "exec")

OUT_DIR.mkdir(parents=True, exist_ok=True)    # files are overwritten on rebuild, never deleted
data_cell = ("# C02R7 data: nartaa/rsna-knee-0949-anatomical-mirror code cells 3 and 4, and\n"
             "# heliosli/0-957-latest-dataset-score-in-one-click code cells 4 and 5, byte-identical,\n"
             "# base64 (sha256 checked before use). Public sources; credits in the next cell's docstring.\n"
             + "".join(f'{k}_B64 = "{v}"\n' for k, v in b64.items()))
nb = json.loads(json.dumps(nb0))
cell_tpl = nb["cells"][0]
nb["cells"] = [dict(cell_tpl, source=data_cell), dict(cell_tpl, source=src)]
# The original build ran on Windows, where text-mode writes turn "\n" into "\r\n"; newline="\r\n"
# reproduces those bytes on any OS. kernel-metadata.json is not written here: the shipped
# kaggle/kernel-metadata.json carries placeholder slugs (dataset_sources = frozen ones + NEW_DS).
for _name, _text in (("rsna-knee-submission.ipynb", json.dumps(nb, indent=1)),
                     ("c02r7_cell.py", src), ("c02r7_data_cell.py", data_cell)):
    with open(OUT_DIR / _name, "w", encoding="utf-8", newline="\r\n") as _fh:
        _fh.write(_text)
_md5 = hashlib.md5((OUT_DIR / "rsna-knee-submission.ipynb").read_bytes()).hexdigest()
print("wrote", OUT_DIR, "md5", _md5, "bytes", (OUT_DIR / "rsna-knee-submission.ipynb").stat().st_size)
assert _md5 == NB_MD5, f"rebuilt notebook md5 {_md5} != {NB_MD5}"
