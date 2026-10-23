"""
package_for_kaggle.py -- assemble and upload the two datasets the submission notebook needs.

  rsna-knee-code     the modules that define preprocessing and the model. Uploading these
                     rather than pasting their contents into the notebook is what keeps
                     scoring-time preprocessing identical to training -- `inference.py
                     --verify` proves that property for these exact files, and a pasted
                     copy would silently diverge the first time either side is edited.

  rsna-knee-weights  the fold checkpoints.

Both are versioned: re-running uploads a new version rather than replacing, so a notebook
pinned to an older version keeps working and a bad upload is recoverable.

In this repository the code dataset is staged from kaggle/code_dataset/ (the exact modules the
submission notebook ran with, Kaggle code dataset v5 = commit 4f96cbd, local paths neutralised) and
the weights from src/checkpoints/ (where src/train.py writes them). Both are uploaded under
your own Kaggle account, licence CC BY-NC 4.0.

Usage:
    python kaggle/package_for_kaggle.py --dry-run      # stage and inspect, upload nothing
    python kaggle/package_for_kaggle.py --code
    python kaggle/package_for_kaggle.py --weights      # default --runs = the 13 checkpoints
    python kaggle/package_for_kaggle.py --all --message "final submission, 13 checkpoints"
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent                  # kaggle/
CODE_DIR = ROOT / "code_dataset"                        # the modules the notebook imports
CKPT_DIR = ROOT.parent / "src" / "checkpoints"          # where src/train.py writes fold*.pt
STAGE = ROOT / "kaggle_upload"
PY = sys.executable
# the 13 checkpoints of the final submission: runs A, B and C of scripts/train_all.sh
FINAL_RUNS = "d43b_mix3u48,d44_ep48,d43b_fdng48_s0,d43b_fdng48_s1,d43b_fdng48_s2"

# Only what the notebook actually imports, plus what those import in turn. Shipping the
# whole repo would drag in the training and audit scripts, which pull pandas/sklearn paths
# the notebook never needs and would fail on Kaggle for no reason.
CODE_FILES = [
    "build_cache.py",        # cache contract: SLOTS, N_SLICE, CROP_MM, IMG, load_slot
    "build_slice_order.py",  # geometric slice ordering
    "build_series_index.py", # sequence recovery + laterality
    "inference.py",          # build_index, preprocess_study
    "model.py",              # KneeNet
    "mil_perdiag.py",        # model.py imports it at MODULE SCOPE (model.py:48).
                             # Omitting it ships a model.py that cannot import, and
                             # the notebook dies before it loads a single weight.
    "validation.py",         # check_submission
    # The f2 family. The notebook imports these only when an f2 checkpoint is attached.
    "build_cache_f2.py",     # select_slots, assemble_study: the f2 cache contract
    "build_series_index_v2.py",  # derive: the fluid_v2 / fatsat_v2 flags f2's ladder reads
    "inference_f2.py",       # build_index_f2, preprocess_study_f2
    "model_f2.py",           # eval_groups_f2, SLOT_PRIOR_F2 (model.knee_net_from_args)
]

EXTERNAL_IMPORTS = set()

# Description of the weights dataset page: licence, the OAI acknowledgement and a pointer to the
# repository's THIRD_PARTY.md (the weights page must carry the OAI notice, like the public
# weights it builds on).
WEIGHTS_DESCRIPTION = (
    "13 ConvNeXt-small 2.5D MIL checkpoints for the RSNA Knee Abnormality Detection competition "
    "(final submission). Licence: CC BY-NC 4.0. The models were trained on the "
    "competition data with auxiliary supervision from the Osteoarthritis Initiative (OAI); use "
    "of the weights grants no right to the underlying data. Third-party credits and licences: "
    "THIRD_PARTY.md in the solution repository.\n\n"
    "Data and/or research tools used in the preparation of this manuscript were obtained and "
    "analyzed from the controlled access datasets distributed from the Osteoarthritis Initiative "
    "(OAI), a data repository housed within the NIMH Data Archive (NDA). OAI is a collaborative "
    "informatics system created by the National Institute of Mental Health and the National "
    "Institute of Arthritis, Musculoskeletal and Skin Diseases (NIAMS) to provide a worldwide "
    "resource to quicken the pace of biomarker identification, scientific investigation and OA "
    "drug development. Dataset identifier(s): 10.15154/0hcg-f676.")


def sh(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd))
    # utf-8 + replace: kaggle CLI progress bars are not cp1252, and a decode failure in the
    # reader thread leaves stdout None, which crashed upload() on 2026-10-06.
    return subprocess.run(cmd, check=False, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", **kw)


def stage_code():
    d = STAGE / "code"
    d.mkdir(parents=True, exist_ok=True)
    missing = [f for f in CODE_FILES if not (CODE_DIR / f).is_file()]
    if missing:
        sys.exit(f"missing source files: {missing}")
    for f in CODE_FILES:
        shutil.copy2(CODE_DIR / f, d / f)
    (d / "dataset-metadata.json").write_text(json.dumps({
        "title": "RSNA Knee Code", "id": "PLACEHOLDER/rsna-knee-code",
        "licenses": [{"name": "CC-BY-NC-4.0"}]}, indent=2), encoding="utf-8")
    total = sum((d / f).stat().st_size for f in CODE_FILES)
    print(f"  staged {len(CODE_FILES)} modules, {total / 1024:.0f} KB -> {d}")
    return d


def stage_weights(runs=None, ema_weights=None, folds=None):
    """Stage checkpoints for the submission.

    `folds` (optional) further restricts the selection to fold{N} for N in folds, e.g. a
    2-fold pilot read; every requested run must then have every requested fold on disk.

    `runs` selects which TRAINING RUNS to ship, by the tag in fold{N}_{tag}.pt. Shipping
    everything in checkpoints/ is wrong twice over: the directory accumulates superseded runs
    (convnext_tiny at gold 0.8330, the 4-slot experiment, convnext_small before augmentation),
    and kaggle_notebook.py treats EVERY .pt it finds as an ensemble member. So an unfiltered
    upload both dilutes the ensemble with weak members and multiplies inference time by the
    number of obsolete runs sitting on disk -- against a 9 h notebook budget.

    Ship the members that were actually measured together, and nothing else.
    """
    src = CKPT_DIR
    if not src.is_dir() or not any(src.glob("*.pt")):
        sys.exit("no checkpoints/*.pt found")
    # Stage into a directory NAMED FOR THE SELECTION. Reusing one "weights" directory is a
    # trap: kaggle_upload/weights still held convnext_tiny checkpoints from an earlier upload,
    # and since the notebook treats every .pt it finds as an ensemble member, those stale files
    # would have shipped as extra members of an ensemble they were never measured in. Nothing
    # is deleted here -- a fresh directory per selection makes leftovers impossible instead.
    d = STAGE / ("weights_" + ("-".join(sorted(runs)) if runs else "all")
                 + ("_f" + "-".join(sorted(folds)) if folds else ""))
    d.mkdir(parents=True, exist_ok=True)
    stale = [p for p in d.glob("*.pt")]
    if stale:
        print(f"  note: {len(stale)} file(s) already in {d.name}; they match this selection")
    n = 0
    chosen = []
    for p in sorted(src.glob("*.pt")):
        tag = p.stem.split("_", 1)[1] if "_" in p.stem else p.stem
        if runs and tag not in runs:
            continue
        if folds and p.stem.split("_", 1)[0] not in {f"fold{f}" for f in folds}:
            continue
        chosen.append(p)
    if (runs or folds) and not chosen:
        sys.exit(f"no checkpoints matched runs={runs} folds={folds}")
    if runs and folds:
        missing = [f"fold{f}_{r}.pt" for r in sorted(runs) for f in sorted(folds)
                   if not (src / f"fold{f}_{r}.pt").is_file()]
        if missing:
            sys.exit(f"requested fold(s) not on disk: {missing}")
    # A --ema checkpoint holds two weight sets: "model" (EMA, the --out OOF) and "model_raw"
    # (oof_parts/<stem>_RAW_fold*.csv). The notebook loads "model" only, so copying one as-is
    # ships EMA even when the blend was measured on RAW. Refuse before staging anything
    # unless the caller names the set, then ship that set alone under "model".
    import torch
    two = [p for p in chosen
           if "model_raw" in torch.load(p, map_location="cpu", weights_only=False)]
    if two and ema_weights not in ("ema", "raw"):
        sys.exit(f"{len(two)} checkpoint(s) were trained with --ema ({two[0].name}, ...): pass "
                 f"--ema-weights ema (the --out OOF) or raw (the _RAW_ OOF parts) -- the set "
                 f"whose OOF the blend was measured on")
    for p in chosen:
        if p in two:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            torch.save({"model": ck["model" if ema_weights == "ema" else "model_raw"],
                        "args": ck["args"], "fold": ck["fold"]}, d / p.name)
        else:
            shutil.copy2(p, d / p.name)
        n += (d / p.name).stat().st_size
    (d / "dataset-metadata.json").write_text(json.dumps({
        "title": "RSNA Knee Weights", "id": "PLACEHOLDER/rsna-knee-weights",
        "licenses": [{"name": "CC-BY-NC-4.0"}],
        "description": WEIGHTS_DESCRIPTION}, indent=2), encoding="utf-8")
    files = sorted(p.name for p in d.glob("*.pt"))
    print(f"  staged {len(files)} checkpoints, {n / 1e9:.2f} GB -> {d}")
    for f in files:
        print(f"    {f}")
    return d



def verify_loadable(d):
    """Load every staged checkpoint the way the notebook will, BEFORE uploading it.

    kaggle_notebook.py rebuilds KneeNet from ck["args"] and calls load_state_dict with the
    default strict=True. Any flag that adds or renames a parameter -- a per-target adapter's
    head.adapter.*, a sharpened-attention run's mil.gain -- produces a checkpoint the notebook
    cannot load. The failure happens on Kaggle, after the upload, and the notebook scores
    NOTHING. That is a whole submission and several hours for a mismatch detectable here in
    seconds, so it is checked here instead.
    """
    import torch
    sys.path.insert(0, str(CODE_DIR))
    from model import knee_net_from_args
    bad = []
    for p in sorted(d.glob("*.pt")):
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            a = dict(ck.get("args", {}) or {})
            a.setdefault("backbone", "convnext_tiny")
            if (a.get("cache_format") or "v1") == "f2" and not a.get("f2_budget"):
                raise ValueError("f2 checkpoint without f2_budget: the notebook skips it")
            net = knee_net_from_args(a)               # the notebook's own constructor
            net.load_state_dict(ck["model"])          # strict, exactly as the notebook does
        except Exception as e:
            bad.append((p.name, f"{type(e).__name__}: {str(e)[:160]}"))
    if bad:
        print(f"\n  REFUSING TO UPLOAD -- {len(bad)} checkpoint(s) will not load the way the "
              f"notebook loads them:")
        for n, m in bad:
            print(f"    {n}\n        {m}")
        print("  Fix kaggle_notebook.py / inference.py to construct the model the way this run "
              "trained it, then re-run.")
        sys.exit(1)
    print(f"  verified {len(list(d.glob('*.pt')))} checkpoints load under strict=True")


def verify_imports(d):
    """Every module a staged file imports at module scope must also be staged.

    CODE_FILES is hand-maintained, so it silently rots the moment someone adds an import.
    model.py gained `from mil_perdiag import ...` at line 48 and the list was not updated; the
    next code upload would have shipped a model.py that cannot import, and the notebook dies
    before loading a single weight -- a whole submission for a one-line omission.

    So the list is checked rather than trusted: parse the staged sources, collect every
    module-scope import, and confirm each one is either stdlib, an installed third-party
    package, or a file we also staged.
    """
    import ast as _ast, sys as _sys, importlib.util as _iu
    staged = {f.stem for f in d.glob("*.py")}
    missing = []
    for f in sorted(d.glob("*.py")):
        try:
            tree = _ast.parse(f.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        names = set()
        for n in _ast.walk(tree):
            if isinstance(n, _ast.Import):
                names |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, _ast.ImportFrom) and n.module and n.level == 0:
                names.add(n.module.split(".")[0])
        for nm in sorted(names):
            if nm in staged or nm in _sys.stdlib_module_names or nm in EXTERNAL_IMPORTS:
                continue
            if (CODE_DIR / f"{nm}.py").is_file():     # ours, and NOT staged
                missing.append((f.name, nm))
            elif _iu.find_spec(nm) is None:
                missing.append((f.name, nm))
    if missing:
        print(f"\n  REFUSING TO UPLOAD -- staged code imports modules that are not in the package:")
        for src, nm in missing:
            print(f"    {src} imports {nm!r}, which is not staged and not installed")
        print("  Add them to CODE_FILES.")
        sys.exit(1)
    print(f"  verified {len(staged)} staged modules; every module-scope import resolves")

def username():
    r = sh([PY, "-m", "kaggle", "config", "view"])
    for line in (r.stdout or "").splitlines():
        if "username" in line.lower():
            return line.split(":")[-1].strip()
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if cfg.is_file():
        return json.loads(cfg.read_text()).get("username")
    return None


def remote_files(ref):
    """File names in the latest version of a dataset, or None if it does not exist."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    names, tok = set(), None
    try:
        while True:
            r = api.dataset_list_files(ref, page_token=tok)
            names |= {f.name for f in r.files}
            tok = getattr(r, "next_page_token", None)
            if not tok:
                return names
    except Exception as e:
        # Kaggle answers 403 (or 404) for a dataset that does not exist: that is "no remote
        # files". Anything else (network, 5xx) re-raises, so the keep_remote guard fails
        # CLOSED -- a transient API error must not wave a file-dropping upload through.
        if getattr(getattr(e, "response", None), "status_code", None) in (403, 404):
            return None
        raise


def upload(d: Path, slug: str, message: str, dry: bool, keep_remote=False):
    """keep_remote: refuse to version a dataset whose latest version holds files the staged
    directory does not. On 2026-10-06 a --vlm-base version of rsna-knee-vlm-base, which a
    parallel job had created holding the base AND its wheels, silently dropped the wheels
    (restored by hand the same hour). The weights upload legitimately drops files, so this
    is opt-in."""
    user = username()
    if not user:
        sys.exit("could not determine Kaggle username from ~/.kaggle/kaggle.json")
    meta = d / "dataset-metadata.json"
    m = json.loads(meta.read_text(encoding="utf-8"))
    m["id"] = f"{user}/{slug}"
    meta.write_text(json.dumps(m, indent=2), encoding="utf-8")
    if keep_remote:
        have = remote_files(m["id"])
        lost = sorted((have or set()) - {p.name for p in d.iterdir()})
        if lost:
            sys.exit(f"REFUSING to version {m['id']}: its latest version holds {len(lost)} "
                     f"file(s) this upload would drop: {lost[:8]}. Copy them into {d} first.")
    if dry:
        print(f"  [dry-run] would upload {d} as {m['id']}")
        return
    # `create` the first time, `version` afterwards; try version first and fall back.
    r = sh([PY, "-m", "kaggle", "datasets", "version", "-p", str(d), "-m", message, "-r", "zip"])
    if r.returncode != 0 and "not found" in (r.stdout + r.stderr).lower():
        r = sh([PY, "-m", "kaggle", "datasets", "create", "-p", str(d), "-r", "zip"])
    print((r.stdout or r.stderr).strip()[-500:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", action="store_true")
    ap.add_argument("--weights", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--message", default="update")
    ap.add_argument("--runs", default=FINAL_RUNS,
                    help="comma-separated run tags to ship (fold{N}_{tag}.pt). Default: the 13 "
                         "released checkpoints, " + FINAL_RUNS + ". Pass '' to ship every "
                         "checkpoint on disk, which is almost never what you want -- see "
                         "stage_weights.")
    ap.add_argument("--ema-weights", choices=["ema", "raw"], default=None,
                    help="REQUIRED when any selected checkpoint was trained with --ema: which "
                         "weight set to ship under 'model' -- the one whose OOF was measured.")
    ap.add_argument("--folds", default=None,
                    help="comma-separated fold numbers to ship, e.g. 0,1. Default ships every "
                         "fold of the selected runs.")
    a = ap.parse_args()
    if not (a.code or a.weights or a.all):
        a.all = True

    if a.code or a.all:
        print("CODE")
        _c = stage_code()
        verify_imports(_c)
        upload(_c, "rsna-knee-code", a.message, a.dry_run)
    if a.weights or a.all:
        print("WEIGHTS")
        runs = set(a.runs.split(",")) if a.runs else None
        folds = set(a.folds.split(",")) if a.folds else None
        _d = stage_weights(runs, a.ema_weights, folds)
        verify_loadable(_d)
        upload(_d, "rsna-knee-weights", a.message, a.dry_run)

    print("\nNotebook setup: point kaggle/kernel-metadata.json at your datasets, then push")
    print("kaggle/rsna-knee-submission.ipynb (see README.md, step 7).")


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()
