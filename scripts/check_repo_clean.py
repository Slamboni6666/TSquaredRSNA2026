"""
check_repo_clean.py -- refuse to publish competition data, OAI data, weights, secrets or local
identifiers. Exit status 1 on any hit, 0 when clean.

WHY THIS FILE EXISTS
A log file once went to a public repository with a verbatim radiology report and real study
UIDs in it. The .gitignore was right about every file anyone thought to list; it was the file
nobody thought of that got through. So the rule is enforced by a program, not by vigilance:
installed as a pre-push hook it blocks the push; run by hand it tells you what would go out.

WHAT IT LOOKS FOR
  file names    weights (*.pt, *.pth, *.ckpt, *.safetensors), arrays and tables (*.npy, *.npz,
                *.parquet, *.csv, *.jsonl, *.xlsx, *.pkl), images (*.dcm, *.nii*), archives and
                wheels, logs, credentials (the Kaggle API token file, *.key, *.pem, .env), and
                working notes (TRACKER.md, CLAUDE.md, PLAN.md).
  content       DICOM study/series UIDs (the competition's UID root and any long 1.2.826 /
                1.2.840 UID), OAI participant IDs (7 digits starting with 9), Kaggle submission
                refs (8 digits starting with 5), absolute local paths (drive-letter paths,
                /c/Users, /home/<user>, /Users/<user>), e-mail addresses, API tokens and keys,
                radiology-report prose (section headers followed by text), and the team's
                identifiers (below).
  identifiers   names, user names and handles, the lab name, the Kaggle kernel id and the
                kernel versions of the picks, submission refs and NDA package IDs of the team
                that wrote this code. They are NOT spelled out in this file (publishing the list would publish the identities): every
                alphanumeric word of the scanned text is lower-cased, hashed with a fixed salt,
                and compared with IDENTITY_DIGESTS. A digest can confirm a guess but does not
                reveal the word. `--digest WORD` prints the digest of a new word to add.
  base64        every base64 string literal of 200+ characters (the Kaggle notebook embeds the
                third-party cells this way) is decoded and scanned for the same content.

WHAT IT DELIBERATELY DOES NOT FLAG
  Clinical vocabulary in extractor patterns (only report section headers followed by prose),
  standard DICOM transfer-syntax UIDs (1.2.840.10008.*, short), the synthetic participant IDs
  of build_oai_full_cache.py's phantom self-test (SYNTHETIC_OAI_IDS below), 7-digit runs
  inside hexadecimal hashes, and heliosli's development paths inside his published notebook
  cells (THIRD_PARTY_PATHS; embedded byte-identically in the Kaggle notebook).

The patterns are written so that this file does not match itself (character classes and
repetition counts break up the literal tokens), so it scans itself like any other file.

Usage:
    python scripts/check_repo_clean.py              # git repo: tracked + untracked-not-ignored
                                                    # files; otherwise every file in the tree
    python scripts/check_repo_clean.py --all-files  # every file in the tree (except .git/)
    python scripts/check_repo_clean.py --staged     # only what is staged
    python scripts/check_repo_clean.py --install    # install as .git/hooks/pre-push
    python scripts/check_repo_clean.py --digest WORD  # digest to add to IDENTITY_DIGESTS
"""

from __future__ import annotations

import argparse
import base64
import binascii
import fnmatch
import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BANNED_NAMES = [
    ("*.pt", "model weights"), ("*.pth", "model weights"), ("*.ckpt", "model weights"),
    ("*.safetensors", "model weights"), ("*.bin", "binary blob / weights"),
    ("*.npy", "cached array"), ("*.npz", "cached array"), ("*.parquet", "derived index"),
    ("*.csv", "table (labels, folds, metadata or competition data)"),
    ("*.jsonl", "label dump"), ("*.xlsx", "spreadsheet"), ("*.pkl", "pickle"),
    ("*.dcm", "DICOM image"), ("*.nii", "image volume"), ("*.nii.gz", "image volume"),
    ("*.zip", "archive"), ("*.7z", "archive"), ("*.tar*", "archive"), ("*.whl", "vendored wheel"),
    ("*.log", "log file"),
    ("kaggle" ".json", "CREDENTIALS"), ("*.key", "CREDENTIALS"), ("*.pem", "CREDENTIALS"),
    (".env", "CREDENTIALS"), ("TRACKER.md", "working notes"), ("CLAUDE.md", "working notes"),
    ("PLAN.md", "working notes"),
]

# Synthetic participant IDs used by build_oai_full_cache.py's phantom self-test (not OAI IDs;
# checked absent from the OAI manifest the pipeline builds).
SYNTHETIC_OAI_IDS = {str(n) for n in (9_999_001, 9_999_002, 9_999_003)}

# heliosli's own development paths inside his published notebook cells (third-party code that
# the Kaggle notebook embeds byte-identically as base64; not ours, not editable).
THIRD_PARTY_PATHS = ("F" ":/rsnaknee", "D" ":/rsnaknee")

# Salted sha256 (first 16 hex digits) of each lower-cased identifier word; see the docstring.
IDENTITY_SALT = b"rsna-knee-solution/check_repo_clean/v1:"
IDENTITY_DIGESTS = frozenset({
    "17c9b1cbfd937508", "1a8e6ee66e8010d7", "1bafa0551f7f8252", "21a4fd18b951f9fd",
    "2aeabde2459a17fb", "2cf56c582982360b", "2d4d01fa7f0b4d8e", "2e94bef288b9294a",
    "2f0d1e988dcc55b7", "3b03d930f388bc6e", "3eb2d0da95f2db68", "425deb7cb275e12a",
    "4399d375514e33d6", "4792a3ba0fae6f1d", "4ad3e99fabeb2b9f", "4af09f42bfbcbcaa",
    "4b2233ff7609d35c", "4d817311ddf594f3", "4ed534fe6a903202", "52be521adec4edc7",
    "52e8e8d2678ffc6f", "545fb486d9f0ddeb", "551d7bf9b088834b", "584df2ed5dc456bd",
    "5f889162c97a0c5d", "616c54c31d092b6c", "640c3e03fadf0502", "66f441b4bbf0580e",
    "6800b049769694da", "6fffeb3ee46e5537", "8163f59e8e304f22", "82b65308f5038fc5",
    "83895fe05df5abec", "8b627437a47d9b11", "9180cf022973a0b9", "933b2598874c1723",
    "98061c165ee58e6f", "a2253713c6a27286", "a59c135e79977b3a", "a8b270c86ec68334",
    "acaa5799a5583088", "b088618f6d4d3cd9", "b370edc63689632a", "b7e59fac65bdc4ae",
    "be8cb7954e635d7d", "c02ae09e7eeef142", "d4e1b1a35d534263", "d7144083e945b2b4",
    "dc4555b0e5a33d8e", "e007789db00da4a1", "e985f1c5fc4e7971", "ee13aaa3974865f0",
    "effaaae1ee2c8c04", "f229894c6489bf47", "f40f55e6c0359542", "f7fbbdaa18448779",
    "fc876195c88f2548", "fde346670848ca48", "0be7cc05af743f0c", "89d7d19d20166142",
})
WORD = re.compile(r"[A-Za-z0-9]+")


def digest(word: str) -> str:
    return hashlib.sha256(IDENTITY_SALT + word.lower().encode("utf-8")).hexdigest()[:16]


def identity_hits(txt: str) -> list[str]:
    seen = {}
    for m in WORD.finditer(txt):
        w = m.group(0)
        if 3 <= len(w) <= 40 and w.lower() not in seen:
            seen[w.lower()] = w
    return [w for lw, w in seen.items() if digest(lw) in IDENTITY_DIGESTS]


CONTENT = [
    ("competition DICOM UID", re.compile(r"1\.2\.826\.0\.1\.36800(?:43)")),
    ("DICOM UID", re.compile(r"1\.2\.(?:826|840)\.[0-9.]{25,}")),
    ("OAI participant ID", re.compile(r"(?<![0-9A-Za-z.])9[0-9]{6}(?![0-9A-Za-z.])")),
    ("Kaggle submission ref", re.compile(r"(?<![0-9A-Za-z.,])5[0-9]{7}(?![0-9A-Za-z.,])")),
    ("absolute local path", re.compile(
        # drive-letter path; "X:\n" (a one-letter name before an escaped newline) is not one
        r"(?<![A-Za-z0-9])[A-Z]:(?:/|\\{1,2}(?![ntr0'\"]))(?=[A-Za-z_~$%])"
        r"|/[cdg]/Users/|/home/[a-z_][a-z0-9_-]*/|(?<![A-Za-z])/Users/[A-Za-z]")),
    ("e-mail address", re.compile(
        r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9-]{1,63}"
        r"(?:\.[A-Za-z0-9-]{1,63}){0,3}\.(?:com|org|net|edu|io|ai|gov|uk|de|cn)(?![A-Za-z])")),
    ("credential", re.compile(
        r"hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
        r"|AKIA[0-9A-Z]{16}|\"key\"\s*:\s*\"[A-Za-z0-9]{16,}|KAGGLE_KEY\s*=\s*['\"]?[A-Za-z0-9]{16,}")),
    ("radiology report text", re.compile(
        r"T[eé]cnica:\s*RMN|Resultados:\s*\w|Impresi[oó]n:\s*\w|FINDINGS:\s*\w|IMPRESSION:\s*\w")),
]

B64 = re.compile(r"[\"']([A-Za-z0-9+/]{200,}={0,2})\\?[\"']")   # also JSON-escaped quotes
SKIP_DIRS = {".git", "__pycache__"}
MAX_TEXT = 50_000_000


def files_to_check(mode: str) -> list[str]:
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)

    in_git = mode != "all-files" and git("rev-parse", "--is-inside-work-tree").returncode == 0
    if in_git and mode == "staged":
        out = git("diff", "--cached", "--name-only", "--diff-filter=ACMR").stdout
        return [f for f in out.splitlines() if f.strip()]
    if in_git:
        out = git("ls-files", "--cached", "--others", "--exclude-standard").stdout
        return [f for f in out.splitlines() if f.strip()]
    return [p.relative_to(ROOT).as_posix() for p in sorted(ROOT.rglob("*"))
            if p.is_file() and not (set(p.relative_to(ROOT).parts) & SKIP_DIRS)]


def scan_text(txt: str):
    hits = []
    for what, pat in CONTENT:
        found = [m.group(0) for m in pat.finditer(txt)]
        if what == "OAI participant ID":
            found = [x for x in found if x not in SYNTHETIC_OAI_IDS]
        if what == "absolute local path":
            found = [m.group(0) for m in pat.finditer(txt)
                     if not txt.startswith(THIRD_PARTY_PATHS, m.start())]
        if found:
            hits.append((what, len(found), found[0]))
    ids = identity_hits(txt)
    if ids:
        # the hit is reported by position, not echoed, so the output can be pasted safely
        first = txt.find(ids[0])
        hits.append(("team identifier", len(ids), f"word at character {first}"))
    return hits


def check(files):
    problems = []
    for f in files:
        name = Path(f).name
        for pat, why in BANNED_NAMES:
            if fnmatch.fnmatch(name, pat):
                problems.append((f, f"file name matches {pat} -- {why}"))
        p = ROOT / f
        if not p.is_file() or p.stat().st_size > MAX_TEXT:
            continue
        txt = p.read_bytes().decode("utf-8", errors="replace")
        for what, n, first in scan_text(txt):
            problems.append((f, f"{n} x {what} (first: {first[:60]!r})"))
        for m in B64.finditer(txt):
            try:
                dec = base64.b64decode(m.group(1), validate=True).decode("utf-8", errors="replace")
            except (binascii.Error, ValueError):
                continue
            for what, n, first in scan_text(dec):
                problems.append((f, f"{n} x {what} inside a base64 literal at offset {m.start()} "
                                    f"(first: {first[:60]!r})"))
    return problems


def install_hook():
    hook = ROOT / ".git" / "hooks" / "pre-push"
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text("#!/bin/sh\nexec python scripts/check_repo_clean.py\n", encoding="utf-8")
    hook.chmod(0o755)
    print(f"installed {hook}")


def main():
    ap = argparse.ArgumentParser(description="scan the repository for data that must not be published")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--all-files", action="store_true", help="every file in the tree, ignoring git")
    g.add_argument("--staged", action="store_true", help="only staged files")
    g.add_argument("--install", action="store_true", help="install as a pre-push hook")
    g.add_argument("--digest", metavar="WORD", help="print the identifier digest of WORD and exit")
    a = ap.parse_args()
    if a.digest:
        print(digest(a.digest), "(listed)" if digest(a.digest) in IDENTITY_DIGESTS else "")
        return 0
    if a.install:
        install_hook()
        return 0
    mode = "all-files" if a.all_files else "staged" if a.staged else "default"
    files = files_to_check(mode)
    problems = check(files)
    if not problems:
        print(f"  clean: {len(files)} file(s) checked; no data, weights, secrets or local identifiers")
        return 0
    print(f"  REFUSING: {len(problems)} problem(s) in {len(files)} file(s) checked\n")
    for f, why in problems:
        print(f"    {f}\n        {why}")
    return 1


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    raise SystemExit(main())
