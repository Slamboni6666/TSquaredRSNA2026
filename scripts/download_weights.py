"""
download_weights.py -- fetch the 13 checkpoints from this repository's GitHub Release and
verify each against release/weights.sha256.

The weights are not in git (13 files, about 203 MB each, 2.64 GB in total). They are attached to
a GitHub Release of this repository, licence CC BY-NC 4.0 (see LICENSE and release/README.md).

usage: python scripts/download_weights.py [--out src/checkpoints] [--only fold0_d43b_mix3u48.pt ...]

A file already present with the right sha256 is kept; a file with the wrong sha256 is never
written under its final name (downloads go to <name>.part and are renamed only after the check).
"""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------------------------
# The GitHub repository is read from this clone's `origin` remote (or --repo owner/name), so no
# account name is written into the code. TAG is the release that carries the 13 .pt files.
TAG = "v1.0"
# ---------------------------------------------------------------------------------------------

URL = "https://github.com/{owner}/{repo}/releases/download/{tag}/{file}"
ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "release" / "weights.sha256"
CHUNK = 1 << 20


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def manifest() -> dict[str, str]:
    out = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            digest, name = line.split(maxsplit=1)
            out[name.lstrip("*")] = digest.lower()
    return out


def origin_repo() -> tuple[str, str] | None:
    """owner, repo of this clone's GitHub `origin` remote (https or ssh form), or None."""
    try:
        url = subprocess.run(["git", "-C", str(ROOT), "remote", "get-url", "origin"],
                             capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    m = re.search(r"github\.com[:/]+([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    return (m.group(1), m.group(2)) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description="download and verify the 13 checkpoints")
    ap.add_argument("--out", default=str(ROOT / "src" / "checkpoints"),
                    help="destination folder (default: src/checkpoints, where train.py writes)")
    ap.add_argument("--only", nargs="*", default=None, help="download only these file names")
    ap.add_argument("--repo", default=None,
                    help="owner/name of the GitHub repository (default: this clone's origin remote)")
    ap.add_argument("--tag", default=TAG, help=f"release tag (default {TAG})")
    a = ap.parse_args()
    if a.repo:
        OWNER, REPO = a.repo.split("/", 1)
    else:
        found = origin_repo()
        if not found:
            sys.exit("download_weights.py: no GitHub origin remote found; pass --repo owner/name")
        OWNER, REPO = found
    files = manifest()
    if a.only:
        unknown = sorted(set(a.only) - set(files))
        if unknown:
            sys.exit(f"not in {MANIFEST.name}: {unknown}")
        files = {k: v for k, v in files.items() if k in a.only}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bad = 0
    for name, digest in files.items():
        dst = out / name
        if dst.is_file() and sha256(dst) == digest:
            print(f"  ok (present) {name}")
            continue
        url = URL.format(owner=OWNER, repo=REPO, tag=a.tag, file=name)
        part = dst.with_name(dst.name + ".part")
        print(f"  get {url}")
        with urllib.request.urlopen(url) as r, open(part, "wb") as fh:
            while True:
                b = r.read(CHUNK)
                if not b:
                    break
                fh.write(b)
        got = sha256(part)
        if got != digest:
            print(f"  SHA256 MISMATCH {name}: {got} != {digest} (left as {part.name})")
            bad += 1
            continue
        part.replace(dst)
        print(f"  ok {name} sha256 {digest[:16]}...")
    print(f"{len(files) - bad}/{len(files)} checkpoint(s) verified in {out}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
