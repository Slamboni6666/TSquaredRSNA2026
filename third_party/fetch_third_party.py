"""
fetch_third_party.py -- download the four public Kaggle notebook cells the blend runs, and
verify each one against the sha256 it had when the C02R7-MIX3 notebook was built.

The cells are NOT redistributed as separate files in this repository: no licence for them could
be confirmed (see THIRD_PARTY.md). They are fetched from their authors' public Kaggle notebooks:

  nartaa/rsna-knee-0949-anatomical-mirror            code cells 3 and 4  (Raptor-384 inference)
  heliosli/0-957-latest-dataset-score-in-one-click   code cells 4 and 5  (R7 stage)

Note: kaggle/rsna-knee-submission.ipynb (the notebook that ran) already embeds these four cells,
base64-encoded and sha256-checked before use, in its first cell.

Output (git-ignored):
  third_party/cache/nartaa/cell3.py, cell4.py
  third_party/cache/heliosli/0957_cell4.txt, 0957_cell5.txt

Kaggle only lets a kernel's owner pull an older version, so the pinned version is tried first and
the current version is used if that is refused. Either way the bytes must match the sha256 below,
or nothing is written for that cell and the script exits non-zero.

Needs the `kaggle` package and your own Kaggle API credentials (see Kaggle's documentation).

usage: python third_party/fetch_third_party.py [--out third_party/cache]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

# kernel ref, version checked on 2026-10-22, {cell index in the notebook's cell list: (out name, sha256)}
SOURCES = {
    "nartaa": ("nartaa/rsna-knee-0949-anatomical-mirror", 2, {
        3: ("cell3.py", "fd96541cbe2e5b37494636dd13b79871a7a6ed9ad94fe528dc6f7a253d4be8b8"),
        4: ("cell4.py", "bcd864cbf349882e876959b49cb348cbb89b4c62fc204d4e4261049c5e3a79c5"),
    }),
    "heliosli": ("heliosli/0-957-latest-dataset-score-in-one-click", 1, {
        4: ("0957_cell4.txt", "cb168cfe82f5a04114894f6c07bbd6daf0373ad2eb0dfcee34df3ecf625986af"),
        5: ("0957_cell5.txt", "68f70dd31cc3544cb5734313d06538355a34b122ff430a32820756175c9c9dd9"),
    }),
}


def pull(api, ref: str, version: int, dest: Path) -> Path:
    """Pull the notebook (pinned version first, current version if refused); return the .ipynb."""
    for r in (f"{ref}/{version}", ref):
        try:
            api.kernels_pull(r, path=str(dest))
            break
        except Exception as e:  # e.g. 403 for a non-owner's old version
            print(f"  pull {r}: {type(e).__name__}: {str(e)[:120]}")
    else:
        raise SystemExit(f"could not pull {ref}")
    nbs = list(dest.glob("*.ipynb"))
    if len(nbs) != 1:
        raise SystemExit(f"{ref}: expected one .ipynb, found {len(nbs)}")
    return nbs[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--out", default=str(HERE / "cache"), help="output folder")
    a = ap.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    bad = 0
    for author, (ref, version, cells) in SOURCES.items():
        with tempfile.TemporaryDirectory() as td:
            nb = json.loads(pull(api, ref, version, Path(td)).read_text(encoding="utf-8"))
        out = Path(a.out) / author
        out.mkdir(parents=True, exist_ok=True)
        for idx, (name, sha) in cells.items():
            src = nb["cells"][idx]["source"]
            b = ("".join(src) if isinstance(src, list) else src).encode("utf-8")
            h = hashlib.sha256(b).hexdigest()
            if h != sha:
                print(f"  MISMATCH {ref} cell {idx}: sha256 {h} != {sha} (the notebook changed)")
                bad += 1
                continue
            (out / name).write_bytes(b)
            print(f"  ok {ref} cell {idx} -> {out / name} ({len(b)} B, sha256 {h[:16]}...)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
