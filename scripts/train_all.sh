#!/usr/bin/env bash
# train_all.sh -- the three training runs behind the 13 released checkpoints.
#
# Each command below is the exact argument set stored in the shipped checkpoints' "args"
# (only the three data paths are placeholders). Run from the repository root; train.py reads
# folds.csv, train.csv and the label CSV from src/ and writes src/checkpoints/fold{k}_<tag>.pt.
#
#   CACHE       competition cache from src/build_cache.py   (6 x 32 x 336 x 336 uint8 per study)
#   OAI_CACHE   OAI cache from src/build_oai_full_cache.py  (7,992 knee volumes)
#   OAI_LADDER  oai_full_labels_ladder.csv from src/build_reasonknee_labels.py
#
# Time on one 96 GB RTX PRO 6000: A ~20.5 h, B ~20.5 h, C ~5.1 h per seed. --mem-frac 0.35 only
# caps GPU memory (35% of 96 GB); raise or drop it on a smaller card. --resume is inert on a fresh
# start and continues a killed run mid-fold. Runs are not bit-identical on repeat (cuDNN
# nondeterminism is left on).
set -euo pipefail

: "${CACHE:?set CACHE to the competition cache folder}"
: "${OAI_CACHE:?set OAI_CACHE to the OAI cache folder}"
: "${OAI_LADDER:?set OAI_LADDER to oai_full_labels_ladder.csv}"
PY="${PYTHON:-python}"

# The commands below run from src/ (train.py's folder). Make the three data paths absolute first,
# so that paths given relative to the directory you start from (README step 5) still resolve.
CACHE="$(cd "$CACHE" && pwd)" || { echo "CACHE folder not found" >&2; exit 1; }
OAI_CACHE="$(cd "$OAI_CACHE" && pwd)" || { echo "OAI_CACHE folder not found" >&2; exit 1; }
_ladder_dir="$(cd "$(dirname "$OAI_LADDER")" && pwd)" || { echo "OAI_LADDER folder not found" >&2; exit 1; }
OAI_LADDER="$_ladder_dir/$(basename "$OAI_LADDER")"
[ -f "$OAI_LADDER" ] || { echo "OAI_LADDER file not found: $OAI_LADDER" >&2; exit 1; }

COMMON=(--backbone convnext_small --bs 8 --cache "$CACHE" --epochs 48 --n-group 10
        --grad-ckpt --aug --slot-drop 0.25 --workers 5
        --oai-cache "$OAI_CACHE" --oai-labels "$OAI_LADDER"
        --oai-frac 0.5 --oai-weight 1.0 --lr-backbone 5e-5 --lr-head 1e-3 --wd 0.01
        --dropout 0.2 --gold-weight 8 --mem-frac 0.35)

cd "$(dirname "$0")/../src"

# A: fold{0..4}_d43b_mix3u48.pt -- mix3 labels, OAI weak targets, uniform per-target weights
"$PY" train.py --all-folds --seed 0 "${COMMON[@]}" --labels report_labels_mix3.csv \
    --oai-targets weak --target-weights uniform --out oof_d43b_mix3u48.csv --resume

# B: fold{0..4}_d44_ep48.pt -- consensus labels, OAI safe targets
"$PY" train.py --all-folds --seed 0 "${COMMON[@]}" --labels report_labels_consensus.csv \
    --oai-targets safe --out oof_d44_ep48.csv --resume

# C: fold5_d43b_fdng48_s{0,1,2}.pt -- all data, gold studies excluded, three seeds
for s in 0 1 2; do
    "$PY" train.py --fold 5 --exclude-gold --seed "$s" "${COMMON[@]}" \
        --labels report_labels_consensus.csv --oai-targets weak \
        --out "oof_d43b_fdng48_s$s.csv" --resume
done
