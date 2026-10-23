# RSNA Knee Abnormality Detection: final solution

Code, training recipe and Kaggle inference notebook for our final submission to the
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
competition (12 findings per knee MRI study, macro AUC).

**Final result: 0.960 macro AUC on the public leaderboard.**

The submission blends our own model, 13 ConvNeXt-small checkpoints, with two public model
families (nartaa's Raptor-384 and heliosli's R7), rank-blended per finding. The blend weights
were fixed in advance and never tuned on the leaderboard.

## Method

- **Model: 2.5D ConvNeXt-small, multiple-instance learning.** Each study is resampled into a
  fixed cache of 6 sequence slots x 32 slices x 336 x 336 px (slot recovery from DICOM headers,
  geometric slice ordering, laterality). The network sees 3-slice windows (10 windows per slot
  during training) through an ImageNet-12k ConvNeXt-small backbone (`timm`
  `convnext_small.in12k_ft_in1k`); a gated-attention MIL head (`src/mil_perdiag.py`) pools the
  windows of each slot and the slots are combined into 12 logits. Augmentation: photometric
  (`--aug`) plus random slot drop (0.25).
- **Labels from reports.** Only 58 training studies have image-level gold labels; the other
  4,349 have a radiology report. Each report was read by **Qwen3-32B** (4-bit, severity per
  finding with an explicit "not mentioned" answer) and by a rule-based extractor; the two are
  merged into a consensus table in which unmentioned findings are masked out of the loss.
  A second table (`mix3`) averages, per cell, the sources that assert it: our consensus, the
  public `stevenleehans` v4_blend labels and the public `nartaa` gemini-low labels. Combining
  readers fills most of the cells a single reader leaves unlabelled, which was our largest
  single improvement.
- **OAI auxiliary supervision.** 7,992 sagittal 3D DESS knee volumes from the Osteoarthritis
  Initiative, labelled with central MOAKS readings (via the 3DReasonKnee release) mapped onto
  the same soft-label ladder, are mixed into training (up to half as many OAI rows as
  competition rows per epoch, rotated at participant level). OAI rows are never scored.
- **Training.** 48 epochs, AdamW with a one-cycle schedule (backbone 5e-5, head 1e-3, weight
  decay 0.01), dropout 0.2, gold studies weighted x8, gradient checkpointing; 5-fold CV on
  site-grouped folds (`src/validation.py`) plus all-data runs. The 13 checkpoints come from
  three runs: A (5 folds, `mix3` labels), B (5 folds, consensus labels) and C (all data,
  3 seeds).
- **Blend.** On Kaggle the notebook runs nartaa's public Raptor-384 inference and heliosli's
  public R7 stage (each as an isolated subprocess, sha256-checked) and our 13 checkpoints, then
  ranks each family's scores per finding over the studies it scored and takes the weighted
  mean of ranks: ours 0.40, Raptor 0.30, R7 0.30 (renormalised when a family did not score a
  study).

## Repository layout

```
src/                 data preparation, label building, OAI pipeline, training (train.py, model.py, ...)
tests/               test_full_data.py (synthetic data, CPU: PYTHONPATH=src python tests/test_full_data.py)
scripts/             train_all.sh, download_weights.py, check_repo_clean.py
kaggle/              rsna-knee-submission.ipynb (the submission notebook), kernel-metadata.json,
                     c02r7_cell.py (readable copy of its code cell), package_for_kaggle.py,
                     code_dataset/ (modules the notebook imports), build/ (rebuilds the notebook)
tools/check_blend.py independent recompute of the blend from a notebook run's output
third_party/         fetch_third_party.py (downloads the public notebook cells the blend runs)
release/             weights.sha256 + README for the GitHub Release that carries the weights
HASHES.sha256        sha256 of every pipeline input/output whose bytes we know
```

The training and data-preparation code is the code that produced the released checkpoints. The
only changes are neutral placeholder paths (`data/...`) in place of the original machine's
absolute paths in defaults, help strings and comments, and `build_reasonknee_labels.py
--teacher` now defaults to `report_labels_consensus.csv` (same output). Runs B and C were
trained with slightly earlier versions of the same code; every later change to `train.py`,
`model.py`, `aug.py` and `oai_aux.py` sits behind a flag that is off in their stored `args`, so
the shipped code runs the same code path for all three runs.

## Hardware and time

- Training: one 96 GB NVIDIA GPU, Python 3.11, CUDA 12.8. Runs A and B about 20.5 h each, run C
  about 5.1 h per seed (about 56 h in total). `--mem-frac 0.35` caps memory at about 36 GB; the
  runs fit on smaller cards with that cap raised or removed.
- Report labelling: Qwen3-32B in 4-bit, roughly 20 GB of GPU memory (estimate).
- Inference: Kaggle 2x T4, internet off. A commit run takes about 8 minutes; a full scored
  submission takes about 6 to 8 hours of Kaggle time, within the 9-hour limit.

## Reproduction

All scripts resolve their tabular inputs and outputs relative to `src/` (the folder they live
in); placeholder defaults under `data/` are relative to the directory you run from. Run every
command from the repository root. `pip install -r requirements.txt` first.

### 1. Get the data

- Competition data from Kaggle: unzip the full competition download (`train.csv`, `test.csv`,
  `sample_submission.csv`, the series CSVs, `train_series/` and `test_series/`) into `src/`
  (or symlink it). `audit_data.py` reads all of it. Nothing under `src/` other than `*.py` is
  tracked by git.
- OAI: SAG_3D_DESS DICOM through your own NIH/NDA OAI access (NDA dataset 10.15154/0hcg-f676),
  for the series listed in the manifest written in step 4. `build_oai_full_cache.py` expects
  the NDA package extraction under `--dicom-root`
  (`<dicom-root>/<any folders>/<release>/<participant>/<date>/<series>/*.dcm`); it indexes
  series by (participant, date, series) at any depth.
- 3DReasonKnee labels from
  [huggingface.co/datasets/rajpurkarlab/3DReasonKnee](https://huggingface.co/datasets/rajpurkarlab/3DReasonKnee)
  (CC BY 4.0) into `data/3dreasonknee/data/` (`labels/` and `split_files/all_studies.csv`).
- The two public label tables used by `mix3` (CC0). We used stevenleehans v6 and the nartaa
  files whose bytes are unchanged in v6 to v9. `kaggle datasets download` fetches the latest
  version, so check the sha256 (step 3, `HASHES.sha256`) before running `build_mix3.py`; if it
  fails, download the version above from the dataset page's version selector.
  ```bash
  kaggle datasets download stevenleehans/rsna-knee-llm-report-labels -p data/public_labels/stevenleehans_rsna-knee-llm-report-labels --unzip
  kaggle datasets download nartaa/rsna-knee-hpo-assets -p data/public_labels/nartaa_rsna-knee-hpo-assets --unzip
  ```

### 2. Build indexes and folds

```bash
python src/audit_data.py --full-headers        # -> src/study_metadata.csv
python src/build_report_labels.py              # -> src/report_labels_ours.csv (rule extractor; validation.py's sanity check reads it)
python src/validation.py                       # -> src/folds.csv
python src/check_patient_leak.py               # -> src/patient_map.csv
python src/build_series_index.py --split train # -> src/series_index.parquet, src/study_index.parquet
python src/build_slice_order.py --split train  # -> src/slice_order.parquet
python src/build_cache.py --out data/rsna_knee_cache   # competition cache, 6 x 32 x 336 x 336 uint8 per study
```

### 3. Build the report labels

```bash
python src/llm_label_reports.py --model Qwen/Qwen3-32B --load-4bit   # -> src/llm_labels.jsonl (needs transformers, accelerate, bitsandbytes)
python src/llm_label_reports.py --score-only                         # -> src/report_labels_llm.csv
python src/teacher_consensus.py                                      # -> src/report_labels_consensus.csv
python src/build_mix3.py src src/report_labels_mix3.csv --dl data/public_labels   # -> mix3 (no --steven-syn-v1-silent)
sha256sum -c --ignore-missing HASHES.sha256
```

Every table except the Qwen output rebuilds byte-identically from its inputs. A new Qwen pass
gives close but not byte-identical labels; our label tables are not published.

### 4. Build the OAI labels and cache

```bash
mkdir -p data/oai_cache
python src/build_reasonknee_labels.py \
    --out data/oai_cache/oai_full_labels_ladder.csv \
    --manifest-out data/oai_cache/oai_full_manifest.csv \
    --folds-out data/oai_cache/folds_oai_full.csv
python src/build_oai_full_cache.py --dicom-root <OAI_DICOM_ROOT> \
    --manifest data/oai_cache/oai_full_manifest.csv \
    --labels data/oai_cache/oai_full_labels_ladder.csv --out data/oai_full_cache
```

### 5. Train the 13 checkpoints

```bash
CACHE=data/rsna_knee_cache OAI_CACHE=data/oai_full_cache \
OAI_LADDER=data/oai_cache/oai_full_labels_ladder.csv bash scripts/train_all.sh
```

The three paths may be relative to the directory you start from; the script makes them
absolute before it changes into `src/`.

Runs A, B and C (see `scripts/train_all.sh`) write `src/checkpoints/fold{0..4}_d43b_mix3u48.pt`,
`fold{0..4}_d44_ep48.pt` and `fold5_d43b_fdng48_s{0,1,2}.pt`. Every argument matches the `args`
stored in the released checkpoints. Retrained weights will differ slightly (cuDNN
nondeterminism is left on), so to reproduce the leaderboard score exactly use the released
weights:

```bash
python scripts/download_weights.py     # GitHub Release -> src/checkpoints/, sha256-verified
```

### 6. Upload to Kaggle

```bash
python kaggle/package_for_kaggle.py --code --weights --message "final submission"
```

This uploads `<KAGGLE_USER>/rsna-knee-code` (from `kaggle/code_dataset/`) and
`<KAGGLE_USER>/rsna-knee-weights` (the 13 checkpoints), both CC BY-NC 4.0. The notebook can
also use an optional `<KAGGLE_USER>/rsna-knee-decoder-wheels` dataset (manylinux wheels of
pylibjpeg 2.1.0, pylibjpeg-libjpeg 2.4.0, pylibjpeg-openjpeg 2.5.0 and python-gdcm 3.0.26 for
cp310 to cp312, from `pip download`); the training data is uncompressed DICOM, so it runs
without them. If you skip that dataset, delete its line from `dataset_sources` in
`kaggle/kernel-metadata.json`.

### 7. Run the notebook

Replace `<KAGGLE_USER>` in `kaggle/kernel-metadata.json`, then
`kaggle kernels push -p kaggle`. The kernel attaches the competition data, the datasets above
and the three public datasets nartaa/rsna-knee-publication-swa-weights-20261007 (v1),
heliosli/rsna-knee-compact-r7cnxfsss (v1) and heliosli/rsna-orthofoundation-fold0-epoch7-20261004
(v1); 2x T4, internet off, docker image `sha256:37c64f7d...` (in the metadata; if the push
rejects that image for your account, remove `docker_image` and Kaggle uses its current image,
which may change the scores slightly). The notebook checks the sha256 of the Raptor
checkpoint, of the R7 bundle manifest and of the four embedded third-party cells before
running them. It finds our weights at `/kaggle/input/rsna-knee-weights` or, failing that, in
the first attached dataset that contains `fold*.pt` files (none of the third-party datasets
does).

A commit run must log `loaded 13 member(s)`, `C02 family raptor: scored 3/3`,
`C02 family r7: scored 3/3` and `ours+raptor+r7 3`. Recompute the blend from a run's output with
`python tools/check_blend.py <run_output_dir>`.

`kaggle/rsna-knee-submission.ipynb` (md5 `c86262fa868ca7252c7c0ff629633796`) differs from the
notebook that was scored in one string only, the weights search path: the scored notebook's
`ours_weights_dir()` first tried an account-specific mount path, and this copy drops that
entry, so the weights are found as described above. Everything else is byte-identical. To
rebuild it from this repository:

```bash
python third_party/fetch_third_party.py   # the public notebook cells, sha256-checked
python kaggle/build/build_c02r7.py        # -> kaggle/build/out/, asserts md5 c86262fa...
```

## Weights

The 13 checkpoints (about 203 MB each, 2.64 GB) are attached to this repository's GitHub
Release, not stored in git; see `release/README.md` and `release/weights.sha256`. The released
files were re-saved with the training machine's data-path strings in their stored `args`
replaced by placeholders; every tensor is identical to the scored checkpoints.

## Licence

Copyright (c) 2026 the authors. Our code and weights are licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (`LICENSE`).
Third-party artifacts keep their own terms; see `THIRD_PARTY.md`. In particular, the four
third-party notebook cells embedded (base64) in `kaggle/rsna-knee-submission.ipynb` are the
work of nartaa and heliosli and are not covered by our licence. This repository contains no
competition data, no OAI data, no label tables and no imaging.

## Credits and acknowledgements

- **nartaa** (Raptor-384 weights and inference notebook), with **dreaddevelopment** (Raptor
  recipe) and the label authors credited in nartaa's dataset (**stevenleehans**, **pilkwang**,
  **riadmohamed42**).
- **heliosli** (R7 ConvNeXt-small bundle, OrthoFoundation dataset and the 0.957 notebook's R7
  stage; fastMRI / fastMRI+, SKM-TEA and KneeXNet upstream).
- **stevenleehans** (LLM report labels, v4_blend including **lixin73**'s labels) and **nartaa**
  (gemini-low report labels), used in `mix3`.
- **3DReasonKnee**: Sambara et al., "3DReasonKnee: Advancing Grounded Reasoning in Medical
  Vision Language Models", 2025, [arXiv:2510.20967](https://arxiv.org/abs/2510.20967)
  (MOAKS annotation release, CC BY 4.0).
- **timm** (ConvNeXt), **Qwen** team (Qwen3-32B).
- RSNA and Kaggle for the competition and its data.
- OAI: Data and/or research tools used in the preparation of this manuscript were obtained and
  analyzed from the controlled access datasets distributed from the Osteoarthritis Initiative
  (OAI), a data repository housed within the NIMH Data Archive (NDA). OAI is a collaborative
  informatics system created by the National Institute of Mental Health and the National
  Institute of Arthritis, Musculoskeletal and Skin Diseases (NIAMS) to provide a worldwide
  resource to quicken the pace of biomarker identification, scientific investigation and OA
  drug development. Dataset identifier(s): 10.15154/0hcg-f676. See `THIRD_PARTY.md` for the
  full OAI acknowledgement.
