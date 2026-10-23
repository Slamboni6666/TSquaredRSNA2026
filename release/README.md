# Release assets: the 13 checkpoints

The weights are **not in git**. They are attached as assets to a GitHub Release of this
repository and are fetched and checked with:

```bash
python scripts/download_weights.py          # -> src/checkpoints/, sha256-verified
```

The script reads the GitHub repository from this clone's `origin` remote (or `--repo owner/name`) and
downloads the assets of release `v1.0` (or `--tag`).

## Files

| File | Run (see `scripts/train_all.sh`) | Bytes |
|---|---|---|
| `fold{0..4}_d43b_mix3u48.pt` | A: 5 folds, `report_labels_mix3.csv`, OAI weak targets, uniform target weights | 202,735,072 each |
| `fold{0..4}_d44_ep48.pt` | B: 5 folds, `report_labels_consensus.csv`, OAI safe targets | 202,733,228 each |
| `fold5_d43b_fdng48_s{0,1,2}.pt` | C: all-data (fold 5, gold studies excluded), seeds 0/1/2 | 202,735,738 each |

Total 2,635,548,714 bytes. `weights.sha256` in this folder holds the sha256 of every file
(`sha256sum -c release/weights.sha256` from inside the download folder also works).

## Contents of each file

`torch.load(path, map_location="cpu", weights_only=True)` returns a dict:

- `model`: the ConvNeXt-small 2.5D MIL network's state dict (359 fp32 tensors, 50.6 M parameters
  including normalisation buffers). No optimizer state, no EMA copy.
- `args`: the `src/train.py` command-line arguments of the run (the notebook rebuilds the network
  from them). The three data paths are placeholders: `cache` = `<CACHE_DIR>`,
  `oai_cache` = `<OAI_CACHE_DIR>`, `oai_labels` = `<OAI_LABELS>`.
- `fold`: the fold number (5 = all-data).

No images, labels, study identifiers or OAI participant identifiers are stored in the weights.

**These files were re-saved for release.** The scored checkpoints stored the training
machine's absolute data paths in `args`. Each released file was loaded, those three strings
were replaced by the placeholders above, and the file was saved again; nothing else changed.
Every one of the 359 tensors of every file was then checked equal (`torch.equal`, same dtype and
shape) to the scored checkpoint, the `fold` value and all other `args` were checked unchanged,
and each file was rebuilt and loaded strictly with the notebook's own model-building code
(`knee_net_from_args`). The sha256 values in `weights.sha256` are those of the released files,
so they differ from the scored checkpoints' file hashes.

## Licence and attribution

CC BY-NC 4.0 (see `LICENSE`). The models were trained on the RSNA Knee Abnormality Detection
competition data with auxiliary supervision from the Osteoarthritis Initiative (OAI); use of the
weights does not grant any right to the underlying data. Required acknowledgement:

> Data and/or research tools used in the preparation of this manuscript were obtained and
> analyzed from the controlled access datasets distributed from the Osteoarthritis Initiative
> (OAI), a data repository housed within the NIMH Data Archive (NDA). OAI is a collaborative
> informatics system created by the National Institute of Mental Health and the National
> Institute of Arthritis, Musculoskeletal and Skin Diseases (NIAMS) to provide a worldwide
> resource to quicken the pace of biomarker identification, scientific investigation and OA
> drug development. Dataset identifier(s): 10.15154/0hcg-f676.
