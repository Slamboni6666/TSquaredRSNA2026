# Third-party artifacts

Everything this solution uses that we did not make, with the version that ran, its hash, its
licence and whether this repository **includes** it or only **links** to it. The only
third-party material inside this repository is the four public notebook cells embedded
(base64, unmodified) in `kaggle/rsna-knee-submission.ipynb`; they are not covered by our
licence (see "The two public notebooks' cells" below). Everything else is linked only.

| Artifact | Version / file | sha256 | Licence | In this repo |
|---|---|---|---|---|
| RSNA Knee Abnormality Detection competition data | Kaggle competition `rsna-knee-abnormality-detection` | n/a | competition rules | link only (never included) |
| Osteoarthritis Initiative (OAI) SAG_3D_DESS MRI | NIMH Data Archive, dataset 10.15154/0hcg-f676 | n/a | NDA Data Use Certification | link only (never included) |
| 3DReasonKnee MOAKS annotations (Sambara et al. 2025, [arXiv:2510.20967](https://arxiv.org/abs/2510.20967)) | [huggingface.co/datasets/rajpurkarlab/3DReasonKnee](https://huggingface.co/datasets/rajpurkarlab/3DReasonKnee): `data/labels/*.csv`, `split_files/all_studies.csv` | rows carry OAI IDs; not hashed here | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | link only |
| [stevenleehans/rsna-knee-llm-report-labels](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels) | v6: `llm_labels_v4_blend.csv`, `llm_labels_full.csv` | `eb4c2b29e7e84adbb4b5ccee825ca20ade6a43dbe295f13be442d549e4f5dca4`, `96444cd3c7d396bae4a71454317d9cbe0689759b5278b30939781e04d0774537` | CC0-1.0 | link only (UID-keyed) |
| [nartaa/rsna-knee-hpo-assets](https://www.kaggle.com/datasets/nartaa/rsna-knee-hpo-assets) | `train_labels_v1.parquet`, `labels_v1_gemlow.parquet` (same bytes in v6 to v9; pin by sha256) | `a134cfce2f751f90f4219d791a563ee677b6223ac28f32e456bb3759c1d3fba9`, `77a3857a25a81d7e1ab4e360b1b86bb2fba8620cb43d14d02b97c6cd16004f00` | CC0-1.0 | link only (UID-keyed) |
| [nartaa/rsna-knee-publication-swa-weights-20261007](https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007) | v1: `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt` (292,840,787 B) | `7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef` | "other": the dataset's model-use terms allow research and educational use including this competition, with credit and the OAI notice; no rights to OAI data. Its "Adaptation and redistribution" clause (added 2026-10-13) allows non-commercial redistribution of the weights or adapted versions, "including licensing a winning Submission built on them under CC BY-NC 4.0", if the OAI notice is kept with any copy (not needed here: the weights are linked, not copied) | link only (attached to the Kaggle notebook) |
| [heliosli/rsna-knee-compact-r7cnxfsss](https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-r7cnxfsss) | v1, R7 bundle `8c7b666e27dc0e8f`: 4 ConvNeXt-small folds, `side_cls.pt`, stackers; trained with a fastMRI+ auxiliary task | `manifest.json`: `c4bb0f7399ada9a020e0c15cae6252033cf0c4335e999961793a176063074ec6` | "other"; no redistribution grant. Upstream: trained with fastMRI+ lesion-box annotations on fastMRI images; the NYU fastMRI Dataset Sharing Agreement allows internal research and education use only and forbids redistributing "any subsequent variables or data files derived from" the dataset | link only (never re-host) |
| [heliosli/rsna-orthofoundation-fold0-epoch7-20261004](https://www.kaggle.com/datasets/heliosli/rsna-orthofoundation-fold0-epoch7-20261004) | v1: decode/render code, pinned pydicom 3.0.1 / timm 1.0.25 wheels, `anatomy.pt` (112,496,037 B, SKM-TEA model-zoo V-Net) | `anatomy.pt`: `5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e` | "other"; "no new open-source license is granted"; upstream licences retained. Upstream: SKM-TEA model-zoo weights (Stanford AIMI dataset terms, non-commercial research use) and KneeXNet auxiliary weights, "unchanged" per the dataset page | link only |
| nartaa notebook [rsna-knee-0949-anatomical-mirror](https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror), code cells 3 and 4 (Raptor-384 inference) | kernel version 2 (current on 2026-10-22) | cell 3 `fd96541cbe2e5b37494636dd13b79871a7a6ed9ad94fe528dc6f7a253d4be8b8` (1,268 B), cell 4 `bcd864cbf349882e876959b49cb348cbb89b4c62fc204d4e4261049c5e3a79c5` (95,348 B) | not confirmed (see below) | included only as base64 inside `kaggle/rsna-knee-submission.ipynb` (unmodified, sha256-checked); not under our LICENSE. `third_party/fetch_third_party.py` downloads them |
| heliosli notebook [0-957-latest-dataset-score-in-one-click](https://www.kaggle.com/code/heliosli/0-957-latest-dataset-score-in-one-click), code cells 4 and 5 (R7 stage) | kernel version 1 (current on 2026-10-22; ipynb md5 `95920079d1bb9b236821401907daa411` when built) | cell 4 `cb168cfe82f5a04114894f6c07bbd6daf0373ad2eb0dfcee34df3ecf625986af` (391,879 B), cell 5 `68f70dd31cc3544cb5734313d06538355a34b122ff430a32820756175c9c9dd9` (2,064 B) | not confirmed (see below) | included only as base64 inside `kaggle/rsna-knee-submission.ipynb` (unmodified, sha256-checked); not under our LICENSE. `third_party/fetch_third_party.py` downloads them |
| timm `convnext_small.in12k_ft_in1k` | timm 1.0.22 default tag, Hugging Face `timm/convnext_small.in12k_ft_in1k` | downloaded at training time | Apache-2.0 | link only |
| Qwen/Qwen3-32B | Hugging Face | downloaded at labelling time | Apache-2.0 | link only |
| DICOM decoder wheels (optional on Kaggle) | pylibjpeg 2.1.0, pylibjpeg-libjpeg 2.4.0, pylibjpeg-openjpeg 2.5.0, python-gdcm 3.0.26 | from PyPI | MIT; GPL-3.0 (pylibjpeg-libjpeg); MIT; BSD | link only (you `pip download` them) |
| Kaggle docker image | `gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461` | (digest) | Kaggle | referenced in `kaggle/kernel-metadata.json` |

## The two public notebooks' cells

**Licence check (2026-10-22).** Kaggle shows a notebook's licence on its web page (public
notebooks default to Apache 2.0, but the author can choose otherwise). We could not confirm the
licence of either notebook: both pages are rendered by JavaScript and returned no licence text
to a plain fetch, and the Kaggle API (`kaggle kernels pull -m`, kernel metadata) has no licence
field. Because no redistribution licence could be confirmed, the cells are **not** included as
separate files, and before this repository is made public the licence line on each notebook
page should be read in a logged-in browser: if it says Apache 2.0, add the licence and a
NOTICE under `third_party/<author>/`; otherwise keep relying on the authors' public sharing
of these cells for the competition, or ask them. `third_party/fetch_third_party.py` downloads them from the authors' notebooks
with your own Kaggle credentials and refuses any cell whose sha256 differs from the table above.
Kaggle lets only a kernel's owner pull an older version (403 otherwise), so the script tries the
pinned version and falls back to the current one; the sha256 check is what pins the bytes.

**The shipped notebook embeds these cells.** `kaggle/rsna-knee-submission.ipynb` is the exact
notebook that produced the submission. Its first code cell holds the four cells above,
base64-encoded and byte-identical; the notebook checks each one's sha256 before running it as an
isolated subprocess, with a few asserted orchestration patches (output paths, temp dir, private
wheel dir, deadline; the model code is untouched). They remain the work of their authors under
their own terms and are not licensed by us under CC BY-NC 4.0.

## Credits

- **nartaa**: Raptor-384 weights and inference source; with **dreaddevelopment** (Raptor
  recipe), timm CoAtNet, and the weak labels of **stevenleehans**, **pilkwang** and
  **riadmohamed42** (see nartaa's dataset README).
- **heliosli**: R7 ConvNeXt-small bundle, the OrthoFoundation dataset and the R7 stage of the
  0.957 notebook; fastMRI and fastMRI+ (auxiliary lesion-box task), the SKM-TEA model zoo and
  KneeXNet upstream.
- **stevenleehans**: LLM report labels (v4_blend includes **lixin73**'s GPT labels).
- **nartaa**: gemini-low report labels (`rsna-knee-hpo-assets`).
- **3DReasonKnee** (Sambara, Kim, Zhang, Luo, Johri, Baharoon, Ro, Rajpurkar; arXiv:2510.20967,
  2025): the MOAKS annotation release, CC BY 4.0.
- **timm** (Ross Wightman et al.), **Qwen** team, **pydicom**, **PyTorch**.
- RSNA and Kaggle for the competition and its data.

## OAI acknowledgement

Data and/or research tools used in the preparation of this manuscript were obtained and
analyzed from the controlled access datasets distributed from the Osteoarthritis Initiative
(OAI), a data repository housed within the NIMH Data Archive (NDA). OAI is a collaborative
informatics system created by the National Institute of Mental Health and the National
Institute of Arthritis, Musculoskeletal and Skin Diseases (NIAMS) to provide a worldwide
resource to quicken the pace of biomarker identification, scientific investigation and OA drug
development. Dataset identifier(s): 10.15154/0hcg-f676.

The OAI is a public-private partnership comprised of five contracts (N01-AR-2-2258;
N01-AR-2-2259; N01-AR-2-2260; N01-AR-2-2261; N01-AR-2-2262) funded by the National Institutes
of Health, a branch of the Department of Health and Human Services, and conducted by the OAI
Study Investigators. Private funding partners include Merck Research Laboratories; Novartis
Pharmaceuticals Corporation, GlaxoSmithKline; and Pfizer, Inc. Private sector funding for the
OAI is managed by the Foundation for the National Institutes of Health. This repository was
prepared using OAI data and does not necessarily reflect the opinions or views of the OAI
investigators, the NIH, or the private funding partners.
