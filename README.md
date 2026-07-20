# ABSS-Net
**Reliable and calibrated diabetic retinopathy grading from Gaussian-filtered fundus images using attention and gated state-space feature mixing**
Pradeep Gupta<sup>1,2</sup>, Shyh-An Yeh<sup>2,3,\*</sup>, Tsair-Fwu Lee<sup>2,4,\*</sup>
<sup>1</sup> Department of Computer Science and Engineering, Ajay Kumar Garg Engineering College, Ghaziabad, India
<sup>2</sup> Medical Physics and Informatics Laboratory of Electronics Engineering, National Kaohsiung University of Science and Technology, Kaohsiung 80778, Taiwan, ROC
<sup>3</sup> Department of Medical Imaging and Radiological Sciences, I-Shou University, Kaohsiung 82445, Taiwan, ROC
<sup>4</sup> Department of Radiation Oncology, E-DA Hospital, Kaohsiung 82445, Taiwan
<sup>\*</sup> Corresponding authors: tflee@nkust.edu.tw, sayeh@outlook.com
## Manuscript citation
> Gupta, P., Yeh, S.-A., & Lee, T.-F. (2026). *Reliable and calibrated diabetic retinopathy grading from Gaussian-filtered fundus images using attention and gated state-space feature mixing*. 
## Technical overview
ABSS-Net performs five-class ICDR diabetic retinopathy grading (No\_DR, Mild, Moderate, Severe, Proliferate\_DR) on 224x224 Ben-Graham Gaussian-filtered colour fundus photographs. The model has **5,470,756 trainable parameters** and comprises six stages:
1. **Input preprocessing** - CLAHE on the L channel in LAB space (clip 2.0, tiles 8x8), resize to 224x224 bilinear, ImageNet normalisation.
2. **Backbone** - `timm` EfficientNet-B0 Noisy-Student (`features_only`, `out_indices=(2,3,4)`), fully fine-tuned, giving F3/F4/F5 at strides 8/16/32 with channels (40, 112, 320).
3. **Attention** - a CBAM (SE-style channel attention with reduction r=16, then 7x7 spatial attention) applied independently to each of F3, F4, F5.
4. **Fusion** - 1x1 projection of each scale to 128 channels, bilinear upsample to F3 resolution, channel concat, then 3x3 conv + BatchNorm + GELU to a 256-channel map U.
5. **Token mixing** - U is flattened to N=784 tokens of dimension D=256 and passed through **two stacked gated blocks**. Each block is LayerNorm, linear up-projection (factor 2), depth-wise 1-D convolution (kernel 5), sigmoid gate, linear out-projection, dropout 0.1, residual.
6. **Dual heads** - mean-pooled embedding feeds a classification MLP [256 to 128 to 5] (dropout 0.3) and a scalar **reliability head** [256 to 64 to 1] with sigmoid, giving a per-sample confidence c in [0,1].

**Training objective:** `L = L_focal + 0.1 * L_cal + 0.5 * L_rel`, where `L_focal` is inverse-frequency class-weighted focal loss (gamma=2.0), `L_cal` is a Brier-style penalty `E[(p_y - 1)^2]`, and `L_rel` is a BCE loss supervising c against the detached binary correctness target `1[argmax(logits) == y]`.

### Terminology note

The gated block is **not** a full selective-scan state-space model in the Mamba sense: it does not implement an explicit recurrent hidden-state update or input-dependent (A, B, C) parameters via a hardware-aware parallel scan. It is a **gated, Mamba-inspired depth-wise-convolutional block** providing O(N) token mixing with input-dependent gating. A drop-in genuine `MambaBlock` (requires `mamba-ssm`) is provided in `code/ablation_ssm_vs_mamba.py` for future controlled comparison. The manuscript uses this same terminology.

---

## Repository structure

```
ABSSNET/
├── README.md                       This file
├── LICENSE                         MIT (code only; not the images)
├── CITATION.cff                    Machine-readable citation metadata
├── requirements.txt                pip dependencies
├── environment.yml                 conda environment
├── .zenodo.json                    Zenodo deposition metadata
├── .gitignore
├── configs/
│   └── abssnet_default.yaml        All hyperparameters in one place
├── code/
│   ├── abss_net.py                     Baseline single-split pipeline (see caveat below)
│   ├── abss_net_all_experiments_colab.py   PRIMARY pipeline - all manuscript experiments
│   ├── calibration_metrics.py          Brier / NLL / ECE / MCE + temperature scaling
│   ├── profile_efficiency.py           Params, GMACs, latency, throughput, GPU memory
│   ├── frequency_attenuation_analysis.py   High-frequency energy + gradient magnitude
│   ├── ablation_raw_vs_gaussian.py     Raw vs Ben-Graham-filtered ablation entry point
│   ├── ablation_ssm_vs_mamba.py        Genuine selective-scan Mamba drop-in block
│   ├── reproduce_baselines.py          Identical-fold baseline re-training entry point
│   ├── generate_revised_figures.py     Regenerates manuscript figures from result tables
│   ├── build_revision_outputs.py       Packages values into CSV/XLSX/LaTeX (NOT training)
│   ├── export_cv_splits.py             Regenerates the exact 5-fold split assignment
│   └── GPU_RUNBOOK.md                  Step-by-step GPU execution guide
├── models/
│   └── abssnet_fold{0..4}.pth         Per-fold checkpoints (~21.2 MB each)
└── results/
    ├── figures/                    19 manuscript figures (PNG)
    ├── tables/                     Result tables (CSV), pooled OOF arrays (NPZ), XLSX workbook
    └── logs/                       Experiment log
```

---

## Installation

```bash
git clone https://github.com/guptapradeep85/ABSSNET.git
cd ABSSNET

# Option A - pip
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# Option B - conda
conda env create -f environment.yml
conda activate abssnet
```

Install the PyTorch build matching your CUDA version from <https://pytorch.org/get-started/locally/>. The optional `mamba-ssm` / `causal-conv1d` packages are needed **only** for the Mamba ablation and require a CUDA toolchain.

---

## Dataset instructions

**The fundus images are third-party data and are NOT included in this repository.** See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for the full statement.

### Primary dataset (required)

Diabetic Retinopathy 224x224 Gaussian Filtered - 3,662 images, five ICDR classes:

<https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered>

It is a pre-processed derivative of APTOS-2019 Blindness Detection: <https://www.kaggle.com/competitions/aptos2019-blindness-detection>

### Expected input directory structure

```
data/DR224/Train_image/
├── No_DR/            1805 images
├── Mild/              370 images
├── Moderate/          999 images
├── Severe/            193 images
└── Proliferate_DR/    295 images
                    = 3662 total
```

Verify your copy against `results/tables/dataset_class_counts.csv` before training. The loaders also accept the folder spelled `Train _image` (with a space), as some Kaggle mirrors use that name.

### Optional datasets

| Purpose | Dataset | Location | Source |
|---|---|---|---|
| Raw-vs-Gaussian ablation | Unfiltered APTOS-2019 | `data/APTOS_RAW/Train_image/<class>/` | Kaggle APTOS-2019 competition |
| External validation | Messidor-2 (1,744 images) | `data/Messidor-2/` | <https://www.adcis.net/en/third-party/messidor2/> (registration required) |
## Preprocessing
Applied by the released code, in this order:
| Stage | Training | Inference |
|---|---|---|
| CLAHE (LAB L channel, clip 2.0, tiles 8x8) | yes | **no** |
| Resize 224x224 bilinear | yes | yes |
| Random horizontal flip | yes | no |
| Random vertical flip | yes | no |
| Random rotation ±15° | yes | no |
| Colour jitter (0.15/0.15/0.10/0.02) | yes | no |
| ImageNet normalisation | yes | yes |
| CutMix (alpha=1.0, p=0.3) / MixUp (alpha=0.4, p=0.3), mutually exclusive per batch | yes | no |

No resampler is used; class imbalance is handled solely by the inverse-frequency focal weights `alpha = [0.2157, 1.0522, 0.3898, 2.0225, 1.3198]`.
## Training
`code/abss_net_all_experiments_colab.py` is the **primary** pipeline and reproduces the manuscript. Set the dataset path in the config block near the top (`DATASET_SEARCH_ROOTS`, `REV_ROOT`) or export `ABSSNET_DATA_ROOT`, then:
```bash
# Single 80/20 stratified development split (seed 42)
python code/abss_net_all_experiments_colab.py --run-mode single
# Full stratified 5-fold cross-validation (this is what the manuscript reports)
python code/abss_net_all_experiments_colab.py --run-mode cv
```
Key settings live in `configs/abssnet_default.yaml`: batch 32, AdamW lr 1e-4 wd 1e-4, CosineAnnealingLR T_max=20, 20 epochs, AMP fp16, seed 42, `RELIABILITY_WEIGHT=0.5`, `CAL_WEIGHT=0.1`.
**Checkpoint selection:** the checkpoint with the best **validation QWK** is saved per fold (`models/abssnet_fold{k}.pth`). No early stopping is used - all 20 epochs always run.
> **Caveat on `code/abss_net.py`.** This is the earlier single-split baseline script. Its training loop implements only `L_focal + 0.1 * L_cal` - it does **not** include the reliability BCE term, CLAHE, CutMix or MixUp, and it uses the legacy `timm` backbone alias `tf_efficientnet_b0_ns`. **It does not reproduce the manuscript results.** Use `abss_net_all_experiments_colab.py`. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
### Five-fold cross-validation
Folds are `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` over the full 3,662-image set in sorted filename order; each image is held out exactly once, and the out-of-fold predictions are pooled for dataset-level metrics. To materialise the exact split assignment as a CSV:
```bash
python code/export_cv_splits.py --img-dir data/DR224/Train_image --out results/tables/cv_folds.csv
```
---

## Evaluation

```bash
# Calibration: Brier, NLL, ECE(15-bin), MCE + fitted temperature scaling
python code/calibration_metrics.py --npz results/tables/pooled_oof_outputs.npz

# Efficiency: parameters, GMACs/GFLOPs, latency, throughput, peak GPU memory
python -c "from code.profile_efficiency import profile; from code.abss_net_all_experiments_colab import ABSSNet; profile(ABSSNet(), device='cuda')"

# Frequency attenuation by DR grade (requires the images)
python code/frequency_attenuation_analysis.py --img-dir data/DR224/Train_image

# Regenerate manuscript figures from the archived result tables
python code/generate_revised_figures.py
```

`results/tables/pooled_oof_outputs.npz` contains `y_prob` (3662x5), `y_true`, `y_pred` and `conf` for all 3,662 images, so the calibration and selective-prediction analyses run **without** needing the images or a GPU.

### Calibration

Two complementary strategies are reported, both computed on the pooled out-of-fold predictions:

- **Training-time** - the Brier-style penalty `L_cal` inside the objective.
- **Post-hoc** - single-parameter temperature scaling fitted by minimising NLL (`calibration_metrics.temperature_scale`).



## Reproducing the principal results

| Manuscript item | Command / script |
|---|---|
| Table 1 (split), Fig 2 | `results/tables/dataset_class_counts.csv` |
| Table 4 (final val), Tables 5-7, Figs 3-9 | `abss_net_all_experiments_colab.py --run-mode single` |
| Tables 8-9 (5-fold), Figs 11-14 | `abss_net_all_experiments_colab.py --run-mode cv` |
| Table 5 (calibration + temperature) | `calibration_metrics.py --npz results/tables/pooled_oof_outputs.npz` |
| Table 10 + Figs 15, 19 (reliability, selective prediction) | `abss_net_all_experiments_colab.py --run-mode cv` |
| Table 11 (Messidor-2) | `abss_net_all_experiments_colab.py` with `RUN_EXTERNAL=True` |
| Table 12 (ablation) | `abss_net_all_experiments_colab.py` ablation cells |
| Table 14 (efficiency) | `profile_efficiency.py` |
| Table 15 (frequency attenuation) | `frequency_attenuation_analysis.py` |
| Figs 10, 18 (Grad-CAM) | Grad-CAM cells in `abss_net_all_experiments_colab.py` |

The authoritative mapping, including every caveat, is in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Expected outputs

Running the CV pipeline writes, under your configured output root:

```
results/  colab_gpu/*.json         per-fold metric dumps
tables/   cv_per_fold_metrics.csv  cv_summary_metrics.csv  oof_per_class_metrics.csv
          reliability_bins.csv     pooled_oof_outputs.npz  ablation_study.csv
figures/  fig2..fig19 *.png
logs/     experiment_log.txt
models/   abssnet_fold{0..4}.pth
```

Reference copies of all of these are committed under `results/` and `models/`, so you can diff your run against ours. Expected magnitudes (5-fold mean ± sd): accuracy 0.7026 ± 0.0420, macro-F1 0.5206 ± 0.0553, QWK 0.7933 ± 0.0418, AUC 0.8862 ± 0.0168, ECE 0.1778 ± 0.0439.

---

## Hardware and software environment

| Component | Value used for the reported results |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB (Google Colab) |
| CUDA | 12.x |
| Python | 3.10 |
| PyTorch | 2.x with `torch.amp` fp16 |
| Backbone weights | `timm` `tf_efficientnet_b0.ns_jft_in1k` (21.4 MB) |
| Batch size / workers | 32 / 2 |
| Seed | 42 (Python, NumPy, PyTorch, CUDA) |
| Checkpoint size | 21.2 MB per fold |
| Peak GPU memory | 0.56 GB (inference, batch 32) |
**Minimum to retrain:** any CUDA GPU with >= 8 GB. **Minimum to re-run the calibration, reliability and selective-prediction analyses:** CPU only, using the committed NPZ.
**Determinism caveat:** seeds are fixed for Python, NumPy and PyTorch, but cuDNN autotuning and AMP are not forced into deterministic mode, so metrics may vary in the third decimal place between runs and across GPU models.
## GitHub release information
The version corresponding to the manuscript is tagged **v1.0.0**: https://github.com/guptapradeep85/ABSSNET/releases/tag/v1.0.0
Cite the tagged release, not the moving `main` branch, so the cited version cannot change after submission.


