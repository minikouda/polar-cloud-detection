# Polar Cloud Detection via MISR Satellite Imagery

**Stat 214 — Lab 2, Spring 2025, UC Berkeley**

A machine learning pipeline for pixel-level cloud detection in Arctic regions using multi-angle satellite radiance data from NASA's Multi-angle Imaging SpectroRadiometer (MISR) sensor aboard the Terra satellite.

---

## Overview

Standard cloud detection algorithms rely on brightness thresholds — clouds are bright, surfaces are dark. This breaks down in the Arctic, where sea ice and snow are equally bright and cold as clouds. This project exploits the fact that **clouds and ice scatter light differently across viewing angles**: ice scatters uniformly, while clouds scatter strongly in the forward direction. The MISR sensor captures the same scene simultaneously from 9 camera angles, providing the angular variation signal needed for classification.

The pipeline proceeds in four stages:
1. **Exploratory Data Analysis** — understand feature structure and discriminative power
2. **Feature Engineering** — engineer new features and extract spatial embeddings via an autoencoder
3. **Classification** — train and evaluate multiple classifiers for pixel-level cloud detection
4. **Stability Analysis** — assess robustness to label noise and distribution shift

---

## Data

The dataset consists of 164 MISR images of the same Arctic geographic region (path 26), collected during the 2002 Arctic daylight season. Each `.npz` file represents one satellite overpass and contains per-pixel measurements:

| Column | Description |
|--------|-------------|
| 0 | y coordinate |
| 1 | x coordinate |
| 2 | NDAI (Normalized Difference Angular Index) |
| 3 | SD (Standard Deviation of local radiance) |
| 4 | CORR (Cross-angle correlation) |
| 5–9 | Radiance angles: DF (70.5°), CF (60°), BF (45.6°), AF (26.1°), AN (nadir 0°) |
| 10 | Expert label: +1 cloud, −1 not cloud, 0 unlabeled *(labeled images only)* |

Only 3 images carry expert labels (`O012791`, `O013257`, `O013490`); the remaining 161 are unlabeled and used for transfer learning pre-training.

**Train/test split:** Temporal holdout — `O012791` and `O013257` (spring) for training, `O013490` (early summer) for testing. This mirrors real-world deployment where models are applied to future satellite passes.

---

## Repository Structure

```
polar-cloud-detection/
├── code/
│   ├── autoencoder.py               # Convolutional autoencoder architecture
│   ├── data.py                      # Data loading utilities
│   ├── feature_engineering.py       # Full feature engineering pipeline
│   ├── feature_engineering_autoencoder.py
│   ├── patchdataset.py              # PyTorch patch dataset for autoencoder training
│   ├── get_embedding.py             # Extract AE embeddings from trained model
│   ├── run_autoencoder.py           # Autoencoder training entry point
│   ├── configs/                     # YAML configs for training runs
│   ├── models/
│   │   ├── ensemble.py              # Random Forest & Histogram Gradient Boosting
│   │   ├── logreg_svm.py            # Logistic Regression & SVM models
│   │   └── logreg_svm_stability.py  # Label noise robustness experiments
│   ├── notebooks/
│   │   ├── 01_eda_jacky.ipynb
│   │   ├── 01_eda_ricardo.ipynb
│   │   ├── 02_autoencoder.ipynb
│   │   ├── 02_feature_eng.ipynb
│   │   ├── 03_hgb_analysis.ipynb
│   │   └── 03_rf_analysis.ipynb
│   ├── run.sh                       # Full pipeline
│   ├── run_all_models.sh
│   ├── run_autoencoder.sh
│   ├── run_feat_eng.sh
│   └── run_logreg_svm.sh
├── data/
│   ├── image_data.zip               # All 164 MISR images (not in repo)
│   └── O*.npz                       # Individual image files
├── feature_eng_dataset/             # Precomputed feature CSVs
├── report/
│   ├── lab2.tex                     # Full report (LaTeX)
│   ├── model.tex
│   └── figures/
├── results/                         # Model checkpoints (.pt) and output figures
├── documents/                       # Reference papers (yu2008.pdf, etc.)
└── environment.yaml                 # Conda environment
```

---

## Methods

### Feature Engineering

**Hand-crafted features (9 total):**
- Raw radiances: DF, CF, BF, AF, AN
- `NDAI_DF_AF`: improved NDAI variant comparing DF (70.5°) to AF (26.1°) — outperforms the original NDAI (AUC 0.833 vs 0.823)
- `PC1`: first principal component of the 5 radiances, capturing 92% of variance (overall brightness)
- `SD` and `CORR` retained from the original feature set

**Feature importance** (ranked by AUC on labeled pixels):

| Rank | Feature | AUC | KS Stat | MI (bits) |
|------|---------|-----|---------|-----------|
| 1 | SD | 0.935 | 0.821 | 0.428 |
| 2 | NDAI | 0.823 | 0.621 | 0.259 |
| 3 | AF | 0.799 | 0.466 | 0.202 |
| ... | ... | ... | ... | ... |
| 8 | CORR | 0.524 | 0.058 | 0.010 |

### Transfer Learning via Autoencoder

With only 3 labeled images (~207,000 labeled pixels), we pre-train a convolutional autoencoder on **all 164 images** (labeled + unlabeled) to learn spatial representations, then transfer the encoder embeddings to downstream classifiers.

- **Input:** 9×9 pixel patches, 8 channels → 648-dimensional
- **Bottleneck:** 32-dimensional embedding
- **Training:** 5,000 randomly sampled patches per image; MSE reconstruction loss; early stopping; lr=5×10⁻⁴, weight decay=10⁻⁵
- **Transfer:** Encoder frozen; 32 AE features appended to hand-crafted features → **41 total features**

### Models

| Model | Accuracy | ROC AUC | F1 | Precision | Recall |
|-------|----------|---------|-----|-----------|--------|
| RF (Literature — 3 features) | 0.9559 | 0.9945 | 0.9558 | 0.9170 | 0.9981 |
| **HGB (Base)** *(selected)* | **0.9608** | **0.9958** | **0.9605** | **0.9283** | **0.9950** |
| RF (Full — 41 features) | 0.9732 | 0.9941 | 0.9725 | 0.9564 | 0.9892 |
| HGB (Full — 41 features) | 0.9787 | 0.9976 | 0.9780 | 0.9660 | 0.9903 |
| Lasso Logistic Regression | 0.4973 | 0.5109 | 0.4957 | — | — |
| Stepwise Logistic Regression | 0.8200 | 0.8101 | 0.8113 | — | — |
| SVM (Base — 3 features) | 0.9563 | 0.9487 | 0.9523 | 0.9465 | 0.9435 |
| SVM (Full — 8 features) | 0.4202 | 0.4148 | 0.3569 | — | — |

**Selected model:** Histogram Gradient Boosting (Base) — best balance of performance and simplicity. Tree-based models are robust to the multicollinearity present in raw radiances; the 3-feature literature baseline already achieves >95% accuracy.

### Key Findings

- **SD is the dominant feature** (AUC=0.935): cloud pixels have ~4.4× higher local radiance standard deviation than non-cloud, reflecting textural contrast between rough clouds and smooth Arctic ice.
- **CORR is nearly useless alone** (AUC=0.524): designed to detect high-altitude clouds, it fails on the predominantly low-altitude clouds in these images.
- **Errors cluster at cloud boundaries**: error rate spikes to ~50% within 6 pixels of a cloud edge, drops below 1% beyond 100 pixels. Two failure modes: (1) extreme-texture clouds missed (high SD, FN), (2) smooth ice falsely flagged (high CORR, FP).
- **Label noise robustness**: RF performance barely changes under 5% training label corruption (ΔAUC=−0.001), owing to ensemble bagging diluting corrupted labels.

---

## Setup

```bash
conda env create -f code/environment.yaml
conda activate <env_name>
```

**Run the full pipeline:**
```bash
cd code
bash run.sh
```

**Or step by step:**
```bash
bash run_autoencoder.sh       # Pre-train autoencoder on all 164 images
bash run_feat_eng.sh          # Extract features + AE embeddings
bash run_logreg_svm.sh        # Train logistic regression and SVM models
bash run_all_models.sh        # Train all ensemble models
```

Trained autoencoder checkpoints are saved as `.pt` files in `results/`.

---

## References

Shi, T., Yu, B., Clothiaux, E. E., & Braverman, A. J. (2008). Daytime Arctic cloud detection based on multi-angle satellite data with case studies. *Journal of the American Statistical Association*, 103(482), 584–597.
