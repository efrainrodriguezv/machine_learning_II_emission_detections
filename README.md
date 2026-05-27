# Emissions from Above
**Classifying Power Plant Pollution Levels via Satellite Imagery and Emissions Data**

Rob Scott · Ananya Sen · Mateo Ronquillo · Efrain Rodriguez

---

## 1. Problem Statement

Power plants are among the largest point-source emitters globally, yet monitoring relies on self-reported data that may be delayed or inaccurate. This project builds a four-task ML pipeline fusing Landsat 8 satellite imagery with EPA eGRID emissions records ultimately building to identify anomalies:

- **Task 1:** Binary classification — high vs. low pollution
- **Task 2:** Multi-class fuel type detection from satellite appearance
- **Task 3:** Continuous CO₂ intensity regression (image + tabular fusion)
- **Task 4:** Anomaly detection — flag underreporting candidates

**Target applications:** regulatory oversight, ESG screening, climate compliance auditing

---

## 2. Dataset Description

**USPPDS imagery** — 4,454 US plants at two resolutions: NAIP 1 m/px (~1,115×1,115 px, annotation ground truth) and Landsat 8 15 m/px (~75×75 px, primary ML input).

**EPA eGRID 2014** — Annual CO₂ emissions, intensity (lbs/MWh), nameplate capacity, capacity factor, net generation, heat rate, coordinates, and primary fuel category per plant.

**Class imbalance** — GAS (1,160) outnumbers NUCLEAR (46) by 25:1, attempted to resolve via `WeightedRandomSampler` and class-weighted loss.

---

## 3. Data Preprocessing

| Labels & Targets | Image & Tabular |
|---|---|
| **Binary:** CO₂ intensity quartile thresholds; middle 50% excluded for cleaner signal. | **Bands:** Per-channel min-max normalization to [0,1] per scene. |
| **Multi-class:** PLFUELCT fuel category; classes <20 samples dropped. | **Resize:** Bilinear upscale 75×75 → 64×64 px for CNN compatibility. |
| **Regression:** log₁(lbs CO₂/MWh) to normalize heavy-tailed distribution. | **Augmentation:** Random H/V flip, ±15° rotation, ±20% brightness/contrast (train only). |
| **Anomaly:** Fossil plants with zero/implausibly low CO₂ held as 'suspicious' test set. | **Tabular:** Median imputation; StandardScaler fit on train split; fuel label-encoded. |

---

## 4. Methodology

### Task 1 — Binary Classification

ResNet-18 fine-tuned from ImageNet; dropout (p=0.3) head; AdamW + cosine LR, 30 epochs. `WeightedRandomSampler` + class-weighted CE. Grad-CAM validates attention on plant infrastructure.

ResNet-18 was chosen for its strong capacity-to-parameter efficiency on small image patches (64×64 px), and ImageNet pretraining provides low-level edge and texture features that transfer meaningfully to satellite imagery despite the domain gap. The quartile-threshold labelling strategy (top vs. bottom 25% of CO₂ intensity) deliberately excludes boundary cases to create a cleaner, more separable signal for the binary decision.

### Task 2 — Multi-class

EfficientNet-B0 fine-tuned; macro-F1 model selection. t-SNE/PCA embedding plots assess latent fuel-type cluster separability. EfficientNet-B0's compound scaling of depth, width, and resolution yields higher accuracy per parameter than ResNet-18, which matters when discriminating across eight visually distinct classes including solar panel grids, cooling tower arrays, and wind turbine layouts. Macro-F1 is used for model selection rather than accuracy to prevent the dominant GAS class (1,160 samples) from masking poor performance on rare classes like NUCLEAR (46 samples).

### Task 3 — Regression

Three-way comparison: GBT tabular baseline, CNN image-only (Huber loss), late-fusion multimodal (CNN + tabular MLP embeddings concatenated before shared regression head). The three-way comparison of GBT tabular baseline, CNN image-only, and late-fusion multimodal is designed to isolate the marginal contribution of each data modality, directly testing whether satellite imagery provides predictive signal beyond what structured eGRID covariates alone can explain. The late-fusion architecture keeps the two branches independent, so each learns domain-appropriate representations before combining, and Huber loss is used in place of MSE to reduce sensitivity to the heavy-tailed outliers common in emissions data.

### Task 4 — Anomaly Detection

Weighted ensemble score: residual Z-score (40%), Isolation Forest on CNN embeddings (25%), One-Class SVM on tabular (20%), MC Dropout uncertainty (15%). Large positive residuals flag underreporting candidates.

No single anomaly signal is reliable in isolation, as residuals can be large due to model error rather than underreporting, and unsupervised detectors can flag visual or structural outliers unrelated to emissions, so a weighted ensemble of residual Z-scores, Isolation Forest, One-Class SVM, and MC Dropout uncertainty is used to require convergent evidence before flagging a facility. Training all detectors exclusively on verified plants ensures the normal distribution is well-defined, making deviations on unverified fossil-fuel facilities a meaningful and defensible signal for regulatory prioritization.

---

## 5. Task 1 Results — Binary Classification

ResNet-18 trained for 30 epochs on quartile-threshold labels.

**Table 2. Quantitative results on held-out test set (n = 443)**

| Metric | Train | Val (Best) | Test |
|---|---|---|---|
| Accuracy | 94.1% | ~79% | 79.9% |
| AUC-ROC | — | 0.852 | 0.866 |
| Recall (High) | — | — | 76.4% (107/140) |
| FP Rate (Low) | — | — | 19.5% (59/303) |

**Interpretation:** AUC-ROC 0.866 is well above the 0.5 random baseline. High-pollution recall (76.4%) and low false-positive rate (19.5% Low→High) are strong for a purely visual signal on 64×64 px patches. The 59 false positives are concentrated near the quartile boundary and motivate Grad-CAM analysis to confirm attention on plant infrastructure rather than confounding background texture.

### 5a. Task 2 Results — Fuel Type Classification

EfficientNet-B0 was trained for 40 epochs on 8 fuel categories. Overall test accuracy was **39.5%** (macro-F1 reported per class), well above the 12.5% random baseline for an 8-class problem.

HYDRO (0.68) and WIND (0.67) are the strongest classes; GAS (0.06) and NUCLEAR (0.14) are the weakest.

Structurally distinct classes (HYDRO, WIND) classify well because their visual footprints are unique, even at 64×64 px resolution. Fossil-fuel classes (GAS, OIL, COAL, BIOMASS) share similar thermal plant configurations and confuse the model heavily, as confirmed by the confusion matrix scatter in those rows. The unstructured t-SNE embedding space indicates the EfficientNet backbone has not yet learned strongly fuel-discriminative features from 15 m Landsat patches alone; incorporating NAIP high-resolution imagery or additional Landsat spectral bands (thermal TIRS) is the primary recommended next step for Task 2.

---

## 6. Evaluation Plan

| Task | Metrics | Validation Strategy |
|---|---|---|
| Task 1 — Binary Classification | Acc, F1, AUC-ROC, confusion matrix | Stratified 80/15/5; WeightedSampler |
| Task 2 — Multi-class Fuel Type | Per-class F1 (macro), confusion matrix | Stratified; class-weighted CE loss |
| Task 3 — Emissions Regression | MAE, RMSE, R²; residual by fuel/size | 3-way model comparison on test set |
| Task 4 — Anomaly Detection | AUC-PR, ensemble rank, watchlist | Train verified; MC Dropout uncertainty |

**Baselines:** Majority-class (Tasks 1–2); tabular GBT vs. image-only vs. fusion (Task 3); residual-only vs. full ensemble (Task 4 ablation).

**Qualitative:** Grad-CAM overlays confirm spatial attention; t-SNE embedding plots assess latent fuel-type separability.

---

## Stack

PyTorch · torchvision · scikit-learn · EfficientNet-B0 · ResNet-18 · Isolation Forest · One-Class SVM · Grad-CAM · pdfplumber

## Notebooks

| Notebook | Task |
|---|---|
| [`task1_binary_classification.ipynb`](task1_binary_classification.ipynb) | Binary classification (ResNet-18) |
| [`task2_multiclass_classification.ipynb`](task2_multiclass_classification.ipynb) | Fuel type detection (EfficientNet-B0) |
| [`task3_regression.ipynb`](task3_regression.ipynb) | CO₂ intensity regression (fusion model) |
| [`task4_anomaly_detection.ipynb`](task4_anomaly_detection.ipynb) | Anomaly detection (ensemble) |
