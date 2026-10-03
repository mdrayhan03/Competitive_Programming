# Pasture Biomass Estimation — Complete Self-Contained Context

> **Purpose of this file.** This single document carries 100% of the context needed to understand and
> continue this project **without any other file, script, notebook, or dataset**. Everything essential —
> problem, data, exact method (with code), all results, the research paper, limitations, and next steps —
> is embedded inline. Hand this one `.md` to any new chat thread, session, or LLM and it can pick up the
> work with full context. (Deliberately excludes machine paths and personal identifiers.)

---

## 0. One-paragraph summary

We estimate **pasture dry biomass (in grams)** from a **single top-down RGB image** — predicting three
botanical fractions (green, dead, clover) and two derived totals (GDM, Total). We benchmark **five deep
vision backbones** (MobileNet-V1, EfficientNet-B0, EfficientNet-B3, DenseNet-121, ViT-B/16) under **one
identical pipeline** on the CSIRO Image2Biomass dataset (357 images). **ViT-B/16 is best** (weighted
R² = 0.7455) and most stable, but **all confidence intervals overlap** (small 72-image test set), so we
report it as *consistently preferable, not statistically proven*. The work is an **architectural
benchmark + ablation study** (supervisor directive: **no architecture novelty**), validated with
bootstrap CIs, paired-loop significance, an ablation on two representative models, and Grad-CAM/SHAP/LIME
explainability. A full IEEE-style paper is written, revised, and number-verified.

---

## 1. Problem definition

- **Input:** one RGB pasture-quadrat photo, resized to **224×224**.
- **Outputs (5 targets, grams):**
  - Predicted directly by the model: `Dry_Green_g`, `Dry_Dead_g`, `Dry_Clover_g`
  - Derived by physical identity: **`GDM_g = Green + Clover`**, **`Dry_Total_g = Green + Dead + Clover`**
  - ⚠️ **GDM = Green + Clover** is confirmed (train.csv error = 0.0000). Some raw output files store a
    wrong `GDM_g = Green + Dead`; always use **G+C**. Predicting 3 fractions and *reconstructing* GDM/Total
    enforces the conservation identities by construction.
- **Metric — competition "Global Weighted R²":**
  weights `w = [0.1, 0.1, 0.1, 0.2, 0.5]` for `[Green, Dead, Clover, GDM, Total]`. Flatten all
  (image, target) pairs, weight, then `R² = 1 − SS_res/SS_tot`. **Total (0.5) + GDM (0.2) = 70%** of the
  score. These weights are the **official competition formula** (not chosen by us).

---

## 2. Dataset (CSIRO Image2Biomass, Kaggle competition training split)

- **357 unique top-down quadrat images** (1785 image–target pairs), **4 Australian states** (Tasmania 138,
  Victoria 112, NSW 75, WA 32), 15 sward types, all **2015**. Each 70 cm × 30 cm quadrat was photographed,
  then clipped/separated/oven-dried/weighed for ground truth. (Full published dataset is larger — 1,162
  images / 19 sites — but only this labelled competition split is used; the competition test set is
  unlabelled, 1 image.)
- **Metadata NDVI + canopy height exist but are deliberately NOT used** — the method is **image-only**.
- **Split:** seed 42, `train_test_split(0.20)` then `0.20` again → **64% train / 16% val / 20% test
  (72 images)**. The same 72-image test set and the same 15 bootstrap draws are used for every model.

**Dataset statistics (grams):**

| Variable | Mean | Std | Min | Median | Max | % zero | Skew |
|---|---|---|---|---|---|---|---|
| Dry_Green_g | 26.62 | 25.40 | 0.00 | 20.80 | 157.98 | 5.0% | 1.74 |
| Dry_Dead_g | 12.04 | 12.40 | 0.00 | 7.98 | 83.84 | 11.2% | 1.75 |
| Dry_Clover_g | 6.65 | 12.12 | 0.00 | 1.42 | 71.79 | **37.8%** | **2.83** |
| GDM_g | 33.27 | 24.94 | 1.04 | 27.11 | 157.98 | 0.0% | 1.55 |
| Dry_Total_g | 45.32 | 27.98 | 1.04 | 40.30 | 185.70 | 0.0% | 1.42 |

**Four intrinsic difficulties (all confirmed in data):** heavy right-skew; **zero-inflated clover (37.8%
exact zeros)**; **dead–soil spectral overlap** (brown thatch ≈ brown soil); vertical occlusion (canopy
shades understory).

---

## 3. Method — one identical pipeline, five backbones

**Shared configuration (the "fair comparison" backbone):**
- Input 224×224, batch 16, seed 42 (full determinism).
- **Raw-gram targets** (NOT log1p — log-space training under-predicts high biomass after expm1).
- **Composite loss** `α·MAE + (1−α)·RMSE` with **α = 0.35** (i.e. 65% RMSE weight), true RMSE
  (mean-before-sqrt).
- **Trimmed augmentation:** horizontal+vertical flip, rotation ±0.1, Gaussian noise σ=0.02 **only**
  (no color jitter — greenness IS the signal; no zoom/translation — changes apparent density, not label).
- **Regression head:** CNNs use a **CBAM** (channel+spatial attention) block over conv maps → the shared
  MLP; ViT uses its pooled **[CLS]** token → the same shared MLP. The MLP head is identical across all
  five: `Dense(128, SiLU) → Dropout(0.3) → Dense(3, linear)`. (Only the pooling front-end differs, by
  necessity — "head parity".)
- **BatchNorm frozen** in CNN backbones (training BN on batch-16 destabilized fine-tuning). ViT uses
  LayerNorm.
- **Two-stage, depth-relative fine-tuning:** Stage-1 "main" unfreezes the **top 30%** of backbone layers +
  head; Stage-2 "fine-tune" unfreezes the **top 40%** at 10× lower LR. AdamW, cosine-warmup (main) /
  ReduceLROnPlateau (ft), EarlyStopping restore-best.
- **Preprocessing is baked inside each saved model** (`Rescaling → preprocess_input`), so inference feeds
  `[0,1]` images.

**Per-model differences (everything else identical):**

| Model | Backbone (Keras) | ~Params | ~GFLOPs | Main LR | Best stage |
|---|---|---|---|---|---|
| MobileNet **(V1)** | `tf.keras.applications.MobileNet` | 4.2M | 0.57 | 1e-4 | ft |
| EfficientNet-B0 | `EfficientNetB0` | 5.3M | 0.39 | 1e-4 | main |
| EfficientNet-B3 | `EfficientNetB3` (run @224², not native 300²) | 12.2M | 1.80 | 1e-4 | main |
| DenseNet-121 | `DenseNet121` | 8.0M | 2.90 | 1e-4 | ft |
| ViT-B/16 | `vit_keras.vit.vit_b16` | 86.6M | 17.6 | **5e-5** | ft |

> It is **MobileNet V1** (depthwise-separable, ReLU, no SE/Hard-Swish) — NOT V3. ViT needs `vit-keras`
> (NOT `tensorflow-addons`, which is broken on TF 2.18). "Best stage" = which of the two stages scored
> higher; fine-tuning helps ViT/DenseNet/MobileNet but slightly hurts the EfficientNets (already at ceiling).

### 3.1 Key code (self-contained — reconstruct the pipeline from these)

**Composite loss:**
```python
def composite_loss(alpha=0.35, eps=1e-6):
    def loss(yt, yp):
        e = yt - yp
        mae  = tf.reduce_mean(tf.abs(e))
        rmse = tf.sqrt(tf.reduce_mean(tf.square(e)) + eps)   # TRUE RMSE (mean inside sqrt)
        return alpha * mae + (1.0 - alpha) * rmse
    return loss
```

**Weighted-R² metric + GDM/Total reconstruction (GDM = G+C):**
```python
import numpy as np
W5 = np.array([0.1, 0.1, 0.1, 0.2, 0.5])          # Green, Dead, Clover, GDM, Total
def reconstruct5(b):                               # b = [green, dead, clover]
    g, d, c = b[:, 0], b[:, 1], b[:, 2]
    return np.column_stack([g, d, c, g + c, g + d + c])   # GDM=G+C, Total=G+D+C
def weighted_r2(true3, pred3):
    yt = reconstruct5(true3).flatten(); yp = reconstruct5(pred3).flatten()
    w = np.tile(W5, len(true3)); ybar = np.sum(w * yt) / np.sum(w)
    return 1 - np.sum(w * (yt - yp) ** 2) / np.sum(w * (yt - ybar) ** 2)
```

**Data pipeline (tf.data):**
```python
IMG_SIZE = 224
def preprocess_fn(path, label):
    img = tf.io.read_file(tf.strings.join([IMG_DIR, "/", path]))
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return tf.cast(img, tf.float32) / 255.0, label   # [0,1]; backbone preprocess is inside the model
# trimmed augmentation layers (train only):
#   RandomFlip("horizontal_and_vertical"), RandomRotation(0.1, "reflect"), GaussianNoise(0.02)
```

**Model head (illustrated on ViT; CNNs are identical after a CBAM block over the conv map):**
```python
feat = backbone(x)                        # ViT: pooled [CLS]; CNN: GlobalAvgPool after CBAM
v = layers.Dense(128, activation="silu")(feat)
v = layers.Dropout(0.3)(v)
out = layers.Dense(3, activation="linear")(v)   # [green, dead, clover] in raw grams
```

**Bootstrap 95% CI (Fix A — centered on the loop mean):**
```python
# 15 loops; each loop draws 10 random test images (SAME draws for all models); compute weighted_r2 per loop
loops = [...]                              # 15 weighted-R² values
xbar   = np.mean(loops)                    # CI center = loop mean (Fix A), NOT the full-test R²
sigma  = np.std(loops)
margin = 1.96 * sigma / np.sqrt(15)
CI = (xbar - margin, xbar + margin)        # full-test R² reported separately as the point estimate
```

---

## 4. Results (verified — GDM = G+C throughout)

### 4a. Leaderboard — weighted R² + bootstrap 95% CI
| Model | Best R² | main / ft | 95% CI (Fix A) | σ |
|---|---|---|---|---|
| **ViT-B/16** | **0.7455** | 0.7398 / **0.7455** | [0.681, 0.769] | **0.087** |
| EfficientNet-B0 | 0.7113 | **0.7113** / 0.7076 | [0.554, 0.724] | 0.168 |
| DenseNet-121 | 0.7070 | 0.6675 / **0.7070** | [0.545, 0.700] | 0.153 |
| EfficientNet-B3 | 0.6817 | **0.6817** / 0.6726 | [0.521, 0.710] | 0.186 |
| MobileNet (V1) | 0.6443 | 0.6211 / **0.6443** | [0.531, 0.668] | 0.135 |
| Mean-predictor (baseline) | 0.000 | — | — | — |

**Reading:** All CIs overlap → no pairwise difference is statistically significant on 72 images. ViT has
the highest center **and** lowest σ (tightest band) → clear **best bet**, not proven. Bigger ≠ better
(B0 > B3). All 5 beat the 0.000 baseline by +0.64 to +0.75.

### 4b. Paired-loop significance (same 15 draws)
ViT-B/16 beats **EfficientNet-B0 10/15**, **DenseNet-121 10/15**, **EfficientNet-B3 9/15**,
**MobileNet 13/15** loops (median Δ vs EffB0 = +0.035). → consistent ordering, not proof of significance.

### 4c. Per-target R² (best stage, GDM = G+C)
| Model | Green | Dead | Clover | GDM | Total |
|---|---|---|---|---|---|
| ViT-B/16 | 0.7054 | **0.3315** | 0.5675 | 0.6900 | **0.6524** |
| EfficientNet-B0 | 0.6394 | 0.2534 | 0.5609 | 0.6660 | 0.6005 |
| DenseNet-121 | **0.7288** | 0.2567 | 0.3537 | **0.7049** | 0.5762 |
| EfficientNet-B3 | 0.6685 | 0.2253 | **0.5709** | 0.5935 | 0.5556 |
| MobileNet (V1) | 0.6311 | 0.2550 | 0.4939 | 0.5425 | 0.4998 |

ViT wins Total + Dead (hardest); DenseNet wins Green + GDM. Dead is weak for all (0.22–0.33). **No single
model wins everything.**

### 4d. Per-target error in grams (best stage): RMSE / MAE (GDM = G+C)
| Model | Green | Dead | Clover | GDM | Total |
|---|---|---|---|---|---|
| ViT-B/16 | 11.77/7.28 | **9.83/6.43** | 8.78/5.02 | 12.21/8.05 | **14.23/9.65** |
| EfficientNet-B0 | 13.02/8.56 | 10.39/6.76 | 8.85/5.64 | 12.67/9.08 | 15.25/10.72 |
| DenseNet-121 | **11.29/7.48** | 10.37/6.97 | 10.73/6.05 | **11.91/8.14** | 15.71/11.08 |
| EfficientNet-B3 | 12.49/7.63 | 10.59/6.71 | **8.74/5.24** | 13.98/9.31 | 16.09/10.45 |
| MobileNet (V1) | 13.17/8.63 | 10.38/6.74 | 9.50/5.46 | 14.83/9.99 | 17.07/11.91 |

ViT lowest Total error (14.23 g on μ45.3 ≈ 31% → adequate for coarse ranking, not precise per-quadrat
budgeting). DenseNet lowest Green/GDM error. MobileNet highest Total error (17.07 g).

### 4e. Ablation studies (change ONE factor from FINAL config)
**EfficientNet-B0 (Table 6):** FINAL **0.6965** · log1p 0.2769 (−0.42) · α=0.65 0.6857 · α=0.50 0.6777 ·
frozen backbone 0.4394 (−0.26) · **BN trainable 0.3783 (−0.32)** · **full aug −0.1675 (−0.86)**.
**ViT-B/16 (Table 6b):** FINAL **0.7398** · log1p 0.2313 (−0.51) · α=0.65 0.6981 · α=0.50 0.7277 ·
frozen 0.2765 · **unfreeze 0.15 (original) 0.2886 (−0.45)** · LR 1e-4 0.7127 · **full aug 0.0917 (−0.65)**.
→ **Impact ranking: augmentation ≫ raw-targets > BatchNorm > unfreezing > loss-α.** Both a CNN and a
transformer agree → the choices generalize across architectural families (why we ablate only these two).

### 4f. Explainability (Grad-CAM / SHAP / LIME)
- DenseNet Green Grad-CAM = focused on foliage (why it's Green champion); EffB0 Dead Grad-CAM = scattered
  onto soil (why Dead is hard). ViT attention-rollout = uninformative → **use LIME/SHAP for ViT**.
- **Quantitative XAI metric (honest NULL result):** % of Grad-CAM attention on vegetation ≈ 90% for all
  CNNs — but vegetation *covers* ≈ 90% of the canopy, so this = chance (vs baseline ≈ 0). The coverage
  metric **cannot discriminate** models on dense canopy; reported transparently. Deletion/insertion
  faithfulness curves = future work.

---

## 5. Key engineering decisions (the journey — shaped the results)

1. **Composite-loss RMSE bug** — original `sqrt(error²+eps)` per element = MAE; fixed to mean-before-sqrt.
2. **Removed log1p** — log-space training under-predicts high biomass → raw targets (MobileNet 0.49→0.63).
3. **α = 0.35** (65% RMSE) — was ignoring the config alpha.
4. **BatchNorm frozen** — BN on batch-16 blew predictions to 211 g.
5. **Depth-relative unfreezing** (30%/40%) — fixed layer counts under-trained EfficientNet.
6. **ViT retune** — 0.15 unfreeze / 3e-5 gave 0.37; 0.30 / 5e-5 gave 0.74 (ViT was under-trained, not weak).
7. **Trimmed augmentation** — color jitter/zoom are catastrophic (ablation −0.65/−0.86).
8. **Fine-tune is backbone-dependent** — helps MobileNet/DenseNet/ViT, hurts EfficientNets.
*(Every one of these is now quantitatively justified by the ablation tables above.)*

---

## 6. The research paper (written, revised, number-verified)

**Title:** *"Tokens vs. Textures: An Architectural Benchmark and Ablation Study of Deep Vision Models for
In-Situ Pasture Biomass Estimation."*
**Authors:** MD Rahat Al Mamun (Dept. of EEE); MD Rayhan Hossain, Sumaiya Islam, Samiul Haque (Dept. of
CSE) — **Independent University, Bangladesh (IUB), Dhaka.**
**Scope (supervisor directive):** **NO architecture novelty** → an architectural **benchmark + ablation
study**; **single split** (not k-fold), bootstrap CI; real results only.

**Section map:** Abstract · I Intro (Research Questions; *empirical* Contributions) · II Related Work
(incl. ViT-vs-CNN in remote sensing) · III Dataset · IV Methodology (backbones; CBAM head-parity;
composite loss; two-stage FT) · V Setup (single-split+bootstrap justified; augmentation; competition
metric; **Reproducibility**) · VI Results (leaderboard+baseline; **paired significance**; per-target R²;
**per-target RMSE/MAE**; winner-per-axis) · VII Diagnostics (7.1 mechanistic "informed speculation"
banner; 7.2–7.6 per-model, table-anchored; **7.7 ablation ×2 models**) · VIII Explainability (honest
null) · IX Deployment (**recommended default = EfficientNet-B0**; 3 tiers; Pareto) · X Limitations ·
XI Conclusion · References [1]–[26] (DOIs verified; Trotter = 10.1071/CP10019).

**Deployment guidance:** Pareto-optimal = EfficientNet-B0 (0.7113 @ 0.39 GFLOPs) and ViT-B/16 (0.7455 @
17.6 GFLOPs, ~45×). **Default recommendation: EfficientNet-B0** (within 0.034 of best, statistically
indistinguishable, runs on edge). Escalate to ViT only when accuracy is paramount and compute is free;
MobileNet only under a hard model-size cap.

**Reviewer objections already pre-empted in the text:** novelty (framed as empirical), metric weights
(official competition formula), single split (comparability + cost + honesty), ablation on 2 of 5
(cross-family by design), XAI null (reported honestly), backbone choice (clean CNN-vs-ViT axis).

**Quality:** independently rated **8.7/10**; full 82-check numeric consistency pass vs source files = 0
real mismatches (one bug found & fixed: MobileNet Total RMSE 15.71 → **17.07 g**; 15.71 was DenseNet's).

**Deliverables that exist:** `paper_v2.md` (markdown draft) and `paper_v2_IEEE_conference.docx` (IEEE
two-column Word render). An `assets/` archive holds every figure and table's source data, renamed by
Fig/Table number.

---

## 7. Limitations (honest)

- **Small test set (72 imgs) → all CIs overlap** — ViT's lead is consistent, not statistically proven.
- **Dead matter ceiling** — dead–soil spectral overlap caps every model at Dead R² ≤ 0.33.
- **Single-year (2015) data** — seasonal/annual generalization untested.
- **Monocular occlusion** — top-down RGB can't see understory biomass.
- **XAI-metric saturation** — vegetation coverage ≈ 90% makes the attention-coverage metric uninformative.
- **No method novelty** — deliberate (supervisor-directed benchmark scope).

---

## 8. Next steps

**To finish the paper for submission:** (1) pick a venue (candidates: IEEE Access, DICTA, MDPI Remote
Sensing, AgriVision) → its template; (2) **LaTeX two-column typeset** (the `.docx` equations are readable
text, not typeset); (3) confirm author order + corresponding author; (4) optional GitHub/Zenodo DOI link
(currently "available on request").

**Deferred future novelty (a separate follow-up paper, NOT this one):** zero-inflated/hurdle head for the
37.8%-zero clover; dead-vs-soil texture disambiguation; image+metadata (NDVI/height) fusion;
uncertainty-aware regression; k-fold CV for added rigor.

---

## 9. Quick-reference constants (for reproduction)

```
SEED=42 · IMG_SIZE=224 · BATCH=16 · USE_LOG_TARGETS=False
LOSS: alpha=0.35 (0.35*MAE + 0.65*trueRMSE) · AUG: flip+rot(0.1)+noise(0.02)
UNFREEZE: main top-30% / finetune top-40% · BatchNorm frozen (CNNs)
LR: 1e-4 main (5e-5 for ViT) → 1e-5 finetune · AdamW · EarlyStopping restore-best
TARGETS: predict [green,dead,clover]; GDM=green+clover; Total=green+dead+clover
METRIC: weighted R², w=[0.1,0.1,0.1,0.2,0.5] over [green,dead,clover,GDM,total]
SPLIT: 64/16/20 (seed 42) → 72-image test · CI: 15 loops × 10 imgs, Fix A (loop-mean centered)
FRAMEWORK: TensorFlow/Keras 2.18, Python 3.11, single P100 GPU; ViT needs vit-keras (not tf-addons)
```

---
*This file is self-contained: it embeds the problem, data, full method + code, every result table, the
paper status, limitations, and next steps. No other file is required to gain full project context.*
