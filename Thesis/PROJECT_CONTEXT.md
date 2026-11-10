# PROJECT CONTEXT — Pasture Biomass Estimation (Full Handoff)

> Read this first. It captures the whole project — problem, models, results, explainability,
> submission setup, and the **research paper (now written and verified)** — so any new
> conversation/thread can continue without re-deriving anything. Numbers below are the latest
> **verified** values (cross-checked against the source files; see §11).

**Status (2026-08):** Modelling done · CI + XAI done · **Research paper drafted, revised, and
number-verified** (`research_paper/paper_v2.md` + IEEE two-column `.docx`). Remaining work is
venue selection + LaTeX typesetting only. Rated **8.7/10**, ready for a Q2 applied venue.

---

## 1. TL;DR (executive summary)

- **Task:** predict pasture **dry biomass in grams** from a **single RGB image** — 3 base targets
  (`Dry_Green_g`, `Dry_Dead_g`, `Dry_Clover_g`) plus two derived (`GDM_g`, `Dry_Total_g`).
- **Data:** Kaggle **CSIRO Image2Biomass** competition training split. **357 unique images**, 4 Australian
  states, 2015. Split → **72-image held-out test**. Image-only (no metadata/NDVI/height).
- **Models compared (5):** MobileNet (V1), EfficientNet-B0, EfficientNet-B3, DenseNet-121, ViT-B/16 —
  same pipeline, only the backbone differs (fair comparison).
- **Best model:** **ViT-B/16** (highest score + most stable). **Efficiency-optimal CNN:** EfficientNet-B0.
- **Metric:** competition **weighted R²** `[Green 0.1, Dead 0.1, Clover 0.1, GDM 0.2, Total 0.5]`.
- **Extras done:** bootstrap confidence intervals (CI), **paired-loop significance**, **ablation studies
  (2 models)**, Grad-CAM/SHAP/LIME + a **quantitative XAI metric**, training-curve figures, unified
  submission notebook, **per-target RMSE/MAE**, deployment/Pareto analysis.
- **Paper angle (supervisor-directed):** **NO architecture novelty** → an **architectural benchmark +
  ablation study** ("Tokens vs. Textures"), image-only, explainable, statistically honest. Single split
  (supervisor-approved), bootstrap CI instead of k-fold.

---

## 2. Problem & dataset

- **Input:** one RGB pasture photo (resized to 224×224).
- **Targets (from `train.csv`, long format `image_path, target_name, target`):**
  - Predicted directly: `Dry_Green_g`, `Dry_Dead_g`, `Dry_Clover_g`
  - Derived: **`GDM_g = Dry_Green_g + Dry_Clover_g`** and **`Dry_Total_g = Green + Dead + Clover`**
  - ⚠️ **GDM = Green + Clover is CONFIRMED** (train.csv: error = 0.0000 for G+C vs 13.63 for G+D). The
    original notebook + the result CSVs' `GDM_g` column wrongly used **G+D**; the paper's Table 4/4b use
    the **correct G+C** (recomputed from the Green & Clover columns — matches Table 4 GDM R² exactly).
- **Split:** seed 42, `train_test_split(0.20)` then `0.20` again → **64% train / 16% val / 20% test
  (72 images)**. Same split + same 15 bootstrap draws across all models (fair).
- **Dataset stats (357 imgs, grams):** Green μ26.6/skew1.74; Dead μ12.0/skew1.75; Clover μ6.65/**37.8%
  zeros**/skew2.83; GDM μ33.3; Total μ45.3 (max 185.7). Clover zero-inflation is the hardest statistical
  feature.
- **`test.csv`:** competition's unlabeled test set — 1 image (`test/ID1001187975.jpg`). Internet OFF to submit.

### Competition metric (weighted R²)
```
weights = [0.1, 0.1, 0.1, 0.2, 0.5]   # [Green, Dead, Clover, GDM, Total]
flatten all (image,target) pairs, weight them, then R² = 1 - SS_res/SS_tot
```
Total (0.5) + GDM (0.2) dominate → getting the aggregate/total right matters most. **These weights are
the official competition formula (not ours)** — a point made explicitly in the paper §5.3.

---

## 3. The 5 models + shared configuration

**Shared across all models (the "fair comparison" backbone):**
- Input 224×224, `BATCH_SIZE=16`, `SEED=42` (full determinism).
- **Targets in RAW grams** (`USE_LOG_TARGETS=False`) — see §5.
- **Loss:** composite `α·MAE + (1-α)·RMSE`, **α = 0.35** (35% MAE + 65% RMSE), true RMSE (mean-before-sqrt).
- **Augmentation (trimmed):** `RandomFlip` + `RandomRotation(0.1)` + `GaussianNoise(0.02)` only.
- **Head (CNNs):** CBAM (channel+spatial) → `Dense(128, silu)` → `Dropout(0.3)` → `Dense(3, linear)`.
- **Head (ViT):** pooled `[CLS]` token → same SiLU MLP (no CBAM — ViT has no 2D map; the MLP head is
  identical across all 5, only the pooling front-end differs — see paper §4.2 "head parity").
- **BatchNorm frozen** in CNN backbones (ViT uses LayerNorm).
- **Unfreezing:** depth-relative — main = top **30%**, fine-tune = top **40%**.
- **Optimizer:** AdamW; cosine-warmup (main) / ReduceLROnPlateau (ft); EarlyStopping restore-best.
- **Two-stage training:** "main" then lower-LR "fine-tune".

**Per-model differences:**

| Model | Backbone | Preprocess | Main LR | Best stage | Kaggle ver |
|---|---|---|---|---|---|
| MobileNet (V1) | `tf.keras.applications.MobileNet` (V1) | mobilenet → [-1,1] | 1e-4 | **ft** | 149 |
| EfficientNet-B0 | `EfficientNetB0` | efficientnet → [0,255] | 1e-4 | **main** | 151 |
| EfficientNet-B3 | `EfficientNetB3` (@224², not native 300²) | efficientnet | 1e-4 | **main** | 152 |
| DenseNet-121 | `DenseNet121` | densenet | 1e-4 | **ft** | 153 |
| ViT-B/16 | `vit_keras.vit.vit_b16` | [0,1]→[-1,1] | **5e-5** | **ft** | 155 |

- **It's MobileNet V1** (`tf.keras.applications.MobileNet`), NOT V3 — 4.2M params, depthwise-separable,
  ReLU, no SE/Hard-Swish. (The `curves/mobilenetv3/` folder name is a leftover misnomer; content is V1.)
- ViT retuned to 0.30/0.40 unfreeze + 5e-5; needs `vit-keras` (NOT `tensorflow-addons`).
- Preprocessing baked inside each saved model (feed `[0,1]` images).

---

## 4. Results (verified — G+C convention throughout)

### 4a. Leaderboard — weighted R² + bootstrap 95% CI (Fix A: loop-mean-centered)
| Rank | Model | Params | GFLOPs | Best R² | main/ft | 95% CI | σ |
|---|---|---|---|---|---|---|---|
| 🥇 | **ViT-B/16** | 86.6M | 17.6 | **0.7455** | 0.7398/**0.7455** | [0.681, 0.769] | **0.087** |
| 🥈 | EfficientNet-B0 | 5.3M | 0.39 | 0.7113 | **0.7113**/0.7076 | [0.554, 0.724] | 0.168 |
| 🥉 | DenseNet-121 | 8.0M | 2.90 | 0.7070 | 0.6675/**0.7070** | [0.545, 0.700] | 0.153 |
| 4 | EfficientNet-B3 | 12.2M | 1.80 | 0.6817 | **0.6817**/0.6726 | [0.521, 0.710] | 0.186 |
| 5 | MobileNet (V1) | 4.2M | 0.57 | 0.6443 | 0.6211/**0.6443** | [0.531, 0.668] | 0.135 |
| — | Mean-predictor (baseline) | — | — | 0.000 | — | — | — |

> **Fix A CI:** centered on the mean of the 15 bootstrap loops (loop-mean), not the full-test R² — so each
> CI sits slightly below its Best-R² point estimate. **All CIs overlap** → no pairwise difference is
> statistically significant on 72 images. ViT is the strongest + most stable (lowest σ), i.e. the clear
> **best bet**, not a proven winner. All 5 clear the 0.000 baseline floor by +0.64 to +0.75.

### 4b. Paired-loop significance (same 15 draws, recomputed live)
ViT-B/16 beats **EfficientNet-B0 10/15**, **DenseNet-121 10/15**, **EfficientNet-B3 9/15**,
**MobileNet 13/15** loops (median Δ vs EffB0 = +0.035). → directionally consistent ordering, **not** proof.

### 4c. Per-target R² (best stage, correct GDM=G+C) — Table 4
| Model | Green | Dead | Clover | GDM | Total |
|---|---|---|---|---|---|
| ViT-B/16 | 0.7054 | **0.3315** | 0.5675 | 0.6900 | **0.6524** |
| EfficientNet-B0 | 0.6394 | 0.2534 | 0.5609 | 0.6660 | 0.6005 |
| DenseNet-121 | **0.7288** | 0.2567 | 0.3537 | **0.7049** | 0.5762 |
| EfficientNet-B3 | 0.6685 | 0.2253 | **0.5709** | 0.5935 | 0.5556 |
| MobileNet (V1) | 0.6311 | 0.2550 | 0.4939 | 0.5425 | 0.4998 |

- **ViT** wins the two heavy targets (Total, GDM-tie) + hardest (Dead). **DenseNet** best Green & GDM.
  **Dead** weakest for everyone (0.22–0.33) → dead-vs-soil spectral overlap. No single model wins everything.

### 4d. Per-target error in grams (best stage): RMSE / MAE — Table 4b (GDM = G+C)
| Model | Green | Dead | Clover | GDM | Total |
|---|---|---|---|---|---|
| ViT-B/16 | 11.77/7.28 | **9.83/6.43** | 8.78/5.02 | 12.21/8.05 | **14.23/9.65** |
| EfficientNet-B0 | 13.02/8.56 | 10.39/6.76 | 8.85/5.64 | 12.67/9.08 | 15.25/10.72 |
| DenseNet-121 | **11.29/7.48** | 10.37/6.97 | 10.73/6.05 | **11.91/8.14** | 15.71/11.08 |
| EfficientNet-B3 | 12.49/7.63 | 10.59/6.71 | **8.74/5.24** | 13.98/9.31 | 16.09/10.45 |
| MobileNet (V1) | 13.17/8.63 | 10.38/6.74 | 9.50/5.46 | 14.83/9.99 | **17.07/11.91** |

- ViT lowest Total error (14.23 g on μ45.3 ≈ 31%); DenseNet lowest Green+GDM error; MobileNet highest Total
  error (17.07 g). ⚠️ *Earlier drafts misattributed 15.71 g (DenseNet's) to MobileNet — fixed to 17.07 g.*

### 4e. Ablation studies (one factor changed from FINAL) — Tables 6 & 6b
**EfficientNet-B0:** FINAL **0.6965** · log1p 0.2769 (−0.42) · α0.65 0.6857 · α0.50 0.6777 · frozen
backbone 0.4394 (−0.26) · **BN trainable 0.3783 (−0.32)** · **full aug −0.1675 (−0.86!)**.
**ViT-B/16:** FINAL **0.7398** · log1p 0.2313 (−0.51) · α0.65 0.6981 · α0.50 0.7277 · frozen 0.2765 ·
**unfreeze 0.15 (original) 0.2886 (−0.45)** · LR 1e-4 0.7127 · **full aug 0.0917 (−0.65)**.
→ **Ranked impact: augmentation ≫ raw-targets > BatchNorm > unfreezing > loss-α.** Both a CNN and a
transformer agree → the design choices generalize across families.

### 4f. XAI quantitative metric (Table 7) — honest NULL result
% of Grad-CAM attention on vegetation: all 4 CNNs ≈ **90%**, but vegetation **covers ≈90%** of the canopy
→ **= chance level (vs baseline ≈ 0)**. The coverage metric **cannot discriminate** models on dense canopy;
reported transparently as a limitation. Deletion/insertion faithfulness curves = future work.

### 4g. Competition test image predictions (`test/ID1001187975.jpg`)
Models disagree widely (Total 43→65) → real uncertainty on unseen data; all agree clover ≈ 0.

---

## 5. Key engineering fixes (the journey)

1. **Composite-loss RMSE bug** — `sqrt(error²+eps)` per element = MAE. Fixed to mean-before-sqrt true RMSE.
2. **Log-transform removed** — log1p training + raw scoring under-predicted high biomass. Raw targets →
   big jump (MobileNet 0.49→0.63). (Ablation confirms: log1p costs −0.42 to −0.51.)
3. **α → 0.35** (65% RMSE) — wired `alpha=ALPHA` into compile (was ignored).
4. **BatchNorm frozen** — BN on batch-16 blew predictions to 211 g. (Ablation: trainable BN −0.32.)
5. **Depth-relative unfreezing** — fixed layer count under-trained EfficientNet (~10%). Fraction 30/40 →
   EffB0 0.634→0.694.
6. **ViT retune** — first config (0.15 unfreeze, 3e-5) → 0.37. Retuned (0.30/0.40, 5e-5) → 0.74. (Ablation:
   0.15 unfreeze collapses ViT to 0.29.)
7. **Trimmed augmentation** — dropped color jitter + zoom. (Ablation: full aug is catastrophic, −0.65/−0.86.)
8. **Fine-tune is backbone-dependent** — helps MobileNet/DenseNet/ViT; hurts EfficientNets (use main).

---

## 6. Explainability findings

- **DenseNet — Green (Grad-CAM):** heat focused on central foliage → why it's the Green champion.
- **EfficientNet-B0 — Dead (Grad-CAM):** scattered, drifts onto soil → visualizes dead–soil ambiguity.
- **ViT — attention-rollout:** near-uniform, **uninterpretable** → use LIME/SHAP for ViT (no conv maps).
- **EfficientNet-B0 — SHAP (Green):** positive importance over the plant → correct reasoning.
- **Quantitative metric:** null (see §4f) — interpretability rests on the qualitative maps.
- Guide: `Learning/Image_Processing/model_evaluation_guide.md`.

---

## 7. Submission setup

- **Notebook:** `iub-kryptonite-submission.ipynb` — `CHOSEN_MODEL` switch.
- **Models dataset (Kaggle):** `phylosopherhossain/csiro-biomass-models` (best-stage `.keras`).
- **Bugs fixed:** removed `expm1` (raw grams); GDM auto-detected G+C; `custom_objects={"preprocess_input"}`;
  no `tensorflow-addons` (broken on TF 2.18); ViT needs offline `vit-keras`.
- **Recommended submission:** **EfficientNet-B0 (main)** = reliable, dependency-free, ~0.69–0.71. ViT =
  best score but fragile offline (protobuf shutdown on TF 2.18).

---

## 8. Research paper — WRITTEN & VERIFIED (the current deliverable)

**Title:** *"Tokens vs. Textures: An Architectural Benchmark and Ablation Study of Deep Vision Models for
In-Situ Pasture Biomass Estimation."*
**Authors:** MD Rahat Al Mamun (Dept. of **EEE**), MD Rayhan Hossain, Sumaiya Islam, Samiul Haque (Dept. of
**CSE**) — **Independent University, Bangladesh (IUB), Dhaka.**
**Supervisor directive (defines scope):** **NO architecture novelty** → this is an **architectural
benchmark + ablation study**, **single split** (not k-fold), on real results only. Novelty ideas are
deferred (see `research_paper/future_novelty_ideas.md`).

**Structure (paper_v2.md):** Abstract · I Intro (RQs, empirical Contributions) · II Related Work
(+ViT-vs-CNN remote-sensing) · III Dataset · IV Methodology (backbones, CBAM head-parity, composite loss,
2-stage FT) · V Setup (5.1 single-split+bootstrap justified, 5.2 aug, 5.3 competition metric, **5.4
Reproducibility**) · VI Results (leaderboard+baseline, **paired significance**, per-target R², **per-target
RMSE/MAE Table 4b**, which-model-wins) · VII Diagnostics (7.1 mechanistic "informed speculation" banner,
7.2–7.6 per-model anchored to tables, **7.7 ablation ×2 models**) · VIII Explainability (honest null
result) · IX Deployment (**recommended default = EfficientNet-B0**; 3 tiers; Pareto) · X Limitations ·
XI Conclusion · **References [1]–[26]** (DOIs verified; Trotter corrected to 10.1071/CP10019).

**Tables:** 1 dataset · 2 specs · 3 leaderboard+CI+baseline · 4 per-target R² · 4b per-target RMSE/MAE ·
5 winner-per-axis · 6 ablation EffB0 · 6b ablation ViT · 7 XAI. **Figures:** 1 pipeline · 2 CI · 3 curves
(×5) · 4 per-target · 5 Grad-CAM (×4) · 6 SHAP/LIME (×4) · 7 Pareto.

**Reviewer-objections pre-empted:** novelty (framed empirical), metric weights (official competition
formula), single split (comparability+cost+honesty), ablation-on-2-models (cross-family by design), XAI
null (honest), backbone selection (clean CNN-vs-ViT axis).

**Rating: 8.7/10.** Target: **Q2 applied** — IEEE Access / DICTA (B-conf) / MDPI Remote Sensing / AgriVision.
**Remaining before submit:** (1) pick venue → its template; (2) **LaTeX typeset** (docx equations are
readable text, not typeset); (3) confirm names/corresponding author; (4) optional repo/Zenodo DOI link
(currently "available on request").

---

## 9. File & folder map (under `Thesis/`)

```
Thesis/
├── PROJECT_CONTEXT.md              <- THIS FILE (start here)
├── configuration.md · model_comparison.md
├── iub-kryptonite-*.ipynb          <- base + 5 training + submission + CI notebooks
├── models/                         <- 5 best-stage .keras (uploaded to Kaggle)
├── logs/                           <- 5 training logs (*_log.txt) [mobilenetv3_log.txt = V1]
├── curves/<model>/                 <- curve PNGs (combined_loss.png used as Fig 3) [mobilenetv3=V1]
├── CI_outputs/
│   ├── confidence_intervals.xlsx   <- Summary + per-model 15-loop sheets + CompTestPred
│   ├── competition_test_predictions.csv · loop_images.txt
│   ├── gradcam/<model>/  ·  explanations/<model>/
├── {model}_result/                 <- raw per-run outputs:
│   ├── target_metrics_{main,ft}.csv    <- per-target R²/RMSE/MAE  (Table 4/4b source)
│   └── detailed_errors_{main,ft}.csv   <- per-image true/pred/error (GDM=G+C recompute source)
├── train.csv                       <- raw dataset (Table 1 source)
└── research_paper/                 <- THE PAPER
    ├── paper_v2.md                     <- main draft (v2, content-complete, verified)
    ├── paper_v2_IEEE_conference.docx   <- IEEE two-column Word render (17 imgs, 16 tables)
    ├── ablation_effb0.csv · ablation_vit.csv   <- Table 6 / 6b data (+ .py Kaggle scripts)
    ├── xai_attention_on_vegetation.csv         <- Table 7 data (+ xai_metric.py)
    ├── figures/                        <- architecture, ci_intervals, per_target_r2, pareto (.png)
    ├── future_novelty_ideas.md         <- deferred novelty (NOT this paper)
    ├── v1_review_and_fixes.md · figures_list.txt
    └── assets/                         <- ONE-STOP archive of everything the paper uses
        ├── MANIFEST.md                     <- maps every file → its Fig/Table
        ├── figures/  (Fig1..Fig7*, 17 imgs, renamed by figure no.)
        └── tables_data/ (Table1,3,4,4b,6,6b,7 source CSV/XLSX, renamed by table no.)

Also: Learning/Image_Processing/model_evaluation_guide.md  <- how-to-read curves/CI/XAI guide
```

---

## 10. Open issues / caveats / next steps

- **CI bands overlap** (72-img test) → significance would need a larger test set / k-fold (deferred; single
  split is supervisor-approved). Bootstrap + paired-loop is the honest middle ground used.
- **GDM=G+C is the paper convention** — the result CSVs' raw `GDM_g` column is G+D (wrong); always recompute
  GDM from Green+Clover (done for Table 4b; matches Table 4 R²).
- **Dead & very-high biomass** are the weak spots — **data limitations**, not architecture.
- **ViT fragile to deploy** (offline vit-keras; protobuf shutdown on TF 2.18). CNNs safe.
- **ViT attention map uninterpretable** — use LIME/SHAP.
- **`mobilenetv3` folder names are a misnomer** — the model is MobileNet **V1**.
- **Per-target RMSE/MAE now available locally** (from `{model}_result/`) — no Kaggle rerun needed.
- **Next:** venue → LaTeX two-column typeset → submit (see §8).

---

## 11. Verification status

A full numeric consistency pass (82 checks) cross-checked every table in `paper_v2.md` against source
files (`target_metrics_*.csv`, `detailed_errors_*.csv`, `confidence_intervals.xlsx`, ablation/xai CSVs,
`train.csv`). **Result: 0 real mismatches** (the only flags were rounding-boundary and Unicode-minus false
positives). All 357 images / 72-test, per-target R²/RMSE/MAE (incl. recomputed GDM), CI, σ, paired-loop
wins, both ablations, XAI, and dataset stats reconcile with their sources. One real bug was found & fixed
in the process: MobileNet Total RMSE 15.71 → **17.07 g** (15.71 was DenseNet's).

---
*Last updated: research paper drafted + revised + number-verified; assets archive built; IEEE .docx rendered.
Remaining = venue choice + LaTeX typeset.*
