# Tokens vs. Textures: An Architectural Benchmark and Ablation Study of Deep Vision Models for In-Situ Pasture Biomass Estimation

*MD Rahat Al Mamun\*, MD Rayhan Hossain, Sumaiya Islam, and Samiul Haque*
*Department of Computer Science & Engineering; \*Department of Electrical & Electronic Engineering*
*Independent University, Bangladesh (IUB), Dhaka, Bangladesh*
*Correspondence: research.pasture.biomass@gmail.com*

> **Working draft v2 (single-column Markdown).** All methods and numbers match the actual
> experiments (`Thesis/CI_outputs/`, result folders, `train.csv`). Format to IEEE two-column later.
> Figure paths point to real files. All tables/figures are filled; the ablation (Table 6) covers EfficientNet-B0 and ViT-B/16.

---

## Abstract

Accurate, non-destructive estimation of pasture biomass and its botanical fractions underpins
precision livestock feeding, rotational-grazing optimization, and grassland carbon management.
Ground-level RGB imagery offers a low-cost, sensor-free alternative to destructive harvesting and
saturation-prone satellite indices, but poses hard regression challenges: heavy right-skew, a
zero-inflated clover component (37.8% true zeros), and spectral confusion between senescent dead
matter and bare soil. We present, to our knowledge, the **first explainable and statistically-validated
architectural benchmark and ablation study** of five deep vision backbones — **MobileNet (V1)**, **EfficientNet-B0**, **EfficientNet-B3**,
**DenseNet-121**, and the **Vision Transformer (ViT-B/16)** — on the multi-site CSIRO Image2Biomass
dataset (357 quadrat images, 4 Australian states, 2015). All models share an identical pipeline on
**raw-gram targets**: an attention-based regression head (**CBAM channel-and-spatial attention** for the
CNNs, pooled **[CLS]** token for ViT), a **composite loss (0.35·MAE + 0.65·RMSE_ε)** chosen to protect
high-biomass fidelity, and a two-stage depth-relative fine-tuning protocol. Using a single held-out split with **15-loop bootstrap 95%
confidence intervals**, ViT-B/16 attains the highest weighted-R² (**0.7455**; bootstrap 95% CI [0.681, 0.769])
and the **lowest variance** (σ = 0.087) — though, as we report up front, **all model confidence intervals
overlap**, so this lead is consistent and practically preferable rather than statistically separable on
this small dataset. ViT leads every architecture on the two highest-weighted targets (Total, GDM) and on
the notoriously hard dead-matter target (R² = 0.33 vs 0.23–0.26 for CNNs).
DenseNet-121 is the best green-grass extractor (R² = 0.729) and EfficientNet-B0 is the
efficiency-optimal choice (0.7113 at 0.39 GFLOPs). We validate biological grounding with Grad-CAM,
SHAP, and LIME, and translate the findings into a three-tier deployment guide.

**Index Terms —** Precision Agriculture, Pasture Biomass, Vision Transformer, Architectural
Benchmark, Multi-Target Regression, Confidence Intervals, Explainable AI, CBAM.

---

## I. Introduction

Grasslands cover the majority of the world's agricultural land and sustain ruminant meat, milk, and
wool production. Accurate above-ground biomass estimation drives stocking-rate decisions,
rotational-grazing timing, and feed budgeting; improved allocation raises gross margins by
$52–96/ha [1], [2]. Traditional in-situ measurement — clipping a quadrat, botanically separating
green/dead/clover, oven-drying, and weighing — yields ground truth but is destructive and
non-scalable [3]. Satellite and UAV indices scale spatially but saturate at Leaf Area Index > 3 and
are vulnerable to cloud and calibration error [4], [5]. Even ground-level classical pipelines that
regress handcrafted vegetation indices (NDVI, ExG, CIVE) on optical/capacitance sensors degrade under
variable illumination and dense multi-species canopies [3], [6], and cannot separate the botanical
fractions (green vs. dead vs. clover) that grazing decisions actually depend on. This ceiling on
index-based regression is what motivates learned, structure-aware visual representations.

Ground-level top-down RGB photography enables non-destructive, low-cost estimation, but framing it
as regression exposes four difficulties, all confirmed in our data (Table 1):

1. **Heavy right-skew.** Green, dead, and clover components are right-skewed (skew 1.7–2.8); a small
   number of overgrown swards (Total up to 185.7 g) dominates squared-error gradients.
2. **Zero-inflated clover.** Clover is 37.8% exact zeros and the most skewed component (2.83).
3. **Dead–soil spectral overlap.** Senescent thatch shares brown reflectance with dry soil.
4. **Vertical occlusion.** Erect green canopies shade understory clover and mulch.

Prior agricultural vision work targets classification, segmentation, or single-variable yield
[8]–[13]; a controlled benchmark isolating **architectural inductive biases** for *simultaneous
multi-component* biomass regression is missing. Notably, the CSIRO dataset ships NDVI and canopy-height
metadata, but we deliberately restrict inputs to the **image alone**, to test what visual
architecture can achieve on commodity photographs.

### 1.1 Research Questions
- **RQ1 (Locality vs. Global Attention):** Do CNN locality priors or ViT global self-attention better
  regress non-contiguous botanical fractions from canopy imagery?
- **RQ2 (Representation Sharing):** How do backbones trade off across green/dead/clover/GDM/total, and
  which minimizes cross-target interference?
- **RQ3 (Accuracy vs. Cost):** What is the accuracy–compute frontier across lightweight edge and
  high-capacity server backbones?

### 1.2 Contributions
This work is deliberately **empirical rather than methodological**: we contribute no new architecture or
loss, but a controlled, honestly-validated benchmark of existing ones — the evidence practitioners need
to choose a backbone for this task. To our knowledge this is the **first benchmark that pairs a
controlled CNN-vs-transformer comparison for multi-component biomass with both statistical (bootstrap
CI) and interpretability (XAI) validation.** Our four empirical contributions are:
1. **Controlled 5-architecture benchmark** on CSIRO Image2Biomass under one identical pipeline
   (raw-gram targets, CBAM head, composite loss, depth-relative two-stage fine-tuning; seed 42) — the
   first apples-to-apples locality-vs-attention comparison for joint green/dead/clover regression.
2. **Statistical rigor:** 15-loop bootstrap 95% CIs; we report overlap honestly rather than claiming
   unsupported significance — a discipline largely absent from prior single-number agricultural leaderboards.
3. **Interpretability audit:** Grad-CAM (CNNs) and model-agnostic SHAP/LIME (all models, incl. ViT),
   including an honestly-reported *negative* result on a naive attention-coverage metric (§8) that
   future work should replace with faithfulness curves.
4. **Ablation + deployment guidance:** a two-model ablation isolating each design choice (§7.7) and an
   accuracy–compute Pareto view mapped to concrete operational tiers with a single recommended default.

We frame these as *empirical* contributions to pre-empt the expectation of an algorithmic novelty:
the value here is a rigorous, reproducible answer to "which vision architecture, and why," not a new model.

---

## II. Related Work

**Ground-level agricultural vision.** Early biomass work used handcrafted indices (NDVI, ExG, CIVE)
with classical regressors and optical/capacitance sensors [3], [5], [6], which fail under variable
illumination and dense multi-species canopies; early neural networks estimated biomass from satellite
imagery [7]. Skovsen et al. released GrassClover for grass–clover segmentation [8]; Sapkota et al. used
synthetic imagery for weed biomass [9]; Albert et al. explored domain adaptation for herbage [10].
These target segmentation or single-variable regression, not joint multi-component dry-matter.

**Deep architectures.** Lightweight CNNs (MobileNet [11], EfficientNet [12]) trade accuracy for
efficiency via depthwise-separable and compound-scaled convolutions; DenseNet [13] maximizes feature
reuse via dense connectivity; ResNet-style deep nets established image recognition [22]; and ViT [15]
replaces convolution with all-to-all self-attention. While deep image-to-image and image regression are
studied [16], the *relative* behavior of these inductive biases on continuous multi-target agricultural
regression is unestablished, and no prior study pairs such a comparison with statistical confidence
intervals and interpretability (XAI) validation — the gap this paper addresses.

**ViT vs. CNN in remote sensing and agriculture.** Transformers have been adopted rapidly for
earth-observation imagery: ViTs match or beat CNNs on remote-sensing scene classification [25], and a
recent survey catalogues their spread across detection, segmentation, and change tasks [26]. Yet these
studies are overwhelmingly *classification/segmentation* on curated overhead imagery; the token-vs-texture
trade-off for *continuous, ground-level, multi-component biomass regression* on small field datasets is
untested. Recent architectural analyses further confirm that ViTs and CNNs learn fundamentally different
representations [23] and that ViTs under-perform on small datasets without added inductive bias [24] —
exactly the small-data, fine-grained regime we probe here, and the reason a careful controlled comparison
(not a leaderboard transplant from ImageNet-scale classification) is warranted.

**Loss formulations.** L2 is outlier-sensitive; L1 is robust but non-smooth at zero; Huber/Smooth-L1
interpolate with a tuned threshold [21]. For heavy-tailed multi-target biomass we adopt a composite
MAE+RMSE loss and, unlike prior work, **weight it toward RMSE** to preserve high-biomass fidelity.

---

## III. Dataset & Problem Formulation

### 3.1 CSIRO Image2Biomass
We use the **Kaggle competition training split** of the CSIRO Image2Biomass dataset [2] — **357
top-down quadrat images** (1785 image–target pairs) across **four Australian states** (Tasmania 138,
Victoria 112, New South Wales 75, Western Australia 32), spanning **15 sward/species types**, all in
**2015 (15 Jan – 10 Nov)**. (The full published dataset [2] comprises 1,162 images across 19 sites; the
held-out competition test images are not publicly labelled.) Each 70 cm × 30 cm
quadrat was photographed, then destructively clipped, botanically separated, oven-dried, and weighed
for ground-truth grams. The dataset also provides `Pre_GSHH_NDVI` and `Height_Ave_cm` metadata, which
we intentionally do not use — our method is image-only.

**TABLE 1 — Dataset statistics (full set, grams).**

| Variable | Mean | Std | Min | Median | Max | % zero | Skew |
|---|---|---|---|---|---|---|---|
| Dry_Green_g | 26.62 | 25.40 | 0.00 | 20.80 | 157.98 | 5.0% | 1.74 |
| Dry_Dead_g | 12.04 | 12.40 | 0.00 | 7.98 | 83.84 | 11.2% | 1.75 |
| Dry_Clover_g | 6.65 | 12.12 | 0.00 | 1.42 | 71.79 | 37.8% | 2.83 |
| GDM_g | 33.27 | 24.94 | 1.04 | 27.11 | 157.98 | 0.0% | 1.55 |
| Dry_Total_g | 45.32 | 27.98 | 1.04 | 40.30 | 185.70 | 0.0% | 1.42 |

Clover's 37.8% zero-inflation and 2.83 skew are the dataset's most severe statistical features.

### 3.2 Problem Formulation
Let X ∈ ℝ^{H×W×3} be a canopy image. A model F_θ predicts the three **directly measured** fractions
**ŷ = [green, dead, clover] ∈ ℝ³_{≥0}**. The two composite targets are reconstructed by the dataset's
exact physical identities (verified, error ≈ 0):

- **GDM = Green + Clover**
- **Total = Green + Dead + Clover**

Predicting the 3 independent fractions and reconstructing GDM/Total (rather than regressing 5 free
outputs) enforces these conservation identities by construction.

### 3.3 Target Scale and Loss Design (raw targets)
Because biomass is heavy-tailed, an initial approach applied a `log1p` variance-stabilizing transform.
However, the competition metric and all reporting are in **raw grams**, and log-space training
systematically **under-predicts high-biomass swards** after `expm1` back-transformation (a small
log-space error amplifies at large magnitudes). We therefore train on **raw-gram targets** and instead
control heavy tails through the **loss weighting** (§4.3), retaining `log1p` only as an ablation
baseline (§7.3). This raw-target choice measurably improved the weighted-R² in development.

---

## IV. Methodology

**Fig. 1 — Model pipeline.** A canopy image is augmented, backbone-specific preprocessing is applied,
one of five ImageNet-pretrained backbones extracts features (BatchNorm frozen), a CBAM attention head
(CNNs) or the `[CLS]` token (ViT) feeds a small SiLU MLP predicting three raw-gram fractions, and
GDM/Total are reconstructed by summation. Trained with an RMSE-weighted composite loss and two-stage
depth-relative fine-tuning.

![Model pipeline](figures/architecture.png)

### 4.1 Backbones
Five ImageNet-pretrained backbones spanning the design spectrum (Table 2). We select these five to cover
the axes our research questions probe rather than to maximize count: **MobileNet (V1)** and
**EfficientNet-B0** anchor the lightweight/edge end (depthwise-separable and compound-scaled convolutions);
**EfficientNet-B3** adds a scaled-up CNN to test capacity-vs-resolution effects; **DenseNet-121** represents
dense feature-reuse convolutional design; and **ViT-B/16** contrasts pure global self-attention against all
four convolutional locality priors — the "tokens vs. textures" comparison at the heart of the study. Newer
hybrids (e.g. ConvNeXt, Swin) deliberately blend both priors and would blur that contrast, so we leave them
to future work and keep a clean CNN-vs-transformer axis here.

**TABLE 2 — Architectural specifications and computational complexity.**

| Backbone | Paradigm | Input | ~Params | ~GFLOPs |
|---|---|---|---|---|
| MobileNet (V1) | Depthwise-separable convolutions | 224² | 4.2M | 0.57 |
| EfficientNet-B0 | Compound-scaled MBConv+SE | 224² | 5.3M | 0.39 |
| EfficientNet-B3 | Compound-scaled MBConv+SE | 224² | 12.2M | 1.80 |
| DenseNet-121 | Dense feature reuse | 224² | 8.0M | 2.90 |
| ViT-B/16 | Patch self-attention | 224² | 86.6M | 17.6 |

**Resolution Standardization and Downsampling Protocol:** To enforce strict architectural parity and computational comparability, all backbones are trained and evaluated at an identical input resolution of **224 × 224 pixels**. Raw high-resolution field photographs (2000 × 1000) are bilinearly downsampled directly to 224 × 224 across all pipelines. We explicitly note that while EfficientNet-B3's standard compound scaling specification prescribes a native input resolution of **300 × 300**, we deliberately evaluate it at **224 × 224** (iso-resolution benchmark). Standardizing resolution across all five backbones serves two critical methodological purposes:
1. *Isolating Structural Capacity:* It eliminates input spatial resolution as a confounding variable, strictly testing whether B3's increased depth (1.2×) and width (1.4×) (12.2M params) yield representational gains on fixed spatial tokens.
2. *Edge/Embedded Parity:* It ensures consistent memory footprint and throughput across all backbones.
As detailed in §7.3, evaluating B3 at 224² reveals that scaling depth and width without scaling input resolution (or expanding dataset volume) induces over-parameterization and feature memorization on small sample sizes.

### 4.2 Attention Regression Head
To give every backbone the same expressive head, CNN feature maps pass through a **CBAM-style
attention block** [14] — sequential **channel attention** (shared MLP over average- and max-pooled
descriptors) then **spatial attention** (a 1×1 convolution) — before global pooling. ViT uses its
pooled `[CLS]` token. The pooled vector h then passes:

**h₁ = SiLU(W₁h + b₁), W₁ ∈ ℝ^{128×D}; h₂ = Dropout(h₁, p=0.3); ŷ = W₂h₂ + b₂, W₂ ∈ ℝ^{3×128}.**

The attention head lets the network emphasize biomass regions over soil clutter before regression.

**A note on head parity.** The *regression head* — the SiLU MLP above — is byte-for-byte identical
across all five backbones; only the **pooling front-end differs by necessity**: CNNs use CBAM
channel-and-spatial attention over their conv feature maps, whereas ViT has no convolutional map to
attend over and instead supplies its pooled `[CLS]` token, which is itself the product of self-attention.
CBAM would be architecturally redundant on ViT (attention on top of attention) and cannot be applied to
a token sequence in the same spatial form. We therefore keep the shared MLP head fixed and let each
backbone contribute its native attention mechanism, and we flag this as the one unavoidable asymmetry in
an otherwise identical pipeline — a design choice, not a confound, since both variants deliver a single
pooled feature vector of the same role to the shared head.

### 4.3 Composite Loss (RMSE-weighted)
We minimize a composite of MAE (robust, linear) and an ε-smoothed RMSE (curvature, outlier-aware):

**L = α·MAE + (1−α)·RMSE_ε,  RMSE_ε = √( (1/K)Σ(yₖ−ŷₖ)² + ε ),  ε = 10⁻⁷.**

Crucially, the RMSE term uses the mean **inside** the square root (true RMSE), so it penalizes large
residuals rather than degenerating into MAE. We set **α = 0.35** — i.e. **65% RMSE weight** — to
counter the high-biomass under-prediction (§3.3). The per-element gradient magnitude is bounded, so
heavy-tailed outliers cannot explode the gradient:

**∂L/∂ŷₖ = −(α/K)·sgn(eₖ) − ((1−α)/K)·eₖ / √((1/K)Σeⱼ² + ε),** bounded by (α+(1−α))/K = 1/K.

### 4.4 Two-Stage Depth-Relative Fine-Tuning
To adapt pretrained features on a small dataset without catastrophic forgetting, we use two stages
with **depth-relative unfreezing** (fair across architectures of differing depth), keeping
**BatchNorm frozen** (training BN on small batches destabilized fine-tuning):

- **Stage 1 (main):** unfreeze the top **30%** of backbone layers + head; AdamW [20], LR 1e-4 (5e-5 for
  ViT), weight decay 1e-2, 5-epoch warmup + cosine decay, EarlyStopping(patience 7).
- **Stage 2 (fine-tune):** unfreeze the top **40%**; AdamW at 10× lower LR (1e-5), weight decay 1e-4,
  ReduceLROnPlateau + EarlyStopping(patience 5).

A key empirical finding (§6) is that **fine-tuning is backbone-dependent**: it helps ViT, DenseNet,
and MobileNet (whose main stage under-fit) but slightly hurts the EfficientNets (already at their
data ceiling), for which the main stage is best.

**Tuning budget (fairness).** All backbones share the identical two-stage protocol, LR schedule, and
depth-relative unfreeze fractions (0.30/0.40). The only deviations are principled and standard:
ViT-B/16 uses a lower peak LR (5e-5 vs 1e-4), since transformers require gentler optimization, and its
unfreeze fraction was aligned to the CNNs after an initial over-conservative setting under-trained it.
No CNN received model-specific hyperparameter search beyond the shared protocol; we report this small
asymmetry transparently rather than tuning each model to its individual optimum.

---

## V. Experimental Setup

### 5.1 Data Split and Statistical Protocol
We use a single deterministic split (seed 42): **64% train / 16% validation / 20% test (≈72 images)**.
To quantify uncertainty on this small test set, we run a **15-loop bootstrap**: each loop draws 10
random test images (identical draws across all models) and computes the weighted-R²; the mean and
standard deviation give a **95% confidence interval** via

**CI = x̄ ± 1.96·(σ/√n), n = 15,**

where **x̄ is the mean of the 15 bootstrap loops** and σ their standard deviation (standard
error-of-the-mean). We separately report the **full-test weighted-R²** as the point estimate; since
10-image subsamples slightly under-estimate the aggregate R², the point estimate lies near the upper
end of the interval.

**Why a single split with bootstrap CIs (rather than k-fold).** We deliberately fix one 64/16/20 split
and quantify its uncertainty with a bootstrap, rather than running k-fold cross-validation, for three
reasons. (i) **Comparability:** every model is trained and scored on *exactly the same* held-out images
and the *same* bootstrap draws, so cross-model differences cannot be an artifact of different folds
seeing different data — the fairest possible head-to-head on a fixed compute budget. (ii) **Cost:**
k-fold would multiply the training cost of five backbones (including an 86.6M-parameter ViT with a
two-stage schedule) by *k*, without changing the paper's central, honestly-stated conclusion that all
CIs overlap. (iii) **Honesty over false precision:** the bootstrap already exposes the wide,
overlapping intervals that a single point number would hide, so it delivers the same cautionary message
k-fold would. We treat this as a defensible middle ground — more informative than one point estimate,
cheaper than k-fold — and flag full k-fold cross-validation as the natural rigor upgrade for a
larger-data follow-up (§X). A k-fold protocol reporting fold-wise mean±std would supersede this bootstrap
and remove the small point-estimate/interval offset noted above.

### 5.2 Augmentation
A trimmed, signal-preserving pipeline: `RandomFlip` (H+V), `RandomRotation(±0.1)`, and additive
Gaussian noise (σ = 0.02). We deliberately exclude color jitter and zoom/translation: greenness is the
biomass signal (color jitter degrades it) and zoom changes apparent density while the label is fixed.

### 5.3 Metric
Our primary metric is the **Global Weighted R²** with weights **w = [0.1, 0.1, 0.1, 0.2, 0.5]** for
[Green, Dead, Clover, GDM, Total]. **These weights are not chosen by us: they are the fixed, official
scoring formula of the CSIRO Image2Biomass Kaggle competition** [2], adopted verbatim so that our
numbers are directly comparable to the public leaderboard and are not open to accusations of
metric-tuning. The weighting is also agronomically sensible — **Total** dry biomass (the quantity that
drives stocking-rate and feed-budget decisions) carries 50% of the weight, and the derived aggregate
**GDM** a further 20%, so 70% of the score rewards the totals a grazier acts on, while the three raw
fractions split the remaining 30%. To guard against any single weighted number hiding per-fraction
behavior, we additionally report **unweighted per-target R²** (Table 4) alongside **RMSE and MAE in
grams**, so a reader who cares about, say, clover specifically can read that column directly rather than
trusting the composite. In short, the composite weight vector is externally imposed (competition-defined)
and we transparently pair it with the raw per-target metrics.

### 5.4 Reproducibility
All experiments use a single fixed seed (42) for the data split, weight initialization, and the 15
bootstrap draws (identical image draws across models). Models are implemented in TensorFlow/Keras 2.18
(Python 3.11) and trained on a single NVIDIA P100 GPU (Kaggle); all backbones initialize from
ImageNet-pretrained weights. The training configuration — LR schedule, depth-relative unfreeze fractions
(0.30/0.40), trimmed augmentation, and the composite loss (α = 0.35) — is identical across the five
backbones and fully specified in §IV. The exact split indices, per-model bootstrap loops, and prediction
outputs are released with the paper; the code and trained models are available from the corresponding
author on reasonable request.

---

## VI. Results

### 6.1 Leaderboard
Table 3 reports the cross-architecture weighted-R² benchmark, with both training stages, parameter
counts, FLOPs, and 95% confidence intervals.

**TABLE 3 — Weighted-R² benchmark (raw targets, GDM = Green+Clover).**

| Model | Params | GFLOPs | main R² | ft R² | **Best R²** | Best stage | 95% CI | σ |
|---|---|---|---|---|---|---|---|---|
| **ViT-B/16** | 86.6M | 17.6 | 0.7398 | 0.7455 | **0.7455** | ft | [0.681, 0.769] | **0.087** |
| EfficientNet-B0 | 5.3M | 0.39 | 0.7113 | 0.7076 | **0.7113** | main | [0.554, 0.724] | 0.168 |
| DenseNet-121 | 8.0M | 2.90 | 0.6675 | 0.7070 | **0.7070** | ft | [0.545, 0.700] | 0.153 |
| EfficientNet-B3 | 12.2M | 1.80 | 0.6817 | 0.6726 | **0.6817** | main | [0.521, 0.710] | 0.186 |
| MobileNet (V1) | 4.2M | 0.57 | 0.6211 | 0.6443 | **0.6443** | ft | [0.531, 0.668] | 0.135 |
| *Mean-predictor (baseline)* | — | — | — | — | *0.000* | — | — | — |

The mean-predictor (always outputs the training-set mean) scores weighted-R² = 0 by construction; all
five backbones clear this floor by a wide margin (+0.64 to +0.75), confirming they learn genuine visual
signal rather than exploiting the label distribution.

> **Best R²** = full-test point estimate; **95% CI** = bootstrap loop-mean ± 1.96·σ/√15 (Fix A). Because
> 10-image subsamples slightly under-estimate the aggregate R², each CI is centered a little below its
> Best-R² point estimate. All CIs overlap → no pairwise difference is statistically significant.

**Key observations (honest):**
1. **ViT-B/16 leads and is the most stable** — highest point estimate (0.7455) and lowest variance
   (σ = 0.087, tightest margin ±0.044).
2. **All 95% CIs overlap, but the ranking is directionally consistent.** No pairwise difference is
   statistically significant on this 357-image dataset. However, a **paired per-loop comparison** (all
   models scored on the *same* 15 bootstrap draws) shows the lead is consistent even where intervals
   overlap: **ViT-B/16 beats EfficientNet-B0 in 10 of 15 loops** (median per-loop Δ = **+0.035**, mean
   +0.086), and beats DenseNet-121 in 10/15, EfficientNet-B3 in 9/15, and MobileNet in **13/15**. We
   present this as suggestive evidence of a stable ordering, **not** as proof of significance.
3. **Fine-tuning is backbone-dependent:** it helps ViT (+0.006), DenseNet (**+0.040**), and MobileNet
   (+0.023), but slightly hurts EfficientNet-B0 (−0.004) and B3 (−0.009).
4. **Bigger ≠ better:** EfficientNet-B0 (0.7113) outperforms the 2× larger B3 (0.6817).

**Fig. 2 — Weighted-R² with 95% confidence intervals (15-loop bootstrap).** ViT is highest and has the
tightest interval (lowest σ); all bands overlap, so no pairwise difference is statistically significant.

![Confidence intervals](figures/ci_intervals.png)

**Fig. 3 — Training dynamics (composite loss, main + fine-tune).** Blue/orange = main train/val;
green/red = fine-tune train/val; red dashed = fine-tune start. ViT shows the smallest train–val gap;
EfficientNet-B3 the largest (overfitting); MobileNet stays high (under-fitting); DenseNet's fine-tune
clearly lowers both curves.

![(a) MobileNet (V1)](../curves/mobilenetv3/combined_loss.png)
![(b) EfficientNet-B0](../curves/efficientnetb0/combined_loss.png)
![(c) EfficientNet-B3](../curves/efficientnetb3/combined_loss.png)
![(d) DenseNet-121](../curves/densenet121/combined_loss.png)
![(e) ViT-B/16](../curves/vitb16/combined_loss.png)

### 6.2 Per-Target Breakdown
Table 4 decomposes accuracy per target, revealing which architecture excels on each biomass fraction.

**TABLE 4 — Per-target R² (best stage; GDM = Green+Clover). Bold = best per column.**

| Model | Green | Dead | Clover | GDM | Total | Weighted |
|---|---|---|---|---|---|---|
| ViT-B/16 | 0.7054 | **0.3315** | 0.5675 | 0.6900 | **0.6524** | **0.7455** |
| EfficientNet-B0 | 0.6394 | 0.2534 | 0.5609 | 0.6660 | 0.6005 | 0.7113 |
| DenseNet-121 | **0.7288** | 0.2567 | 0.3537 | **0.7049** | 0.5762 | 0.7070 |
| EfficientNet-B3 | 0.6685 | 0.2253 | **0.5709** | 0.5935 | 0.5556 | 0.6817 |
| MobileNet (V1) | 0.6311 | 0.2550 | 0.4939 | 0.5425 | 0.4998 | 0.6443 |

To complement the scale-free R², Table 4b reports per-target error in grams (best stage). The ordering
mirrors the R² results: **ViT-B/16 attains the lowest Total-biomass error (RMSE 14.23 g on a mean Total
of 45.3 g) and the lowest Dead-matter error**, while **DenseNet-121 has the lowest Green and
GDM error (RMSE 11.29 g and 11.91 g)** — consistent with its top Green/GDM R². Total-biomass RMSE spans 14.23 g (ViT) to 17.07 g
(MobileNet); errors are largest in absolute terms for the high-magnitude Total target and smallest for
Clover, and MAE tracks RMSE closely throughout, indicating no model is uniquely outlier-driven.

**TABLE 4b — Per-target error in grams (best stage): RMSE / MAE. Bold = lowest RMSE per column.
GDM = Green + Clover (paper convention).**

| Model | Green | Dead | Clover | GDM | Total |
|---|---|---|---|---|---|
| ViT-B/16 | 11.77 / 7.28 | **9.83 / 6.43** | 8.78 / 5.02 | 12.21 / 8.05 | **14.23 / 9.65** |
| EfficientNet-B0 | 13.02 / 8.56 | 10.39 / 6.76 | 8.85 / 5.64 | 12.67 / 9.08 | 15.25 / 10.72 |
| DenseNet-121 | **11.29 / 7.48** | 10.37 / 6.97 | 10.73 / 6.05 | **11.91 / 8.14** | 15.71 / 11.08 |
| EfficientNet-B3 | 12.49 / 7.63 | 10.59 / 6.71 | **8.74 / 5.24** | 13.98 / 9.31 | 16.09 / 10.45 |
| MobileNet (V1) | 13.17 / 8.63 | 10.38 / 6.74 | 9.50 / 5.46 | 14.83 / 9.99 | 17.07 / 11.91 |

**Fig. 4 — Per-target R² by architecture.** DenseNet leads Green/GDM; ViT leads Dead/Total; dead
matter is the universal low target across all models.

![Per-target R²](figures/per_target_r2.png)

### 6.3 Which model wins which evaluation (evidence-based)
Table 5 names the best model on each axis. We stress once here, to keep the rest of the discussion
uncluttered, that these are **rankings of point estimates whose confidence intervals overlap** (§6.1):
each "winner" is the *consistently preferable* choice on that axis, not a statistically separable one.
With that caveat fixed, the per-axis picture is clear and — importantly — **no single model wins
everything**, which is the practically useful result.

**TABLE 5 — Best model per evaluation axis, with the supporting number.**

| Evaluation axis | Winner | Evidence |
|---|---|---|
| Overall accuracy (weighted R²) | **ViT-B/16** | 0.7455 |
| Statistical stability (lowest σ) | **ViT-B/16** | σ = 0.087 (tightest CI) |
| Total biomass (weight 0.5) | **ViT-B/16** | 0.6524 |
| Dead matter (hardest) | **ViT-B/16** | 0.3315 (vs CNN 0.225–0.257) |
| Green grass | **DenseNet-121** | 0.7288 |
| GDM (Green+Clover) | **DenseNet-121** | 0.7049 |
| Clover | **EfficientNet-B3** | 0.5709 (ViT 0.5675, tie) |
| Efficiency (accuracy per FLOP) | **EfficientNet-B0** | 0.7113 at 0.39 GFLOPs |
| Smallest footprint | MobileNet (V1) | 4.2M params (fewest); 0.57 GFLOPs |

*Dead matter is the universal weak target (R² 0.23–0.33): its brown reflectance overlaps bare soil,
so no architecture resolves it well.*

**Practical accuracy.** The best model's Total-biomass error (RMSE ≈ 14 g on a mean of 45 g) is ≈ 31%
— adequate for **coarse paddock-level ranking and relative high/low-biomass comparisons**, but above the
~15–20% tolerance typically desired for **precise per-quadrat feed budgeting**. Image-only estimation is
therefore best positioned as a **low-cost screening tool** that flags high- and low-biomass zones, with
destructive sampling reserved for calibration — a realistic near-term role for commodity-camera biomass.

Figures: training curves in `Thesis/curves/<model>/` (loss/MAE/RMSE, main+fine-tune);
per-model CI in `Thesis/CI_outputs/confidence_intervals.xlsx`.

---

## VII. Architectural Diagnostics (grounded in the results)

We interpret the benchmark strictly from measured evidence (per-target R², loss curves, and the
explainability maps of §8). Because the dataset is small and CIs overlap, we frame these as observed
tendencies, not proven causes.

### 7.1 Mechanistic Hypotheses: "Tokens vs. Textures" (Inductive Bias & Receptive Field)

> **Read this subsection as informed speculation, not proof.** The equations below are standard,
> textbook descriptions of how convolution and self-attention *operate*; we use them only to articulate
> *why* the measured ranking is architecturally plausible. They are **not evidence** that these
> mechanisms caused our specific results — with 357 images and fully overlapping confidence intervals
> (§6), we cannot and do not claim causal attribution. The verifiable claims of this paper live in the
> tables (§6, §7.7) and the honestly-reported XAI audit (§8); everything in §7.1 is interpretive
> scaffolding that a skeptical reader may discount without affecting those results.

To *motivate* why ViT-B/16 tends to lead on aggregate and senescent targets while DenseNet-121 excels
at local green textures, we outline the architectural mechanisms that plausibly contribute. We stress
that these are **hypotheses consistent with our measurements, not proven causes** — the small dataset
and overlapping confidence intervals (§6) preclude causal attribution.

#### 1. Convolutional Spatial Locality & Receptive Field Bottlenecks (Textures)
In standard CNNs (MobileNet, DenseNet-121, EfficientNet), feature extraction at spatial coordinate $p = (x, y)$ in layer $l$ is governed by translation-invariant, static parameter kernels $\mathbf{W} \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times k \times k}$:
$$\mathbf{z}_l(p) = \sigma \left( \sum_{\delta \in \Omega_k} \mathbf{W}(\delta) \mathbf{z}_{l-1}(p + \delta) + \mathbf{b} \right), \quad \Omega_k = \left\{ \delta \in \mathbb{Z}^2 \;\middle|\; \|\delta\|_\infty \le \left\lfloor \frac{k}{2} \right\rfloor \right\}$$

The Effective Receptive Field (ERF) follows a 2D Gaussian distribution that expands slowly with layer depth $L$:
$$\text{ERF}_L \propto k \sqrt{L}$$

**The Locality Dilemma in Canopies:** Convolutions impose an explicit *spatial locality bias*: output features are strictly computed from adjacent pixels. However, pasture swards are inherently non-contiguous: clover leaflets (*Trifolium repens*) and dry thatch patches are dispersed across the $70\,\text{cm} \times 30\,\text{cm}$ quadrat. CNNs cannot model correlations between distant legume clusters without repeated spatial pooling, by which point high-frequency structural boundaries are blurred.

#### 2. Vision Transformer: Global Receptive Field from Layer 1 (Tokens)
ViT-B/16 tokenizes the canopy image $\mathbf{X} \in \mathbb{R}^{H \times W \times C}$ into $N = \frac{HW}{P^2} = 196$ non-overlapping patches ($P = 16$), linearly projected into latent dimension $D = 768$. For each visual token $\mathbf{z}_i$, Multi-Head Self-Attention (MHSA) computes pairwise affinity across all tokens simultaneously:
$$\mathbf{Q} = \mathbf{Z} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{Z} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{Z} \mathbf{W}_V$$
$$\mathbf{A}_{i,j} = \frac{\exp\left( \frac{\mathbf{q}_i^T \mathbf{k}_j}{\sqrt{d_k}} \right)}{\sum_{m=1}^{N+1} \exp\left( \frac{\mathbf{q}_i^T \mathbf{k}_m}{\sqrt{d_k}} \right)}, \quad \operatorname{MHSA}(\mathbf{z}_i) = \sum_{j=1}^{N+1} \mathbf{A}_{i,j} \mathbf{v}_j \mathbf{W}_O$$

Because $\mathbf{A}_{i,j} > 0$ for all $i, j \in \{1, \dots, 196\}$, ViT establishes an **unconstrained global receptive field at Layer 1**, enabling immediate aggregation of dispersed botanical fractions without downsampling artifacts.

#### 3. Dynamic Background Soil Attenuation (hypothesized)
A key theoretical distinction lies in the static vs. dynamic nature of filtering:
- In CNNs, filter weights $\mathbf{W}(\delta)$ are static post-training; every spatial coordinate undergoes identical filtering regardless of whether it contains soil or foliage.
- In ViT, attention coefficients $\mathbf{A}_{i,j}(\mathbf{X})$ are **dynamic input-conditioned variables**. When token $j$ represents bare soil or dark ground clutter, its projected key vector $\mathbf{k}_j$ aligns orthogonally to the global classification query vector $\mathbf{q}_{\text{class}}$:
$$\lim_{\mathbf{z}_j \to \text{Soil}} \mathbf{q}_{\text{class}}^T \mathbf{k}_j \ll \mathbf{q}_{\text{class}}^T \mathbf{k}_{\text{foliage}} \implies \mathbf{A}_{\text{class}, j} \longrightarrow 0$$
In principle this *could* let ViT down-weight bare-soil tokens. We caution, however, that our
quantitative attention-coverage metric (§8) did **not** detect selective soil suppression on this
canopy-dominated dataset (attention ≈ vegetation coverage for all models); we therefore present soil
attenuation as a mechanism *hypothesis*, not a demonstrated effect.

#### 4. Resolving the Dead Matter Disambiguation Bottleneck (`Dry_Dead_g`)
Senescent dead vegetation (`Dry_Dead_g`) exhibits severe spectral overlap with brown loamy/sandy soil, and remains the hardest target for every model ($R^2 \le 0.2567$ for CNNs). ViT retains $16 \times 16$ patch-token resolution without spatial pooling, which *may* better preserve thin dead-stalk contours; empirically it attains the **highest dead-matter R² (0.3315), a roughly +29% to +47% relative gain** over the CNNs' 0.225–0.257. We report this gain as an observation; the resolution-preservation mechanism is a plausible but unproven explanation.

**7.2 ViT-B/16 — best on the aggregate and hardest targets.** ViT achieves the highest Dead-matter R²
(0.3315 vs 0.225–0.257 for CNNs, **Table 4**), the highest Total R² (0.6524, **Table 4**), the highest
weighted-R² (0.7455) and the lowest variance (σ = 0.087, **Table 3**). Its SHAP/LIME attributions (§8)
are distributed broadly across the canopy rather than concentrated on a few blades, and its training
curve shows the smallest train–val gap of any model — consistent with a model that generalizes rather
than memorizes. We attribute its edge on aggregate quantities to the whole-image token mixing of
multi-head self-attention [15], reported as an observation supported by these measurements. This global-versus-local distinction is well
documented — ViTs learn more uniform, globally-informed representations than CNNs [23] — as is the
difficulty of training ViTs on small datasets without added inductive bias [24], consistent with our
finding that ViT required the most careful fine-tuning to reach its potential.

**7.3 DenseNet-121 — best green/GDM extractor.** DenseNet's dense feature reuse [13] leads on Green
(0.7288) and GDM (0.7049) — both column-best in **Table 4** — and its fine-tuning gain is the largest of
any model (+0.040, **Table 3**), consistent with a backbone whose main stage under-fit. Its Grad-CAM
(`CI_outputs/gradcam/DenseNet121/`, Fig. 5a) places heat tightly on central green foliage. It is weakest
on Clover (0.3537, **Table 4**), consistent with feature saturation when isolating sparse legumes among
grass.

**7.4 EfficientNet family & the Iso-Resolution Scaling Effect.** Both variants incorporate compound scaling principles [12]. EfficientNet-B0 emerges as the efficiency champion (0.7113 at 0.39 GFLOPs, **Table 3**), striking the optimal capacity-to-receptive-field balance for 224² inputs on small agricultural datasets. In contrast, EfficientNet-B3 (12.2M params), despite having over 2.3× the capacity of B0, achieves a lower weighted-R² (0.6817, **Table 3**) and exhibits the **largest train–val generalization gap** among all models (Fig. 3c).

This result provides key empirical insight into compound scaling: when B3 is evaluated at 224 × 224 (below its native 300 × 300 resolution), its expanded convolutional channels and deeper layers lack the high-frequency pixel granularity they were designed to process. Rather than learning richer botanical representations, the additional parameters more likely overfit — directly evidenced by B3's largest train–val gap — though the precise failure mode is not separately measured. Furthermore, both EfficientNets exhibit slight performance degradation during Stage 2 fine-tuning (B0: 0.7113 -> 0.7076; B3: 0.6817 -> 0.6726), indicating that compound-scaled inverted bottlenecks reach their optimal representation in Stage 1 and risk representational drift when unfrozen on limited sample sizes (N=357).

**7.5 MobileNet (V1).** Built on depthwise-separable convolutions with ReLU activations [11], MobileNet
is the smallest model (4.2M parameters, **Table 2**) but records the lowest overall accuracy (0.6443,
**Table 3**) and the highest Total-biomass error (RMSE = 17.07 g). Its main-stage loss reflects
under-fitting on dense, multi-layer swards (Fig. 3a, curve stays high); Stage-2 fine-tuning yields a
+0.023 gain (**Table 3**) — among the largest relative improvements — indicating that its limited
capacity benefits most from direct backbone adaptation. Note that MobileNet V1 is *not* the most
compute-efficient model here — EfficientNet-B0 attains higher accuracy at fewer FLOPs — so its practical
niche is a small parameter footprint rather than raw inference speed.

**7.6 Universal failure — dead matter.** Every model plateaus at Dead R² ≤ 0.33. The Grad-CAM/SHAP
maps for the Dead target are scattered and frequently land on soil (§8), directly visualizing the
dead–soil spectral ambiguity.

**7.7 Ablation of design choices (EfficientNet-B0).** Table 6 ablates each pipeline choice by changing
one factor at a time from the final configuration; every choice is justified (all Δ < 0). We ablate on
**two representative backbones — EfficientNet-B0 (the strongest CNN) and ViT-B/16 (the transformer)** —
rather than all five, by design: the aim is to show each design decision transfers *across architectural
families* at tractable compute, and cross-family agreement (below) is stronger evidence than repeating
the sweep within the CNN family. Ablating all five would multiply training cost fivefold (including the
86.6M-parameter ViT's two-stage schedule) for negligible added insight.

**TABLE 6 — Ablation on EfficientNet-B0.** Each row changes ONE factor from the final config;
weighted-R² on the held-out test set. (Raw data: `research_paper/ablation_effb0.csv`.)

| Configuration (one factor changed) | Weighted R² | Δ vs final | MAE (g) | RMSE (g) |
|---|---|---|---|---|
| **FINAL** (raw, α=0.35, unfreeze 0.30, BN frozen, trimmed aug) | **0.6965** | — | 7.03 | 10.95 |
| Target: **log1p** (vs raw) | 0.2769 | **−0.4196** | 7.70 | 14.17 |
| Loss: **α = 0.65** (MAE-heavy) | 0.6857 | −0.0108 | 6.73 | 10.77 |
| Loss: **α = 0.50** | 0.6777 | −0.0189 | 7.06 | 11.04 |
| Unfreeze: **frozen backbone** (head only) | 0.4394 | −0.2572 | 9.17 | 14.46 |
| BatchNorm: **trainable** (vs frozen) | 0.3783 | −0.3182 | 7.95 | 13.97 |
| Augmentation: **full** (color jitter + zoom) | **−0.1675** | **−0.8641** | 11.14 | 17.91 |

The ranking of importance is informative:
1. **Augmentation is the most critical choice.** Restoring full color-jitter + zoom augmentation is
   **catastrophic** (R² collapses to **−0.17**, Δ = −0.86): color jitter destroys the greenness cue and
   zoom corrupts the fixed-label canopy density — strongly validating the trimmed pipeline.
2. **Raw targets matter next.** `log1p` conditioning drops R² by **0.42** (to 0.28), confirming §3.3.
3. **BatchNorm must stay frozen.** Trainable BN drops R² by **0.32** (to 0.38).
4. **Backbone adaptation is necessary.** Freezing the whole backbone drops R² by **0.26** (to 0.44).
5. **Loss balance α is the least sensitive knob.** α = 0.35 marginally beats α = 0.50 (−0.019) and
   α = 0.65 (−0.011) — a modest RMSE-lean is optimal but not decisive.

Ranked by impact: **augmentation ≫ target-scale > BatchNorm > unfreezing > loss-α.** (The single-stage
ablation baseline R² = 0.6965 is consistent with the 0.7113 of the full two-stage model.)

To test whether these choices generalize beyond a CNN, we repeat the ablation on **ViT-B/16** (Table 6b).
Since ViT has no BatchNorm and no CBAM head, the BN ablation is replaced by two transformer-relevant
factors: the peak learning rate and the over-conservative unfreeze that originally under-trained it.

**TABLE 6b — Ablation on ViT-B/16.** Same protocol; single-stage weighted-R² (raw data:
`research_paper/ablation_vit.csv`).

| Configuration (one factor changed) | Weighted R² | Δ vs final | MAE (g) | RMSE (g) |
|---|---|---|---|---|
| **FINAL** (raw, α=0.35, unfreeze 0.30, LR 5e-5, trimmed aug) | **0.7398** | — | 6.39 | 10.27 |
| Target: **log1p** (vs raw) | 0.2313 | **−0.5084** | 8.18 | 14.41 |
| Loss: **α = 0.65** (MAE-heavy) | 0.6981 | −0.0417 | 6.66 | 10.76 |
| Loss: **α = 0.50** | 0.7277 | −0.0120 | 6.47 | 10.46 |
| Unfreeze: **frozen backbone** (head only) | 0.2765 | −0.4633 | 8.81 | 14.91 |
| Unfreeze: **0.15** (over-conservative, original) | 0.2886 | −0.4512 | 8.78 | 14.83 |
| LR: **1e-4** (vs 5e-5, too hot for ViT) | 0.7127 | −0.0271 | 6.20 | 10.20 |
| Augmentation: **full** (color jitter + zoom) | **0.0917** | **−0.6481** | 10.46 | 16.65 |

**Cross-model agreement.** Both backbones agree on the ranking: **augmentation and raw targets are the
most critical choices** (removing either costs 0.42–0.86 R² on EffB0 and 0.51–0.65 on ViT), followed by
backbone unfreezing, with the loss-α knob least sensitive. ViT adds two transformer-specific
confirmations: the **over-conservative 0.15 unfreeze collapses it to 0.29** (justifying the deeper 0.30
unfreeze that fixed ViT), and the **hotter 1e-4 LR costs 0.03** (justifying the gentler 5e-5). That the
same design choices transfer from a CNN to a transformer strengthens their generality.

---

## VIII. Explainability Audit (SHAP, LIME, Grad-CAM)

**What this audit does and does not establish.** Our goal here is a *trust check*, not a headline
number: we ask whether the models' predictions rest on plant structure rather than on background soil or
framing artifacts. This section reports **two findings, one positive and one deliberately negative, both
stated plainly.** (1) *Qualitatively*, the saliency maps are reassuring — attention concentrates on
foliage, and where a model struggles (dead matter) the maps visibly drift onto soil, so the explanations
are consistent with the accuracy results. (2) *Quantitatively*, a naive "fraction of attention on
vegetation" metric turns out to be **uninformative on this dataset**, because the canopy is already ≈90%
vegetation, so every model scores ≈90% by construction (Table 7). We report this null result openly
rather than dropping it: it is a methodological warning that attention-*coverage* is the wrong faithfulness
metric for dense-canopy imagery, and it motivates the deletion/insertion curves we flag as future work
(§X). Read together, the audit supports biological plausibility qualitatively while being candid that our
quantitative interpretability metric could not discriminate the models — an honest boundary on what
current XAI buys us here, not a positive claim dressed up as one.

We audit whether models rely on plant structures rather than background. CNNs use **Grad-CAM** [19]
(gradient-based, needs conv maps); all models — including ViT — use **model-agnostic LIME [18] and
SHAP [17]**.
For ViT specifically, attention-rollout maps were uninformative (near-uniform), so we rely on LIME/SHAP
for its interpretation — itself a methodological finding.

Real outputs in `Thesis/CI_outputs/{gradcam,explanations}/`. Observed:
- **DenseNet-121 (Green, Grad-CAM):** heat focused on central green foliage → explains its top Green R².
- **EfficientNet-B0 (Dead, Grad-CAM):** diffuse/scattered heat → visualizes the dead–soil ambiguity.
- **EfficientNet-B0 (Green, SHAP):** positive attributions over the plant → correct reasoning.
- **ViT-B/16 (Green, LIME):** superpixels distributed across canopy foliage; attention map uninformative.

**Fig. 5 — Grad-CAM on a high-biomass sample.** DenseNet's Green attention is tightly focused on
foliage (a); EfficientNet-B0's Dead attention is scattered and drifts onto soil (c), directly
visualizing the dead–soil ambiguity; ViT's attention-rollout map (d) is near-uniform and uninformative.

![(a) DenseNet-121 | Green — focused](../CI_outputs/gradcam/DenseNet121/img4_Green.png)
![(b) EfficientNet-B0 | Green](../CI_outputs/gradcam/EfficientNetB0/img4_Green.png)
![(c) EfficientNet-B0 | Dead — scattered](../CI_outputs/gradcam/EfficientNetB0/img4_Dead.png)
![(d) ViT-B/16 | attention — uninformative](../CI_outputs/gradcam/ViT-B16/img4_attention.png)

**Fig. 6 — Model-agnostic explanations (Green target).** SHAP places positive (red) attribution on the
plant; LIME superpixels track foliage. These are the interpretation of choice for ViT, whose attention
map is uninformative.

![(a) EfficientNet-B0 | SHAP](../CI_outputs/explanations/EfficientNetB0/fixed_testset/img4_Green_shap.png)
![(b) EfficientNet-B0 | LIME](../CI_outputs/explanations/EfficientNetB0/fixed_testset/img4_Green_lime.png)
![(c) ViT-B/16 | SHAP](../CI_outputs/explanations/ViT-B16/fixed_testset/img4_Green_shap.png)
![(d) ViT-B/16 | LIME — distributed canopy](../CI_outputs/explanations/ViT-B16/fixed_testset/img4_Green_lime.png)

**Quantitative check (and its limit).** To move beyond single-image inspection, we measured the
fraction of Grad-CAM attention mass falling on vegetation (Excess-Green > 0) across all 72 test images
for the four CNNs (Table 7).

**TABLE 7 — Grad-CAM attention on vegetation (% of attention mass, all test images).**

| Model | Green | Dead | Clover | Overall | vs. baseline |
|---|---|---|---|---|---|
| MobileNet (V1) | 89.8 | 90.0 | 89.6 | 89.8 | −0.4 |
| EfficientNet-B0 | 90.5 | 90.4 | 90.0 | 90.3 | +0.1 |
| EfficientNet-B3 | 90.6 | 90.2 | 90.2 | 90.4 | +0.2 |
| DenseNet-121 | 89.9 | 90.2 | 90.7 | 90.3 | +0.1 |

All models place ≈ 90% of attention on vegetation — but **vegetation already covers ≈ 90% of these
dense-canopy quadrats**, so this equals the chance level (vs. baseline ≈ 0). This metric therefore
**cannot discriminate selective plant-focus from the high vegetation coverage** on a canopy-dominated
dataset, and attention is near-identical across targets (no Green/Dead/Clover specialization visible at
this resolution). We report this transparently: quantitative *coverage* is uninformative here, so
interpretability rests on the qualitative maps (Figs 5–6), and stronger **faithfulness metrics
(deletion/insertion curves, or dead-vs-soil sub-region attention) are left as future work**.

---

## IX. Computational Efficiency & Deployment

Two models are Pareto-optimal on the accuracy–compute frontier (Fig. 7): **EfficientNet-B0**
(0.7113 at 0.39 GFLOPs) and **ViT-B/16** (0.7455 at 17.6 GFLOPs, ~45×). EfficientNet-B3 and DenseNet are
dominated; MobileNet is dominated on FLOPs but retains the **smallest parameter count**. We map the
results to three operational deployment tiers:

- **Tier 1 — Cloud / GPU:** **ViT-B/16** (0.7455) for maximum accuracy where server compute is available
  (regional feed budgeting, satellite/UAV ground-truthing).
- **Tier 2 — Edge compute:** **EfficientNet-B0** (0.7113 at 0.39 GFLOPs) — the Pareto-optimal
  accuracy-per-FLOP choice for on-vehicle / embedded inference.
- **Tier 3 — Mobile footprint:** **MobileNet (V1)** (0.6443, 4.2M parameters — the fewest) for
  storage/memory-constrained on-device apps, accepting lower accuracy. *(Caveat: EfficientNet-B0 is both
  more accurate and lower-FLOP; MobileNet's niche is smallest model size, not inference speed.)*

**Recommended default: EfficientNet-B0.** If a practitioner must pick *one* model without knowing the
deployment target, we recommend **EfficientNet-B0**. It sits on the Pareto frontier at the lowest FLOPs
of any model (0.39 GFLOPs), its 0.7113 weighted-R² is within **0.034** of the best (ViT's 0.7455) and
statistically indistinguishable from it (overlapping CIs, §6), and it runs on edge hardware where ViT
cannot. ViT-B/16 is the right choice **only** when (a) server/GPU compute is guaranteed and (b) the
marginal accuracy on Total and dead-matter targets is operationally worth ≈45× the compute — e.g.
regional feed budgeting or satellite/UAV ground-truthing. Put simply: **default to EfficientNet-B0;
escalate to ViT-B/16 when accuracy is paramount and compute is free; drop to MobileNet only under a hard
model-size cap.**

We report FLOPs rather than wall-clock latency, which depends on the target device and is left to
on-device measurement.

**Fig. 7 — Accuracy vs. compute Pareto frontier.** EfficientNet-B0 and ViT-B/16 form the frontier;
the other three are dominated (more FLOPs, lower or equal R²).

![Pareto frontier](figures/pareto.png)

---

## X. Limitations & Future Work

- **Small test set → overlapping CIs.** ViT leads on every axis but its superiority is not
  statistically separable on 357 images; larger multi-year data is needed for significance.
- **Monocular occlusion.** Top-down RGB cannot see understory biomass under tall canopies.
- **Dead-matter ceiling.** Dead–soil spectral overlap caps all models at R² ≤ 0.33.
- **Single-year data.** All samples are 2015; seasonal/annual generalization is untested.
- **XAI-metric saturation.** Vegetation covers ≈ 90% of the canopy, so the attention-on-vegetation
  coverage metric is at chance and cannot discriminate models (Table 7); interpretability therefore
  rests on the qualitative maps (Figs 5–6).

Future work: NIR/red-edge fusion and LiDAR height to break occlusion; a zero-inflated hurdle model for
the 37.8%-zero clover component; and discriminative interpretability via **deletion/insertion
faithfulness curves** (which, unlike coverage, are not saturated by high vegetation density).

---

## XI. Conclusion

Under one controlled pipeline, ViT-B/16 is the strongest and most stable pasture-biomass estimator
(weighted R² 0.7455, σ 0.087), leading on total and dead-matter targets; DenseNet-121 is the best
green/GDM extractor (0.7288/0.7049) and EfficientNet-B0 the efficiency-optimal choice
(0.7113 at 0.39 GFLOPs). All confidence intervals overlap, so we present ViT's lead as consistent and
practically preferable rather than statistically proven — an honest benchmark that also validates
biological grounding via Grad-CAM/SHAP/LIME. The raw-target, RMSE-weighted composite loss, CBAM head,
and depth-relative two-stage fine-tuning together provide a reproducible recipe for multi-component
biomass regression.

---

## References

[1] I. L. Bretas et al., "Prediction of aboveground biomass and dry-matter content in brachiaria pastures by combining meteorological data and satellite imagery," *Grass and Forage Science*, vol. 76, no. 3, pp. 340–352, 2021, doi: 10.1111/gfs.12517.

[2] Q. Liao et al., "Estimating pasture biomass from top-view images: A dataset for precision agriculture," *arXiv:2510.22916*, 2025.

[3] M. T. Schaefer and D. W. Lamb, "A combination of plant NDVI and LiDAR measurements improve the estimation of pasture biomass in tall fescue," *Remote Sensing*, vol. 8, no. 2, art. 109, 2016, doi: 10.3390/rs8020109.

[4] A. Edirisinghe, M. J. Hill, G. E. Donald, and M. Hyder, "Quantitative mapping of pasture biomass using satellite imagery," *Int. J. Remote Sens.*, vol. 32, no. 10, pp. 2699–2724, 2011, doi: 10.1080/01431161003743181.

[5] I. Sinde-González et al., "Biomass estimation of pasture plots with multitemporal UAV-based photogrammetric surveys," *Int. J. Appl. Earth Obs. Geoinf.*, vol. 101, art. 102355, 2021, doi: 10.1016/j.jag.2021.102355.

[6] M. G. Trotter, D. W. Lamb, G. E. Donald, and D. A. Schneider, "Evaluating an active optical sensor for quantifying and mapping green herbage mass and growth in a perennial grass pasture," *Crop and Pasture Science*, vol. 61, no. 5, pp. 389–398, 2010, doi: 10.1071/CP10019.

[7] P. Migolet, L. Coulibaly, H. Adegbidi, and E. Hervet, "Utilization of neural networks for the estimation of aboveground forest biomass from ikonos satellite image," in *IEEE IGARSS*, 2007, pp. 4339–4342.

[8] S. Skovsen et al., "The GrassClover image dataset for semantic and hierarchical species understanding in agriculture," in *Proc. IEEE/CVF CVPR Workshops (CVPRW)*, 2019, pp. 2676–2684.

[9] B. B. Sapkota et al., "Use of synthetic images for training a deep learning model for weed detection and biomass estimation in cotton," *Sci. Rep.*, vol. 12, art. 19580, 2022, doi: 10.1038/s41598-022-23399-z.

[10] P. Albert et al., "Unsupervised domain adaptation and super resolution on drone images for autonomous dry herbage biomass estimation," *arXiv:2204.08271*, 2022.

[11] A. G. Howard et al., "MobileNets: Efficient convolutional neural networks for mobile vision applications," *arXiv:1704.04861*, 2017.

[12] M. Tan and Q. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2019, pp. 6105–6114.

[13] G. Huang, Z. Liu, L. Van Der Maaten, and K. Q. Weinberger, "Densely connected convolutional networks," in *Proc. IEEE CVPR*, 2017, pp. 4700–4708.

[14] S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, "CBAM: Convolutional block attention module," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2018, pp. 3–19.

[15] A. Dosovitskiy et al., "An image is worth 16×16 words: Transformers for image recognition at scale," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2021.

[16] V. Santhanam, V. I. Morariu, and L. S. Davis, "Generalized deep image to image regression," in *Proc. IEEE CVPR*, 2017, pp. 5395–5405, doi: 10.1109/CVPR.2017.573.

[17] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2017.

[18] M. T. Ribeiro, S. Singh, and C. Guestrin, "'Why should I trust you?': Explaining the predictions of any classifier," in *Proc. ACM SIGKDD (KDD)*, 2016, pp. 1135–1144.

[19] R. R. Selvaraju et al., "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2017, pp. 618–626.

[20] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2019.

[21] O. Elharrouss et al., "Task-based loss functions in computer vision: A comprehensive review," *arXiv:2504.04242*, 2025.

[22] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE CVPR*, 2016, pp. 770–778.

[23] M. Raghu, T. Unterthiner, S. Kornblith, C. Zhang, and A. Dosovitskiy, "Do vision transformers see like convolutional neural networks?" in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 34, 2021, pp. 12116–12128.

[24] Z. Lu, H. Xie, C. Liu, and Y. Zhang, "Bridging the gap between vision transformers and convolutional neural networks on small datasets," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2022.

[25] Y. Bazi, L. Bashmal, M. M. Al Rahhal, R. Al Dayil, and N. Al Ajlan, "Vision transformers for remote sensing image classification," *Remote Sensing*, vol. 13, no. 3, art. 516, 2021, doi: 10.3390/rs13030516.

[26] A. A. Aleissaee, A. Kumar, R. M. Anwer, S. Khan, H. Cholakkal, G.-S. Xia, and F. S. Khan, "Transformers in remote sensing: A survey," *Remote Sensing*, vol. 15, no. 7, art. 1860, 2023, doi: 10.3390/rs15071860.

---
*End of draft v2. Figure paths are relative (`../curves`, `../CI_outputs`). Format to IEEE two-column
in Word/LaTeX for submission; verify all reference DOIs.*
