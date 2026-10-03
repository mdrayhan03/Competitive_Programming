# Chapter 1 — Introduction

## 1.1 Background and Research Context

Grasslands cover the majority of the world’s agricultural land and sustain ruminant meat, milk, and wool production. Accurate estimation of above-ground pasture biomass is therefore a foundational input for stocking-rate decisions, rotational-grazing timing, feed budgeting, and grassland carbon management. Prior work has shown that improved biomass allocation can raise gross margins by approximately \$52–96 per hectare [1], [2]. Despite this economic and environmental importance, reliable in-field biomass estimation remains difficult at scale.

The classical ground-truth procedure—clipping a fixed quadrat, botanically separating green, dead, and clover fractions, oven-drying, and weighing—is accurate but destructive, labour-intensive, and non-scalable for routine farm decisions [3]. Remote sensing alternatives based on satellite or UAV vegetation indices can cover large areas, yet they commonly saturate at high leaf-area index (LAI > 3) and remain vulnerable to cloud cover, atmospheric effects, and calibration error [4], [5]. Even ground-level classical pipelines that regress handcrafted vegetation indices such as NDVI, Excess Green (ExG), or Colour Index of Vegetation Extraction (CIVE) from optical or capacitance sensors degrade under variable illumination and dense multi-species canopies [3], [6]. Critically, index-based methods typically cannot separate the botanical fractions—green grass, senescent dead matter, and clover—that grazing and feeding decisions actually depend on.

This ceiling on handcrafted-index regression motivates a shift toward learned, structure-aware visual representations. Ground-level top-down RGB photography of a pasture quadrat offers a particularly attractive sensing modality: it is non-destructive, low-cost, sensor-light, and compatible with commodity cameras already available in the field. Recent public resources, including the CSIRO Image2Biomass dataset and its associated Kaggle challenge [2], have made multi-site, laboratory-validated pasture imagery available for machine-learning research. The dataset pairs top-view quadrat photographs with destructively measured dry biomass components, creating a controlled setting in which deep vision models can be evaluated for in-situ biomass estimation.

Deep convolutional networks (CNNs) and Vision Transformers (ViTs) provide complementary inductive biases for this task. CNNs emphasise local texture through locality and translation equivariance, whereas ViTs model long-range dependencies through global self-attention. Agricultural and remote-sensing applications have adopted both families extensively for classification and segmentation, but their relative behaviour on continuous, multi-component, ground-level biomass regression—especially on small field datasets—remains insufficiently established. This thesis therefore investigates that architectural question under a controlled experimental design.

## 1.2 Motivation

The motivation for this study arises from three practical and scientific needs.

First, livestock managers require more than a single total-biomass number. Feed quality, grazing pressure, and clover contribution depend on botanical composition. A method that predicts green, dead, and clover dry matter jointly is therefore more operationally useful than a single aggregated score.

Second, although deep learning has become common in agricultural vision, fair architectural comparison is often confounded by mismatched pipelines: different preprocessing, losses, fine-tuning schedules, data splits, or input modalities. Without a controlled benchmark that isolates the backbone while holding everything else fixed, it is difficult to answer a simple but consequential question: for ground-level pasture biomass regression, should one prefer local CNN texture features or global transformer attention?

Third, deployment constraints matter. A high-capacity model that is accurate on a server may be unsuitable for edge or mobile use, while a lightweight CNN may be preferred if its accuracy–compute trade-off is acceptable. Practitioners therefore need not only a leaderboard ranking, but also statistically honest uncertainty estimates, interpretability checks, and concrete guidance for choosing among models under different compute budgets.

Together, these motivations define the scope of the present work: an image-only, empirically rigorous architectural benchmark and ablation study for multi-component pasture biomass estimation, rather than the proposal of a new network architecture or loss function.

## 1.3 Problem Statement and Regression Challenges

This thesis addresses the following problem. Given a single top-down RGB photograph of a pasture quadrat, estimate the dry biomass of three botanical fractions—green grass, dead matter, and clover—in grams, and reconstruct two physically consistent aggregates: Green Dry Matter (GDM = Green + Clover) and Total biomass (Total = Green + Dead + Clover). The method is deliberately restricted to the image alone; available NDVI and canopy-height metadata are not used, so that the evaluation isolates what visual architecture can achieve from commodity photographs.

Framing this task as multi-target regression exposes four intrinsic difficulties, all confirmed in the CSIRO Image2Biomass competition training split used in this study (357 labelled images):

1. **Heavy right-skew.** Green, dead, and clover components are right-skewed (skew approximately 1.7–2.8). A small number of overgrown swards—with Total biomass up to 185.7 g—dominates squared-error gradients and can distort learning if the loss is not carefully designed.
2. **Zero-inflated clover.** Clover is exactly zero in 37.8% of samples and is the most skewed component (skew 2.83), creating a sparse, hard-to-predict target.
3. **Dead–soil spectral overlap.** Senescent thatch shares brown reflectance with dry bare soil, so colour alone is an unreliable cue for the dead-matter fraction.
4. **Vertical occlusion.** Erect green canopies shade understory clover and mulch, hiding mass that contributes to the laboratory labels but is only partially visible in a top-down photograph.

Prior agricultural vision work has largely targeted classification, segmentation, or single-variable yield estimation [8]–[13]. A controlled benchmark that isolates architectural inductive biases for simultaneous multi-component biomass regression—validated with statistical confidence intervals and explainability methods—has been missing. That gap is the focus of this thesis.

## 1.4 Objectives and Project Outcome

The aim of this study is to provide a reproducible, statistically honest answer to which deep vision architecture is preferable for image-only multi-component pasture biomass estimation, and under what deployment constraints.

The specific objectives are to:

1. Benchmark five ImageNet-pretrained vision backbones—MobileNet (V1), EfficientNet-B0, EfficientNet-B3, DenseNet-121, and Vision Transformer ViT-B/16—under one identical training and evaluation pipeline on the CSIRO Image2Biomass dataset.
2. Predict the three independent botanical fractions (green, dead, clover) in raw grams and reconstruct GDM and Total by physical identity, thereby enforcing conservation constraints by construction.
3. Quantify performance using the competition’s global weighted R² metric, supplemented by per-target R², RMSE, and MAE, with 15-loop bootstrap 95% confidence intervals on a shared held-out test set.
4. Conduct controlled ablations on representative CNN and transformer models to isolate the effect of major design choices (target scale, loss weighting, augmentation, unfreezing, and BatchNorm behaviour).
5. Audit biological grounding with Grad-CAM, SHAP, and LIME, and translate the accuracy–compute findings into a practical three-tier deployment guide.

The intended outcome is not a new architecture, but evidence that practitioners and researchers can use to choose among existing ones for this task.

## 1.5 Research Questions

This thesis is organised around three research questions:

- **RQ1 (Locality vs. Global Attention):** Do CNN locality priors or ViT global self-attention better regress non-contiguous botanical fractions from canopy imagery?
- **RQ2 (Representation Sharing):** How do backbones trade off across green, dead, clover, GDM, and total biomass, and which architecture minimises harmful cross-target interference?
- **RQ3 (Accuracy vs. Cost):** What is the accuracy–compute frontier across lightweight edge-oriented and high-capacity server-oriented backbones?

## 1.6 Research Contributions

This work is deliberately empirical rather than methodological. We contribute no new architecture or loss function. Instead, we provide a controlled, honestly validated benchmark of existing models—the evidence needed to choose a backbone for multi-component pasture biomass estimation. To our knowledge, this is the first study that pairs a controlled CNN-versus-transformer comparison for multi-component biomass with both statistical (bootstrap confidence interval) and interpretability (XAI) validation.

The principal contributions are:

1. **Controlled five-architecture benchmark.** All models share an identical pipeline—raw-gram targets, attention-based regression head (CBAM for CNNs; pooled [CLS] token for ViT), composite loss (0.35·MAE + 0.65·RMSE), and depth-relative two-stage fine-tuning with a fixed seed—yielding an apples-to-apples locality-versus-attention comparison for joint green/dead/clover regression.
2. **Statistical rigor with honest uncertainty.** Performance is reported with 15-loop bootstrap 95% confidence intervals on a shared held-out split. Interval overlap is reported transparently rather than overstated as statistical significance—a discipline largely absent from single-number agricultural leaderboards.
3. **Interpretability audit.** Grad-CAM (CNNs) and model-agnostic SHAP/LIME (all models, including ViT) are used to inspect biological grounding, including an honestly reported negative result on a naive attention-coverage metric that future work should replace with faithfulness curves.
4. **Ablation and deployment guidance.** A two-model ablation isolates each major design choice across both a CNN and a transformer family, and an accuracy–compute Pareto view is mapped to concrete operational tiers with a single recommended default for practical deployment.

These are framed as empirical contributions to pre-empt the expectation of algorithmic novelty: the value of this thesis is a rigorous, reproducible answer to “which vision architecture, and why,” not a new model.

## 1.7 Scope and Delimitations

The empirical scope of this thesis is intentionally bounded so that claims remain commensurate with the evidence.

- **Task and modality.** The study is restricted to image-only multi-target regression from top-down RGB quadrat photographs. NDVI, canopy height, and other sensor metadata are available in the source dataset but are excluded by design.
- **Data.** Experiments use the labelled CSIRO Image2Biomass competition training split (357 unique images across four Australian states, all from 2015). The larger published collection and the unlabelled competition test images are outside the supervised evaluation reported here.
- **Models and input resolution.** Five backbones are compared at a uniform 224×224 input resolution. No new architecture, fusion block, or loss function is proposed.
- **Evaluation protocol.** A single held-out split (seed 42; 72-image test set) is used for all models, with bootstrap confidence intervals rather than k-fold cross-validation. This choice prioritises fair cross-model comparability and computational tractability; it also implies that pairwise differences should be interpreted cautiously when intervals overlap.
- **Interpretation limits.** Predicted biomass values are statistical estimates from visual appearance. They are not direct measurements of nutritive value, animal intake, or farm profitability, and should not be treated as causal economic indicators.

These delimitations keep the study focused on architectural comparison under controlled conditions and leave metadata fusion, larger multi-year datasets, and stronger statistical protocols as directions for future work.

## 1.8 Thesis Organization

The remainder of this thesis is organised as follows.

- **Chapter 2** reviews related work on ground-level agricultural vision, CNN versus transformer inductive biases, applications of ViTs in agronomy and remote sensing, loss functions for heavy-tailed regression, and explainable AI in agriculture, and then states the research gap.
- **Chapter 3** describes the CSIRO Image2Biomass dataset, target statistics, multi-target formulation with conservation identities, the rationale for raw-gram training, and the data split.
- **Chapter 4** presents the unified methodology: backbone selection, attention regression head, composite loss, two-stage fine-tuning, augmentation policy, and the evaluation framework.
- **Chapter 5** reports the architectural benchmark results, per-target errors, paired-loop comparisons, and controlled ablations.
- **Chapter 6** discusses explainability findings, architectural diagnostics, the dead-matter bottleneck, and accuracy–compute deployment guidance.
- **Chapter 7** concludes by answering the research questions, stating limitations, outlining future work, and reflecting on ethical and societal implications.

---

*Note for editing: citation numbers [1]–[13] follow the IEEE paper draft (`paper_v2.md` / `ref.bib`). Remap them to your final thesis bibliography when pasting into the document.*
