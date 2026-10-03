# Thesis — Table of Contents + Writing Guide (aligned to supervisor's LaTeX template)

**Title:** *Tokens vs. Textures: An Architectural Benchmark and Ablation Study of Deep Vision Models for
In-Situ Pasture Biomass Estimation*

**Authors:** MD Rahat Al Mamun (Dept. of EEE); MD Rayhan Hossain, Sumaiya Islam, Samiul Haque (Dept. of CSE)
**Institution:** Independent University, Bangladesh (IUB), Dhaka

> **Matched to the supervisor's template** (`Thesis_Report_Template/`, LaTeX `report`, 12pt, A4). Every
> chapter maps to an existing `.tex` file wired in `main.tex`. **Each entry below carries, inline, a plain
> one–two sentence description of what to write (according to our research) plus which table/figure to
> place.** Content comes from `paper_v2.md`; novelty/future-work is in `../future_work.md` (excluded here).

### Template facts to respect
- 7 chapters, fixed order (`main.tex`): Introduction → Literature Review → Dataset → **Methodology**
  (`proposed-method.tex`) → Results and Analysis → Discussion → Conclusion.
- **Continuous figure numbering** (`\counterwithout{figure}{chapter}`) → figures are Fig. 1–7 across the
  whole thesis. Tables number normally.
- Section depth = 3; chapter titles render "**Chapter-N : Title**"; references = `ref.bib` + IEEEtran.
- Existing front-matter files: `attestation.tex`, `evalcommittee.tex`, `acknow.tex`,
  `list-of-publications.tex`, `abstract.tex`.

---

## Front Matter  *(front-matter `.tex` files + `main.tex` preamble)*
- **Title Page** *(`main.tex`)* — title, author names + IDs, department, IUB, supervisor, month/year, degree line.
- **Attestation** *(`attestation.tex`)* — short "this is our own work, no plagiarism" statement + signatures. **Adapt to 4 authors** (template is single-author "I").
- **Evaluation Committee** *(`evalcommittee.tex`)* — blank examiner name/designation/signature slots.
- **Acknowledgement** *(`acknow.tex`)* — thank supervisor, CCDS lab, and department; one short paragraph.
- **List of Publications** *(`list-of-publications.tex`)* — cite our IEEE paper ("Tokens vs. Textures", authors, venue, status).
- **Abstract** *(`abstract.tex`)* — the whole thesis in ~250 words: problem, data, 5 models, best result (ViT 0.7455, CIs overlap), XAI, deployment.
- **Table of Contents / List of Figures / List of Tables** *(auto in `main.tex`)*.
- *(Optional) List of Abbreviations* — template has none; add if the committee wants (ViT, CBAM, GDM, RMSE, XAI…).

---

## Chapter 1 — Introduction  → `introduction.tex`
- **1.1 Background & Agricultural Context** — what pasture biomass is and why farmers need it (rotational grazing, feed budgeting, carbon). Big-picture, simple.
- **1.2 Motivation** — why deep learning (learned features) beats old hand-made indices like NDVI/ExG for this task.
- **1.3 Problem Statement & Regression Challenges** — the 4 hard things: skewed data, clover mostly zero (37.8%), dead grass looks like soil, tall grass hides understory.
- **1.4 Objectives & Project Outcome** — what we set out to do: fairly compare 5 vision models, image-only, and report honestly.
- **1.5 Research Questions** — RQ1 (local CNN vs global ViT), RQ2 (how the 5 targets share features), RQ3 (accuracy vs compute).
- **1.6 Research Contributions** — state clearly these are *empirical* contributions: controlled benchmark, honest bootstrap CIs, XAI audit, deployment guide (no new model).
- **1.7 Scope & Delimitations** — limits: image-only, 357 images, 224² input, single split — so we don't over-claim.
- **1.8 Thesis Organization** — one line per chapter saying what it covers.

## Chapter 2 — Literature Review  → `literature-review.tex`
- **2.1 Prior Studies**
  - **2.1.1 Ground-Level Agricultural Vision & Biomass Sensing** — past work: plate meters, NDVI/ExG/CIVE indices, segmentation; why they struggle in dense/variable canopies.
  - **2.1.2 CNN Priors vs. Self-Attention** — CNNs look locally (texture), Transformers look everywhere (global); explain the trade-off simply.
  - **2.1.3 ViT in Agronomy & Remote Sensing** — where transformers have been used on satellite/crop images, and that small datasets are hard for them.
  - **2.1.4 Loss Functions for Heavy-Tailed Data** — quick tour of MSE/MAE/Huber and why we use a composite RMSE-heavy loss.
  - **2.1.5 Explainable AI in Agriculture** — what Grad-CAM, SHAP, LIME are and why trust matters for farm decisions.
- **2.2 Challenges Faced** — summarize what makes fair multi-component biomass comparison hard (confounds: mismatched inputs, losses, loops).
- **2.3 Research Gap & Scope of Study** — the punchline: nobody has done a *fair, controlled, honest* CNN-vs-ViT benchmark for multi-part biomass — that gap is ours.

## Chapter 3 — Dataset  → `dataset.tex`
- **3.1 Dataset Description** — CSIRO Image2Biomass: 357 top-down photos, 4 Australian states, 2015; how ground truth was measured (clip → separate → oven-dry → weigh); note we use image only (ignore NDVI/height).
- **3.2 Dataset Statistics & Structural Challenges** — present the numbers; point out clover is 37.8% zeros and most skewed. → **Table 1 (dataset stats).**
- **3.3 Multi-Target Formulation & Conservation Identities** — predict green/dead/clover; reconstruct GDM = Green+Clover and Total = Green+Dead+Clover; note the confirmed G+C fix.
- **3.4 Target Scale & Raw-Gram Rationale** — explain we train on raw grams, not log1p, because log under-predicts high biomass after back-transform.
- **3.5 Data Split & Observational Unit** — seed 42, 64/16/20 split → 72-image test; resize to 224².

## Chapter 4 — Methodology  → `proposed-method.tex`
- **4.1 Proposed Method**
  - **4.1.1 Unified Pipeline Overview** — one paragraph describing the whole flow (image → aug → backbone → head → 3 fractions → reconstruct). → **Fig. 1 (pipeline).**
  - **4.1.2 Backbone Selection & Iso-Resolution Standardization** — the 5 models and why chosen (light→heavy, CNN→ViT); all run at 224² for fairness. → **Table 2 (specs/params/FLOPs).**
  - **4.1.3 Unified Attention Regression Head** — CNNs use CBAM over conv maps; ViT uses pooled [CLS]; same small MLP head after ("head parity").
  - **4.1.4 RMSE-Weighted Composite Loss** — the α·MAE + (1−α)·RMSE formula with α=0.35, true RMSE, and why RMSE-heavy protects big values; show the equation.
- **4.2 Training Protocol**
  - **4.2.1 Two-Stage Depth-Relative Fine-Tuning** — unfreeze top 30% then 40%, keep BatchNorm frozen; explain why.
  - **4.2.2 Data Augmentation & Signal Preservation** — flip + small rotation + light noise only; why we dropped color jitter/zoom.
  - **4.2.3 Reproducibility & Implementation** — seed 42, TensorFlow/Keras 2.18, single P100; models/code available on request.
- **4.3 Evaluation Framework**
  - **4.3.1 Statistical Validation Protocol** — single split + 15-loop bootstrap 95% CI; why this over k-fold (fair, cheaper, honest).
  - **4.3.2 Global Weighted R²** — the competition metric + its weights `[0.1,0.1,0.1,0.2,0.5]`; show the formula.
  - **4.3.3 Unweighted Per-Target R²** — why we also report each fraction separately.
  - **4.3.4 Root Mean Squared Error (RMSE)** — definition, in grams.
  - **4.3.5 Mean Absolute Error (MAE)** — definition, in grams.

## Chapter 5 — Results and Analysis  → `result.tex`
- **5.1 Architectural Benchmark & Model Selection**
  - **5.1.1 Leaderboard & Statistical Overlap** — ViT best (0.7455), all CIs overlap → best bet, not proven; include the mean-predictor baseline row. → **Table 3 + Fig. 2 (CI plot).**
  - **5.1.2 Paired-Loop Significance** — ViT wins most bootstrap loops (10/10/9/13 of 15) → consistent ordering, still not "proven".
  - **5.1.3 Training Dynamics & Overfitting** — read the loss curves: ViT smallest gap, EffB3 overfits, MobileNet under-fits. → **Fig. 3 (5 curves).**
- **5.2 Botanical Fraction Decomposition & Error**
  - **5.2.1 Per-Target R²** — which model wins each fraction: DenseNet best Green/GDM, ViT best Dead/Total; dead is hard for all. → **Table 4 + Fig. 4.**
  - **5.2.2 Absolute Error in Grams (RMSE / MAE)** — errors in real units per target; ViT lowest Total error (14.23 g). → **Table 4b.**
- **5.3 Controlled Ablation on Representative Backbones** — change one setting at a time; show augmentation and raw-targets matter most, and it holds on both a CNN and ViT. → **Tables 6 & 6b.**

## Chapter 6 — Discussion  → `discussion.tex`
- **6.1 Explainability Audit**
  - **6.1.1 Qualitative Saliency (Grad-CAM, SHAP, LIME)** — show the maps; models look at plants (good), dead-matter maps drift onto soil. → **Figs. 5–6.**
  - **6.1.2 Cross-Architecture Attention: Foliage vs. Soil** — describe what each model's maps reveal about its focus.
  - **6.1.3 Quantitative Faithfulness & Null Result** — be honest: "% attention on plant" ≈90% for all, but plants cover ≈90% anyway, so it can't separate models. → **Table 7.**
- **6.2 Architectural Diagnostics**
  - **6.2.1 Mechanistic Hypotheses: Locality vs. Global Attention** — why ViT may win (global mixing) and DenseNet wins green (local texture) — clearly marked as ideas, not proof.
  - **6.2.2 Per-Backbone Characterization** — one short paragraph each (ViT, DenseNet, EfficientNet, MobileNet), tied to the numbers. → **Table 5.**
  - **6.2.3 The Dead-Matter Bottleneck** — why every model fails on dead grass (brown dead ≈ brown soil).
- **6.3 Efficiency & Deployment**
  - **6.3.1 Accuracy–Compute Pareto Frontier** — accuracy vs compute; EffB0 and ViT are the smart choices. → **Fig. 7.**
  - **6.3.2 Three-Tier Deployment Guide & Recommended Default** — cloud/edge/mobile tiers + the one default: EfficientNet-B0.

## Chapter 7 — Conclusion  → `conclusion.tex`
- **7.1 Summary** — restate the findings in plain words and **directly answer RQ1/RQ2/RQ3** with our evidence.
- **7.2 Limitations** — small 72-image test, single year (2015), dead-matter ceiling, XAI-metric saturation; be honest.
- **7.3 Future Work** — point to `../future_work.md` (clover hurdle head, metadata fusion, faithfulness curves, k-fold).
- **7.4 Ethical Considerations & Use of Language Models** — public data, no personal data, and how AI tools were used in writing/analysis.
- **7.5 Societal, Environmental & Economic Impact** — how cheap camera-based biomass helps farming, environment, and cost.

---

## Back Matter
- **Bibliography** *(`ref.bib` + IEEEtran)* — the paper's [1]–[26], expandable.
- *(Optional) Appendices* — template has no appendix `\input`s; if wanted: A Full Hyperparameters · B Extra Training Curves · C Extended XAI Gallery · D Reproducibility (needs new `.tex` files + `\input`).

---

### Content-source map (paper_v2.md → thesis `.tex`)
| Thesis chapter (.tex) | Source in paper_v2.md |
|---|---|
| Introduction | Abstract + §I |
| Literature Review | §II (expanded) |
| Dataset | §III |
| Methodology (proposed-method) | §IV + §V |
| Results and Analysis | §VI + §VII.7 (ablation) |
| Discussion | §VII (diagnostics) + §VIII (XAI) + §IX (deployment) |
| Conclusion | §X + §XI |

**Newly written for the thesis (not in the paper):** §1.8 organization; fuller Ch 2; §7.4 ethics/LLM-use;
§7.5 impact; all front matter (incl. List of Publications = our IEEE paper).

### Quick figure/table placement (continuous figure numbering per template)
- **Fig. 1** → 4.1.1 · **Fig. 2** → 5.1.1 · **Fig. 3** → 5.1.3 · **Fig. 4** → 5.2.1 · **Fig. 5–6** → 6.1.1 · **Fig. 7** → 6.3.1
- **Table 1** → 3.2 · **Table 2** → 4.1.2 · **Table 3** → 5.1.1 · **Table 4** → 5.2.1 · **Table 4b** → 5.2.2 · **Table 5** → 6.2.2 · **Tables 6 & 6b** → 5.3 · **Table 7** → 6.1.3
