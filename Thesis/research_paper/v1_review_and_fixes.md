# v1 Paper Review — "Tokens vs. Textures"
Content corrections + per-section strength/weakness/improvement analysis.
Ground truth = the actual experiments in `Thesis/` (results are the source of truth).

---

## PART A — Content that is WRONG and must be fixed (methods must match the code/results)

| # | Where | Currently says (WRONG) | Fix to (REALITY) | Why |
|---|---|---|---|---|
| 1 | Abstract, §3.3 (Eq 4, Prop 1), Fig 1 caption | **log1p target transform** used | **RAW-gram targets** (`USE_LOG_TARGETS=False`); reframe §3.3 as an *ablation finding* ("log tested, raw + RMSE-weighting won") | We removed log; it hurt the high end. Results came from raw targets. |
| 2 | Abstract, §4.3 (Eq 15), §4.3.1, Table 5 | **α = 0.65** (65% MAE, 35% RMSE), "optimal" | **α = 0.35** (35% MAE, **65% RMSE**) | Real config; RMSE-heavy to punish high-biomass under-prediction. |
| 3 | §1, §2.2, §4.1.1, Table 2, §7.2.2, §9.1.3, Tables 3–4 | **MobileNetV3-Large** (5.4M, 0.45G) | **MobileNet V1** (`tf.keras.applications.MobileNet`, ~4.2M params, ~1.1 GFLOPs) | Code used V1. Update name, params, FLOPs, and the "occlusion" narrative. |
| 4 | §5.1, §1.2 | **5-fold cross-validation (80/20)** | **single 64/16/20 split + 15-loop bootstrap CI** | K-fold was commented out; CI came from bootstrap on one test set. |
| 5 | §3.2 (Eq 1, R⁵), §4.2, Fig 1 caption | model outputs **5** targets; head = GAP→SiLU→Dense(5) | model outputs **3** (Green, Dead, Clover); **CBAM (channel+spatial attention) head**→Dense(3); GDM/Total **reconstructed** (G+C, G+D+C) | Real architecture — and CBAM is a *stronger* contribution to state. |
| 6 | Tables 3–4 (MobileNet row) | FT Global R² = **0.6345**; per-target slightly off | FT R² = **0.6463**, CI **0.6443 ± 0.0682**; per-target from `mobilenetV0_result` (Green .631, Dead .255, Clover .494, GDM .579, Total .500) | Reconcile to actual result files. |
| 7 | Abstract, §6.1, §7.1 | ViT **"outperforms all CNN baselines"** (implies significance) | **"ViT leads on every measure and is most stable, though 95% CIs overlap"** | Your CI bands ALL overlap → NOT statistically separable. Overclaiming = reviewer red flag. |
| 8 | Table 5 (ablations) | values like MAE = 0.661 g (don't match real ~8.8 g MAE → look synthetic) | **Real ablation runs**, or clearly mark "to be regenerated" | Cannot publish fabricated ablations. |
| 9 | Table 1 (dataset stats) | Max Total 148.9 g (wrong); "2014–2017 three-year" | **REAL (computed from train.csv):** 357 images, 1785 pairs, 4 states, **2015 only (Jan–Nov)**, Max Total **185.70 g** | Numbers must be real. Date range is a hard factual error. |
| 10 | §3.3 Fig 2 caption, loss "65/35" wording | "combining 65% MAE and 35% RMSE" | "**35% MAE + 65% RMSE**" | Flows from fix #2. |

> Net effect: fixes #1, #2, #5 rewrite the *method story* from "log + MAE-heavy + 5-output MLP head" to
> "**raw targets + RMSE-heavy composite loss + CBAM attention head + 3-output reconstruction**" — which
> is your real, reproducible, and arguably *stronger* method. Fix #7 is the integrity-critical one.

---

## REAL Table 1 (computed from `train.csv` — use these exact numbers)

**Dataset:** 357 quadrat images, 1785 image–target pairs, 4 Australian states
(Tas 138, Vic 112, NSW 75, WA 32), 15 species/sward types, **collected in 2015 (15 Jan – 10 Nov)**.
Note: the CSV also contains `Pre_GSHH_NDVI` and `Height_Ave_cm` metadata columns which we
**deliberately do not use** (image-only method) — worth stating as a design choice.

| Biomass Variable | Mean (g) | Std (g) | Min (g) | Median (g) | Max (g) | % zero | Skew |
|---|---|---|---|---|---|---|---|
| Dry_Green_g | 26.62 | 25.40 | 0.00 | 20.80 | 157.98 | 5.0% | 1.74 |
| Dry_Dead_g | 12.04 | 12.40 | 0.00 | 7.98 | 83.84 | 11.2% | 1.75 |
| Dry_Clover_g | 6.65 | 12.12 | 0.00 | 1.42 | 71.79 | **37.8%** | **2.83** |
| GDM_g | 33.27 | 24.94 | 1.04 | 27.11 | 157.98 | 0.0% | 1.55 |
| Dry_Total_g | 45.32 | 27.98 | 1.04 | 40.30 | 185.70 | 0.0% | 1.42 |

Identities verified: **GDM = Green + Clover** (err ≈ 0) and **Total = Green + Dead + Clover** (err ≈ 0).

**Correction to the paper's skew claim:** "skewness > 2.5" is only true for **Clover (2.83)**; Green/Dead ≈ 1.75,
Total ≈ 1.42. Reframe as: *"Clover is extremely right-skewed (2.83) and zero-inflated (37.8% true zeros);
all components are right-skewed (1.4–2.8)."* This actually **strengthens** the zero-inflation motivation.

---

## PART B — Per-section Strength / Weakness / How-to-improve

| Section | Strength | Weakness | How to improve |
|---|---|---|---|
| **Title** ("Tokens vs. Textures") | Memorable, frames the CNN-vs-ViT hook | — | Keep. |
| **Abstract** | Complete, quantified, all contributions present | States log-transform + α=0.65 (wrong); overclaims ViT significance | Apply fixes #1,#2,#7,#10; add "differences within 95% CI" honesty. |
| **1. Introduction** | Strong agronomic motivation ($/ha), 4 clear vision challenges | Slightly long; MobileNetV3 mention | Fix #3; tighten challenge list to 3–4 crisp points. |
| **1.1 Research Questions** | Excellent — RQ1/RQ2/RQ3 give the paper a spine | RQ3 (Pareto) is weaker than RQ1/RQ2 | Keep RQ1 (locality vs attention) as the headline; tie each result back to an RQ. |
| **1.2 Contributions** | Clear 3-part list | Claims "5-fold CV" and log-transform | Fix #1,#4; add CBAM head + honest-CI as explicit contributions. |
| **2.1 Related Work — Ag biomass** | Good coverage (NDVI, GrassClover, domain adaptation) | Could position the *gap* more sharply | Add 1 line: "no prior work benchmarks inductive biases for *multi-component* biomass." |
| **2.2 Evolution of architectures** | Clean CNN→ViT arc, cites the right papers | Mentions MobileNetV3 | Fix #3; add 1 recent ViT-for-regression or ViT-in-agriculture cite. |
| **2.3 Loss formulations** | Motivates composite loss well (L1 vs L2 vs Huber) | Sets up α=0.65 framing | Reframe toward RMSE-heavy rationale (fix #2). |
| **3.1 Dataset description** | Rich detail (19 sites, seasons, quadrat protocol) | Stats unverified (Table 1) | Fix #9 — recompute from train.csv; confirm N (images/pairs). |
| **3.2 Problem formulation** | Clean math; states conservation identities (GDM, Total) | Says model outputs R⁵ | Fix #5 — model outputs 3, GDM/Total derived; keep identities. |
| **3.3 Log pre-conditioning + Prop 1** | Rigorous (error-propagation proof) — impressive | **Describes a method you didn't use** (log) | Fix #1 — reframe: present skew problem + Prop 1 as *motivation*, then state raw+RMSE was chosen (log as ablation baseline). Keeps the math, fixes the honesty. |
| **4.1 Backbones (4.1.1–4.1.5)** | Good per-architecture detail + FLOPs | MobileNetV3 (4.1.1); ConvNeXt mentioned but not in suite | Fix #3; drop ConvNeXt references. |
| **4.2 Regression head** | Fair-comparison rationale is good | Omits CBAM; says Dense(5) | Fix #5 — describe channel+spatial attention head; this is a *selling point*. |
| **4.3 Composite loss + 4.3.1 gradient** | Strong — bounded-gradient proof is a real contribution | α=0.65 value; RMSE eps term framing | Fix #2 — set α=0.35; the gradient proof stays valid, just re-plot Fig 2 at α=0.35. |
| **4.4 Two-phase fine-tuning (Alg 1)** | Clear, reproducible algorithm block | Says "Stage 3/DenseBlock4" named unfreezing; N=358 | State **depth-relative unfreezing (top 30% main / 40% ft, BN frozen)** — your real method; note fine-tune helped/hurt per backbone (a finding!). |
| **5.1 CV strategy** | — | **Claims 5-fold CV (didn't happen)** | Fix #4 — single split + 15-loop bootstrap; explain why bootstrap CIs are the rigor. |
| **5.2 Augmentation** | Matches code (flip, rot 0.1, noise 0.02) ✓ | Doesn't say *why* trimmed (dropped color jitter) | Add 1 line: color jitter removed because greenness IS the signal. |
| **5.3 Evaluation metrics** | Correct weighted-R² formula + weights ✓ | — | Keep; note GDM uses Green+Clover explicitly. |
| **6.1 Leaderboard (Tables 3–4)** | Comprehensive, well-organized | MobileNet numbers off; overclaims dominance | Fix #6,#7; add a sentence on CI overlap. |
| **7.1 Why ViT wins (7.1.1–7.1.4)** | The paper's intellectual core — receptive-field + dynamic-filtering theory is compelling | Post-hoc; asserts causation the ablations don't prove | Frame as "we *hypothesize*/consistent with"; back the soil-suppression claim with the SHAP/attention evidence you actually have. |
| **7.2 CNN failure modes (7.2.1–7.2.3)** | Nice per-model diagnostics (B3 overfit, DenseNet clover saturation, Mobile occlusion) | MobileNetV3 framing; B3/Mobile claims need evidence | Fix #3; tie each claim to a curve or heatmap you actually have. |
| **7.3 Ablation studies** | Right idea (α, unfreeze depth, weight decay) | **Table 5 appears fabricated** | Fix #8 — run real ablations or clearly scope down; report α=0.35 as the winner. |
| **8. SHAP/LIME (8.1–8.2)** | Strong trust angle; Shapley formula; failure-case reading | Claims (e.g., ViT "uniform canopy coverage") need to match your real images; ViT attention map is actually uninterpretable | Use your real `CI_outputs/explanations` images; state ViT uses **LIME/SHAP** (attention map uninformative) — honest and matches your finding. |
| **9. Deployment tiers** | Practical "so what"; ties FLOPs to use-cases | Latency numbers (<35ms, <15ms) unmeasured; MobileNetV3 | Fix #3; either measure latency or soften to "estimated." |
| **10. Limitations + future** | Honest (monocular depth, occlusion); good roadmap (NIR, ZIM) | Doesn't mention small-test-set / CI-overlap limit | Add: "differences within CI on a small test set; larger data needed for significance." |
| **11. Conclusion** | Ties back to findings cleanly | Repeats overclaim | Mirror the honest-CI framing. |
| **References** | Solid, top-venue (ICLR, ICML, CVPR, NeurIPS, KDD) | A few generic; [17] Elharrouss loss survey, [16] SAM loosely relevant | Add 1–2 recent ViT-regression / agriculture-XAI cites; verify all DOIs. |

---

## Priority order for fixing
1. **Integrity-critical:** #7 (CI overlap / no overclaim), #8 (real ablations), #1 & #2 (methods match results).
2. **Factual:** #3 (MobileNet V1), #4 (single split + bootstrap), #5 (CBAM head + 3 outputs), #6 (MobileNet numbers), #9 (Table 1 stats).
3. **Polish:** tighten intro, tie results→RQs, real XAI image descriptions, latency caveats.

## The corrected method story (one paragraph, for reference)
> "Top-down RGB quadrat images (224×224) are fed to five ImageNet-pretrained backbones. Each CNN
> backbone (BatchNorm frozen) feeds a **CBAM channel+spatial attention head** → Dense(128, SiLU) →
> Dropout(0.3) → **Dense(3)** predicting Green/Dead/Clover in **raw grams**; GDM and Total are
> reconstructed by summation. ViT-B/16 uses its pooled [CLS] token into the same Dense head. Training
> minimizes a **composite loss (0.35·MAE + 0.65·RMSEε)** with AdamW, in two stages (top-30% then
> top-40% depth-relative unfreezing). Evaluation uses a single 64/16/20 split with **15-loop bootstrap
> 95% CIs** on the competition weighted-R²."
