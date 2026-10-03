# Future Work & Novelty Options — Pasture Biomass Thesis

> **Purpose.** The IEEE paper (`paper_v2.md`) is a supervisor-scoped **benchmark + ablation study with
> no architecture novelty**. The thesis can go further. This file lists **7 concrete ways to make the
> research novel**, each with the exact contribution claim it unlocks, how to do it, how to prove it
> worked, effort, and whether it needs data/supervisor sign-off — plus **recommended "novelty story"
> packages** that combine them into a coherent thesis contribution.
>
> Companion file: `future_novelty_ideas.md` (earlier notes). This file supersedes and formalizes it.

---

## Legend
- **Gate 🟢 SAFE** = rigor/analysis only (no architecture novelty) → allowed even under the "no novelty" rule.
- **Gate 🟡 APPROVAL** = adds *method* novelty the paper deliberately avoided → **confirm with supervisor**
  before doing it in the thesis.
- **Data?** = whether it needs a fresh Kaggle training run, or reuses existing saved models/files.

---

## The 7 novelty options (ranked by value ÷ effort)

### 1. ⭐ Zero-inflated "hurdle" head for clover  — Gate 🟡 APPROVAL · Data: needs 2 retrains
- **Novelty claim:** *"We identify clover (37.8% exact zeros) as the dominant failure mode and fix it with
  a two-branch zero-inflated head."*
- **How:** split clover into (a) a **presence classifier** `p(clover > 0)` (sigmoid, BCE loss) and (b) an
  **amount regressor** `E[clover | clover > 0]` (composite loss, masked to positive samples); predict
  `ŷ_clover = p × amount`. One extra sigmoid branch on the existing head; train the two jointly.
- **Prove it worked:** Clover R² improves vs. the plain regressor on 1–2 models (e.g. ViT + EffB0).
- **Effort:** ~½ day code + 2–3 short retrains. **Highest value-per-effort novelty.**

### 2. Image + metadata fusion ablation  — Gate 🟡 APPROVAL · Data: needs 1 run
- **Novelty claim:** *"We quantify exactly what image-only estimation sacrifices by ignoring the dataset's
  free NDVI and canopy-height sensors."*
- **How:** a small **fusion branch** concatenating `Pre_GSHH_NDVI` + `Height_Ave_cm` with the pooled image
  features before the MLP head.
- **Prove it worked:** report **image-only vs. image+metadata** weighted-R² gap (how much the camera alone loses).
- **Effort:** low–medium. Nice clean "multimodal" story; could headline a follow-up.

### 3. Deletion / insertion faithfulness curves  — Gate 🟢 SAFE · Data: reuses saved models (no retrain)
- **Novelty claim:** *"We replace the saturated attention-coverage metric with a faithfulness metric that
  actually discriminates the models."*  → **turns the paper's XAI null result into a positive method.**
- **How:** progressively **mask** (deletion) or **restore** (insertion) the most-salient pixels from the
  Grad-CAM/SHAP map and record the prediction drop/recovery; summarize as area-under-curve (AUC).
- **Prove it worked:** the curves separate models where the coverage metric (Table 7) could not.
- **Effort:** low–medium. **Best weakness→strength conversion in the whole list.**

### 4. Dead-vs-soil texture disambiguation  — Gate 🟡 APPROVAL · Data: needs runs
- **Novelty claim:** *"We tackle the universal dead-matter ceiling (R² ≤ 0.33) with a texture/edge cue."*
- **How:** feed a **texture channel** (Local Binary Pattern / Sobel) alongside RGB, or add an **auxiliary
  segmentation loss** separating dead vegetation from bare soil.
- **Prove it worked:** Dead R² rises; Grad-CAM heat moves off soil onto dead stalks.
- **Effort:** medium–high (harder, riskier — brown dead grass ≈ brown soil is genuinely hard).

### 5. k-fold cross-validation  — Gate 🟢 SAFE · Data: needs 5× runs
- **Novelty claim:** *"We move beyond a single split to k-fold CV, giving the first statistically robust
  ranking of these architectures on the task."*
- **How:** 5-fold CV; report fold-wise mean ± std and a proper significance test (e.g. paired t / Wilcoxon).
- **Prove it worked:** tighter, defensible intervals; directly answers the "single-split" reviewer objection.
- **Effort:** medium (5× the training compute across five backbones).

### 6. Cost / carbon efficiency analysis  — Gate 🟢 SAFE · Data: uses existing FLOPs
- **Novelty claim:** *"We add an energy/carbon axis to the accuracy–compute trade-off — a Green-AI
  deployment view of agricultural vision."*
- **How:** measure **train + inference energy** (kWh, gCO₂e) per model alongside the existing FLOPs/params;
  extend the Pareto frontier with a sustainability dimension.
- **Prove it worked:** a 3-axis (accuracy / compute / carbon) comparison and updated deployment tiers.
- **Effort:** low. Timely; fills Discussion (Ch. 6) with something fresh.

### 7. Uncertainty-aware regression  — Gate 🟡 APPROVAL · Data: needs runs
- **Novelty claim:** *"The model reports per-prediction confidence, not just a point estimate."*
- **How:** a **heteroscedastic loss** predicting a variance per target (or MC-dropout / deep-ensemble
  uncertainty); optionally fuse multiple crops/views of the quadrat.
- **Prove it worked:** calibration plot; uncertainty correlates with error.
- **Effort:** high — realistically a **separate paper**, not this thesis.

---

## Recommended novelty stories (pick ONE narrative, not scattered add-ons)

### 🥇 Story A — "A benchmark that finds the failures — and fixes them"  *(recommended)*
**Core:** the existing rigorous benchmark → **#1 (clover hurdle head)** + **#3 (faithfulness curves)** +
**#2 (metadata fusion)**.
- **Why it's strong:** converts *"we compared models"* → *"we compared models, diagnosed the two failure
  modes (clover zeros, dead–soil), **fixed clover**, quantified the image-only cost, and delivered a
  **faithfulness-verified** interpretability method."* Cheap, coherent, and genuinely novel.
- **Gate:** #1 and #2 need 🟡 supervisor approval; #3 is 🟢 safe.

### 🥈 Story B — "Rigor-first honest benchmark"  *(if supervisor keeps the no-novelty rule)*
**Core:** benchmark → **#3 (faithfulness curves)** + **#5 (k-fold CV)** + **#6 (cost/carbon)**.
- **Why:** all three are 🟢 **SAFE** (analysis/rigor, not architecture novelty). Still meaningfully
  upgrades the thesis: statistically robust ranking, a working interpretability metric, and a Green-AI angle.
- **Gate:** none — do it even under the strict rule.

### 🥉 Story C — "Toward the full multi-component solver"  *(most ambitious)*
**Core:** benchmark → **#1 (clover)** + **#4 (dead–soil)** + **#7 (uncertainty)**.
- **Why:** attacks *all three* weak spots (clover, dead matter, confidence). Biggest contribution, but
  highest risk/effort — closer to a journal extension than a thesis add-on.
- **Gate:** all 🟡 approval; realistically needs more time.

---

## Suggested sequencing
1. **#3 faithfulness curves** — do first: safe, no retrain, fixes the biggest weakness.
2. **#1 clover hurdle head** — the flagship novelty (once approved).
3. **#2 metadata fusion** — cheap, strengthens the story.
4. **#6 cost/carbon** — low effort, fills Discussion.
5. **#5 k-fold** — if compute allows; answers the single-split objection.
6. **#4 dead–soil / #7 uncertainty** — only if time; otherwise list as genuine future work.

---

## ⚠️ Decision gate before implementing
The paper is capped at **no architecture novelty** (supervisor). Options **#1, #2, #4, #7 add method
novelty** → **get supervisor sign-off** before putting them in the thesis. Options **#3, #5, #6 are rigor/
analysis** → safe to add regardless. Recommended default: **Story A if approved, Story B if not.**

---
*These options build on already-verified results and existing saved models/data; none require restarting
the project. Full config + code hooks live in `shareable_context.md` / `paper_v2.md`.*
