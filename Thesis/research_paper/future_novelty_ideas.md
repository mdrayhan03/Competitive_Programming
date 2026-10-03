# Future Novelty / Method Ideas — DEFERRED (do later, NOT for this paper)

**Context:** Supervisor's instruction is **no architecture novelty** → this paper is an
**architectural benchmark + ablation study** (single split). The ideas below are **method
contributions for a *future* paper** (a "v2" that adds novelty), kept here so they aren't lost.
Do NOT implement these for the current benchmark paper.

---

## Idea 1 — Zero-inflated / hurdle head for clover  ⭐ (cheapest, highest-value novelty)
- **Problem it targets:** Clover is **37.8% exact zeros** and the weakest target (R² ≈ 0.35–0.57).
  A plain regressor can't cleanly predict "exactly zero."
- **What it is:** Split clover into **two branches** — (a) a presence classifier *p(clover > 0)*,
  (b) an amount regressor *E[clover | clover > 0]* — and multiply: `ŷ_clover = p × amount`
  (a "hurdle" / zero-inflated model).
- **How to build:** one extra sigmoid output branch on the existing head; train the two branches
  jointly (BCE for presence + composite loss for amount, masked to positive samples).
- **Prove it worked:** show Clover R² improves vs the plain regressor on 1–2 models.
- **Effort:** ~½ day code + 2–3 short retrains. **Novelty gain: high** (genuine method that fixes a
  documented failure).

## Idea 2 — Dead-vs-soil disambiguation
- **Problem:** Dead matter fails for all models (R² ≤ 0.33) — brown dead grass ≈ brown soil.
- **What it is:** feed a **texture/edge cue** (e.g., Local Binary Pattern / Sobel channel) alongside
  RGB, or add an **auxiliary segmentation loss** that separates dead-vegetation from soil.
- **Prove it worked:** Dead R² improves; Grad-CAM heat moves off soil onto dead stalks.
- **Effort:** medium–high (harder than Idea 1). **Novelty gain: high but riskier.**

## Idea 3 — Image + metadata fusion (ablation → could become a contribution)
- **Problem/opportunity:** the CSIRO dataset ships **NDVI (`Pre_GSHH_NDVI`) and height
  (`Height_Ave_cm`)** that we deliberately ignore (image-only).
- **What it is:** a small fusion branch that concatenates NDVI/height with the image features.
- **Prove it worked:** show image-only vs image+metadata R² — quantifies how much the camera alone loses.
- **Effort:** low–medium. **Gain: moderate** (nice ablation; could headline a "multimodal" follow-up).

## Idea 4 — Uncertainty-aware / multi-view regression (bigger future work)
- Predict a **variance** per output (heteroscedastic loss) so the model reports confidence, or fuse
  **multiple views/crops** of the quadrat. Larger effort; a separate paper.

---

## Suggested sequencing (for the *future* novelty paper)
1. **Idea 1 (clover hurdle head)** — best novelty-per-effort; do first.
2. **Idea 3 (metadata fusion ablation)** — cheap, strengthens the story.
3. **Idea 2 (dead-matter)** — if time; harder.
4. **k-fold CV** — add for the novelty paper's rigor (this benchmark paper uses single split per supervisor).

**One-line reminder:** these turn "we benchmarked" → "we benchmarked, found the failure modes, and
**fixed clover** with a new head." Save for the next paper; not for the current benchmark submission.
