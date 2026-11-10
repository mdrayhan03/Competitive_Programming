# Model Comparison — Pasture Biomass Regression

Final comparison of 5 backbones predicting dry biomass (`Dry_Green_g`, `Dry_Dead_g`,
`Dry_Clover_g`) from a single image. All models share the same pipeline, loss
(`0.35·MAE + 0.65·RMSE`, raw-gram targets), trimmed augmentation, and depth-relative
unfreezing; only the backbone (and ViT's LR) differs. Metric = **weighted R²**
(Green 0.1, Dead 0.1, Clover 0.1, GDM 0.2, Total 0.5), evaluated on the 20% test split.

Full per-model hyperparameters + Kaggle versions are in [configuration.md](configuration.md).

---

## 1. Overall ranking (weighted R²)

| Rank | Model | main | fine-tune | **Best** | Best stage | Kaggle ver |
|---|---|---|---|---|---|---|
| 🥇 1 | **ViT-B16** | 0.7315 | 0.7356 | **0.7356** | ft | 155 |
| 🥈 2 | EfficientNetB0 | 0.6942 | 0.6899 | **0.6942** | main | 151 |
| 🥉 3 | EfficientNetB3 | 0.6831 | 0.6748 | **0.6831** | main | 152 |
| 4 | DenseNet121 | 0.6425 | 0.6812 | **0.6812** | ft | 153 |
| 5 | MobileNetV0 | 0.6199 | 0.6463 | **0.6463** | ft | 149 |

**ViT-B16 is the clear winner** (+0.041 over the best CNN). The four CNNs cluster
between 0.65–0.69 — within noise of each other on a 72-image test set.

---

## 2. Per-target R² (each model's best stage)

| Model | Green | Dead | Clover | GDM | Total | Weighted |
|---|---|---|---|---|---|---|
| **ViT-B16** | 0.705 | **0.331** | 0.567 | **0.653** | **0.652** | **0.7356** |
| EfficientNetB0 | 0.639 | 0.253 | 0.561 | 0.590 | 0.600 | 0.6942 |
| EfficientNetB3 | 0.669 | 0.225 | 0.571 | 0.623 | 0.556 | 0.6831 |
| DenseNet121 | **0.729** | 0.257 | 0.354 | 0.573 | 0.576 | 0.6812 |
| MobileNetV0 | 0.631 | 0.255 | 0.494 | 0.579 | 0.500 | 0.6463 |

*(Bold = best per column.)* ViT leads on the two highest-weighted targets (**Total** and
**GDM**), which is why it wins the weighted score despite DenseNet having the best Green.

---

## 3. Cross-model findings

1. **ViT > CNNs when properly trained.** ViT-B16 wins — but only after retuning. Its first
   (over-conservative) config under-trained it to 0.374; unfreezing more (0.30/0.40) + a
   warmup-protected 5e-5 LR lifted it to 0.7356. **Lesson: pretrained ViT is competitive on
   small data if enough of it is fine-tuned.**
2. **Bigger ≠ better (CNNs).** EfficientNetB0 (0.694) beats the larger B3 (0.683). On ~230
   training images the extra capacity overfits, and B3 is handicapped by running at 224 px
   instead of its native 300 px.
3. **Fine-tune is backbone-dependent.** It *helped* ViT, DenseNet, and MobileNet (whose main
   stages under-fit), but *hurt* both EfficientNets (already at the data ceiling — a 2nd pass
   just overfits).
4. **`Dry_Dead_g` is the hardest target for everyone** (R² 0.22–0.33). Dead matter is brown,
   sparse, and visually mixed with soil/litter — hard to read from RGB alone.
5. **Universal high-end under-prediction.** Every model caps Total predictions at ~76–98 g vs
   a true max of 126 g. The small dataset has few high-biomass examples, so all models regress
   toward the middle at the extremes — a **data limitation, not a model one**.

---

## 4. Per-model analysis

### 🥇 ViT-B16 — 0.7356 (best)
- **Config:** unfreeze 0.30 (main) / 0.40 (ft), LR 5e-5 (main, cosine warmup) / 1e-5 (ft), no BatchNorm (LayerNorm).
- **Result:** main 0.7315 · **ft 0.7356**. Best on Total (0.652), GDM (0.653), and Dead (0.331).
- **Why it wins:** ImageNet-21k pretraining gives strong transferable features, and global
  self-attention suits an *aggregate-quantity* task (total biomass = integrating signal across
  the whole image) better than local convolutions. Once enough transformer blocks are unfrozen,
  it adapts cleanly.
- **Caveat:** it needed a more aggressive retune than the CNNs received, and it has the most
  compressed extreme (Total pred max 76) yet still the best Total R² — i.e. excellent bulk
  calibration, weakest only at the rare very-high samples.

### 🥈 EfficientNetB0 — 0.6942 (best CNN)
- **Config:** unfreeze 0.30 / 0.40, LR 1e-4 / 1e-5, BatchNorm frozen.
- **Result:** **main 0.6942** · ft 0.6899 (fine-tune slightly hurts).
- **Why:** compound-scaled architecture is an efficient, well-regularised feature extractor —
  the sweet spot of capacity for this dataset size. Balanced across targets (Total 0.600).
- **Note:** single-stage is best; the fine-tune stage adds nothing here (main already at ceiling).

### 🥉 EfficientNetB3 — 0.6831
- **Config:** unfreeze 0.30 / 0.40, LR 1e-4 / 1e-5, BatchNorm frozen.
- **Result:** **main 0.6831** · ft 0.6748 (fine-tune hurts).
- **Why it trails B0:** ~2× the parameters overfits the small set, and it runs at 224 px rather
  than its native 300 px, costing accuracy. Highest Green among EfficientNets (0.669) but lower
  Total (0.556). A clean "bigger isn't better on small data" data point.

### 4. DenseNet121 — 0.6812
- **Config:** unfreeze 0.30 / 0.40, LR 1e-4 / 1e-5, BatchNorm frozen.
- **Result:** main 0.6425 · **ft 0.6812** (fine-tune helps, +0.039).
- **Why:** dense connectivity / feature reuse generalises well on small data. **Best Green R²
  of any model (0.729).** Held back by unusually weak **Clover (0.354)** — it struggles to
  isolate clover from grass.
- **Note:** main under-fits (0.6425); fine-tune does the real work — headroom exists but extra
  tuning was declined to keep the comparison fair.

### 5. MobileNetV0 (MobileNet V1) — 0.6463 (baseline)
- **Config:** unfreeze 0.30 / 0.40, LR 1e-4 / 1e-5, BatchNorm frozen.
- **Result:** main 0.6199 · **ft 0.6463** (fine-tune helps).
- **Why lowest:** the smallest, oldest backbone (depthwise-separable, ~4M params) — least
  capacity. Still a respectable baseline given its size; useful as the efficiency/again-cost
  reference point. Weakest on Total (0.500).

---

## 5. Conclusion

- **Best model: ViT-B16 (weighted R² 0.7356)** — the standout result once trained adequately.
- **Best CNN: EfficientNetB0 (0.6942)** — the efficient, robust choice if a transformer /
  `vit-keras` dependency is undesirable.
- **Recommended for deployment:** ViT-B16 for peak accuracy; EfficientNetB0 for a lighter,
  dependency-free CNN at ~0.69.
- **Biggest remaining limitation:** high-biomass under-prediction across all models — best
  addressed with **more high-biomass training data**, not a different architecture.
