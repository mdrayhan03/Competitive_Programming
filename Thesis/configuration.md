# Pasture Biomass — Model Configurations

Experiment log for the CSIRO biomass regression models (predicting `Dry_Green_g`,
`Dry_Dead_g`, `Dry_Clover_g` from a single image). One backbone per notebook; the
pipeline, loss, and targets are shared for a fair comparison.

**Status legend:** ✅ finalized · 🔄 pending run/re-run · ⬜ not run yet

---

## Summary table

| Model | Kaggle ver | Backbone | Preprocess | Main unfreeze | α (MAE:RMSE) | FT unfreeze | FT LR | FT α | R² main | R² ft | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MobileNetV0 | **149** | `MobileNet` | mobilenet ([-1,1]) | 0.30 | 0.35 (35:65) | 0.40 | 1e-5 | 0.35 | 0.6199 | **0.6463** | ✅ |
| EfficientNetB0 | **151** | `EfficientNetB0` | efficientnet ([0,255]) | 0.30 | 0.35 | 0.40 | 1e-5 | 0.35 | **0.6942** | 0.6899 | ✅ |
| EfficientNetB3 | **152** | `EfficientNetB3` | efficientnet ([0,255]) | 0.30 | 0.35 | 0.40 | 1e-5 | 0.35 | **0.6831** | 0.6748 | ✅ |
| DenseNet121 | **153** | `DenseNet121` | densenet (caffe) | 0.30 | 0.35 | 0.40 | 1e-5 | 0.35 | 0.6425 | **0.6812** | ✅ |
| ViT-B16 | **155** | `vit.vit_b16` | [-1,1] | 0.30 | 0.35 | 0.40 | 1e-5 | 0.35 | 0.7315 | **0.7356** | ✅ |

Overall best: **ViT-B16 (ft 0.7356)** after retuning — see its note below.
For EfficientNetB0, **main (0.6942)** is its best stage; fine-tune (0.6899) is kept for consistency
but does not beat single-stage on that backbone.

---

## Shared configuration (all models)

**Input / data**
- `IMG_SIZE` = 224 × 224, `BATCH_SIZE` = 16
- Targets: `['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g']`
- `USE_LOG_TARGETS` = **False** (train on RAW grams — aligns loss with the raw-gram metric)
- Split: 64% train / 16% val / 20% test, `SEED` = 42 (full determinism)

**Augmentation (trimmed)**
- `RandomFlip("horizontal_and_vertical")`, `RandomRotation(0.1)`, `GaussianNoise(0.02)`
- (dropped: color jitter, zoom, translation — they fight the biomass signal)

**Head**
- CNNs: CBAM (channel + spatial attention) → `Dense(128, silu)` → `Dropout(0.3)` → `Dense(3, linear)`
- ViT: pooled token → `Dense(128, silu)` → `Dropout(0.3)` → `Dense(3, linear)` (no CBAM — no 2D map)

**Loss**
- Composite `α·MAE + (1-α)·RMSE`, true RMSE (mean-before-sqrt)

**BatchNorm**
- Frozen everywhere in the CNN backbones (ViT uses LayerNorm — trainable)

**Optimizer / schedule**
- AdamW; main cosine warmup (horizon 50, peak = LEARNING_RATE) + EarlyStopping(val_loss, patience 7)
- Fine-tune: ReduceLROnPlateau(val_loss, factor 0.6, patience 10) + EarlyStopping(val_loss, patience 5), both `restore_best_weights`

**Unfreezing**
- Depth-relative: unfreeze top `UNFREEZE_FRACTION` of the backbone (main) / `FT_UNFREEZE_FRACTION` of the flattened model (fine-tune), BN skipped.

---

## Per-model details

### MobileNetV0 ✅  (Kaggle version 149)
- Backbone: `tf.keras.applications.MobileNet`, preprocess: `mobilenet.preprocess_input` → [-1, 1]
- Main: `UNFREEZE_FRACTION` 0.30, `LEARNING_RATE` 1e-4, `WEIGHT_DECAY` 1e-2, `ALPHA` 0.35
- Fine-tune: `FT_UNFREEZE_FRACTION` 0.40, `FT_LEARNING_RATE` 1e-5, `FT_WEIGHT_DECAY` 1e-4, `FT_ALPHA` 0.35
- **Result:** main R² 0.6199 · **ft R² 0.6463**

### EfficientNetB0 ✅  (Kaggle version 151)
- Backbone: `tf.keras.applications.EfficientNetB0`, preprocess: `efficientnet.preprocess_input` → [0, 255]
- Main: `UNFREEZE_FRACTION` 0.30, `LEARNING_RATE` 1e-4, `WEIGHT_DECAY` 1e-2, `ALPHA` 0.35
- Fine-tune: `FT_UNFREEZE_FRACTION` 0.40, `FT_LEARNING_RATE` 1e-5, `FT_WEIGHT_DECAY` 1e-4, `FT_ALPHA` 0.35
- **Result:** main R² **0.6942** (best) · ft R² 0.6899
- Note: fine-tune does not beat single-stage here; deep config kept (gentle 0.20/5e-6/α0.20 was worse at 0.6790).

### EfficientNetB3 ✅  (Kaggle version 152)
- Backbone: `tf.keras.applications.EfficientNetB3`, preprocess: `efficientnet.preprocess_input` → [0, 255]
- Main: `UNFREEZE_FRACTION` 0.30, `LEARNING_RATE` 1e-4, `WEIGHT_DECAY` 1e-2, `ALPHA` 0.35
- Fine-tune: `FT_UNFREEZE_FRACTION` 0.40, `FT_LEARNING_RATE` 1e-5, `FT_WEIGHT_DECAY` 1e-4, `FT_ALPHA` 0.35
- Note: heavier model — drop `BATCH_SIZE` to 8 if it OOMs.
- **Result:** main R² **0.6831** (best) · ft R² 0.6748
- Note: below EfficientNetB0 (0.6942) — bigger model overfits on this small set; fine-tune hurts (same as B0).

### DenseNet121 ✅  (Kaggle version 153)
- Backbone: `tf.keras.applications.DenseNet121`, preprocess: `densenet.preprocess_input` (caffe/torch)
- Main: `UNFREEZE_FRACTION` 0.30, `LEARNING_RATE` 1e-4, `WEIGHT_DECAY` 1e-2, `ALPHA` 0.35
- Fine-tune: `FT_UNFREEZE_FRACTION` 0.40, `FT_LEARNING_RATE` 1e-5, `FT_WEIGHT_DECAY` 1e-4, `FT_ALPHA` 0.35
- **Result:** main R² 0.6425 · **ft R² 0.6812** (best) — fine-tune helps here (like MobileNet)
- Note: 3rd overall (just behind B3 0.6831 and B0 0.6942). Best Green R² of any model (0.729); weak Clover (~0.35).

### ViT-B16 ✅  (Kaggle version 155)  — 🥇 BEST OVERALL
- Backbone: `vit_keras.vit.vit_b16` (pooled token feature), preprocess: map [0,1] → [-1, 1]
- Main: `UNFREEZE_FRACTION` 0.30, `LEARNING_RATE` 5e-5, `WEIGHT_DECAY` 1e-2, `ALPHA` 0.35
- Fine-tune: `FT_UNFREEZE_FRACTION` 0.40, `FT_LEARNING_RATE` 1e-5, `FT_WEIGHT_DECAY` 1e-4, `FT_ALPHA` 0.35
- Note: needs `vit-keras` + `tensorflow-addons`; no BatchNorm (LayerNorm).
- **Result:** main R² 0.7315 · **ft R² 0.7356** (best of all models).
- Note: the first (conservative 0.15/3e-5/0.15) config badly UNDER-trained ViT (ft 0.3736). Retuning
  to 0.30 unfreeze + 5e-5 LR (warmup-protected) + 0.40 fine-tune lifted it to 0.7356. ViT was under-trained,
  not incapable. Unfreeze now matches the CNNs (0.30/0.40); only LR differs (5e-5 vs 1e-4), a standard ViT choice.
