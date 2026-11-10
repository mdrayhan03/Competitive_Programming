# Paper Assets — paper_v2 ("Tokens vs. Textures")

Every figure and table source used in `paper_v2.md` / `paper_v2_IEEE_conference.docx`,
collected here so nothing has to be hunted for across folders.

## figures/  (all 7 figures, 17 image files)

| File | Paper | What it is |
|---|---|---|
| Fig1_architecture.png | Fig. 1 | Model pipeline diagram |
| Fig2_ci_intervals.png | Fig. 2 | Weighted-R² with 95% bootstrap CIs |
| Fig3a_MobileNet_loss.png | Fig. 3a | MobileNet (V1) train/val loss (main+ft) |
| Fig3b_EfficientNetB0_loss.png | Fig. 3b | EfficientNet-B0 loss |
| Fig3c_EfficientNetB3_loss.png | Fig. 3c | EfficientNet-B3 loss (largest gap) |
| Fig3d_DenseNet121_loss.png | Fig. 3d | DenseNet-121 loss |
| Fig3e_ViT_loss.png | Fig. 3e | ViT-B/16 loss (smallest gap) |
| Fig4_per_target_r2.png | Fig. 4 | Per-target R² bar chart |
| Fig5a_DenseNet_Green_gradcam.png | Fig. 5a | Grad-CAM, DenseNet Green (focused) |
| Fig5b_EffB0_Green_gradcam.png | Fig. 5b | Grad-CAM, EffB0 Green |
| Fig5c_EffB0_Dead_gradcam.png | Fig. 5c | Grad-CAM, EffB0 Dead (scattered onto soil) |
| Fig5d_ViT_attention.png | Fig. 5d | ViT attention-rollout (uninformative) |
| Fig6a_EffB0_Green_SHAP.png | Fig. 6a | SHAP, EffB0 Green |
| Fig6b_EffB0_Green_LIME.png | Fig. 6b | LIME, EffB0 Green |
| Fig6c_ViT_Green_SHAP.png | Fig. 6c | SHAP, ViT Green |
| Fig6d_ViT_Green_LIME.png | Fig. 6d | LIME, ViT Green (distributed canopy) |
| Fig7_pareto.png | Fig. 7 | Accuracy–compute Pareto frontier |

## tables_data/  (source data behind each table)

| File | Paper | What it is |
|---|---|---|
| Table1_dataset_raw_train.csv | Table 1 | Raw CSIRO train.csv → dataset statistics (357 imgs) |
| Table3_leaderboard_CI_bootstrap.xlsx | Table 3, Fig 2 | Best R², 95% CI, σ, and the 15 per-loop R² (also source of the paired-significance 10/10/9/13-of-15) |
| Table4_*_per_target_metrics.csv (×5) | Table 4 | Per-target R²/RMSE/MAE, best stage per model |
| Table4b_*_detailed_errors.csv (×5) | Table 4b | Per-image true/pred/error; GDM=Green+Clover recomputed from Green & Clover columns |
| Table6_ablation_EfficientNetB0.csv | Table 6 | Ablation on EfficientNet-B0 |
| Table6b_ablation_ViT.csv | Table 6b | Ablation on ViT-B/16 |
| Table7_xai_attention_on_vegetation.csv | Table 7 | Grad-CAM attention-on-vegetation metric |

## Derived tables (no separate source file)

- **Table 2** (architecture specs / params / GFLOPs) — standard published values, written directly in the paper.
- **Table 4b GDM column** — recomputed from the `Table4b_*_detailed_errors.csv` Green+Clover columns (matches Table 4 GDM R²).
- **Table 5** (best model per axis) — a summary distilled from Tables 3 & 4.

## Best stage per model (used for Table 3 "Best R²", Table 4, Table 4b)
ViT-B/16 = ft · EfficientNet-B0 = main · DenseNet-121 = ft · EfficientNet-B3 = main · MobileNet (V1) = ft
