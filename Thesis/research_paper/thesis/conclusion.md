# Chapter 7 — Conclusion

This final chapter restates the thesis findings in plain terms, answers the research questions, acknowledges limitations, outlines future work, and briefly addresses ethics and broader impact.

---

## 7.1 Summary

This thesis conducted a controlled architectural benchmark and ablation study of five deep vision backbones — MobileNet (V1), EfficientNet-B0, EfficientNet-B3, DenseNet-121, and ViT-B/16 — for **image-only, multi-component pasture biomass estimation** on the CSIRO Image2Biomass competition training split (357 labelled quadrat images). All models shared one identical pipeline: raw-gram targets, an attention-based regression head (CBAM for CNNs; pooled [CLS] for ViT), a composite loss (0.35·MAE + 0.65·RMSE), depth-relative two-stage fine-tuning, trimmed augmentation, and seed 42. Green, dead, and clover were predicted directly; GDM and Total were reconstructed by physical identity (GDM = Green + Clover; Total = Green + Dead + Clover).

**Principal empirical findings.**

1. **ViT-B/16** attained the highest weighted R² (**0.7455**) and the lowest bootstrap variance (σ = **0.087**), and led on Total and Dead.
2. **DenseNet-121** was the best Green and GDM extractor (R² = **0.7288** / **0.7049**).
3. **EfficientNet-B0** was the efficiency-optimal choice (**0.7113** at **0.39 GFLOPs**).
4. **All 95% confidence intervals overlap** on the 72-image test set; ViT’s lead is *consistently preferable* (paired-loop wins in 9–13 of 15 draws), not statistically proven.
5. **Ablations** on EfficientNet-B0 and ViT agree: trimmed augmentation and raw-gram targets are the most critical design choices, followed by adequate backbone unfreezing; loss-α is least sensitive.
6. **Explainability** maps support plant-focused reasoning for Green and visualise dead–soil confusion for Dead; a quantitative attention-coverage metric was a null result (≈90% for all models ≈ chance on dense canopies).
7. **Deployment default:** EfficientNet-B0; escalate to ViT-B/16 when server compute is free and maximum accuracy matters.

### Answers to the research questions

**RQ1 (Locality vs. Global Attention).**  
On this dataset and pipeline, **ViT global self-attention is the consistently stronger choice for overall and hard targets** (weighted R², Total, Dead), while **CNN locality remains competitive — and superior — for contiguous green texture** (DenseNet on Green/GDM). The evidence supports a *task-dependent* answer, not a universal winner. Overlapping CIs preclude claiming statistical superiority of ViT.

**RQ2 (Representation Sharing).**  
Backbones trade off across targets: no model wins every column. ViT minimises harm on the high-weight aggregates and on Dead; DenseNet specialises in Green/GDM; EfficientNet-B3 edges Clover narrowly but loses on Total. Cross-target interference is therefore architecture-specific — choosing a backbone is choosing which botanical fraction to prioritise.

**RQ3 (Accuracy vs. Cost).**  
The accuracy–compute frontier is formed by **EfficientNet-B0** (edge) and **ViT-B/16** (cloud). EfficientNet-B3 and DenseNet are dominated; MobileNet is useful only under a hard parameter-size cap. The recommended single default is **EfficientNet-B0**.

---

## 7.2 Limitations

The following limits bound how far the conclusions may be generalised:

1. **Small held-out test set (72 images).** Bootstrap confidence intervals overlap; pairwise differences are not statistically significant. A larger test set or k-fold cross-validation would be required for stronger significance claims.
2. **Single year and geography (Australia, 2015).** Seasonal and inter-annual generalisation is untested. Models trained on this split may degrade under different lighting, species mixes, or management regimes.
3. **Image-only modality.** NDVI and canopy-height metadata exist in the source dataset but were excluded by design. The benchmark therefore understates what a multimodal system could achieve.
4. **Monocular occlusion.** Top-down RGB cannot see understory biomass under tall canopies; vertical structure is invisible.
5. **Dead-matter ceiling.** Dead–soil spectral overlap caps all models at Dead R² ≤ 0.33. This is a data/appearance limit more than an architecture limit.
6. **XAI-metric saturation.** The attention-on-vegetation coverage score equals chance on dense canopies and cannot rank models; qualitative maps remain the primary interpretability evidence.
7. **Empirical scope.** This thesis proposes no new architecture or loss. Its contribution is a controlled, honest comparison — valuable for practice, but not a methodological invention.

**Practical accuracy caveat.** Even the best Total RMSE (~14 g ≈ 31% of mean Total) supports coarse paddock ranking and screening, not precise per-quadrat feed budgeting without calibration against destructive samples.

---

## 7.3 Future Work

Future work can strengthen either the *rigour* of the present empirical study or, with supervisor approval, add *method* novelty that this thesis deliberately deferred.

**Rigor / analysis extensions (safe under the current empirical scope).**

- **Deletion/insertion faithfulness curves** to replace the saturated attention-coverage metric with a discriminative XAI measure.
- **k-fold cross-validation** (e.g. 5-fold) with paired significance tests for a statistically stronger ranking.
- **Energy / carbon accounting** alongside FLOPs, extending the Pareto view to a Green-AI deployment axis.

**Method directions (require explicit approval if framed as thesis novelty).**

- A **zero-inflated hurdle head** for clover (37.8% exact zeros): classify presence, then regress amount.
- **Image + metadata fusion** (NDVI, canopy height) to quantify what image-only estimation sacrifices.
- **Dead-vs-soil disambiguation** via texture/edge channels or auxiliary segmentation.
- **NIR / red-edge / LiDAR** fusion to reduce occlusion and spectral ambiguity.
- **Uncertainty-aware regression** (heteroscedastic heads or ensembles) so each prediction carries a confidence estimate.

A coherent next step that converts a documented weakness into a positive contribution is to replace the null coverage metric with faithfulness curves, then (optionally) add a clover hurdle head — both map directly onto failure modes already measured in Chapters 5–6.

---

## 7.4 Ethical Considerations and Use of Language Models

**Data and subjects.** The study uses the publicly released CSIRO Image2Biomass competition training imagery and laboratory biomass labels. The data contain no personally identifiable information about individuals. No animal experiments were conducted by the authors; biomass labels were produced by the dataset providers through standard agronomic clipping and drying protocols.

**Use of AI tools.** Large language models and coding assistants were used as productivity aids for literature organisation, LaTeX/Markdown drafting, debugging training scripts, and polishing prose. All experimental numbers reported in this thesis were computed from the authors’ training runs, result CSVs, and bootstrap notebooks, and were cross-checked against source files. Final scientific claims, model selection decisions, and interpretation of results remain the responsibility of the authors. AI-generated text was reviewed and edited for accuracy; the work was not produced by autonomous generation without human oversight.

**Responsible deployment.** Camera-based biomass estimates should not replace agronomic judgement or destructive calibration where feeding decisions have animal-welfare or financial consequences. Models may inherit geographic and seasonal biases from the 2015 Australian training distribution; deployment outside that domain requires local validation.

---

## 7.5 Societal, Environmental, and Economic Impact

Accurate, low-cost pasture biomass estimation supports three linked outcomes:

1. **Economic.** Better stocking-rate and rotational-grazing decisions can raise gross margins (prior literature cites on the order of \$52–96/ha under improved allocation). An RGB-camera workflow that screens high- and low-biomass zones reduces the need for wall-to-wall destructive sampling.
2. **Environmental.** Avoiding chronic overgrazing protects soil cover, reduces erosion risk, and supports grassland carbon management by keeping utilisation closer to sustainable targets.
3. **Accessibility.** Commodity cameras and edge-friendly models (especially EfficientNet-B0) lower the barrier relative to satellite indices that saturate at high LAI, or to specialised optical/capacitance sensors.

The present work contributes an evidence base for *which* vision architecture to choose under different compute budgets, with honest uncertainty and interpretability caveats — a practical step toward trustworthy precision-grazing tools rather than a finished farm product.

---

## 7.6 Closing Statement

Under one controlled pipeline, **ViT-B/16** is the strongest and most stable image-only pasture-biomass estimator on this benchmark; **DenseNet-121** is the best green/GDM specialist; and **EfficientNet-B0** is the recommended default for real-world deployment. Overlapping confidence intervals keep these conclusions honest: the ranking is consistent and useful, not statistically sealed. Together with ablations and explainability audits, the thesis provides a reproducible recipe — raw targets, RMSE-weighted composite loss, attention head, depth-relative fine-tuning, and trimmed augmentation — for multi-component biomass regression, and a clear map of what remains unsolved (dead matter, clover zeros, dense-canopy XAI metrics, and domain shift).

---

*End of Chapter 7.*
