# Pipeline — Mermaid Diagrams

## 1. High-level overview (Fig. 4.1)

```mermaid
flowchart TD
    A["Raw RGB quadrat image<br/>(2000 × 1000)"] --> B["Preprocessing<br/>bilinear resize → 224×224, scale to [0,1]<br/>+ backbone-specific normalization"]
    B --> C["Signal-preserving augmentation<br/><i>(training only)</i><br/>flip · small rotation · light noise"]
    C --> D["Backbone feature extractor<br/>one of 5: MobileNet · EfficientNet-B0/B3 · DenseNet-121 · ViT-B/16<br/><i>(ImageNet-pretrained, BatchNorm frozen)</i>"]
    D --> E["Attention pooling → feature vector<br/>CBAM (CNNs) · pooled [CLS] token (ViT)"]
    E --> F["Shared regression head<br/>Dense(128, SiLU) → Dropout(0.3) → Dense(3, linear)"]
    F --> G["Predicted fractions (raw grams)<br/>Green · Dead · Clover<br/><i>(clamped ≥ 0 at inference)</i>"]
    G --> H["Conservation reconstruction<br/>GDM = Green + Clover<br/>Total = Green + Dead + Clover"]
    H --> I["Final output: 5 values<br/>Green · Dead · Clover · GDM · Total"]
```

---

## 2. Detailed pipeline (preprocessing → training → output)

```mermaid
flowchart TD
    A["Competition RGB image<br/>~2000 × 1000 × 3"]

    subgraph PRE["Preprocessing"]
        direction TB
        B1["Bilinear resize → 224 × 224 × 3"]
        B2["Scale pixels /255 → [0, 1]"]
        B3["Backbone-specific normalization (baked in model)<br/>MobileNet → [-1,1] · EfficientNet → [0,255]<br/>DenseNet → caffe-style · ViT → [0,1]→[-1,1]"]
        B1 --> B2 --> B3
    end

    subgraph AUG["Augmentation — TRAINING ONLY"]
        direction TB
        C1["RandomFlip (horizontal + vertical), p=0.5"]
        C2["RandomRotation ±0.1 (±36°), reflect fill"]
        C3["GaussianNoise σ = 0.02"]
        C4["Excluded on purpose:<br/>color jitter · zoom / translation"]
        C1 --> C2 --> C3
    end

    subgraph BK["Backbone (one of 5) — ImageNet-pretrained · BatchNorm frozen"]
        direction TB
        D["MobileNet-V1 (4.2M) · EfficientNet-B0 (5.3M)<br/>EfficientNet-B3 (12.2M) · DenseNet-121 (8.0M)<br/>ViT-B/16 (86.6M) — all @ 224×224"]
    end

    subgraph POOL["Attention pooling → feature vector h"]
        direction TB
        E1["CNNs: CBAM<br/>channel attn (avg+max, shared MLP, r=16, sigmoid)<br/>→ spatial attn (1×1 conv, sigmoid)<br/>→ GlobalAvgPool → h ∈ R^C"]
        E2["ViT: pooled [CLS] token → h ∈ R^768"]
    end

    F["Shared regression head (head parity)<br/>Dense(128, SiLU) → Dropout(0.3) → Dense(3, LINEAR)"]

    subgraph TRAIN["Two-stage depth-relative fine-tuning"]
        direction TB
        T1["Stage 1 — MAIN: unfreeze top 30% of backbone + head<br/>AdamW (wd 1e-2) · warmup 1e-6→1e-4 (ViT 5e-5) · cosine decay<br/>EarlyStopping patience 7"]
        T2["Stage 2 — FINE-TUNE: unfreeze top 40% of backbone<br/>AdamW LR 1e-5 (wd 1e-4) · ReduceLROnPlateau (×0.5, pat 3)<br/>EarlyStopping patience 5 · best stage is model-dependent"]
        L["Composite loss = 0.35·MAE + 0.65·RMSE_ε (ε=1e-6)<br/>true RMSE (mean inside sqrt) · raw-gram targets"]
        T1 --> T2
    end

    G["Predicted fractions (raw grams)<br/>ŷ = [Green, Dead, Clover]"]
    NN["Non-negativity: clamp ŷ = max(ŷ, 0) at INFERENCE<br/>(linear output — NO ReLU/Sigmoid activation)"]
    H["Conservation reconstruction<br/>GDM = Green + Clover<br/>Total = Green + Dead + Clover"]
    I["Final output — 5 values<br/>Green · Dead · Clover · GDM · Total"]

    A --> PRE
    PRE -->|training path| AUG
    AUG --> BK
    PRE -->|val / test path| BK
    BK --> POOL
    POOL --> F
    F --> G --> NN --> H --> I

    L -.loss.-> F
    TRAIN -.controls unfrozen layers.-> BK
```

> **Notes**
> - Unfreezing is **depth-relative** (top 30% then 40% *of each backbone's layers*), so the actual layer
>   count differs per model — that is intentional (fair across different depths).
> - **Augmentation runs only during training**; validation/test images skip it.
> - **Non-negativity** is enforced only at inference by clamping (`max(ŷ, 0)`); the output layer itself is
>   **linear** — there is no ReLU/Sigmoid on the outputs.
> - GDM/Total are **reconstructed** from the 3 predicted fractions, never regressed directly.
