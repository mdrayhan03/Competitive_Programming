# =====================================================================
# #6 — QUANTITATIVE XAI METRIC:  "% of attention that lands on vegetation"
# Runs Grad-CAM on ALL test images for the 4 CNNs and measures what
# fraction of the attention mass falls on plant pixels vs soil.
# (ViT excluded: Grad-CAM needs conv maps; its attention map is uninformative.)
#
# Run on Kaggle: GPU on, internet off. Attach:
#   - competition dataset  (csiro-biomass)
#   - your models dataset  (the models/ folder)
# Paste this whole block into ONE cell and Run.  -> xai_attention_on_vegetation.csv
# =====================================================================
import os, gc, glob, importlib
import numpy as np, pandas as pd, tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import cv2

SEED     = 42
PATH     = "/kaggle/input/csiro-biomass"
IMG_DIR  = PATH
IMG_SIZE = 224
TARGETS  = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g"]
TNAMES   = ["Green", "Dead", "Clover"]

# EDIT to your uploaded models dataset (auto-resolver also searches /kaggle/input)
MODELS_DIR = "/kaggle/input/datasets/phylosopherhossain/csiro-biomass-models"

# 4 CNNs (best-stage files) + their preprocess module. ViT is intentionally excluded.
CNN = {
    "MobileNetV0":    ("mobilenet.keras",     "mobilenet"),
    "EfficientNetB0": ("efficientnetb0.keras","efficientnet"),
    "EfficientNetB3": ("efficientnetb3.keras","efficientnet"),
    "DenseNet121":    ("densenet121.keras",   "densenet"),
}

# ---- rebuild the exact held-out test split (seed 42) ----
df  = pd.read_csv(os.path.join(PATH, "train.csv"))
dfp = df.pivot_table(index=["image_path"], columns="target_name",
                     values="target").reset_index().fillna(0)
_, test = train_test_split(dfp, test_size=0.20, random_state=SEED)
PATHS = test["image_path"].tolist()
print(f"Test images: {len(PATHS)}")

def resolve(fn):
    p = os.path.join(MODELS_DIR, fn)
    if os.path.exists(p): return p
    hits = glob.glob(f"/kaggle/input/**/{fn}", recursive=True)
    if hits: return hits[0]
    raise FileNotFoundError(fn)

def load_model(fn, mod):
    m = importlib.import_module(f"tensorflow.keras.applications.{mod}")
    co = {"preprocess_input": m.preprocess_input}
    try:    return tf.keras.models.load_model(resolve(fn), compile=False, safe_mode=False, custom_objects=co)
    except TypeError: return tf.keras.models.load_model(resolve(fn), compile=False, custom_objects=co)

def load_img(path):
    img = tf.io.read_file(os.path.join(IMG_DIR, path))
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return (tf.cast(img, tf.float32) / 255.0).numpy()

def vegetation_mask(img01):
    """Excess-Green (ExG = 2G - R - B) > 0  ->  plant pixel. Simple, training-free."""
    r, g, b = img01[..., 0], img01[..., 1], img01[..., 2]
    return (2 * g - r - b > 0).astype("float32")

def find_last_conv(model):
    for l in reversed(model.layers):
        try: s = l.output.shape
        except Exception: continue
        if len(s) == 4 and s[-1] and s[-1] > 1:
            return l.name
    return None

def grad_cam(model, img01, tidx, conv):
    gm = tf.keras.models.Model(model.inputs, [model.get_layer(conv).output, model.output])
    with tf.GradientTape() as t:
        conv_out, pred = gm(img01[None, ...]); loss = pred[:, tidx]
    grads  = t.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    hm = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
    hm = tf.maximum(hm, 0.0); hm = hm / (tf.reduce_max(hm) + 1e-8)
    return cv2.resize(hm.numpy().astype("float32"), (IMG_SIZE, IMG_SIZE))

# ---- baseline: average % of each image that is vegetation (random-attention reference) ----
veg_cov = np.mean([vegetation_mask(load_img(p)).mean() for p in PATHS]) * 100
print(f"Baseline vegetation coverage (random attention ≈ this): {veg_cov:.1f}%")

# ---- per-model metric ----
rows = []
for name, (fn, mod) in CNN.items():
    print("=" * 50, "\nProcessing", name)
    model = load_model(fn, mod); conv = find_last_conv(model)
    per_t = {t: [] for t in TNAMES}
    for p in PATHS:
        img = load_img(p); vm = vegetation_mask(img)
        for ti, tn in enumerate(TNAMES):
            hm = grad_cam(model, img, ti, conv)
            per_t[tn].append(float((hm * vm).sum() / (hm.sum() + 1e-8)))
    row = {"model": name}
    allv = []
    for tn in TNAMES:
        row[f"{tn}_%onVeg"] = round(100 * np.mean(per_t[tn]), 1); allv += per_t[tn]
    row["Overall_%onVeg"] = round(100 * np.mean(allv), 1)
    row["vs_baseline(+pts)"] = round(row["Overall_%onVeg"] - veg_cov, 1)
    rows.append(row); print(row)
    del model; gc.collect(); tf.keras.backend.clear_session()

res = pd.DataFrame(rows)
res.attrs["baseline_veg_%"] = round(veg_cov, 1)
res.to_csv("xai_attention_on_vegetation.csv", index=False)
print("\n=== XAI METRIC: % of Grad-CAM attention on vegetation (higher = looks at plants) ===")
print(f"(baseline vegetation coverage = {veg_cov:.1f}%; a model above this over-attends to plants)")
print(res.to_string(index=False))
print("\nSaved -> xai_attention_on_vegetation.csv")
