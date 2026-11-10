# =====================================================================
# TABLE 5-B  —  Ablation study on EfficientNet-B0
# One factor changed from the FINAL config; report weighted-R2 (+ MAE/RMSE).
# Run on Kaggle (GPU, csiro-biomass dataset attached, no internet needed).
# Paste this whole block into ONE cell and Run.
# =====================================================================
import os, gc, math, time
import numpy as np, pandas as pd, tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.model_selection import train_test_split

# ---------------- config ----------------
SEED       = 42
PATH       = "/kaggle/input/csiro-biomass"
IMG_DIR    = PATH
IMG_SIZE   = 224
BATCH      = 16
EPOCHS     = 500                       # early stopping cuts this ~30-40
TARGETS    = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g"]
W5         = np.array([0.1, 0.1, 0.1, 0.2, 0.5])   # Green,Dead,Clover,GDM,Total

def set_seed(s=SEED):
    os.environ["PYTHONHASHSEED"] = str(s)
    tf.keras.utils.set_random_seed(s)
    try: tf.config.experimental.enable_op_determinism()
    except Exception: pass

# ---------------- data ----------------
df = pd.read_csv(os.path.join(PATH, "train.csv"))
dfp = df.pivot_table(index=["image_path"], columns="target_name",
                     values="target").reset_index().fillna(0)
trainval, test = train_test_split(dfp, test_size=0.20, random_state=SEED)
train, val = train_test_split(trainval, test_size=0.20, random_state=SEED)`ikjj`
print(f"train={len(train)} val={len(val)} test={len(test)}")

def targets_of(d, use_log):
    y = d[TARGETS].values.astype("float32")
    return np.log1p(y) if use_log else y

def preprocess_fn(path, label):
    img = tf.io.read_file(tf.strings.join([IMG_DIR, "/", path]))
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return tf.cast(img, tf.float32) / 255.0, label

def make_ds(d, use_log, shuffle):
    ds = tf.data.Dataset.from_tensor_slices(
        (d["image_path"].values.astype(str), targets_of(d, use_log)))
    if shuffle:
        ds = ds.shuffle(len(d), seed=SEED, reshuffle_each_iteration=True)
    return ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH).prefetch(tf.data.AUTOTUNE)

# ---------------- loss ----------------
def composite_loss(alpha=0.35, eps=1e-6):
    def loss(yt, yp):
        e = yt - yp
        mae = tf.reduce_mean(tf.abs(e))
        rmse = tf.sqrt(tf.reduce_mean(tf.square(e)) + eps)   # true RMSE (mean before sqrt)
        return alpha * mae + (1.0 - alpha) * rmse
    return loss

# ---------------- model ----------------
def build(unfreeze_frac, bn_frozen, aug_full):
    inp = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    x = layers.RandomFlip("horizontal_and_vertical")(inp)
    x = layers.RandomRotation(0.1, fill_mode="reflect")(x)
    if aug_full:                       # the "full augmentation" ablation
        x = layers.RandomTranslation(0.1, 0.1, fill_mode="reflect")(x)
        x = layers.RandomZoom(0.2)(x)
        x = layers.RandomBrightness(0.2)(x)
        x = layers.RandomContrast(0.2)(x)
    x = layers.GaussianNoise(0.02)(x)
    x = layers.Rescaling(255.0)(x)
    x = layers.Lambda(preprocess_input)(x)

    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_tensor=x, pooling=None)
    base.trainable = True
    if unfreeze_frac <= 0:             # frozen-backbone ablation (head only)
        for l in base.layers: l.trainable = False
    else:
        n = max(1, int(len(base.layers) * unfreeze_frac))
        for l in base.layers[:-n]: l.trainable = False
    if bn_frozen:
        for l in base.layers:
            if isinstance(l, layers.BatchNormalization): l.trainable = False

    f = base.output; ch = f.shape[-1]
    ap = layers.GlobalAveragePooling2D()(f); mp = layers.GlobalMaxPooling2D()(f)
    s1 = layers.Dense(ch // 16, activation="relu", use_bias=False)
    s2 = layers.Dense(ch, use_bias=False)
    ca = layers.Activation("sigmoid")(layers.Add()([s2(s1(ap)), s2(s1(mp))]))
    ca = layers.Reshape((1, 1, ch))(ca)
    cw = layers.Multiply()([f, ca])
    sa = layers.Conv2D(1, 1, activation="sigmoid")(cw)
    fw = layers.Multiply()([cw, sa])
    v = layers.GlobalAveragePooling2D()(fw)
    v = layers.Dense(128, activation="silu")(v)
    v = layers.Dropout(0.3)(v)
    out = layers.Dense(3, activation="linear")(v)
    return models.Model(inp, out)

def lr_cb(epochs=50):
    lo, hi, warm = 1e-6, 1e-4, 5
    def fn(ep):
        if ep < warm: return float((hi - lo) / warm * ep + lo)
        p = min(1.0, (ep - warm) / max(1, epochs - warm))
        return float(lo + 0.5 * (hi - lo) * (1 + math.cos(3.14159 * p)))
    return tf.keras.callbacks.LearningRateScheduler(fn, verbose=0)

# ---------------- metrics ----------------
def reconstruct5(b):
    g, d, c = b[:, 0], b[:, 1], b[:, 2]
    return np.column_stack([g, d, c, g + c, g + d + c])   # GDM=G+C, Total=G+D+C

def weighted_r2(true3, pred3):
    yt = reconstruct5(true3).flatten(); yp = reconstruct5(pred3).flatten()
    w = np.tile(W5, len(true3)); ybar = np.sum(w * yt) / np.sum(w)
    return 1 - np.sum(w * (yt - yp) ** 2) / np.sum(w * (yt - ybar) ** 2)

# ---------------- one ablation run ----------------
def run(cfg):
    set_seed(SEED); tf.keras.backend.clear_session()
    tr = make_ds(train, cfg["use_log"], True)
    va = make_ds(val,   cfg["use_log"], False)
    m = build(cfg["unfreeze"], cfg["bn_frozen"], cfg["aug_full"])
    m.compile(optimizer=tf.keras.optimizers.AdamW(1e-4, weight_decay=1e-2),
              loss=composite_loss(cfg["alpha"]))
    cbs = [lr_cb(50),
           tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)]
    m.fit(tr, validation_data=va, epochs=EPOCHS, callbacks=cbs, verbose=0)

    te = make_ds(test, cfg["use_log"], False)
    pred = m.predict(te, verbose=0)
    if cfg["use_log"]: pred = np.expm1(pred)
    pred = np.maximum(pred, 0.0)
    true = test[TARGETS].values.astype("float32")          # raw grams
    r2 = weighted_r2(true, pred)
    mae = float(np.mean(np.abs(true - pred)))
    rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
    del m; gc.collect()
    return r2, mae, rmse

# ---------------- ablation grid ----------------
# FINAL must be first (delta is measured against it).
CONFIGS = [
    dict(id="FINAL (raw, a=0.35, unfreeze=0.30, BN frozen, trimmed aug)",
         use_log=False, alpha=0.35, unfreeze=0.30, bn_frozen=True,  aug_full=False),
    dict(id="Target: log1p (vs raw)",
         use_log=True,  alpha=0.35, unfreeze=0.30, bn_frozen=True,  aug_full=False),
    dict(id="Loss: alpha=0.65 (MAE-heavy)",
         use_log=False, alpha=0.65, unfreeze=0.30, bn_frozen=True,  aug_full=False),
    dict(id="Loss: alpha=0.50",
         use_log=False, alpha=0.50, unfreeze=0.30, bn_frozen=True,  aug_full=False),
    dict(id="Unfreeze: frozen backbone (head only)",
         use_log=False, alpha=0.35, unfreeze=0.0,  bn_frozen=True,  aug_full=False),
    dict(id="BatchNorm: trainable (vs frozen)",
         use_log=False, alpha=0.35, unfreeze=0.30, bn_frozen=False, aug_full=False),
    dict(id="Augmentation: full (color jitter + zoom)",
         use_log=False, alpha=0.35, unfreeze=0.30, bn_frozen=True,  aug_full=True),
]

rows = []; final_r2 = None
for c in CONFIGS:
    t0 = time.time(); r2, mae, rmse = run(c); dt = int(time.time() - t0)
    if final_r2 is None: final_r2 = r2
    rows.append(dict(configuration=c["id"], weighted_R2=round(r2, 4),
                     delta_vs_final=round(r2 - final_r2, 4),
                     MAE_g=round(mae, 3), RMSE_g=round(rmse, 3), seconds=dt))
    print(f"{c['id']:52s} R2={r2:.4f}  d={r2-final_r2:+.4f}  MAE={mae:.2f}  RMSE={rmse:.2f}  ({dt}s)")

res = pd.DataFrame(rows)
res.to_csv("ablation_effb0.csv", index=False)
print("\n==================  TABLE 5-B (EfficientNet-B0 ablation)  ==================")
print(res.to_string(index=False))
print("\nSaved -> ablation_effb0.csv")
