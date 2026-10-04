"""
Train a Random Forest on the SisFall Enhanced windows (Musci et al. labelling).

Data layout (found from file sizes):
  x_*_3 : float32, shape (N, 256, 6)   256 samples ~ 1.28 s at 200 Hz, 6 channels
  y_*_3 : uint8,   shape (N, 3)        one-hot: 0 = normal, 1 = alert (pre-impact), 2 = fall

Run from the project folder:   python train_model.py
"""
import os
import numpy as np
import joblib
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "fall_model.joblib")
W, C, FS = 256, 6, 200
CLASSES = ["normal", "alert", "fall"]


def load_xy(split):
    x = np.memmap(os.path.join(DATA, f"x_{split}_3"), dtype="<f4", mode="r")
    y = np.fromfile(os.path.join(DATA, f"y_{split}_3"), dtype=np.uint8)
    n = len(y) // 3
    assert len(y) % 3 == 0 and len(x) == n * W * C, f"{split}: unexpected file size"
    return x.reshape(n, W, C), y.reshape(n, 3).argmax(1)


def window_features(a):
    """a: (n, W, 3) accelerometer in g  ->  (n, F) features (axis-independent)."""
    n, w, _ = a.shape
    smv = np.linalg.norm(a, axis=2)
    f = {}
    f["smv_mean"] = smv.mean(1)
    f["smv_std"] = smv.std(1)
    f["smv_max"] = smv.max(1)
    f["smv_min"] = smv.min(1)
    f["smv_range"] = f["smv_max"] - f["smv_min"]
    f["smv_rms"] = np.sqrt((smv ** 2).mean(1))
    f["smv_skew"] = np.nan_to_num(skew(smv, axis=1))
    f["smv_kurt"] = np.nan_to_num(kurtosis(smv, axis=1))
    jerk = np.abs(np.diff(smv, axis=1)) * FS
    f["jerk_max"] = jerk.max(1)
    f["jerk_mean"] = jerk.mean(1)

    ipk = smv.argmax(1)
    f["peak_pos"] = ipk / w
    t = np.arange(w)[None, :]
    pre, post = t < ipk[:, None], t > ipk[:, None]
    n_post = post.sum(1)
    pre_min = np.where(pre, smv, np.inf).min(1)
    f["pre_min"] = np.where(np.isfinite(pre_min), pre_min, 1.0)
    pm = (smv * post).sum(1) / np.maximum(n_post, 1)
    pv = (((smv - pm[:, None]) ** 2) * post).sum(1) / np.maximum(n_post, 1)
    f["post_mean"] = np.where(n_post > 5, pm, 1.0)
    f["post_std"] = np.where(n_post > 5, np.sqrt(pv), 0.0)
    f["freefall_frac"] = (smv < 0.6).mean(1)
    f["impact_frac"] = (smv > 1.8).mean(1)

    k, h = 40, w // 2
    a0, a1 = a[:, :k].mean(1), a[:, -k:].mean(1)
    den = np.linalg.norm(a0, axis=1) * np.linalg.norm(a1, axis=1)
    cosang = np.clip((a0 * a1).sum(1) / np.maximum(den, 1e-9), -1, 1)
    f["tilt_change_deg"] = np.degrees(np.arccos(cosang))
    f["std_first_half"] = smv[:, :h].std(1)
    f["std_second_half"] = smv[:, h:].std(1)
    f["mean_shift"] = smv[:, h:].mean(1) - smv[:, :h].mean(1)
    names = list(f)
    return np.stack([f[k] for k in names], axis=1), names


FEATURE_NAMES = window_features(np.tile([0.0, 0.0, 1.0], (2, W, 1)))[1]


def estimate_units_per_g(x, y, n=20000):
    """Dataset values are scaled. Find how many dataset units equal 1 g by using
    quiet 'normal' windows, where the accelerometer only measures gravity (1 g)."""
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(x), size=min(n, len(x)), replace=False))
    a = np.asarray(x[idx][:, :, :3])
    idx_ok = y[idx] == 0
    smv = np.linalg.norm(a, axis=2)
    sd = smv.std(1)
    still = idx_ok & (sd <= np.quantile(sd[idx_ok], 0.10))
    g_units = np.linalg.norm(a.mean(1), axis=1)[still]
    med = float(np.median(g_units))
    spread = float((np.quantile(g_units, .75) - np.quantile(g_units, .25)) / med)
    return med, spread


def featurize(x, scale, chunk=5000):
    out = []
    for i in range(0, len(x), chunk):
        a = np.asarray(x[i:i + chunk, :, :3]) / scale
        out.append(window_features(a)[0])
    return np.vstack(out)


def report(name, model, X, y):
    p = model.predict(X)
    print(f"\n===== {name} =====")
    print(confusion_matrix(y, p))
    print(classification_report(y, p, target_names=CLASSES, digits=3, zero_division=0))


def main():
    xtr, ytr = load_xy("train")
    xva, yva = load_xy("val")
    xte, yte = load_xy("test")
    print("windows  train/val/test:", len(ytr), len(yva), len(yte))
    print("train class counts:", np.bincount(ytr, minlength=3).tolist())

    scale, spread = estimate_units_per_g(xtr, ytr)
    print(f"\n1 g  =  {scale:.4f} dataset units   (relative spread {spread:.3f}; "
          f"small = channels 0-2 really are the accelerometer)")

    print("computing features ...")
    Xtr, Xva, Xte = (featurize(v, scale) for v in (xtr, xva, xte))

    print("training ...")
    clf = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, n_jobs=-1,
        class_weight="balanced_subsample", random_state=42)
    clf.fit(Xtr, ytr)

    report("validation", clf, Xva, yva)
    report("test", clf, Xte, yte)
    imp = sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda t: -t[1])[:8]
    print("top features:", ", ".join(f"{n}={v:.3f}" for n, v in imp))

    joblib.dump({"model": clf, "feature_names": FEATURE_NAMES, "classes": CLASSES,
                 "fs": FS, "win": W, "acc_units_per_g": scale,
                 "units": "acc in g (features), 200 Hz, 256-sample window"}, OUT, compress=3)
    print("\nsaved", OUT)


if __name__ == "__main__":
    main()
