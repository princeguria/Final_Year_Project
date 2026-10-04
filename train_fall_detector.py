#!/usr/bin/env python3
"""
Fall-detection training script (SisFall -> ADXL335-style accelerometer-only model)
==================================================================================

Goal: train a Random Forest that detects falls from a SINGLE 3-axis accelerometer
that saturates at +-3 g (ADXL335 on an ESP32 ADC, 200 Hz), using the SisFall dataset.

SisFall file layout (one activity per file, 200 Hz, 9 integer columns, "bits"):
    col 0,1,2 : ADXL345  accel X,Y,Z   (13-bit, +-16 g)   <-- used as proxy for our ADXL335
    col 3,4,5 : ITG3200  gyro  X,Y,Z   (16-bit, +-2000 deg/s)  <-- ablation only
    col 6,7,8 : MMA8451Q accel X,Y,Z   (14-bit, +-8 g)    <-- not used

Conversion (from the SisFall readme):  value = (2*Range / 2^Resolution) * raw_bits
    ADXL345 : 2*16   / 2^13 = 0.00390625   g per bit
    ITG3200 : 2*2000 / 2^16 = 0.06103515625 deg/s per bit
    MMA8451Q: 2*8    / 2^14 = 0.0009765625 g per bit   (unused)

Pipeline:
    1. parse files  ->  2. convert to g / deg/s  ->  3. clip accel to +-3 g (per axis)
    4. 2 s windows (400 samples, 50 % overlap)  ->  5. features
    6. Random Forest (class-weighted)  ->  7. subject-wise CV (StratifiedGroupKFold)
    8. threshold tuned for ~95 % fall recall on out-of-fold predictions
    9. report All vs Elderly(SE*) + gyro ablation  ->  10. save .joblib

Usage:
    python train_fall_detector.py --data_dir /path/to/SisFall_dataset --out fall_model.joblib

The feature function `extract_accel_features` is importable (this file has a
__main__ guard), so the FastAPI service can reuse EXACTLY the same code.
"""

import argparse
import io
import re
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # save figures to files (no GUI window needed)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore", category=RuntimeWarning)  # skew/kurtosis on flat signals

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
DATA_DIR = "SisFall_dataset"          # default dataset folder (relative to where you run the script)
MODEL_OUT = "fall_model.joblib"       # default output model file
FEATURE_CACHE = "feats.joblib"        # default feature cache (speeds up re-runs)
PLOT_DIR = "plots"                    # folder where the PNG graphs are saved

FS = 200                      # sample rate (Hz) - native SisFall rate, no downsampling
WIN = 400                     # 2 s window
STRIDE = 200                  # 50 % overlap -> a new window every 1 s

# SisFall conversion constants: (2*Range)/(2^Resolution)
ACC_SCALE_G = (2 * 16) / 2 ** 13         # ADXL345, g per bit
GYRO_SCALE_DPS = (2 * 2000) / 2 ** 16    # ITG3200, deg/s per bit
MMA_SCALE_G = (2 * 8) / 2 ** 14          # MMA8451Q (documented for completeness)

ADXL335_LIMIT_G = 3.0         # our real sensor saturates here

# Feature parameters (stored in the .joblib so inference uses the same values)
FREEFALL_G = 0.6              # SMV below this  -> "free-fall-like" sample
IMPACT_G = 2.0                # SMV above this  -> "impact-like" sample
PRE_N = 100                   # 0.5 s before the peak
POST_SKIP = 20                # skip 0.1 s right after the peak (impact ringing)
POST_N = 200                  # 1 s of post-impact signal
TILT_N = 100                  # 0.5 s used to estimate gravity direction at window start/end
MIN_SEG = 10                  # minimum samples for a pre/post segment to be trusted

# A fall window must contain the impact peak, located between these fractions of the window,
# so that pre-impact AND post-impact samples exist inside the window.
FALL_PEAK_RANGE = (0.2, 0.8)

FILE_RE = re.compile(r"^([DF]\d{2})_(S[AE]\d{2})_R(\d{2})\.txt$", re.IGNORECASE)


# ----------------------------------------------------------------------------
# 1. File parsing + unit conversion
# ----------------------------------------------------------------------------
def load_sisfall_file(path):
    """Read one SisFall file -> (acc_g [N,3], gyro_dps [N,3]) in physical units.

    Real SisFall lines look like '  -16,  -23, ... ,-1000;' (trailing semicolon),
    so we strip ';' before parsing with pandas' fast C parser.
    """
    raw = Path(path).read_bytes().replace(b";", b"")
    arr = pd.read_csv(io.BytesIO(raw), header=None).to_numpy(dtype=np.float64)
    if arr.shape[1] != 9:
        raise ValueError(f"{path}: expected 9 columns, got {arr.shape[1]}")
    arr = arr[~np.isnan(arr).any(axis=1)]
    acc_g = arr[:, 0:3] * ACC_SCALE_G        # ADXL345
    gyro_dps = arr[:, 3:6] * GYRO_SCALE_DPS  # ITG3200
    return acc_g, gyro_dps


def simulate_adxl335(acc_g, limit_g=ADXL335_LIMIT_G):
    """Simulate the ADXL335's hard saturation.

    Saturation happens per sensor axis (each axis output rails at about +-3 g), so we clip
    each axis, then compute SMV afterwards. (SMV can therefore reach 3*sqrt(3) = 5.2 g.)
    """
    return np.clip(acc_g, -limit_g, limit_g)


# ----------------------------------------------------------------------------
# 2. Feature extraction (windows are already in g and already clipped)
# ----------------------------------------------------------------------------
def _angle_deg(u, v):
    """Angle between two 3-vectors in degrees (0 if either is ~zero)."""
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-6 or nv < 1e-6:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))))


def _seg(x, fn, default):
    """Apply fn to a segment if it is long enough, else return a fallback value."""
    return float(fn(x)) if len(x) >= MIN_SEG else float(default)


def extract_accel_features(acc_win, fs=FS):
    """Accelerometer-only features for one window. acc_win: (N, 3) array in g, post-clipping.

    SMV (signal magnitude vector) = sqrt(ax^2 + ay^2 + az^2) is independent of how the
    sensor is mounted, and the tilt features use angles between gravity vectors, so none
    of this assumes which physical axis is "vertical".
    """
    acc_win = np.asarray(acc_win, dtype=np.float64)
    smv = np.linalg.norm(acc_win, axis=1)
    f = {}

    # --- SMV statistics ---
    f["smv_mean"] = smv.mean()
    f["smv_std"] = smv.std()
    f["smv_min"] = smv.min()
    f["smv_max"] = smv.max()
    f["smv_range"] = f["smv_max"] - f["smv_min"]
    f["smv_rms"] = np.sqrt(np.mean(smv ** 2))
    f["smv_skew"] = np.nan_to_num(skew(smv))
    f["smv_kurtosis"] = np.nan_to_num(kurtosis(smv))

    # --- Jerk: magnitude of the derivative of the acceleration vector (g/s) ---
    jerk = np.linalg.norm(np.diff(acc_win, axis=0), axis=1) * fs
    f["jerk_mean"] = jerk.mean()
    f["jerk_std"] = jerk.std()
    f["jerk_max"] = jerk.max()

    # --- Free-fall and impact fractions ---
    f["freefall_frac"] = np.mean(smv < FREEFALL_G)
    f["impact_frac"] = np.mean(smv > IMPACT_G)

    # --- Pre-/post-impact statistics around the SMV peak ---
    i = int(np.argmax(smv))
    pre = smv[max(0, i - PRE_N):i]
    post = smv[i + POST_SKIP:i + POST_SKIP + POST_N]
    pre_mean = _seg(pre, np.mean, f["smv_mean"])
    post_mean = _seg(post, np.mean, f["smv_mean"])
    f["pre_impact_mean"] = pre_mean
    f["pre_impact_std"] = _seg(pre, np.std, f["smv_std"])
    f["post_impact_mean"] = post_mean
    f["post_impact_std"] = _seg(post, np.std, f["smv_std"])
    # After a fall the person usually lies still: SMV ~ 1 g with very little variation.
    f["post_impact_dev_from_1g"] = _seg(post, lambda x: np.mean(np.abs(x - 1.0)),
                                        np.mean(np.abs(smv - 1.0)))
    f["post_pre_ratio"] = post_mean / (pre_mean + 1e-6)
    f["peak_pre_ratio"] = smv[i] / (pre_mean + 1e-6)

    # --- Orientation / tilt change ---
    # Mean acceleration vector over a short segment ~ gravity direction (body orientation).
    g_start = acc_win[:TILT_N].mean(axis=0)
    g_end = acc_win[-TILT_N:].mean(axis=0)
    f["tilt_change_window"] = _angle_deg(g_start, g_end)

    pre_vec = acc_win[max(0, i - PRE_N):i]
    pre_vec = pre_vec.mean(axis=0) if len(pre_vec) >= MIN_SEG else g_start
    post_vec = acc_win[i + 50:i + 50 + POST_N]
    post_vec = post_vec.mean(axis=0) if len(post_vec) >= MIN_SEG else acc_win[-20:].mean(axis=0)
    f["tilt_change_impact"] = _angle_deg(pre_vec, post_vec)

    return {k: float(v) for k, v in f.items()}


def extract_gyro_features(gyro_win, impact_idx, fs=FS):
    """Gyro features, ONLY for the ablation (our real device has no gyroscope)."""
    gm = np.linalg.norm(np.asarray(gyro_win, dtype=np.float64), axis=1)
    i = impact_idx
    f = {
        "gyro_mean": gm.mean(),
        "gyro_std": gm.std(),
        "gyro_max": gm.max(),
        "gyro_range": gm.max() - gm.min(),
        "gyro_rms": np.sqrt(np.mean(gm ** 2)),
        "gyro_peak_near_impact": gm[max(0, i - 50):i + 50].max(),
        "gyro_post_impact_mean": _seg(gm[i + POST_SKIP:i + POST_SKIP + POST_N], np.mean, gm.mean()),
    }
    return {k: float(v) for k, v in f.items()}


# ----------------------------------------------------------------------------
# 3. Dataset building (parse -> window -> label -> features)
# ----------------------------------------------------------------------------
def build_dataset(data_dir):
    """Returns X_acc (DataFrame), X_gyro (DataFrame), meta (DataFrame)."""
    files = sorted(p for p in Path(data_dir).rglob("*.txt") if FILE_RE.match(p.name))
    if not files:
        raise SystemExit(f"No SisFall files (e.g. D01_SA01_R01.txt) found under {data_dir}")
    print(f"Found {len(files)} activity files. Extracting features ...")

    acc_rows, gyro_rows, meta_rows = [], [], []
    t0 = time.time()
    for k, path in enumerate(files):
        act, subj, trial = FILE_RE.match(path.name).groups()
        act, subj = act.upper(), subj.upper()
        is_fall = act.startswith("F")           # F* = fall (label 1), D* = normal (label 0)
        try:
            acc_g, gyro_dps = load_sisfall_file(path)
        except Exception as e:                  # skip corrupt files but tell the user
            print(f"  [skip] {path.name}: {e}")
            continue
        n = len(acc_g)
        if n < WIN:
            continue

        acc_clipped = simulate_adxl335(acc_g)   # <-- ADXL335 saturation applied BEFORE features

        if is_fall:
            # Locate the impact in the *unclipped* signal (offline labelling only).
            peak = int(np.argmax(np.linalg.norm(acc_g, axis=1)))

        for s in range(0, n - WIN + 1, STRIDE):
            e = s + WIN
            if is_fall:
                # Only windows that contain the impact (not too close to an edge) are "fall".
                # Other windows in a fall file (pre-fall walking, lying afterwards) are ambiguous
                # and are DROPPED rather than mislabelled.
                rel = (peak - s) / WIN
                if not (FALL_PEAK_RANGE[0] <= rel < FALL_PEAK_RANGE[1]):
                    continue
            a_win = acc_clipped[s:e]
            acc_rows.append(extract_accel_features(a_win))
            imp = int(np.argmax(np.linalg.norm(a_win, axis=1)))
            gyro_rows.append(extract_gyro_features(gyro_dps[s:e], imp))
            meta_rows.append(dict(subject=subj, activity=act, trial=int(trial),
                                  label=int(is_fall), elderly=subj.startswith("SE")))
        if (k + 1) % 500 == 0:
            print(f"  {k + 1}/{len(files)} files ({time.time() - t0:.0f}s)")

    X_acc, X_gyro, meta = pd.DataFrame(acc_rows), pd.DataFrame(gyro_rows), pd.DataFrame(meta_rows)
    print(f"Windows: {len(meta)}  (falls={meta.label.sum()}, normal={(meta.label == 0).sum()}), "
          f"subjects={meta.subject.nunique()}")
    return X_acc, X_gyro, meta


# ----------------------------------------------------------------------------
# 4. Model, cross-validation, threshold, metrics
# ----------------------------------------------------------------------------
def make_rf(seed=42):
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced_subsample",   # falls are rare -> up-weight them in every tree
        n_jobs=-1,
        random_state=seed,
    )


def subject_wise_oof(X, y, groups, n_splits=5, seed=42):
    """Out-of-fold fall probabilities. No subject is ever in both train and test."""
    oof = np.zeros(len(y))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(cv.split(X, y, groups)):
        assert set(groups[tr]).isdisjoint(groups[te]), "subject leakage!"
        clf = make_rf(seed).fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
        auc = roc_auc_score(y[te], oof[te]) if len(set(y[te])) == 2 else float("nan")
        print(f"    fold {k + 1}: test subjects={len(set(groups[te]))}, "
              f"falls={int(y[te].sum())}, AUC={auc:.4f}")
    return oof


def pick_threshold(y, p, target_recall=0.95):
    """Highest threshold that still catches >= target_recall of falls (fewest false alarms)."""
    fall_p = np.sort(p[y == 1])
    k = int(np.floor((1 - target_recall) * len(fall_p)))
    return float(fall_p[k])


def metrics_at(y, p, thr):
    """Window-level metrics at a given threshold."""
    pred = p >= thr
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    nan = float("nan")
    recall = tp / (tp + fn) if tp + fn else nan
    spec = tn / (tn + fp) if tn + fp else nan
    prec = tp / (tp + fp) if tp + fp else nan
    f1 = 2 * prec * recall / (prec + recall) if (tp + fp and tp + fn and tp > 0) else nan
    hours = (tn + fp) * (STRIDE / FS) / 3600.0       # each window advances 1 s
    both = len(set(y)) == 2
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return dict(n_fall=int(tp + fn), n_normal=int(tn + fp), accuracy=accuracy, recall=recall, specificity=spec,
                precision=prec, f1=f1, false_alarms_per_hour=fp / hours if hours else nan,
                roc_auc=roc_auc_score(y, p) if both else nan,
                pr_auc=average_precision_score(y, p) if both else nan,
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def print_metrics(title, m):
    print(f"  {title}")
    print(f"    windows: {m['n_fall']} fall / {m['n_normal']} normal")
    print(f"    accuracy={m['accuracy']:.3f}  (misleading here: ~95% of windows are normal)")
    print(f"    recall={m['recall']:.3f}  specificity={m['specificity']:.3f}  "
          f"precision={m['precision']:.3f}  F1={m['f1']:.3f}")
    print(f"    ROC-AUC={m['roc_auc']:.3f}  PR-AUC={m['pr_auc']:.3f}  "
          f"false alarms/hour={m['false_alarms_per_hour']:.1f}")
    print(f"    confusion: TP={m['tp']} FN={m['fn']} FP={m['fp']} TN={m['tn']}")


def evaluate_feature_set(name, X, y, groups, elderly, n_splits, target_recall, seed):
    """Run subject-wise CV, tune threshold on out-of-fold predictions, report All + Elderly."""
    print(f"\n=== {name}  ({X.shape[1]} features) ===")
    oof = subject_wise_oof(X, y, groups, n_splits, seed)
    thr = pick_threshold(y, oof, target_recall)
    print(f"  Threshold for ~{target_recall:.0%} fall recall (pooled out-of-fold): {thr:.4f}")

    all_m = metrics_at(y, oof, thr)
    print_metrics("ALL subjects", all_m)
    el_m = metrics_at(y[elderly], oof[elderly], thr)
    print_metrics("ELDERLY (SE*) only", el_m)
    if el_m["n_fall"]:
        print("    NOTE: the only elderly fall data is subject SE06 (a judo expert) -> elderly "
              "fall recall is based on ONE person. Elderly false-alarm rate is the reliable number.")
    return dict(oof=oof, thr=thr, all=all_m, elderly=el_m)



# ----------------------------------------------------------------------------
# 5. Graphs (saved as PNG files)
# ----------------------------------------------------------------------------
def make_plots(y, elderly, results, plot_dir):
    """Save confusion matrices, metric bars, ROC and Precision-Recall curves.

    results: {"Accel-only": res_dict, "Accel+gyro": res_dict} as returned by evaluate_feature_set.
    All numbers come from out-of-fold (subject-wise) predictions at the tuned threshold.
    """
    out = Path(plot_dir)
    out.mkdir(parents=True, exist_ok=True)
    subsets = [("All subjects", "all", np.ones(len(y), dtype=bool)),
               ("Elderly (SE*)", "elderly", elderly)]
    colors = {"Accel-only": "tab:blue", "Accel+gyro": "tab:orange"}

    # ---- 1. Confusion matrices (rows: feature set, cols: subject group) ----
    fig, axes = plt.subplots(len(results), 2, figsize=(9, 4.3 * len(results)), squeeze=False)
    for r, (rname, res) in enumerate(results.items()):
        for c, (sname, key, _) in enumerate(subsets):
            m = res[key]
            cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
            pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)   # row-normalised
            ax = axes[r, c]
            ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
            names = [["TN", "FP"], ["FN", "TP"]]
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{names[i][j]}\n{cm[i, j]}\n({pct[i, j]:.1%})", ha="center",
                            va="center", fontsize=11, color="white" if pct[i, j] > 0.5 else "black")
            ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Fall"])
            ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Fall"])
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.set_title(f"{rname} - {sname}\n(threshold {res['thr']:.3f})", fontsize=10)
    fig.suptitle("Confusion matrices (window level, subject-wise out-of-fold)")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices.png", dpi=150)
    plt.close(fig)

    # ---- 2. Accuracy / Precision / Recall / F1 (+ specificity) bars ----
    metric_names = ["accuracy", "precision", "recall", "f1", "specificity"]
    combos = [(f"{rname} - {sname}", res[key]) for rname, res in results.items()
              for sname, key, _ in subsets]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(metric_names))
    w = 0.8 / len(combos)
    for k, (label, m) in enumerate(combos):
        vals = [float(np.nan_to_num(m[n])) for n in metric_names]
        bars = ax.bar(x - 0.4 + w / 2 + k * w, vals, w, label=label)
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([n.capitalize() if n != "f1" else "F1 score" for n in metric_names])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Classification metrics at the high-recall threshold\n"
                 "(accuracy is inflated by class imbalance - focus on recall / precision / F1)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "metrics_bar.png", dpi=150)
    plt.close(fig)

    # ---- 3. ROC curves with AUC ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (sname, key, mask) in zip(axes, subsets):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess (AUC = 0.50)")
        if len(set(y[mask])) == 2:
            for rname, res in results.items():
                fpr, tpr, _ = roc_curve(y[mask], res["oof"][mask])
                m = res[key]
                ax.plot(fpr, tpr, color=colors[rname], lw=2, label=f"{rname} (AUC = {m['roc_auc']:.3f})")
                ax.scatter([1 - m["specificity"]], [m["recall"]], color=colors[rname], s=60,
                           edgecolor="k", zorder=5)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate (Recall)")
        ax.set_title(f"ROC curve - {sname}\n(dots = chosen threshold)")
        ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "roc_curves.png", dpi=150)
    plt.close(fig)

    # ---- 4. Precision-Recall curves ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (sname, key, mask) in zip(axes, subsets):
        if len(set(y[mask])) == 2:
            ax.axhline(y[mask].mean(), color="k", ls="--", lw=1,
                       label=f"Random guess (precision = {y[mask].mean():.3f})")
            for rname, res in results.items():
                prec, rec, _ = precision_recall_curve(y[mask], res["oof"][mask])
                m = res[key]
                ax.plot(rec, prec, color=colors[rname], lw=2, label=f"{rname} (AP = {m['pr_auc']:.3f})")
                ax.scatter([m["recall"]], [m["precision"]], color=colors[rname], s=60,
                           edgecolor="k", zorder=5)
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05)
        ax.set_title(f"Precision-Recall curve - {sname}\n(dots = chosen threshold)")
        ax.legend(loc="lower left", fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "pr_curves.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved graphs to {out.resolve()}: confusion_matrices.png, metrics_bar.png, "
          f"roc_curves.png, pr_curves.png")

# ----------------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DATA_DIR, help="Folder containing SA01/, SE01/, ... subfolders")
    ap.add_argument("--out", default=MODEL_OUT)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--target_recall", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot_dir", default=PLOT_DIR, help="Folder to save the graphs (PNG)")
    ap.add_argument("--cache", default=FEATURE_CACHE, help="Joblib path to cache extracted features (use '' to disable)")
    args = ap.parse_args()

    # ---- build (or load cached) features ----
    if args.cache and Path(args.cache).exists():
        X_acc, X_gyro, meta = joblib.load(args.cache)
        print(f"Loaded cached features from {args.cache}")
    else:
        X_acc, X_gyro, meta = build_dataset(args.data_dir)
        if args.cache:
            joblib.dump((X_acc, X_gyro, meta), args.cache)

    y = meta["label"].to_numpy()
    groups = meta["subject"].to_numpy()          # subject-wise splitting
    elderly = meta["elderly"].to_numpy()
    acc_names = list(X_acc.columns)
    X_a = X_acc.to_numpy()
    X_ag = pd.concat([X_acc, X_gyro], axis=1).to_numpy()

    # ---- main experiment (accelerometer only) + ablation (accel + gyro) ----
    res_acc = evaluate_feature_set("ACCEL-ONLY (what the ADXL335 device can do)", X_a, y, groups,
                                   elderly, args.n_splits, args.target_recall, args.seed)
    res_ag = evaluate_feature_set("ACCEL + GYRO (ablation, not deployable)", X_ag, y, groups,
                                  elderly, args.n_splits, args.target_recall, args.seed)

    # ---- ablation summary ----
    rows = []
    for name, r in [("accel-only", res_acc), ("accel+gyro", res_ag)]:
        for subset in ("all", "elderly"):
            m = r[subset]
            rows.append(dict(features=name, subjects=subset, threshold=round(r["thr"], 3),
                             accuracy=round(m["accuracy"], 3), recall=round(m["recall"], 3),
                             specificity=round(m["specificity"], 3),
                             roc_auc=round(m["roc_auc"], 3), pr_auc=round(m["pr_auc"], 3),
                             false_alarms_per_hour=round(m["false_alarms_per_hour"], 1)))
    print("\n=== Gyro ablation summary (out-of-fold, subject-wise) ===")
    print(pd.DataFrame(rows).to_string(index=False))

    # ---- graphs ----
    make_plots(y, elderly, {"Accel-only": res_acc, "Accel+gyro": res_ag}, args.plot_dir)

    # ---- per-activity alarm rate (accel-only): shows which ADLs cause false alarms ----
    pred = res_acc["oof"] >= res_acc["thr"]
    per_act = (meta.assign(alarm=pred).groupby("activity")
               .agg(windows=("alarm", "size"), alarm_rate=("alarm", "mean")).round(3))
    print("\n=== Per-activity alarm rate, accel-only (D*: false alarms, F*: recall) ===")
    print(per_act.T.to_string())

    # ---- final model: train on ALL data, accelerometer features only ----
    print("\nTraining final accel-only model on all subjects ...")
    final = make_rf(args.seed).fit(X_a, y)
    imp = pd.Series(final.feature_importances_, index=acc_names).sort_values(ascending=False)
    print("Top features:\n" + imp.head(8).round(3).to_string())

    payload = {
        "model": final,
        "feature_names": acc_names,                 # column order the model expects
        "threshold": res_acc["thr"],                # classify as fall if P(fall) >= threshold
        "target_recall": args.target_recall,
        "sample_rate_hz": FS,
        "window_size": WIN,
        "window_stride": STRIDE,
        "accel_clip_g": ADXL335_LIMIT_G,            # clip each axis to +-this before features
        "sisfall_conversion": {
            "adxl345":  {"range_g": 16, "bits": 13, "g_per_bit": ACC_SCALE_G},
            "itg3200":  {"range_dps": 2000, "bits": 16, "dps_per_bit": GYRO_SCALE_DPS},
            "mma8451q": {"range_g": 8, "bits": 14, "g_per_bit": MMA_SCALE_G},
            "formula": "value = (2*Range / 2**Resolution) * raw",
        },
        "feature_params": dict(freefall_g=FREEFALL_G, impact_g=IMPACT_G, pre_n=PRE_N,
                               post_skip=POST_SKIP, post_n=POST_N, tilt_n=TILT_N, min_seg=MIN_SEG),
        "cv_summary": {
            "accel_only_all": {k: res_acc["all"][k] for k in ("recall", "specificity", "roc_auc", "pr_auc", "false_alarms_per_hour")},
            "accel_only_elderly": {k: res_acc["elderly"][k] for k in ("recall", "specificity", "roc_auc", "pr_auc", "false_alarms_per_hour")},
            "accel_gyro_all": {k: res_ag["all"][k] for k in ("recall", "specificity", "roc_auc", "pr_auc", "false_alarms_per_hour")},
        },
        "notes": "Trained on SisFall ADXL345 (waist) data, clipped per-axis to +-3 g to mimic ADXL335. "
                 "Inference input: (400, 3) array in g, ALREADY clipped, 200 Hz.",
    }
    joblib.dump(payload, args.out)
    print(f"\nSaved model bundle -> {args.out}")


if __name__ == "__main__":
    main()