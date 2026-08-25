"""QC + preprocessing sanity check for out-ear EEG (ear_eeg_out.ads1299) emotion epochs.

1. Flag epochs that hit the ADC's hardware rail (clipping/saturation).
2. Drop those epochs, DC-remove + bandpass the rest.
3. Bandpower features -> logistic regression (grouped by participant) vs.
   chance, to check whether any linear signal survives at all.
"""
import numpy as np
from pathlib import Path
from scipy import signal as sps
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier

EXPORT_DIR = Path(__file__).parents[1] / "data" / "npy_export"  # dataset.export_npy --emotion-only output
FS = 200.0
FULL_SCALE = 2**23  # ADS1299 signed 24-bit ADC rail (+-8388608 raw counts)
CLIP_MARGIN = 0.001  # within 0.1% of the rail counts as clipped

PARTICIPANTS = sorted(p.name for p in EXPORT_DIR.iterdir() if p.is_dir() and p.name != "pooled")
sos = sps.butter(4, [1, 40], btype="bandpass", fs=FS, output="sos")

all_X, all_y, all_pid, clip_rates = [], [], [], {}

for pid in PARTICIPANTS:
    X = np.load(EXPORT_DIR / pid / "emotion_trial_X.npy")  # (N, 8ch, 200) raw ADC counts
    y = np.load(EXPORT_DIR / pid / "emotion_trial_y.npy")
    if X.shape[0] == 0:
        continue

    clipped = (np.abs(X) > FULL_SCALE * (1 - CLIP_MARGIN)).any(axis=(1, 2))  # (N,)
    clip_rates[pid] = float(clipped.mean())

    X_clean, y_clean = X[~clipped], y[~clipped]
    if X_clean.shape[0] == 0:
        continue
    X_filt = sps.sosfiltfilt(sos, X_clean - X_clean.mean(axis=2, keepdims=True), axis=2)

    all_X.append(X_filt)
    all_y.append(y_clean)
    all_pid.extend([pid] * len(y_clean))

for pid, rate in sorted(clip_rates.items(), key=lambda kv: -kv[1]):
    print(f"{pid}: {rate:.1%} of epochs clipped")

X_all, y_all, pid_all = np.concatenate(all_X), np.concatenate(all_y), np.array(all_pid)
print(f"\n{X_all.shape[0]} clean epochs remain (of {sum(len(np.load(EXPORT_DIR / p / 'emotion_trial_y.npy')) for p in PARTICIPANTS)})")

# per-epoch bandpower features (delta/theta/alpha/beta/gamma x 8 channels)
freqs, psd = sps.welch(X_all, fs=FS, nperseg=X_all.shape[-1], axis=-1)
bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
feat = np.concatenate([psd[..., (freqs >= lo) & (freqs <= hi)].mean(axis=-1) for lo, hi in bands], axis=1)
feat = StandardScaler().fit_transform(np.log1p(feat))

gkf = GroupKFold(n_splits=5)
acc = cross_val_score(LogisticRegression(max_iter=2000), feat, y_all, groups=pid_all, cv=gkf, scoring="accuracy")
base = cross_val_score(DummyClassifier(strategy="most_frequent"), feat, y_all, groups=pid_all, cv=gkf, scoring="accuracy")
print(f"\nLogReg accuracy:  {acc.mean():.3f} +/- {acc.std():.3f}")
print(f"Majority baseline: {base.mean():.3f} +/- {base.std():.3f}  (chance = 0.333)")
