"""
RHD Screening - PCG + ECG attention fusion (single-file Streamlit app)

Place this file in the RHD_Project root, next to:
    processed/PCG/best_pcg_cnn_clean.keras
    processed/ECG/best_ecg_1dcnn_clean.keras
    processed/PCG_ECG_best_attention_fusion_clean.keras

Run:  streamlit run app.py        (not "python app.py")

Research prototype only - not clinically validated.

IMPORTANT: the saved fusion model takes ONLY PCG and ECG embeddings.
Clinical inputs entered in the sidebar are recorded in the report but are
NOT used by the model. The app never claims clinical-data fusion.
"""

import os
import io
import json
import re
from datetime import datetime
from fractions import Fraction

import numpy as np
import pandas as pd
import streamlit as st
import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.signal import butter, filtfilt, resample_poly, find_peaks
from scipy.ndimage import gaussian_filter1d, zoom

# ==================================================
# CONFIGURATION
# ==================================================
ROOT = os.path.dirname(os.path.abspath(__file__))

PCG_PATH = os.path.join(ROOT, "processed", "PCG", "best_pcg_cnn_clean.keras")
ECG_PATH = os.path.join(ROOT, "processed", "ECG", "best_ecg_1dcnn_clean.keras")
FUSION_PATH = os.path.join(
    ROOT, "processed", "PCG_ECG_best_attention_fusion_clean.keras"
)

DEFAULT_THRESHOLD = 0.4031  # validation-selected (Youden) threshold

# Training-time signal settings (must match the notebook pipeline)
PCG_SR = 2000
PCG_SEG_SEC = 1.2
PCG_FRAMES = 19
PCG_MELS = 40
ECG_FS = 2000
ECG_LEN = 4000

# Input validation limits
MIN_ECG_SECONDS = ECG_LEN / ECG_FS      # 2 s
MIN_PCG_SECONDS = PCG_SEG_SEC           # 1.2 s
MAX_MISSING_FRACTION = 0.05             # reject ECG above 5% NaN/inf
MIN_PCG_NATIVE_SR = 1000                # need >= 1 kHz to keep <= 500 Hz content
METADATA_WORDS = {
    "recording_id", "label", "labels", "segment", "segment_id", "filename",
    "file", "path", "patient", "patient_id", "subject", "split", "fold",
    "class", "target", "record", "recording", "id",
}

# Copied from notebook cell 74. Re-verify against the live output of
# cells 69, 73 and 27/41 before putting these numbers in a report.
REFERENCE_METRICS = pd.DataFrame(
    {
        "Metric": ["ROC-AUC", "Accuracy", "Precision",
                   "Sensitivity", "Specificity", "F1-score"],
        "PCG 2D-CNN": [0.6744, 0.7049, 0.7778, 0.8140, 0.4444, 0.7955],
        "ECG 1D-CNN": [0.8023, 0.6885, 0.8529, 0.6744, 0.7222, 0.7532],
        "PCG-ECG Fusion": [0.8295, 0.7541, 0.8333, 0.8140, 0.6111, 0.8235],
    }
)

st.set_page_config(page_title="RHD Screening (PCG + ECG)", page_icon="🫀",
                   layout="wide")

# ==================================================
# STYLING
# ==================================================
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; max-width: 1250px;}
    .hero {padding: 24px 28px; border-radius: 18px; color: #fff;
           background: linear-gradient(120deg,#0b2545 0%,#13607a 60%,#1b8a8f 100%);
           box-shadow: 0 6px 22px rgba(11,37,69,.25);}
    .hero h1 {color:#fff; margin:0; font-size: 1.9rem;}
    .hero p {margin:6px 0 0; opacity:.92;}
    .pill {display:inline-block; padding:3px 12px; margin:10px 6px 0 0;
           border-radius:999px; background:rgba(255,255,255,.16);
           font-size:.8rem;}
    .result-card {padding: 22px 26px; border-radius: 16px; border: 1px solid;
                  margin: 6px 0 14px 0;}
    .result-card h2 {margin:0 0 4px 0; font-size:1.7rem;}
    .result-card p {margin:0; font-size:.98rem;}
    .abn {background:#fef3f2; border-color:#fda29b; color:#7a271a;}
    .nor {background:#ecfdf3; border-color:#75e0a7; color:#05603a;}
    .bor {background:#fffaeb; border-color:#fec84b; color:#7a4b06;}
    .step {padding:10px 14px; border-radius:10px; background:#f2f6f9;
           margin-bottom:8px; color:#12304a;}
    .mod {padding:10px 14px; border-radius:10px; margin-bottom:6px;
          border:1px solid #d0d5dd;}
    .mod-on {background:#ecfdf3; border-color:#75e0a7; color:#05603a;}
    .mod-off {background:#f2f4f7; color:#475467;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🫀 RHD Screening Prototype: PCG + ECG</h1>
        <p>Heart-sound and ECG branch CNNs combined by an attention fusion
           model, with branch-level Grad-CAM explanations.</p>
        <span class="pill">PCG + ECG fusion</span>
        <span class="pill">Grad-CAM</span>
        <span class="pill">Strict input validation</span>
        <span class="pill">Research use only</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")
st.warning(
    "Research prototype only. This system is not clinically validated and "
    "must not be used to diagnose or treat a patient."
)


# ==================================================
# MODEL LOADING
# ==================================================
def _rebuild(model, want_conv=False):
    """Rebuild a saved model as a clean Functional graph.

    want_conv=False -> encoder (everything before the final Dense layer)
    want_conv=True  -> model returning [last conv feature maps, output]
    """
    inp = tf.keras.Input(shape=model.input_shape[1:])
    x = inp
    conv_out = None
    layers = [l for l in model.layers
              if not isinstance(l, tf.keras.layers.InputLayer)]
    body = layers if want_conv else layers[:-1]

    for lyr in body:
        x = lyr(x)
        if isinstance(lyr, (tf.keras.layers.Conv1D, tf.keras.layers.Conv2D)):
            conv_out = x

    if want_conv:
        if conv_out is None:
            raise ValueError("No Conv1D/Conv2D layer found for Grad-CAM.")
        return tf.keras.Model(inp, [conv_out, x])
    return tf.keras.Model(inp, x)


@st.cache_resource(show_spinner="Loading trained models...")
def load_models():
    for path in (PCG_PATH, ECG_PATH, FUSION_PATH):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Model file not found: {os.path.relpath(path, ROOT)}"
            )
    pcg = tf.keras.models.load_model(PCG_PATH, compile=False)
    ecg = tf.keras.models.load_model(ECG_PATH, compile=False)
    fusion = tf.keras.models.load_model(FUSION_PATH, compile=False)
    return {
        "pcg": pcg, "ecg": ecg, "fusion": fusion,
        "pcg_encoder": _rebuild(pcg), "ecg_encoder": _rebuild(ecg),
        "pcg_cam": _rebuild(pcg, want_conv=True),
        "ecg_cam": _rebuild(ecg, want_conv=True),
    }


# ==================================================
# PCG VALIDATION + PREPROCESSING
# ==================================================
def preprocess_pcg(audio_bytes):
    """Validate the WAV, resample to 2 kHz, return segments. Raises on bad input."""
    warnings = []
    if not audio_bytes:
        raise ValueError("The PCG audio file is empty.")

    try:
        y, sr0 = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
    except Exception as exc:
        raise ValueError(f"Could not read the PCG file as audio: {exc}") from exc

    if len(y) == 0:
        raise ValueError("The PCG recording has no samples.")
    bad = ~np.isfinite(y)
    if bad.any():
        raise ValueError("The PCG recording contains NaN/infinite values.")
    if sr0 < MIN_PCG_NATIVE_SR:
        raise ValueError(
            f"PCG sampling rate is {sr0} Hz; at least {MIN_PCG_NATIVE_SR} Hz "
            "is needed to keep heart-sound frequencies."
        )

    duration = len(y) / sr0
    if duration < MIN_PCG_SECONDS:
        raise ValueError(
            f"PCG is {duration:.2f} s long; at least {MIN_PCG_SECONDS} s is needed."
        )

    peak = float(np.max(np.abs(y)))
    if peak < 1e-8:
        raise ValueError("The PCG recording is silent.")
    if np.mean(np.abs(y) >= 0.99 * peak) > 0.02 and peak >= 0.99:
        warnings.append("PCG shows clipping in more than 2% of samples.")
    if np.std(y / peak) < 0.005:
        warnings.append("PCG has very low variation; it may be mostly noise or silence.")

    if sr0 != PCG_SR:
        y = librosa.resample(y, orig_sr=sr0, target_sr=PCG_SR)
        warnings.append(f"PCG resampled from {sr0} Hz to {PCG_SR} Hz.")
    y = y / (np.max(np.abs(y)) + 1e-12)

    seg_len = int(PCG_SEG_SEC * PCG_SR)
    specs = []
    for i in range(0, len(y) - seg_len + 1, seg_len):
        mel = librosa.feature.melspectrogram(
            y=y[i:i + seg_len], sr=PCG_SR, n_mels=PCG_MELS,
            n_fft=256, hop_length=128,
        )
        logmel = librosa.power_to_db(mel, ref=np.max)
        if logmel.shape[1] < PCG_FRAMES:
            logmel = np.pad(logmel, ((0, 0), (0, PCG_FRAMES - logmel.shape[1])))
        else:
            logmel = logmel[:, :PCG_FRAMES]
        specs.append(logmel.astype(np.float32))

    X = np.asarray(specs, dtype=np.float32)[..., np.newaxis]
    if X.shape[1:] != (PCG_MELS, PCG_FRAMES, 1):
        raise ValueError(f"Unexpected PCG input shape: {X.shape}")
    return X, y, warnings


# ==================================================
# ECG FILE SCANNING (rejects metadata / index / label files)
# ==================================================
def scan_ecg_table(raw_text, filename, has_header):
    """Return (numeric_df, valid_columns, excluded{col: reason}).

    Raises ValueError when the file is clearly not an ECG signal file,
    for example ecg_metadata.csv.
    """
    if "metadata" in filename.lower():
        raise ValueError(
            f"'{filename}' looks like a metadata file, not an ECG signal. "
            "Upload the raw ECG waveform (one column of samples) or a .npy "
            "file containing a single 1-D signal."
        )
    if not raw_text.strip():
        raise ValueError("The ECG file is empty.")

    first_line = raw_text.strip().splitlines()[0]
    tokens = {t.strip().strip('"').lower() for t in re.split(r"[,;\t ]+", first_line)}
    found = tokens & METADATA_WORDS
    if found:
        raise ValueError(
            "The first row contains metadata-style column names "
            f"({', '.join(sorted(found))}). This looks like a metadata/label "
            "table, not an ECG waveform. Upload the signal samples only."
        )

    # Detect the separator ourselves: csv sniffing fails on single-column
    # files of signed decimals.
    probe = raw_text.strip().splitlines()[0]
    sep = "," if "," in probe else ";" if ";" in probe else "\t" if "\t" in probe else r"\s+"

    try:
        raw_df = pd.read_csv(io.StringIO(raw_text), sep=sep, engine="python",
                             header=0 if has_header else None, dtype=str)
    except Exception as exc:
        raise ValueError(f"Could not parse the ECG file: {exc}") from exc

    num_df = raw_df.apply(pd.to_numeric, errors="coerce")
    valid, excluded = [], {}

    for col in num_df.columns:
        s = num_df[col]
        if s.notna().mean() < 0.95:
            excluded[col] = "not numeric (text values)"
            continue
        v = s.dropna().to_numpy(dtype=np.float64)
        if len(v) < 10:
            excluded[col] = "too few values"
            continue
        d = np.diff(v)
        if np.all(d > 0) and np.allclose(d, d[0], rtol=1e-6, atol=1e-9):
            excluded[col] = "looks like an index / time column"
            continue
        if len(np.unique(v)) <= 5:
            excluded[col] = "looks like a label/category column"
            continue
        valid.append(col)

    if not valid:
        reasons = "; ".join(f"column {c}: {r}" for c, r in excluded.items())
        raise ValueError(
            "No usable ECG signal column found. " + (reasons or "")
        )
    return num_df, valid, excluded


def read_ecg(file_bytes, filename, column=None, has_header=False):
    """Return raw ECG samples (may contain NaN; validated later)."""
    if filename.lower().endswith(".npy"):
        try:
            x = np.load(io.BytesIO(file_bytes), allow_pickle=False)
        except Exception as exc:
            raise ValueError(f"Could not read the .npy file: {exc}") from exc
        if not np.issubdtype(x.dtype, np.number):
            raise ValueError("The .npy file does not contain numeric data.")
        x = np.asarray(x, dtype=np.float64).squeeze()
        if x.ndim != 1:
            raise ValueError(
                f"The .npy array has shape {x.shape}. It looks like a "
                "pre-segmented dataset (for example ecg_segments.npy), not a "
                "single ECG recording. Upload one 1-D waveform."
            )
        return x

    raw = file_bytes.decode("utf-8-sig", errors="replace")
    num_df, valid, _ = scan_ecg_table(raw, filename, has_header)
    if column is None:
        column = valid[0]
    if column not in valid:
        raise ValueError("The selected column is not a valid ECG signal column.")
    return num_df[column].to_numpy(dtype=np.float64)


# ==================================================
# ECG VALIDATION + PREPROCESSING
# ==================================================
def preprocess_ecg(signal, fs):
    warnings = []
    x = np.asarray(signal, dtype=np.float64).reshape(-1)

    if fs < 100 or fs > 10000:
        raise ValueError("Sampling rate must be between 100 and 10000 Hz.")
    if len(x) == 0:
        raise ValueError("The ECG has no samples.")

    # Missing values
    bad = ~np.isfinite(x)
    frac = bad.mean()
    if frac > MAX_MISSING_FRACTION:
        raise ValueError(
            f"{frac * 100:.1f}% of ECG samples are missing/invalid "
            f"(limit {MAX_MISSING_FRACTION * 100:.0f}%)."
        )
    if bad.any():
        idx = np.arange(len(x))
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
        warnings.append(
            f"{int(bad.sum())} missing ECG samples ({frac * 100:.2f}%) were "
            "filled by linear interpolation."
        )

    # Length at the declared sampling rate
    duration = len(x) / fs
    if duration < MIN_ECG_SECONDS:
        raise ValueError(
            f"ECG is {duration:.2f} s at {int(fs)} Hz ({len(x)} samples); "
            f"at least {MIN_ECG_SECONDS:.0f} s is needed. If the length looks "
            "wrong, check the sampling rate."
        )

    if np.ptp(x) < 1e-8:
        raise ValueError("The ECG signal is flat.")
    if (np.mean(x >= x.max()) > 0.02) or (np.mean(x <= x.min()) > 0.02):
        warnings.append("ECG appears clipped (flat at its min or max for >2% of samples).")

    # Band-pass at the native rate
    if fs > 80:
        nyq = fs / 2
        low, high = max(0.5 / nyq, 1e-4), min(40 / nyq, 0.99)
        if low < high:
            b, a = butter(3, [low, high], btype="bandpass")
            try:
                x = filtfilt(b, a, x)
            except ValueError:
                warnings.append("Band-pass filter skipped (signal too short).")

    # Robust normalisation, then plausibility check on heart rate
    med = np.median(x)
    scale = 1.4826 * np.median(np.abs(x - med))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = np.std(x)
    if not np.isfinite(scale) or scale < 1e-8:
        raise ValueError("ECG signal is flat or invalid.")
    xn = (x - med) / scale

    peaks, _ = find_peaks(np.abs(xn), height=3.0, distance=max(int(0.25 * fs), 1))
    bpm = len(peaks) / duration * 60.0
    if not 30 <= bpm <= 220:
        warnings.append(
            f"Estimated rate is {bpm:.0f} beats/min, outside 30-220. The "
            "sampling rate may be wrong or this may not be an ECG."
        )

    # Resample to the training rate
    if int(fs) != ECG_FS:
        ratio = Fraction(ECG_FS, int(fs)).limit_denominator(1000)
        xn = resample_poly(xn, ratio.numerator, ratio.denominator)
        warnings.append(f"ECG resampled from {int(fs)} Hz to {ECG_FS} Hz.")

    if len(xn) < ECG_LEN:
        raise ValueError(
            f"Only {len(xn)} samples remain after resampling; {ECG_LEN} needed."
        )

    segs = [xn[i:i + ECG_LEN] for i in range(0, len(xn) - ECG_LEN + 1, ECG_LEN)]
    X = np.asarray(segs, dtype=np.float32)[..., np.newaxis]
    if X.shape[1:] != (ECG_LEN, 1):
        raise ValueError(f"Unexpected ECG input shape: {X.shape}")
    return X, xn.astype(np.float32), warnings, float(bpm)


# ==================================================
# GRAD-CAM + CONSISTENCY CHECK
# ==================================================
def gradcam(cam_model, x):
    with tf.GradientTape() as tape:
        conv, pred = cam_model(tf.convert_to_tensor(x, tf.float32), training=False)
        score = pred[:, 0]
    grads = tape.gradient(score, conv)
    if grads is None:
        return None
    axes = tuple(range(1, len(grads.shape) - 1))
    weights = tf.reduce_mean(grads, axis=axes, keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(conv * weights, axis=-1))[0].numpy()
    peak = float(cam.max())
    if not np.isfinite(peak) or peak < 1e-8:
        return None
    return cam / peak


def consistency_check(model, cam_model, x):
    """Compare the original model with the rebuilt Grad-CAM model on one segment."""
    original = float(model.predict(x, verbose=0).ravel()[0])
    _, out = cam_model(tf.convert_to_tensor(x, tf.float32), training=False)
    rebuilt = float(np.asarray(out).ravel()[0])
    return {"original": original, "gradcam_model": rebuilt,
            "abs_diff": abs(original - rebuilt)}


# ==================================================
# SCREENING PIPELINE
# ==================================================
def run_screening(m, pcg_bytes, ecg_bytes, ecg_name, column, has_header, fs,
                  clinical, ecg_prepared=None, ecg_dataset_label=None):
    Xp, pcg_wave, pcg_warn = preprocess_pcg(pcg_bytes)

    if ecg_prepared is not None:
        # Dataset segments (ecg_segments.npy) are already filtered, normalised
        # and cut to 4000 samples, so they go to the model unchanged.
        seg = np.asarray(ecg_prepared, dtype=np.float32)
        if seg.ndim != 2 or seg.shape[1] != ECG_LEN or len(seg) == 0:
            raise ValueError(f"Dataset ECG segments must have shape (N, {ECG_LEN}).")
        if not np.all(np.isfinite(seg)):
            raise ValueError("The selected ECG segments contain NaN/infinite values.")
        Xe = seg[..., np.newaxis]
        ecg_wave = seg.reshape(-1)
        ecg_warn = ["ECG taken from pre-processed dataset segments; filtering, "
                    "resampling and normalisation were skipped."]
        bpm = None
        fs = ECG_FS
    else:
        raw_ecg = read_ecg(ecg_bytes, ecg_name, column, has_header)
        Xe, ecg_wave, ecg_warn, bpm = preprocess_ecg(raw_ecg, float(fs))

    pf_seg = m["pcg_encoder"].predict(Xp, verbose=0)
    ef_seg = m["ecg_encoder"].predict(Xe, verbose=0)
    if not (np.all(np.isfinite(pf_seg)) and np.all(np.isfinite(ef_seg))):
        raise ValueError("Embeddings contain invalid values.")

    pf = pf_seg.mean(axis=0, keepdims=True)
    ef = ef_seg.mean(axis=0, keepdims=True)

    try:
        out = m["fusion"].predict({"pcg_features": pf, "ecg_features": ef}, verbose=0)
    except Exception:
        out = m["fusion"].predict([pf, ef], verbose=0)

    prob = float(np.asarray(out).reshape(-1)[0])
    if not np.isfinite(prob) or not 0.0 <= prob <= 1.0:
        raise ValueError(
            f"Fusion output {prob:.4f} is not a valid probability. "
            "Check that the saved model ends in a sigmoid."
        )

    pcg_seg_prob = m["pcg"].predict(Xp, verbose=0).ravel()
    ecg_seg_prob = m["ecg"].predict(Xe, verbose=0).ravel()
    pcg_idx, ecg_idx = int(np.argmax(pcg_seg_prob)), int(np.argmax(ecg_seg_prob))

    pcg_x = Xp[pcg_idx:pcg_idx + 1]
    ecg_x = Xe[ecg_idx:ecg_idx + 1]

    return {
        "prob": prob,
        "pcg_seg_prob": pcg_seg_prob, "ecg_seg_prob": ecg_seg_prob,
        "pcg_spec": Xp[pcg_idx, :, :, 0], "ecg_seg": Xe[ecg_idx, :, 0],
        "pcg_idx": pcg_idx, "ecg_idx": ecg_idx,
        "pcg_cam": gradcam(m["pcg_cam"], pcg_x),
        "ecg_cam": gradcam(m["ecg_cam"], ecg_x),
        "pcg_check": consistency_check(m["pcg"], m["pcg_cam"], pcg_x),
        "ecg_check": consistency_check(m["ecg"], m["ecg_cam"], ecg_x),
        "pcg_wave": pcg_wave, "ecg_wave": ecg_wave,
        "warnings": pcg_warn + ecg_warn,
        "ecg_bpm": bpm,
        "ecg_source": "dataset segments" if ecg_prepared is not None else "uploaded signal",
        "dataset_label": ecg_dataset_label,
        "ecg_fs_in": int(fs),
        "n_pcg": int(len(Xp)), "n_ecg": int(len(Xe)),
        "clinical": clinical,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ==================================================
# PLOTS
# ==================================================
def plot_gauge(prob, thr):
    fig, ax = plt.subplots(figsize=(8, 1.5))
    ax.imshow(np.linspace(0, 1, 256)[None, :], extent=[0, 1, 0, 1],
              aspect="auto", cmap="RdYlGn_r", alpha=0.9)
    ax.axvline(thr, color="black", lw=2, ls="--")
    ax.text(thr, 1.12, f"threshold {thr:.2f}", ha="center", fontsize=8)
    ax.plot([prob], [0.5], marker="v", ms=15, color="#0b2545", mec="white")
    ax.text(prob, -0.28, f"score {prob:.3f}", ha="center", fontsize=9,
            fontweight="bold", color="#0b2545")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.4, 1.35)
    ax.set_yticks([])
    ax.set_xlabel("Model abnormal score")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def plot_segment_scores(seg_prob, title, color):
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.bar(np.arange(1, len(seg_prob) + 1), seg_prob, color=color, alpha=0.85)
    ax.axhline(seg_prob.mean(), color="black", ls="--", lw=1,
               label=f"mean {seg_prob.mean():.2f}")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Segment")
    ax.set_ylabel("Abnormal score")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    if len(seg_prob) <= 20:
        ax.set_xticks(np.arange(1, len(seg_prob) + 1))
    fig.tight_layout()
    return fig


def plot_pcg_cam(spec, cam, idx, prob):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    ext = [0, PCG_SEG_SEC, 0, PCG_MELS]
    axes[0].imshow(spec, aspect="auto", origin="lower", extent=ext, cmap="magma")
    axes[0].set(title="Log-Mel spectrogram", xlabel="Time (s)", ylabel="Mel band")
    axes[1].imshow(spec, aspect="auto", origin="lower", extent=ext, cmap="gray")
    if cam is not None:
        up = zoom(cam, (PCG_MELS / cam.shape[0], PCG_FRAMES / cam.shape[1]), order=1)
        axes[1].imshow(up, aspect="auto", origin="lower", extent=ext,
                       cmap="jet", alpha=0.45)
    axes[1].set(title="Grad-CAM overlay", xlabel="Time (s)")
    fig.suptitle(f"PCG segment {idx + 1} (most abnormal) | score {prob:.3f}", fontsize=10)
    fig.tight_layout()
    return fig


def plot_ecg_cam(seg, cam, idx, prob):
    t = np.arange(len(seg)) / ECG_FS
    fig, ax = plt.subplots(figsize=(9, 3.2))
    if cam is not None:
        c = np.interp(np.linspace(0, len(cam) - 1, len(seg)), np.arange(len(cam)), cam)
        c = gaussian_filter1d(c, sigma=40)
        c = c / (c.max() + 1e-8)
        ax.imshow(np.tile(c, (60, 1)), extent=[t[0], t[-1], seg.min(), seg.max()],
                  aspect="auto", origin="lower", cmap="jet", alpha=0.30)
    ax.plot(t, seg, color="black", lw=0.9)
    ax.set(xlabel="Time (s)", ylabel="Normalized amplitude",
           title=f"ECG segment {idx + 1} (most abnormal) | score {prob:.3f}")
    ax.set_xlim(t[0], t[-1])
    fig.tight_layout()
    return fig


def plot_waveform(wave, sr, title, color):
    fig, ax = plt.subplots(figsize=(9, 2.2))
    ax.plot(np.arange(len(wave)) / sr, wave, lw=0.7, color=color)
    ax.set(xlabel="Time (s)", ylabel="Amplitude", title=title)
    fig.tight_layout()
    return fig


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


# ==================================================
# RESULT RENDERING
# ==================================================
def render_check(label, chk):
    ok = chk["abs_diff"] < 1e-4
    msg = (f"{label}: original {chk['original']:.6f} | Grad-CAM model "
           f"{chk['gradcam_model']:.6f} | difference {chk['abs_diff']:.2e}")
    if ok:
        st.success(msg + " (consistent)")
    else:
        st.error(msg + " - mismatch. Check that the Grad-CAM model has the "
                       "same layers and weights as the saved model.")


def render_result(r, thr):
    prob = r["prob"]
    abnormal = prob >= thr
    borderline = abs(prob - thr) < 0.05

    if borderline:
        css, head = "bor", "Borderline"
        msg = ("The score is very close to the decision threshold. Treat this "
               "as inconclusive and consider a repeat recording.")
    elif abnormal:
        css, head = "abn", "Abnormal"
        msg = ("The model flags this PCG + ECG pair as abnormal. Clinical "
               "evaluation is needed; this is not a diagnosis.")
    else:
        css, head = "nor", "Normal"
        msg = ("The model does not flag this PCG + ECG pair. This does not "
               "rule out RHD or replace clinical evaluation.")

    tab_res, tab_xai, tab_sig, tab_rep = st.tabs(
        ["Result", "Explainability", "Signals & validation", "Report"]
    )

    with tab_res:
        st.markdown(f'<div class="result-card {css}"><h2>{head}</h2><p>{msg}</p></div>',
                    unsafe_allow_html=True)

        if r["dataset_label"] is not None:
            true_txt = "Abnormal" if r["dataset_label"] == 1 else "Normal"
            st.info(f"Dataset label for the selected ECG recording: **{true_txt}** "
                    "(shown for comparison only; not used by the model).")
        st.markdown("**Inputs used by the model**")
        st.markdown('<div class="mod mod-on">✅ PCG heart sound: used (2D-CNN branch)</div>'
                    '<div class="mod mod-on">✅ ECG waveform: used (1D-CNN branch)</div>'
                    '<div class="mod mod-on">✅ PCG + ECG attention fusion: used for the final score</div>'
                    '<div class="mod mod-off">⛔ Clinical data: recorded in the report only. '
                    'The saved fusion model was not trained with clinical inputs, so they '
                    'do not affect the score.</div>',
                    unsafe_allow_html=True)
        st.write("")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fusion score", f"{prob:.3f}")
        c2.metric("Threshold", f"{thr:.3f}", f"{prob - thr:+.3f} vs threshold",
                  delta_color="off")
        c3.metric("PCG-only score", f"{r['pcg_seg_prob'].mean():.3f}")
        c4.metric("ECG-only score", f"{r['ecg_seg_prob'].mean():.3f}")
        show(plot_gauge(prob, thr))
        st.caption(
            "The score is the model's raw output, not a calibrated clinical "
            "probability. PCG-only and ECG-only scores are the branch models' "
            "mean segment outputs; their own operating thresholds differ from "
            "the fusion threshold."
        )

        a, b = st.columns(2)
        with a:
            show(plot_segment_scores(r["pcg_seg_prob"],
                                     f"PCG segment scores ({r['n_pcg']})", "#13607a"))
        with b:
            show(plot_segment_scores(r["ecg_seg_prob"],
                                     f"ECG segment scores ({r['n_ecg']})", "#b54708"))

        with st.expander("Reference performance (internal test split)"):
            st.dataframe(REFERENCE_METRICS.set_index("Metric"))
            st.caption(
                "Values copied from notebook cell 74 (61 held-out recordings). "
                "Confirm them against the live output of the evaluation cells "
                "before using them in a report. With 61 recordings the "
                "uncertainty on every figure is wide, and fusion specificity "
                "near 61% means many normal recordings are still flagged."
            )

    with tab_xai:
        st.caption(
            "Grad-CAM is computed per branch on the segment that branch scored "
            "as most abnormal. It does not explain the fusion model, and the "
            "highlighted regions are attribution signals, not clinical evidence."
        )
        render_check("PCG check", r["pcg_check"])
        render_check("ECG check", r["ecg_check"])
        left, right = st.columns(2)
        with left:
            st.subheader("PCG heart sound")
            if r["pcg_cam"] is None:
                st.warning("PCG Grad-CAM unavailable (zero or missing gradients).")
            show(plot_pcg_cam(r["pcg_spec"], r["pcg_cam"], r["pcg_idx"],
                              float(r["pcg_seg_prob"][r["pcg_idx"]])))
        with right:
            st.subheader("ECG waveform")
            if r["ecg_cam"] is None:
                st.warning("ECG Grad-CAM unavailable (zero or missing gradients).")
            show(plot_ecg_cam(r["ecg_seg"], r["ecg_cam"], r["ecg_idx"],
                              float(r["ecg_seg_prob"][r["ecg_idx"]])))

    with tab_sig:
        if r["warnings"]:
            for w in r["warnings"]:
                st.warning(w)
        else:
            st.success("All input validation checks passed with no warnings.")
        q1, q2, q3 = st.columns(3)
        q1.metric("PCG duration used", f"{r['n_pcg'] * PCG_SEG_SEC:.1f} s",
                  f"{r['n_pcg']} segments", delta_color="off")
        q2.metric("ECG duration used", f"{r['n_ecg'] * ECG_LEN / ECG_FS:.1f} s",
                  f"{r['n_ecg']} segments", delta_color="off")
        q3.metric("Rough ECG rate",
                  "n/a" if r["ecg_bpm"] is None else f"{r['ecg_bpm']:.0f} bpm")
        if r["n_pcg"] < 3 or r["n_ecg"] < 3:
            st.info("Fewer than 3 segments in one modality; longer recordings "
                    "give a more stable recording-level score.")
        show(plot_waveform(r["pcg_wave"], PCG_SR,
                           "Uploaded PCG (resampled to 2 kHz, peak-normalized)", "#13607a"))
        show(plot_waveform(r["ecg_wave"], ECG_FS,
                           "Uploaded ECG (band-pass, robust-normalized, 2 kHz)", "#b54708"))

    with tab_rep:
        report = {
            "timestamp": r["timestamp"],
            "predicted_class": head,
            "fusion_score": round(prob, 4),
            "threshold": round(float(thr), 4),
            "pcg_only_mean_score": round(float(r["pcg_seg_prob"].mean()), 4),
            "ecg_only_mean_score": round(float(r["ecg_seg_prob"].mean()), 4),
            "pcg_segments": r["n_pcg"],
            "ecg_segments": r["n_ecg"],
            "ecg_input_sampling_rate_hz": r["ecg_fs_in"],
            "model_inputs": "PCG + ECG only",
            "ecg_source": r["ecg_source"],
            "dataset_label_if_known": r["dataset_label"],
            "clinical_data_used_by_model": False,
            "clinical_context_recorded_only": r["clinical"],
            "input_validation_warnings": r["warnings"],
            "gradcam_consistency_abs_diff": {
                "pcg": r["pcg_check"]["abs_diff"],
                "ecg": r["ecg_check"]["abs_diff"],
            },
            "disclaimer": "Research prototype. Not clinically validated. "
                          "Not for diagnosis or treatment decisions.",
        }
        st.json(report)
        d1, d2 = st.columns(2)
        d1.download_button("Download report (JSON)", json.dumps(report, indent=2),
                           file_name="rhd_screening_report.json",
                           mime="application/json")
        flat = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                for k, v in report.items()}
        d2.download_button("Download report (CSV)",
                           pd.DataFrame([flat]).to_csv(index=False),
                           file_name="rhd_screening_report.csv", mime="text/csv")


# ==================================================
# LOAD MODELS
# ==================================================
try:
    models = load_models()
    with st.expander("Model details"):
        st.write("PCG input:", models["pcg"].input_shape)
        st.write("ECG input:", models["ecg"].input_shape)
        st.write("Fusion inputs:", models["fusion"].input_shape)
        st.write("Fusion output:", models["fusion"].output_shape)
except Exception as exc:
    st.error(f"Unable to load models: {type(exc).__name__}: {exc}")
    st.info("Check that app.py is in the RHD_Project root and all three "
            "model files exist under processed/.")
    st.stop()


# ==================================================
# SIDEBAR INPUTS
# ==================================================
with st.sidebar:
    st.header("1. Upload recordings")
    pcg_file = st.file_uploader("PCG heart sound (.wav)", type=["wav"])
    ecg_file = st.file_uploader("ECG signal (.csv, .txt, .npy)",
                                type=["csv", "txt", "npy"])
    st.caption("A single ECG signal is expected. ecg_metadata.csv is rejected as a "
               "signal; it is only accepted as a recording picker for ecg_segments.npy.")

    st.header("2. ECG settings")
    fs = st.number_input("ECG sampling rate (Hz)", min_value=100,
                         max_value=10000, value=2000, step=100)
    ecg_has_header = st.checkbox("ECG CSV/TXT has a header row", value=False)

    ecg_bytes = ecg_file.getvalue() if ecg_file is not None else None
    ecg_column, ecg_error = None, None
    ecg_prepared, ecg_dataset_label = None, None

    if ecg_file is not None and ecg_file.name.lower().endswith(".npy"):
        try:
            _arr = np.load(io.BytesIO(ecg_bytes), allow_pickle=False)
        except Exception as exc:
            _arr = None
            ecg_error = f"Could not read the .npy file: {exc}"
            st.error(ecg_error)

        if _arr is not None and _arr.squeeze().ndim == 2:
            st.info(f"This array has shape {_arr.shape}: it holds many "
                    "pre-segmented recordings. Upload ecg_metadata.csv below "
                    "and pick ONE recording to test.")
            meta_file = st.file_uploader(
                "ecg_metadata.csv (only used to pick a recording)", type=["csv"],
                key="ecg_meta_picker")
            if _arr.shape[1] != ECG_LEN:
                ecg_error = (f"Segments must be {ECG_LEN} samples long; "
                             f"got {_arr.shape[1]}.")
                st.error(ecg_error)
            elif meta_file is None:
                ecg_error = "Upload ecg_metadata.csv to choose a recording."
                st.warning(ecg_error)
            else:
                try:
                    _meta = pd.read_csv(io.BytesIO(meta_file.getvalue()))
                    if "recording_id" not in _meta.columns:
                        raise ValueError("ecg_metadata.csv has no 'recording_id' column.")
                    if len(_meta) != len(_arr):
                        raise ValueError(
                            f"Metadata has {len(_meta)} rows but the array has "
                            f"{len(_arr)} segments; they do not match.")
                    rid = st.selectbox("Recording", sorted(_meta["recording_id"].unique()))
                    _mask = (_meta["recording_id"] == rid).to_numpy()
                    ecg_prepared = _arr[_mask].astype(np.float32)
                    if "label" in _meta.columns:
                        ecg_dataset_label = int(_meta.loc[_mask, "label"].iloc[0])
                    st.caption(f"{int(_mask.sum())} segments selected "
                               "(sampling-rate setting is ignored for dataset segments).")
                except Exception as exc:
                    ecg_error = f"{type(exc).__name__}: {exc}"
                    st.error(ecg_error)

    if ecg_file is not None and not ecg_file.name.lower().endswith(".npy"):
        try:
            _, valid_cols, excluded = scan_ecg_table(
                ecg_bytes.decode("utf-8-sig", errors="replace"),
                ecg_file.name, ecg_has_header,
            )
            ecg_column = st.selectbox("ECG signal column", options=valid_cols,
                                      format_func=lambda v: f"Column {v}")
            for c, why in excluded.items():
                st.caption(f"Column {c} ignored: {why}")
        except ValueError as exc:
            ecg_error = str(exc)
            st.error(ecg_error)

    st.header("3. Clinical context (optional)")
    with st.expander("Enter clinical details"):
        st.caption("Saved in the report only. The model does not use these values.")
        age = st.number_input("Age (years)", 0, 120, 0, 1)
        sex = st.selectbox("Sex", ["Not stated", "Female", "Male", "Other"])
        symptoms = st.multiselect(
            "Reported symptoms / history",
            ["Breathlessness", "Chest pain", "Fatigue", "Palpitations",
             "Previous rheumatic fever", "Known heart murmur"],
        )
        notes = st.text_area("Notes", "", height=70)
    clinical = {
        "age_years": int(age) if age > 0 else None,
        "sex": sex,
        "symptoms_history": symptoms,
        "notes": notes.strip(),
    }

    st.header("4. Decision threshold")
    threshold = st.slider(
        "Abnormal if score >=", 0.05, 0.95, float(DEFAULT_THRESHOLD), 0.01,
        help="Default is the validation-selected Youden threshold.",
    )

    run = st.button("Run PCG + ECG screening", type="primary",
                    disabled=(pcg_file is None or ecg_file is None
                              or ecg_error is not None))


# ==================================================
# MAIN
# ==================================================
if run:
    try:
        with st.spinner("Validating inputs and running models..."):
            st.session_state["result"] = run_screening(
                models, pcg_file.getvalue(), ecg_bytes, ecg_file.name,
                ecg_column, ecg_has_header, fs, clinical,
                ecg_prepared=ecg_prepared, ecg_dataset_label=ecg_dataset_label,
            )
    except ValueError as exc:
        st.session_state.pop("result", None)
        st.error(f"Input rejected: {exc}")
    except Exception as exc:
        st.session_state.pop("result", None)
        st.error(f"Prediction failed: {type(exc).__name__}: {exc}")

if "result" in st.session_state:
    render_result(st.session_state["result"], threshold)
else:
    st.markdown("### How to use")
    for i, text in enumerate(
        [
            "Upload a PCG <code>.wav</code> recording (at least 1.2 s, native rate at least 1 kHz).",
            "Upload the ECG <b>signal</b> as <code>.csv</code>, <code>.txt</code> or a 1-D <code>.npy</code> (at least 2 s). Metadata files are rejected.",
            "Set the true ECG sampling rate, the header option and the signal column.",
            "Optionally enter clinical context (report only, not used by the model).",
            "Click <b>Run PCG + ECG screening</b>.",
        ],
        start=1,
    ):
        st.markdown(f'<div class="step"><b>{i}.</b> {text}</div>',
                    unsafe_allow_html=True)

st.caption("Research prototype only. Not clinically validated. Do not use "
           "this output for diagnosis or treatment decisions.")
