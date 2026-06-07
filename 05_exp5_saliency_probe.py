# =============================================================================
# Program      : 05_exp5_saliency_probe.py
# Version      : 1.0
# Description  : Experiment 5 — Frequency-Axis Gradient Saliency Probe
#
#                Computes per-genre frequency-axis saliency maps from the
#                best-fold models saved by Program 01 (M=40 and M=128),
#                using the full GTZAN corpus (all 1000 clips, no train/val
#                split) as the analysis set.
#
#                For each genre g and model M, the saliency map is:
#                  s_g^(m) = (1/T) * sum_t | d(y_hat_g) / d(X_{m,t}) |
#                averaged over all clips of genre g. This gives a vector
#                of length M indicating which mel bands drive genre
#                prediction for each genre.
#
#                Produces paper Figure 2: two-panel saliency comparison
#                for hip-hop (high-spread) and classical (low-spread)
#                under M=40 and M=128, showing how increased band count
#                shifts attention differently across genre groups.
#
#                Also computes:
#                  - Jensen-Shannon divergence between saliency distributions
#                    of low-spread and high-spread genre groups
#                  - Spearman correlation between saliency mass below 500 Hz
#                    and per-genre accuracy (from exp1_results.json)
#
# INPUT        :
#                  Google Drive:
#                    /content/drive/MyDrive/datasets/GTZAN.zip
#                  Drive:
#                    exp1_best_model_M40.pth
#                    exp1_best_model_M128.pth
#                    precompute_genre_centroids.json
#                    exp1_results.json
#
# STEPS        :
#                  Step 1  Mount Drive, copy and extract GTZAN
#                  Step 2  Download models and support files from Drive
#                  Step 3  Build full-corpus mel-spectrograms for M=40
#                            and M=128 (f_max=8000, Slaney, all clips)
#                  Step 4  Compute per-genre saliency maps for each model
#                  Step 5  Compute JS divergence between group saliency
#                            distributions
#                  Step 6  Compute Spearman correlation (sub-500Hz saliency
#                            mass vs per-genre accuracy)
#                  Step 7  Generate Figure 2 (two-panel saliency comparison)
#                  Step 8  Save results JSON and figure, upload to Drive
#
# OUTPUT FILES :
#                  exp5_saliency_results.json
#                      Per (M, genre): saliency vector (length M),
#                      sub-500Hz saliency mass, group label
#                      JS divergence between group saliency distributions
#                      Spearman r and p for sub-500Hz mass vs accuracy
#
#                  fig_05_02_saliency_profiles.png
#                      Paper Figure 2: two-panel saliency profiles for
#                      hip-hop and classical under M=40 and M=128
#
# GPU Required : YES (gradient computation is faster on GPU)
# Dependencies : torch, librosa, numpy, scipy, matplotlib, tqdm
#
# Change Log   :
#   v1.0  2026-06-07  Initial version
# =============================================================================

import subprocess
import sys

for pkg in ["torch", "librosa", "numpy", "scipy", "matplotlib", "tqdm"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

import torch
if not torch.cuda.is_available():
    print("\n[ERROR] GPU not detected!")
    print("This script requires a GPU.")
    print("Please switch your Colab runtime to a T4 GPU and restart.")
    sys.exit(1)
print("CUDA available: True. Proceeding...")

# ── Standard imports ──────────────────────────────────────────────────────────
import os
import json
import zipfile
import shutil
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
import torch
import torch.nn as nn
import librosa

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR     = "/content/drive/MyDrive/paper/mel-fairness-probe/"  # Persistent storage
DRIVE_DIR       = "/content/drive/MyDrive/datasets"
GTZAN_ZIP       = os.path.join(DRIVE_DIR, "GTZAN.zip")
LOCAL_WORK_DIR  = "/tmp/gtzan_fairness"
GTZAN_LOCAL_ZIP = os.path.join(LOCAL_WORK_DIR, "GTZAN.zip")
GTZAN_EXTRACT   = os.path.join(LOCAL_WORK_DIR, "gtzan")
OUTPUT_DIR      = os.path.join(LOCAL_WORK_DIR, "outputs_exp5")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Audio ─────────────────────────────────────────────────────────────────────
SR         = 22050
HOP_LENGTH = 512
N_FFT      = 2048
FMAX       = 8000
FMIN       = 0.0
NORM       = "slaney"
DURATION   = 29.0

# ── Experiment ────────────────────────────────────────────────────────────────
M_VALUES     = [40, 128]
DEVICE       = torch.device("cuda")
SUB500_HZ    = 500.0     # saliency mass threshold for Spearman correlation

GTZAN_GENRES = [
    "blues", "classical", "country", "disco",
    "hiphop", "jazz", "metal", "pop", "reggae", "rock"
]
N_CLASSES = len(GTZAN_GENRES)

# Genres for Figure 2 panels
FIG2_LOW_SPREAD  = "classical"
FIG2_HIGH_SPREAD = "hiphop"

# =============================================================================
# GOOGLE DRIVE HELPERS
# =============================================================================

def ensure_project_dir():
    """Create project directory in Google Drive if it doesn't exist."""
    os.makedirs(PROJECT_DIR, exist_ok=True)

def save_to_drive(local_filepath, remote_filename):
    """Copy a local file to Google Drive project folder."""
    ensure_project_dir()
    dest_path = os.path.join(PROJECT_DIR, remote_filename)
    try:
        shutil.copy2(local_filepath, dest_path)
        print(f"  [DRIVE] Uploaded: {remote_filename}")
    except Exception as e:
        print(f"  [DRIVE] Upload failed for {remote_filename}: {e}")

def load_from_drive(remote_filename, local_filepath):
    """Copy a file from Google Drive project folder to local path."""
    ensure_project_dir()
    src_path = os.path.join(PROJECT_DIR, remote_filename)
    if os.path.exists(src_path):
        try:
            shutil.copy2(src_path, local_filepath)
            return True
        except Exception as e:
            print(f"  [DRIVE] Download failed for {remote_filename}: {e}")
            return False
    else:
        return False

def list_drive_files():
    """List files in the Google Drive project directory."""
    ensure_project_dir()
    try:
        return [f for f in os.listdir(PROJECT_DIR) if os.path.isfile(os.path.join(PROJECT_DIR, f))]
    except Exception as e:
        print(f"  [DRIVE] Could not list files: {e}")
        return []

# =============================================================================
# STEP 1 — PREPARE DATASET
# =============================================================================

def step1_prepare_dataset():
    print("\n=== STEP 1: Prepare GTZAN dataset ===")
    from google.colab import drive
    drive.mount("/content/drive")

    for candidate in ["genres_original", "genres",
                      "Data/genres_original", "data/genres_original"]:
        p = os.path.join(GTZAN_EXTRACT, candidate)
        if os.path.isdir(p):
            print(f"  [SKIP] Already extracted at {p}")
            return p

    print(f"  Copying {GTZAN_ZIP} -> {GTZAN_LOCAL_ZIP} ...")
    os.makedirs(LOCAL_WORK_DIR, exist_ok=True)
    shutil.copy2(GTZAN_ZIP, GTZAN_LOCAL_ZIP)
    print("  Copy complete.")

    print(f"  Extracting to {GTZAN_EXTRACT} ...")
    with zipfile.ZipFile(GTZAN_LOCAL_ZIP, "r") as zf:
        zf.extractall(GTZAN_EXTRACT)
    print("  Extraction complete.")

    genre_root = None
    for candidate in ["genres_original", "genres",
                      "Data/genres_original", "data/genres_original"]:
        p = os.path.join(GTZAN_EXTRACT, candidate)
        if os.path.isdir(p):
            genre_root = p
            break
    if genre_root is None:
        for root, dirs, files in os.walk(GTZAN_EXTRACT):
            if any(f.endswith(".wav") for f in files):
                genre_root = os.path.dirname(root)
                break
    print(f"  Genre root: {genre_root}")
    return genre_root

# =============================================================================
# STEP 2 — DOWNLOAD MODELS AND SUPPORT FILES
# =============================================================================

def step2_download_inputs():
    print("\n=== STEP 2: Download models and support files from Drive ===")
    drive_files = list_drive_files()
    
    files_needed = {
        "exp1_best_model_M40.pth"          : "exp1_best_model_M40.pth",
        "exp1_best_model_M128.pth"         : "exp1_best_model_M128.pth",
        "precompute_genre_centroids.json"  : "precompute_genre_centroids.json",
        "exp1_results.json"                : "exp1_results.json"
    }
    
    for remote, local_name in files_needed.items():
        local_path = os.path.join(OUTPUT_DIR, local_name)
        if os.path.exists(local_path):
            print(f"  [SKIP] {local_name} already local.")
            continue
        if remote in drive_files:
            if load_from_drive(remote, local_path):
                print(f"  Downloaded: {local_name}")
            else:
                raise RuntimeError(f"Cannot download {remote} from Drive. "
                                   f"Ensure Program 01 completed successfully.")
        else:
            raise RuntimeError(f"File {remote} not found on Drive. "
                               f"Ensure Program 01 completed successfully.")

    with open(os.path.join(OUTPUT_DIR,
                           "precompute_genre_centroids.json")) as f:
        centroid_data = json.load(f)

    with open(os.path.join(OUTPUT_DIR, "exp1_results.json")) as f:
        exp1_results = json.load(f)

    return centroid_data, exp1_results

# =============================================================================
# MODEL
# =============================================================================

class GenreCNN(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        self.block1 = conv_block(1,   32)
        self.block2 = conv_block(32,  64)
        self.block3 = conv_block(64, 128)
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# =============================================================================
# STEP 3 — BUILD FULL-CORPUS SPECTROGRAMS
# =============================================================================

def compute_melspec(wav_path, n_mels):
    y, _ = librosa.load(wav_path, sr=SR, mono=True, duration=DURATION)
    S    = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=n_mels, fmin=FMIN, fmax=FMAX,
        norm=NORM, power=2.0
    )
    return librosa.power_to_db(S, ref=np.max).astype(np.float32)


def step3_build_spectrograms(genre_root):
    """
    Build and cache full-corpus mel-spectrograms for M=40 and M=128.
    Returns dict: M -> {genre -> list of (M, T) np.float32 arrays}
    Each genre retains its clips as a list (variable T handled per clip
    during saliency computation).
    """
    print("\n=== STEP 3: Build full-corpus mel-spectrograms ===")
    result = {}

    for M in M_VALUES:
        cache_path = os.path.join(OUTPUT_DIR,
                                  f"saliency_specs_M{M}.npz")
        if os.path.exists(cache_path):
            print(f"  [CACHE] M={M}: loading from local cache.")
            loaded = np.load(cache_path, allow_pickle=True)
            result[M] = {g: loaded[g] for g in GTZAN_GENRES
                         if g in loaded}
            continue

        print(f"  Computing spectrograms for M={M} ...")
        genre_specs = {g: [] for g in GTZAN_GENRES}

        for genre in tqdm(GTZAN_GENRES, desc=f"    M={M}"):
            genre_dir = None
            for cand in [genre, genre.capitalize()]:
                p = os.path.join(genre_root, cand)
                if os.path.isdir(p):
                    genre_dir = p
                    break
            if genre_dir is None:
                print(f"    [WARN] Genre dir not found: {genre}")
                continue

            wav_files = sorted([f for f in os.listdir(genre_dir)
                                 if f.endswith(".wav")])
            for wav_name in wav_files:
                wav_path = os.path.join(genre_dir, wav_name)
                try:
                    spec = compute_melspec(wav_path, n_mels=M)
                    genre_specs[genre].append(spec)
                except Exception as e:
                    print(f"    [WARN] Skipping {wav_name}: {e}")

        # Save as npz with object arrays per genre
        save_dict = {g: np.array(genre_specs[g], dtype=object)
                     for g in GTZAN_GENRES}
        np.savez(cache_path, **save_dict)
        print(f"    M={M}: cached {sum(len(v) for v in genre_specs.values())} clips.")
        result[M] = genre_specs

    return result

# =============================================================================
# STEP 4 — COMPUTE PER-GENRE SALIENCY MAPS
# =============================================================================

def load_model(M):
    """Load GenreCNN from Drive-downloaded PTH file."""
    pth_path = os.path.join(OUTPUT_DIR, f"exp1_best_model_M{M}.pth")
    ckpt     = torch.load(pth_path, map_location=DEVICE)
    model    = GenreCNN(n_classes=N_CLASSES).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def compute_saliency_one_clip(model, spec_np, genre_idx):
    """
    Compute frequency-axis saliency for one clip and one genre class.
    Returns saliency vector of shape (M,) = mean over time of |dY/dX|.
    """
    # spec_np: (M, T) -> tensor (1, 1, M, T)
    spec_t = torch.from_numpy(spec_np[np.newaxis, np.newaxis, :, :]).to(DEVICE)
    spec_t.requires_grad_(True)

    logits = model(spec_t)                          # (1, n_classes)
    score  = logits[0, genre_idx]                   # scalar for genre class
    model.zero_grad()
    score.backward()

    # Gradient shape: (1, 1, M, T)
    grad = spec_t.grad.detach().cpu().numpy()[0, 0]  # (M, T)
    return np.abs(grad).mean(axis=1)                 # (M,)


def step4_compute_saliency(spectrogram_data, genre_idx_map):
    """
    For each M and each genre, compute mean saliency vector across all clips.
    Returns dict: M -> {genre -> saliency array (M,)}
    """
    print("\n=== STEP 4: Compute per-genre saliency maps ===")

    saliency_data = {}

    for M in M_VALUES:
        print(f"\n  M = {M}")
        model      = load_model(M)
        genre_sals = {}

        for genre in tqdm(GTZAN_GENRES, desc=f"    M={M} genres"):
            clips      = spectrogram_data[M][genre]
            genre_idx  = genre_idx_map[genre]
            sal_accum  = np.zeros(M, dtype=np.float64)
            count      = 0

            for spec in clips:
                try:
                    sal       = compute_saliency_one_clip(model, spec, genre_idx)
                    sal_accum += sal
                    count     += 1
                except Exception as e:
                    print(f"      [WARN] Saliency failed for {genre}: {e}")

            if count > 0:
                genre_sals[genre] = sal_accum / count
            else:
                genre_sals[genre] = np.zeros(M)

            print(f"    {genre:<12s}  clips={count}  "
                  f"peak_band={genre_sals[genre].argmax()}")

        saliency_data[M] = genre_sals

    return saliency_data

# =============================================================================
# STEP 5 — JS DIVERGENCE BETWEEN GROUP SALIENCY DISTRIBUTIONS
# =============================================================================

def step5_js_divergence(saliency_data, centroid_data):
    """
    For each M, compute JS divergence between the mean saliency distribution
    of the low-spread group and the high-spread group.
    """
    print("\n=== STEP 5: JS divergence between group saliency distributions ===")
    js_results = {}

    for M in M_VALUES:
        low_genres  = [g for g in GTZAN_GENRES
                       if centroid_data.get(g, {}).get("group") == "low_spread"]
        high_genres = [g for g in GTZAN_GENRES
                       if centroid_data.get(g, {}).get("group") == "high_spread"]

        # Mean saliency vector per group, normalised to probability distribution
        low_sal  = np.mean([saliency_data[M][g] for g in low_genres],  axis=0)
        high_sal = np.mean([saliency_data[M][g] for g in high_genres], axis=0)

        # Normalise to sum to 1 (probability distribution)
        low_sal  = low_sal  / (low_sal.sum()  + 1e-12)
        high_sal = high_sal / (high_sal.sum() + 1e-12)

        js = float(jensenshannon(low_sal, high_sal))
        js_results[M] = round(js, 6)
        print(f"  M={M:3d}  JS divergence (low vs high spread) = {js:.6f}")

    return js_results

# =============================================================================
# STEP 6 — SPEARMAN CORRELATION: SUB-500 HZ SALIENCY MASS VS ACCURACY
# =============================================================================

def step6_spearman_correlation(saliency_data, exp1_results, centroid_data):
    """
    For each M, compute Spearman r between sub-500Hz saliency mass and
    per-genre mean accuracy from Experiment 1.
    Sub-500Hz saliency mass = sum of saliency values for mel bands whose
    centre frequency <= 500 Hz.
    """
    print("\n=== STEP 6: Spearman correlation "
          "(sub-500Hz saliency mass vs accuracy) ===")

    def mel_filter_centres(M, fmin, fmax):
        hz_to_mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
        mel_to_hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
        m_min  = hz_to_mel(fmin if fmin > 0 else 1.0)
        m_max  = hz_to_mel(fmax)
        m_pts  = np.linspace(m_min, m_max, M + 2)
        return mel_to_hz(m_pts)[1:-1]

    spearman_results = {}

    for M in M_VALUES:
        centres    = mel_filter_centres(M, FMIN, FMAX)
        sub500_idx = np.where(centres <= SUB500_HZ)[0]

        sub500_mass = []
        accuracies  = []

        for genre in GTZAN_GENRES:
            sal  = saliency_data[M][genre]
            mass = float(sal[sub500_idx].sum()) if len(sub500_idx) > 0 else 0.0
            acc  = (exp1_results
                    .get("per_M", {})
                    .get(str(M), {})
                    .get("per_genre", {})
                    .get(genre, {})
                    .get("mean_acc", None))
            if acc is not None:
                sub500_mass.append(mass)
                accuracies.append(acc)

        if len(sub500_mass) >= 3:
            r, p = spearmanr(sub500_mass, accuracies)
            print(f"  M={M:3d}  Spearman r = {r:.4f}  p = {p:.4f}  "
                  f"n_sub500_bands = {len(sub500_idx)}")
            spearman_results[M] = {
                "spearman_r"      : round(float(r), 4),
                "spearman_p"      : round(float(p), 4),
                "n_sub500_bands"  : int(len(sub500_idx)),
                "sub500_mass_per_genre": {
                    g: round(float(saliency_data[M][g][sub500_idx].sum()
                                   if len(sub500_idx) > 0 else 0.0), 6)
                    for g in GTZAN_GENRES
                }
            }
        else:
            print(f"  M={M:3d}  No sub-500Hz bands found.")
            spearman_results[M] = {"spearman_r": None, "spearman_p": None}

    return spearman_results

# =============================================================================
# STEP 7 — FIGURE 2: TWO-PANEL SALIENCY PROFILES
# =============================================================================

def step7_make_figure2(saliency_data, centroid_data):
    """
    Paper Figure 2: two-panel comparison of saliency profiles.
    Left panel  : classical (low-spread)  — M=40 vs M=128
    Right panel : hiphop   (high-spread)  — M=40 vs M=128

    For comparability across M values, saliency vectors are plotted against
    normalised mel-band index (0 to 1) rather than absolute band number,
    and each saliency vector is L1-normalised so areas are comparable.
    Actual Hz tick labels are added using the M=128 filter centres.
    """
    print("\n=== STEP 7: Generate Figure 2 ===")

    def mel_filter_centres(M, fmin, fmax):
        hz_to_mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
        mel_to_hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
        m_pts = np.linspace(hz_to_mel(fmin if fmin > 0 else 1.0),
                            hz_to_mel(fmax), M + 2)
        return mel_to_hz(m_pts)[1:-1]

    genres_to_plot = [FIG2_LOW_SPREAD, FIG2_HIGH_SPREAD]
    panel_titles   = [
        f"{FIG2_LOW_SPREAD.capitalize()} (low-spread, "
        f"$\\sigma_g$ = {centroid_data[FIG2_LOW_SPREAD]['sigma_hz']:.0f} Hz)",
        f"Hip-Hop (high-spread, "
        f"$\\sigma_g$ = {centroid_data[FIG2_HIGH_SPREAD]['sigma_hz']:.0f} Hz)"
    ]
    colors = {40: "#1B9E77", 128: "#D95F02"}
    labels = {40: "$M = 40$", 128: "$M = 128$"}

    fig = plt.figure(figsize=(10, 4.5))
    gs  = gridspec.GridSpec(1, 2, wspace=0.38)

    for panel_idx, genre in enumerate(genres_to_plot):
        ax = fig.add_subplot(gs[panel_idx])

        for M in M_VALUES:
            sal      = saliency_data[M][genre].copy()
            sal_norm = sal / (sal.max() + 1e-12)
            centres  = mel_filter_centres(M, FMIN, FMAX)

            # Plot against actual Hz on x-axis
            ax.plot(centres, sal_norm,
                    color=colors[M], linewidth=2.5,
                    label=labels[M], alpha=0.88)
            ax.fill_between(centres, sal_norm,
                            alpha=0.08, color=colors[M])

        ax.axvline(SUB500_HZ, color="gray", linestyle=":",
                   linewidth=1.0, alpha=0.7, label="500 Hz")
        ax.set_title(panel_titles[panel_idx], fontsize=10, pad=8)
        ax.set_xlabel("Mel Filter Centre Frequency (Hz)", fontsize=9)
        if panel_idx == 0:
            ax.set_ylabel("Normalised Saliency", fontsize=9)
        ax.legend(fontsize=8.5, loc="upper right")
        ax.set_xlim(0, FMAX)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
        ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.4)

        # Shade sub-500Hz region
        ax.axvspan(0, SUB500_HZ, alpha=0.06, color="gray", zorder=0)

        # Group label in corner
        group = centroid_data[genre]["group"]
        group_color = "#1B9E77" if group == "low_spread" else "#D95F02"
        ax.text(0.97, 0.97,
                "Low-spread" if group == "low_spread" else "High-spread",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=group_color, fontweight="bold")

    fig.suptitle(
        "Frequency-Axis Gradient Saliency: Classical vs Hip-Hop\n"
        "($f_{\\max}$ = 8 kHz, Slaney norm; shaded region: $f < 500$ Hz)",
        fontsize=10, y=1.02
    )
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "fig_05_02_saliency_profiles.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path

# =============================================================================
# STEP 8 — SAVE RESULTS JSON AND UPLOAD
# =============================================================================

def step8_save_results(saliency_data, js_results,
                       spearman_results, centroid_data, fig_path):
    print("\n=== STEP 8: Save results JSON and upload ===")

    results = {
        "experiment"    : "Exp5_SaliencyProbe",
        "fixed_params"  : {
            "fmax_hz"        : FMAX,
            "norm"           : NORM,
            "corpus"         : "GTZAN_full",
            "sub500_threshold": SUB500_HZ
        },
        "js_divergence_group_saliency": {
            str(M): js_results[M] for M in M_VALUES
        },
        "spearman_sub500_vs_accuracy": {
            str(M): spearman_results[M] for M in M_VALUES
        },
        "per_M_per_genre_saliency": {}
    }

    for M in M_VALUES:
        results["per_M_per_genre_saliency"][str(M)] = {}
        centres = []
        hz_to_mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
        mel_to_hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
        m_pts   = np.linspace(hz_to_mel(1.0), hz_to_mel(FMAX), M + 2)
        centres = mel_to_hz(m_pts)[1:-1]

        for genre in GTZAN_GENRES:
            sal = saliency_data[M][genre]
            results["per_M_per_genre_saliency"][str(M)][genre] = {
                "saliency_vector"   : [round(float(v), 6) for v in sal],
                "filter_centres_hz" : [round(float(c), 2) for c in centres],
                "peak_band_hz"      : round(float(centres[sal.argmax()]), 2),
                "group"             : centroid_data.get(genre, {}).get("group"),
                "sigma_hz"          : centroid_data.get(genre, {}).get("sigma_hz")
            }

    local_json = os.path.join(OUTPUT_DIR, "exp5_saliency_results.json")
    with open(local_json, "w") as f:
        json.dump(results, f, indent=2)
    save_to_drive(local_json, "exp5_saliency_results.json")
    save_to_drive(fig_path, "fig_05_02_saliency_profiles.png")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Program 05 — Experiment 5: Frequency-Axis Saliency Probe  (v1.0)")
    print(f"  Models: M={M_VALUES}")
    print(f"  Full GTZAN corpus (all clips, no train/val split)")
    print("=" * 70)

    genre_idx_map = {g: i for i, g in enumerate(sorted(GTZAN_GENRES))}

    genre_root                  = step1_prepare_dataset()
    centroid_data, exp1_results = step2_download_inputs()
    spectrogram_data            = step3_build_spectrograms(genre_root)
    saliency_data               = step4_compute_saliency(
                                      spectrogram_data, genre_idx_map)
    js_results                  = step5_js_divergence(
                                      saliency_data, centroid_data)
    spearman_results            = step6_spearman_correlation(
                                      saliency_data, exp1_results,
                                      centroid_data)
    fig_path                    = step7_make_figure2(
                                      saliency_data, centroid_data)
    step8_save_results(saliency_data, js_results,
                       spearman_results, centroid_data, fig_path)

    # ── Sync to ensure all writes are flushed ────────────────────────────────
    print("\n[SYNC] Flushing file system buffers...")
    os.sync()
    print("[SYNC] Complete.")

    print("\nProgram 05 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()