# =============================================================================
# Program      : 00_precompute_filterbank_fairness.py
# Version      : 2.0
# Description  : Prerequisite Computation — Filterbank Fairness Pre-Analysis
#
#                Before any model training, this program computes three
#                analytical quantities that validate whether the experimental
#                programme is worth running:
#
#                  1. Spectral centre-of-mass (f_bar_g) per GTZAN genre:
#                       f_bar_g = sum(f_n * S_g(n)) / sum(S_g(n))
#                     where S_g(n) is the mean linear magnitude spectrum
#                     averaged over all clips in genre g.
#
#                  2. Spectral spread (sigma_g) per genre:
#                       sigma_g = sqrt(sum(f_n^2 * S_g(n))/sum(S_g(n))
#                                      - f_bar_g^2)
#
#                  3. Effective filter density D_g(theta) per genre per
#                     filterbank configuration theta = (M, f_max, norm):
#                       D_g(theta) = |{k : |f_k - f_bar_g| <= sigma_g}|
#                     i.e. the number of mel filter centres that fall
#                     within the effective spectral support
#                     [f_bar_g - sigma_g, f_bar_g + sigma_g].
#
#                Genre grouping uses a DATA-DRIVEN MEDIAN SPLIT on sigma_g:
#                  high_spread : sigma_g >= median(sigma_g across all genres)
#                  low_spread  : sigma_g <  median(sigma_g across all genres)
#
#                This replaces the fixed centroid thresholds used in v1.0,
#                which produced an empty low-centroid group because all
#                GTZAN genres have f_bar_g > 1500 Hz. The fairness variable
#                is now spectral spread (coverage) rather than centroid
#                position. A filterbank with low M or low f_max clips the
#                tails of high-spread genres disproportionately.
#
#                A large Delta_D = mean(D_g | high_spread)
#                                - mean(D_g | low_spread)
#                across configurations confirms the fairness hypothesis is
#                empirically grounded before GPU compute is spent.
#
# INPUT        :
#                  Google Drive:
#                    /content/drive/MyDrive/datasets/GTZAN.zip
#                  Expected internal structure after extraction:
#                    Data/genres_original/<genre>/*.wav
#                  (standard GTZAN layout)
#
# STEPS        :
#                  Step 1  Mount Drive, copy GTZAN.zip to local disk, unzip
#                  Step 2  For each genre, load all WAV clips and compute
#                          mean linear magnitude spectrum S_g(n)
#                  Step 3  Compute f_bar_g and sigma_g from S_g(n)
#                  Step 4  Assign spread groups via median split on sigma_g
#                  Step 5  For each filterbank config (M, f_max, norm),
#                          compute mel filter centre frequencies f_k
#                          and D_g(theta) for every genre
#                  Step 6  Compute Delta_D between high-spread and
#                          low-spread genre groups per configuration
#                  Step 7  Save results JSON and figures, upload to Drive
#
# OUTPUT FILES :
#                  precompute_genre_centroids.json
#                      Per-genre: f_bar_g (Hz), sigma_g (Hz),
#                      group label (low_spread / high_spread),
#                      corpus median_sigma_hz
#
#                  precompute_filter_density.json
#                      Per (genre, M, f_max, norm): D_g(theta),
#                      filter centre frequencies f_k,
#                      Delta_D per configuration
#
#                  fig_00_01_genre_spread.png
#                      Bar chart of sigma_g per genre sorted ascending,
#                      colour-coded by spread group (low/high),
#                      horizontal line at median sigma_g,
#                      f_bar_g annotated on each bar
#
#                  fig_00_02_filter_density_heatmap.png
#                      Heatmap: genres (y, sorted by sigma_g ascending) x
#                      configurations (x, labelled M/f_max),
#                      cell value = D_g(theta),
#                      Delta_D (high_spread minus low_spread) annotated
#                      per configuration column
#
# GPU Required : NO
# Dependencies : numpy, scipy, matplotlib, librosa, tqdm
#
# Change Log   :
#   v1.0  2026-06-06  Initial version (fixed centroid thresholds)
#   v2.0  2026-06-06  Replace fixed thresholds with median-split on sigma_g;
#                     update figures and Delta_D accordingly
# =============================================================================

import subprocess
import sys

for pkg in ["numpy", "scipy", "matplotlib", "librosa", "tqdm"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

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
import matplotlib.ticker as mticker
from tqdm import tqdm
import librosa

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

# ── Drive / local paths ───────────────────────────────────────────────────────
PROJECT_DIR      = "/content/drive/MyDrive/paper/mel-fairness-probe/"  # Persistent storage
DRIVE_DIR        = "/content/drive/MyDrive/datasets"
GTZAN_ZIP        = os.path.join(DRIVE_DIR, "GTZAN.zip")
LOCAL_WORK_DIR   = "/tmp/gtzan_fairness"
GTZAN_LOCAL_ZIP  = os.path.join(LOCAL_WORK_DIR, "GTZAN.zip")
GTZAN_EXTRACT    = os.path.join(LOCAL_WORK_DIR, "gtzan")
OUTPUT_DIR       = os.path.join(LOCAL_WORK_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GTZAN_EXTRACT, exist_ok=True)

# ── Audio ─────────────────────────────────────────────────────────────────────
SR         = 22050
N_FFT      = 2048
HOP_LENGTH = 512

# ── Filterbank configurations to probe ───────────────────────────────────────
M_VALUES     = [40, 64, 80, 128]
FMAX_VALUES  = [4000, 8000, 11025]
FMIN         = 0.0
NORM_SCHEMES = ["slaney", "htk"]

# ── GTZAN genres ─────────────────────────────────────────────────────────────
GTZAN_GENRES = [
    "blues", "classical", "country", "disco",
    "hiphop", "jazz", "metal", "pop", "reggae", "rock"
]

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
        print(f"  [DRIVE OK] Uploaded: {remote_filename}")
    except Exception as e:
        print(f"  [DRIVE FAIL] Upload failed for {remote_filename}: {e}")

def load_from_drive(remote_filename, local_filepath):
    """Copy a file from Google Drive project folder to local path."""
    ensure_project_dir()
    src_path = os.path.join(PROJECT_DIR, remote_filename)
    if os.path.exists(src_path):
        try:
            shutil.copy2(src_path, local_filepath)
            return True
        except Exception as e:
            print(f"  [DRIVE FAIL] Download failed for {remote_filename}: {e}")
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
# CHECKPOINT HELPERS
# =============================================================================

CENTROID_JSON   = os.path.join(OUTPUT_DIR, "precompute_genre_centroids.json")
DENSITY_JSON    = os.path.join(OUTPUT_DIR, "precompute_filter_density.json")
CENTROID_REMOTE = "precompute_genre_centroids.json"
DENSITY_REMOTE  = "precompute_filter_density.json"

def load_centroid_checkpoint():
    """Try to resume centroid computation from Drive, then local."""
    if os.path.exists(CENTROID_JSON):
        print("  [CHECKPOINT] Resuming genre centroids from local disk.")
        with open(CENTROID_JSON) as f:
            return json.load(f)
    if load_from_drive(CENTROID_REMOTE, CENTROID_JSON):
        print("  [CHECKPOINT] Resumed genre centroids from Drive.")
        with open(CENTROID_JSON) as f:
            return json.load(f)
    return {}

def save_centroid_checkpoint(data):
    with open(CENTROID_JSON, "w") as f:
        json.dump(data, f, indent=2)
    save_to_drive(CENTROID_JSON, CENTROID_REMOTE)

# =============================================================================
# STEP 1 — MOUNT DRIVE, COPY AND EXTRACT GTZAN
# =============================================================================

def step1_prepare_dataset():
    print("\n=== STEP 1: Prepare GTZAN dataset ===")

    from google.colab import drive
    drive.mount("/content/drive")

    # Check if already extracted
    for candidate in ["genres_original", "genres",
                       "Data/genres_original", "data/genres_original"]:
        p = os.path.join(GTZAN_EXTRACT, candidate)
        if os.path.isdir(p):
            print(f"  [SKIP] Already extracted at {p}")
            return p

    # Copy zip to local
    print(f"  Copying {GTZAN_ZIP} -> {GTZAN_LOCAL_ZIP} ...")
    os.makedirs(LOCAL_WORK_DIR, exist_ok=True)
    shutil.copy2(GTZAN_ZIP, GTZAN_LOCAL_ZIP)
    print("  Copy complete.")

    # Extract
    print(f"  Extracting to {GTZAN_EXTRACT} ...")
    with zipfile.ZipFile(GTZAN_LOCAL_ZIP, "r") as zf:
        zf.extractall(GTZAN_EXTRACT)
    print("  Extraction complete.")

    # Locate genre root
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
# STEP 2 — COMPUTE MEAN LINEAR MAGNITUDE SPECTRUM PER GENRE
# =============================================================================

def compute_mean_spectrum(genre_dir):
    wav_files = sorted([f for f in os.listdir(genre_dir) if f.endswith(".wav")])
    if not wav_files:
        return None

    accum = np.zeros(N_FFT // 2 + 1, dtype=np.float64)
    count = 0

    for wav_name in wav_files:
        wav_path = os.path.join(genre_dir, wav_name)
        try:
            y, _ = librosa.load(wav_path, sr=SR, mono=True, duration=29.0)
            S     = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
            accum += S.mean(axis=1)
            count += 1
        except Exception as e:
            print(f"    [WARN] Skipping {wav_name}: {e}")

    return accum / count if count > 0 else None

# =============================================================================
# STEP 3 — COMPUTE CENTROID AND SPREAD PER GENRE
# =============================================================================

def compute_centroid_and_spread(mean_spectrum):
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    S     = mean_spectrum.astype(np.float64)
    S_sum = S.sum()
    if S_sum == 0:
        return 0.0, 0.0
    f_bar = np.sum(freqs * S) / S_sum
    sigma = np.sqrt(np.sum((freqs ** 2) * S) / S_sum - f_bar ** 2)
    return float(f_bar), float(sigma)

# =============================================================================
# STEP 4 — ASSIGN SPREAD GROUPS VIA MEDIAN SPLIT
# =============================================================================

def assign_spread_groups(centroid_data):
    """
    Partition genres into low_spread / high_spread using the corpus median
    of sigma_g. This is data-driven and avoids arbitrary fixed thresholds.
    Returns updated centroid_data and the median sigma value used.
    """
    sigmas       = [v["sigma_hz"] for v in centroid_data.values()]
    median_sigma = float(np.median(sigmas))

    for genre, v in centroid_data.items():
        v["group"] = "high_spread" if v["sigma_hz"] >= median_sigma else "low_spread"

    return centroid_data, median_sigma

# =============================================================================
# STEP 5 — COMPUTE MEL FILTER CENTRES AND D_g(theta)
# =============================================================================

def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)

def mel_filter_centres(M, fmin, fmax):
    """
    Return M interior mel filter centre frequencies in Hz,
    uniformly spaced on the mel scale between fmin and fmax.
    """
    m_min = hz_to_mel(fmin if fmin > 0 else 1.0)
    m_max = hz_to_mel(fmax)
    m_pts = np.linspace(m_min, m_max, M + 2)
    return mel_to_hz(m_pts)[1:-1]

def effective_filter_density(f_centres, f_bar, sigma):
    """
    D_g(theta) = number of filter centres within the spectral support
    [f_bar - sigma, f_bar + sigma] of genre g.
    """
    return int(np.sum(np.abs(f_centres - f_bar) <= sigma))

# =============================================================================
# STEP 6 — FIGURES
# =============================================================================

def make_fig_spread(centroid_data, median_sigma):
    """
    Fig 00_01: Bar chart of sigma_g per genre, sorted ascending.
    Colour-coded by spread group. Horizontal line at median_sigma.
    f_bar_g annotated inside each bar.
    """
    genres  = sorted(centroid_data.keys(), key=lambda g: centroid_data[g]["sigma_hz"])
    sigmas  = [centroid_data[g]["sigma_hz"] for g in genres]
    f_bars  = [centroid_data[g]["f_bar_hz"] for g in genres]
    groups  = [centroid_data[g]["group"]    for g in genres]

    color_map = {"low_spread": "#1B9E77", "high_spread": "#D95F02"}
    colors    = [color_map[grp] for grp in groups]

    fig, ax = plt.subplots(figsize=(10, 5))
    x    = np.arange(len(genres))
    bars = ax.bar(x, sigmas, color=colors, edgecolor="white",
                  linewidth=0.8, alpha=0.88, width=0.6)

    ax.axhline(median_sigma, color="black", linestyle="--", linewidth=1.2,
               label=f"Corpus median $\\sigma_g$ = {median_sigma:.0f} Hz")

    # Annotate f_bar inside bars
    for xi, (fb, sg) in enumerate(zip(f_bars, sigmas)):
        ax.text(xi, sg * 0.05, f"$\\bar{{f}}$={fb:.0f}", ha="center",
                va="bottom", fontsize=7.5, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([g.capitalize() for g in genres],
                       rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Spectral Spread $\\sigma_g$ (Hz)", fontsize=11)
    ax.set_title(
        "Per-Genre Spectral Spread $\\sigma_g$ (GTZAN) — Median-Split Grouping\n"
        "Green = low-spread  |  Orange = high-spread  |  $\\bar{f}_g$ annotated in bars",
        fontsize=10
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)

    # Group label top of bar
    for xi, grp in enumerate(groups):
        label = "H" if grp == "high_spread" else "L"
        ax.text(xi, sigmas[xi] + 40, label, ha="center", va="bottom",
                fontsize=8, color=color_map[grp], fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig_00_01_genre_spread.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


def make_fig_density_heatmap(centroid_data, density_data, median_sigma):
    """
    Fig 00_02: Heatmap of D_g(theta).
    Rows = genres sorted by sigma_g ascending.
    Columns = (M, f_max) configurations (slaney norm only for primary heatmap).
    Delta_D (high_spread minus low_spread) annotated per column.
    """
    genres_sorted = sorted(centroid_data.keys(),
                           key=lambda g: centroid_data[g]["sigma_hz"])

    configs = [(M, fmax, "slaney") for M in M_VALUES for fmax in FMAX_VALUES]

    n_genres  = len(genres_sorted)
    n_configs = len(configs)
    matrix    = np.zeros((n_genres, n_configs), dtype=float)

    for ci, (M, fmax, norm) in enumerate(configs):
        cfg_key = f"M{M}_fmax{fmax}_{norm}"
        for ri, genre in enumerate(genres_sorted):
            matrix[ri, ci] = (density_data
                              .get(genre, {})
                              .get(cfg_key, {})
                              .get("D_g", 0))

    # Delta_D per config
    delta_d_vals = []
    for ci, (M, fmax, norm) in enumerate(configs):
        cfg_key   = f"M{M}_fmax{fmax}_{norm}"
        low_vals  = [density_data.get(g, {}).get(cfg_key, {}).get("D_g", 0)
                     for g in genres_sorted
                     if centroid_data[g]["group"] == "low_spread"]
        high_vals = [density_data.get(g, {}).get(cfg_key, {}).get("D_g", 0)
                     for g in genres_sorted
                     if centroid_data[g]["group"] == "high_spread"]
        delta_d_vals.append(
            (np.mean(high_vals) if high_vals else 0) -
            (np.mean(low_vals)  if low_vals  else 0)
        )

    col_labels = [f"M={M}\n$f_{{max}}$={fmax//1000}k" for (M, fmax, _) in configs]
    row_labels  = [
        f"{g.capitalize()}\n$\\sigma$={centroid_data[g]['sigma_hz']:.0f} Hz"
        for g in genres_sorted
    ]

    fig, ax = plt.subplots(figsize=(max(12, n_configs * 0.9), 5.5))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_xticks(np.arange(n_configs))
    ax.set_xticklabels(col_labels, fontsize=7.5)
    ax.set_yticks(np.arange(n_genres))
    ax.set_yticklabels(row_labels, fontsize=8.5)

    # Annotate cells
    vmax = matrix.max()
    for ri in range(n_genres):
        for ci in range(n_configs):
            val = matrix[ri, ci]
            ax.text(ci, ri, f"{int(val)}", ha="center", va="center",
                    fontsize=7,
                    color="white" if val > vmax * 0.7 else "black")

    # Delta_D secondary x-axis below heatmap
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(np.arange(n_configs))
    ax2.set_xticklabels([f"$\\Delta D$={d:.1f}" for d in delta_d_vals],
                        fontsize=7, color="#D95F02")
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.spines["bottom"].set_position(("outward", 38))

    plt.colorbar(im, ax=ax, label="$D_g(\\theta)$ — Effective Filter Density")
    ax.set_title(
        "Effective Filter Density $D_g(\\theta)$ per Genre per Filterbank Configuration\n"
        "(Slaney norm; genres sorted by $\\sigma_g$ ascending; "
        "$\\Delta D$ = high-spread minus low-spread)",
        fontsize=10, pad=12
    )

    # Shade group boundaries
    low_rows  = [ri for ri, g in enumerate(genres_sorted)
                 if centroid_data[g]["group"] == "low_spread"]
    high_rows = [ri for ri, g in enumerate(genres_sorted)
                 if centroid_data[g]["group"] == "high_spread"]
    if low_rows:
        ax.axhspan(min(low_rows) - 0.5, max(low_rows) + 0.5,
                   alpha=0.08, color="#1B9E77", zorder=0)
    if high_rows:
        ax.axhspan(min(high_rows) - 0.5, max(high_rows) + 0.5,
                   alpha=0.08, color="#D95F02", zorder=0)

    # Group boundary line
    if low_rows and high_rows:
        boundary = (max(low_rows) + min(high_rows)) / 2.0
        ax.axhline(boundary, color="black", linewidth=1.2,
                   linestyle="-", alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig_00_02_filter_density_heatmap.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Program 00 — Filterbank Fairness Pre-Computation  (v2.0)")
    print("=" * 70)

    # ── STEP 1: Prepare dataset ───────────────────────────────────────────────
    genre_root = step1_prepare_dataset()

    # ── STEP 2-3: Compute centroids per genre (with checkpoint) ──────────────
    print("\n=== STEP 2-3: Compute genre spectral centroids ===")
    centroid_data = load_centroid_checkpoint()

    # Strip any old group assignments so Step 4 recomputes cleanly
    for v in centroid_data.values():
        v.pop("group", None)
        v.pop("median_sigma_hz", None)

    genres_to_process = [g for g in GTZAN_GENRES if g not in centroid_data]
    if not genres_to_process:
        print("  [SKIP] All genre centroids already computed (checkpoint found).")
    else:
        for genre in tqdm(genres_to_process, desc="  Genres"):
            genre_dir = None
            for candidate in [genre, genre.capitalize()]:
                p = os.path.join(genre_root, candidate)
                if os.path.isdir(p):
                    genre_dir = p
                    break

            if genre_dir is None:
                print(f"  [WARN] Directory not found for genre '{genre}', skipping.")
                continue

            S_mean = compute_mean_spectrum(genre_dir)
            if S_mean is None:
                print(f"  [WARN] No valid WAV files for genre '{genre}', skipping.")
                continue

            f_bar, sigma = compute_centroid_and_spread(S_mean)
            centroid_data[genre] = {
                "f_bar_hz": round(f_bar, 2),
                "sigma_hz": round(sigma, 2)
            }
            print(f"    {genre:12s}  f_bar={f_bar:.1f} Hz  sigma={sigma:.1f} Hz")
            # Save raw centroid checkpoint (no group yet — median not known)
            save_centroid_checkpoint(centroid_data)

    # ── STEP 4: Assign spread groups via median split ─────────────────────────
    print("\n=== STEP 4: Assign spread groups (median split on sigma_g) ===")
    centroid_data, median_sigma = assign_spread_groups(centroid_data)

    print(f"  Corpus median sigma_g = {median_sigma:.1f} Hz")
    print(f"  {'Genre':<12}  {'sigma_g':>8}  {'f_bar_g':>8}  {'Group'}")
    print("  " + "-" * 48)
    for g in sorted(centroid_data.keys(),
                    key=lambda x: centroid_data[x]["sigma_hz"]):
        v = centroid_data[g]
        print(f"  {g:<12}  {v['sigma_hz']:>8.1f}  "
              f"{v['f_bar_hz']:>8.1f}  {v['group']}")

    # Add median to each record and save final centroid JSON
    for v in centroid_data.values():
        v["median_sigma_hz"] = round(median_sigma, 2)
    save_centroid_checkpoint(centroid_data)

    # ── STEP 5: Compute D_g(theta) across all configurations ─────────────────
    print("\n=== STEP 5: Compute effective filter density D_g(theta) ===")
    density_data  = {}
    total_configs = len(M_VALUES) * len(FMAX_VALUES) * len(NORM_SCHEMES)
    print(f"  Configurations to probe: {total_configs}")

    for genre, cdata in centroid_data.items():
        f_bar = cdata["f_bar_hz"]
        sigma = cdata["sigma_hz"]
        density_data.setdefault(genre, {})

        for M in M_VALUES:
            for fmax in FMAX_VALUES:
                for norm in NORM_SCHEMES:
                    cfg_key   = f"M{M}_fmax{fmax}_{norm}"
                    f_centres = mel_filter_centres(M, FMIN, fmax)
                    D_g       = effective_filter_density(f_centres, f_bar, sigma)
                    density_data[genre][cfg_key] = {
                        "M"             : M,
                        "f_max_hz"      : fmax,
                        "norm"          : norm,
                        "D_g"           : D_g,
                        "f_bar_hz"      : f_bar,
                        "sigma_hz"      : sigma,
                        "filter_centres": [round(float(fc), 2) for fc in f_centres]
                    }

    # ── STEP 6: Compute Delta_D and print summary ─────────────────────────────
    print("\n=== STEP 6: Delta_D summary (high_spread minus low_spread) ===")
    print(f"  {'Config':<25}  {'Mean D low_spread':>17}  "
          f"{'Mean D high_spread':>18}  {'Delta_D':>8}")
    print("  " + "-" * 75)

    delta_summary = {}
    for M in M_VALUES:
        for fmax in FMAX_VALUES:
            for norm in NORM_SCHEMES:
                cfg_key   = f"M{M}_fmax{fmax}_{norm}"
                low_vals  = [density_data[g][cfg_key]["D_g"]
                             for g in centroid_data
                             if centroid_data[g]["group"] == "low_spread"
                             and cfg_key in density_data.get(g, {})]
                high_vals = [density_data[g][cfg_key]["D_g"]
                             for g in centroid_data
                             if centroid_data[g]["group"] == "high_spread"
                             and cfg_key in density_data.get(g, {})]
                mean_low  = float(np.mean(low_vals))  if low_vals  else 0.0
                mean_high = float(np.mean(high_vals)) if high_vals else 0.0
                delta_d   = mean_high - mean_low
                delta_summary[cfg_key] = {
                    "mean_D_low_spread" : round(mean_low,  3),
                    "mean_D_high_spread": round(mean_high, 3),
                    "delta_D"           : round(delta_d,   3)
                }

    # Print sorted by delta_D descending
    for cfg_key, vals in sorted(delta_summary.items(),
                                key=lambda x: x[1]["delta_D"], reverse=True):
        print(f"  {cfg_key:<25}  {vals['mean_D_low_spread']:>17.3f}  "
              f"{vals['mean_D_high_spread']:>18.3f}  {vals['delta_D']:>8.3f}")

    # ── STEP 7: Save JSONs and upload ─────────────────────────────────────────
    print("\n=== STEP 7: Save and upload outputs ===")

    with open(CENTROID_JSON, "w") as f:
        json.dump(centroid_data, f, indent=2)
    save_to_drive(CENTROID_JSON, CENTROID_REMOTE)

    density_output = {
        "configurations": {
            "M_values"    : M_VALUES,
            "fmax_values" : FMAX_VALUES,
            "fmin_hz"     : FMIN,
            "norm_schemes": NORM_SCHEMES
        },
        "median_sigma_hz": round(median_sigma, 2),
        "delta_summary"  : delta_summary,
        "per_genre"      : density_data
    }
    with open(DENSITY_JSON, "w") as f:
        json.dump(density_output, f, indent=2)
    save_to_drive(DENSITY_JSON, DENSITY_REMOTE)

    fig1_path = make_fig_spread(centroid_data, median_sigma)
    save_to_drive(fig1_path, "fig_00_01_genre_spread.png")

    fig2_path = make_fig_density_heatmap(centroid_data, density_data, median_sigma)
    save_to_drive(fig2_path, "fig_00_02_filter_density_heatmap.png")

    # ── Final interpretation ──────────────────────────────────────────────────
    print("\n=== INTERPRETATION ===")
    max_cfg = max(delta_summary.items(), key=lambda x: x[1]["delta_D"])
    min_cfg = min(delta_summary.items(), key=lambda x: x[1]["delta_D"])
    print(f"  Largest  Delta_D : {max_cfg[0]}  ->  {max_cfg[1]['delta_D']:.3f} filters")
    print(f"  Smallest Delta_D : {min_cfg[0]}  ->  {min_cfg[1]['delta_D']:.3f} filters")
    print()
    print("  If Delta_D >= 3 across the majority of configurations,")
    print("  the filter density differential is substantial enough to")
    print("  justify the full experimental programme (Experiments 1-5).")
    print()
    print("  High-spread genres (above median sigma_g):")
    for g, v in centroid_data.items():
        if v["group"] == "high_spread":
            print(f"    {g:<12}  sigma={v['sigma_hz']:.1f} Hz")
    print()
    print("  Low-spread genres (below median sigma_g):")
    for g, v in centroid_data.items():
        if v["group"] == "low_spread":
            print(f"    {g:<12}  sigma={v['sigma_hz']:.1f} Hz")
    print()
    print("  Use max-Delta_D and min-Delta_D configs as anchor")
    print("  configurations for Experiment 4 (FMA-Small replication).")
    print()
    print("Program 00 complete.")
    print("=" * 70)

    # ── Sync to ensure all writes are flushed ────────────────────────────────
    print("\n[SYNC] Flushing file system buffers...")
    os.sync()
    print("[SYNC] Complete.")


if __name__ == "__main__":
    main()