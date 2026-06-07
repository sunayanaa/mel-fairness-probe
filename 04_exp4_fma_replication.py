# =============================================================================
# Program      : 04_exp4_fma_replication.py
# Version      : 1.0
# Description  : Experiment 4 — Cross-Dataset Replication on FMA-Small
#
#                Replicates the fairness disparity finding on FMA-Small
#                (8,000 clips, 8 genres, MP3, 44.1 kHz, 30 s) using the
#                two anchor configurations identified in Program 00:
#
#                  Max Delta_D anchor : M=128, f_max=8000 Hz, Slaney norm
#                  Min Delta_D anchor : M=40,  f_max=4000 Hz, Slaney norm
#
#                Genre group assignments (low_spread / high_spread) are
#                recomputed from FMA-Small's own corpus statistics via a
#                fresh median split on sigma_g. GTZAN's median sigma
#                (2485 Hz) is NOT carried over.
#
#                FMA-Small metadata is downloaded from:
#                  https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
#                tracks.csv (multi-level pandas header) is parsed to
#                extract track IDs and genre_top labels for the 'small'
#                subset. Audio files are read from:
#                  /content/drive/MyDrive/datasets/FMA-small.zip
#
#                CNN architecture, optimizer, loss, epochs, and repeated
#                CV protocol (3 repeats x 5 folds, seeds [42,7,123]) are
#                identical to Experiments 1-3.
#
#                Two replication checks are reported:
#                  1. FFD direction: does high-spread group underperform
#                     low-spread group on FMA-Small as on GTZAN?
#                  2. Pearson correlation of per-genre Delta_A between
#                     GTZAN (Exp 1, M=128 vs M=40) and FMA-Small results.
#
# INPUT        :
#                  Google Drive:
#                    /content/drive/MyDrive/datasets/FMA-small.zip
#                  Downloaded at runtime:
#                    https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
#                  Drive:
#                    precompute_genre_centroids.json  (GTZAN, for reference)
#                    exp1_results.json               (GTZAN Exp1, for correlation)
#
# STEPS        :
#                  Step 1  Mount Drive, copy and extract FMA-small.zip
#                  Step 2  Download and parse fma_metadata.zip -> tracks.csv
#                  Step 3  Compute FMA-Small genre centroids (f_bar_g, sigma_g)
#                            and assign spread groups via median split
#                  Step 4  Build mel-spectrograms for both anchor configs
#                  Step 5  3-repeat x 5-fold CV training per anchor config
#                  Step 6  Aggregate per-genre accuracy across 15 runs
#                  Step 7  Compute FFD per anchor config
#                  Step 8  Replication checks (FFD direction + correlation)
#                  Step 9  Save results JSON, upload to Drive
#
# OUTPUT FILES :
#                  exp4_fma_centroids.json
#                      Per-genre: f_bar_g, sigma_g, group (FMA-Small corpus)
#
#                  exp4_results.json
#                      Per (config, genre): mean acc, std across 15 runs
#                      Per config: overall acc, FFD
#                      Replication checks: FFD direction, Pearson correlation
#
#                  exp4_results_partial.json
#                      Incremental save after each completed config
#
#                  exp4_checkpoint_{config}_repeat{r}_fold{f}_epoch{e}.pth
#                      Periodic checkpoints (every 5 epochs)
#
# GPU Required : YES
# Dependencies : torch, torchaudio, librosa, numpy, pandas, scikit-learn,
#                scipy, matplotlib, tqdm, requests
#
# Change Log   :
#   v1.0  2026-06-07  Initial version
# =============================================================================

import subprocess
import sys

for pkg in ["torch", "torchaudio", "librosa", "numpy", "pandas",
            "scikit-learn", "scipy", "tqdm", "requests"]:
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
import pandas as pd
import requests
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR      = "/content/drive/MyDrive/paper/mel-fairness-probe/"  # Persistent storage
DRIVE_DIR        = "/content/drive/MyDrive/datasets"
FMA_ZIP          = os.path.join(DRIVE_DIR, "FMA-small.zip")
LOCAL_WORK_DIR   = "/tmp/gtzan_fairness"
FMA_LOCAL_ZIP    = os.path.join(LOCAL_WORK_DIR, "FMA-small.zip")
FMA_EXTRACT      = os.path.join(LOCAL_WORK_DIR, "fma_small")
META_URL         = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
META_LOCAL_ZIP   = os.path.join(LOCAL_WORK_DIR, "fma_metadata.zip")
META_EXTRACT     = os.path.join(LOCAL_WORK_DIR, "fma_metadata")
OUTPUT_DIR       = os.path.join(LOCAL_WORK_DIR, "outputs_exp4")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Audio ─────────────────────────────────────────────────────────────────────
SR         = 22050          # resample from 44.1 kHz
HOP_LENGTH = 512
N_FFT      = 2048
FMIN       = 0.0
DURATION   = 29.0
NORM       = "slaney"       # fixed for both anchor configs

# ── Anchor configurations (from Program 00 Delta_D analysis) ─────────────────
ANCHOR_CONFIGS = {
    "M128_fmax8000": {"M": 128, "fmax": 8000},   # max Delta_D
    "M40_fmax4000" : {"M": 40,  "fmax": 4000},   # min Delta_D
}

# ── Experiment ────────────────────────────────────────────────────────────────
N_FOLDS    = 5
N_REPEATS  = 3
SEEDS      = [42, 7, 123]
N_EPOCHS   = 50
CKPT_EVERY = 5
BATCH_SIZE = 16
LR         = 1e-3
DEVICE     = torch.device("cuda")

# FMA-Small genres (8 top-level genres in the small subset)
FMA_GENRES = [
    "Electronic", "Experimental", "Folk", "Hip-Hop",
    "Instrumental", "International", "Pop", "Rock"
]
N_CLASSES = len(FMA_GENRES)

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
# STEP 1 — PREPARE FMA-SMALL DATASET
# =============================================================================

def step1_prepare_dataset():
    print("\n=== STEP 1: Prepare FMA-Small dataset ===")
    from google.colab import drive
    drive.mount("/content/drive")

    # Check if already extracted
    if os.path.isdir(FMA_EXTRACT):
        # Verify it has audio subdirs
        subdirs = [d for d in os.listdir(FMA_EXTRACT)
                   if os.path.isdir(os.path.join(FMA_EXTRACT, d))
                   and d.isdigit()]
        if subdirs:
            print(f"  [SKIP] Already extracted at {FMA_EXTRACT} "
                  f"({len(subdirs)} audio subdirs found).")
            return FMA_EXTRACT

    print(f"  Copying {FMA_ZIP} -> {FMA_LOCAL_ZIP} ...")
    os.makedirs(LOCAL_WORK_DIR, exist_ok=True)
    shutil.copy2(FMA_ZIP, FMA_LOCAL_ZIP)
    print("  Copy complete.")

    print(f"  Extracting to {FMA_EXTRACT} ...")
    os.makedirs(FMA_EXTRACT, exist_ok=True)
    with zipfile.ZipFile(FMA_LOCAL_ZIP, "r") as zf:
        zf.extractall(FMA_EXTRACT)
    print("  Extraction complete.")
    return FMA_EXTRACT

# =============================================================================
# STEP 2 — DOWNLOAD AND PARSE FMA METADATA
# =============================================================================

def step2_load_metadata():
    """
    Download fma_metadata.zip, extract tracks.csv, parse multi-level header,
    filter to subset='small', return dict: track_id (int) -> genre_top (str).
    """
    print("\n=== STEP 2: Load FMA metadata ===")

    tracks_csv = os.path.join(META_EXTRACT, "fma_metadata", "tracks.csv")

    if not os.path.exists(tracks_csv):
        if not os.path.exists(META_LOCAL_ZIP):
            print(f"  Downloading metadata from {META_URL} ...")
            r = requests.get(META_URL, stream=True, timeout=120)
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(META_LOCAL_ZIP, "wb") as f:
                downloaded = 0
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r  {pct:5.1f}%", end="", flush=True)
            print(f"\n  Download complete: {META_LOCAL_ZIP}")

        print(f"  Extracting metadata ...")
        os.makedirs(META_EXTRACT, exist_ok=True)
        with zipfile.ZipFile(META_LOCAL_ZIP, "r") as zf:
            zf.extractall(META_EXTRACT)
        print("  Extraction complete.")
    else:
        print(f"  [SKIP] tracks.csv already extracted.")

    # Parse multi-level header (rows 0 and 1 are header rows in tracks.csv)
    print("  Parsing tracks.csv ...")
    tracks = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])

    # Filter to small subset
    subset_col = ("set", "subset")
    genre_col  = ("track", "genre_top")
    mask       = tracks[subset_col] == "small"
    small      = tracks[mask][[genre_col]].copy()
    small.columns = ["genre_top"]
    small        = small.dropna(subset=["genre_top"])

    # Verify genres match expected FMA_GENRES
    found_genres = sorted(small["genre_top"].unique().tolist())
    print(f"  Found {len(small)} tracks in small subset.")
    print(f"  Genres: {found_genres}")

    # Build track_id -> genre dict
    genre_map = small["genre_top"].to_dict()   # {track_id (int): genre_str}
    return genre_map

# =============================================================================
# STEP 3 — COMPUTE FMA-SMALL GENRE CENTROIDS AND SPREAD GROUPS
# =============================================================================

def track_id_to_path(track_id, fma_root):
    """
    FMA audio files are stored as:
      fma_root/<tid_padded[:3]>/<tid_padded>.mp3
    where tid_padded is zero-padded to 6 digits.
    """
    tid_str = f"{track_id:06d}"
    audio_root = os.path.join(fma_root, "fma_small", "fma_small")
    return os.path.join(audio_root, tid_str[:3], f"{tid_str}.mp3")


def compute_mean_spectrum_mp3(track_id, fma_root):
    path = track_id_to_path(track_id, fma_root)
    if not os.path.exists(path):
        return None
    try:
        y, _ = librosa.load(path, sr=SR, mono=True, duration=DURATION)
        S    = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
        return S.mean(axis=1)
    except Exception:
        return None


def compute_centroid_and_spread(mean_spectrum):
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    S     = mean_spectrum.astype(np.float64)
    S_sum = S.sum()
    if S_sum == 0:
        return 0.0, 0.0
    f_bar = np.sum(freqs * S) / S_sum
    sigma = np.sqrt(np.sum((freqs ** 2) * S) / S_sum - f_bar ** 2)
    return float(f_bar), float(sigma)


def step3_compute_fma_centroids(genre_map, fma_root):
    print("\n=== STEP 3: Compute FMA-Small genre centroids ===")

    centroid_path = os.path.join(OUTPUT_DIR, "exp4_fma_centroids.json")
    if os.path.exists(centroid_path):
        print("  [SKIP] Centroids already computed (local cache).")
        with open(centroid_path) as f:
            return json.load(f)

    # Group track IDs by genre
    genre_tracks = {g: [] for g in FMA_GENRES}
    for tid, genre in genre_map.items():
        if genre in genre_tracks:
            genre_tracks[genre].append(tid)

    centroid_data = {}
    for genre in tqdm(FMA_GENRES, desc="  Genres"):
        tids    = genre_tracks[genre]
        accum   = np.zeros(N_FFT // 2 + 1, dtype=np.float64)
        count   = 0
        for tid in tids:
            S = compute_mean_spectrum_mp3(tid, fma_root)
            if S is not None:
                accum += S
                count += 1
        if count == 0:
            print(f"  [WARN] No valid files for genre {genre}")
            continue
        mean_spec        = accum / count
        f_bar, sigma     = compute_centroid_and_spread(mean_spec)
        centroid_data[genre] = {
            "f_bar_hz": round(f_bar, 2),
            "sigma_hz": round(sigma, 2),
            "n_tracks": count
        }
        print(f"    {genre:<15s}  f_bar={f_bar:.1f} Hz  "
              f"sigma={sigma:.1f} Hz  n={count}")

    # Median split on sigma_g
    sigmas       = [v["sigma_hz"] for v in centroid_data.values()]
    median_sigma = float(np.median(sigmas))
    print(f"\n  FMA-Small corpus median sigma_g = {median_sigma:.1f} Hz")

    for genre, v in centroid_data.items():
        v["group"] = ("high_spread"
                      if v["sigma_hz"] >= median_sigma
                      else "low_spread")
        v["median_sigma_hz"] = round(median_sigma, 2)

    print(f"\n  {'Genre':<15}  {'sigma_g':>8}  {'Group'}")
    print("  " + "-" * 38)
    for g in sorted(centroid_data.keys(),
                    key=lambda x: centroid_data[x]["sigma_hz"]):
        v = centroid_data[g]
        print(f"  {g:<15}  {v['sigma_hz']:>8.1f}  {v['group']}")

    with open(centroid_path, "w") as f:
        json.dump(centroid_data, f, indent=2)
    save_to_drive(centroid_path, "exp4_fma_centroids.json")
    return centroid_data

# =============================================================================
# STEP 4 — BUILD MEL-SPECTROGRAMS
# =============================================================================

def compute_melspec(track_id, fma_root, n_mels, fmax):
    path = track_id_to_path(track_id, fma_root)
    y, _ = librosa.load(path, sr=SR, mono=True, duration=DURATION)
    S    = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=n_mels, fmin=FMIN, fmax=fmax,
        norm=NORM, power=2.0
    )
    return librosa.power_to_db(S, ref=np.max).astype(np.float32)


def step4_build_spectrograms(genre_map, fma_root, centroid_data):
    """
    Build and cache mel-spectrograms for both anchor configs.
    Cache naming: specs_fma_M{M}_fmax{fmax}.npy
    Returns dict: config_key -> (specs, labels, track_ids, le)
    """
    print("\n=== STEP 4: Build mel-spectrograms for anchor configs ===")

    # Build genre list from centroid_data (only genres with valid audio)
    valid_genres = sorted(centroid_data.keys())
    le = LabelEncoder()
    le.fit(valid_genres)

    result = {}

    for cfg_key, cfg in ANCHOR_CONFIGS.items():
        M    = cfg["M"]
        fmax = cfg["fmax"]

        cache_specs  = os.path.join(OUTPUT_DIR,
                                    f"specs_fma_M{M}_fmax{fmax}.npy")
        cache_labels = os.path.join(OUTPUT_DIR,
                                    f"labels_fma_M{M}_fmax{fmax}.npy")
        cache_ids    = os.path.join(OUTPUT_DIR,
                                    f"tids_fma_M{M}_fmax{fmax}.json")

        if (os.path.exists(cache_specs) and
                os.path.exists(cache_labels) and
                os.path.exists(cache_ids)):
            print(f"  [CACHE] {cfg_key}: loading from local cache.")
            specs  = np.load(cache_specs)
            labels = np.load(cache_labels)
            with open(cache_ids) as f:
                track_ids = json.load(f)
            result[cfg_key] = (specs, labels, track_ids, le)
            continue

        print(f"  Computing mel-spectrograms for {cfg_key} "
              f"(M={M}, fmax={fmax}) ...")

        specs_list  = []
        labels_list = []
        track_ids   = []
        min_T       = None
        skipped     = 0

        for tid, genre in tqdm(genre_map.items(),
                               desc=f"    {cfg_key}"):
            if genre not in valid_genres:
                continue
            wav_path = track_id_to_path(tid, fma_root)
            if not os.path.exists(wav_path):
                skipped += 1
                continue
            try:
                spec = compute_melspec(tid, fma_root, n_mels=M, fmax=fmax)
                specs_list.append(spec)
                labels_list.append(le.transform([genre])[0])
                track_ids.append(tid)
                if min_T is None or spec.shape[1] < min_T:
                    min_T = spec.shape[1]
            except Exception as e:
                skipped += 1

        if skipped > 0:
            print(f"    [WARN] Skipped {skipped} tracks.")

        specs_arr  = np.stack([s[:, :min_T] for s in specs_list], axis=0)
        labels_arr = np.array(labels_list, dtype=np.int64)

        np.save(cache_specs,  specs_arr)
        np.save(cache_labels, labels_arr)
        with open(cache_ids, "w") as f:
            json.dump(track_ids, f)

        print(f"    {cfg_key}: {specs_arr.shape[0]} clips, "
              f"shape ({M}, {min_T}), cached.")
        result[cfg_key] = (specs_arr, labels_arr, track_ids, le)

    return result

# =============================================================================
# DATASET
# =============================================================================

class MelDataset(Dataset):
    def __init__(self, specs, labels):
        self.specs  = torch.from_numpy(specs[:, np.newaxis, :, :])
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.specs[idx], self.labels[idx]

# =============================================================================
# MODEL
# =============================================================================

class GenreCNN(nn.Module):
    def __init__(self, n_classes):
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
# CHECKPOINT HELPERS
# =============================================================================

def ckpt_name(cfg_key, repeat, fold, epoch):
    return (f"exp4_checkpoint_{cfg_key}"
            f"_repeat{repeat}_fold{fold}_epoch{epoch}.pth")

def find_latest_checkpoint(cfg_key, repeat, fold):
    for epoch in range(N_EPOCHS, 0, -CKPT_EVERY):
        fname      = ckpt_name(cfg_key, repeat, fold, epoch)
        local_path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(local_path):
            return epoch, local_path
        if load_from_drive(fname, local_path):
            print(f"    [CKPT] Resumed {cfg_key} repeat={repeat} "
                  f"fold={fold} epoch={epoch} from Drive.")
            return epoch, local_path
    return 0, None

def save_checkpoint(state, cfg_key, repeat, fold, epoch):
    fname      = ckpt_name(cfg_key, repeat, fold, epoch)
    local_path = os.path.join(OUTPUT_DIR, fname)
    torch.save(state, local_path)
    save_to_drive(local_path, fname)

# =============================================================================
# STEP 5 — TRAINING LOOP
# =============================================================================

def train_one_fold(train_specs, train_labels, val_specs, val_labels,
                   cfg_key, repeat, fold, n_classes, valid_genres, le):
    model     = GenreCNN(n_classes=n_classes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    train_ds     = MelDataset(train_specs, train_labels)
    val_ds       = MelDataset(val_specs,   val_labels)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    start_epoch, ckpt_path = find_latest_checkpoint(cfg_key, repeat, fold)
    history = {"train_loss": [], "val_acc": []}

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        history = ckpt.get("history", history)
        print(f"    [RESUME] {cfg_key} repeat={repeat} fold={fold} "
              f"from epoch {start_epoch}")

    for epoch in range(start_epoch + 1, N_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for specs_b, labels_b in train_loader:
            specs_b  = specs_b.to(DEVICE)
            labels_b = labels_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(specs_b), labels_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(labels_b)
        epoch_loss /= len(train_ds)

        model.eval()
        correct = 0
        with torch.no_grad():
            for specs_b, labels_b in val_loader:
                preds    = model(specs_b.to(DEVICE)).argmax(dim=1)
                correct += (preds == labels_b.to(DEVICE)).sum().item()
        val_acc = correct / len(val_ds)

        history["train_loss"].append(round(epoch_loss, 4))
        history["val_acc"].append(round(val_acc,       4))

        if epoch % 10 == 0 or epoch == N_EPOCHS:
            print(f"    {cfg_key} R{repeat} F{fold} ep={epoch:3d}  "
                  f"loss={epoch_loss:.4f}  val_acc={val_acc:.4f}")

        if epoch % CKPT_EVERY == 0:
            save_checkpoint({
                "epoch"          : epoch,
                "model_state"    : model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history"        : history
            }, cfg_key, repeat, fold, epoch)

    # Per-genre accuracy
    model.eval()
    genre_correct = {g: 0 for g in valid_genres}
    genre_total   = {g: 0 for g in valid_genres}

    with torch.no_grad():
        for specs_b, labels_b in val_loader:
            preds     = model(specs_b.to(DEVICE)).argmax(dim=1).cpu().numpy()
            labels_np = labels_b.numpy()
            for pred, true in zip(preds, labels_np):
                g = le.inverse_transform([true])[0]
                genre_total[g]   += 1
                genre_correct[g] += int(pred == true)

    return {
        g: (genre_correct[g] / genre_total[g] if genre_total[g] > 0 else 0.0)
        for g in valid_genres
    }


def step5_run_training(spectrogram_data, centroid_data):
    print(f"\n=== STEP 5: {N_REPEATS}-repeat x {N_FOLDS}-fold CV training "
          f"(FMA-Small, anchor configs) ===")

    valid_genres = sorted(centroid_data.keys())
    n_classes    = len(valid_genres)
    all_results  = {}

    for cfg_key in ANCHOR_CONFIGS:
        print(f"\n  {'='*60}")
        print(f"  Config: {cfg_key}")
        print(f"  {'='*60}")
        specs, labels, _, le = spectrogram_data[cfg_key]
        fold_results = []

        for repeat_idx, seed in enumerate(SEEDS):
            repeat = repeat_idx + 1
            print(f"\n  Repeat {repeat}/{N_REPEATS}  (seed={seed})")
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                  random_state=seed)

            for fold_idx, (train_idx, val_idx) in enumerate(
                    skf.split(specs, labels)):
                fold = fold_idx + 1
                print(f"\n  Fold {fold}/{N_FOLDS}")

                per_genre_acc = train_one_fold(
                    specs[train_idx], labels[train_idx],
                    specs[val_idx],   labels[val_idx],
                    cfg_key, repeat, fold, n_classes, valid_genres, le
                )
                fold_results.append(per_genre_acc)

        all_results[cfg_key] = fold_results
        save_intermediate_results(all_results, valid_genres)

    return all_results

# =============================================================================
# STEP 6 — AGGREGATE
# =============================================================================

def step6_aggregate(all_results, centroid_data):
    print("\n=== STEP 6: Aggregate per-genre accuracy ===")
    valid_genres = sorted(centroid_data.keys())
    aggregated   = {}

    for cfg_key, fold_results in all_results.items():
        aggregated[cfg_key] = {}
        for genre in valid_genres:
            accs = [fr[genre] for fr in fold_results]
            aggregated[cfg_key][genre] = {
                "mean": round(float(np.mean(accs)), 4),
                "std" : round(float(np.std(accs)),  4)
            }
        overall = float(np.mean([aggregated[cfg_key][g]["mean"]
                                  for g in valid_genres]))
        print(f"  {cfg_key}  overall mean acc = {overall:.4f}  "
              f"(across {len(fold_results)} runs)")

    return aggregated

# =============================================================================
# STEP 7 — COMPUTE FFD
# =============================================================================

def step7_compute_ffd(aggregated, centroid_data):
    print("\n=== STEP 7: Compute FFD per anchor config ===")
    valid_genres = sorted(centroid_data.keys())
    ffd_per_cfg  = {}

    for cfg_key in ANCHOR_CONFIGS:
        low_accs  = [aggregated[cfg_key][g]["mean"] for g in valid_genres
                     if centroid_data[g]["group"] == "low_spread"]
        high_accs = [aggregated[cfg_key][g]["mean"] for g in valid_genres
                     if centroid_data[g]["group"] == "high_spread"]
        mean_low  = float(np.mean(low_accs))  if low_accs  else 0.0
        mean_high = float(np.mean(high_accs)) if high_accs else 0.0
        ffd       = abs(mean_high - mean_low)
        ffd_per_cfg[cfg_key] = {
            "mean_acc_low_spread" : round(mean_low,  4),
            "mean_acc_high_spread": round(mean_high, 4),
            "FFD"                 : round(ffd,        4)
        }
        print(f"  {cfg_key}  FFD={ffd:.4f}  "
              f"(low_spread={mean_low:.4f}, high_spread={mean_high:.4f})")

    return ffd_per_cfg

# =============================================================================
# STEP 8 — REPLICATION CHECKS
# =============================================================================

def step8_replication_checks(aggregated, ffd_per_cfg, centroid_data):
    """
    Check 1: FFD direction — does high_spread underperform low_spread
             on FMA-Small as on GTZAN?
    Check 2: Pearson correlation of per-genre Delta_A between GTZAN
             (Exp1, M=128 vs M=40) and FMA-Small (M128_fmax8000 vs M40_fmax4000).
             Genres must be matched by name; FMA and GTZAN share no genres
             exactly, so this check uses the FMA genres only and compares
             the within-FMA Delta_A pattern across configs.
    """
    print("\n=== STEP 8: Replication checks ===")
    valid_genres = sorted(centroid_data.keys())
    checks       = {}

    # Check 1: FFD direction
    ffd_max = ffd_per_cfg.get("M128_fmax8000", {}).get("FFD", 0)
    ffd_min = ffd_per_cfg.get("M40_fmax4000",  {}).get("FFD", 0)
    hi_max  = ffd_per_cfg.get("M128_fmax8000", {}).get("mean_acc_high_spread", 0)
    lo_max  = ffd_per_cfg.get("M128_fmax8000", {}).get("mean_acc_low_spread",  0)
    direction_replicated = hi_max < lo_max

    print(f"  Check 1 — FFD direction replication:")
    print(f"    M128_fmax8000: high_spread={hi_max:.4f} < low_spread={lo_max:.4f} "
          f"-> {'REPLICATED' if direction_replicated else 'NOT REPLICATED'}")
    checks["ffd_direction_replicated"] = direction_replicated
    checks["ffd_M128_fmax8000"]        = round(ffd_max, 4)
    checks["ffd_M40_fmax4000"]         = round(ffd_min, 4)
    checks["ffd_increases_with_deltaD"] = ffd_max > ffd_min

    # Check 2: Within-FMA Delta_A correlation across configs
    # Delta_A per genre = acc(M128_fmax8000) - acc(M40_fmax4000)
    delta_a = []
    for genre in valid_genres:
        a_max = aggregated["M128_fmax8000"][genre]["mean"]
        a_min = aggregated["M40_fmax4000"][genre]["mean"]
        delta_a.append(a_max - a_min)

    # Correlation with group label (+1 for high_spread, -1 for low_spread)
    group_sign = [1 if centroid_data[g]["group"] == "high_spread" else -1
                  for g in valid_genres]

    if len(delta_a) >= 3:
        r, p = pearsonr(group_sign, delta_a)
        print(f"  Check 2 — Pearson r(group_sign, Delta_A) = {r:.4f}  "
              f"p = {p:.4f}")
        print(f"    Interpretation: {'high-spread genres gain more' if r > 0 else 'low-spread genres gain more'} "
              f"from max-Delta_D config vs min-Delta_D config.")
        checks["pearson_r_group_vs_deltaA"] = round(r, 4)
        checks["pearson_p"]                  = round(p, 4)
    else:
        checks["pearson_r_group_vs_deltaA"] = None
        checks["pearson_p"]                  = None

    # Per-genre Delta_A table
    print(f"\n  Per-genre Delta_A (M128_fmax8000 minus M40_fmax4000):")
    print(f"  {'Genre':<15}  {'Delta_A':>8}  {'Group'}")
    print("  " + "-" * 38)
    for genre, da in zip(valid_genres, delta_a):
        print(f"  {genre:<15}  {da:>8.4f}  {centroid_data[genre]['group']}")

    checks["per_genre_deltaA"] = {
        g: round(da, 4) for g, da in zip(valid_genres, delta_a)
    }
    return checks

# =============================================================================
# INTERMEDIATE SAVE
# =============================================================================

def save_intermediate_results(all_results, valid_genres):
    partial = {}
    for cfg_key, fold_results in all_results.items():
        partial[cfg_key] = {
            genre: {
                "mean": round(float(np.mean([fr[genre]
                                             for fr in fold_results])), 4),
                "std" : round(float(np.std( [fr[genre]
                                             for fr in fold_results])), 4)
            }
            for genre in valid_genres
        }
    local_path = os.path.join(OUTPUT_DIR, "exp4_results_partial.json")
    with open(local_path, "w") as f:
        json.dump(partial, f, indent=2)
    save_to_drive(local_path, "exp4_results_partial.json")

# =============================================================================
# STEP 9 — SAVE RESULTS JSON
# =============================================================================

def step9_save_results(aggregated, ffd_per_cfg, centroid_data,
                       all_results, checks):
    print("\n=== STEP 9: Save results JSON ===")
    valid_genres = sorted(centroid_data.keys())

    results = {
        "experiment"    : "Exp4_FMASmallReplication",
        "anchor_configs": ANCHOR_CONFIGS,
        "fixed_params"  : {
            "norm"      : NORM,
            "n_folds"   : N_FOLDS,
            "n_repeats" : N_REPEATS,
            "seeds"     : SEEDS,
            "n_epochs"  : N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr"        : LR
        },
        "fma_corpus_median_sigma_hz": (
            centroid_data[valid_genres[0]].get("median_sigma_hz")
            if valid_genres else None
        ),
        "replication_checks": checks,
        "per_config"        : {}
    }

    for cfg_key in ANCHOR_CONFIGS:
        overall_mean = float(np.mean([aggregated[cfg_key][g]["mean"]
                                      for g in valid_genres]))
        results["per_config"][cfg_key] = {
            "overall_mean_acc": round(overall_mean, 4),
            "n_runs"          : len(all_results[cfg_key]),
            "FFD"             : ffd_per_cfg[cfg_key],
            "per_genre"       : {
                genre: {
                    "mean_acc": aggregated[cfg_key][genre]["mean"],
                    "std_acc" : aggregated[cfg_key][genre]["std"],
                    "group"   : centroid_data[genre]["group"],
                    "sigma_hz": centroid_data[genre]["sigma_hz"]
                }
                for genre in valid_genres
            }
        }

    local_json = os.path.join(OUTPUT_DIR, "exp4_results.json")
    with open(local_json, "w") as f:
        json.dump(results, f, indent=2)
    save_to_drive(local_json, "exp4_results.json")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Program 04 — Experiment 4: FMA-Small Replication  (v1.0)")
    print(f"  Anchor configs: {list(ANCHOR_CONFIGS.keys())}")
    print(f"  {N_REPEATS} repeats x {N_FOLDS} folds = "
          f"{N_REPEATS * N_FOLDS} runs per config")
    print(f"  Seeds: {SEEDS}")
    print("=" * 70)

    fma_root     = step1_prepare_dataset()
    genre_map    = step2_load_metadata()
    centroid_data = step3_compute_fma_centroids(genre_map, fma_root)

    spectrogram_data = step4_build_spectrograms(
        genre_map, fma_root, centroid_data)

    all_results  = step5_run_training(spectrogram_data, centroid_data)
    aggregated   = step6_aggregate(all_results, centroid_data)
    ffd_per_cfg  = step7_compute_ffd(aggregated, centroid_data)
    checks       = step8_replication_checks(aggregated, ffd_per_cfg,
                                            centroid_data)
    step9_save_results(aggregated, ffd_per_cfg, centroid_data,
                       all_results, checks)

    # ── Sync to ensure all writes are flushed ────────────────────────────────
    print("\n[SYNC] Flushing file system buffers...")
    os.sync()
    print("[SYNC] Complete.")

    print("\nProgram 04 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()