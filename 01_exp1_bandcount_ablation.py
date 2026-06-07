# =============================================================================
# Program      : 01_exp1_bandcount_ablation.py
# Version      : 2.0
# Description  : Experiment 1 — Band Count Ablation
#
#                Trains a fixed CNN backbone on GTZAN mel-spectrograms
#                with four filterbank band counts M in {40, 64, 80, 128},
#                holding f_max=8000 Hz and Slaney normalization constant.
#                3-repeat x 5-fold stratified cross-validation (15 runs
#                per M) is used to reduce per-genre accuracy variance on
#                GTZAN's small per-genre sample size (100 clips/genre).
#
#                CNN architecture (identical across all M):
#                  Input : (1, M, T)  — single-channel mel-spectrogram
#                  Block 1: Conv2D(1->32,  3x3, pad=1) + BN + ReLU + MaxPool(2x2)
#                  Block 2: Conv2D(32->64, 3x3, pad=1) + BN + ReLU + MaxPool(2x2)
#                  Block 3: Conv2D(64->128,3x3, pad=1) + BN + ReLU + MaxPool(2x2)
#                  Global Average Pooling -> FC(128->10) -> Softmax
#
#                Each GTZAN clip (30 s) is converted to a full mel-spectrogram
#                and used as a single training sample. No segmentation.
#
#                Training: 50 epochs, Adam lr=1e-3, CrossEntropyLoss.
#                Checkpoints saved every 5 epochs per (M, repeat, fold) to Drive.
#                Resumption logic detects existing checkpoints and continues
#                from the last saved epoch.
#
#                The trained models at M=40 and M=128 (best fold across all
#                repeats by val accuracy) are saved as final PTH files for
#                reuse by Program 05 (Experiment 5, gradient saliency probe).
#
# INPUT        :
#                  Google Drive:
#                    /content/drive/MyDrive/datasets/GTZAN.zip
#                  Drive:
#                    precompute_genre_centroids.json
#                    precompute_filter_density.json
#
# STEPS        :
#                  Step 1  Mount Drive, copy and extract GTZAN
#                  Step 2  Download precompute_genre_centroids.json from Drive
#                  Step 3  Build full clip mel-spectrograms for all M values
#                  Step 4  3-repeat x 5-fold CV training loop for each M,
#                            with checkpoint save/resume per (M, repeat, fold)
#                  Step 5  Aggregate per-genre accuracy across 15 runs
#                  Step 6  Compute FFD per M
#                  Step 7  Save best models for M=40 and M=128
#                  Step 8  Save results JSON and figures, upload to Drive
#
# OUTPUT FILES :
#                  exp1_results.json
#                      Per (M, genre): mean accuracy, std across 15 runs
#                      Per M: overall accuracy, FFD
#                      D_g values from precompute_filter_density.json
#
#                  exp1_results_partial.json
#                      Incremental save after each completed M
#
#                  exp1_checkpoint_M{m}_repeat{r}_fold{f}_epoch{e}.pth
#                      Periodic checkpoints (every 5 epochs)
#
#                  exp1_best_model_M40.pth
#                  exp1_best_model_M128.pth
#                      Best-run final models for saliency probe (Program 05)
#
#                  fig_01_01_accuracy_heatmap.png
#                      Heatmap: genres (y, sorted by sigma_g) x M values (x)
#                      Cell = mean per-genre accuracy across 15 runs
#                      FFD overlaid as line on secondary axis
#                      (Contributes M-axis panel of paper Figure 1)
#
# GPU Required : YES
# Dependencies : torch, torchaudio, librosa, numpy, scikit-learn,
#                matplotlib, tqdm
#
# Change Log   :
#   v1.0  2026-06-06  Initial version (5-fold CV)
#   v2.0  2026-06-06  Upgraded to 3-repeat x 5-fold repeated CV;
#                     checkpoint keys now include repeat index;
#                     seeds = [42, 7, 123]
# =============================================================================

import subprocess
import sys

for pkg in ["torch", "torchaudio", "librosa", "numpy",
            "scikit-learn", "matplotlib", "tqdm"]:
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
import matplotlib.ticker as mticker
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
OUTPUT_DIR      = os.path.join(LOCAL_WORK_DIR, "outputs_exp1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Audio ─────────────────────────────────────────────────────────────────────
SR          = 22050
HOP_LENGTH  = 512
N_FFT       = 2048
FMAX        = 8000
FMIN        = 0.0
NORM_SCHEME = "slaney"
DURATION    = 29.0

# ── Experiment ────────────────────────────────────────────────────────────────
M_VALUES    = [40, 64, 80, 128]
N_FOLDS     = 5
N_REPEATS   = 3
SEEDS       = [42, 7, 123]
N_EPOCHS    = 50
CKPT_EVERY  = 5
BATCH_SIZE  = 16
LR          = 1e-3
DEVICE      = torch.device("cuda")

GTZAN_GENRES = [
    "blues", "classical", "country", "disco",
    "hiphop", "jazz", "metal", "pop", "reggae", "rock"
]
N_CLASSES = len(GTZAN_GENRES)

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
# STEP 2 — LOAD CENTROID DATA
# =============================================================================

def step2_load_centroids():
    print("\n=== STEP 2: Load genre centroids ===")
    local_path = os.path.join(OUTPUT_DIR, "precompute_genre_centroids.json")
    if not os.path.exists(local_path):
        if not load_from_drive("precompute_genre_centroids.json", local_path):
            raise RuntimeError("Cannot find precompute_genre_centroids.json "
                               "on Drive. Run Program 00 first.")
    with open(local_path) as f:
        data = json.load(f)
    print(f"  Loaded centroids for {len(data)} genres.")
    return data

# =============================================================================
# STEP 3 — BUILD MEL-SPECTROGRAMS
# =============================================================================

def compute_melspec(wav_path, n_mels, fmax=FMAX):
    y, _ = librosa.load(wav_path, sr=SR, mono=True, duration=DURATION)
    S    = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=n_mels, fmin=FMIN, fmax=fmax,
        norm=NORM_SCHEME, power=2.0
    )
    return librosa.power_to_db(S, ref=np.max).astype(np.float32)


def step3_build_spectrograms(genre_root, centroid_data):
    """
    Build and cache mel-spectrograms for all M values.
    Returns dict: M -> (specs array, labels array, file_ids list)
      specs  : (N, n_mels, T_min)
      labels : (N,) int64
    """
    print("\n=== STEP 3: Build mel-spectrograms ===")

    le = LabelEncoder()
    le.fit(GTZAN_GENRES)
    result = {}

    for M in M_VALUES:
        cache_specs  = os.path.join(OUTPUT_DIR, f"specs_M{M}.npy")
        cache_labels = os.path.join(OUTPUT_DIR, f"labels_M{M}.npy")
        cache_ids    = os.path.join(OUTPUT_DIR, f"fileids_M{M}.json")

        if (os.path.exists(cache_specs) and
                os.path.exists(cache_labels) and
                os.path.exists(cache_ids)):
            print(f"  [CACHE] M={M}: loading from local cache.")
            specs  = np.load(cache_specs)
            labels = np.load(cache_labels)
            with open(cache_ids) as f:
                file_ids = json.load(f)
            result[M] = (specs, labels, file_ids)
            continue

        print(f"  Computing mel-spectrograms for M={M} ...")
        specs_list  = []
        labels_list = []
        file_ids    = []
        min_T       = None

        for genre in tqdm(GTZAN_GENRES, desc=f"    M={M} genres"):
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
                    specs_list.append(spec)
                    labels_list.append(le.transform([genre])[0])
                    file_ids.append(f"{genre}/{wav_name}")
                    if min_T is None or spec.shape[1] < min_T:
                        min_T = spec.shape[1]
                except Exception as e:
                    print(f"    [WARN] Skipping {wav_name}: {e}")

        specs_arr  = np.stack([s[:, :min_T] for s in specs_list], axis=0)
        labels_arr = np.array(labels_list, dtype=np.int64)

        np.save(cache_specs,  specs_arr)
        np.save(cache_labels, labels_arr)
        with open(cache_ids, "w") as f:
            json.dump(file_ids, f)

        print(f"    M={M}: {specs_arr.shape[0]} clips, "
              f"shape ({M}, {min_T}), cached.")
        result[M] = (specs_arr, labels_arr, file_ids)

    return result, le

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
    """
    Fixed 3-block CNN. Channel progression: 32 -> 64 -> 128.
    Input: (B, 1, M, T)
    """
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
# CHECKPOINT HELPERS
# =============================================================================

def ckpt_name(M, repeat, fold, epoch):
    return f"exp1_checkpoint_M{M}_repeat{repeat}_fold{fold}_epoch{epoch}.pth"

def find_latest_checkpoint(M, repeat, fold):
    """
    Scan local disk then Drive for the latest checkpoint for (M, repeat, fold).
    Returns (epoch, local_path) or (0, None) if none found.
    """
    for epoch in range(N_EPOCHS, 0, -CKPT_EVERY):
        fname      = ckpt_name(M, repeat, fold, epoch)
        local_path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(local_path):
            return epoch, local_path
        if load_from_drive(fname, local_path):
            print(f"    [CKPT] Resumed M={M} repeat={repeat} "
                  f"fold={fold} epoch={epoch} from Drive.")
            return epoch, local_path
    return 0, None

def save_checkpoint(state, M, repeat, fold, epoch):
    fname      = ckpt_name(M, repeat, fold, epoch)
    local_path = os.path.join(OUTPUT_DIR, fname)
    torch.save(state, local_path)
    save_to_drive(local_path, fname)

# =============================================================================
# STEP 4 — TRAINING LOOP
# =============================================================================

def train_one_fold(train_specs, train_labels, val_specs, val_labels,
                   M, repeat, fold, le):
    """
    Train GenreCNN for N_EPOCHS on one (repeat, fold).
    Returns (model, per_genre_acc dict, history dict).
    Resumes from checkpoint if available.
    """
    model     = GenreCNN(n_classes=N_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    train_ds     = MelDataset(train_specs, train_labels)
    val_ds       = MelDataset(val_specs,   val_labels)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    start_epoch, ckpt_path = find_latest_checkpoint(M, repeat, fold)
    history = {"train_loss": [], "val_acc": []}

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        history = ckpt.get("history", history)
        print(f"    [RESUME] M={M} repeat={repeat} fold={fold} "
              f"from epoch {start_epoch}")

    for epoch in range(start_epoch + 1, N_EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────────────────────
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

        # ── Validate ──────────────────────────────────────────────────────────
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
            print(f"    M={M} R{repeat} F{fold} ep={epoch:3d}  "
                  f"loss={epoch_loss:.4f}  val_acc={val_acc:.4f}")

        if epoch % CKPT_EVERY == 0:
            save_checkpoint({
                "epoch"          : epoch,
                "model_state"    : model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history"        : history
            }, M, repeat, fold, epoch)

    # ── Per-genre accuracy on validation set ─────────────────────────────────
    model.eval()
    genre_correct = {g: 0 for g in GTZAN_GENRES}
    genre_total   = {g: 0 for g in GTZAN_GENRES}

    with torch.no_grad():
        for specs_b, labels_b in val_loader:
            preds     = model(specs_b.to(DEVICE)).argmax(dim=1).cpu().numpy()
            labels_np = labels_b.numpy()
            for pred, true in zip(preds, labels_np):
                g = le.inverse_transform([true])[0]
                genre_total[g]   += 1
                genre_correct[g] += int(pred == true)

    per_genre_acc = {
        g: (genre_correct[g] / genre_total[g] if genre_total[g] > 0 else 0.0)
        for g in GTZAN_GENRES
    }
    return model, per_genre_acc, history


def step4_run_ablation(spectrogram_data, le):
    print(f"\n=== STEP 4: {N_REPEATS}-repeat x {N_FOLDS}-fold CV training ===")

    all_results = {}   # M -> list of (N_REPEATS * N_FOLDS) per-genre-acc dicts
    best_models = {}   # M -> best model state_dict (for M=40 and M=128)

    for M in M_VALUES:
        print(f"\n  {'='*60}")
        print(f"  M = {M}")
        print(f"  {'='*60}")
        specs, labels, _ = spectrogram_data[M]
        fold_results = []
        best_val_acc = -1.0

        for repeat_idx, seed in enumerate(SEEDS):
            repeat = repeat_idx + 1
            print(f"\n  Repeat {repeat}/{N_REPEATS}  (seed={seed})")

            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                  random_state=seed)

            for fold_idx, (train_idx, val_idx) in enumerate(
                    skf.split(specs, labels)):
                fold = fold_idx + 1
                print(f"\n  Fold {fold}/{N_FOLDS}")

                model, per_genre_acc, history = train_one_fold(
                    specs[train_idx], labels[train_idx],
                    specs[val_idx],   labels[val_idx],
                    M, repeat, fold, le
                )
                fold_results.append(per_genre_acc)

                # Track best run model for M=40 and M=128
                run_overall = float(np.mean(list(per_genre_acc.values())))
                if M in [40, 128] and run_overall > best_val_acc:
                    best_val_acc = run_overall
                    best_models[M] = model.state_dict()

        all_results[M] = fold_results

        # Save partial results after each M completes
        save_intermediate_results(all_results)

    return all_results, best_models

# =============================================================================
# STEP 5 — AGGREGATE PER-GENRE ACCURACY
# =============================================================================

def step5_aggregate(all_results):
    """
    For each (M, genre): mean and std across N_REPEATS * N_FOLDS = 15 runs.
    """
    print("\n=== STEP 5: Aggregate per-genre accuracy ===")
    aggregated = {}

    for M, fold_results in all_results.items():
        aggregated[M] = {}
        for genre in GTZAN_GENRES:
            accs = [fr[genre] for fr in fold_results]
            aggregated[M][genre] = {
                "mean": round(float(np.mean(accs)), 4),
                "std" : round(float(np.std(accs)),  4)
            }
        overall = float(np.mean([aggregated[M][g]["mean"] for g in GTZAN_GENRES]))
        print(f"  M={M:3d}  overall mean acc = {overall:.4f}  "
              f"(across {len(fold_results)} runs)")

    return aggregated

# =============================================================================
# STEP 6 — COMPUTE FFD
# =============================================================================

def step6_compute_ffd(aggregated, centroid_data):
    """
    FFD(theta) = |mean_acc(high_spread) - mean_acc(low_spread)|
    """
    print("\n=== STEP 6: Compute FFD per M ===")
    ffd_per_M = {}

    for M in M_VALUES:
        low_accs  = [aggregated[M][g]["mean"] for g in GTZAN_GENRES
                     if centroid_data.get(g, {}).get("group") == "low_spread"]
        high_accs = [aggregated[M][g]["mean"] for g in GTZAN_GENRES
                     if centroid_data.get(g, {}).get("group") == "high_spread"]
        mean_low  = float(np.mean(low_accs))  if low_accs  else 0.0
        mean_high = float(np.mean(high_accs)) if high_accs else 0.0
        ffd       = abs(mean_high - mean_low)
        ffd_per_M[M] = {
            "mean_acc_low_spread" : round(mean_low,  4),
            "mean_acc_high_spread": round(mean_high, 4),
            "FFD"                 : round(ffd,        4)
        }
        print(f"  M={M:3d}  FFD={ffd:.4f}  "
              f"(low_spread={mean_low:.4f}, high_spread={mean_high:.4f})")

    return ffd_per_M

# =============================================================================
# STEP 7 — SAVE BEST MODELS FOR M=40 AND M=128
# =============================================================================

def step7_save_best_models(best_models):
    print("\n=== STEP 7: Save best models for M=40 and M=128 ===")
    for M in [40, 128]:
        if M not in best_models:
            print(f"  [WARN] No best model recorded for M={M}")
            continue
        fname      = f"exp1_best_model_M{M}.pth"
        local_path = os.path.join(OUTPUT_DIR, fname)
        torch.save({"M": M, "model_state": best_models[M]}, local_path)
        save_to_drive(local_path, fname)
        print(f"  Saved and uploaded: {fname}")

# =============================================================================
# INTERMEDIATE SAVE
# =============================================================================

def save_intermediate_results(all_results):
    partial = {}
    for M, fold_results in all_results.items():
        partial[str(M)] = {
            genre: {
                "mean": round(float(np.mean([fr[genre] for fr in fold_results])), 4),
                "std" : round(float(np.std( [fr[genre] for fr in fold_results])), 4)
            }
            for genre in GTZAN_GENRES
        }
    local_path = os.path.join(OUTPUT_DIR, "exp1_results_partial.json")
    with open(local_path, "w") as f:
        json.dump(partial, f, indent=2)
    save_to_drive(local_path, "exp1_results_partial.json")

# =============================================================================
# STEP 8 — FIGURES AND FINAL JSON
# =============================================================================

def make_fig_accuracy_heatmap(aggregated, ffd_per_M, centroid_data):
    """
    Fig 01_01: Heatmap of per-genre accuracy vs M.
    Genres sorted by sigma_g ascending (rows). M values as columns.
    FFD overlaid as line on secondary y-axis.
    """
    genres_sorted = sorted(
        GTZAN_GENRES,
        key=lambda g: centroid_data.get(g, {}).get("sigma_hz", 0)
    )
    n_g = len(genres_sorted)
    n_M = len(M_VALUES)

    matrix = np.zeros((n_g, n_M))
    for ci, M in enumerate(M_VALUES):
        for ri, genre in enumerate(genres_sorted):
            matrix[ri, ci] = aggregated[M][genre]["mean"]

    fig, ax1 = plt.subplots(figsize=(8, 6))
    im = ax1.imshow(matrix, aspect="auto", cmap="RdYlGn",
                    vmin=0.0, vmax=1.0, interpolation="nearest")

    ax1.set_xticks(np.arange(n_M))
    ax1.set_xticklabels([f"M={M}" for M in M_VALUES], fontsize=10)
    ax1.set_yticks(np.arange(n_g))
    ax1.set_yticklabels(
        [f"{g.capitalize()}\n"
         f"$\\sigma$={centroid_data.get(g,{}).get('sigma_hz',0):.0f} Hz"
         for g in genres_sorted],
        fontsize=8.5
    )

    for ri in range(n_g):
        for ci in range(n_M):
            val = matrix[ri, ci]
            ax1.text(ci, ri, f"{val:.2f}", ha="center", va="center",
                     fontsize=7.5,
                     color="black" if 0.3 < val < 0.85 else "white")

    # Group boundary line
    low_rows  = [ri for ri, g in enumerate(genres_sorted)
                 if centroid_data.get(g, {}).get("group") == "low_spread"]
    high_rows = [ri for ri, g in enumerate(genres_sorted)
                 if centroid_data.get(g, {}).get("group") == "high_spread"]
    if low_rows and high_rows:
        boundary = (max(low_rows) + min(high_rows)) / 2.0
        ax1.axhline(boundary, color="black", linewidth=1.5,
                    linestyle="--", alpha=0.6, label="Spread group boundary")

    plt.colorbar(im, ax=ax1, label="Per-Genre Accuracy")

    # FFD overlay
    ax2      = ax1.twinx()
    ffd_vals = [ffd_per_M[M]["FFD"] for M in M_VALUES]
    ax2.plot(np.arange(n_M), ffd_vals, color="#D95F02", marker="o",
             linewidth=2.0, markersize=7, label="FFD", zorder=5)
    ax2.set_ylabel("FFD (Filterbank Fairness Disparity)",
                   color="#D95F02", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="#D95F02")
    ax2.set_ylim(0, max(ffd_vals) * 1.5 if max(ffd_vals) > 0 else 0.1)
    ax2.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    if low_rows:
        ax1.axhspan(min(low_rows) - 0.5, max(low_rows) + 0.5,
                    alpha=0.05, color="#1B9E77", zorder=0)
    if high_rows:
        ax1.axhspan(min(high_rows) - 0.5, max(high_rows) + 0.5,
                    alpha=0.05, color="#D95F02", zorder=0)

    ax1.set_title(
        "Per-Genre Accuracy vs Band Count $M$\n"
        f"(GTZAN, $f_{{max}}$=8 kHz, Slaney norm, "
        f"{N_REPEATS}x{N_FOLDS}-fold repeated CV)\n"
        "Genres sorted by $\\sigma_g$ ascending; orange line = FFD",
        fontsize=10
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="lower right", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig_01_01_accuracy_heatmap.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


def step8_save_and_upload(aggregated, ffd_per_M, centroid_data, all_results):
    print("\n=== STEP 8: Save results JSON and figures ===")

    # Load D_g values
    dg_per_M     = {M: {} for M in M_VALUES}
    density_path = os.path.join(OUTPUT_DIR, "precompute_filter_density.json")
    if not os.path.exists(density_path):
        load_from_drive("precompute_filter_density.json", density_path)
    if os.path.exists(density_path):
        with open(density_path) as f:
            density_data = json.load(f)
        for M in M_VALUES:
            cfg = f"M{M}_fmax{FMAX}_slaney"
            for genre in GTZAN_GENRES:
                dg_per_M[M][genre] = (density_data
                                      .get("per_genre", {})
                                      .get(genre, {})
                                      .get(cfg, {})
                                      .get("D_g", None))

    results = {
        "experiment" : "Exp1_BandCountAblation",
        "fixed_params": {
            "fmax_hz"   : FMAX,
            "norm"      : NORM_SCHEME,
            "n_folds"   : N_FOLDS,
            "n_repeats" : N_REPEATS,
            "seeds"     : SEEDS,
            "n_epochs"  : N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr"        : LR
        },
        "per_M": {}
    }

    for M in M_VALUES:
        overall_mean = float(np.mean([aggregated[M][g]["mean"]
                                      for g in GTZAN_GENRES]))
        results["per_M"][str(M)] = {
            "overall_mean_acc": round(overall_mean, 4),
            "n_runs"          : len(all_results[M]),
            "FFD"             : ffd_per_M[M],
            "per_genre"       : {
                genre: {
                    "mean_acc": aggregated[M][genre]["mean"],
                    "std_acc" : aggregated[M][genre]["std"],
                    "D_g"     : dg_per_M[M].get(genre),
                    "group"   : centroid_data.get(genre, {}).get("group"),
                    "sigma_hz": centroid_data.get(genre, {}).get("sigma_hz")
                }
                for genre in GTZAN_GENRES
            }
        }

    local_json = os.path.join(OUTPUT_DIR, "exp1_results.json")
    with open(local_json, "w") as f:
        json.dump(results, f, indent=2)
    save_to_drive(local_json, "exp1_results.json")

    fig_path = make_fig_accuracy_heatmap(aggregated, ffd_per_M, centroid_data)
    save_to_drive(fig_path, "fig_01_01_accuracy_heatmap.png")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Program 01 — Experiment 1: Band Count Ablation  (v2.0)")
    print(f"  {N_REPEATS} repeats x {N_FOLDS} folds = "
          f"{N_REPEATS * N_FOLDS} runs per M value")
    print(f"  Seeds: {SEEDS}")
    print("=" * 70)

    genre_root    = step1_prepare_dataset()
    centroid_data = step2_load_centroids()

    spectrogram_data, le = step3_build_spectrograms(genre_root, centroid_data)

    all_results, best_models = step4_run_ablation(spectrogram_data, le)

    aggregated = step5_aggregate(all_results)
    ffd_per_M  = step6_compute_ffd(aggregated, centroid_data)

    step7_save_best_models(best_models)
    step8_save_and_upload(aggregated, ffd_per_M, centroid_data, all_results)

    # ── Sync to ensure all writes are flushed ────────────────────────────────
    print("\n[SYNC] Flushing file system buffers...")
    os.sync()
    print("[SYNC] Complete.")

    print("\nProgram 01 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()