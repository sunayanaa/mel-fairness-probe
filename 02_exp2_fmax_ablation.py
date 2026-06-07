# =============================================================================
# Program      : 02_exp2_fmax_ablation.py
# Version      : 1.0
# Description  : Experiment 2 — Frequency Ceiling Ablation
#
#                Trains the fixed GenreCNN backbone on GTZAN mel-spectrograms
#                with three frequency ceilings f_max in {4000, 8000, 11025} Hz,
#                holding M=64 and Slaney normalization constant.
#                3-repeat x 5-fold stratified cross-validation (15 runs per
#                f_max) is used throughout, identical to Experiment 1.
#
#                CNN architecture (identical to Program 01):
#                  Input : (1, 64, T)
#                  Block 1: Conv2D(1->32,  3x3, pad=1) + BN + ReLU + MaxPool(2x2)
#                  Block 2: Conv2D(32->64, 3x3, pad=1) + BN + ReLU + MaxPool(2x2)
#                  Block 3: Conv2D(64->128,3x3, pad=1) + BN + ReLU + MaxPool(2x2)
#                  Global Average Pooling -> FC(128->10) -> Softmax
#
#                Spectrogram cache files use fully qualified naming:
#                  specs_M{M}_fmax{fmax}.npy
#
#                No new standalone figure is produced. Results feed into
#                the combined Figure 1 (f_max panel) written at paper
#                compilation time. Output is exp2_results.json only.
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
#                  Step 2  Load precompute_genre_centroids.json from Drive
#                  Step 3  Build mel-spectrograms for all f_max values (M=64)
#                  Step 4  3-repeat x 5-fold CV training loop for each f_max,
#                            with checkpoint save/resume
#                  Step 5  Aggregate per-genre accuracy across 15 runs
#                  Step 6  Compute FFD per f_max configuration
#                  Step 7  Save results JSON, upload to Drive
#
# OUTPUT FILES :
#                  exp2_results.json
#                      Per (f_max, genre): mean accuracy, std across 15 runs
#                      Per f_max: overall accuracy, FFD
#                      D_g values from precompute_filter_density.json
#
#                  exp2_results_partial.json
#                      Incremental save after each completed f_max
#
#                  exp2_checkpoint_M64_fmax{f}_repeat{r}_fold{f}.pth
#                      Periodic checkpoints (every 5 epochs)
#
# GPU Required : YES
# Dependencies : torch, torchaudio, librosa, numpy, scikit-learn,
#                matplotlib, tqdm
#
# Change Log   :
#   v1.0  2026-06-06  Initial version
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
OUTPUT_DIR      = os.path.join(LOCAL_WORK_DIR, "outputs_exp2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Audio ─────────────────────────────────────────────────────────────────────
SR         = 22050
HOP_LENGTH = 512
N_FFT      = 2048
M_FIXED    = 64           # fixed for Experiment 2
FMIN       = 0.0
NORM       = "slaney"     # fixed for Experiment 2
DURATION   = 29.0

# ── Experiment ────────────────────────────────────────────────────────────────
FMAX_VALUES = [4000, 8000, 11025]
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

def compute_melspec(wav_path, n_mels, fmax):
    y, _ = librosa.load(wav_path, sr=SR, mono=True, duration=DURATION)
    S    = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=n_mels, fmin=FMIN, fmax=fmax,
        norm=NORM, power=2.0
    )
    return librosa.power_to_db(S, ref=np.max).astype(np.float32)


def step3_build_spectrograms(genre_root):
    """
    Build and cache mel-spectrograms for all f_max values at M=64.
    Cache naming: specs_M{M}_fmax{fmax}.npy
    Returns dict: fmax -> (specs array, labels array, file_ids list)
    """
    print("\n=== STEP 3: Build mel-spectrograms (M=64, all f_max) ===")

    le = LabelEncoder()
    le.fit(GTZAN_GENRES)
    result = {}

    for fmax in FMAX_VALUES:
        cache_specs  = os.path.join(OUTPUT_DIR,
                                    f"specs_M{M_FIXED}_fmax{fmax}.npy")
        cache_labels = os.path.join(OUTPUT_DIR,
                                    f"labels_M{M_FIXED}_fmax{fmax}.npy")
        cache_ids    = os.path.join(OUTPUT_DIR,
                                    f"fileids_M{M_FIXED}_fmax{fmax}.json")

        if (os.path.exists(cache_specs) and
                os.path.exists(cache_labels) and
                os.path.exists(cache_ids)):
            print(f"  [CACHE] M={M_FIXED} fmax={fmax}: loading from local cache.")
            specs  = np.load(cache_specs)
            labels = np.load(cache_labels)
            with open(cache_ids) as f:
                file_ids = json.load(f)
            result[fmax] = (specs, labels, file_ids)
            continue

        print(f"  Computing mel-spectrograms for M={M_FIXED}, fmax={fmax} ...")
        specs_list  = []
        labels_list = []
        file_ids    = []
        min_T       = None

        for genre in tqdm(GTZAN_GENRES, desc=f"    fmax={fmax} genres"):
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
                    spec = compute_melspec(wav_path, n_mels=M_FIXED, fmax=fmax)
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

        print(f"    fmax={fmax}: {specs_arr.shape[0]} clips, "
              f"shape ({M_FIXED}, {min_T}), cached.")
        result[fmax] = (specs_arr, labels_arr, file_ids)

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

def ckpt_name(fmax, repeat, fold, epoch):
    return (f"exp2_checkpoint_M{M_FIXED}_fmax{fmax}"
            f"_repeat{repeat}_fold{fold}_epoch{epoch}.pth")

def find_latest_checkpoint(fmax, repeat, fold):
    for epoch in range(N_EPOCHS, 0, -CKPT_EVERY):
        fname      = ckpt_name(fmax, repeat, fold, epoch)
        local_path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(local_path):
            return epoch, local_path
        if load_from_drive(fname, local_path):
            print(f"    [CKPT] Resumed fmax={fmax} repeat={repeat} "
                  f"fold={fold} epoch={epoch} from Drive.")
            return epoch, local_path
    return 0, None

def save_checkpoint(state, fmax, repeat, fold, epoch):
    fname      = ckpt_name(fmax, repeat, fold, epoch)
    local_path = os.path.join(OUTPUT_DIR, fname)
    torch.save(state, local_path)
    save_to_drive(local_path, fname)

# =============================================================================
# STEP 4 — TRAINING LOOP
# =============================================================================

def train_one_fold(train_specs, train_labels, val_specs, val_labels,
                   fmax, repeat, fold, le):
    model     = GenreCNN(n_classes=N_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    train_ds     = MelDataset(train_specs, train_labels)
    val_ds       = MelDataset(val_specs,   val_labels)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    start_epoch, ckpt_path = find_latest_checkpoint(fmax, repeat, fold)
    history = {"train_loss": [], "val_acc": []}

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        history = ckpt.get("history", history)
        print(f"    [RESUME] fmax={fmax} repeat={repeat} fold={fold} "
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
            print(f"    fmax={fmax} R{repeat} F{fold} ep={epoch:3d}  "
                  f"loss={epoch_loss:.4f}  val_acc={val_acc:.4f}")

        if epoch % CKPT_EVERY == 0:
            save_checkpoint({
                "epoch"          : epoch,
                "model_state"    : model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history"        : history
            }, fmax, repeat, fold, epoch)

    # ── Per-genre accuracy ────────────────────────────────────────────────────
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
    return per_genre_acc


def step4_run_ablation(spectrogram_data, le):
    print(f"\n=== STEP 4: {N_REPEATS}-repeat x {N_FOLDS}-fold CV training "
          f"(M={M_FIXED}, varying f_max) ===")

    all_results = {}   # fmax -> list of 15 per-genre-acc dicts

    for fmax in FMAX_VALUES:
        print(f"\n  {'='*60}")
        print(f"  f_max = {fmax} Hz")
        print(f"  {'='*60}")
        specs, labels, _ = spectrogram_data[fmax]
        fold_results     = []

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
                    fmax, repeat, fold, le
                )
                fold_results.append(per_genre_acc)

        all_results[fmax] = fold_results
        save_intermediate_results(all_results)

    return all_results

# =============================================================================
# STEP 5 — AGGREGATE PER-GENRE ACCURACY
# =============================================================================

def step5_aggregate(all_results):
    print("\n=== STEP 5: Aggregate per-genre accuracy ===")
    aggregated = {}

    for fmax, fold_results in all_results.items():
        aggregated[fmax] = {}
        for genre in GTZAN_GENRES:
            accs = [fr[genre] for fr in fold_results]
            aggregated[fmax][genre] = {
                "mean": round(float(np.mean(accs)), 4),
                "std" : round(float(np.std(accs)),  4)
            }
        overall = float(np.mean([aggregated[fmax][g]["mean"]
                                  for g in GTZAN_GENRES]))
        print(f"  fmax={fmax:6d}  overall mean acc = {overall:.4f}  "
              f"(across {len(fold_results)} runs)")

    return aggregated

# =============================================================================
# STEP 6 — COMPUTE FFD
# =============================================================================

def step6_compute_ffd(aggregated, centroid_data):
    print("\n=== STEP 6: Compute FFD per f_max ===")
    ffd_per_fmax = {}

    for fmax in FMAX_VALUES:
        low_accs  = [aggregated[fmax][g]["mean"] for g in GTZAN_GENRES
                     if centroid_data.get(g, {}).get("group") == "low_spread"]
        high_accs = [aggregated[fmax][g]["mean"] for g in GTZAN_GENRES
                     if centroid_data.get(g, {}).get("group") == "high_spread"]
        mean_low  = float(np.mean(low_accs))  if low_accs  else 0.0
        mean_high = float(np.mean(high_accs)) if high_accs else 0.0
        ffd       = abs(mean_high - mean_low)
        ffd_per_fmax[fmax] = {
            "mean_acc_low_spread" : round(mean_low,  4),
            "mean_acc_high_spread": round(mean_high, 4),
            "FFD"                 : round(ffd,        4)
        }
        print(f"  fmax={fmax:6d}  FFD={ffd:.4f}  "
              f"(low_spread={mean_low:.4f}, high_spread={mean_high:.4f})")

    return ffd_per_fmax

# =============================================================================
# INTERMEDIATE SAVE
# =============================================================================

def save_intermediate_results(all_results):
    partial = {}
    for fmax, fold_results in all_results.items():
        partial[str(fmax)] = {
            genre: {
                "mean": round(float(np.mean([fr[genre]
                                             for fr in fold_results])), 4),
                "std" : round(float(np.std( [fr[genre]
                                             for fr in fold_results])), 4)
            }
            for genre in GTZAN_GENRES
        }
    local_path = os.path.join(OUTPUT_DIR, "exp2_results_partial.json")
    with open(local_path, "w") as f:
        json.dump(partial, f, indent=2)
    save_to_drive(local_path, "exp2_results_partial.json")

# =============================================================================
# STEP 7 — SAVE RESULTS JSON
# =============================================================================

def step7_save_results(aggregated, ffd_per_fmax, centroid_data, all_results):
    print("\n=== STEP 7: Save results JSON ===")

    # Load D_g values from precompute_filter_density.json
    dg_per_fmax  = {fmax: {} for fmax in FMAX_VALUES}
    density_path = os.path.join(OUTPUT_DIR, "precompute_filter_density.json")
    if not os.path.exists(density_path):
        load_from_drive("precompute_filter_density.json", density_path)
    if os.path.exists(density_path):
        with open(density_path) as f:
            density_data = json.load(f)
        for fmax in FMAX_VALUES:
            cfg = f"M{M_FIXED}_fmax{fmax}_slaney"
            for genre in GTZAN_GENRES:
                dg_per_fmax[fmax][genre] = (density_data
                                            .get("per_genre", {})
                                            .get(genre, {})
                                            .get(cfg, {})
                                            .get("D_g", None))

    results = {
        "experiment"  : "Exp2_FreqCeilingAblation",
        "fixed_params": {
            "M"         : M_FIXED,
            "norm"      : NORM,
            "n_folds"   : N_FOLDS,
            "n_repeats" : N_REPEATS,
            "seeds"     : SEEDS,
            "n_epochs"  : N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr"        : LR
        },
        "per_fmax": {}
    }

    for fmax in FMAX_VALUES:
        overall_mean = float(np.mean([aggregated[fmax][g]["mean"]
                                      for g in GTZAN_GENRES]))
        results["per_fmax"][str(fmax)] = {
            "overall_mean_acc": round(overall_mean, 4),
            "n_runs"          : len(all_results[fmax]),
            "FFD"             : ffd_per_fmax[fmax],
            "per_genre"       : {
                genre: {
                    "mean_acc": aggregated[fmax][genre]["mean"],
                    "std_acc" : aggregated[fmax][genre]["std"],
                    "D_g"     : dg_per_fmax[fmax].get(genre),
                    "group"   : centroid_data.get(genre, {}).get("group"),
                    "sigma_hz": centroid_data.get(genre, {}).get("sigma_hz")
                }
                for genre in GTZAN_GENRES
            }
        }

    local_json = os.path.join(OUTPUT_DIR, "exp2_results.json")
    with open(local_json, "w") as f:
        json.dump(results, f, indent=2)
    save_to_drive(local_json, "exp2_results.json")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Program 02 — Experiment 2: Frequency Ceiling Ablation  (v1.0)")
    print(f"  M fixed = {M_FIXED}")
    print(f"  f_max values: {FMAX_VALUES}")
    print(f"  {N_REPEATS} repeats x {N_FOLDS} folds = "
          f"{N_REPEATS * N_FOLDS} runs per f_max")
    print(f"  Seeds: {SEEDS}")
    print("=" * 70)

    genre_root    = step1_prepare_dataset()
    centroid_data = step2_load_centroids()

    spectrogram_data, le = step3_build_spectrograms(genre_root)

    all_results = step4_run_ablation(spectrogram_data, le)

    aggregated   = step5_aggregate(all_results)
    ffd_per_fmax = step6_compute_ffd(aggregated, centroid_data)

    step7_save_results(aggregated, ffd_per_fmax, centroid_data, all_results)

    # ── Sync to ensure all writes are flushed ────────────────────────────────
    print("\n[SYNC] Flushing file system buffers...")
    os.sync()
    print("[SYNC] Complete.")

    print("\nProgram 02 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()