# mel-fairness-probe

Reproducibility code for the paper:

> **Mel-Filterbank Design as a Fairness Probe for Music Genre Classifiers**
> Sridharan Sankaran, IEEE Signal Processing Letters, 2026

This repository contains the complete experimental pipeline demonstrating that mel-filterbank hyperparameters — band count *M*, frequency ceiling *f*_max, and normalization scheme — constitute an unchecked fairness variable in music genre classification. Genres with wide spectral spread (rock, disco, hip-hop, reggae, pop) are systematically disadvantaged relative to spectrally concentrated genres (classical, blues, jazz, metal, country) under standard filterbank configurations. The inter-group accuracy gap reaches **22.7 percentage points** at *M* = 80, *f*_max = 8 kHz, and replicates in direction on FMA-Small.

---

## Repository Structure

```
filterbank-fairness/
├── 00_precompute_filterbank_fairness.py   # Pre-analysis: centroids, spread groups, filter density
├── 01_exp1_bandcount_ablation.py          # Ablation: band count M ∈ {40, 64, 80, 128}
├── 02_exp2_fmax_ablation.py               # Ablation: frequency ceiling f_max ∈ {4k, 8k, 11k} Hz
├── 03_exp3_normalization_ablation.py      # Ablation: normalization ∈ {Slaney, HTK, Area}
├── 04_exp4_fma_replication.py             # Cross-dataset replication on FMA-Small
├── 05_exp5_saliency_probe.py              # Gradient saliency probe (paper Figure 2)
└── README.md
```

---

## Programs

### `00_precompute_filterbank_fairness.py` — Pre-Analysis
**GPU required:** No

Computes three analytical quantities before any model training:

1. Spectral centre-of-mass *f̄*_g and spectral spread σ_g per GTZAN genre from mean linear magnitude spectra.
2. Effective filter density *D*_g(θ) for every (genre, filterbank configuration) pair — the number of mel filter centres falling within the genre's spectral support [*f̄*_g − σ_g, *f̄*_g + σ_g].
3. Fairness differential ΔD per configuration (high-spread minus low-spread group mean).

Genre groups are assigned by a **data-driven median split** on σ_g (corpus median = 2,485 Hz), replacing fixed frequency thresholds. ΔD ≥ 3 in 22 of 24 configurations confirms the pre-training fairness hypothesis and validates the experimental programme.

**Outputs:** `precompute_genre_centroids.json`, `precompute_filter_density.json`, `fig_00_01_genre_spread.png`, `fig_00_02_filter_density_heatmap.png` (saved to Google Drive)

---

### `01_exp1_bandcount_ablation.py` — Ablation: Band Count
**GPU required:** Yes

Trains the fixed GenreCNN backbone on GTZAN mel-spectrograms with four band counts *M* ∈ {40, 64, 80, 128}, holding *f*_max = 8,000 Hz and Slaney normalization fixed. Uses 3-repeat × 5-fold repeated stratified cross-validation (seeds {42, 7, 123}, 15 runs per *M*) to reduce variance arising from GTZAN's limited per-genre sample size (100 clips/genre). Best-fold models at *M* = 40 and *M* = 128 are saved for reuse by Program 05.

**Key finding:** FFD widens monotonically from *M* = 40 (0.146) to *M* = 80 (0.227). High-spread group accuracy drops 9.1 pp as *M* increases from 40 to 80; low-spread accuracy is stable at 0.638–0.659 throughout.

**Outputs:** `exp1_results.json`, `exp1_best_model_M40.pth`, `exp1_best_model_M128.pth`, `fig_01_01_accuracy_heatmap.png`, periodic checkpoints `exp1_checkpoint_M{m}_repeat{r}_fold{f}_epoch{e}.pth` (saved to Google Drive)

---

### `02_exp2_fmax_ablation.py` — Ablation: Frequency Ceiling
**GPU required:** Yes

Trains GenreCNN with three frequency ceilings *f*_max ∈ {4000, 8000, 11025} Hz, holding *M* = 64 and Slaney normalization fixed. Identical CV protocol to Program 01. Spectrogram cache files use fully qualified naming `specs_M{M}_fmax{fmax}.npy`.

**Key finding:** FFD peaks at *f*_max = 8 kHz (0.217) rather than at the narrow extreme (4 kHz: 0.142). High-spread group accuracy varies by only 2.5 pp across the full *f*_max range; no choice of *f*_max alone closes the fairness gap.

**Outputs:** `exp2_results.json`, `exp2_results_partial.json`, periodic checkpoints `exp2_checkpoint_M64_fmax{f}_repeat{r}_fold{f}_epoch{e}.pth` (saved to Google Drive)

---

### `03_exp3_normalization_ablation.py` — Ablation: Normalization Scheme
**GPU required:** Yes

Trains GenreCNN with three normalization schemes, holding *M* = 64 and *f*_max = 8,000 Hz fixed:

- **Slaney**: amplitude-normalized triangular filters (librosa `norm="slaney"`)
- **HTK**: peak-normalized filters (librosa `norm=None, htk=True`)
- **Area**: manually constructed — each filter row divided by its L1 norm, equalising filter gain across the filterbank regardless of centre frequency

Note: Slaney and HTK produce identical *D*_g(θ) values (normalization affects filter gain, not centre positions). Accuracy differences between schemes therefore reflect gain equalization effects, not coverage.

**Key finding:** HTK produces the largest FFD (0.190); Slaney the smallest (0.142). Area normalization, despite fully equalising gains, yields intermediate FFD (0.176), dissociating gain equalization from fairness.

**Outputs:** `exp3_results.json`, `exp3_results_partial.json`, periodic checkpoints `exp3_checkpoint_{norm}_repeat{r}_fold{f}_epoch{e}.pth` (saved to Google Drive)

---

### `04_exp4_fma_replication.py` — Cross-Dataset Replication
**GPU required:** Yes

Replicates the fairness disparity on FMA-Small (8,000 clips, 8 genres, MP3, 44.1 kHz) using two anchor configurations identified in Program 00:

| Anchor | *M* | *f*_max | ΔD |
|--------|-----|--------|-----|
| Max-ΔD | 128 | 8,000 Hz | 15.4 |
| Min-ΔD | 40  | 4,000 Hz | 2.0  |

FMA-Small metadata is downloaded at runtime from `https://os.unil.cloud.switch.ch/fma/fma_metadata.zip`. Genre labels are parsed from `tracks.csv` (multi-level pandas header, filtered to `subset == 'small'`). Genre spread groups are recomputed from FMA-Small's own corpus statistics via a fresh median split — the GTZAN median (2,485 Hz) is not carried over. Two replication checks are reported: FFD direction replication and Pearson correlation of per-genre accuracy gain across configurations.

**Key finding:** High-spread group underperforms low-spread group under both anchor configurations on FMA-Small, replicating the GTZAN direction across corpora differing in taxonomy, format, and provenance.

**Outputs:** `exp4_fma_centroids.json`, `exp4_results.json`, `exp4_results_partial.json`, periodic checkpoints `exp4_checkpoint_{config}_repeat{r}_fold{f}_epoch{e}.pth` (saved to Google Drive)

---

### `05_exp5_saliency_probe.py` — Gradient Saliency Probe
**GPU required:** Yes (gradient computation)

Computes per-genre frequency-axis gradient saliency maps using the best-fold models from Program 01 (*M* = 40 and *M* = 128) on the full GTZAN corpus (all 1,000 clips, no train/val split). For each genre *g* and model *M*:

```
s_g^(m) = (1/T) * Σ_t | ∂ŷ_g / ∂X_{m,t} |
```

averaged over all clips of genre *g* and normalised to unit maximum for cross-*M* comparability. Also computes Jensen-Shannon divergence between low-spread and high-spread group saliency distributions, and Spearman correlation between sub-500 Hz saliency mass and per-genre accuracy.

**Key finding:** Under *M* = 40, hip-hop attention is locked to the sub-500 Hz region (poorly resolved by coarse filterbank); *M* = 128 redistributes attention toward the 1,000–6,000 Hz mid-range. Classical attention is robust to *M*. JS divergence decreases from 0.045 (*M* = 40) to 0.026 (*M* = 128).

**Outputs:** `exp5_saliency_results.json`, `fig_05_02_saliency_profiles.png` (paper Figure 2) (saved to Google Drive)

---

## Datasets

| Dataset | Size | Genres | Format | Source |
|---------|------|--------|--------|--------|
| GTZAN | 1,000 clips × 30 s | 10 | WAV, 22.05 kHz | [Tzanetakis & Cook, 2002] |
| FMA-Small | 8,000 clips × 30 s | 8 | MP3, 44.1 kHz | [Defferrard et al., 2017] |

Place datasets in Google Drive at:
```
/content/drive/MyDrive/datasets/GTZAN.zip
/content/drive/MyDrive/datasets/FMA-small.zip
```
FMA metadata is downloaded automatically by Program 04 at runtime.

---

## CNN Backbone (GenreCNN)

All experiments use an identical fixed backbone — only the filterbank geometry varies:

```
Input: (1, M, T)
Block 1: Conv2D(1→32,  3×3, pad=1) + BatchNorm + ReLU + MaxPool(2×2)
Block 2: Conv2D(32→64, 3×3, pad=1) + BatchNorm + ReLU + MaxPool(2×2)
Block 3: Conv2D(64→128,3×3, pad=1) + BatchNorm + ReLU + MaxPool(2×2)
Global Average Pooling → FC(128→N_classes) → Softmax
```

Training: Adam (lr = 1e-3), cross-entropy loss, 50 epochs, batch size 16.
Evaluation: 3-repeat × 5-fold stratified CV, seeds {42, 7, 123}, 15 runs per configuration.

---

## Execution Order

Programs must be run in the following sequence due to data dependencies:

```
00  →  01  →  02  →  03  →  05
                ↘
                04
```

Specifically:
- **00** must run first — produces `precompute_genre_centroids.json` and `precompute_filter_density.json` required by 01, 02, 03, 04.
- **01** must run before **05** — produces `exp1_best_model_M40.pth` and `exp1_best_model_M128.pth` required by the saliency probe.
- **02**, **03**, **04** can run independently after 00.
- **05** requires 01 to be complete.

---

## Resilience and Checkpointing

All programs are designed to survive Google Colab session disconnects:

- Spectrogram arrays are cached locally as `.npy` / `.npz` files and reloaded on resume.
- Model checkpoints are saved every 5 epochs to the Google Drive project folder as `*_epoch{e}.pth`.
- On restart, each program scans for the latest checkpoint (local disk first, then Google Drive) and resumes from the last completed epoch.
- JSON results are saved incrementally after each completed configuration (partial results files `*_partial.json`).

---

## Google Drive Storage Configuration

All outputs (JSON results, figures, checkpoints) are written to a Google Drive project folder configured at the top of each program:

```python
PROJECT_DIR = "/content/drive/MyDrive/paper/mel-fairness-probe/"
```

The script expects the following datasets in `/content/drive/MyDrive/datasets/`:
- `GTZAN.zip`
- `FMA-small.zip`

FMA metadata is downloaded automatically at runtime.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | CNN training and gradient computation |
| `torchaudio` | Audio utilities |
| `librosa` | Mel-spectrogram computation, filterbank construction |
| `numpy` | Numerical computation |
| `pandas` | FMA metadata parsing (Program 04) |
| `scikit-learn` | Stratified cross-validation, label encoding |
| `scipy` | Jensen-Shannon divergence, Spearman correlation |
| `matplotlib` | Figure generation |
| `tqdm` | Progress bars |
| `requests` | FMA metadata download (Program 04) |

All packages are installed automatically via `pip` at the top of each program. No manual environment setup is required.

---

## Key Metrics

**Filterbank Fairness Disparity (FFD):**
```
FFD(θ) = | Ā_{G_high}(θ) − Ā_{G_low}(θ) |
```
where Ā_G(θ) is mean per-genre accuracy over spread group G under configuration θ.

**Effective Filter Density:**
```
D_g(θ) = |{k : |f_k − f̄_g| ≤ σ_g}|
```
Number of mel filter centres within the spectral support of genre g.

**Spectral Spread:**
```
σ_g = sqrt( Σ_n f_n² S_g(n) / Σ_n S_g(n)  −  f̄_g² )
```
```

